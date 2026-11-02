"""NBA evaluation: next-event NLL, and full-game simulations of held-out games against Elo and
the betting spread."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data.ratings import RATING_COLS, elo_win_prob, scale_ratings
from .engine.rules import new_game
from .model.dataset import encode_games
from .sim.sampler import Simulator
from .train import evaluate as model_nll, split_games

log = logging.getLogger(__name__)


def _pre_ratings(ratings: pd.DataFrame, game_id: str, scale: float) -> tuple[dict, dict]:
    rg = ratings[ratings.game_id == game_id].set_index("side")
    return tuple(scale_ratings({k: float(rg.at[sd, f"pre_{k}"]) for k in RATING_COLS}, scale)
                 for sd in ("home", "away"))


def _sim_games(sim: Simulator, games: pd.DataFrame, ratings: pd.DataFrame | None, sims: int, scale: float,
               keep: int = 0) -> tuple[pd.DataFrame, list]:
    rows, logs_all, t0 = [], [], time.time()
    for i, g in enumerate(games.itertuples()):
        pre = _pre_ratings(ratings, g.game_id, scale) if ratings is not None and sim.model.cfg.n_cont else None
        finals, logs = sim.rollout(new_game(g.home, g.away, g.season, g.game_type), sims, seed=i,
                                   keep_plays=keep, ratings=pre)
        h = np.array([s.score[0] for s in finals]); a = np.array([s.score[1] for s in finals])
        rows.append(dict(game_id=g.game_id, actual_margin=g.home_final - g.away_final,
                         actual_total=g.home_final + g.away_final,
                         p_home=float((h > a).mean() + 0.5 * (h == a).mean()), pred_margin=float((h - a).mean()),
                         pred_total=float((h + a).mean()), sim_points=list(h) + list(a)))
        logs_all += logs
        if i % 10 == 0:
            log.info("simulated %d/%d games (%.0fs)", i + 1, len(games), time.time() - t0)
    return pd.DataFrame(rows), logs_all


def run(ckpt: Path, data_dir: Path, sims_per_game: int = 100, max_games: int | None = 150,
        strength_scale: float | None = None) -> dict:
    sim = Simulator.load(ckpt)
    scale = sim.strength_scale if strength_scale is None else strength_scale
    vs = sim.meta.get("val_season")
    games = pd.read_parquet(data_dir / "games.parquet")
    plays = pd.read_parquet(data_dir / "plays.parquet")
    ratings = pd.read_parquet(data_dir / "ratings.parquet") if (data_dir / "ratings.parquet").exists() else None
    _, val_g = split_games(games, vs, sim.meta.get("val_fraction", 0.4))
    if val_g.empty:
        raise ValueError("checkpoint has no held-out games (train with --val-season)")
    report = {"val_season": vs, "val_games": len(val_g), "strength_scale": scale}
    items = encode_games(plays[plays.game_id.isin(val_g.game_id)], val_g, sim.ts, sim.model.cfg.max_len,
                         ratings if sim.model.cfg.n_cont else None)
    report["next_event_nll"] = model_nll(sim.model, items)
    log.info("NLL %s", report["next_event_nll"])

    test = val_g.sample(min(max_games or len(val_g), len(val_g)), random_state=0).sort_values("game_date")
    df, logs = _sim_games(sim, test, ratings, sims_per_game, scale, keep=0)
    df = df.merge(test[["game_id", "home_spread"]] if "home_spread" in test else test[["game_id"]], on="game_id")
    y = (df.actual_margin > 0).astype(float)
    clip = lambda p: np.clip(p, 1e-3, 1 - 1e-3)
    out = {
        "games": len(df),
        "brier_model": float(((df.p_home - y) ** 2).mean()), "brier_coinflip": 0.25,
        "logloss_model": float(-(y * np.log(clip(df.p_home)) + (1 - y) * np.log(1 - clip(df.p_home))).mean()),
        "accuracy_model": float(((df.p_home > 0.5) == (df.actual_margin > 0)).mean()),
        "margin_mae_model": float((df.pred_margin - df.actual_margin).abs().mean()),
        "total_mean_pred": float(df.pred_total.mean()), "total_mean_actual": float(df.actual_total.mean()),
        "points_per_team_std": [float(np.std(np.concatenate(df.sim_points.to_list()))),
                                float(np.std(list(test.home_final) + list(test.away_final)))],
    }
    if ratings is not None:
        pre = [_pre_ratings(ratings, gid, 1.0) for gid in df.game_id]
        p_elo = np.array([elo_win_prob(h["elo"], a["elo"]) for h, a in pre])
        out["brier_elo"] = float(((p_elo - y) ** 2).mean())
        out["accuracy_elo"] = float(((p_elo > 0.5) == (df.actual_margin > 0)).mean())
    if "home_spread" in df and df.home_spread.notna().any():
        sp = df.dropna(subset=["home_spread"])
        # The sign convention of the feed's spread is checked against results, not assumed.
        sign = 1.0 if np.corrcoef(sp.home_spread, sp.actual_margin)[0, 1] >= 0 else -1.0
        exp_margin = sign * sp.home_spread
        ys = (sp.actual_margin > 0).astype(float)
        p_spread = 1 / (1 + np.exp(-exp_margin / 6.5))
        out["spread_games"] = len(sp)
        out["brier_spread"] = float(((p_spread - ys) ** 2).mean())
        out["accuracy_spread"] = float(((exp_margin > 0) == (sp.actual_margin > 0)).mean())
        out["margin_mae_spread"] = float((exp_margin - sp.actual_margin).abs().mean())
        out["corr_pred_margin_vs_spread"] = float(np.corrcoef(sp.pred_margin, exp_margin)[0, 1])
    report["game"] = out

    # realism of the simulated event mix vs the real held-out games
    _, logs = _sim_games(sim, test.head(20), ratings, 10, scale, keep=10)
    sim_counts = pd.DataFrame([{**_count([o for _, o in lg])} for lg in logs])
    real = plays[plays.game_id.isin(test.game_id)]
    real_counts = pd.DataFrame([_count(list(g.itertuples())) for _, g in real.groupby("game_id")])
    report["realism"] = {k: [float(sim_counts[k].mean()), float(real_counts[k].mean())] for k in sim_counts}
    report["realism"]["_format"] = "[simulated, actual] per game, both teams"
    return report


def _count(outcomes) -> dict:
    types = [o.play_type for o in outcomes]
    res = [(o.play_type, o.result) for o in outcomes]
    return {
        "events": len(outcomes), "fga2": types.count("shot2"), "fga3": types.count("shot3"),
        "fg3_made": sum(t == "shot3" and r.startswith("made") for t, r in res),
        "fta": types.count("free_throw"), "turnovers": types.count("turnover"),
        "off_rebounds": sum(r == "missed_off" for _, r in res), "timeouts": types.count("timeout"),
    }


def calibrate(ckpt: Path, data_dir: Path, season: int, n_games: int = 100, sims_per_game: int = 100,
              scales: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0), write: bool = False) -> dict:
    """Grid-search the team-strength scale that minimizes Brier score on `season` games."""
    sim = Simulator.load(ckpt)
    games = pd.read_parquet(data_dir / "games.parquet")
    ratings = pd.read_parquet(data_dir / "ratings.parquet")
    g = games[games.season == season]
    g = g.sample(min(n_games, len(g)), random_state=0)
    results = {}
    for scale in scales:
        df, _ = _sim_games(sim, g, ratings, sims_per_game, scale)
        y = (df.actual_margin > 0).astype(float)
        results[scale] = {"brier": float(((df.p_home - y) ** 2).mean()),
                          "pred_margin_std": float(df.pred_margin.std())}
        log.info("scale %.2f: %s", scale, results[scale])
    best = min(results, key=lambda k: results[k]["brier"])
    if write:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        ck["strength_scale"] = best
        torch.save(ck, ckpt)
    return {"season": season, "games": len(g), "best_scale": best, "results": results}
