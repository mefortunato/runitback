"""Evaluation: next-play NLL vs an empirical baseline, and full-game simulation quality."""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data.download import load_raw
from .data.ratings import scale_ratings
from .engine.rules import new_game
from .model.dataset import collate, encode_games
from .sim.sampler import Simulator
from .train import evaluate as model_nll, split_games

log = logging.getLogger(__name__)


def _baseline_nll(train_items: list[dict], val_items: list[dict]) -> dict[str, float]:
    """Smoothed empirical conditionals on coarse state: a strong non-sequential baseline."""
    def frame(items):
        cols = {}
        for k in ("f_phase", "f_down", "f_ytg", "f_yl", "f_qtr", "f_qsec", "type", "result", "yards", "elapsed",
                  "y_lo", "y_hi", "el_hi"):
            cols[k] = np.concatenate([g[k] for g in items])
        cols["tmask"] = np.concatenate([g["type_mask"] for g in items])
        cols["rmask"] = np.concatenate([g["result_mask"] for g in items])
        return cols

    tr, va = frame(train_items), frame(val_items)

    def keyed(c, parts):
        k = np.zeros(len(c["type"]), dtype=np.int64)
        for p, m in parts:
            k = k * m + np.minimum(c[p], m - 1)
        return k

    def ytg_b(c):
        return np.minimum(c["f_ytg"], 15)

    for c in (tr, va):
        c["ytgb"], c["ylb"], c["late"] = ytg_b(c), c["f_yl"] // 5, ((c["f_qtr"] % 2 == 0) & (c["f_qsec"] < 8)).astype(int)
    heads = {
        "type": ([("f_phase", 3), ("f_down", 5), ("ytgb", 16), ("ylb", 21), ("late", 2)], "type", 11, "tmask"),
        "result": ([("type", 11), ("f_down", 5), ("ylb", 21)], "result", 16, "rmask"),
        "yards": ([("type", 11), ("result", 16), ("ylb", 21)], "yards", 200, None),
        "elapsed": ([("type", 11), ("result", 16), ("late", 2)], "elapsed", 61, None),
    }
    out = {}
    for name, (parts, tgt, ncls, mkey) in heads.items():
        ktr, kva = keyed(tr, parts), keyed(va, parts)
        counts: dict[int, np.ndarray] = {}
        for k in np.unique(ktr):
            counts[k] = np.bincount(tr[tgt][ktr == k], minlength=ncls).astype(float)
        glob = np.bincount(tr[tgt], minlength=ncls).astype(float)
        nll = 0.0
        for i in range(len(kva)):
            c = counts.get(kva[i], np.zeros(ncls)) + glob / glob.sum() + 1e-3  # one pseudo-count of global prior
            if mkey:
                mask = va[mkey][i]
            elif name == "yards":
                ar = np.arange(ncls) - 99
                mask = (ar >= va["y_lo"][i]) & (ar <= va["y_hi"][i])
            else:
                mask = np.arange(ncls) <= va["el_hi"][i]
            c = c * mask
            nll -= math.log(c[va[tgt][i]] / c.sum())
        out[name] = nll / len(kva)
    out["total"] = sum(out.values())
    return out


def _spreads(seasons: list[int], raw_dir: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(raw_dir / f"pbp_{s}.parquet", columns=["game_id", "spread_line"]) for s in seasons]
    return pd.concat(frames).groupby("game_id").spread_line.first().reset_index()


def run(ckpt: Path, data_dir: Path, sims_per_game: int = 200, raw_dir: Path = Path("data/nfl/raw"),
        max_games: int | None = None, strength_scale: float | None = None) -> dict:
    sim = Simulator.load(ckpt)
    scale = sim.strength_scale if strength_scale is None else strength_scale
    vs, vw = sim.meta.get("val_season"), sim.meta.get("val_week", 10)
    games = pd.read_parquet(data_dir / "games.parquet")
    plays = pd.read_parquet(data_dir / "plays.parquet")
    train_g, val_g = split_games(games, vs, vw)
    if max_games:
        val_g = val_g.head(max_games)
    max_len = sim.model.cfg.max_len
    rpath = data_dir / "ratings.parquet"
    ratings = pd.read_parquet(rpath) if rpath.exists() else None
    model_ratings = ratings if sim.model.cfg.n_cont else None
    val_items = encode_games(plays[plays.game_id.isin(val_g.game_id)], val_g, sim.ts, max_len, model_ratings)
    tr_sample = train_g[train_g.season >= 2015]
    train_items = encode_games(plays[plays.game_id.isin(tr_sample.game_id)], tr_sample, sim.ts, max_len)
    report = {"val_season": vs, "val_week": vw, "val_games": len(val_g), "strength_scale": scale}
    report["next_play_nll_model"] = model_nll(sim.model, val_items)
    report["next_play_nll_baseline"] = _baseline_nll(train_items, val_items)
    log.info("NLL model %s", report["next_play_nll_model"])
    log.info("NLL baseline %s", report["next_play_nll_baseline"])

    # full-game simulations for each held-out game
    sp = _spreads(sorted(val_g.season.unique()), raw_dir)
    val_g = val_g.merge(sp, on="game_id", how="left")
    rows, t0 = [], time.time()
    stats = {"plays": [], "pass": [], "run": [], "punt": [], "turnovers": [], "fg": [], "points": []}
    for i, g in enumerate(val_g.itertuples()):
        start = new_game(g.home, g.away, g.season, g.game_type, opening_kicker=int(g.opening_kicker))
        pre = None
        if ratings is not None:
            rg = ratings[ratings.game_id == g.game_id].set_index("side")
            raw_pre = tuple({k: float(rg.at[sd, f"pre_{k}"]) for k in ("off_epa", "def_epa", "elo")}
                            for sd in ("home", "away"))
            pre = tuple(scale_ratings(r, scale) for r in raw_pre)
        finals, logs = sim.rollout(start, sims_per_game, seed=i, keep_plays=20,
                                   ratings=pre if sim.model.cfg.n_cont else None)
        h = np.array([s.score[0] for s in finals]); a = np.array([s.score[1] for s in finals])
        rows.append(dict(game_id=g.game_id, home=g.home, away=g.away, actual_margin=g.home_final - g.away_final,
                         actual_total=g.home_final + g.away_final, spread=g.spread_line,
                         p_home=float((h > a).mean() + 0.5 * (h == a).mean()), pred_margin=float((h - a).mean()),
                         pred_total=float((h + a).mean()),
                         elo_diff=(raw_pre[0]["elo"] - raw_pre[1]["elo"]) if pre else np.nan))
        for lg, s in zip(logs, finals):
            types = [o.play_type for _, o in lg]
            stats["plays"].append(len(lg))
            stats["pass"].append(types.count("pass")); stats["run"].append(types.count("run"))
            stats["punt"].append(types.count("punt")); stats["fg"].append(types.count("field_goal"))
            stats["turnovers"].append(sum(o.result in ("interception", "fumble_lost") for _, o in lg))
            stats["points"] += s.score
        if i % 10 == 0:
            log.info("simulated %d/%d games (%.0fs)", i + 1, len(val_g), time.time() - t0)
    df = pd.DataFrame(rows)
    y = (df.actual_margin > 0).astype(float) + 0.5 * (df.actual_margin == 0)
    p_spread = 1 / (1 + np.exp(-df.spread.fillna(0) / 5.9))  # logistic approx of spread->win prob
    clip = lambda p: np.clip(p, 1e-3, 1 - 1e-3)
    report["game"] = {
        "brier_model": float(((df.p_home - y) ** 2).mean()),
        "brier_spread": float(((p_spread - y) ** 2).mean()),
        "brier_coinflip": 0.25,
        "brier_elo": float(((1 / (1 + 10 ** (-(df.elo_diff + 48) / 400)) - y) ** 2).mean()),
        "logloss_model": float(-(y * np.log(clip(df.p_home)) + (1 - y) * np.log(1 - clip(df.p_home))).mean()),
        "logloss_spread": float(-(y * np.log(p_spread) + (1 - y) * np.log(1 - p_spread)).mean()),
        "accuracy_model": float(((df.p_home > 0.5) == (df.actual_margin > 0)).mean()),
        "accuracy_spread": float(((df.spread > 0) == (df.actual_margin > 0)).mean()),
        "margin_mae_model": float((df.pred_margin - df.actual_margin).abs().mean()),
        "margin_mae_spread": float((df.spread - df.actual_margin).abs().mean()),
        "corr_pred_margin_vs_spread": float(np.corrcoef(df.pred_margin, df.spread.fillna(0))[0, 1]),
        "pred_margin_std": float(df.pred_margin.std()), "spread_std": float(df.spread.std()),
        "total_mean_pred": float(df.pred_total.mean()), "total_mean_actual": float(df.actual_total.mean()),
    }
    actual = plays[plays.game_id.isin(val_g.game_id)]
    per = actual.groupby("game_id").play_type.value_counts().unstack(fill_value=0)
    turn = actual[actual.result.isin(["interception", "fumble_lost"])].groupby("game_id").size()
    report["realism"] = {
        "plays_per_game": [float(np.mean(stats["plays"])), float(actual.groupby("game_id").size().mean())],
        "pass_per_game": [float(np.mean(stats["pass"])), float(per["pass"].mean())],
        "run_per_game": [float(np.mean(stats["run"])), float(per["run"].mean())],
        "punts_per_game": [float(np.mean(stats["punt"])), float(per["punt"].mean())],
        "fg_att_per_game": [float(np.mean(stats["fg"])), float(per["field_goal"].mean())],
        "turnovers_per_game": [float(np.mean(stats["turnovers"])), float(turn.reindex(per.index, fill_value=0).mean())],
        "points_per_team": [float(np.mean(stats["points"])), float(np.mean(list(val_g.home_final) + list(val_g.away_final)))],
        "points_per_team_std": [float(np.std(stats["points"])), float(np.std(list(val_g.home_final) + list(val_g.away_final)))],
        "_format": "[simulated, actual]",
    }
    return report


def calibrate(ckpt: Path, data_dir: Path, season: int = 2024, n_games: int = 100, sims_per_game: int = 100,
              scales: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0), write: bool = False) -> dict:
    """Grid-search the team-strength scale that minimizes Brier score on `season` games."""
    sim = Simulator.load(ckpt)
    games = pd.read_parquet(data_dir / "games.parquet")
    ratings = pd.read_parquet(data_dir / "ratings.parquet")
    g = games[games.season == season].sample(min(n_games, (games.season == season).sum()), random_state=0)
    y = ((g.home_final > g.away_final) + 0.5 * (g.home_final == g.away_final)).to_numpy()
    results = {}
    for scale in scales:
        p, sd = [], []
        for i, row in enumerate(g.itertuples()):
            rg = ratings[ratings.game_id == row.game_id].set_index("side")
            pre = tuple(scale_ratings({k: float(rg.at[s, f"pre_{k}"]) for k in ("off_epa", "def_epa", "elo")}, scale)
                        for s in ("home", "away"))
            start = new_game(row.home, row.away, row.season, row.game_type, opening_kicker=int(row.opening_kicker))
            finals, _ = sim.rollout(start, sims_per_game, seed=i, ratings=pre)
            m = np.array([f.score[0] - f.score[1] for f in finals])
            p.append((m > 0).mean() + 0.5 * (m == 0).mean())
            sd.append(m.mean())
        results[scale] = {"brier": float(((np.array(p) - y) ** 2).mean()), "pred_margin_std": float(np.std(sd))}
        log.info("scale %.2f: %s", scale, results[scale])
    best = min(results, key=lambda k: results[k]["brier"])
    if write:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        ck["strength_scale"] = best
        torch.save(ck, ckpt)
    return {"season": season, "games": len(g), "best_scale": best, "results": results}
