"""NHL generator evaluation: next-event loss, and the simulated score distribution against actual
results.

For a sample of games of one season (by default the fold's test season, which the model never
saw), every game is simulated `sims` times from the opening faceoff with its real pre-game ratings.
The report compares, simulated vs actual:

- goals per team (mean, sd and the distribution 0..8+), total goals (mean, sd, distribution) and
  home margin;
- how games end: regulation, overtime, shootout (regular season);
- the event mix per game: shot attempts, goals, saves, missed and blocked shots, penalties
  (minors), power-play goals, goalie pulls and empty-net goals, faceoffs, hits;
- prediction quality on those games: moneyline Brier and log loss of the simulated home win rate
  (a shootout win counts as a win), against pre-game Elo and a constant home rate; total-goals
  MAE and the correlation of the mean simulated total with the actual total.
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .data.preprocess import load
from .data.ratings import RATING_COLS, elo_win_prob
from .engine.rules import AWAY, HOME, new_game
from .model.dataset import encode_games
from .sim.sampler import Simulator
from .train import evaluate as model_nll, split_games

log = logging.getLogger(__name__)
GOAL_BINS = 9   # 0..7, 8+ goals per team
TOTAL_BINS = 13  # 0..11, 12+ total


def _pre(ratings: pd.DataFrame | None, game_id: str) -> tuple[dict, dict] | None:
    if ratings is None:
        return None
    rg = ratings[ratings.game_id == game_id].set_index("side")
    if set(rg.index) != {"home", "away"}:
        return None
    return tuple({k: float(rg.at[sd, f"pre_{k}"]) for k in RATING_COLS} for sd in ("home", "away"))


def _ending(periods: int, shootout: bool) -> str:
    return "shootout" if shootout else "overtime" if periods > 3 else "regulation"


def event_counts(outcomes, states=None) -> dict:
    """Per-game counts (both teams) from a list of Outcomes (and the states before them)."""
    c = Counter((o.play_type, o.result) for o in outcomes)
    types = Counter(o.play_type for o in outcomes)
    out = {
        "events": len(outcomes), "shot_attempts": types["shot"], "goals": c[("shot", "goal")],
        "saves": c[("shot", "saved")], "missed": c[("shot", "missed")], "blocked": c[("shot", "blocked")],
        "faceoffs": types["faceoff"], "hits": types["hit"], "giveaways": types["giveaway"],
        "takeaways": types["takeaway"], "stoppages": types["stoppage"], "penalties": types["penalty"],
        "minors": c[("penalty", "minor")], "goalie_pulls": types["goalie_pull"],
    }
    if states is not None:
        pp = en = 0
        for s, o in zip(states, outcomes):
            if o.play_type == "shot" and o.result == "goal" and o.side is not None and s.phase == 1:
                sk = s.skaters
                pp += sk[o.side] > sk[1 - o.side] and not s.empty_net[o.side]
                en += bool(s.empty_net[1 - o.side])
        out["power_play_goals"], out["empty_net_goals"] = pp, en
    return out


def _hist(values, bins: int) -> list[float]:
    v = np.minimum(np.asarray(values, dtype=int), bins - 1)
    return (np.bincount(v, minlength=bins) / max(len(v), 1)).round(4).tolist()


def score_report(sims: pd.DataFrame, actual: pd.DataFrame) -> dict:
    """`sims`: one row per simulated game (game_id, home, away, periods, shootout); `actual`: the
    games table rows of the same games. Goals exclude the shootout's winner goal."""
    def goals(df):
        h, a = df.home_final.to_numpy(), df.away_final.to_numpy()
        so = df.shootout.to_numpy(bool)
        h, a = h - (so & (h > a)), a - (so & (a > h))  # in-play goals only
        return h, a
    sh, sa = goals(sims)
    ah, aa = goals(actual)
    reg = actual.game_type.to_numpy() == "REG"
    sreg = sims.game_type.to_numpy() == "REG"
    pair = lambda f: [round(float(f(sh, sa)), 4), round(float(f(ah, aa)), 4)]
    end_s = Counter(_ending(p, s) for p, s in zip(sims.periods[sreg], sims.shootout[sreg]))
    end_a = Counter(_ending(p, s) for p, s in zip(actual.periods[reg], actual.shootout[reg]))
    return {
        "_format": "[simulated, actual]",
        "goals_per_team_mean": pair(lambda h, a: np.r_[h, a].mean()),
        "goals_per_team_sd": pair(lambda h, a: np.r_[h, a].std()),
        "home_goals_mean": pair(lambda h, a: h.mean()), "away_goals_mean": pair(lambda h, a: a.mean()),
        "total_mean": pair(lambda h, a: (h + a).mean()), "total_sd": pair(lambda h, a: (h + a).std()),
        "margin_sd": pair(lambda h, a: (h - a).std()),
        "home_win_rate": [round(float((sims.home_final > sims.away_final).mean()), 4),
                          round(float((actual.home_final > actual.away_final).mean()), 4)],
        "goals_per_team_dist": [_hist(np.r_[sh, sa], GOAL_BINS), _hist(np.r_[ah, aa], GOAL_BINS)],
        "total_dist": [_hist(sh + sa, TOTAL_BINS), _hist(ah + aa, TOTAL_BINS)],
        "regular_season_endings": {k: [round(end_s[k] / max(sreg.sum(), 1), 4),
                                       round(end_a[k] / max(reg.sum(), 1), 4)]
                                   for k in ("regulation", "overtime", "shootout")},
    }


def simulate_games(sim: Simulator, games: pd.DataFrame, ratings: pd.DataFrame | None, sims: int,
                   seed: int = 0, batch_size: int = 256, keep_logs: int = 0) -> tuple[pd.DataFrame, list]:
    """`sims` simulations of each game; one row per simulation."""
    starts, rat, keys = [], [], []
    for g in games.itertuples(index=False):
        pre = _pre(ratings, g.game_id) if sim.model.cfg.n_cont else None
        for _ in range(sims):
            starts.append(new_game(g.home, g.away, int(g.season), g.game_type))
            rat.append(pre)
            keys.append((g.game_id, g.game_type))
    t0 = time.time()
    finals, logs = sim.rollout_games(starts, rat, seed=seed, batch_size=batch_size, keep_logs=keep_logs)
    log.info("simulated %d games in %.0fs", len(finals), time.time() - t0)
    df = pd.DataFrame({
        "game_id": [k[0] for k in keys], "game_type": [k[1] for k in keys],
        "home_final": [s.score[HOME] for s in finals], "away_final": [s.score[AWAY] for s in finals],
        "periods": [min(s.period, 4) for s in finals], "shootout": [s.phase == 3 for s in finals],
        "over": [s.over for s in finals],
    })
    return df, logs


def run(ckpt: Path, data_dir: Path, season: int | None = None, games: int = 200, sims: int = 50,
        seed: int = 0, nll: bool = True) -> dict:
    """Evaluate a checkpoint on `games` sampled games of `season` (default: the fold's test season,
    the validation season + 1; else the validation season's held-out part)."""
    sim = Simulator.load(ckpt)
    all_games, plays, ratings = load(data_dir)
    vs = sim.meta.get("val_season")
    season = season or ((vs + 1) if vs is not None and (all_games.season == vs + 1).any() else vs)
    report: dict = {"checkpoint": str(ckpt), "val_season": vs, "season": season}
    if nll and vs is not None:
        _, val_g = split_games(all_games, vs, sim.meta.get("val_fraction", 0.4))
        items = encode_games(plays[plays.game_id.isin(val_g.game_id)], val_g, sim.ts, sim.model.cfg.max_len,
                             ratings if sim.model.cfg.n_cont else None)
        report["val_nll"] = {k: round(v, 4) for k, v in model_nll(sim.model, items).items()}
    pool = all_games[all_games.season == season]
    if season == vs:  # only the held-out part of the validation season
        _, pool = split_games(all_games, vs, sim.meta.get("val_fraction", 0.4))
    test = pool.sample(min(games, len(pool)), random_state=seed).sort_values(["game_date", "game_id"])
    df, logs = simulate_games(sim, test, ratings, sims, seed, keep_logs=min(sims * 20, len(test) * sims))
    report["games"], report["sims_per_game"] = len(test), sims
    report["unfinished_sims"] = int((~df.over).sum())
    report["scores"] = score_report(df[df.over], test)

    # per-game predictions
    g = df[df.over].assign(win=lambda d: (d.home_final > d.away_final).astype(float),
                           total=lambda d: d.home_final + d.away_final)
    per = g.groupby("game_id").agg(p_home=("win", "mean"), pred_total=("total", "mean"))
    t = test.set_index("game_id").join(per, how="inner")
    y = (t.home_final > t.away_final).astype(float).to_numpy()
    p = np.clip(t.p_home.to_numpy(), 1e-3, 1 - 1e-3)
    q = {"brier_model": float(((p - y) ** 2).mean()),
         "logloss_model": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
         "brier_home_rate": float(((y.mean() - y) ** 2).mean())}
    if ratings is not None:
        pre = [_pre(ratings, gid) for gid in t.index]
        pe = np.array([elo_win_prob(x[0]["elo"], x[1]["elo"]) if x else np.nan for x in pre])
        ok = ~np.isnan(pe)
        q["brier_elo"] = float(((pe[ok] - y[ok]) ** 2).mean())
        q["corr_model_elo"] = float(np.corrcoef(p[ok], pe[ok])[0, 1])
    actual_total = (t.home_final + t.away_final - t.shootout.astype(int)).to_numpy(float)
    sim_total = t.pred_total.to_numpy() - (g.groupby("game_id").apply(
        lambda d: (d.shootout & (d.home_final != d.away_final)).mean(), include_groups=False)
        .reindex(t.index).to_numpy())
    q["total_mae_model"] = float(np.abs(sim_total - actual_total).mean())
    q["total_mae_mean"] = float(np.abs(actual_total.mean() - actual_total).mean())
    q["corr_total"] = float(np.corrcoef(sim_total, actual_total)[0, 1])
    report["prediction"] = {k: round(v, 4) for k, v in q.items()}

    # event mix of the kept simulated games against the real ones
    sim_counts = pd.DataFrame([event_counts([o for _, o in lg], [s for s, _ in lg]) for lg in logs if lg])
    real = plays[plays.game_id.isin(test.game_id)]
    real_counts = pd.DataFrame([_real_counts(gp) for _, gp in real.groupby("game_id")])
    report["events_per_game"] = {k: [round(float(sim_counts[k].mean()), 2), round(float(real_counts[k].mean()), 2)]
                                 for k in sim_counts if k in real_counts}
    report["events_per_game"]["_format"] = "[simulated, actual] per game, both teams"
    return report


def _real_counts(gp: pd.DataFrame) -> dict:
    from .engine.rules import Outcome
    outs = [Outcome(t, r, None if s == 2 else int(s), int(e))
            for t, r, s, e in zip(gp.play_type, gp.result, gp.side, gp.elapsed)]
    out = event_counts(outs)
    goal = (gp.play_type == "shot") & (gp.result == "goal") & (gp.phase == 1)
    side = gp.side.to_numpy()
    sk_own = np.where(side == 0, gp.sk_h, gp.sk_a)
    sk_opp = np.where(side == 0, gp.sk_a, gp.sk_h)
    en_own = np.where(side == 0, gp.en_h, gp.en_a)
    en_opp = np.where(side == 0, gp.en_a, gp.en_h)
    out["power_play_goals"] = int((goal & (sk_own > sk_opp) & ~en_own.astype(bool)).sum())
    out["empty_net_goals"] = int((goal & en_opp.astype(bool)).sum())
    return out
