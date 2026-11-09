"""A direct game model on player values: who plays, plus a team rating for what the players don't
explain, straight to the margin and total.

    margin = home field + b . (home - away lineup values: offense, defense, kicking)
             + (R_home - R_away)
    total  = a0 + a . (both offenses' values, both defenses' values) + T_home + T_away

The lineup values are the projected pre-game lineups' points over replacement
(scripts/player_values.py, leak-free weekly fits). R and T are team ratings learned from results:
after each game they move toward the part of the margin / total the lineups did not predict
(step k), and shrink toward 0 between seasons. b, a and k are fit on the seasons before the test
season; within it the ratings keep updating after each game. Margins and totals are normal around
the predictions (sd from the training seasons), giving the win / cover / over probabilities.

With --market, the predicted margin and total are blended with the market's (weights fit on the
training seasons): a model that uses the line. Writes <out>/<name>/<season>.csv in the format the
wager experiments score (rows and market columns copied from a baseline configuration's files).

    python scripts/direct_game_model.py --out experiments/nfl/playervalues --name pvg
    python scripts/direct_game_model.py --out experiments/nfl/playervalues --name pvg_mkt --market
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

SEASON_KEEP = 0.5
GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB", "ST"]     # share of a team rating carried into the next season


def load(pv_path: Path, games_csv: Path) -> pd.DataFrame:
    g = pd.read_csv(games_csv, low_memory=False)
    g = g[(g.location >= 0) & (g.season >= 2016)].sort_values("location", ascending=False).drop_duplicates("game_id")
    g = g.sort_values(["gameday", "gametime", "game_id"], na_position="last")
    pv = pd.read_parquet(pv_path).set_index(["game_id", "team"])
    cols = [f"_pv_{i}" for i in range(9)]   # QB, RB, WR, TE, OL, DL, LB, DB, kicking
    h = pv.reindex(pd.MultiIndex.from_arrays([g.game_id, g.team]))[cols].to_numpy()
    a = pv.reindex(pd.MultiIndex.from_arrays([g.game_id, g.opp_team]))[cols].to_numpy()
    d = pd.DataFrame({"game_id": g.game_id.to_numpy(), "season": g.season.to_numpy(), "home": g.team.to_numpy(),
                      "away": g.opp_team.to_numpy(), "margin": g.result.to_numpy(float), "total": g.total.to_numpy(float),
                      "mkt": g.spread_line.to_numpy(float), "mkt_total": g.total_line.to_numpy(float),
                      "neutral": (g.location == 0).to_numpy()})
    known = ~(np.isnan(h).any(1) | np.isnan(a).any(1))
    h, a = np.nan_to_num(h), np.nan_to_num(a)
    d["known"] = known
    for i, g in enumerate(GROUPS):
        d[f"x_{g}"] = h[:, i] - a[:, i]
    d["x_off"], d["x_def"], d["x_st"] = (h[:, :5] - a[:, :5]).sum(1), (h[:, 5:8] - a[:, 5:8]).sum(1), h[:, 8] - a[:, 8]
    d["t_off"], d["t_def"] = (h[:, :5] + a[:, :5]).sum(1), (h[:, 5:8] + a[:, 5:8]).sum(1)
    return d.reset_index(drop=True)


def run_ratings(d: pd.DataFrame, base_m: np.ndarray, base_t: np.ndarray, k: float, kt: float
                ) -> tuple[np.ndarray, np.ndarray]:
    """Pre-game rating terms (R_h - R_a, T_h + T_a) for every game, updating after each one."""
    R, T = {}, {}
    rm, rt = np.zeros(len(d)), np.zeros(len(d))
    season = None
    for i, r in enumerate(d.itertuples()):
        if r.season != season:
            season = r.season
            R = {t: v * SEASON_KEEP for t, v in R.items()}
            T = {t: v * SEASON_KEEP for t, v in T.items()}
        rh, ra, th, ta = R.get(r.home, 0.0), R.get(r.away, 0.0), T.get(r.home, 0.0), T.get(r.away, 0.0)
        rm[i], rt[i] = rh - ra, th + ta
        if np.isfinite(r.margin):
            e = r.margin - (base_m[i] + rm[i])
            R[r.home], R[r.away] = rh + k * e / 2, ra - k * e / 2
            et = r.total - (base_t[i] + rt[i])
            T[r.home], T[r.away] = th + kt * et / 2, ta + kt * et / 2
    return rm, rt


def _lstsq(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def fit_predict(d: pd.DataFrame, test: int, market: bool, by_group: bool = False) -> pd.DataFrame:
    tr = ((d.season < test) & d.known & d.margin.notna()).to_numpy()
    xcols = [f"x_{g}" for g in GROUPS] if by_group else ["x_off", "x_def", "x_st"]
    Xm = np.column_stack([d[xcols].to_numpy(), (~d.neutral).to_numpy(float)])
    Xt = d[["t_off", "t_def"]].to_numpy()
    # lineup coefficients first without ratings, then the rating steps given them, then the
    # coefficients again with the ratings as an offset
    bm, bt = _lstsq(Xm[tr], d.margin[tr]), _lstsq(Xt[tr], d.total[tr])
    best = None
    for _ in range(2):
        base_m = np.column_stack([np.ones(len(d)), Xm]) @ bm
        base_t = np.column_stack([np.ones(len(d)), Xt]) @ bt
        for k in (0.0, 0.03, 0.06, 0.1, 0.15):
            for kt in (0.0, 0.03, 0.06, 0.1):
                rm, rt = run_ratings(d, base_m, base_t, k, kt)
                err = np.mean((d.margin[tr] - base_m[tr] - rm[tr]) ** 2) + np.mean((d.total[tr] - base_t[tr] - rt[tr]) ** 2)
                if best is None or err < best[0]:
                    best = (err, k, kt, rm, rt)
        _, k, kt, rm, rt = best
        bm = _lstsq(Xm[tr], d.margin[tr] - rm[tr])
        bt = _lstsq(Xt[tr], d.total[tr] - rt[tr])
        best = None
    base_m = np.column_stack([np.ones(len(d)), Xm]) @ bm
    base_t = np.column_stack([np.ones(len(d)), Xt]) @ bt
    rm, rt = run_ratings(d, base_m, base_t, k, kt)
    pm, pt = base_m + rm, base_t + rt
    if market:
        mk = tr & d.mkt.notna().to_numpy() & d.mkt_total.notna().to_numpy()
        wm = _lstsq(np.column_stack([pm, d.mkt])[mk], d.margin[mk])
        wt = _lstsq(np.column_stack([pt, d.mkt_total])[mk], d.total[mk])
        pm = np.where(d.mkt.notna(), wm[0] + wm[1] * pm + wm[2] * d.mkt.fillna(0), pm)
        pt = np.where(d.mkt_total.notna(), wt[0] + wt[1] * pt + wt[2] * d.mkt_total.fillna(0), pt)
    sm = float(np.std((d.margin - pm)[tr]))
    st = float(np.std((d.total - pt)[tr]))
    te = (d.season == test).to_numpy()
    out = d[te][["game_id", "home"]].copy()
    out["pred_margin"], out["pred_total"] = pm[te], pt[te]
    out["win_prob"] = norm.cdf(pm[te] / sm)
    out["cover_prob"] = norm.cdf((pm[te] - d.mkt[te].to_numpy()) / sm)
    out["over_prob"] = norm.cdf((pt[te] - d.mkt_total[te].to_numpy()) / st)
    out.attrs |= dict(k=k, kt=kt, bm=bm.round(3).tolist(), bt=bt.round(3).tolist(), sd=(round(sm, 2), round(st, 2)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pv", type=Path, default=Path("experiments/nfl/playervalues/pv.parquet"))
    ap.add_argument("--games", type=Path, default=Path("data/nfl/games.csv"))
    ap.add_argument("--rows", type=Path, default=Path("experiments/nfl/lineups/pbp32"),
                    help="a configuration's per-season files to take the rows and market columns from")
    ap.add_argument("--out", type=Path, default=Path("experiments/nfl/playervalues"))
    ap.add_argument("--name", default="pvg")
    ap.add_argument("--market", action="store_true")
    ap.add_argument("--by-group", action="store_true", help="a margin coefficient per position group")
    ap.add_argument("--seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    a = ap.parse_args()
    d = load(a.pv, a.games)
    (a.out / a.name).mkdir(parents=True, exist_ok=True)
    for s in a.seasons:
        p = fit_predict(d, s, a.market, a.by_group)
        print(s, p.attrs)
        rows = pd.read_csv(a.rows / f"{s}.csv", low_memory=False)
        keep = rows.drop(columns=["pred_margin", "win_prob", "cover_prob", "pred_total", "over_prob"])
        res = keep.merge(p, on="game_id", how="inner")
        flip = (res.team != res.home).to_numpy()          # the row is the other team's side
        res.loc[flip, "pred_margin"] = -res.pred_margin[flip]
        res.loc[flip, "win_prob"] = 1 - res.win_prob[flip]
        res.loc[flip, "cover_prob"] = 1 - res.cover_prob[flip]
        res = res.drop(columns="home")
        res.to_csv(a.out / a.name / f"{s}.csv", index=False)


if __name__ == "__main__":
    main()
