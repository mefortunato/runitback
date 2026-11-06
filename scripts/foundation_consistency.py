"""Is the play-by-play generator consistent with the wager model?

For every test game of a walk-forward fold, sample full games from that fold's foundation model
(pre-game ratings, each team's previous-game lineups: what is known before kickoff) and turn the
rollouts into the three markets' probabilities: home win (moneyline), home cover and over, the last
two without pushes. Writes one CSV per fold to --out; `--compare` scores them against wager
backtest results (see experiments/nfl/README.md).

    python scripts/foundation_consistency.py simulate --season 2021 --sims 100
    python scripts/foundation_consistency.py compare --wager experiments/nfl/foundation/prod
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

log = logging.getLogger("consistency")
DATA = Path("data/nfl/processed")


def strength_scale(sim, ld, season: int, games: int, sims: int) -> float:
    """The fold's team-strength scale: Brier-best on `season` (before the test season)."""
    from runitback.nfl.evaluate import game_lineups
    from runitback.nfl.engine.rules import new_game
    from runitback.nfl.sim.sampler import scale_ratings

    g = pd.read_parquet(DATA / "games.parquet")
    ratings = pd.read_parquet(DATA / "ratings.parquet")
    g = g[g.season == season].sample(games, random_state=0)
    y = ((g.home_final > g.away_final) + 0.5 * (g.home_final == g.away_final)).to_numpy()
    best, best_brier = 1.0, np.inf
    for scale in (1.0, 1.25, 1.5, 1.75, 2.0):
        p = []
        for i, row in enumerate(g.itertuples()):
            rg = ratings[ratings.game_id == row.game_id].set_index("side")
            pre = tuple(scale_ratings({k: float(rg.at[s, f"pre_{k}"]) for k in ("off_epa", "def_epa", "elo")}, scale)
                        for s in ("home", "away"))
            start = new_game(row.home, row.away, row.season, row.game_type, opening_kicker=int(row.opening_kicker))
            finals, _ = sim.rollout(start, sims, seed=i, ratings=pre,
                                    lineups=game_lineups(ld, row, "previous") if ld is not None else None)
            m = np.array([f.score[0] - f.score[1] for f in finals])
            p.append((m > 0).mean() + 0.5 * (m == 0).mean())
        brier = float(((np.array(p) - y) ** 2).mean())
        log.info("season %d scale %.2f brier %.4f", season, scale, brier)
        if brier < best_brier:
            best, best_brier = scale, brier
    return best


def simulate(ckpt: Path, season: int, sims: int, out: Path, lines: Path, max_games: int | None,
             calib_games: int, calib_sims: int) -> Path:
    from runitback.nfl.engine.rules import new_game
    from runitback.nfl.evaluate import game_lineups, load_lineup_data
    from runitback.nfl.sim.sampler import Simulator, scale_ratings

    sim = Simulator.load(ckpt)
    ld = load_lineup_data(sim, DATA)
    scale = strength_scale(sim, ld, season - 1, calib_games, calib_sims)
    log.info("fold %d: strength scale %.2f", season, scale)
    games = pd.read_parquet(DATA / "games.parquet")
    ratings = pd.read_parquet(DATA / "ratings.parquet")
    wl = pd.read_csv(lines, low_memory=False)
    wl = wl[wl.location == 1][["game_id", "spread_line", "total_line"]]   # the home side's row
    g = games[games.season == season].merge(wl, on="game_id", how="inner")
    if max_games:
        g = g.head(max_games)
    rows, t0 = [], time.time()
    for i, row in enumerate(g.itertuples()):
        rg = ratings[ratings.game_id == row.game_id].set_index("side")
        pre = tuple(scale_ratings({k: float(rg.at[s, f"pre_{k}"]) for k in ("off_epa", "def_epa", "elo")}, scale)
                    for s in ("home", "away"))
        lu = game_lineups(ld, row, "previous") if ld is not None else None
        finals = []
        for k, n in ((0, sims // 2), (1, sims - sims // 2)):   # both opening kickers
            start = new_game(row.home, row.away, row.season, row.game_type, opening_kicker=k)
            f, _ = sim.rollout(start, n, seed=1000 * i + k, ratings=pre, lineups=lu)
            finals += f
        h = np.array([s.score[0] for s in finals], dtype=float)
        a = np.array([s.score[1] for s in finals], dtype=float)
        m, t = h - a, h + a

        def no_push(x, line):
            decided = x != line
            return float((x[decided] > line).mean()) if decided.any() else np.nan

        rows.append(dict(game_id=row.game_id, season=season, sims=len(finals),
                         gen_win_prob=float((m > 0).mean() + 0.5 * (m == 0).mean()),
                         gen_cover_prob=no_push(m, row.spread_line), gen_over_prob=no_push(t, row.total_line),
                         gen_margin=float(m.mean()), gen_total=float(t.mean()),
                         gen_margin_sd=float(m.std()), gen_total_sd=float(t.std())))
        if i % 25 == 0:
            log.info("fold %d: %d/%d games (%.0fs)", season, i + 1, len(g), time.time() - t0)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{season}.csv"
    df = pd.DataFrame(rows)
    df.attrs["strength_scale"] = scale
    df.assign(strength_scale=scale).to_csv(path, index=False)
    return path


def _home_rows(wager_dir: Path) -> pd.DataFrame:
    w = pd.concat([pd.read_csv(p) for p in sorted(wager_dir.glob("*.csv"))], ignore_index=True)
    return w[w.location == 1] if (w.location == 1).any() else w


def compare(gen_dir: Path, wager_dirs: list[Path]) -> pd.DataFrame:
    """Per market: each source's Brier and calibration slope, and the generator vs wager-head agreement."""
    gen = pd.concat([pd.read_csv(p) for p in sorted(gen_dir.glob("*.csv"))], ignore_index=True)
    out = []
    for wdir in wager_dirs:
        w = _home_rows(wdir)
        df = w.merge(gen, on="game_id", how="inner")
        y_win = (df.result > 0).astype(float) + 0.5 * (df.result == 0)
        markets = {
            "moneyline": ("win_prob", "gen_win_prob", y_win, df.result != 0),
            "spread": ("cover_prob", "gen_cover_prob", (df.result > df.market_margin).astype(float),
                       df.result != df.market_margin),
            "total": ("over_prob", "gen_over_prob", (df.total > df.total_line).astype(float),
                      df.total != df.total_line),
        }
        for market, (wcol, gcol, y, decided) in markets.items():
            d = decided & df[wcol].notna() & df[gcol].notna()
            pw, pg, yy = df.loc[d, wcol].to_numpy(), df.loc[d, gcol].to_numpy(), y[d].to_numpy()
            bw, bg = (pw - yy) ** 2, (pg - yy) ** 2
            diff = bg - bw
            out.append(dict(
                wager=wdir.name, market=market, n=int(d.sum()),
                brier_wager=bw.mean(), brier_generator=bg.mean(),
                d_brier=diff.mean(), d_brier_ci=1.96 * diff.std(ddof=1) / np.sqrt(len(diff)),
                mean_abs_gap=float(np.abs(pg - pw).mean()), corr=float(np.corrcoef(pg, pw)[0, 1]),
                slope_wager=_slope(pw, yy), slope_generator=_slope(pg, yy),
                mean_wager=pw.mean(), mean_generator=pg.mean(), rate=yy.mean()))
        out.append(dict(wager=wdir.name, market="margin_points", n=len(df),
                        mean_abs_gap=float((df.gen_margin - df.pred_margin).abs().mean()),
                        corr=float(np.corrcoef(df.gen_margin, df.pred_margin)[0, 1])))
        out.append(dict(wager=wdir.name, market="total_points", n=len(df),
                        mean_abs_gap=float((df.gen_total - df.pred_total).abs().mean()),
                        corr=float(np.corrcoef(df.gen_total, df.pred_total)[0, 1])))
    return pd.DataFrame(out)


def _slope(p: np.ndarray, y: np.ndarray) -> float:
    """Calibration slope: logistic regression of the outcome on logit(p) (1 = calibrated spread)."""
    x = np.log(np.clip(p, 1e-4, 1 - 1e-4) / np.clip(1 - p, 1e-4, 1))
    X = np.stack([np.ones_like(x), x], 1)
    w = np.zeros(2)
    for _ in range(50):   # Newton
        q = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - q)
        H = (X * (q * (1 - q))[:, None]).T @ X + 1e-9 * np.eye(2)
        w += np.linalg.solve(H, g)
    return float(w[1])


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--season", type=int, required=True)
    s.add_argument("--ckpt", default="runs/foundation/fold{season}/model.pt")
    s.add_argument("--sims", type=int, default=100)
    s.add_argument("--out", type=Path, default=Path("experiments/nfl/foundation/generator"))
    s.add_argument("--lines", type=Path, default=Path("data/nfl/games.csv"))
    s.add_argument("--max-games", type=int)
    s.add_argument("--calib-games", type=int, default=60)
    s.add_argument("--calib-sims", type=int, default=60)
    s.add_argument("--threads", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("--generator", type=Path, default=Path("experiments/nfl/foundation/generator"))
    c.add_argument("--wager", type=Path, nargs="+", required=True)
    c.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.cmd == "simulate":
        if a.threads:
            torch.set_num_threads(a.threads)
        print(simulate(Path(a.ckpt.format(season=a.season)), a.season, a.sims, a.out, a.lines, a.max_games,
                       a.calib_games, a.calib_sims))
    else:
        df = compare(a.generator, a.wager)
        print(df.round(4).to_string(index=False))
        if a.out:
            df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
