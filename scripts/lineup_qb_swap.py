"""Do the foundation model's player embeddings carry who plays? The QB-swap check.

For games of one season, roll out the season's foundation model (pre-game ratings, previous-game
lineups, both opening kickers), then again with the same named quarterbacks in every home team's
QB1 slot (elite starters, journeyman backups, the unknown "new QB" token), on the same random
seeds. If the model has learned who matters, the home team should win more with the elite ones.

    python scripts/lineup_qb_swap.py --ckpt runs/foundation/fold2025/model.pt --season 2025 --games 60
"""
from __future__ import annotations

import argparse
import copy
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

log = logging.getLogger("qb_swap")
DATA = Path("data/nfl/processed")


def rollouts(sim, row, pre, lineups, sims: int, seed: int) -> np.ndarray:
    from runitback.nfl.engine.rules import new_game

    finals = []
    for k, n in ((0, sims // 2), (1, sims - sims // 2)):
        start = new_game(row.home, row.away, row.season, row.game_type, opening_kicker=k)
        f, _ = sim.rollout(start, n, seed=seed + k, ratings=pre, lineups=lineups)
        finals += f
    return np.array([f.score[0] - f.score[1] for f in finals], dtype=float)


def main() -> None:
    from runitback.nfl.evaluate import game_lineups, load_lineup_data
    from runitback.nfl.model.lineups import SLOT_IDX, new_token
    from runitback.nfl.sim.sampler import Simulator, scale_ratings

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--scale", type=float, default=1.5, help="team-strength scale (the experiment's folds: 1-2)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("experiments/nfl/lineups/qb_swap.csv"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.threads:
        torch.set_num_threads(a.threads)

    sim = Simulator.load(a.ckpt)
    ld = load_lineup_data(sim, DATA)
    games = pd.read_parquet(DATA / "games.parquet")
    ratings = pd.read_parquet(DATA / "ratings.parquet")
    g = games[(games.season == a.season) & (games.week > 1)].sample(a.games, random_state=0)
    q1 = SLOT_IDX[("QB", 0)]
    # the same named QBs in every home team's QB1 slot: elite starters, journeyman backups, and the
    # unknown "new QB" token. Career games: each player's at the end of the fold's data.
    career = ld.pg.groupby("player_id").career_games.max()
    refs = {"Mahomes": "00-0033873", "J.Allen": "00-0034857", "C.Rush": "00-0033662",
            "T.Siemian": "00-0032156", "B.Rypien": "00-0034955"}
    tok = {name: ld.pvocab.token(pid, "QB", float(career.get(pid, 0))) for name, pid in refs.items()}
    tok["NEW"] = new_token("QB")
    log.info("QB tokens: %s", tok)
    rows, t0 = [], time.time()
    for i, row in enumerate(g.itertuples()):
        rg = ratings[ratings.game_id == row.game_id].set_index("side")
        pre = tuple(scale_ratings({k: float(rg.at[s, f"pre_{k}"]) for k in ("off_epa", "def_epa", "elo")}, a.scale)
                    for s in ("home", "away"))
        lu = game_lineups(ld, row, "previous")
        if lu.tokens[0, q1] == 0:
            continue
        out = dict(game_id=row.game_id, home=row.home)
        for name, t in [("base", None), *tok.items()]:
            x = copy.deepcopy(lu)
            if t is not None:
                x.tokens[0, q1] = t
            m = rollouts(sim, row, pre, x, a.sims, 1000 * i)        # same seeds in every condition
            out[f"win_{name}"] = float((m > 0).mean() + 0.5 * (m == 0).mean())
            out[f"margin_{name}"] = float(m.mean())
        rows.append(out)
        if i % 10 == 0:
            log.info("%d/%d games (%.0fs)", i + 1, len(g), time.time() - t0)
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} games, home QB1 replaced; change vs the real previous-game lineup (95% CI):")
    for name in ["base", *tok]:
        d = df[f"win_{name}"] - df.win_base
        dm = df[f"margin_{name}"] - df.margin_base
        print(f"  {name:10s} home win {df[f'win_{name}'].mean():.3f}  {d.mean():+.3f} ± {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.3f}"
              f"   margin {dm.mean():+.2f} ± {1.96 * dm.std(ddof=1) / np.sqrt(len(dm)):.2f}")


if __name__ == "__main__":
    main()
