"""A synthetic NFL-shaped play table for the player-values tests (no data files needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from runitback.nfl.data.players import GROUPS

# per team: group -> players on the field each play (roster has a couple more per group)
ON_FIELD = {"QB": 1, "RB": 1, "WR": 3, "TE": 1, "OL": 5, "DL": 4, "LB": 3, "DB": 4}
ROSTER_EXTRA = 2


def synthetic_plays(seed: int = 0, teams: int = 4, seasons=(2019, 2020, 2021, 2022), weeks: int = 6,
                    plays_per_game: int = 60):
    """(PlayData, special) with known per-player effects: every play's EPA is the offense's
    effects minus the defense's plus situation terms and noise."""
    from runitback.nfl.player_values import PlayData

    rng = np.random.default_rng(seed)
    roster, effect = {}, {}
    for t in range(teams):
        for g, n in ON_FIELD.items():
            roster[(t, g)] = [f"T{t}{g}{i}" for i in range(n + ROSTER_EXTRA)]
            for p in roster[(t, g)]:
                effect[p] = rng.normal(0, 0.15 if g == "QB" else 0.03)
    off_groups, def_groups = ("QB", "RB", "WR", "TE", "OL"), ("DL", "LB", "DB")
    rows, offs, dfns = [], [], []
    for s in seasons:
        for w in range(1, weeks + 1):
            order = rng.permutation(teams)
            for gi_ in range(teams // 2):
                h, a = order[2 * gi_], order[2 * gi_ + 1]
                gid = f"{s}_{w:02d}_{h}_{a}"
                for k in range(plays_per_game):
                    o, d = (h, a) if k % 2 == 0 else (a, h)
                    op = [p for g in off_groups for p in rng.choice(roster[(o, g)], ON_FIELD[g], replace=False)]
                    dp = [p for g in def_groups for p in rng.choice(roster[(d, g)], ON_FIELD[g], replace=False)]
                    if rng.random() < 0.03:     # rarely used players: replacement level
                        op[0], dp[-1] = f"T{o}QB9", f"T{d}DB9"
                    down, ytg, yl = int(rng.integers(1, 5)), int(rng.integers(1, 15)), int(rng.integers(1, 99))
                    epa = (sum(effect.get(p, -0.1) for p in op) - sum(effect.get(p, 0.1) for p in dp) + 0.02 * (down == 1)
                           + rng.normal(0, 1.2))
                    if k % 97 == 0:
                        epa *= 8          # a few outliers past the clip
                    rows.append((gid, s * 100 + w, s, down, ytg, yl, epa))
                    offs.append(op)
                    dfns.append(dp)
    plays = pd.DataFrame(rows, columns=["game_id", "t", "season", "down", "ytg", "yl", "epa"])
    ids = np.array(sorted({p for lst in offs + dfns for p in lst}))
    code = {p: i for i, p in enumerate(ids)}
    gi = {g: i for i, g in enumerate(GROUPS)}
    group = np.array([gi[p[2:-1]] for p in ids], dtype=np.int32)
    group[ids == "T0WR4"] = -1                  # a player with no known group is ignored

    def to_codes(lists, width=13):
        out = np.full((len(lists), width), -1, dtype=np.int32)
        for i, lst in enumerate(lists):
            out[i, :len(lst)] = [code[p] for p in lst]
        return out

    data = PlayData(plays, to_codes(offs), to_codes(dfns), ids, group)
    sp_rows = []
    for s in seasons:
        for w in range(1, weeks + 1):
            for t in range(teams):
                for kind, n in (("K", 2), ("P", 4)):
                    for _ in range(n):
                        sp_rows.append((s * 100 + w, s, kind, f"T{t}{kind}{int(rng.integers(0, 2))}",
                                        float(rng.normal(0.1 * t, 1.0))))
    special = pd.DataFrame(sp_rows, columns=["t", "season", "kind", "player_id", "epa"])
    return data, special, effect


# cut-offs for the golden fits: early (short window), mid-season, and a later season
BEFORES = (201904, 202003, 202101, 202206)
