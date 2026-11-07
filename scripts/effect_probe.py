"""Do a foundation's per-player effects (--all-effects) agree with the direct player values, at
every position? For each player with an embedding, his own additive term (in his group's starter
role) on a league-average pass and run play, against the group's replacement-level "new player"
token: the change in expected yards (offense: + is better; defense: the yards he takes away) and
in completion / sack rates. Correlated per group with the direct values (adjusted plus-minus fit
before the fold's season, scripts/player_values.py) and listed best / worst.

    python scripts/effect_probe.py runs/ae/fold2025/model.pt --values values_2025.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from runitback.nfl.evaluate import load_lineup_data
from runitback.nfl.model.lineups import GROUP_IDX, SIDE, new_token
from runitback.nfl.sim.sampler import Simulator
from runitback.nfl.vocab import N_YARDS, PT, RESULTS, RS, idx_to_yards

GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"]
DEFENSE = {"DL", "LB", "DB"}
PASS_SHARE = 0.58


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--values", type=Path, help="direct values (side, player_id, group, plays, value_play)")
    ap.add_argument("--data", type=Path, default=Path("data/nfl/processed"))
    ap.add_argument("--min-games", type=int, default=16)
    a = ap.parse_args()
    sim = Simulator.load(a.ckpt)
    m = sim.model.eval()
    if getattr(m, "fx", None) is None:
        raise SystemExit("this model has no --all-effects terms")
    ld = load_lineup_data(sim, a.data)
    names = pd.read_parquet(a.data / "players.parquet").set_index("player_id")
    games = ld.pg.groupby("player_id").game_id.nunique()
    yv = torch.tensor([float(idx_to_yards(i)) for i in range(N_YARDS)])
    nR = len(RESULTS)

    def play_value(pers, grp, t):
        """Expected yards and result probabilities of a league-average play of type t with only
        these players' effects, (N,) and (N, nR)."""
        hd = torch.zeros_like(pers, dtype=torch.bool)
        lr = m.fx.base_r(t) + m.all_result_effect(pers, grp, t, hd)
        p = torch.softmax(lr, -1)
        ey = torch.zeros(len(t))
        for r in range(nR):
            rr = torch.full_like(t, r)
            py = torch.softmax(m.fx.base_y(t * nR + rr) + m.all_yards_effect(pers, grp, t, hd, rr), -1)
            ey = ey + p[:, r] * (py * yv).sum(-1)
        return ey, p

    rows = []
    with torch.no_grad():
        for g in GROUPS:
            pid = [p for p in ld.pvocab.ids if names.group.get(p) == g and games.get(p, 0) >= a.min_games]
            if not pid:
                continue
            toks = torch.tensor([ld.pvocab.idx[p] for p in pid] + [new_token(g)])
            N = len(toks)
            pers = torch.zeros(N, 2 * SIDE, dtype=torch.long)
            grp = torch.zeros(N, 2 * SIDE, dtype=torch.long)
            col = SIDE if g in DEFENSE else 0
            pers[:, col], grp[:, col] = toks, GROUP_IDX[g] + 1
            out = {}
            for kind in ("pass", "run"):
                t = torch.full((N,), PT[kind])
                ey, p = play_value(pers, grp, t)
                out[kind] = (ey - ey[-1], p - p[-1])
            dy = PASS_SHARE * out["pass"][0] + (1 - PASS_SHARE) * out["run"][0]
            sign = -1.0 if g in DEFENSE else 1.0
            for i, p in enumerate(pid):
                rows.append(dict(group=g, player_id=p, name=names.name.get(p), games=int(games.get(p, 0)),
                                 d_yards=float(sign * dy[i]),
                                 d_comp=float(sign * out["pass"][1][i, RS["complete"]]),
                                 d_sack=float(sign * out["pass"][1][i, RS["sack"]])))
    df = pd.DataFrame(rows)
    print(f"{len(df)} players with {a.min_games}+ games: each one's own effect on a league-average play vs "
          "a replacement-level player (yards per play; defenders: yards taken away)")
    val = None
    if a.values and a.values.exists():
        v = pd.read_csv(a.values)
        val = {k: d.drop_duplicates("player_id").set_index("player_id").value_play for k, d in v.groupby("side")}
    for g, d in df.groupby("group", sort=False):
        line = f"  {g:3s} n={len(d):4d}  sd Δyards {d.d_yards.std():.3f}"
        if val is not None:
            j = d.assign(v=d.player_id.map(val["def" if g in DEFENSE else "off"])).dropna(subset=["v"])
            if len(j) > 10:
                line += f"   corr with direct value {j.d_yards.corr(j.v):+.2f} (n={len(j)})"
        d = d.sort_values("d_yards")
        line += "   best: " + ", ".join(d.name.iloc[-3:][::-1].fillna("?")) + "   worst: " + ", ".join(d.name.iloc[:2].fillna("?"))
        print(line)


if __name__ == "__main__":
    main()
