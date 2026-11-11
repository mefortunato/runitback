"""A small synthetic college league for the C3 tests: ESPN-style player box rows."""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from runitback.college.data import STATS


def league(seasons=(2021, 2022, 2023), teams=8, roster=10, weeks=8, seed=0) -> pd.DataFrame:
    """Raw box rows (the release's columns) for a round of games every week. Each player has a skill
    that raises his team's scoring and his own counts; one player (id 1000) transfers from team 0 to
    team 1 after the first season."""
    rng = np.random.default_rng(seed)
    players = {t: [1000 + t * roster + i for i in range(roster)] for t in range(teams)}
    skill = {p: rng.normal(0, 1) for ps in players.values() for p in ps}
    skill[1000] = 3.0
    rows, gid = [], 1
    for s in seasons:
        if s > seasons[0]:
            players[0] = [p for p in players[0] if p != 1000]
            if 1000 not in players[1]:
                players[1] = players[1] + [1000]
        start = datetime.date(s - 1, 11, 6)
        for w in range(weeks):
            order = rng.permutation(teams)
            day = start + datetime.timedelta(days=7 * w + 1)   # Tuesdays
            for a, b in zip(order[::2], order[1::2]):
                gid += 1
                lines = {}
                for t in (a, b):
                    ps = players[t]
                    out = rng.random(len(ps)) < 0.08               # didn't play
                    mins = np.where(out, 0.0, rng.uniform(5, 35, len(ps)))
                    mins = np.where(out, 0.0, mins * 200 / mins.sum())
                    lines[t] = (ps, out, mins)
                pts = {}
                for t in (a, b):
                    ps, out, mins = lines[t]
                    q = sum(skill[p] * m / 40 for p, m in zip(ps, mins))
                    pts[t] = int(70 + 3 * q + rng.normal(0, 8))
                for t, o in ((a, b), (b, a)):
                    ps, out, mins = lines[t]
                    for i, (p, m) in enumerate(zip(ps, mins)):
                        r = {"game_id": gid, "season": s, "season_type": 2, "game_date": day,
                             "athlete_id": float(p), "athlete_display_name": f"Player {p}",
                             "team_id": 100 + t, "team_location": f"School {t}", "opponent_team_id": 100 + o,
                             "home_away": "home" if t == a else "away", "minutes": None if out[i] else m,
                             "starter": i < 5, "did_not_play": bool(out[i]),
                             "athlete_position_abbreviation": ["G", "F", "C", "PG", "ATH"][i % 5],
                             "team_score": pts[t], "opponent_team_score": pts[o]}
                        k = m / 40
                        base = {"points": 12, "field_goals_made": 5, "field_goals_attempted": 11,
                                "three_point_field_goals_made": 1.5, "free_throws_made": 2.5,
                                "free_throws_attempted": 3.5, "offensive_rebounds": 1.5,
                                "defensive_rebounds": 4, "assists": 2.5, "steals": 1, "blocks": 0.6,
                                "turnovers": 2, "fouls": 2.5}
                        for c in STATS.values():
                            lam = max(base.get(c, 1.0) * k * (1 + 0.25 * skill[p] * (c != "turnovers")), 0)
                            r[c] = None if out[i] else float(rng.poisson(lam))
                        rows.append(r)
    return pd.DataFrame(rows)
