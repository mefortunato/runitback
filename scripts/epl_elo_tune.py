"""Tune the football Elo (`EloParams(mov="football")`) on the EPL, without leakage.

    python scripts/epl_elo_tune.py [--cache data/epl/epl_cache]

Grid-searches `k`, `home`, `regress` and the first rating of a new team over the tuning seasons
(2006-2016 by default, by ending year), with 2001-2005 as burn-in: Elo is replayed from 2001 and only
the tuning seasons are scored, so every season used is before the first test season (2022). For each
setting:

- `points_per_elo` is the least-squares slope of the goal difference on the pre-game rating gap
  (home edge included) over the tuning seasons;
- the margin is that slope times the gap, and its ordered probit (the spread and the draw cut,
  `fit_outcome_calibration`) turns it into win / draw / loss probabilities;
- the score is their ranked probability score (RPS). The probit's two parameters are fit on the same
  seasons: two numbers on about 4,000 games, shared by every setting.

Then, with the best setting, it reports the home edge that wins on three windows, to show what the
2020-21 season (no crowds, ending year 2021) does to it: 2006-2016 (the tuning window), 2006-2021, and
2006-2021 without 2021.
"""

from __future__ import annotations

import argparse
import itertools

import pandas as pd

from runitback.wager.core.metrics import (
    fit_outcome_calibration,
    outcome_probabilities,
    three_way_scores,
)
from runitback.wager.core.process import process
from runitback.wager.core.sport import get_sport
from runitback.wager.epl.data.download import download
from runitback.wager.models.baselines import EloModel, EloParams

GRID = {
    "k": (10.0, 15.0, 20.0, 25.0, 30.0),
    # the plan's grid was home 40-85 and regress 0.1-0.33; its best settings sat on those edges,
    # so both are extended (regress down to 0, home up to 100)
    "home": (40.0, 55.0, 70.0, 85.0, 100.0),
    "regress": (0.0, 0.05, 0.1, 0.2, 0.33),
    "new_team_offset": (0.0, 50.0, 100.0, 150.0),  # below the mean
}
MEAN = EloParams().mean


def params_for(k, home, regress, new_team_offset) -> EloParams:
    return EloParams(
        k=k,
        home=home,
        regress=regress,
        mean=MEAN,
        mov="football",
        new_team=MEAN - new_team_offset,
    )


def gaps(games: pd.DataFrame, params: EloParams) -> pd.DataFrame:
    """Every played home row's season, goal difference and pre-game rating gap."""
    schema = get_sport("epl").schema
    gap = EloModel(schema, params).replay(games)
    rows = games.loc[gap.index]
    home = rows[schema.location] == 1
    return pd.DataFrame(
        {"season": rows[schema.season], "y": rows[schema.target], "gap": gap}
    )[home]


def score(frame: pd.DataFrame, seasons) -> dict[str, float]:
    """RPS of the probit probabilities of the rating gap, over ``seasons``."""
    window = frame[frame["season"].isin(list(seasons))]
    y, gap = window["y"].to_numpy(), window["gap"].to_numpy()
    slope = float((gap * y).sum() / (gap * gap).sum())  # goals per Elo point
    sigma, cut = fit_outcome_calibration(y, slope * gap)
    win, draw = outcome_probabilities(slope * gap, sigma, cut)
    rps = three_way_scores(y, win, draw)["rps"]
    return {"rps": float(rps.mean()), "points_per_elo": slope, "sigma": sigma, "cut": cut,
            "n": len(window)}  # fmt: skip


def search(games: pd.DataFrame, seasons) -> pd.DataFrame:
    rows = []
    for values in itertools.product(*GRID.values()):
        setting = dict(zip(GRID, values, strict=True))
        rows.append({**setting, **score(gaps(games, params_for(**setting)), seasons)})
    return pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)


def home_windows(games: pd.DataFrame, best: dict) -> pd.DataFrame:
    """The home edge with the best RPS on three windows, the rest of the setting held."""
    windows = {
        "2006-2016 (tuning window)": range(2006, 2017),
        "2006-2021": range(2006, 2022),
        "2006-2021 without 2021": [s for s in range(2006, 2022) if s != 2021],
    }
    rows = []
    for name, seasons in windows.items():
        scores = {
            home: score(gaps(games, params_for(**{**best, "home": home})), seasons)
            for home in range(20, 111, 10)
        }
        top = min(scores, key=lambda h: scores[h]["rps"])
        rows.append({"window": name, "best_home": top, "rps": scores[top]["rps"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="data/epl/epl_cache")
    parser.add_argument("--first", type=int, default=2006)
    parser.add_argument("--last", type=int, default=2016)
    args = parser.parse_args()
    games = process("epl", download(cache_dir=args.cache))
    games = games[games["season"] <= max(args.last, 2021)]
    table = search(games, range(args.first, args.last + 1))
    fmt = {"float_format": lambda v: f"{v:.5f}"}
    print("Best 10 of", len(table), "\n" + table.head(10).to_string(**fmt))
    for name in GRID:
        print(
            f"\nMean RPS by {name}\n"
            + table.groupby(name)["rps"].mean().to_string(**fmt)
        )
    best = table.iloc[0][list(GRID)].to_dict()
    print("\nNFL settings, as a reference:", end=" ")
    nfl = score(gaps(games, EloParams()), range(args.first, args.last + 1))
    print({k: round(v, 5) for k, v in nfl.items()})
    print(
        "\nHome edge on three windows (best setting otherwise)\n"
        + home_windows(games, best).to_string(index=False, **fmt)
    )
