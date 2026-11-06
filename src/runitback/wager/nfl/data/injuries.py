"""Pre-game injury features: how much of each unit's usual playing time is
expected to be missing.

For every team and game, each player on that week's injury report counts
``P(misses | report status) x`` his share of the unit's snaps over the
team's previous ``WINDOW`` games, summed per unit (1.0 is one full-time
player). Shares are relative to the team's recent games, the same games the
models already learn from: a player who has been out for weeks has a small
share, so only news since then counts.

Injury reports are published before kickoff and the shares use earlier
games only, so the features are safe for backtests. Snap counts start in
2012 and the injury reports in 2009 (with ``Probable`` until 2015); a game
whose window has no snap data gets missing values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .names import move_map

# P(no snaps | final report status), from 2016-2020 reports joined to snap
# counts (Probable, retired in 2016, from 2013-2015).
MISS_PROBABILITY = {
    "Out": 1.0,
    "Doubtful": 0.99,
    "Questionable": 0.33,
    "Probable": 0.05,
}

UNITS = {
    "qb": {"QB"},
    "ol": {"T", "G", "C", "OT", "OG", "OL"},
    "skill": {"RB", "FB", "HB", "WR", "TE"},
    "front": {"DE", "DT", "NT", "DL", "EDGE", "LB", "OLB", "ILB", "MLB"},
    "secondary": {"CB", "S", "FS", "SS", "SAF", "DB"},
}
OFFENSE_UNITS = {"qb", "ol", "skill"}
INJURY_COLUMNS = [f"inj_{unit}" for unit in UNITS]
OPP_INJURY_COLUMNS = [f"opp_{col}" for col in INJURY_COLUMNS]
WINDOW = 4

_UNIT_OF = {
    position: unit for unit, positions in UNITS.items() for position in positions
}
_KEYS = ["season", "week", "team"]


def _team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Each team's games in order: ``season, week, team, n`` (n = 0, 1, ...)."""
    keys = games[_KEYS].drop_duplicates().sort_values(["team", "season", "week"])
    return keys.assign(n=keys.groupby("team").cumcount())


def injury_features(
    games: pd.DataFrame,
    injuries: pd.DataFrame,
    snaps: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    """``season, week, team`` + one ``inj_<unit>`` column per unit.

    Args:
        games: Team-perspective games (``season``, ``week``, ``team``), played
            and upcoming.
        injuries: nflverse injury reports (``load_injuries``).
        snaps: nflverse snap counts (``load_snap_counts``).
        players: nflverse players (``gsis_id`` -> ``pfr_id``).
    """
    order = _team_games(games)

    snaps = snaps.assign(team=snaps["team"].replace(move_map))
    snaps = snaps.merge(order, on=_KEYS)[
        ["team", "n", "pfr_player_id", "offense_pct", "defense_pct"]
    ].drop_duplicates(["team", "n", "pfr_player_id"])
    covered = snaps[["team", "n"]].drop_duplicates()

    reports = injuries[injuries["report_status"].isin(MISS_PROBABILITY)].copy()
    reports["team"] = reports["team"].replace(move_map)
    reports["unit"] = reports["position"].map(_UNIT_OF)
    reports = (
        reports.dropna(subset=["unit"])
        .sort_values("date_modified")
        .drop_duplicates(["season", "week", "team", "gsis_id"], keep="last")
        .merge(players[["gsis_id", "pfr_id"]].dropna(), on="gsis_id")
        .merge(order, on=_KEYS)
    )

    share = np.zeros(len(reports))
    games_seen = np.zeros(len(order))
    offense = reports["unit"].isin(OFFENSE_UNITS).to_numpy()
    for lag in range(1, WINDOW + 1):
        earlier = reports[["team", "n", "pfr_id"]].assign(n=reports["n"] - lag)
        pct = earlier.merge(
            snaps.rename(columns={"pfr_player_id": "pfr_id"}),
            on=["team", "n", "pfr_id"],
            how="left",
        )
        # a player absent from a covered game played no snaps in it
        share += np.nan_to_num(
            np.where(offense, pct["offense_pct"], pct["defense_pct"]).astype(float)
        )
        seen = order[["team", "n"]].assign(n=order["n"] - lag)
        games_seen += (
            seen.merge(covered.assign(c=1), on=["team", "n"], how="left")["c"]
            .fillna(0)
            .to_numpy()
        )

    window = order[["team", "n"]].assign(games_seen=games_seen)
    reports = reports.merge(window, on=["team", "n"])
    reports["missing"] = (
        reports["report_status"].map(MISS_PROBABILITY)
        * share
        / reports["games_seen"].replace(0, np.nan)
    )
    by_unit = reports.pivot_table(
        index=_KEYS, columns="unit", values="missing", aggfunc="sum"
    ).reindex(columns=list(UNITS))
    out = order.assign(games_seen=games_seen).merge(
        by_unit.add_prefix("inj_").reset_index(), on=_KEYS, how="left"
    )
    known = out["games_seen"] > 0
    out[INJURY_COLUMNS] = out[INJURY_COLUMNS].fillna(0.0).where(known)
    return out[_KEYS + INJURY_COLUMNS].reset_index(drop=True)
