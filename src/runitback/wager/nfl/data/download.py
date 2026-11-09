from pathlib import Path

import pandas as pd

from .features import STATS_COLUMNS
from .injuries import INJURY_COLUMNS, injury_features
from .names import move_map
from .pbp import PBP_COLUMNS, load_efficiency

LOCATIONS = {"Home": 1, "Away": -1, "Neutral": 0}


def invert_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game from nflverse's one-row-per-game schedule."""
    home = schedule.copy()
    away = schedule.copy()

    away["result"] *= -1
    away["spread_line"] *= -1
    away["location"] = away["location"].replace("Home", "Away")

    home_col_renamer = {
        col: col.removeprefix("home_") for col in home if col.startswith("home_")
    }
    home_col_renamer.update(
        {col: col.replace("away_", "opp_") for col in home if col.startswith("away_")}
    )
    home = home.rename(columns=home_col_renamer)

    away_col_renamer = {
        col: col.removeprefix("away_") for col in away if col.startswith("away_")
    }
    away_col_renamer.update(
        {col: col.replace("home_", "opp_") for col in away if col.startswith("home_")}
    )
    away = away.rename(columns=away_col_renamer)

    games = pd.concat([home, away], ignore_index=True)
    games["location"] = games["location"].map(LOCATIONS)
    return games


def build_games(
    schedule: pd.DataFrame,
    team_stats: pd.DataFrame,
    injuries: pd.DataFrame | None = None,
    efficiency: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join weekly team stats onto both perspectives of every scheduled game.

    The schedule is the source of truth for which games exist, so games whose
    stats nflverse lacks keep their result with missing stats, and each row's
    opponent stats always come from the opponent's own stats row.
    ``injuries`` (:func:`~.injuries.injury_features` of these games) adds
    ``inj_*`` and ``opp_inj_*`` the same way, and ``efficiency``
    (:func:`~.pbp.load_efficiency`) the play-by-play ``pbp_*`` /
    ``opp_pbp_*`` columns.
    """
    schedule = schedule.copy()
    for col in ("home_team", "away_team"):
        schedule[col] = schedule[col].replace(move_map)
    games = invert_schedule(schedule)

    stats = team_stats.dropna(subset=["team"]).copy()
    stats["team"] = stats["team"].replace(move_map)
    stats = stats.drop_duplicates(["season", "week", "team"])
    stats = stats[["season", "week", "team"] + STATS_COLUMNS]

    games = games.merge(stats, on=["season", "week", "team"], how="left")
    opp_stats = stats.rename(
        columns={"team": "opp_team", **{col: f"opp_{col}" for col in STATS_COLUMNS}}
    )
    games = games.merge(opp_stats, on=["season", "week", "opp_team"], how="left")
    if efficiency is not None:
        eff = efficiency.copy()
        eff["team"] = eff["team"].replace(move_map)
        eff = eff.drop_duplicates(["season", "week", "team"])
        eff = eff[["season", "week", "team", *PBP_COLUMNS]]
        games = games.merge(eff, on=["season", "week", "team"], how="left")
        games = games.merge(
            eff.rename(
                columns={"team": "opp_team", **{c: f"opp_{c}" for c in PBP_COLUMNS}}
            ),
            on=["season", "week", "opp_team"],
            how="left",
        )
    if injuries is not None:
        injuries = injuries[["season", "week", "team"] + INJURY_COLUMNS]
        games = games.merge(injuries, on=["season", "week", "team"], how="left")
        games = games.merge(
            injuries.rename(
                columns={
                    "team": "opp_team",
                    **{col: f"opp_{col}" for col in INJURY_COLUMNS},
                }
            ),
            on=["season", "week", "opp_team"],
            how="left",
        )
    return games.sort_values(["season", "week", "game_id", "location"]).reset_index(
        drop=True
    )


def download(output_file: str = None, seasons: list[int] = None) -> pd.DataFrame:
    """Download nflverse schedules, weekly team stats, injury features and
    play-by-play efficiency (all seasons by default)."""
    import nflreadpy

    schedule = nflreadpy.load_schedules(seasons or True).to_pandas()
    team_stats = nflreadpy.load_team_stats(seasons=seasons or True).to_pandas()
    games = build_games(schedule, team_stats)
    # injury reports start in 2009, snap counts in 2012
    have = sorted(int(s) for s in games["season"].unique() if s >= 2012)
    injuries = injury_features(
        games,
        nflreadpy.load_injuries(have).to_pandas(),
        nflreadpy.load_snap_counts(have).to_pandas(),
        nflreadpy.load_players().to_pandas(),
    )
    played_seasons = sorted(
        int(s) for s in games.loc[games["result"].notna(), "season"].unique()
    )
    efficiency = load_efficiency(played_seasons)
    games = build_games(schedule, team_stats, injuries, efficiency)

    if output_file is not None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        games.to_csv(output_file, index=False)

    return games
