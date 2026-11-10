import pandas as pd

from ...nhl.data.teams import load_teams
from ..views import (
    SportView,
    Stat,
    bracket_playoffs,
    calendar_date,
    column,
    eastern_kickoff,
    ratio,
)


def playoffs(games: pd.DataFrame) -> dict[int, dict[str, str]]:
    """Stanley Cup winner and runner-up, and the conference finals losers,
    from the bracket."""
    return bracket_playoffs(games, games["game_type"].eq("playoffs"), lambda s: 4)


def save_pct(rows: pd.DataFrame) -> pd.Series:
    shots = rows["opp_shots"]
    return 100 * (1 - rows["opp_goals"] / shots.where(shots > 0))


VIEW = SportView(
    sport="nhl",
    label="NHL",
    date_label="Day",
    load_teams=load_teams,
    score=("goals", "opp_goals"),
    kickoff=eastern_kickoff,
    playoffs=playoffs,
    playoff_labels={
        "champion": "Won Stanley Cup",
        "runner_up": "Lost Stanley Cup Final",
        "semifinal": "Lost conference final",
    },
    date_name=calendar_date,
    upcoming_dates=7,
    recent_dates=14,
    split_season=True,
    recent_games=8,
    stats=(
        Stat("goals", "Goals scored", column("goals"), 2),
        Stat("goals_allowed", "Goals allowed", column("opp_goals"), 2, False),
        Stat("shots", "Shots", column("shots")),
        Stat("shots_allowed", "Shots allowed", column("opp_shots"), 1, False),
        Stat("shooting_pct", "Shooting %", ratio("goals", "shots")),
        Stat("save_pct", "Save %", save_pct),
        Stat("pp_pct", "Power play %", ratio("pp_goals", "pp_opportunities")),
        Stat("pk_pct", "Penalty kill %", column("pk_pct")),
        Stat("faceoff_pct", "Faceoff %", column("faceoff_pct")),
        Stat("hits", "Hits", column("hits")),
        Stat("blocked_shots", "Blocked shots", column("blocked_shots")),
        Stat("giveaways", "Giveaways", column("giveaways"), 1, False),
    ),
)
