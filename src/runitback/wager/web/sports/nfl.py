import pandas as pd

from ...nfl.data.teams import load_teams
from ..views import SportView, Stat, column, total

TURNOVERS = (
    "passing_interceptions",
    "sack_fumbles_lost",
    "rushing_fumbles_lost",
    "receiving_fumbles_lost",
)


def kickoff(row: pd.Series) -> str | None:
    """``YYYY-MM-DD HH:MM`` (US Eastern, as nflverse lists it)."""
    day, time = row.get("gameday"), row.get("gametime")
    if not isinstance(day, str):
        return None
    return f"{day} {time}" if isinstance(time, str) else day


def playoffs(games: pd.DataFrame) -> dict[int, dict[str, str]]:
    """Super Bowl winner and loser, and the conference championship losers."""
    out: dict[int, dict[str, str]] = {}
    last_rounds = games[games["game_type"].isin(["SB", "CON"])]
    for row in last_rounds.itertuples():
        if row.game_type == "SB":
            stage = "champion" if row.result > 0 else "runner_up"
        elif row.result < 0:
            stage = "semifinal"
        else:
            continue  # conference champions are marked by the Super Bowl
        out.setdefault(int(row.season), {})[str(row.team)] = stage
    return out


VIEW = SportView(
    sport="nfl",
    label="NFL",
    date_label="Week",
    load_teams=load_teams,
    score=("score", "opp_score"),
    kickoff=kickoff,
    playoffs=playoffs,
    playoff_labels={
        "champion": "Won Super Bowl",
        "runner_up": "Lost Super Bowl",
        "semifinal": "Lost conference championship",
    },
    stats=(
        Stat("points", "Points scored", column("score")),
        Stat("points_allowed", "Points allowed", column("opp_score"), 1, False),
        Stat("yards", "Total yards", total("passing_yards", "rushing_yards"), 0),
        Stat(
            "yards_allowed",
            "Yards allowed",
            total("opp_passing_yards", "opp_rushing_yards"),
            0,
            False,
        ),
        Stat("passing_yards", "Passing yards", column("passing_yards"), 0),
        Stat("rushing_yards", "Rushing yards", column("rushing_yards"), 0),
        Stat("passing_epa", "Passing EPA", column("passing_epa")),
        Stat("rushing_epa", "Rushing EPA", column("rushing_epa")),
        Stat("turnovers", "Turnovers", total(*TURNOVERS), 1, False),
        Stat("takeaways", "Takeaways", total(*(f"opp_{c}" for c in TURNOVERS))),
        Stat("sacks", "Sacks", column("def_sacks")),
        Stat("penalty_yards", "Penalty yards", column("penalty_yards"), 0, False),
    ),
)
