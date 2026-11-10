import pandas as pd

from ...cfb.data.teams import load_teams
from ..views import SportView, Stat, column


def kickoff(row: pd.Series) -> str | None:
    """``YYYY-MM-DD HH:MM`` US Eastern (date only when the time is TBD)."""
    start = pd.to_datetime(row.get("start_date"), utc=True, errors="coerce")
    if pd.isna(start):
        return None
    start = start.tz_convert("America/New_York")
    if row.get("start_time_tbd") in (True, "True", "true", 1):
        return start.strftime("%Y-%m-%d")
    return start.strftime("%Y-%m-%d %H:%M")


def playoffs(games: pd.DataFrame) -> dict[int, dict[str, str]]:
    """National champion and runner-up, and the playoff semifinal losers."""
    out: dict[int, dict[str, str]] = {}
    notes = games["notes"].fillna("").astype(str)
    semifinal = games["game_type"].eq("playoff") & notes.str.contains(
        "semifinal", case=False
    )
    for row in games[games["game_type"].eq("final") | semifinal].itertuples():
        if row.game_type == "final":
            stage = "champion" if row.result > 0 else "runner_up"
        elif row.result < 0:
            stage = "semifinal"
        else:
            continue  # semifinal winners are marked by the final
        out.setdefault(int(row.season), {})[str(row.team)] = stage
    return out


def per_play(yards: str, rushes: str, passes: str):
    def value(rows: pd.DataFrame) -> pd.Series:
        plays = rows[rushes] + rows[passes]
        return rows[yards] / plays.where(plays > 0)

    return value


def third_down_pct(rows: pd.DataFrame) -> pd.Series:
    att = rows["thirdDownAtt"]
    return 100 * rows["thirdDownConv"] / att.where(att > 0)


VIEW = SportView(
    sport="cfb",
    label="CFB",
    date_label="Week",
    load_teams=load_teams,
    score=("points", "opp_points"),
    kickoff=kickoff,
    playoffs=playoffs,
    playoff_labels={
        "champion": "Won national championship",
        "runner_up": "Lost national championship",
        "semifinal": "Lost playoff semifinal",
    },
    stats=(
        Stat("points", "Points scored", column("points")),
        Stat("points_allowed", "Points allowed", column("opp_points"), 1, False),
        Stat("yards", "Total yards", column("totalYards"), 0),
        Stat("yards_allowed", "Yards allowed", column("opp_totalYards"), 0, False),
        Stat(
            "yards_per_play",
            "Yards per play",
            per_play("totalYards", "rushingAttempts", "passingAttempts"),
        ),
        Stat("passing_yards", "Passing yards", column("netPassingYards"), 0),
        Stat("rushing_yards", "Rushing yards", column("rushingYards"), 0),
        Stat("third_down_pct", "Third-down conversion %", third_down_pct, 0),
        Stat("turnovers", "Turnovers", column("turnovers"), 1, False),
        Stat("takeaways", "Takeaways", column("opp_turnovers")),
        Stat("penalty_yards", "Penalty yards", column("totalPenaltiesYards"), 0, False),
    ),
)
