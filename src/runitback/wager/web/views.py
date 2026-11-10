"""How each sport is presented in the web app.

A :class:`SportView` is the web-side counterpart of
:class:`runitback.wager.core.sport.Sport`: which stats to show, what a "date" is
called, and where team names, colors and logos come from. Adding a sport to
the app means registering one view; the export and frontend are generic.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

# How a team's season ended, for the last rounds of the playoffs.
PLAYOFF_STAGES = ("champion", "runner_up", "semifinal")

# Team metadata columns every view's ``load_teams`` returns (it may add
# ``abbr``, a short chip label, where team keys are long names).
TEAM_COLUMNS = (
    "team", "name", "nickname", "color", "color2", "logo", "conference", "division",
)  # fmt: skip


@dataclass(frozen=True)
class Stat:
    """A per-game stat shown in team comparisons.

    Attributes:
        key: Identifier in the exported data.
        label: Display label.
        value: Per-row value from the team's perspective (games table rows).
        decimals: Digits shown.
        higher_is_better: Which side of a comparison is highlighted.
    """

    key: str
    label: str
    value: Callable[[pd.DataFrame], pd.Series]
    decimals: int = 1
    higher_is_better: bool = True


def column(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda rows: rows[name]


def total(*names: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda rows: rows[list(names)].sum(axis=1, min_count=1)


def ratio(numerator: str, denominator: str, scale: float = 100.0):
    """``scale * numerator / denominator`` per row (NaN where it is 0)."""

    def value(rows: pd.DataFrame) -> pd.Series:
        den = rows[denominator]
        return scale * rows[numerator] / den.where(den > 0)

    return value


def calendar_date(row: pd.Series) -> str | None:
    """``Oct 28``: for sports whose dates are game days (``game_date``)."""
    day = pd.to_datetime(row.get("game_date"), errors="coerce")
    return None if pd.isna(day) else f"{day:%b} {day.day}"


def eastern_kickoff(row: pd.Series) -> str | None:
    """``YYYY-MM-DD HH:MM`` US Eastern, from ``start_date`` (UTC)."""
    start = pd.to_datetime(row.get("start_date"), utc=True, errors="coerce")
    if pd.isna(start):
        return None
    return start.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M")


def bracket_playoffs(
    games: pd.DataFrame,
    playoff: pd.Series,
    wins_needed: Callable[[int], int],
    season: str = "season",
    date: str = "day",
) -> dict[int, dict[str, str]]:
    """Champion, runner-up and semifinal losers from a bracket's results.

    The final is the series (or game) of the season's last ``playoff``
    game, and each finalist's last earlier playoff opponent lost the
    semifinal. A season is marked once a finalist has won
    ``wins_needed(season)`` final games (1 for a one-game final).
    """
    out: dict[int, dict[str, str]] = {}
    rows = games[playoff].sort_values([season, date], kind="stable")
    for year, season_rows in rows.groupby(season):
        last = season_rows.iloc[-1]
        finalists = {last["team"], last["opp_team"]}
        finals = season_rows[
            season_rows["team"].isin(finalists)
            & season_rows["opp_team"].isin(finalists)
        ]
        wins = finals[finals["result"] > 0]["team"].value_counts()
        if wins.empty or wins.max() < wins_needed(int(year)):
            continue  # the playoffs are still on
        champion = wins.idxmax()
        stages = {
            str(team): "champion" if team == champion else "runner_up"
            for team in finalists
        }
        for team in finalists:
            earlier = season_rows[
                season_rows["team"].eq(team) & ~season_rows["opp_team"].isin(finalists)
            ]
            if len(earlier):
                stages.setdefault(str(earlier.iloc[-1]["opp_team"]), "semifinal")
        out[int(year)] = stages
    return out


@dataclass(frozen=True)
class SportView:
    """Presentation of one sport.

    Attributes:
        sport: Name in the sport registry (``core.sport``).
        label: Display name, e.g. ``"NFL"``.
        date_label: What the schema's date column is called, e.g. ``"Week"``.
        stats: Stats compared on game and challenge pages.
        load_teams: Returns team metadata with :data:`TEAM_COLUMNS`.
        score: ``(team, opponent)`` points columns of the games table.
        kickoff: Kickoff time of a games-table row, for display.
        recent_games: How many recent games to show per team.
        playoffs: Played games -> ``{season: {team: stage}}`` for the teams
            that reached the last rounds (stages from
            :data:`PLAYOFF_STAGES`); ``None`` if the sport has no playoffs.
        playoff_labels: Display label per stage.
        date_name: Display name of a games-table row's date (e.g. its
            calendar day, where dates are game days); ``None``: the site
            shows ``"<date_label> <date>"``.
        upcoming_dates: How many upcoming dates to export (``None``: every
            date left in the schedule).
        recent_dates: How many of the last played dates the recent-games
            page shows (e.g. 4 weeks, or 14 game days).
        split_season: Seasons span two calendar years and are keyed by the
            year they end (the site shows 2026 as "2025–26").
        ranking_every: Power rankings at most this often, in the schema's
            date units (:func:`runitback.wager.web.build.ranking_dates`); ``None``:
            before every date. For sports with hundreds of teams and a game
            day most days, where rating every pair of teams before each
            date is the most expensive part of a backtest.
    """

    sport: str
    label: str
    date_label: str
    stats: tuple[Stat, ...]
    load_teams: Callable[[], pd.DataFrame]
    score: tuple[str, str]
    kickoff: Callable[[pd.Series], str | None] = lambda row: None
    recent_games: int = 5
    playoffs: Callable[[pd.DataFrame], dict[int, dict[str, str]]] | None = None
    playoff_labels: dict[str, str] = field(default_factory=dict)
    date_name: Callable[[pd.Series], str | None] | None = None
    upcoming_dates: int | None = None
    recent_dates: int = 4
    split_season: bool = False
    ranking_every: int | None = None


BUILTIN_VIEWS = ("nfl", "nba", "cbb", "wnba", "wcbb")  # nhl, mlb and cfb are ported later
_views: dict[str, SportView] = {}


def register_view(view: SportView) -> SportView:
    _views[view.sport] = view
    return view


def get_view(sport: str) -> SportView:
    if sport not in _views and sport in BUILTIN_VIEWS:
        register_view(importlib.import_module(f"runitback.wager.web.sports.{sport}").VIEW)
    if sport not in _views:
        raise KeyError(f"no web view for sport {sport!r}")
    return _views[sport]
