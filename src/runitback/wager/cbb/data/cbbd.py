"""Men's college basketball betting lines from CollegeBasketballData.

https://api.collegebasketballdata.com (key: ``CBBD_API_KEY``, free at
CollegeBasketballData.com). ``/lines`` returns every book's lines of up to
3,000 games per request, so a season takes about a dozen requests instead
of ESPN's one per game (~5,000 a season, which gets a runner refused with
403). Its games carry CBBD's own ids; ``/games`` maps them to ESPN's
(``sourceId``), the games table's ``game_id``.

CBBD has the spread, total and both moneylines (and opening lines), but no
spread or over/under prices: those stay empty. It lists some games without
any lines (about 6% of 2024-25, and whole stretches of older seasons) that
ESPN has; those are left to ESPN (see :func:`season_lines`).
"""

from __future__ import annotations

import datetime
import logging
import os

import numpy as np
import pandas as pd

from ... import espn

logger = logging.getLogger(__name__)

API = "https://api.collegebasketballdata.com"
KEY_ENV = "CBBD_API_KEY"

# The most games one request returns; a window that fills it is split.
LIMIT = 3000


def api_key() -> str | None:
    return os.environ.get(KEY_ENV) or None


def _rows(path: str, key: str, **params) -> list[dict]:
    response = espn._get(
        f"{API}/{path}", headers={"Authorization": f"Bearer {key}"}, **params
    )
    response.raise_for_status()
    return response.json()


def _windows(
    season: int, first=None, last=None
) -> list[tuple[datetime.datetime, datetime.datetime]]:
    """A season (by ending year, October to May), or ``[first, last]`` of
    it, in windows of at most a month (well under the 3,000 games a
    request returns; a fuller window is split)."""
    start = pd.Timestamp(season - 1, 10, 1, tz="UTC")
    end = pd.Timestamp(season, 5, 1, tz="UTC")
    start = max(start, first) if first is not None else start
    end = min(end, last) if last is not None else end
    windows = []
    while start < end:
        stop = min(start + pd.Timedelta(days=31), end)
        windows.append((start.to_pydatetime(), stop.to_pydatetime()))
        start = stop
    return windows


def _fetch(
    path: str,
    key: str,
    season: int,
    start: datetime.datetime,
    end: datetime.datetime,
) -> list[dict]:
    """Every row of ``path`` in ``[start, end]``, splitting full windows."""
    rows = _rows(
        path,
        key,
        season=season,
        startDateRange=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        endDateRange=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    if len(rows) >= LIMIT and end - start > datetime.timedelta(hours=2):
        middle = start + (end - start) / 2
        return _fetch(path, key, season, start, middle) + _fetch(
            path, key, season, middle, end
        )
    return rows


def fetch_season(path: str, key: str, season: int, first=None, last=None) -> list[dict]:
    """Every row of ``path`` (``games`` or ``lines``) in ``season`` (between
    ``first`` and ``last`` when given), once per game id."""
    rows = {}
    for start, end in _windows(season, first, last):
        for row in _fetch(path, key, season, start, end):
            rows[row.get("id", row.get("gameId"))] = row
    return list(rows.values())


def _book(name: str) -> str:
    return "".join(name.lower().split())


PREFERENCE = {_book(name): i for i, name in enumerate(espn.PROVIDERS)}
NOT_BOOKS = {_book(name) for name in espn.NOT_BOOKS}


def _books(lines: list[dict]) -> list[dict]:
    books = [
        line
        for line in lines
        if (name := _book(line.get("provider") or ""))
        and name not in NOT_BOOKS
        and "live" not in name
    ]
    return sorted(
        books, key=lambda line: PREFERENCE.get(_book(line["provider"]), len(PREFERENCE))
    )


def parse_lines(lines: list[dict]) -> dict:
    """CBBD's books of one game -> the home team's lines, as
    :func:`espn.parse_odds` gives them: each market from the most preferred
    book that has it; ``spread_line`` is the home team's expected margin
    (CBBD's ``spread`` is the home handicap, negative when it is favored)."""
    books = _books(lines)
    out = {}
    for line in books:
        spread = espn._number(line.get("spread"))
        if np.isfinite(spread):
            out |= {
                "spread_line": -spread if spread else 0.0,
                "spread_provider": line["provider"],
            }
            break
    for line in books:
        total = espn._number(line.get("overUnder"))
        if np.isfinite(total) and total > 0:
            out |= {"total_line": total, "total_provider": line["provider"]}
            break
    for line in books:
        home = espn._moneyline(line.get("homeMoneyline"))
        away = espn._moneyline(line.get("awayMoneyline"))
        if np.isfinite(home) and np.isfinite(away):
            out |= {
                "moneyline": home,
                "opp_moneyline": away,
                "moneyline_provider": line["provider"],
            }
            break
    return out


def season_lines(season: int, games: pd.DataFrame, key: str) -> dict[str, dict]:
    """``{game_id: lines}`` of finished ``games`` (a :func:`espn.schedule_frame`
    subset: ``game_id``, ``start``, ``home_score``, ``away_score``) from CBBD,
    keyed by ESPN's game id. Only games CBBD has lines for are in it: one it
    lists without any, does not list, or has with home and away the other
    way around (by the score) is left out, for ESPN to fetch."""
    if games.empty:
        return {}
    # only the dates of these games: a daily refresh asks for a day or two
    starts = games["start"].dropna()
    first = starts.min() - pd.Timedelta(days=1) if len(starts) else None
    last = starts.max() + pd.Timedelta(days=1) if len(starts) else None
    espn_ids = {
        row["id"]: str(row["sourceId"])
        for row in fetch_season("games", key, season, first, last)
        if row.get("sourceId")
    }
    by_game = {}
    for row in fetch_season("lines", key, season, first, last):
        game_id = espn_ids.get(row.get("gameId"))
        if game_id is not None:
            by_game[game_id] = row
    out, flipped = {}, 0
    for game in games.itertuples(index=False):
        row = by_game.get(game.game_id)
        if row is None:
            continue
        home, away = row.get("homeScore"), row.get("awayScore")
        if (
            home is not None
            and away is not None
            and home != away
            and (home, away) == (game.away_score, game.home_score)
        ):
            flipped += 1
            continue
        if lines := parse_lines(row.get("lines") or []):
            out[game.game_id] = lines
    if flipped:
        logger.warning(
            "%d: %d CBBD games have home and away swapped; their lines from ESPN",
            season,
            flipped,
        )
    return out
