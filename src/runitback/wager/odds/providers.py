"""Betting lines from several providers, normalized to one table.

Each provider returns one row per game per sportsbook, from the home team's
side: ``book``, ``home`` / ``away`` (the sport's team keys), ``commence``
(UTC kickoff), ``spread`` (the home team's expected margin: positive when
the home team is favored, the opposite sign of the handicap books quote),
``spread_home_odds`` / ``spread_away_odds``, ``ml_home`` / ``ml_away``
(American moneylines), ``total``, ``over_odds`` / ``under_odds`` and, where
the provider has them, ``spread_open`` / ``total_open``. Missing markets
are NaN.

* :class:`OddsAPI`: https://the-odds-api.com (``ODDS_API_KEY``), every book
  in a region, one request per sport and refresh.
* :class:`ESPN`: ESPN's odds API (no key), college football lines from
  every book ESPN lists, opening lines included.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import logging
import os
from collections.abc import Callable

import httpx
import pandas as pd

from ..cfb.data import download as cfb
from ..cfb.data.download import FCS

logger = logging.getLogger(__name__)

COLUMNS = [
    "book", "home", "away", "commence", "spread", "spread_home_odds",
    "spread_away_odds", "ml_home", "ml_away", "total", "over_odds",
    "under_odds", "spread_open", "total_open",
]  # fmt: skip


def empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


class Provider:
    """Fetches current lines for a sport; ``remaining`` is the provider's
    reported request quota after the last fetch (None if it reports none)."""

    name = "provider"
    remaining: int | None = None

    def fetch(self) -> pd.DataFrame:
        raise NotImplementedError


# The Odds API's sport keys.
ODDS_API_SPORTS = {
    "nfl": "americanfootball_nfl",
    "cfb": "americanfootball_ncaaf",
    "nba": "basketball_nba",
    "cbb": "basketball_ncaab",
    "wcbb": "basketball_wncaab",
    "wnba": "basketball_wnba",
    "nhl": "icehockey_nhl",
    "mlb": "baseball_mlb",
}

# Sports without a point spread: their "spreads" are the puck / run line, a
# fixed 1.5 handicap, so only moneylines and totals are requested.
NO_SPREAD = {"nhl", "mlb", "icehockey_nhl", "baseball_mlb"}


class OddsAPI(Provider):
    """Every US book's spread, moneyline and total for a sport's upcoming
    games, in one request (about 3 credits: one per market; no spreads for
    :data:`NO_SPREAD` sports).

    Args:
        sport: Sport name (``ODDS_API_SPORTS``) or an Odds API sport key.
        names: Maps the API's team names to the sport's team keys (e.g. the
            NFL's ``odds_map``); unmapped names are kept as they are.
        api_key: Default ``ODDS_API_KEY``.
        get: ``get(url, params) -> (json, headers)``; tests pass a fake.
    """

    name = "oddsapi"
    host = "https://api.the-odds-api.com/v4"

    def __init__(
        self,
        sport: str,
        names: dict | None = None,
        api_key: str | None = None,
        regions: str = "us",
        get: Callable | None = None,
    ):
        self.sport = ODDS_API_SPORTS.get(sport, sport)
        self.markets = "h2h,totals" if self.sport in NO_SPREAD else "h2h,spreads,totals"
        self.names = names or {}
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        self.regions = regions
        self._get = get or _http_get

    def fetch(self) -> pd.DataFrame:
        if not self.api_key:
            raise RuntimeError("set ODDS_API_KEY to fetch from The Odds API")
        data, headers = self._get(
            f"{self.host}/sports/{self.sport}/odds",
            {
                "apiKey": self.api_key,
                "regions": self.regions,
                "markets": self.markets,
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
        )
        remaining = headers.get("x-requests-remaining")
        self.remaining = int(float(remaining)) if remaining is not None else None
        rows = []
        for game in data:
            home, away = game["home_team"], game["away_team"]
            for book in game.get("bookmakers", []):
                row = {
                    "book": book.get("title") or book["key"],
                    "home": self.names.get(home, home),
                    "away": self.names.get(away, away),
                    "commence": game["commence_time"],
                }
                for market in book.get("markets", []):
                    prices = {o["name"]: o for o in market["outcomes"]}
                    if market["key"] == "h2h":
                        row["ml_home"] = prices.get(home, {}).get("price")
                        row["ml_away"] = prices.get(away, {}).get("price")
                    elif market["key"] == "spreads" and home in prices:
                        # books quote the home handicap: -3 is home by 3
                        row["spread"] = -prices[home]["point"]
                        row["spread_home_odds"] = prices[home]["price"]
                        row["spread_away_odds"] = prices.get(away, {}).get("price")
                    elif market["key"] == "totals" and "Over" in prices:
                        row["total"] = prices["Over"]["point"]
                        row["over_odds"] = prices["Over"]["price"]
                        row["under_odds"] = prices.get("Under", {}).get("price")
                rows.append(row)
        return _frame(rows)


class ESPN(Provider):
    """College football lines from ESPN's odds API: every book's line on the
    FBS games kicking off within ``days``, one request per game (no key, no
    quota). The games come from sportsdataverse's schedule, so team keys are
    the CFB games table's school names, and
    :data:`~runitback.wager.cfb.data.download.FCS` for an FCS side. Opening lines
    are each book's own.

    Args:
        days: Lookahead from now.
        schedule: The season's sportsdataverse schedule (default:
            downloaded); tests pass a frame.
        odds: ``odds(game_id) -> ESPN odds items`` (default: the API).
    """

    name = "espn"

    def __init__(
        self,
        days: int = 8,
        schedule: pd.DataFrame | None = None,
        odds: Callable | None = None,
        today: datetime.datetime | None = None,
    ):
        self.days = days
        self.schedule = schedule
        self._odds = odds or cfb.odds_items
        self.today = today

    def games(self) -> pd.DataFrame:
        """The FBS games kicking off within ``days`` (or started in the last
        few hours)."""
        now = pd.Timestamp(self.today or datetime.datetime.now(datetime.timezone.utc))
        schedule = self.schedule
        if schedule is None:
            season = cfb.season_of(now.date())
            url = cfb.SCHEDULE_URL.format(season=season)
            schedule = cfb.season_file(url, "", final=False, cache=cfb.Cache(None))
        games = cfb.schedule_frame(schedule)
        if games.empty:
            return games
        soon = games["start"].between(
            now - pd.Timedelta(hours=4), now + pd.Timedelta(days=self.days)
        )
        return games[cfb.fbs(games) & ~games["completed"] & soon]

    def fetch(self) -> pd.DataFrame:
        games = self.games()
        ids = list(games["id"]) if len(games) else []
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            items = dict(zip(ids, pool.map(self._odds, ids)))
        rows = []
        for game in games.itertuples():
            # FCS opponents are one pooled team in the games table
            home = game.home_team if game.home_classification == "fbs" else FCS
            away = game.away_team if game.away_classification == "fbs" else FCS
            for item in items[game.id]:
                book = (item.get("provider") or {}).get("name") or ""
                if not book or book in cfb.NOT_BOOKS or "live" in book.lower():
                    continue
                home_odds = item.get("homeTeamOdds") or {}
                away_odds = item.get("awayTeamOdds") or {}
                opening = (home_odds.get("open") or {}).get("pointSpread") or {}
                total_open = ((item.get("open") or {}).get("total") or {}).get(
                    "american"
                )
                rows.append(
                    {
                        "book": book,
                        "home": home,
                        "away": away,
                        "commence": game.start_date,
                        # ESPN's spread is the home handicap: -3 is home by 3
                        "spread": -cfb._number(item.get("spread")),
                        "spread_home_odds": home_odds.get("spreadOdds"),
                        "spread_away_odds": away_odds.get("spreadOdds"),
                        "ml_home": cfb._moneyline(home_odds.get("moneyLine")),
                        "ml_away": cfb._moneyline(away_odds.get("moneyLine")),
                        "total": item.get("overUnder") or None,
                        "over_odds": item.get("overOdds"),
                        "under_odds": item.get("underOdds"),
                        "spread_open": -cfb._number(opening.get("american")),
                        "total_open": cfb._number(total_open),
                    }
                )
        return _frame(rows)


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).reindex(columns=COLUMNS)
    numeric = [c for c in COLUMNS if c not in ("book", "home", "away", "commence")]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame["commence"] = pd.to_datetime(frame["commence"], utc=True, errors="coerce")
    return frame


def _http_get(url: str, params: dict, headers: dict | None = None):
    response = httpx.get(url, params=params, headers=headers or {}, timeout=30)
    response.raise_for_status()
    return response.json(), response.headers


def provider_for(sport: str, name: str) -> Provider:
    """A sport's provider by name (``oddsapi``, or ``espn`` for college
    football), with the sport's sportsbook team-name map where it has one."""
    if name == "espn":
        if sport != "cfb":
            raise ValueError("the espn odds provider is college football's")
        return ESPN()
    if name == "oddsapi":
        names = {}
        try:
            import importlib

            names = importlib.import_module(f"runitback.wager.{sport}.data.names").odds_map
        except (ImportError, AttributeError):
            logger.info("no odds_map for %s: keeping sportsbook team names", sport)
        return OddsAPI(sport, names=names)
    raise ValueError(f"unknown odds provider {name!r} (oddsapi, espn)")
