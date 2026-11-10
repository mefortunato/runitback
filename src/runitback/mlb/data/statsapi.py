"""MLB Stats API client: season schedules, game live feeds and pre-game probables / lineups.

- **Rate limits.** The Stats API publishes no limit, so requests are spaced at least
  ``min_interval`` seconds apart across all threads (default 0.1 s, at most 10 requests a second),
  and a 429 or 5xx response is retried with exponential backoff (honoring ``Retry-After``).
- **Cache.** A finished game's feed cannot change, so it is written once to
  ``<cache_dir>/<season>/<game_pk>.json.gz`` and never requested again. Games that are not final
  (in progress, postponed, suspended) are always fetched fresh and never cached. Before caching,
  the feed is pruned of what this package does not read (pitch tracking, hit coordinates, player
  season stats, hot/cold zones): about 110 KB gzipped per game becomes about 25 KB.
"""
from __future__ import annotations

import gzip
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator

import httpx

log = logging.getLogger(__name__)

API = "https://statsapi.mlb.com/api"

# MLB game types kept: R regular season, F wild card, D division series, L league championship
# series, W World Series (spring training, exhibitions and the All-Star game are dropped).
GAME_TYPES = ("R", "F", "D", "L", "W")

# abstractGameState of a game whose feed can no longer change.
FINAL = "Final"
# detailedState of finished listings that were not (fully) played on that date.
NOT_PLAYED = {"Postponed", "Cancelled", "Suspended"}

SCHEDULE_FIELDS = ("dates,date,games,gamePk,gameType,season,gameDate,officialDate,status,"
                   "abstractGameState,detailedState,teams,away,home,team,id,doubleHeader,gameNumber")


class Client:
    """A throttled, retrying Stats API client, safe to share between threads."""

    def __init__(self, min_interval: float = 0.1, retries: int = 5, timeout: float = 60.0):
        self.min_interval = min_interval
        self.retries = retries
        self._lock = threading.Lock()
        self._next = 0.0
        self._http = httpx.Client(timeout=timeout, follow_redirects=True,
                                  limits=httpx.Limits(max_connections=8, max_keepalive_connections=8))
        self.requests = 0

    def _wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.min_interval
        if delay > 0:
            time.sleep(delay)

    def get(self, path: str, **params) -> dict:
        """GET ``API + path`` as JSON. Retries 429 / 5xx / transport errors; raises on the last."""
        url = f"{API}{path}"
        for attempt in range(self.retries + 1):
            self._wait()
            try:
                response = self._http.get(url, params=params or None)
                self.requests += 1
            except httpx.TransportError:
                if attempt == self.retries:
                    raise
                time.sleep(2 ** attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == self.retries:
                    response.raise_for_status()
                retry_after = response.headers.get("retry-after")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
                log.warning("%s: HTTP %d, retrying in %.0fs", url, response.status_code, wait)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError("unreachable")


_default: Client | None = None


def default_client() -> Client:
    global _default
    if _default is None:
        _default = Client()
    return _default


# ----------------------------------------------------------------- schedule
def schedule(season: int, client: Client | None = None) -> list[dict]:
    """A season's games (regular season and postseason), one dict per listing: ``game_pk``,
    ``season``, ``game_type``, ``date``, ``final`` and ``state``. A game listed twice (a suspended
    game resumed on a later date keeps its id) keeps its last listing; postponed and cancelled
    listings are dropped."""
    client = client or default_client()
    data = client.get("/v1/schedule", sportId=1, season=season, gameType=",".join(GAME_TYPES),
                      fields=SCHEDULE_FIELDS)
    games: dict[int, dict] = {}
    for date in data.get("dates") or []:
        for game in date.get("games") or []:
            status = game.get("status") or {}
            if game.get("gameType") not in GAME_TYPES or status.get("detailedState") in NOT_PLAYED:
                continue
            games[int(game["gamePk"])] = {
                "game_pk": int(game["gamePk"]), "season": int(game["season"]),
                "game_type": game["gameType"], "date": game.get("officialDate") or date.get("date"),
                "final": status.get("abstractGameState") == FINAL, "state": status.get("detailedState"),
            }
    return sorted(games.values(), key=lambda g: (g["date"], g["game_pk"]))


# --------------------------------------------------------------------- feeds
# Keys dropped (at any depth) from a feed before it is cached.
PRUNE = {"pitchData", "hitData", "seasonStats", "batterHotColdZones", "pitcherHotColdZones",
         "batterHotColdZoneStats", "pitcherHotColdZoneStats", "link", "nameSlug", "playId",
         "ballColor", "trailColor", "leaders", "scoringPlays", "playsByInning", "currentPlay",
         "highlights", "weather", "alerts", "review", "officialScorer", "primaryDatacaster",
         "flags", "moundVisits", "gameInfo", "info", "note", "teamStats", "splits"}
# Per-player box score stats are kept only for these groups' plate appearances / batters faced.
KEEP_STATS = {"batting": ("plateAppearances",), "pitching": ("battersFaced",)}


def prune(node):
    """A copy of a feed without the :data:`PRUNE` keys and with player box stats cut down to
    :data:`KEEP_STATS` (enough to check the PA table against the box score)."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key in PRUNE:
                continue
            if key == "stats" and isinstance(value, dict) and ("batting" in value or "pitching" in value):
                out[key] = {g: {k: (value.get(g) or {}).get(k) for k in keys if k in (value.get(g) or {})}
                            for g, keys in KEEP_STATS.items()}
                continue
            out[key] = prune(value)
        return out
    if isinstance(node, list):
        return [prune(v) for v in node]
    return node


def is_final(feed: dict) -> bool:
    status = (feed.get("gameData") or {}).get("status") or {}
    return status.get("abstractGameState") == FINAL and status.get("detailedState") not in NOT_PLAYED


def cache_path(cache_dir: Path, season: int, game_pk: int) -> Path:
    return Path(cache_dir) / str(season) / f"{game_pk}.json.gz"


def read_cached(path: Path) -> dict | None:
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_cached(path: Path, feed: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(feed, f, separators=(",", ":"))
    tmp.replace(path)  # atomic: an interrupted download never leaves a truncated cache file


def feed(game_pk: int, season: int, cache_dir: Path | None = None, client: Client | None = None,
         force: bool = False) -> dict:
    """One game's pruned live feed (``/v1.1/game/{pk}/feed/live``); finished games are cached."""
    path = cache_path(cache_dir, season, game_pk) if cache_dir is not None else None
    if path is not None and not force:
        cached = read_cached(path)
        if cached is not None:
            return cached
    data = prune((client or default_client()).get(f"/v1.1/game/{game_pk}/feed/live"))
    if path is not None and is_final(data):
        write_cached(path, data)
    return data


def iter_feeds(games: Iterable[dict], cache_dir: Path | None, client: Client | None = None,
               workers: int = 4, force: bool = False, chunk: int = 64) -> Iterator[dict]:
    """Feeds of ``games`` (``schedule`` rows) in order, cached ones read from disk and the rest
    downloaded by ``workers`` threads under the client's shared throttle. Works through ``chunk``
    games at a time so a season's feeds are never all in memory."""
    client = client or default_client()
    games = list(games)

    def one(g: dict) -> dict:
        return feed(g["game_pk"], g["season"], cache_dir, client, force)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for start in range(0, len(games), chunk):
            yield from pool.map(one, games[start:start + chunk])


def feeds(games: Iterable[dict], cache_dir: Path | None, client: Client | None = None,
          workers: int = 4, force: bool = False) -> list[dict]:
    """:func:`iter_feeds` as a list."""
    return list(iter_feeds(games, cache_dir, client, workers, force))


# ----------------------------------------------------------- pre-game state
def probables(date: str, client: Client | None = None) -> list[dict]:
    """Announced probable pitchers and posted batting orders for the games on ``date``
    (``YYYY-MM-DD``), as listed right now: one dict per team-game with ``game_pk``, ``game_type``,
    ``state``, ``side``, ``team_id``, ``opp_id``, ``probable_sp_id`` (None until announced) and
    ``lineup`` (batting order ids 1-9, empty until posted).

    This is a live view. For a past date the Stats API returns the pitchers who started and the
    lineups that were used, not what was announced beforehand, so it is not a pre-game history.
    """
    client = client or default_client()
    data = client.get("/v1/schedule", sportId=1, date=date, gameType=",".join(GAME_TYPES),
                      hydrate="probablePitcher,lineups")
    rows = []
    for day in data.get("dates") or []:
        for game in day.get("games") or []:
            teams = game.get("teams") or {}
            lineups = game.get("lineups") or {}
            for side, other in (("home", "away"), ("away", "home")):
                team = teams.get(side) or {}
                rows.append({
                    "game_pk": int(game["gamePk"]), "game_type": game.get("gameType"),
                    "date": game.get("officialDate") or day.get("date"),
                    "state": (game.get("status") or {}).get("detailedState"), "side": side,
                    "team_id": (team.get("team") or {}).get("id"),
                    "opp_id": ((teams.get(other) or {}).get("team") or {}).get("id"),
                    "probable_sp_id": (team.get("probablePitcher") or {}).get("id"),
                    "lineup": [p["id"] for p in lineups.get(f"{side}Players") or []],
                })
    return rows
