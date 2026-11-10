"""College football games, box scores and betting lines from sportsdataverse
and ESPN.

The schedule comes from sportsdataverse's ``cfb_schedules`` release files
(https://github.com/sportsdataverse/sportsdataverse-data, one parquet file per
season, refreshed daily). It mirrors CollegeFootballData's schedule, so team
keys are CFBD's school names, and is the source of truth for which games
exist: upcoming games are kept with a missing result. Team box scores come
from the ``espn_cfb_team_box`` release (ESPN's box scores, joined by ESPN's
team id, which CFBD shares).

Betting lines come from ESPN's odds API, one request per game: every book
ESPN lists, from 2013 on. Lines of finished games never change, so they are
cached per season on disk (``cache_dir``) with the finished seasons' parquet
files; a daily refresh only requests the newest games and the next week's
upcoming ones. Nothing needs an API key or has a request quota.

Only games with at least one FBS team are kept. FCS (and lower-division)
opponents are pooled into one team, :data:`FCS`: an FCS school plays one or
two FBS games a year, too few to give it a team state of its own, while the
pooled team's last games are a fair picture of a typical FCS opponent. The
school's own name stays in ``team_name`` / ``opp_name``.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import io
import json
import logging
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RELEASES = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
SCHEDULE_URL = RELEASES + "/cfb_schedules/cfb_schedules_{season}.parquet"
TEAM_BOX_URL = RELEASES + "/espn_cfb_team_box/team_box_{season}.parquet"
TEAM_INFO_URL = RELEASES + "/cfb_team_info/cfb_team_info_{season}.parquet"
ODDS_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
    "/events/{0}/competitions/{0}/odds"
)

# Team key of every FCS / lower-division opponent.
FCS = "FCS"

# Sportsbooks in order of preference for prices and moneylines; any other
# book comes after them. Prediction sites (numberfire, teamrankings,
# accuscore), the "Opening" line and in-game ("Live Odds") lines are never
# used. ESPN's "consensus" is used only when no book has a line (2013): in
# 2014-2016 it is the opening line.
PROVIDERS = (
    "DraftKings",
    "ESPN BET",
    "Caesars Sportsbook (Colorado)",
    "Caesars Sportsbook (New Jersey)",
    "Caesars Sportsbook",
    "Caesars",
    "Westgate",
    "BOVADA.lv",
    "BETONLINE.ag",
    "5Dimes.eu",
)
NOT_BOOKS = {"numberfire", "teamrankings", "accuscore", "Opening", "consensus"}

# Box-score categories that come as "made-attempted" (or "made/attempted")
# strings.
SPLIT_STATS = {
    "totalPenaltiesYards": ("totalPenalties", "totalPenaltiesYards"),
    "completionAttempts": ("passingCompletions", "passingAttempts"),
    "fourthDownEff": ("fourthDownConv", "fourthDownAtt"),
    "thirdDownEff": ("thirdDownConv", "thirdDownAtt"),
}

# Longest rest counted, in days (season openers get it too).
MAX_REST = 21

# Odds lookahead: upcoming games starting within this many days get lines.
LINES_DAYS = 8

# The first season with betting lines.
FIRST_LINES_SEASON = 2013


def season_of(day: datetime.date) -> int:
    """The season a date belongs to (bowls in January count as last year's)."""
    return day.year if day.month >= 3 else day.year - 1


def season_complete(season: int, today: datetime.datetime) -> bool:
    """Whether nothing in ``season`` can change (bowls end in mid-January)."""
    return today.date() > datetime.date(season + 1, 2, 15)


# -------------------------------------------------------------------- cache
class Cache:
    """Files that cannot change any more, under ``directory`` (``None``:
    nothing is cached)."""

    def __init__(self, directory: str | None):
        self.directory = Path(directory) if directory else None

    def path(self, name: str) -> Path | None:
        return self.directory / name if self.directory else None

    def read_bytes(self, name: str) -> bytes | None:
        path = self.path(name)
        return path.read_bytes() if path is not None and path.exists() else None

    def write_bytes(self, name: str, data: bytes) -> None:
        path = self.path(name)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def read_json(self, name: str) -> dict:
        data = self.read_bytes(name)
        return json.loads(data) if data else {}

    def write_json(self, name: str, data: dict) -> None:
        self.write_bytes(name, json.dumps(data, sort_keys=True).encode())


def _get(url: str, retries: int = 4) -> httpx.Response:
    for attempt in range(retries + 1):
        try:
            response = httpx.get(url, timeout=60, follow_redirects=True)
            if response.status_code != 429 and response.status_code < 500:
                return response
        except httpx.TransportError:
            if attempt == retries:
                raise
        if attempt < retries:
            time.sleep(2 ** (attempt + 1))
    return response


def season_file(url: str, name: str, final: bool, cache: Cache) -> pd.DataFrame:
    """A season's sportsdataverse parquet file (empty if it has none yet)."""
    data = cache.read_bytes(name) if final else None
    if data is None:
        response = _get(url)
        if response.status_code == 404:
            return pd.DataFrame()
        response.raise_for_status()
        data = response.content
        if final:
            cache.write_bytes(name, data)
    return pd.read_parquet(io.BytesIO(data))


# ----------------------------------------------------------------- schedule
def schedule_frame(schedule: pd.DataFrame) -> pd.DataFrame:
    """sportsdataverse schedule -> one row per game, typed, with ``id`` (the
    game id), ``home_team_id`` / ``away_team_id`` and ``*_classification``
    (``fbs``, ``fcs``, ``ii``, ``iii``)."""
    if schedule.empty:
        return pd.DataFrame()
    games = schedule.rename(
        columns={
            "game_id": "id",
            "home_id": "home_team_id",
            "away_id": "away_team_id",
            "home_division": "home_classification",
            "away_division": "away_classification",
        }
    )
    games["start"] = pd.to_datetime(games["start_date"], utc=True, errors="coerce")
    for col in ("neutral_site", "conference_game", "completed", "start_time_tbd"):
        if col not in games:
            games[col] = False
        games[col] = games[col].astype("boolean").fillna(False).astype(bool)
    for col in ("home_points", "away_points", "home_team_id", "away_team_id"):
        games[col] = pd.to_numeric(games[col], errors="coerce")
    games["id"] = games["id"].astype(str)
    return games.drop_duplicates("id").reset_index(drop=True)


def fbs(games: pd.DataFrame) -> pd.Series:
    """Whether each game has an FBS team."""
    home, away = games["home_classification"], games["away_classification"]
    return home.eq("fbs") | away.eq("fbs")


def chronological_weeks(games: pd.DataFrame) -> pd.Series:
    """Week of the season that orders every game, bowls included.

    Regular-season games keep the schedule's week. CFBD files the whole bowl
    season under postseason week 1, which would predict every playoff round
    from the same state; postseason games are numbered instead by the
    7-day block they fall in after the first bowl, following the last
    regular week.
    """
    week = pd.to_numeric(games["week"], errors="coerce")
    post = games["season_type"].eq("postseason")
    out = week.copy()
    for season, rows in games.groupby("season"):
        regular = rows.index[~post.loc[rows.index]]
        bowls = rows.index[post.loc[rows.index]]
        if not len(bowls):
            continue
        last = week.loc[regular].max() if len(regular) else 0
        # US time: kickoffs late on a Saturday are already Sunday in UTC
        day = (games.loc[bowls, "start"] - pd.Timedelta(hours=6)).dt.normalize()
        out.loc[bowls] = last + 1 + ((day - day.min()).dt.days // 7)
    return out.astype(int)


def game_types(games: pd.DataFrame) -> pd.Series:
    """``regular``, ``bowl``, ``playoff`` or ``final`` (national championship)."""
    notes = games.get("notes", pd.Series("", index=games.index)).fillna("")
    post = games["season_type"].eq("postseason")
    final = post & notes.str.contains("national championship", case=False)
    playoff = post & notes.str.contains(r"playoff|\bCFP\b", case=False)
    out = pd.Series("regular", index=games.index)
    out[post] = "bowl"
    out[playoff] = "playoff"
    out[final] = "final"
    return out


# ---------------------------------------------------------------- box score
def possession_time(time: str) -> float:
    minutes, seconds = str(time).split(":")
    return int(minutes) + int(seconds) / 60


def box_scores(team_box: pd.DataFrame) -> pd.DataFrame:
    """sportsdataverse team box scores -> one row per team per game:
    ``game_id``, ``team_id`` and its stats as numbers."""
    if team_box.empty:
        return pd.DataFrame(columns=["game_id", "team_id"])
    box = team_box.copy()
    for category, (made, att) in SPLIT_STATS.items():
        if category not in box:
            continue
        parts = box[category].astype("string").str.extract(r"^(\d+)\s*[-/]+\s*(\d+)$")
        box[made], box[att] = parts[0], parts[1]
    if "possessionTime" in box:
        box["possessionTime"] = [
            possession_time(t) if isinstance(t, str) and ":" in t else np.nan
            for t in box["possessionTime"]
        ]
    box = box.drop(columns=["team_abbreviation", "team_name", "home_away", "season"],
                   errors="ignore")  # fmt: skip
    box["game_id"] = box["game_id"].astype(str)
    stats = [c for c in box if c != "game_id"]
    box[stats] = box[stats].apply(pd.to_numeric, errors="coerce")
    return box


# -------------------------------------------------------------------- lines
def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _moneyline(value) -> float:
    """American odds, or NaN for the placeholders some books report."""
    value = _number(value)
    return value if abs(value) >= 100 else np.nan


def _name(item: dict) -> str:
    return (item.get("provider") or {}).get("name") or ""


def _books(items: list[dict]) -> list[dict]:
    rank = {p: i for i, p in enumerate(PROVIDERS)}
    books = [
        item
        for item in items
        if (name := _name(item))
        and name not in NOT_BOOKS
        and "live" not in name.lower()
    ]
    if not books:
        books = [item for item in items if _name(item) == "consensus"]
    return sorted(books, key=lambda i: rank.get(_name(i), len(rank)))


def _median(books: list[dict], line) -> tuple[float, dict | None]:
    """The median of the books' ``line(book)`` to the half point, and the
    most preferred book quoting exactly it (None if none does)."""
    lines = [(book, line(book)) for book in books]
    lines = [(book, value) for book, value in lines if np.isfinite(value)]
    if not lines:
        return np.nan, None
    median = float(np.floor(np.median([value for _, value in lines]) * 2 + 0.5) / 2)
    quoting = [book for book, value in lines if value == median]
    return median, quoting[0] if quoting else None


def parse_odds(data: dict) -> dict:
    """ESPN's odds for one game -> the home team's lines.

    ``spread_line`` (the home team's expected margin, positive = home
    favored; ESPN's ``spread`` is the home handicap) and ``total_line`` are
    the median over the books, so one book's stale or broken number cannot
    pick the line; their prices come from the most preferred book quoting
    exactly that line. Both moneylines come from the most preferred book
    that has them.
    """
    books = _books(data.get("items") or [])
    out = {}
    spread, book = _median(books, lambda b: -_number(b.get("spread")))
    if np.isfinite(spread):
        home = (book or {}).get("homeTeamOdds") or {}
        away = (book or {}).get("awayTeamOdds") or {}
        out |= {
            "spread_line": spread + 0.0,  # no -0.0
            "spread_odds": _moneyline(home.get("spreadOdds")),
            "opp_spread_odds": _moneyline(away.get("spreadOdds")),
        }
    total, book = _median(
        books, lambda b: _number(b.get("overUnder")) if b.get("overUnder") else np.nan
    )
    if np.isfinite(total):
        out |= {
            "total_line": total,
            "over_odds": _moneyline((book or {}).get("overOdds")),
            "under_odds": _moneyline((book or {}).get("underOdds")),
        }
    for item in books:
        home = _moneyline((item.get("homeTeamOdds") or {}).get("moneyLine"))
        away = _moneyline((item.get("awayTeamOdds") or {}).get("moneyLine"))
        if np.isfinite(home) and np.isfinite(away):
            out |= {
                "moneyline": home,
                "opp_moneyline": away,
                "moneyline_provider": _name(item),
            }
            break
    return {k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in out.items()}  # fmt: skip


def odds_items(game_id: str) -> list[dict]:
    """Every book's line on one game from ESPN's odds API."""
    response = _get(ODDS_URL.format(game_id))
    if response.status_code == 404:
        return []
    response.raise_for_status()
    return response.json().get("items") or []


def fetch_odds(game_id: str) -> dict:
    return parse_odds({"items": odds_items(game_id)})


def season_lines(
    season: int,
    final_ids: list[str],
    upcoming_ids: list[str],
    cache: Cache,
    workers: int = 8,
) -> dict[str, dict]:
    """``{game_id: lines}`` for a season's finished games (cached) and the
    given upcoming ones (fetched every time)."""
    name = f"odds_{season}.json"
    cached = cache.read_json(name)
    missing = [g for g in final_ids if g not in cached]
    wanted = missing + [g for g in upcoming_ids if g not in cached]
    if wanted:
        logger.info("%d: fetching lines of %d games", season, len(wanted))
        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            fetched = dict(zip(wanted, pool.map(fetch_odds, wanted)))
        if missing:
            cache.write_json(name, cached | {g: fetched[g] for g in missing})
        cached = cached | fetched
    return {g: cached[g] for g in final_ids + upcoming_ids if g in cached}


# -------------------------------------------------------------------- games
def invert(home: pd.DataFrame) -> pd.DataFrame:
    """Both perspectives of one-row-per-game ``home_*`` / ``away_*`` columns."""

    def side(team: str, opp: str, location) -> pd.DataFrame:
        renames = {}
        for col in home:
            if col.startswith(f"{team}_"):
                renames[col] = col.removeprefix(f"{team}_")
            elif col.startswith(f"{opp}_"):
                renames[col] = "opp_" + col.removeprefix(f"{opp}_")
        out = home.rename(columns=renames)
        out["location"] = location
        return out

    neutral = home["neutral_site"].to_numpy()
    away = side("away", "home", np.where(neutral, 0, -1))
    away["spread_line"] = -away["spread_line"]
    home = side("home", "away", np.where(neutral, 0, 1))
    return pd.concat([home, away], ignore_index=True)


def rest_days(rows: pd.DataFrame) -> pd.Series:
    """Days since each school's previous game (capped; openers get the cap)."""
    order = rows.sort_values("start")
    previous = order.groupby(["season", "team_name"])["start"].shift()
    days = (order["start"] - previous).dt.days.clip(upper=MAX_REST)
    return days.fillna(MAX_REST).reindex(rows.index)


LINE_COLUMNS = ["spread_line", "spread_odds", "opp_spread_odds", "total_line",
                "over_odds", "under_odds", "moneyline", "opp_moneyline"]  # fmt: skip


def build_games(
    schedule: pd.DataFrame,
    team_box: pd.DataFrame,
    lines: dict[str, dict] | None = None,
    today: datetime.datetime | None = None,
) -> pd.DataFrame:
    """One row per team per game from the schedule, box scores and lines.

    Games without an FBS team are dropped, FCS sides are pooled into
    :data:`FCS`, and past games that were never completed (cancelled or
    postponed) are dropped; future ones are kept as upcoming games.
    """
    from .features import STATS_COLUMNS

    today = today or datetime.datetime.now(datetime.timezone.utc)
    games = schedule_frame(schedule)
    if games.empty:
        return games
    games = games[fbs(games)].copy()
    upcoming = ~games["completed"] | games["home_points"].isna()
    cancelled = upcoming & (games["start"] < pd.Timestamp(today) - pd.Timedelta(days=1))
    games = games[~cancelled].copy()
    games.loc[upcoming, ["home_points", "away_points"]] = np.nan

    games["api_week"] = games["week"]
    games["week"] = chronological_weeks(games)
    games["game_type"] = game_types(games)
    games["postseason"] = games["season_type"].eq("postseason").astype(int)
    games["conference_game"] = games["conference_game"].astype(int)

    market = pd.DataFrame.from_dict(lines or {}, orient="index")
    market = market.reindex(columns=LINE_COLUMNS).rename(
        columns={
            "moneyline": "home_moneyline",
            "opp_moneyline": "away_moneyline",
            "spread_odds": "home_spread_odds",
            "opp_spread_odds": "away_spread_odds",
        }
    )
    games = games.merge(market, left_on="id", right_index=True, how="left")
    # over / under prices belong to the game, not a side
    games = games.rename(columns={"over_odds": "ou_over", "under_odds": "ou_under"})

    keep = ["id", "season", "week", "api_week", "season_type", "game_type",
            "start_date", "start_time_tbd", "neutral_site", "conference_game",
            "postseason", "venue", "notes", "spread_line", "total_line",
            "ou_over", "ou_under", "start"]  # fmt: skip
    sides = ["team", "team_id", "conference", "classification", "points",
             "moneyline", "spread_odds"]  # fmt: skip
    cols = keep + [f"{s}_{c}" for s in ("home", "away") for c in sides]
    rows = invert(games.reindex(columns=cols))
    rows = rows.rename(
        columns={
            "id": "game_id",
            "team": "team_name",
            "opp_team": "opp_name",
            "ou_over": "over_odds",
            "ou_under": "under_odds",
        }
    )

    rows["team"] = rows["team_name"].where(rows["classification"].eq("fbs"), FCS)
    rows["opp_team"] = rows["opp_name"].where(rows["opp_classification"].eq("fbs"), FCS)
    for prefix in ("", "opp_"):
        fcs = rows[f"{prefix}team"].eq(FCS)
        rows.loc[fcs, f"{prefix}conference"] = FCS
    rows["result"] = rows["points"] - rows["opp_points"]
    rows["total"] = rows["points"] + rows["opp_points"]
    rows["rest"] = rest_days(rows)
    rows.loc[rows["team"].eq(FCS), "rest"] = np.nan
    opp_rest = rows[["game_id", "team_name", "rest"]].rename(
        columns={"team_name": "opp_name", "rest": "opp_rest"}
    )
    rows = rows.merge(opp_rest, on=["game_id", "opp_name"], how="left")

    box = box_scores(team_box).drop_duplicates(["game_id", "team_id"])
    stats = [c for c in STATS_COLUMNS if c != "points"]
    box = box.reindex(columns=["game_id", "team_id", *stats])
    rows = rows.merge(box, on=["game_id", "team_id"], how="left")
    opp = {"team_id": "opp_team_id", **{c: f"opp_{c}" for c in stats}}
    rows = rows.merge(
        box.rename(columns=opp),
        on=["game_id", "opp_team_id"],
        how="left",
    )
    # finished games' stats only: a stray box score cannot leak into a preview
    rows.loc[rows["result"].isna(), stats + [f"opp_{c}" for c in stats]] = np.nan
    rows = rows.drop(columns=["start", "classification", "opp_classification",
                              "team_id", "opp_team_id"])  # fmt: skip
    return rows.sort_values(["season", "week", "game_id", "location"]).reset_index(
        drop=True
    )


# ----------------------------------------------------------------- download
def fetch_season(
    season: int,
    today: datetime.datetime,
    cache: Cache,
    lines_days: int = LINES_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """``(schedule, team box scores, lines)`` of one season."""
    final = season_complete(season, today)
    schedule = season_file(
        SCHEDULE_URL.format(season=season), f"schedule_{season}.parquet", final, cache
    )
    box = season_file(
        TEAM_BOX_URL.format(season=season), f"team_box_{season}.parquet", final, cache
    )
    lines = {}
    games = schedule_frame(schedule)
    if season >= FIRST_LINES_SEASON and len(games):
        games = games[fbs(games)]
        now = pd.Timestamp(today)
        done = games["completed"]
        soon = ~done & games["start"].between(
            now - pd.Timedelta(hours=12), now + pd.Timedelta(days=lines_days)
        )
        lines = season_lines(
            season,
            list(games.loc[done, "id"]),
            list(games.loc[soon, "id"]),
            cache,
        )
    return schedule, box, lines


def download(
    output_file: str | None = None,
    first_season: int = 2004,
    last_season: int | None = None,
    cache_dir: str | None = "data/cfb/espn_cache",
) -> pd.DataFrame:
    """Download every season's games, box scores and lines.

    Args:
        output_file: Also write the games table here.
        first_season / last_season: Seasons to download (default: through
            the current season). Box scores start in 2004, betting lines in
            2013.
        cache_dir: Reuse finished seasons' files and finished games' lines
            from here (``None``: always download). A full download makes
            about 12,000 odds requests; a daily refresh about a hundred.
    """
    today = datetime.datetime.now(datetime.timezone.utc)
    last_season = last_season or season_of(today.date())
    cache = Cache(cache_dir)
    schedules, boxes, lines = [], [], {}
    for season in range(first_season, last_season + 1):
        schedule, box, season_lines_ = fetch_season(season, today, cache)
        schedules.append(schedule)
        boxes.append(box)
        lines |= season_lines_
        logger.info("%d: %d games, %d with lines", season, len(schedule),
                    len(season_lines_))  # fmt: skip
    games = build_games(
        pd.concat(schedules, ignore_index=True),
        pd.concat(boxes, ignore_index=True),
        lines,
        today=today,
    )
    if output_file is not None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        games.to_csv(output_file, index=False)
    return games
