"""ESPN and sportsdataverse data shared by every sport that uses them.

- Betting lines from ESPN's odds API (``sports.core.api.espn.com``), one
  request per game: every book ESPN lists. Each market comes from the most
  preferred book that has it (:data:`PROVIDERS`).
- ESPN's scoreboard by month (``site.api.espn.com``), for the ESPN ids of
  sports whose games come from the league's own API (NHL, MLB), so their
  lines can be fetched.
- Team metadata (names, colors, logos, conferences) from ESPN's ``/teams``
  and ``/groups``.
- sportsdataverse's ESPN basketball release files
  (https://github.com/sportsdataverse/sportsdataverse-data, one parquet file
  per season, refreshed daily with the previous day's games) and the games
  table built from them: :func:`build_basketball_games`, used by the NBA,
  WNBA and men's and women's college basketball.

Files and lines that cannot change any more (finished seasons, finished
games) are cached on disk (:class:`Cache`), so a daily refresh only fetches
the current season.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RELEASES = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
SITE = "https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}"
ODDS_URL = (
    "https://sports.core.api.espn.com/v2/sports/{sport}/leagues/{league}"
    "/events/{game}/competitions/{game}/odds"
)

# Sportsbooks in order of preference; any other book comes after them.
# Prediction sites (numberfire, teamrankings, accuscore), the "Opening" line
# and in-game ("Live Odds") lines are never used.
PROVIDERS = (
    "consensus",
    "DraftKings",
    "ESPN BET",
    "Caesars Sportsbook",
    "Caesars Sportsbook (New Jersey)",
    "Westgate",
    "BOVADA.lv",
    "BETONLINE.ag",
    "5Dimes.eu",
)
NOT_BOOKS = {"numberfire", "teamrankings", "accuscore", "Opening"}

# Lines of one game, from the home team's perspective (see parse_odds).
LINE_COLUMNS = ["spread_line", "spread_odds", "opp_spread_odds", "total_line",
                "over_odds", "under_odds", "moneyline", "opp_moneyline"]  # fmt: skip


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

    def write_json(self, name: str, data) -> None:
        self.write_bytes(name, json.dumps(data, sort_keys=True).encode())


_client: httpx.Client | None = None
_ODDS_FORBIDDEN: set[tuple[str, str]] = set()


def client() -> httpx.Client:
    """One client for every request, shared by threads: reusing its
    connections makes a season of odds requests ~10x faster than a new TLS
    connection per request."""
    global _client
    if _client is None:
        limits = httpx.Limits(max_connections=16, max_keepalive_connections=16)
        _client = httpx.Client(timeout=60, follow_redirects=True, limits=limits)
    return _client


def _get(
    url: str, retries: int = 4, headers: dict | None = None, **params
) -> httpx.Response:
    for attempt in range(retries + 1):
        try:
            response = client().get(url, params=params or None, headers=headers)
            if response.status_code != 429 and response.status_code < 500:
                return response
        except httpx.TransportError:
            if attempt == retries:
                raise
        if attempt < retries:
            time.sleep(2 ** (attempt + 1))
    return response


def get_json(url: str, name: str | None = None, cache: Cache | None = None, **params):
    """A JSON response; with ``name``, cached in ``cache`` and reused."""
    if name is not None and cache is not None:
        data = cache.read_bytes(name)
        if data is not None:
            return json.loads(data)
    response = _get(url, **params)
    response.raise_for_status()
    if name is not None and cache is not None:
        cache.write_bytes(name, response.content)
    return response.json()


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


def _books(items: list[dict]) -> list[dict]:
    rank = {p: i for i, p in enumerate(PROVIDERS)}
    books = [
        item
        for item in items
        if (name := (item.get("provider") or {}).get("name") or "")
        and name not in NOT_BOOKS
        and "live" not in name.lower()
    ]
    return sorted(books, key=lambda i: rank.get(i["provider"]["name"], len(rank)))


def parse_odds(data: dict) -> dict:
    """ESPN's odds for one game -> the home team's lines.

    ``spread_line`` is the home team's expected margin (positive = home
    favored; ESPN's ``spread`` is the home handicap), with both sides' spread
    prices; ``total_line`` with the over / under prices; both moneylines.
    Each market comes from the most preferred book that has it, its prices
    from the same book. (In hockey and baseball the spread is the puck or
    run line, a handicap of 1.5 goals or runs priced like a moneyline.)
    """
    books = _books(data.get("items") or [])
    out = {}
    for item in books:
        spread = _number(item.get("spread"))
        if np.isfinite(spread):
            home, away = item.get("homeTeamOdds") or {}, item.get("awayTeamOdds") or {}
            out |= {
                "spread_line": -spread if spread else 0.0,
                "spread_odds": _moneyline(home.get("spreadOdds")),
                "opp_spread_odds": _moneyline(away.get("spreadOdds")),
                "spread_provider": item["provider"]["name"],
            }
            break
    for item in books:
        total = _number(item.get("overUnder"))
        if np.isfinite(total) and total > 0:
            out |= {
                "total_line": total,
                "over_odds": _moneyline(item.get("overOdds")),
                "under_odds": _moneyline(item.get("underOdds")),
                "total_provider": item["provider"]["name"],
            }
            break
    for item in books:
        home = _moneyline((item.get("homeTeamOdds") or {}).get("moneyLine"))
        away = _moneyline((item.get("awayTeamOdds") or {}).get("moneyLine"))
        if np.isfinite(home) and np.isfinite(away):
            out |= {
                "moneyline": home,
                "opp_moneyline": away,
                "moneyline_provider": item["provider"]["name"],
            }
            break
    return {k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in out.items()}  # fmt: skip


def odds_refused() -> list[str]:
    """The ``sport/league``s whose odds ESPN refused (403) in this process."""
    return sorted(f"{sport}/{league}" for sport, league in _ODDS_FORBIDDEN)


def fetch_odds(sport: str, league: str, game_id: str) -> dict | None:
    """One game's :func:`parse_odds` (empty if ESPN has none; ``None`` if denied).

    A 403 disables further odds requests for this league in the current
    process; schedule and box-score downloads remain unaffected.
    """
    key = (sport, league)
    if key in _ODDS_FORBIDDEN:
        return None
    response = _get(ODDS_URL.format(sport=sport, league=league, game=game_id))
    if response.status_code == 404:
        return {}
    if response.status_code == 403:
        _ODDS_FORBIDDEN.add(key)
        logger.warning(
            "ESPN odds API returned 403 for %s/%s; skipping further odds requests",
            sport,
            league,
        )
        return None
    response.raise_for_status()
    return parse_odds(response.json())


def season_lines(
    season: int,
    final_ids: list[str],
    upcoming_ids: list[str],
    cache: Cache,
    fetch: Callable[[str], dict],
    workers: int = 8,
    bulk: Callable[[list[str]], dict[str, dict]] | None = None,
) -> dict[str, dict]:
    """``{game_id: lines}`` for a season's finished games (cached) and the
    given upcoming ones (fetched every time), ``fetch``-ed by game id.

    With ``bulk``, finished games come from ``bulk(game_ids)`` first
    (``{game_id: lines}``); the ones it leaves out are ``fetch``-ed, and
    cached, like any other.
    """
    name = f"odds_{season}.json"
    cached = cache.read_json(name)
    missing = [g for g in final_ids if g not in cached]
    if bulk is not None:
        if missing:
            found = bulk(missing)
            final = {g: found[g] for g in missing if g in found}
            logger.info("%d: %d of %d finished games' lines in bulk",
                        season, len(final), len(missing))  # fmt: skip
            if final:
                cache.write_json(name, cached | final)
                cached = cached | final
            missing = [g for g in missing if g not in final]
    wanted = missing + [g for g in upcoming_ids if g not in cached]
    if wanted:
        logger.info("%d: fetching lines of %d games", season, len(wanted))
        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            fetched = dict(zip(wanted, pool.map(fetch, wanted)))
        available = {g: lines for g, lines in fetched.items() if lines is not None}
        final = {g: lines for g, lines in available.items() if g in missing}
        if final:
            cache.write_json(name, cached | final)
        cached = cached | available
    return {g: cached[g] for g in final_ids + upcoming_ids if g in cached}


def lines_to_fetch(
    games: pd.DataFrame, today, days: int
) -> tuple[list[str], list[str]]:
    """``(finished, upcoming)`` game ids of a :func:`schedule_frame`-like table
    (``game_id``, ``completed``, ``start``) whose lines are wanted: every
    finished game and the upcoming ones starting within ``days`` (books post
    lines about a day ahead)."""
    now = pd.Timestamp(today)
    done = games["completed"]
    soon = ~done & games["start"].between(
        now - pd.Timedelta(hours=12), now + pd.Timedelta(days=days)
    )
    return list(games.loc[done, "game_id"]), list(games.loc[soon, "game_id"])


def join_lines(games: pd.DataFrame, lines: dict[str, dict] | None) -> pd.DataFrame:
    """One-row-per-game ``games`` (``game_id``) with each game's lines as
    ``spread_line``, ``total_line``, ``ou_over`` / ``ou_under`` and
    ``home_*`` / ``away_*`` moneylines and spread prices, ready for
    :func:`invert`."""
    market = pd.DataFrame.from_dict(lines or {}, orient="index")
    market = market.reindex(columns=LINE_COLUMNS).rename(
        columns={
            "moneyline": "home_moneyline",
            "opp_moneyline": "away_moneyline",
            "spread_odds": "home_spread_odds",
            "opp_spread_odds": "away_spread_odds",
        }
    )
    games = games.merge(market, left_on="game_id", right_index=True, how="left")
    # over / under prices belong to the game, not a side
    return games.rename(columns={"over_odds": "ou_over", "under_odds": "ou_under"})


# --------------------------------------------------------------- scoreboard
def month_events(sport: str, league: str, month: str) -> pd.DataFrame:
    """ESPN's games of one month (``"202501"``): ``espn_id``, ``start`` and
    both teams' ESPN abbreviations (``home`` / ``away``)."""
    data = get_json(SITE.format(sport=sport, league=league) + "/scoreboard",
                    dates=month, limit=1000)  # fmt: skip
    return events_frame(data.get("events") or [])


def events_frame(events: list[dict]) -> pd.DataFrame:
    rows = []
    for event in events:
        competition = (event.get("competitions") or [{}])[0]
        sides = {
            c.get("homeAway"): (c.get("team") or {}).get("abbreviation")
            for c in competition.get("competitors") or []
        }
        rows.append(
            {
                "espn_id": str(event["id"]),
                "start": event.get("date"),
                "home": sides.get("home"),
                "away": sides.get("away"),
            }
        )
    out = pd.DataFrame(rows, columns=["espn_id", "start", "home", "away"])
    out["start"] = pd.to_datetime(out["start"], utc=True, errors="coerce")
    return out


def match_events(
    games: pd.DataFrame, events: pd.DataFrame, abbreviations: dict[str, str]
) -> pd.Series:
    """ESPN id of each of ``games`` (``game_date``: US Eastern date,
    ``home_team`` / ``away_team`` keys, ``start``), or NaN.

    Games are matched on date and both teams; a team's second game of a day
    (a doubleheader) to the second ESPN game of those teams that day.
    ``abbreviations`` maps ESPN abbreviations to team keys.
    """
    events = events.dropna(subset=["start"]).copy()
    local = events["start"].dt.tz_convert("America/New_York")
    events["game_date"] = local.dt.strftime("%Y-%m-%d")
    for side in ("home", "away"):
        events[f"{side}_team"] = events[side].map(lambda a: abbreviations.get(a, a))
    keys = ["game_date", "home_team", "away_team"]
    events = events.sort_values("start").drop_duplicates("espn_id")
    events["_n"] = events.groupby(keys).cumcount()
    order = games.sort_values("start", kind="stable")
    order = order.assign(_n=order.groupby(keys).cumcount())
    matched = order[keys + ["_n"]].merge(
        events[keys + ["_n", "espn_id"]], on=keys + ["_n"], how="left"
    )
    return pd.Series(matched["espn_id"].to_numpy(), index=order.index).reindex(
        games.index
    )


# -------------------------------------------------------------------- teams
def color(value) -> str | None:
    if not isinstance(value, str) or not value.strip("#"):
        return None
    return value if value.startswith("#") else f"#{value}"


def site_json(sport: str, league: str, path: str, **params) -> dict:
    response = httpx.get(
        f"{SITE.format(sport=sport, league=league)}/{path}", params=params, timeout=30
    )
    response.raise_for_status()
    return response.json()


def teams_frame(
    teams: list[dict],
    groups: list[dict],
    key: Callable[[dict], str],
    nested: bool = True,
    abbr: bool = False,
) -> pd.DataFrame:
    """ESPN ``/teams`` and ``/groups`` -> web team metadata.

    ``key(team)`` is a team's games-table key. With ``nested`` groups are
    conferences holding divisions (the NBA's East / West); otherwise each
    group is a conference of its own (college basketball) and teams have no
    division. With ``abbr`` each team also gets ESPN's abbreviation as its
    short label (for sports whose keys are long names).
    """
    placed = {}
    for conference in groups:
        name = conference.get("abbreviation") or conference.get("name")
        if nested:
            for division in conference.get("children") or []:
                for team in division.get("teams") or []:
                    placed[str(team["id"])] = (name, division["name"])
        else:
            short = conference.get("shortName") or name
            for team in conference.get("teams") or []:
                placed[str(team["id"])] = (short, None)
    rows = []
    for team in teams:
        team = team.get("team", team)
        logos = team.get("logos") or []
        conference, division = placed.get(str(team["id"]), (None, None))
        rows.append(
            {
                "team": key(team),
                "name": team["displayName"],
                "nickname": team.get("shortDisplayName") or team.get("name"),
                "color": color(team.get("color")),
                "color2": color(team.get("alternateColor")),
                "logo": logos[0]["href"] if logos else None,
                "conference": conference,
                "division": division,
            }
            | ({"abbr": team.get("abbreviation")} if abbr else {})
        )
    return pd.DataFrame(rows).drop_duplicates("team").reset_index(drop=True)


# --------------------------------------------------------------- basketball
# sportsdataverse team box-score columns kept, with their games-table names.
BOX_COLUMNS = {
    "field_goals_made": "fgm",
    "field_goals_attempted": "fga",
    "three_point_field_goals_made": "fg3m",
    "three_point_field_goals_attempted": "fg3a",
    "free_throws_made": "ftm",
    "free_throws_attempted": "fta",
    "offensive_rebounds": "oreb",
    "defensive_rebounds": "dreb",
    "assists": "ast",
    "steals": "stl",
    "blocks": "blk",
    "total_turnovers": "tov",
    "fouls": "pf",
    "points_in_paint": "paint_points",
    "fast_break_points": "fast_break_points",
    "turnover_points": "turnover_points",
}


@dataclass(frozen=True)
class League:
    """An ESPN league with sportsdataverse release files.

    Attributes:
        sport / league: ESPN's path, e.g. ``"basketball"`` / ``"nba"``.
        schedule_url / team_box_url: Release file URLs with ``{season}``.
        season_types: ESPN season type -> ``game_type``; other types
            (preseason, All-Star) are dropped.
        first_lines_season: First season with betting lines.
    """

    sport: str
    league: str
    schedule_url: str
    team_box_url: str
    season_types: dict[int, str] = field(default_factory=dict)
    first_lines_season: int = 0

    def fetch_odds(self, game_id: str) -> dict:
        return fetch_odds(self.sport, self.league, game_id)


def schedule_frame(
    schedule: pd.DataFrame,
    season_types: dict[int, str],
    keep: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> pd.DataFrame:
    """sportsdataverse schedule -> one row per game (home / away columns),
    typed. Games of other season types, and those ``keep(games)`` rejects,
    are dropped."""
    if schedule.empty:
        return pd.DataFrame()
    games = schedule.copy()
    games["season_type"] = pd.to_numeric(games["season_type"], errors="coerce")
    games["game_type"] = games["season_type"].map(season_types)
    games = games.dropna(subset=["game_type"])
    for side in ("home", "away"):
        games[f"{side}_id"] = pd.to_numeric(games[f"{side}_id"], errors="coerce")
        games[f"{side}_score"] = pd.to_numeric(games[f"{side}_score"], errors="coerce")
    games = games.dropna(subset=["home_id", "away_id"])
    if keep is not None:
        games = games[keep(games)]
    games["game_id"] = games["game_id"].astype(str)
    games["start"] = pd.to_datetime(games["date"], utc=True, errors="coerce")
    games["start_date"] = games["start"].dt.strftime("%Y-%m-%dT%H:%MZ")
    games["completed"] = games["status_type_completed"].fillna(False).astype(bool)
    games["neutral_site"] = games["neutral_site"].fillna(False).astype(bool)
    return games.drop_duplicates("game_id").reset_index(drop=True)


def box_frame(team_box: pd.DataFrame) -> pd.DataFrame:
    """sportsdataverse team box scores -> ``game_id``, ``team_id`` and the
    :data:`BOX_COLUMNS` stats."""
    cols = ["game_id", "team_id", *BOX_COLUMNS.values()]
    if team_box.empty:
        return pd.DataFrame(columns=cols)
    box = team_box.rename(columns=BOX_COLUMNS).reindex(columns=cols)
    box["game_id"] = box["game_id"].astype(str)
    box["team_id"] = pd.to_numeric(box["team_id"], errors="coerce")
    stats = list(BOX_COLUMNS.values())
    box[stats] = box[stats].apply(pd.to_numeric, errors="coerce")
    return box.drop_duplicates(["game_id", "team_id"])


def day_numbers(
    games: pd.DataFrame, tz: str = "America/New_York"
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """``(game_date, day, week)`` of every game (``start``, ``season``).

    ``game_date`` is the local (US Eastern) date, unless ``games`` already
    has one (the league's own, ``YYYY-MM-DD``); ``day`` the days since the
    season's first game and ``week`` the Monday-to-Sunday week of the season
    (1: the opener's).
    """
    if "game_date" in games:
        date = pd.to_datetime(games["game_date"]).dt.normalize()
    else:
        local = games["start"].dt.tz_convert(tz)
        date = local.dt.tz_localize(None).dt.normalize()
    first = date.groupby(games["season"]).transform("min")
    monday = first - pd.to_timedelta(first.dt.weekday, unit="D")
    day = (date - first).dt.days
    week = (date - monday).dt.days // 7 + 1
    return date.dt.strftime("%Y-%m-%d"), day, week


def invert(
    home: pd.DataFrame, negated: tuple[str, ...] = ("spread_line",)
) -> pd.DataFrame:
    """Both perspectives of one-row-per-game ``home_*`` / ``away_*`` columns;
    the away rows' ``negated`` columns (the home team's spread) flip sign."""

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

    neutral = home["neutral_site"].to_numpy(dtype=bool)
    away = side("away", "home", np.where(neutral, 0, -1))
    for col in negated:
        if col in away:
            away[col] = -away[col]
    home = side("home", "away", np.where(neutral, 0, 1))
    return pd.concat([home, away], ignore_index=True)


def rest_days(rows: pd.DataFrame, max_rest: int, by: str = "team") -> pd.Series:
    """Days since each team's previous game (1: a back-to-back; capped, and
    openers get the cap). ``by`` names the team column."""
    order = rows.sort_values("game_date", kind="stable")
    dates = pd.to_datetime(order["game_date"])
    previous = dates.groupby([order["season"], order[by]]).shift()
    days = (dates - previous).dt.days.clip(upper=max_rest)
    return days.fillna(max_rest).reindex(rows.index)


def with_opp(rows: pd.DataFrame, col: str, by: str = "team") -> pd.DataFrame:
    """Adds ``opp_<col>``: the opponent row's ``col``, matched on the team
    column ``by`` (``team`` or ``id``) and its ``opp_`` counterpart."""
    opp = rows[["game_id", by, col]].rename(
        columns={by: f"opp_{by}", col: f"opp_{col}"}
    )
    return rows.merge(opp, on=["game_id", f"opp_{by}"], how="left")


def latest_names(rows: pd.DataFrame, column: str) -> pd.Series:
    """Each ESPN team id's ``column`` (abbreviation, school) in its latest
    game of ``rows`` (``id``, ``season``, ``day``), indexed by id.

    Keys built from them keep a renamed or relocated team's history in one
    place. Two ids that end up with the same name (a defunct team whose name
    a new one took over) are told apart by the older one's id.
    """
    rows = rows.dropna(subset=["id", column]).sort_values(["season", "day"])
    latest = rows.drop_duplicates("id", keep="last")[["id", column, "season"]]
    latest = latest.sort_values("season", ascending=False, kind="stable")
    names = latest[column].astype(str)
    taken = names.duplicated()
    ids = latest["id"].astype(int).astype(str)
    names = names.where(~taken, names + " (" + ids + ")")
    return pd.Series(names.to_numpy(), index=latest["id"].to_numpy())


def keep_league_teams(min_games: int) -> Callable[[pd.DataFrame], pd.Series]:
    """A ``keep`` for :func:`build_basketball_games`: games between teams with
    at least ``min_games`` games in the season (drops All-Star and
    exhibition teams)."""

    def keep(games: pd.DataFrame) -> pd.Series:
        sides = pd.concat(
            [games[["season", "home_id"]].set_axis(["season", "id"], axis=1),
             games[["season", "away_id"]].set_axis(["season", "id"], axis=1)]
        )  # fmt: skip
        counts = sides.groupby(["season", "id"]).size()
        home = pd.MultiIndex.from_frame(games[["season", "home_id"]])
        away = pd.MultiIndex.from_frame(games[["season", "away_id"]])
        return pd.Series(
            (counts.reindex(home).to_numpy() >= min_games)
            & (counts.reindex(away).to_numpy() >= min_games),
            index=games.index,
        )

    return keep


def build_basketball_games(
    schedule: pd.DataFrame,
    team_box: pd.DataFrame,
    lines: dict[str, dict] | None,
    today,
    season_types: dict[int, str],
    team_keys: Callable[[pd.DataFrame], tuple[pd.Series, pd.Series]],
    max_rest: int,
    keep: Callable[[pd.DataFrame], pd.Series] | None = None,
    extra: tuple[str, ...] = (),
    side_extra: tuple[str, ...] = (),
) -> pd.DataFrame:
    """One row per team per game from ESPN's schedule, box scores and lines.

    Past games that were never completed (postponed, cancelled) are dropped;
    future ones are kept as upcoming games with missing results and stats.

    Args:
        team_keys: Both-perspective rows (``id``, ``abbreviation``,
            ``school``, ``team_name``, ... and their ``opp_`` columns) ->
            ``(team, opp_team)`` keys.
        keep: Rejects one-row-per-game schedule rows (exhibitions).
        extra / side_extra: Further schedule columns kept, per game and per
            side (without the ``home_`` / ``away_`` prefix).
    """
    games = schedule_frame(schedule, season_types, keep)
    if games.empty:
        return games
    upcoming = ~games["completed"] | games["home_score"].isna()
    stale = upcoming & (games["start"] < pd.Timestamp(today) - pd.Timedelta(days=1))
    games = games[~stale].copy()
    games.loc[upcoming, ["home_score", "away_score"]] = np.nan
    games["game_date"], games["day"], games["week"] = day_numbers(games)
    games["postseason"] = games["game_type"].ne("regular").astype(int)
    games = join_lines(games, lines)

    keep_cols = ["game_id", "season", "day", "week", "game_date", "start_date",
                 "game_type", "postseason", "neutral_site", "notes_headline",
                 "spread_line", "total_line", "ou_over", "ou_under",
                 *extra]  # fmt: skip
    sides = ["id", "abbreviation", "school", "display_name", "score",
             "moneyline", "spread_odds", *side_extra]  # fmt: skip
    cols = keep_cols + [f"{s}_{c}" for s in ("home", "away") for c in sides]
    # ESPN's team "location" is the school / city; ours is home or away
    games = games.rename(
        columns={f"{s}_location": f"{s}_school" for s in ("home", "away")}
    )
    rows = invert(games.reindex(columns=cols))
    rows = rows.rename(
        columns={
            "display_name": "team_name",
            "opp_display_name": "opp_name",
            "score": "points",
            "opp_score": "opp_points",
            "ou_over": "over_odds",
            "ou_under": "under_odds",
            "notes_headline": "notes",
        }
    )
    rows["team"], rows["opp_team"] = team_keys(rows)
    rows["result"] = rows["points"] - rows["opp_points"]
    rows["total"] = rows["points"] + rows["opp_points"]
    # by ESPN id: every pooled team (college's non-Division I) has its own rest
    rows["rest"] = rest_days(rows, max_rest, by="id")
    rows = with_opp(rows, "rest", by="id")

    box = box_frame(team_box)
    stats = list(BOX_COLUMNS.values())
    rows = rows.merge(
        box.rename(columns={"team_id": "id"}), on=["game_id", "id"], how="left"
    )
    rows = rows.merge(
        box.rename(columns={"team_id": "opp_id", **{c: f"opp_{c}" for c in stats}}),
        on=["game_id", "opp_id"],
        how="left",
    )
    # finished games' stats only: a stray box score cannot leak into a preview
    rows.loc[rows["result"].isna(), stats + [f"opp_{c}" for c in stats]] = np.nan
    rows = rows.drop(columns=["id", "opp_id", "abbreviation", "opp_abbreviation",
                              "school", "opp_school"])  # fmt: skip
    return rows.sort_values(["season", "day", "game_id", "location"]).reset_index(
        drop=True
    )


def basketball_season(
    league: League,
    season: int,
    final: bool,
    today,
    cache: Cache,
    lines_days: int,
    wants_lines: Callable[[pd.DataFrame], pd.Series] | None = None,
    bulk_lines: Callable[[int, pd.DataFrame], dict[str, dict]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """``(schedule, team box scores, lines)`` of one season of ``league``.

    ``wants_lines(games)`` limits which games' lines are fetched (default:
    all; college basketball skips games against non-Division I teams).
    ``bulk_lines(season, games)`` gives finished games' lines from another
    source (see :func:`season_lines`); ESPN then has upcoming games' and
    the finished ones it leaves out.
    """
    schedule = season_file(
        league.schedule_url.format(season=season),
        f"schedule_{season}.parquet",
        final,
        cache,
    )
    box = season_file(
        league.team_box_url.format(season=season),
        f"team_box_{season}.parquet",
        final,
        cache,
    )
    lines = {}
    games = schedule_frame(schedule, league.season_types)
    if season >= league.first_lines_season and len(games):
        if wants_lines is not None:
            games = games[wants_lines(games)]
        finished, soon = lines_to_fetch(games, today, lines_days)
        bulk = None
        if bulk_lines is not None:

            def bulk(ids: list[str]) -> dict[str, dict]:
                return bulk_lines(season, games[games["game_id"].isin(ids)])

        lines = season_lines(
            season, finished, soon, cache, league.fetch_odds, bulk=bulk
        )
    return schedule, box, lines
