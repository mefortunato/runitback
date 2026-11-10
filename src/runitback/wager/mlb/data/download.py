"""MLB games, team box scores and starting pitchers from MLB's Stats API,
betting lines from ESPN.

A season is about 70 requests to MLB's own Stats API
(``statsapi.mlb.com``): the schedule with each side's announced starting
pitcher (one request, upcoming games included), every team's per-game
hitting, pitching and fielding logs (two requests per team: regular season
and postseason), and the announced starters' own pitching logs (batches of
50 pitchers). The schedule is the source of truth for which games exist;
postponed and cancelled games are dropped (a postponed game is played
later under the same id).

Starting pitchers are pre-game features: ``sp_*`` / ``opp_sp_*`` describe
each side's announced starter from his earlier starts this season and last
season, shrunk toward a league-average starter (:func:`starter_features`).

Betting lines come from ESPN's odds API, one request per game, from 2013 on
(see :mod:`runitback.wager.espn`). ESPN's game ids come from its scoreboard, one
request per month, matched to MLB's games on date and teams (doubleheaders
in order). ESPN's "spread" is the run line (a 1.5-run handicap priced like a
moneyline); it is kept as ``run_line`` with its prices, while the market's
win probability comes from the moneylines. Finished seasons' responses and
finished games' lines are cached on disk (``cache_dir``).

Team keys are MLB's current abbreviations, by team id, so a renamed or
moved team keeps one history (Indians -> Guardians, Oakland -> the
Athletics); the name it played under stays in ``team_name`` / ``opp_name``.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ... import espn

logger = logging.getLogger(__name__)

API = "https://statsapi.mlb.com/api/v1"

# MLB game types kept: R regular season, F wild card, D division series,
# L league championship series, W World Series (spring training,
# exhibitions and the All-Star game are dropped).
GAME_TYPES = {"R": "regular", "F": "wild_card", "D": "division", "L": "lcs",
              "W": "world_series"}  # fmt: skip

# detailedState of scheduled games that will not be played on that date.
NOT_PLAYED = {"Postponed", "Cancelled", "Suspended"}

# ESPN abbreviations that differ from MLB's current ones.
ESPN_ABBREVIATIONS = {"CHW": "CWS", "ARI": "AZ", "OAK": "ATH", "WSH": "WSH",
                      "FLA": "MIA"}  # fmt: skip

# Schedule fields requested (the full response is ~15 MB a season).
SCHEDULE_FIELDS = (
    "dates,date,games,gamePk,gameType,season,gameDate,officialDate,status,"
    "detailedState,abstractGameState,teams,away,home,team,id,name,score,"
    "probablePitcher,fullName,doubleHeader,gameNumber"
)

# Per-game team logs: (group, stat) -> games-table name.
BOX_STATS = {
    ("hitting", "runs"): "runs",
    ("hitting", "hits"): "hits",
    ("hitting", "doubles"): "doubles",
    ("hitting", "triples"): "triples",
    ("hitting", "homeRuns"): "home_runs",
    ("hitting", "baseOnBalls"): "walks",
    ("hitting", "hitByPitch"): "hit_by_pitch",
    ("hitting", "strikeOuts"): "strikeouts",
    ("hitting", "atBats"): "at_bats",
    ("hitting", "plateAppearances"): "plate_appearances",
    ("hitting", "totalBases"): "total_bases",
    ("hitting", "stolenBases"): "stolen_bases",
    ("hitting", "caughtStealing"): "caught_stealing",
    ("hitting", "leftOnBase"): "left_on_base",
    ("hitting", "groundIntoDoublePlay"): "double_plays_grounded",
    ("hitting", "sacFlies"): "sac_flies",
    ("pitching", "numberOfPitches"): "pitches",
    ("pitching", "earnedRuns"): "earned_runs_allowed",
    ("fielding", "errors"): "errors",
}
BOX_COLUMNS = list(BOX_STATS.values())

# Starting-pitcher features, with the league-average prior each is shrunk
# toward and its weight (in innings, batters faced or starts).
STARTER_COLUMNS = ["sp_era", "sp_k_bb", "sp_ip", "sp_starts"]
PRIOR_ERA, PRIOR_INNINGS = 4.40, 30.0
PRIOR_K_BB, PRIOR_BATTERS = 0.13, 120.0
PRIOR_IP, PRIOR_STARTS = 5.2, 5.0

# The first season with betting lines on ESPN (moneylines; earlier seasons
# list placeholder prices).
FIRST_LINES_SEASON = 2013

# Longest rest counted, in days (season openers get it too).
MAX_REST = 5

# Odds lookahead: upcoming games starting within this many days get lines.
LINES_DAYS = 2


def season_of(day: datetime.date) -> int:
    return day.year


def season_complete(season: int, today: datetime.datetime) -> bool:
    """Whether nothing in ``season`` can change (the World Series ends by
    early November)."""
    return today.date() > datetime.date(season, 11, 30)


def months(season: int) -> list[str]:
    """``YYYYMM`` of every month with games (March openers to November)."""
    return [f"{season}{m:02d}" for m in range(3, 12)]


def team_keys(records: list[dict]) -> dict[int, str]:
    """MLB ``/teams`` (the latest season) -> team id: abbreviation."""
    return {int(t["id"]): t["abbreviation"] for t in records if t.get("abbreviation")}


def innings(value) -> float:
    """``"5.2"`` (5 innings and 2 outs) -> 5.667."""
    try:
        whole, _, outs = str(value).partition(".")
        return int(whole) + int(outs or 0) / 3
    except ValueError:
        return np.nan


# ----------------------------------------------------------------- schedule
def schedule_frame(data: dict, keys: dict[int, str]) -> pd.DataFrame:
    """MLB ``/schedule`` -> one row per game (home / away columns), typed.

    Postponed, cancelled and suspended listings are dropped; a game listed
    twice (a resumed or makeup game keeps its id) keeps its last listing.
    """
    rows = []
    for date in data.get("dates") or []:
        for game in date.get("games") or []:
            status = game.get("status") or {}
            row = {
                "game_id": str(game["gamePk"]),
                "season": int(game["season"]),
                "type": game.get("gameType"),
                "start": game.get("gameDate"),
                "game_date": game.get("officialDate"),
                "state": status.get("detailedState"),
                "final": status.get("abstractGameState") == "Final",
                "doubleheader": game.get("doubleHeader"),
                "game_number": game.get("gameNumber"),
            }
            for side in ("home", "away"):
                team = (game.get("teams") or {}).get(side) or {}
                pitcher = team.get("probablePitcher") or {}
                row |= {
                    f"{side}_id": (team.get("team") or {}).get("id"),
                    f"{side}_display_name": (team.get("team") or {}).get("name"),
                    f"{side}_score": team.get("score"),
                    f"{side}_sp_id": pitcher.get("id"),
                    f"{side}_sp_name": pitcher.get("fullName"),
                }
            rows.append(row)
    games = pd.DataFrame(rows)
    if games.empty:
        return games
    games["game_type"] = games["type"].map(GAME_TYPES)
    games = games.dropna(subset=["game_type"])
    games = games[~games["state"].isin(NOT_PLAYED)].copy()
    games["start"] = pd.to_datetime(games["start"], utc=True, errors="coerce")
    games["start_date"] = games["start"].dt.strftime("%Y-%m-%dT%H:%MZ")
    games["completed"] = games["final"]
    for side in ("home", "away"):
        games[f"{side}_id"] = pd.to_numeric(games[f"{side}_id"], errors="coerce")
        games[f"{side}_team"] = games[f"{side}_id"].map(keys)
        games[f"{side}_score"] = pd.to_numeric(games[f"{side}_score"], errors="coerce")
    games["neutral_site"] = False
    games = games.dropna(subset=["home_team", "away_team"])
    games = games.sort_values(["completed", "start"], kind="stable")
    return games.drop_duplicates("game_id", keep="last").reset_index(drop=True)


def box_frame(logs: list[dict]) -> pd.DataFrame:
    """Teams' ``gameLog`` responses -> ``game_id``, ``id`` (team) and the
    :data:`BOX_COLUMNS` stats."""
    rows = {}
    for log in logs:
        for stats in log.get("stats") or []:
            group = (stats.get("group") or {}).get("displayName")
            for split in stats.get("splits") or []:
                key = (str(split["game"]["gamePk"]), split["team"]["id"])
                row = rows.setdefault(key, {"game_id": key[0], "id": key[1]})
                for name, value in (split.get("stat") or {}).items():
                    if (group, name) in BOX_STATS:
                        row[BOX_STATS[group, name]] = value
    box = pd.DataFrame(list(rows.values())).reindex(columns=["game_id", "id",
                                                             *BOX_COLUMNS])  # fmt: skip
    box[BOX_COLUMNS] = box[BOX_COLUMNS].apply(pd.to_numeric, errors="coerce")
    box["id"] = pd.to_numeric(box["id"], errors="coerce")
    return box


# ------------------------------------------------------------------ starters
def starts_frame(people: list[dict]) -> pd.DataFrame:
    """Pitchers' ``gameLog`` (``/people?hydrate=stats(...)``) -> one row per
    start: ``sp_id``, ``season``, ``date``, ``game_id`` and the start's
    ``outs``, ``earned_runs``, ``strikeouts``, ``walks``, ``batters``."""
    rows = []
    for person in people:
        for stats in person.get("stats") or []:
            for split in stats.get("splits") or []:
                stat = split.get("stat") or {}
                if not stat.get("gamesStarted"):
                    continue
                rows.append(
                    {
                        "sp_id": int(person["id"]),
                        "season": int(split.get("season") or 0),
                        "date": split.get("date"),
                        "game_id": str((split.get("game") or {}).get("gamePk")),
                        "outs": stat.get("outs"),
                        "earned_runs": stat.get("earnedRuns"),
                        "strikeouts": stat.get("strikeOuts"),
                        "walks": stat.get("baseOnBalls"),
                        "batters": stat.get("battersFaced"),
                    }
                )
    cols = ["sp_id", "season", "date", "game_id", "outs", "earned_runs",
            "strikeouts", "walks", "batters"]  # fmt: skip
    starts = pd.DataFrame(rows, columns=cols)
    starts[cols[4:]] = starts[cols[4:]].apply(pd.to_numeric, errors="coerce")
    starts["date"] = pd.to_datetime(starts["date"], errors="coerce")
    return starts.dropna(subset=["date"]).drop_duplicates(["sp_id", "game_id"])


TOTALS = ["starts", "outs", "earned_runs", "strikeouts", "walks", "batters"]


def starter_features(rows: pd.DataFrame, starts: pd.DataFrame) -> pd.DataFrame:
    """Each row's announced starter (``sp_id``) going into the game.

    From his starts before the game's date this season plus all of last
    season's (``starts``): ``sp_era`` (earned runs per 9 innings, shrunk
    toward :data:`PRIOR_ERA` with :data:`PRIOR_INNINGS` innings),
    ``sp_k_bb`` (strikeouts minus walks per batter faced, shrunk the same
    way by batters), ``sp_ip`` (innings per start) and ``sp_starts`` (how
    many starts that is). A starter with no record (a debut) gets the
    priors; rows without an announced starter get NaN.
    """
    out = pd.DataFrame(np.nan, index=rows.index, columns=STARTER_COLUMNS)
    have = rows["sp_id"].notna() & rows["game_date"].notna()
    if not have.any():
        return out
    starts = starts.assign(starts=1.0).sort_values("date")
    # this season through each start, then last season's totals
    current = starts.groupby(["sp_id", "season"])[TOTALS].cumsum()
    current = starts[["sp_id", "season", "date"]].join(current)
    last = starts.groupby(["sp_id", "season"])[TOTALS].sum().reset_index()
    last["season"] += 1

    games = rows.loc[have, ["sp_id", "season", "game_date"]].copy()
    games["sp_id"] = games["sp_id"].astype(int)
    games["date"] = pd.to_datetime(games["game_date"])
    games = games.reset_index().sort_values("date")
    before = pd.merge_asof(
        games,
        current.sort_values("date"),
        on="date",
        by=["sp_id", "season"],
        allow_exact_matches=False,
    )
    before = before.merge(last, on=["sp_id", "season"], how="left",
                          suffixes=("", "_last"))  # fmt: skip
    total = {c: before[c].fillna(0) + before[f"{c}_last"].fillna(0) for c in TOTALS}
    ip = total["outs"] / 3
    frame = pd.DataFrame(
        {
            "sp_era": (9 * total["earned_runs"] + PRIOR_ERA * PRIOR_INNINGS)
            / (ip + PRIOR_INNINGS),
            "sp_k_bb": (
                total["strikeouts"] - total["walks"] + PRIOR_K_BB * PRIOR_BATTERS
            )
            / (total["batters"] + PRIOR_BATTERS),
            "sp_ip": (ip + PRIOR_IP * PRIOR_STARTS) / (total["starts"] + PRIOR_STARTS),
            "sp_starts": total["starts"],
        }
    )
    frame.index = before["index"].to_numpy()
    out.loc[frame.index, STARTER_COLUMNS] = frame[STARTER_COLUMNS].to_numpy()
    return out


# -------------------------------------------------------------------- games
def build_games(
    schedule: list[dict],
    logs: list[dict],
    people: list[dict],
    teams: list[dict],
    lines: dict[str, dict] | None = None,
    today: datetime.datetime | None = None,
) -> pd.DataFrame:
    """One row per team per game from MLB's schedules (``/schedule``
    responses), team logs, starters' logs and teams, and ESPN's lines (keyed
    by MLB's game id).

    Past games that were never completed are dropped; future ones are kept
    as upcoming games with missing results and stats.
    """
    from .features import STATS_COLUMNS

    today = today or datetime.datetime.now(datetime.timezone.utc)
    keys = team_keys(teams)
    games = pd.concat([schedule_frame(s, keys) for s in schedule], ignore_index=True)
    if games.empty:
        return games
    games = games.drop_duplicates("game_id", keep="last")
    upcoming = ~games["completed"] | games["home_score"].isna()
    stale = upcoming & (games["start"] < pd.Timestamp(today) - pd.Timedelta(days=1))
    games = games[~stale].copy()
    games.loc[upcoming, ["home_score", "away_score"]] = np.nan
    games["game_date"], games["day"], games["week"] = espn.day_numbers(games)
    games["postseason"] = games["game_type"].ne("regular").astype(int)
    games = espn.join_lines(games, lines)
    games = games.rename(columns={"spread_line": "run_line"})
    for side in ("home", "away"):
        games = games.rename(columns={f"{side}_spread_odds": f"{side}_run_line_odds"})

    keep = ["game_id", "season", "day", "week", "game_date", "start_date",
            "game_type", "postseason", "neutral_site", "doubleheader",
            "game_number", "run_line", "total_line", "ou_over", "ou_under"]  # fmt: skip
    sides = ["id", "team", "display_name", "score", "moneyline", "run_line_odds",
             "sp_id", "sp_name"]  # fmt: skip
    cols = keep + [f"{s}_{c}" for s in ("home", "away") for c in sides]
    rows = espn.invert(games.reindex(columns=cols), negated=("run_line",))
    rows = rows.rename(
        columns={
            "display_name": "team_name",
            "opp_display_name": "opp_name",
            "score": "runs",
            "opp_score": "opp_runs",
            "ou_over": "over_odds",
            "ou_under": "under_odds",
        }
    )
    rows["result"] = rows["runs"] - rows["opp_runs"]
    rows["total"] = rows["runs"] + rows["opp_runs"]
    rows["rest"] = espn.rest_days(rows, MAX_REST)
    rows = espn.with_opp(rows, "rest")

    rows = rows.join(starter_features(rows, starts_frame(people)))
    for col in STARTER_COLUMNS:
        rows = espn.with_opp(rows, col)

    box = box_frame(logs)
    stats = [c for c in STATS_COLUMNS if c != "runs"]
    box = box.reindex(columns=["game_id", "id", *stats])
    rows = rows.merge(box, on=["game_id", "id"], how="left")
    rows = rows.merge(
        box.rename(columns={"id": "opp_id", **{c: f"opp_{c}" for c in stats}}),
        on=["game_id", "opp_id"],
        how="left",
    )
    # finished games' stats only: a stray box score cannot leak into a preview
    rows.loc[rows["result"].isna(), stats + [f"opp_{c}" for c in stats]] = np.nan
    rows = rows.drop(columns=["id", "opp_id", "neutral_site"])
    return rows.sort_values(["season", "day", "game_id", "location"]).reset_index(
        drop=True
    )


# ----------------------------------------------------------------- download
def espn_keys(events: pd.DataFrame) -> dict[str, str]:
    """ESPN abbreviation -> games-table key, for every team in ``events``."""
    return {
        a: ESPN_ABBREVIATIONS.get(a, a)
        for a in set(events["home"]) | set(events["away"])
        if isinstance(a, str)
    }


def espn_ids(
    games: pd.DataFrame, season: int, final: bool, cache: espn.Cache
) -> pd.Series:
    """ESPN's id of each of a season's games (``schedule_frame`` rows), from
    its monthly scoreboards (finished seasons' cached)."""
    events = pd.concat(
        [
            espn.events_frame(
                espn.get_json(
                    espn.SITE.format(sport="baseball", league="mlb") + "/scoreboard",
                    f"scoreboard_{month}.json" if final else None,
                    cache,
                    dates=month,
                    limit=1000,
                ).get("events")
                or []
            )
            for month in months(season)
        ],
        ignore_index=True,
    )
    local = games["start"].dt.tz_convert("America/New_York")
    frame = games.assign(game_date=local.dt.strftime("%Y-%m-%d"))
    return espn.match_events(frame, events, espn_keys(events))


def fetch_season(
    season: int,
    today: datetime.datetime,
    cache: espn.Cache,
    teams: list[dict],
    lines_days: int = LINES_DAYS,
) -> tuple[dict, list[dict], list[dict], dict]:
    """``(schedule, team logs, starters' logs, lines)`` of one season."""
    final = season_complete(season, today)

    def get(path: str, name: str, **params):
        return espn.get_json(f"{API}{path}", name if final else None, cache, **params)

    schedule = get(
        "/schedule",
        f"schedule_{season}.json",
        sportId=1,
        season=season,
        gameType=",".join(GAME_TYPES),
        hydrate="probablePitcher",
        fields=SCHEDULE_FIELDS,
    )
    games = schedule_frame(schedule, team_keys(teams))
    logs = []
    if len(games) and games["completed"].any():
        played = set(games.loc[games["completed"], "home_id"]) | set(
            games.loc[games["completed"], "away_id"]
        )
        postseason = set(
            games.loc[games["completed"] & games["game_type"].ne("regular"),
                      ["home_id", "away_id"]].to_numpy().ravel()
        )  # fmt: skip
        for team_id in sorted(int(t) for t in played):
            for game_type in ("R", "P") if team_id in postseason else ("R",):
                logs.append(
                    get(
                        f"/teams/{team_id}/stats",
                        f"log_{season}_{team_id}_{game_type}.json",
                        stats="gameLog",
                        group="hitting,pitching,fielding",
                        season=season,
                        gameType=game_type,
                    )
                )
    pitchers = sorted(
        {int(p) for side in ("home", "away") for p in games.get(f"{side}_sp_id",
                                                                 []) if pd.notna(p)}
    )  # fmt: skip
    people = []
    for start in range(0, len(pitchers), 50):
        batch = pitchers[start : start + 50]
        people += get(
            "/people",
            f"people_{season}_{start // 50}.json",
            personIds=",".join(map(str, batch)),
            hydrate=f"stats(group=pitching,type=gameLog,season={season})",
        ).get("people", [])

    lines = {}
    if season >= FIRST_LINES_SEASON and len(games):
        games["espn_id"] = espn_ids(games, season, final, cache)
        games = games.dropna(subset=["espn_id"])
        finished, soon = espn.lines_to_fetch(games, today, lines_days)
        to_espn = dict(zip(games["game_id"], games["espn_id"]))
        by_espn = espn.season_lines(
            season,
            [to_espn[g] for g in finished],
            [to_espn[g] for g in soon],
            cache,
            lambda g: espn.fetch_odds("baseball", "mlb", g),
        )
        lines = {g: by_espn[e] for g, e in to_espn.items() if e in by_espn}
    return schedule, logs, people, lines


def download(
    output_file: str | None = None,
    first_season: int = 2008,
    last_season: int | None = None,
    cache_dir: str | None = "data/mlb/mlb_cache",
) -> pd.DataFrame:
    """Download every season's games, box scores, starters and lines.

    Args:
        output_file: Also write the games table here.
        first_season / last_season: Seasons to download (default: 2008
            through the current season). ESPN's betting lines start in 2013.
        cache_dir: Reuse finished seasons' responses and finished games'
            lines from here (``None``: always download). A full download
            makes about 35,000 odds requests; a daily refresh a few dozen.
    """
    today = datetime.datetime.now(datetime.timezone.utc)
    last_season = last_season or season_of(today.date())
    cache = espn.Cache(cache_dir)
    teams = espn.get_json(f"{API}/teams", sportId=1)["teams"]
    schedules, logs, people, lines = [], [], [], {}
    for season in range(first_season, last_season + 1):
        schedule, season_logs, season_people, season_lines = fetch_season(
            season, today, cache, teams
        )
        schedules.append(schedule)
        logs += season_logs
        people += season_people
        lines |= season_lines
        logger.info(
            "%d: %d team logs, %d starters, %d games with lines",
            season,
            len(season_logs),
            len(season_people),
            len(season_lines),
        )
    games = build_games(schedules, logs, people, teams, lines, today=today)
    if output_file is not None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        games.to_csv(output_file, index=False)
    return games
