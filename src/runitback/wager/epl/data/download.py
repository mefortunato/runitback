"""Premier League games, box scores and betting odds from football-data.co.uk.

One CSV per season (``/mmz4281/<YYZZ>/E0.csv``) holds every game's score,
half-time score, referee, shots, shots on target, corners, fouls, cards and
bookmakers' prices (1X2, over / under 2.5 goals, Asian handicap); the
current season's file is complete up to the last game played. The
all-leagues ``/fixtures.csv`` lists the coming week's games, with their
opening prices; its Premier League rows are kept as upcoming games.

Seasons are keyed by their ending year (2025-26 is 2026; ``build_games``
takes a ``season`` column from each file) and games by
``<season>_<home>_<away>`` (a pair meets once at each ground a season).
Team keys are three-letter codes (:mod:`.teams`). Every game is at the home
team's ground (``location`` is +1 / -1), so there is no neutral site.

The files are hand-maintained, and the parser copes with their history:
latin-1 files, two- and four-digit years, blank trailing rows and the
2003-04 and 2004-05 files whose later rows are wider than their header (they
list more bookmakers; the columns the header names line up, the extras are
dropped).

**Odds.** Bookmakers and column names changed over the years, so each price
comes from the first bookmaker in a preference order that quotes a full set
for the game: closing Pinnacle, closing market average, closing bet365, then
the same before the closing prices existed (opening), then older bookmakers
(William Hill, Interwetten, Ladbrokes). ``odds_source`` names the one used
for the 1X2 prices; totals and handicaps use their own order. All prices are
converted to American odds (like the other sports'), the Asian handicap to
the home team's expected margin (``-AHh``). Odds start in 2000-01; there are
none before 2000-01 and for the unfinished games the coming week's opening
prices only.
"""

from __future__ import annotations

import csv
import datetime
import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ... import espn
from .teams import team_codes

logger = logging.getLogger(__name__)

BASE = "https://www.football-data.co.uk"

# The first season (by ending year) downloaded by default: 2000-01, the first
# with odds, shots and cards.
FIRST_SEASON = 2001

# Longest rest counted, in days (season openers get it too).
MAX_REST = 14

# football-data column -> (home, away) games-table stat
BOX = {"goals": ("FTHG", "FTAG"), "half_time_goals": ("HTHG", "HTAG"),
       "shots": ("HS", "AS"), "shots_on_target": ("HST", "AST"),
       "corners": ("HC", "AC"), "fouls": ("HF", "AF"),
       "yellow_cards": ("HY", "AY"), "red_cards": ("HR", "AR")}  # fmt: skip

# (label, home, draw, away) columns, in order of preference.
BOOKS_1X2 = [
    ("pinnacle_close", "PSCH", "PSCD", "PSCA"),
    ("pinnacle_close", "PPCH", "PPCD", "PPCA"),
    ("average_close", "AvgCH", "AvgCD", "AvgCA"),
    ("bet365_close", "B365CH", "B365CD", "B365CA"),
    ("pinnacle", "PSH", "PSD", "PSA"),
    ("pinnacle", "PPH", "PPD", "PPA"),
    ("average", "AvgH", "AvgD", "AvgA"),
    ("bet365", "B365H", "B365D", "B365A"),
    ("williamhill", "WHH", "WHD", "WHA"),
    ("interwetten", "IWH", "IWD", "IWA"),
    ("ladbrokes", "LBH", "LBD", "LBA"),
]
# The closing market average: its own columns (from 2019-20), whatever bookmaker the
# prices above come from, so one baseline can score every test season.
AVERAGE_1X2 = ("AvgCH", "AvgCD", "AvgCA")
AVERAGE_TOTAL = ("AvgC>2.5", "AvgC<2.5")
# (label, over, under) of the 2.5-goal line.
BOOKS_TOTAL = [
    ("pinnacle_close", "PC>2.5", "PC<2.5"),
    ("average_close", "AvgC>2.5", "AvgC<2.5"),
    ("bet365_close", "B365C>2.5", "B365C<2.5"),
    ("pinnacle", "P>2.5", "P<2.5"),
    ("average", "Avg>2.5", "Avg<2.5"),
    ("bet365", "B365>2.5", "B365<2.5"),
    ("average", "BbAv>2.5", "BbAv<2.5"),
]
# Home team's handicap, closing first.
HANDICAPS = ["AHCh", "AHh", "BbAHh"]
TOTAL_LINE = 2.5


def season_of(day: datetime.date) -> int:
    """The season a date belongs to, by its ending year (2025-26 is 2026)."""
    return day.year + 1 if day.month >= 7 else day.year


def season_complete(season: int, today: datetime.datetime) -> bool:
    """Whether nothing in ``season`` can change (the latest end was 26 July)."""
    return today.date() > datetime.date(season, 7, 31)


def season_code(season: int) -> str:
    """``2026`` -> ``"2526"``, football-data's directory name."""
    return f"{(season - 1) % 100:02d}{season % 100:02d}"


# ------------------------------------------------------------------ parsing
def read_csv(data: bytes) -> pd.DataFrame:
    """A football-data.co.uk file as a frame of strings, blank rows dropped
    and rows wider than the header cut to it."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return pd.DataFrame()
    header = [h.strip() for h in rows[0]]
    while header and not header[-1]:
        header.pop()
    out = []
    for row in rows[1:]:
        while row and not row[-1].strip():
            row.pop()
        if not row or not any(cell.strip() for cell in row):
            continue
        # later rows of some files carry extra bookmakers past the header's
        # columns, which the first ones still line up with: drop the extras
        out.append(row[: len(header)] + [""] * (len(header) - len(row)))
    frame = pd.DataFrame(out, columns=header).replace("", np.nan)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    return frame.dropna(subset=["HomeTeam", "AwayTeam", "Date"])


def american(decimal: pd.Series) -> pd.Series:
    """Decimal odds -> American odds (rounded; NaN where not a price)."""
    d = pd.to_numeric(decimal, errors="coerce").where(lambda s: s > 1)
    return np.where(d >= 2, (d - 1) * 100, -100 / (d - 1)).round()


def pick(raw: pd.DataFrame, books: list[tuple]) -> tuple[pd.DataFrame, pd.Series]:
    """The first bookmaker of ``books`` (``(label, *columns)``) with a price
    in every column, per game: ``(prices, labels)``."""
    width = len(books[0]) - 1
    prices = pd.DataFrame(np.nan, index=raw.index, columns=range(width))
    labels = pd.Series(np.nan, index=raw.index, dtype=object)
    for label, *columns in books:
        if not all(c in raw for c in columns):
            continue
        values = raw[columns].apply(pd.to_numeric, errors="coerce")
        ok = (values > 1).all(axis=1) & prices[0].isna()
        prices.loc[ok] = values[ok].to_numpy()
        labels[ok] = label
    return prices, labels


def parse_dates(raw: pd.DataFrame) -> pd.DataFrame:
    """``game_date`` (``YYYY-MM-DD``) and ``start_date`` (UTC, where the file
    has kickoff times) of every game."""
    date = pd.to_datetime(raw["Date"], format="mixed", dayfirst=True)
    out = pd.DataFrame({"game_date": date.dt.strftime("%Y-%m-%d")}, index=raw.index)
    out["start_date"] = np.nan
    if "Time" in raw:
        clock = pd.to_datetime(raw["Time"], format="%H:%M", errors="coerce")
        local = date + pd.to_timedelta(clock.dt.hour * 60 + clock.dt.minute, unit="m")
        local = local.dt.tz_localize("Europe/London", ambiguous="NaT",
                                      nonexistent="shift_forward")  # fmt: skip
        start = local.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%MZ")
        out["start_date"] = start.where(clock.notna())
    return out


# -------------------------------------------------------------------- games
def build_games(
    raw: pd.DataFrame, today: datetime.datetime | None = None
) -> pd.DataFrame:
    """One row per team per game from the concatenated season files
    (``read_csv`` frames; ``fixtures.csv``'s rows are upcoming games).

    Past games without a result (postponed, not yet played) are dropped;
    future ones are kept as upcoming games with missing results and stats.
    """
    today = today or datetime.datetime.now(datetime.timezone.utc)
    raw = raw.reset_index(drop=True)
    games = parse_dates(raw)
    # a file's own season (the 2019-20 one ran to August 2020); fixtures' by date
    games["season"] = (
        raw["season"]
        .fillna(pd.to_datetime(games["game_date"]).dt.date.map(season_of))
        .astype(int)
    )
    for stat, (home, away) in BOX.items():
        for side, col in (("home", home), ("away", away)):
            games[f"{side}_{stat}"] = pd.to_numeric(raw.get(col), errors="coerce")
    played = games["home_goals"].notna() & games["away_goals"].notna()
    cutoff = (pd.Timestamp(today).tz_localize(None) - pd.Timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    games = games[played | (games["game_date"] >= cutoff)].copy()
    raw = raw.loc[games.index]
    played = played.loc[games.index]
    # a game still to be played has no result, whatever else a file holds
    stats = [c for c in games if c.startswith(("home_", "away_"))]
    games.loc[~played, stats] = np.nan

    names = team_codes(set(raw["HomeTeam"]) | set(raw["AwayTeam"]))
    for side, col in (("home", "HomeTeam"), ("away", "AwayTeam")):
        games[f"{side}_team"] = raw[col].map(lambda n: names[n][0])
        games[f"{side}_name"] = raw[col].map(lambda n: names[n][1])
    games["game_id"] = (games["season"].astype(str) + "_" + games["home_team"]
                        + "_" + games["away_team"])  # fmt: skip

    prices, games["odds_source"] = pick(raw, BOOKS_1X2)
    games["home_moneyline"] = american(prices[0])
    games["draw_moneyline"] = american(prices[1])
    games["away_moneyline"] = american(prices[2])
    totals, _ = pick(raw, BOOKS_TOTAL)
    games["over_odds"], games["under_odds"] = american(totals[0]), american(totals[1])
    games["total_line"] = np.where(totals[0].notna(), TOTAL_LINE, np.nan)
    average, _ = pick(raw, [("average_close", *AVERAGE_1X2)])
    games["home_avg_moneyline"] = american(average[0])
    games["avg_draw_moneyline"] = american(average[1])
    games["away_avg_moneyline"] = american(average[2])
    average_total, _ = pick(raw, [("average_close", *AVERAGE_TOTAL)])
    games["avg_over_odds"] = american(average_total[0])
    games["avg_under_odds"] = american(average_total[1])
    handicap = raw.reindex(columns=HANDICAPS).apply(pd.to_numeric, errors="coerce")
    games["spread_line"] = -handicap.bfill(axis=1).iloc[:, 0]
    games["referee"] = raw.get("Referee")

    games["played"] = played
    games = games.sort_values(
        ["played", "game_date"], ascending=[False, True], kind="stable"
    )
    games = games.drop_duplicates("game_id").drop(columns="played")  # a played row wins
    games["game_date"], games["day"], games["week"] = espn.day_numbers(games)
    games["neutral_site"] = False
    rows = espn.invert(games, negated=("spread_line",))
    rows = rows.rename(columns={"name": "team_name", "opp_name": "opp_name"})
    rows["result"] = rows["goals"] - rows["opp_goals"]
    rows["total"] = rows["goals"] + rows["opp_goals"]
    rows["rest"] = espn.rest_days(rows, MAX_REST)
    rows = espn.with_opp(rows, "rest")
    rows = rows.drop(columns=["neutral_site"])
    return rows.sort_values(["season", "day", "game_id", "location"],
                            ascending=[True, True, True, False]).reset_index(drop=True)  # fmt: skip


# ----------------------------------------------------------------- download
def fetch(url: str, name: str | None, cache: espn.Cache) -> bytes:
    """A file's bytes; with ``name``, cached in (and read from) ``cache``."""
    data = cache.read_bytes(name) if name else None
    if data is None:
        response = espn._get(url)
        response.raise_for_status()
        data = response.content
        if name:
            cache.write_bytes(name, data)
    return data


def download(
    output_file: str | None = None,
    first_season: int = FIRST_SEASON,
    last_season: int | None = None,
    cache_dir: str | None = "data/epl/epl_cache",
) -> pd.DataFrame:
    """Download every season's games, statistics and odds.

    Args:
        output_file: Also write the games table here.
        first_season / last_season: Seasons to download, by ending year
            (default: 2000-01, the first with odds, through the current
            season). Earlier files only have results.
        cache_dir: Reuse finished seasons' files from here (``None``: always
            download). A full download is 27 small files.
    """
    today = datetime.datetime.now(datetime.timezone.utc)
    last_season = last_season or season_of(today.date())
    cache = espn.Cache(cache_dir)
    frames = []
    for season in range(first_season, last_season + 1):
        final = season_complete(season, today)
        code = season_code(season)
        data = fetch(
            f"{BASE}/mmz4281/{code}/E0.csv", f"E0_{code}.csv" if final else None, cache
        )
        frame = read_csv(data).assign(season=season)
        frames.append(frame)
        logger.info("%d: %d games", season, len(frame))
    if last_season >= season_of(today.date()):
        fixtures = read_csv(fetch(f"{BASE}/fixtures.csv", None, cache))
        if len(fixtures):
            frames.append(fixtures[fixtures["Div"] == "E0"])
    raw = pd.concat(frames, ignore_index=True)
    games = build_games(raw, today=today)
    if output_file is not None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        games.to_csv(output_file, index=False)
    return games
