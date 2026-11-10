"""Men's college basketball possessions with the ten players on the floor, from CollegeBasketballData.

CBBD (``CBBD_API_KEY``) lists the ten players on the floor (``onFloor``) on every play from the
2023-24 season on (``experiments/cbb/lineup_probe.md``). Its API has no bulk lineup endpoint: the
cheapest source is ``/plays/date``, **one call per game day** (about 150 a season, 35-80 MB each).
The key's monthly call limit (``x-calllimit-remaining``) is the constraint, so:

- :func:`fetch_days` caches each day as a compact parquet file (``plays_<date>.parquet``, about
  1 MB) in ``cache_dir`` and never fetches a cached day again; it stops before the remaining calls
  fall below ``reserve``.
- Player ids: ``onFloor`` carries CBBD's own athlete ids. They are matched to ESPN athlete ids
  (which carry across transfers, as every other table here) **by name within the game's team**,
  from the ESPN box score (:func:`espn_ids`), so no extra calls are needed.

:func:`season_possessions` cuts the plays into possessions with the NBA's rules
(:func:`runitback.nba.player_values.game_possessions`): a possession is a run of one team's actions,
ended by a made shot (unless an and-one follows), a made last free throw, a turnover, the other
team's defensive rebound or the end of the period. Its target is the offense's points (technical
free throws excluded) and its players the ten on the floor at its first action. Possessions without
five matched players per side are dropped (C1: some 2024 games have stretches without a full ten).
"""
from __future__ import annotations

import datetime
import itertools
import logging
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

API = "https://api.collegebasketballdata.com"
KEY_ENV = "CBBD_API_KEY"
FIRST_SEASON = 2024          # onFloor is empty on every play before 2023-24
RESERVE = 25                 # calls left untouched for the wager pipeline's lines

PLAY_COLUMNS = ["cbbd_game_id", "game_id", "play_id", "period", "seconds", "play_type", "text", "side",
                "scoring", "shooting", "score_value", "home_ids", "home_names", "away_ids", "away_names"]
POSS_COLUMNS = ["game_id", "season", "game_date", "t", "period", "off_home", "start", "points",
                *[f"off_{i}" for i in range(1, 6)], *[f"def_{i}" for i in range(1, 6)]]
FT_OF = re.compile(r"(\d)\s*of\s*(\d)")
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


# ----------------------------------------------------------------------------------- fetching

def _get(path: str, key: str, **params) -> tuple[list[dict], int | None]:
    """One CBBD request (a day of plays is tens of MB, so a transfer cut off midway is retried)."""
    import requests

    for attempt in range(4):
        try:
            r = requests.get(f"{API}/{path}", headers={"Authorization": f"Bearer {key}"}, params=params,
                             timeout=600)
            r.raise_for_status()
            left = r.headers.get("x-calllimit-remaining")
            return r.json(), (int(left) if left is not None else None)
        except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError):
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def compact_plays(plays: list[dict]) -> pd.DataFrame:
    """A day of CBBD plays -> :data:`PLAY_COLUMNS` (one row per play; ``side`` 0 home / 1 away /
    -1 neither; the ten on the floor split by team as CBBD ids and names)."""
    home_team = {}
    for p in plays:
        if p.get("isHomeTeam") is True:
            home_team.setdefault(p["gameId"], p.get("team"))
        elif p.get("isHomeTeam") is False:
            home_team.setdefault(p["gameId"], p.get("opponent"))
    rows = []
    for p in plays:
        ht = home_team.get(p["gameId"])
        on = p.get("onFloor") or []
        home = [x for x in on if x.get("team") == ht]
        away = [x for x in on if x.get("team") != ht]
        side = 0 if p.get("isHomeTeam") is True else 1 if p.get("isHomeTeam") is False else -1
        rows.append((p["gameId"], str(p.get("gameSourceId") or ""), p["id"], p.get("period"),
                     p.get("secondsRemaining"), p.get("playType") or "", p.get("playText") or "", side,
                     bool(p.get("scoringPlay")), bool(p.get("shootingPlay")), p.get("scoreValue") or 0,
                     [x["id"] for x in home], [x.get("name") or "" for x in home],
                     [x["id"] for x in away], [x.get("name") or "" for x in away]))
    return pd.DataFrame(rows, columns=PLAY_COLUMNS)


def season_days(pg: pd.DataFrame, season: int) -> list[str]:
    """The dates of a season's games in ESPN's box scores (``YYYY-MM-DD``), the days to fetch."""
    return sorted(pg.loc[pg.season == season, "game_date"].astype(str).str[:10].unique())


def fetch_days(days: list[str], cache_dir: Path, key: str, reserve: int = RESERVE,
               max_calls: int | None = None) -> dict:
    """Fetch every day in ``days`` not yet in ``cache_dir`` (one ``/plays/date`` call each), oldest
    first, stopping before the remaining calls drop below ``reserve`` or after ``max_calls``.
    Returns ``{"fetched": n, "cached": n, "left": remaining calls, "missing": [days not fetched]}``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [d for d in days if not (cache_dir / f"plays_{d}.parquet").exists()]
    fetched, left = 0, None
    for d in todo:
        if max_calls is not None and fetched >= max_calls:
            break
        if left is not None and left <= reserve:
            log.warning("CBBD calls left %d <= reserve %d: stopping", left, reserve)
            break
        plays, left = _get("plays/date", key, date=d)
        compact_plays(plays).to_parquet(cache_dir / f"plays_{d}.parquet", index=False)
        fetched += 1
        log.info("%s: %d plays, calls left %s", d, len(plays), left)
    missing = [d for d in days if not (cache_dir / f"plays_{d}.parquet").exists()]
    return {"fetched": fetched, "cached": len(days) - len(todo), "left": left, "missing": missing}


def load_days(cache_dir: Path, days: list[str] | None = None) -> pd.DataFrame:
    paths = sorted(cache_dir.glob("plays_*.parquet")) if days is None else \
        [cache_dir / f"plays_{d}.parquet" for d in days if (cache_dir / f"plays_{d}.parquet").exists()]
    if not paths:
        return pd.DataFrame(columns=PLAY_COLUMNS)
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


# ----------------------------------------------------------------------------------- id matching

def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", " ", s.replace("'", "").replace(".", ""))
    return " ".join(SUFFIX.sub(" ", s).split())


def _match(names: dict[str, int], name: str) -> int | None:
    n = norm_name(name)
    if n in names:
        return names[n]
    last = n.split()[-1] if n else ""
    hits = {pid for k, pid in names.items() if k.split() and k.split()[-1] == last}
    if len(hits) == 1:
        return hits.pop()
    first = n.split()[0] if n else ""
    hits = {pid for k, pid in names.items() if k.split() and k.split()[0] == first
            and last and (k.split()[-1].startswith(last[:4]) or last.startswith(k.split()[-1][:4]))}
    return hits.pop() if len(hits) == 1 else None


def espn_ids(plays: pd.DataFrame, pg: pd.DataFrame) -> tuple[dict, dict, dict]:
    """``({(game_id, side, cbbd_id): espn_id}, {(game_id, side): espn home flag}, stats)``: each CBBD
    player on the floor matched by name to the ESPN box score of the same game. A CBBD side (0 home /
    1 away) is paired with the ESPN team whose roster matches more of its names, since the two
    sources disagree on home and away for some neutral-site games."""
    box = pg[pg.game_id.isin(set(plays.game_id))]
    rosters = {}
    for (g, home), d in box.groupby(["game_id", "home"]):
        rosters[(g, bool(home))] = {norm_name(n): int(p) for n, p in zip(d.name, d.player_id)}
    seen = {}
    for col_ids, col_names, side in (("home_ids", "home_names", 0), ("away_ids", "away_names", 1)):
        for g, ids, names in zip(plays.game_id, plays[col_ids], plays[col_names]):
            for i, n in zip(ids, names):
                seen.setdefault((g, side, int(i)), n)
    by_side: dict = {}
    for (g, side, i), n in seen.items():
        by_side.setdefault((g, side), []).append((i, n))
    out, home_of, missed = {}, {}, 0
    for (g, side), players in by_side.items():
        best = None
        for home in (side == 0, side != 0):
            m = {i: _match(rosters.get((g, home), {}), n) for i, n in players}
            hits = sum(v is not None for v in m.values())
            if best is None or hits > best[0]:
                best = (hits, home, m)
        _, home, m = best
        home_of[(g, side)] = home
        for i, pid in m.items():
            if pid is None:
                missed += 1
            else:
                out[(g, side, i)] = pid
    swapped = sum(home != (side == 0) for (g, side), home in home_of.items())
    return out, home_of, {"players": len(seen), "matched": len(out), "unmatched": missed,
                          "swapped_sides": swapped}


# ----------------------------------------------------------------------------------- possessions

def classify(plays: pd.DataFrame) -> pd.DataFrame:
    """Per play: ``side``, ``kind`` (the NBA adapter's action kinds), ``made``, ``last_ft``, ``pts``."""
    from ..nba.player_values import DREB, FT, OREB, OTHER, SHOT, TO

    pt = plays.play_type.astype(str)
    text = plays.text.astype(str).str.lower()
    is_ft = pt.str.contains("FreeThrow") | text.str.contains("free throw")
    ft = text.str.extract(FT_OF)
    k, n = pd.to_numeric(ft[0], errors="coerce"), pd.to_numeric(ft[1], errors="coerce")
    tech = is_ft & (text.str.contains("technical") | k.isna())
    shot = plays.shooting.astype(bool) & ~is_ft
    to = pt.str.contains("Turnover")
    kind = np.select([is_ft & ~tech, shot, to, pt == "Offensive Rebound", pt == "Defensive Rebound"],
                     [FT, SHOT, TO, OREB, DREB], OTHER)
    made = plays.scoring.astype(bool).to_numpy()
    value = pd.to_numeric(plays.score_value, errors="coerce").fillna(0).clip(0, 3).to_numpy()
    pts = np.where((kind == SHOT) & made, value, np.where((kind == FT) & made, 1, 0))
    return pd.DataFrame({"side": plays.side.astype(np.int8).to_numpy(), "kind": kind.astype(np.int8),
                         "made": made, "last_ft": ((kind == FT) & (k == n)).to_numpy(),
                         "pts": pts.astype(np.int8)}, index=plays.index)


def season_possessions(plays: pd.DataFrame, pg: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Possessions (:data:`POSS_COLUMNS`, ESPN game and player ids) of every game in ``plays`` that is
    in ``pg``; returns them and match statistics."""
    from ..nba.player_values import game_possessions

    games = pg.drop_duplicates("game_id").set_index("game_id")
    plays = plays[plays.game_id.isin(games.index)]
    plays = plays.sort_values(["game_id", "period", "seconds", "play_id"], ascending=[True, True, False, True],
                              kind="stable").reset_index(drop=True)
    ids, home_of, stats = espn_ids(plays, pg)
    cl = classify(plays)
    period = plays.period.to_numpy()
    side, kind, made, last_ft, pts = (cl[c].to_numpy() for c in ("side", "kind", "made", "last_ft", "pts"))
    gid = plays.game_id.to_numpy()
    bounds = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1], True])
    rows = []
    for a, b in itertools.pairwise(bounds):
        g = gid[a]
        for r, s, st, p in game_possessions(period[a:b], side[a:b], kind[a:b], made[a:b], last_ft[a:b], pts[a:b]):
            i = a + r
            home = [ids.get((g, 0, int(x))) for x in plays.home_ids.iat[i]]
            away = [ids.get((g, 1, int(x))) for x in plays.away_ids.iat[i]]
            if len(home) != 5 or len(away) != 5 or None in home or None in away:
                continue
            off, dfn = (home, away) if s == 0 else (away, home)
            rows.append((g, int(period[i]), bool(home_of.get((g, int(s)), s == 0)), st, p, *off, *dfn))
    poss = pd.DataFrame(rows, columns=["game_id", "period", "off_home", "start", "points",
                                       *[f"off_{i}" for i in range(1, 6)], *[f"def_{i}" for i in range(1, 6)]])
    poss["season"] = poss.game_id.map(games.season).astype(int)
    poss["game_date"] = poss.game_id.map(games.game_date).astype(str)
    poss["t"] = pd.to_datetime(poss.game_date).sub(pd.Timestamp("1970-01-01")).dt.days.astype(np.int32)
    poss["start"] = poss.start.astype(np.int8)
    poss["points"] = poss.points.astype(np.int8)
    total = sum(len(game_possessions(period[a:b], side[a:b], kind[a:b], made[a:b], last_ft[a:b], pts[a:b]))
                for a, b in itertools.pairwise(bounds))
    stats |= {"games": int(len(bounds) - 1), "possessions": int(total), "kept": len(poss)}
    return poss[POSS_COLUMNS].reset_index(drop=True), stats


def today() -> str:
    return datetime.datetime.now(datetime.timezone.utc).date().isoformat()
