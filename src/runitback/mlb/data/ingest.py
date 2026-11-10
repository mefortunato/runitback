"""Season tables from MLB Stats API live feeds.

``build_seasons`` downloads (or reads from the cache) every finished game's feed of each season,
parses it (:mod:`.feed`), adds RE24 run values fit on the previous season (:mod:`.run_values`) and
writes, per season S, to ``out_dir``:

- ``pa_<S>.parquet``: one row per plate appearance (batter, pitcher, handedness, lineup spot,
  base / out state at the start, just before the result and after it, result, runs, run value,
  the defense's fielders 2-9);
- ``games_<S>.parquet``: one row per game (date, venue, teams, final score, the feed's probable
  and the actual starting pitchers);
- ``lineups_<S>.parquet``: everyone who played, per team-game (batting order code and spot,
  starter flag, starting and all positions, pitching order);
- ``substitutions_<S>.parquet``: every substitution in order (pinch hitters and runners,
  defensive substitutions and switches, pitching changes), with the play it happened in;
- ``players_<S>.parquet``: name, bats, throws and primary position of everyone in the feeds;
- ``re_<S>.parquet``: the run-expectancy matrix used for season S's run values;
- ``checks_<S>.json``: replay checks summed over the season (see :mod:`.feed`).

Only finished games are kept (an in-progress season has its finished games so far); upcoming
games' probable pitchers and posted lineups come from :func:`.statsapi.probables`.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import statsapi
from .feed import parse_feed
from .run_values import add_run_values, run_expectancy

log = logging.getLogger(__name__)

TABLES = ("pa", "games", "lineups", "substitutions", "players")


def parse_season(feeds: Iterable[dict]) -> tuple[dict[str, pd.DataFrame], dict]:
    """Parsed feeds -> (season tables, summed checks). Plays that are not plate appearances
    (a caught stealing that ends an inning with the batter at the plate) are dropped from ``pa``
    after their runs have counted toward the half inning's ``runs_rest``."""
    rows = {t: [] for t in TABLES}
    checks: Counter = Counter()
    for fd in feeds:
        if not statsapi.is_final(fd):
            continue
        out = parse_feed(fd)
        rows["pa"] += [p for p in out["plays"] if p["is_pa"]]
        rows["games"].append(out["game"])
        rows["lineups"] += out["lineups"]
        rows["substitutions"] += out["substitutions"]
        rows["players"] += out["players"]
        checks.update(out["checks"])
        checks["games"] += 1
    tables = {t: pd.DataFrame(r) for t, r in rows.items()}
    if len(tables["players"]):
        tables["players"] = tables["players"].drop_duplicates("player_id", keep="last") \
            .sort_values("player_id").reset_index(drop=True)
    if len(tables["pa"]):
        tables["pa"] = tables["pa"].drop(columns=["is_pa"])
    return tables, dict(checks)


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def build_season(season: int, cache_dir: Path, out_dir: Path, client: statsapi.Client | None = None,
                 workers: int = 4) -> dict:
    """Download / parse one season and write its tables (run values are added by
    :func:`build_seasons`, which needs the season before). Returns the season's checks."""
    games = [g for g in statsapi.schedule(season, client) if g["final"]]
    log.info("%d: %d finished games", season, len(games))
    tables, checks = parse_season(statsapi.iter_feeds(games, cache_dir, client, workers))
    for name, df in tables.items():
        _write(df, Path(out_dir) / f"{name}_{season}.parquet")
    (Path(out_dir) / f"checks_{season}.json").write_text(json.dumps(checks, indent=1, sort_keys=True))
    log.info("%d: %s", season, checks)
    return checks


def add_season_run_values(season: int, out_dir: Path, lookback: int = 1) -> bool:
    """Adds run values to ``pa_<season>.parquet`` from the run expectancy of the ``lookback``
    seasons before it in ``out_dir``. Returns False (values NaN) when none of them is there."""
    out_dir = Path(out_dir)
    earlier = [out_dir / f"pa_{s}.parquet" for s in range(season - lookback, season)]
    earlier = [pd.read_parquet(p, columns=["start_bases", "start_outs", "runs_rest", "half_complete"])
               for p in earlier if p.exists()]
    path = out_dir / f"pa_{season}.parquet"
    pa = pd.read_parquet(path)
    pa = pa.drop(columns=[c for c in ("re_start", "re_pre", "re_end", "run_value", "run_value_pa")
                          if c in pa.columns])
    table = run_expectancy(pd.concat(earlier, ignore_index=True)) if earlier else None
    if table is None:
        log.warning("%d: no earlier season in %s, run values left NaN", season, out_dir)
    else:
        _write(table.reset_index(), out_dir / f"re_{season}.parquet")
    _write(add_run_values(pa, table), path)
    return table is not None


def build_seasons(seasons: list[int], cache_dir: Path, out_dir: Path,
                  client: statsapi.Client | None = None, workers: int = 4, lookback: int = 1) -> dict:
    """:func:`build_season` for each season, then run values in season order."""
    summary = {}
    for season in sorted(seasons):
        summary[season] = build_season(season, cache_dir, out_dir, client, workers)
    for season in sorted(seasons):
        summary[season]["run_values"] = add_season_run_values(season, out_dir, lookback)
    return summary


def download(seasons: list[int], cache_dir: Path, client: statsapi.Client | None = None,
             workers: int = 4) -> dict[int, int]:
    """Cache every finished game's feed of ``seasons``; returns games per season."""
    counts = {}
    for season in seasons:
        games = [g for g in statsapi.schedule(season, client) if g["final"]]
        for _ in statsapi.iter_feeds(games, cache_dir, client, workers):
            pass
        counts[season] = len(games)
        log.info("%d: %d finished games cached", season, len(games))
    return counts
