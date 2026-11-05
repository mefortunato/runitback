"""Refresh a checkpoint's data without retraining.

A checkpoint carries data next to its weights: pre-game team ratings and, for lineup models, each
team's default depth chart (its most recent game), the player directory (career games, usage
multipliers, recent-usage features), head-coach records and role snap shares. All of it is frozen at
training time. `refresh_checkpoint` rebuilds it from the latest nflverse data and writes a new
checkpoint. The model's weights and vocabularies, and what was fit to training data (personnel package
frequencies, role on-field rates), stay as trained. Players and coaches the model has never seen
use its NEW / OTHER tokens, as they would in any lineup.
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd
import torch

from .data.ratings import RatingsTable, build_ratings
from .release import sha256

log = logging.getLogger(__name__)


def refresh_checkpoint(ckpt: Path, out: Path, raw_dir: Path, processed_dir: Path, start: int, end: int,
                       download: bool = True) -> dict:
    """Rebuild ratings and lineup data for seasons start..end and write `out`. Returns a summary."""
    seasons = list(range(start, end + 1))
    if download:
        from .data.download import download_season
        from .data.players import download_players
        for s in seasons:
            # the latest season is still being played: always re-download it
            download_season(s, raw_dir, force=s == end)
        download_players(seasons, raw_dir)
    missing = [s for s in seasons if not (raw_dir / f"pbp_{s}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"no play-by-play for seasons {missing} in {raw_dir} (run with download)")

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    summary: dict = {"checkpoint": str(ckpt), "out": str(out)}
    before = _data_marks(ck)

    ratings = build_ratings(seasons, raw_dir, processed_dir)
    ck["ratings"] = RatingsTable(ratings).to_records()
    last = ratings.sort_values(["season", "week"]).iloc[-1]
    summary["ratings_through"] = f"{int(last.season)} week {int(last.week)}"

    lm = ck.get("lineup_meta")
    if lm:
        from .data.players import build_players
        from .model.lineups import CoachVocab, LineupData, PlayerVocab, inference_meta
        build_players(seasons, raw_dir, processed_dir)
        ld = LineupData.load(processed_dir, [], PlayerVocab(lm["player_vocab"]), CoachVocab(lm["coach_vocab"]))
        if ld is None:
            raise FileNotFoundError(f"no player data in {processed_dir}")
        fresh = inference_meta(ld, pd.read_parquet(processed_dir / "coach_games.parquet"))
        # vocabularies must match the weights; fitted tables stay as trained
        for k in ("player_vocab", "coach_vocab"):
            assert fresh[k] == lm[k], f"{k} changed while refreshing"
        keep = {k: lm[k] for k in ("personnel_sampler", "role_inclusion") if k in lm}
        ck["lineup_meta"] = fresh | keep
        latest = max((d["season"], d["week"]) for d in fresh["lineups"].values())
        summary["lineups_through"] = f"{latest[0]} week {latest[1]}"
        summary["teams_behind_latest_week"] = sorted(
            t for t, d in fresh["lineups"].items() if (d["season"], d["week"]) < latest)
        summary["players"] = len(fresh["players"])

    ck["refreshed"] = {"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                       "from_checkpoint": str(ckpt), "seasons": [start, end]} | {
        k: summary[k] for k in ("ratings_through", "lineups_through") if k in summary}
    after = _data_marks(ck)
    summary["previous"] = {k: f"{v[0]} week {v[1]}" for k, v in before.items() if v and k != "games"}
    # anything new since the checkpoint's data? (the weekly job skips publishing when not)
    summary["changed"] = any(after.get(k) != before.get(k) for k in after)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    torch.save(ck, tmp)
    tmp.replace(out)
    summary["sha256"] = sha256(out)
    summary["bytes"] = out.stat().st_size
    log.info("refreshed %s -> %s: %s", ckpt, out, summary)
    return summary


def _data_marks(ck: dict) -> dict:
    """Latest (season, week) in the checkpoint's ratings and default lineups."""
    marks: dict = {"ratings": None, "lineups": None, "games": 0}
    if ck.get("ratings"):
        marks["ratings"] = max((int(r["season"]), int(r["week"])) for r in ck["ratings"])
        marks["games"] = len({r["game_id"] for r in ck["ratings"]})  # catches late games in the same week
    lm = ck.get("lineup_meta") or {}
    if lm.get("lineups"):
        marks["lineups"] = max((int(d["season"]), int(d["week"])) for d in lm["lineups"].values())
    return marks
