"""Per-game team efficiency from nflverse play-by-play.

Box scores count yards and points; play-by-play says how much each play
moved the expected score (EPA) and whether it kept the offense on schedule
(``success``: EPA > 0). Per game, EPA per play and success rate predict a
team's future margins better than its points or yards do, so they are
summarized here for each offense (``pbp_*``); the opponent's columns
(``opp_pbp_*``) are the same offense numbers allowed by this team's defense.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PBP_COLUMNS = [
    "pbp_plays",  # offensive snaps (dropbacks and designed runs)
    "pbp_epa_play",  # EPA per offensive play
    "pbp_success",  # share of plays with EPA > 0
    "pbp_pass_epa",  # EPA per dropback (sacks and scrambles included)
    "pbp_rush_epa",  # EPA per designed run
    "pbp_early_epa",  # EPA per 1st/2nd-down play
    "pbp_early_pass_rate",  # dropback share on 1st/2nd down
    "pbp_neutral_epa",  # EPA per play with win probability 20-80%
    "pbp_explosive",  # share of plays gaining 20+ (pass) or 10+ (run)
    "pbp_turnover_epa",  # EPA lost on interceptions and lost fumbles
    "pbp_st_epa",  # special-teams EPA (kicks, punts, returns) per game
    "pbp_drive_score",  # share of drives ending in a TD or FG
]

# columns read from the play-by-play files
READ = [
    "game_id", "season", "week", "posteam", "defteam", "play_type", "epa",
    "success", "down", "qb_dropback", "rush", "yards_gained", "wp",
    "interception", "fumble_lost", "special_teams_play", "drive",
    "fixed_drive_result", "pass", "qb_scramble",
]  # fmt: skip


def _ratio(num, den) -> pd.Series:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (num / den.where(den > 0)).astype(float)


def game_efficiency(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team) of a season's play-by-play with the
    :data:`PBP_COLUMNS` of that team's offense."""
    pbp = pbp[[c for c in READ if c in pbp]].copy()
    pbp = pbp[pbp["posteam"].notna() & pbp["posteam"].ne("")]
    keys = ["season", "week", "posteam"]

    scrim = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()].copy()
    dropback = scrim["qb_dropback"].fillna(0) == 1
    run = ~dropback
    early = scrim["down"].isin([1, 2])
    neutral = scrim["wp"].between(0.2, 0.8)
    explosive = np.where(dropback, scrim["yards_gained"] >= 20,
                         scrim["yards_gained"] >= 10)  # fmt: skip
    turnover = (scrim["interception"].fillna(0) == 1) | (
        scrim["fumble_lost"].fillna(0) == 1
    )
    scrim = scrim.assign(
        _one=1.0,
        _db=dropback.astype(float),
        _db_epa=scrim["epa"].where(dropback, 0.0),
        _run=run.astype(float),
        _run_epa=scrim["epa"].where(run, 0.0),
        _early=early.astype(float),
        _early_epa=scrim["epa"].where(early, 0.0),
        _early_db=(early & dropback).astype(float),
        _neutral=neutral.astype(float),
        _neutral_epa=scrim["epa"].where(neutral, 0.0),
        _explosive=explosive.astype(float),
        _to_epa=scrim["epa"].where(turnover, 0.0),
        _success=scrim["success"].fillna(0.0),
    )
    g = scrim.groupby(keys)
    s = g[[c for c in scrim if c.startswith("_")] + ["epa"]].sum()
    out = pd.DataFrame(index=s.index)
    out["pbp_plays"] = s["_one"]
    out["pbp_epa_play"] = _ratio(s["epa"], s["_one"])
    out["pbp_success"] = _ratio(s["_success"], s["_one"])
    out["pbp_pass_epa"] = _ratio(s["_db_epa"], s["_db"])
    out["pbp_rush_epa"] = _ratio(s["_run_epa"], s["_run"])
    out["pbp_early_epa"] = _ratio(s["_early_epa"], s["_early"])
    out["pbp_early_pass_rate"] = _ratio(s["_early_db"], s["_early"])
    out["pbp_neutral_epa"] = _ratio(s["_neutral_epa"], s["_neutral"])
    out["pbp_explosive"] = _ratio(s["_explosive"], s["_one"])
    out["pbp_turnover_epa"] = s["_to_epa"]

    st = pbp[pbp["special_teams_play"].fillna(0) == 1]
    out["pbp_st_epa"] = st.groupby(keys)["epa"].sum().reindex(out.index).fillna(0.0)

    drives = pbp.dropna(subset=["drive"]).drop_duplicates([*keys, "drive"])
    scored = drives["fixed_drive_result"].isin(["Touchdown", "Field goal"])
    drive_rate = scored.groupby([drives[k] for k in keys]).mean()
    out["pbp_drive_score"] = drive_rate.reindex(out.index)

    return out.reset_index().rename(columns={"posteam": "team"})


def load_efficiency(seasons: list[int]) -> pd.DataFrame:
    """:func:`game_efficiency` of every season in ``seasons`` (play-by-play
    starts in 1999), one season's file at a time."""
    import nflreadpy

    frames = []
    for season in seasons:
        if season < 1999:
            continue
        try:
            pbp = nflreadpy.load_pbp([season]).to_pandas()
        except Exception as error:  # a season not published yet
            print(f"no play-by-play for {season}: {error}")
            continue
        frames.append(game_efficiency(pbp))
    if not frames:
        return pd.DataFrame(columns=["season", "week", "team", *PBP_COLUMNS])
    return pd.concat(frames, ignore_index=True)
