"""College football players from the play-by-play: ESPN ids for passers and rushers, the
quarterbacks' plays, and each team-game's starting quarterback.

College play-by-play has no on-field personnel, only the passer / rusher / receiver of each play.
``espn_cfb_pbp`` carries ESPN athlete ids next to the names for most plays (95% of 2005's passes,
over 99% from 2014), the same ids as ``espn_cfb_player_box``. :func:`map_ids` keeps those and fills
the rest by matching the name to the player box scores (:func:`norm_name`): the same name on the
same team in that game, then the last name alone if it is unique on the team that game, then the same
name on the same team that season (some games' box scores miss a player). "TEAM" plays (team sacks,
spikes) stay unmapped, even where ESPN gives the placeholder an athlete id.

A team-game's **starting quarterback** (:func:`team_games`) is the passer with the most of the
team's first :data:`STARTER_WINDOW` dropbacks (ties: the earliest), so a trick-play pass on the
first snap doesn't count. Dropbacks are pass attempts and sacks (cfbfastR's ``pass`` flag on a
scrimmage play); scrambles are logged as runs and can't be told apart from designed runs.
"""
from __future__ import annotations

import datetime
import logging
import re
import unicodedata

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROLES = ("passer", "rusher")
STARTER_WINDOW = 5          # a team-game's starter has the most of its first 5 dropbacks
QB_MIN_DROPBACKS = 20       # a rusher counts as a quarterback with this many dropbacks in the season
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
# 2012-13 play text sometimes runs the pass's route into the passer's name ("Ryan Griffin slant")
ROUTE = re.compile(r"\s+(slant|deep|sideline|middle|short|left|right|screen|flat|over the middle)$")
PLAY_COLUMNS = ["game_id", "season", "t", "order", "off", "def", "home", "qb", "kind", "epa", "down",
                "distance", "ytg", "score_diff", "def_fbs"]
DROPBACK, RUSH = 0, 1


def norm_name(name) -> str:
    """Lower case, accents and punctuation dropped, suffixes (Jr., III, ...) removed:
    ``"C.J. Stroud"`` and ``"CJ Stroud"`` match, as do ``"Kenneth Walker III"`` and ``"Kenneth Walker"``."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().strip()
    if s.count(",") == 1:                       # "Plitt,Drew"
        last, first = s.split(",")
        s = f"{first} {last}"
    if len(s.split()) >= 3:                     # "Ryan Griffin slant", never "Kawann Short"
        s = ROUTE.sub("", s)
    s = re.sub(r"[.'`’]", "", s)
    s = re.sub(r"[^a-z ]", " ", s)
    s = SUFFIX.sub(" ", s)
    return " ".join(s.split())


def _flag(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0) > 0


def _id(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _unique(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Rows of ``df`` (keys + athlete_id) whose keys name exactly one athlete."""
    d = df.drop_duplicates([*keys, "athlete_id"])
    n = d.groupby(keys).athlete_id.transform("size")
    return d[n == 1][[*keys, "athlete_id"]]


def map_ids(pbp: pd.DataFrame, box: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """``pbp`` with ``passer_id`` / ``rusher_id`` (ESPN athlete ids, ``<NA>`` if unknown) and
    ``passer_src`` / ``rusher_src`` (``pbp``: the play-by-play's own id; ``name``: matched by name to
    the game's box scores; ``none``). Also a summary per role: the rows with a name, the share with an
    id from each source, and how often the name match agrees with the play-by-play's id where both
    exist (the match's accuracy)."""
    pbp = pbp.copy()
    pbp["game_id"] = pbp.game_id.astype(str)
    b = box.dropna(subset=["athlete_id"]).assign(
        game_id=lambda d: d.game_id.astype(str), team_id=lambda d: _id(d.team_id),
        athlete_id=lambda d: _id(d.athlete_id), nm=lambda d: d.athlete_name.map(norm_name))
    b = b[(b.nm != "") & (b.nm != "team")]
    b["last"] = b.nm.str.split().str[-1]
    full = _unique(b, ["game_id", "team_id", "nm"])
    last = _unique(b, ["game_id", "team_id", "last"])
    season = _unique(b.assign(season=b.season.astype(int)), ["season", "team_id", "nm"])
    summary = {}
    for role in ROLES:
        name = pbp[f"{role}_player_name"]
        nm = name.map(norm_name)
        key = pd.DataFrame({"game_id": pbp.game_id.to_numpy(), "team_id": _id(pbp.pos_team_id).to_numpy(),
                            "season": pd.to_numeric(pbp.season).astype(int).to_numpy(),
                            "nm": nm.to_numpy(), "last": nm.str.split().str[-1].fillna("").to_numpy()})
        by_full = key.merge(full, on=["game_id", "team_id", "nm"], how="left").athlete_id
        by_last = key.merge(last, on=["game_id", "team_id", "last"], how="left").athlete_id
        by_season = key.merge(season, on=["season", "team_id", "nm"], how="left").athlete_id
        matched = by_full.fillna(by_last).fillna(by_season).to_numpy()
        own = _id(pbp[f"{role}_player_id"]).to_numpy()
        team = (nm == "team").to_numpy()                # ESPN's "TEAM" placeholder has an athlete id
        own = np.where(team, pd.NA, own)
        has_name = ((nm != "") & (nm != "team")).to_numpy()
        has_own = ~pd.isna(own)
        has_match = ~pd.isna(matched) & has_name
        pbp[f"{role}_id"] = pd.array(np.where(has_own, own, np.where(has_match, matched, pd.NA)), dtype="Int64")
        pbp[f"{role}_src"] = np.where(has_own, "pbp", np.where(has_match, "name", "none"))
        both = has_own & has_match
        n = int(has_name.sum())
        summary[role] = {
            "rows_with_name": n,
            "pbp_id": float((has_own & has_name).sum() / max(n, 1)),
            "name_match": float((~has_own & has_match).sum() / max(n, 1)),
            "unmapped": float((has_name & ~has_own & ~has_match).sum() / max(n, 1)),
            "name_match_agrees": float((own[both] == matched[both]).mean()) if both.any() else float("nan"),
        }
    return pbp, summary


def game_days(schedule: pd.DataFrame) -> pd.DataFrame:
    """``game_id, t, neutral`` per scheduled game: ``t`` is the kickoff's US day (UTC minus 6 hours)
    as days since 1970-01-01, the fits' time key."""
    start = pd.to_datetime(schedule.start_date, utc=True, errors="coerce", format="ISO8601")
    day = (start - pd.Timedelta(hours=6)).dt.tz_localize(None).dt.normalize()
    t = (day - pd.Timestamp("1970-01-01")).dt.days
    neutral = schedule.neutral_site if "neutral_site" in schedule else pd.Series(False, index=schedule.index)
    out = pd.DataFrame({"game_id": schedule.game_id.astype(str), "t": t,
                        "neutral": neutral.astype("boolean").fillna(False).astype(bool)})
    return out.dropna(subset=["t"]).astype({"t": int}).drop_duplicates("game_id")


def day_number(date) -> int:
    """Days since 1970-01-01 of a ``YYYY-MM-DD`` date."""
    return (datetime.date.fromisoformat(str(date)[:10]) - datetime.date(1970, 1, 1)).days


def fbs_teams(schedule: pd.DataFrame) -> set[tuple[int, int]]:
    """``{(season, ESPN team id)}`` of the FBS teams in a schedule."""
    out = set()
    for side in ("home", "away"):
        d = schedule[schedule[f"{side}_division"] == "fbs"]
        ids = pd.to_numeric(d[f"{side}_id"], errors="coerce")
        out |= {(int(s), int(i)) for s, i in zip(d.season, ids) if pd.notna(i)}
    return out


def qb_plays(pbp: pd.DataFrame, days: pd.DataFrame, rushes: bool = True,
             fbs: set[tuple[int, int]] | None = None) -> pd.DataFrame:
    """Every quarterback play of ``pbp`` (after :func:`map_ids`), in game order: dropbacks (``kind``
    :data:`DROPBACK`; the passer's) and, with ``rushes``, runs by a quarterback (:data:`RUSH`; a
    rusher with at least :data:`QB_MIN_DROPBACKS` dropbacks that season; kneel-downs excluded).

    Columns :data:`PLAY_COLUMNS`: ``t`` (the game's day, from ``days``), ``order`` (play order in
    the game), ``off`` / ``def`` (ESPN team ids), ``home`` (1 home offense, -1 away, 0 neutral site),
    ``qb``, ``epa``, ``down``, ``distance``, ``ytg`` (yards to the end zone) and ``score_diff`` (the
    offense's lead), ``def_fbs`` (1 if the defense is an FBS team that season, per ``fbs``
    (:func:`fbs_teams`); all 1 without it). Plays of games without a schedule date or an EPA are
    dropped."""
    scrim = _flag(pbp.scrimmage_play)
    order = pd.to_numeric(pbp.game_play_number, errors="coerce")
    if "id" in pbp:
        order = order.fillna(pd.to_numeric(pbp["id"], errors="coerce"))
    drop = scrim & _flag(pbp["pass"]) & pbp.passer_id.notna()
    base = pbp.assign(order=order)
    parts = [base[drop].assign(qb=base.passer_id[drop], kind=DROPBACK)]
    if rushes:
        db = parts[0]
        qbs = db.groupby(["season", "qb"]).size()
        qbs = set(qbs.index[qbs >= QB_MIN_DROPBACKS])
        kneel = _flag(pbp.kneel_down) if "kneel_down" in pbp else pd.Series(False, index=pbp.index)
        run = scrim & _flag(pbp.rush) & ~kneel & pbp.rusher_id.notna()
        r = base[run]
        r = r[[(s, q) in qbs for s, q in zip(r.season, r.rusher_id)]]
        parts.append(r.assign(qb=r.rusher_id, kind=RUSH))
    p = pd.concat(parts)
    p = p.assign(off=_id(p.pos_team_id), **{"def": _id(p.def_pos_team_id)}).merge(days, on="game_id", how="inner")
    home = np.where((p.off == _id(p.homeTeamId)).fillna(False).to_numpy(dtype=bool), 1, -1)
    p["home"] = np.where(p.neutral.to_numpy(), 0, home).astype(np.int8)
    p["epa"] = pd.to_numeric(p.EPA, errors="coerce")
    p["down"] = pd.to_numeric(p.down, errors="coerce").fillna(1).clip(1, 4).astype(np.int8)
    p["distance"] = pd.to_numeric(p.distance, errors="coerce").fillna(10).clip(0, 99)
    p["ytg"] = pd.to_numeric(p["start.yardsToEndzone"], errors="coerce").fillna(75).clip(0, 100)
    p["score_diff"] = pd.to_numeric(p.pos_score_diff_start, errors="coerce").fillna(0)
    p = p.dropna(subset=["epa", "off", "def", "qb"])
    p = p.astype({"season": int, "qb": "int64", "off": "int64", "def": "int64", "kind": np.int8})
    p["def_fbs"] = (np.ones(len(p), dtype=np.int8) if fbs is None else
                    np.array([(s, d) in fbs for s, d in zip(p.season, p["def"])], dtype=np.int8))
    p = p.sort_values(["t", "game_id", "order", "kind"], kind="stable").reset_index(drop=True)
    return p[PLAY_COLUMNS]


def team_games(plays: pd.DataFrame) -> pd.DataFrame:
    """Each team-game's quarterbacks: ``game_id, season, t, team_id, starter, leader, dropbacks,
    starter_dropbacks`` (``starter``: most of the team's first :data:`STARTER_WINDOW` dropbacks, ties
    to the earliest; ``leader``: most dropbacks in the game; ``dropbacks``: the team's)."""
    db = plays[plays.kind == DROPBACK]
    db = db.assign(i=db.groupby(["game_id", "off"]).cumcount())
    first = db[db.i < STARTER_WINDOW]
    c = first.groupby(["game_id", "off", "qb"]).agg(n=("i", "size"), at=("i", "min")).reset_index()
    starter = c.sort_values(["game_id", "off", "n", "at"], ascending=[True, True, False, True]).drop_duplicates(
        ["game_id", "off"])[["game_id", "off", "qb"]].rename(columns={"qb": "starter"})
    per = db.groupby(["game_id", "off", "qb"]).agg(n=("i", "size"), at=("i", "min")).reset_index()
    leader = per.sort_values(["game_id", "off", "n", "at"], ascending=[True, True, False, True]).drop_duplicates(
        ["game_id", "off"])[["game_id", "off", "qb"]].rename(columns={"qb": "leader"})
    tg = db.groupby(["game_id", "off"]).agg(season=("season", "first"), t=("t", "first"),
                                             dropbacks=("i", "size")).reset_index()
    tg = tg.merge(starter, on=["game_id", "off"]).merge(leader, on=["game_id", "off"])
    sd = per.rename(columns={"qb": "starter", "n": "starter_dropbacks"})[["game_id", "off", "starter",
                                                                          "starter_dropbacks"]]
    tg = tg.merge(sd, on=["game_id", "off", "starter"], how="left").rename(columns={"off": "team_id"})
    return tg.sort_values(["t", "game_id", "team_id"], kind="stable").reset_index(drop=True)


def build_players(seasons: list[int], raw_dir, processed_dir, rushes: bool = True) -> dict:
    """``qb_plays.parquet``, ``team_games.parquet`` and ``players.parquet`` (processed_dir) from the
    seasons' raw play-by-play, player box scores and schedules; returns the id-mapping summary per
    season and role."""
    from pathlib import Path

    from .download import load

    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    schedule = load("schedule", seasons, raw_dir)
    days, fbs = game_days(schedule), fbs_teams(schedule)
    plays, summary = [], {}
    for s in seasons:
        pbp, box = load("pbp", [s], raw_dir), load("player_box", [s], raw_dir)
        if pbp.empty or box.empty:
            continue
        pbp, summary[s] = map_ids(pbp, box)
        p = qb_plays(pbp, days, rushes=rushes, fbs=fbs)
        summary[s]["qb_plays"] = len(p)
        summary[s]["dropbacks"] = int((p.kind == DROPBACK).sum())
        log.info("season %d: %s", s, summary[s])
        plays.append(p)
    plays = pd.concat(plays, ignore_index=True)
    plays.to_parquet(processed_dir / "qb_plays.parquet", index=False)
    team_games(plays).to_parquet(processed_dir / "team_games.parquet", index=False)
    from ..player_values import players_table

    players_table(load("player_box", seasons, raw_dir)).to_parquet(processed_dir / "players.parquet", index=False)
    return summary
