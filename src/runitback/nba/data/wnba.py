"""WNBA play-by-play as engine events (W1).

ESPN's WNBA feed (sportsdataverse ``espn_wnba_pbp``, 2006 on) has the NBA's format, "Free Throw - 1 of 2"
and all (``runitback.nba.data.preprocess``), and this module turns it into the event tables of
:data:`~runitback.nba.league.WNBA`. What differs from the NBA's reader:

- **Fouls are events.** The WNBA engine tracks the bonus (``League.bonus``), so a personal foul by the
  defense that sends nobody to the line is a ``foul`` with result ``fts_0``, and a foul whose free throws
  follow is that foul's event (the NBA's reader adds a ``foul`` only at the trip). Team fouls are personal,
  loose-ball, take, away-from-play, clear-path, flagrant and inbound fouls, and shooting fouls (counted by
  the shot they come with, an and-one or a missed shot that drew a foul). Offensive fouls (the turnover row
  is the event) and technicals are not team fouls.
- **Fouls that keep the ball.** Flagrant and clear-path free throws, and the single shot of a transition take
  foul or of an away-from-play foul, are followed by the fouled team's ball. They are a ``foul`` with no free
  throws (it still counts toward the bonus) and technical-style shots, which keep possession.
- **A foul by the team with the ball** that sends nobody to the line (a loose-ball or inbound foul) loses the
  ball: it is a turnover, as in the college reader.
- **Timeouts** are full, 20-second (``ShortTimeOut``, to 2023) and, in 2026, reset timeouts; official
  (media) timeouts have no team and are dropped.
- **Teams** are keyed by ESPN team id as the franchise's latest abbreviation (``WNBA_TEAM_IDS``: Detroit,
  Tulsa and Dallas are ``DAL``; San Antonio and Las Vegas ``LV``), as ``runitback.wager.wnba`` keys them, so
  ``pbp_32.parquet`` joins the wager model's rows. All-Star, Olympic and exhibition teams are dropped.

The rows are then turned into states and outcomes by the college reader's ``process_game`` (shared: it tracks
team fouls with the league's rules, independently of the engine) and replayed through the engine
(``replay_agreement``): each real outcome applied to the real state must give the real next state.
"""
from __future__ import annotations

import json
import logging
import math
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..engine.rules import AWAY, HOME
from ..league import WNBA, WNBA_TEAM_IDS, League
from . import college
from .download import numpy_dtypes

log = logging.getLogger(__name__)

PBP_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
           "espn_wnba_pbp/play_by_play_{season}.parquet")
COLUMNS = [
    "game_id", "game_play_number", "sequence_number", "season", "season_type", "game_date", "period_number",
    "start_quarter_seconds_remaining", "type_text", "text", "team_id", "home_team_id", "away_team_id",
    "home_score", "away_score", "score_value", "scoring_play", "shooting_play", "home_team_spread",
]
FT_OF = re.compile(r"(\d)\s*of\s*(\d)")
POSSESSION_KINDS = ("shot", "ft", "to", "foul", "jump")
TIMEOUTS = {"Full Timeout", "Timeout", "RegularTimeOut", "ShortTimeOut", "Short Timeout", "Reset Timeout"}
SHOOTING_FOULS = {"Shooting Foul", "ShootingFoul", "Shooting Block Foul"}
# team fouls that are not shooting fouls (offensive fouls and technicals are not team fouls)
PERSONAL_FOULS = {
    "Personal Foul", "PersonalFoul", "PersonalBlock", "Personal Take Foul", "PersonalTake", "Transition Take Foul",
    "Loose Ball Foul", "LooseBallFoul", "Away from Play Foul", "Away From Play Foul", "Clear Path Foul",
    "ClearPath Foul", "Flagrant Foul Type 1", "Flagrant Foul Type 2", "InBound Foul", "Inbound Foul",
}
# a single free throw after one of these keeps the ball for the fouled team
ONE_SHOT_AND_BALL = {"Transition Take Foul", "Away from Play Foul", "Away From Play Foul"}


# ---------------------------------------------------------------- download

def download_season(season: int, raw_dir: Path, force: bool = False) -> Path | None:
    """One season's play-by-play (``pbp_<season>.parquet``; seasons are calendar years). None for a season
    that isn't published."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"pbp_{season}.parquet"
    if force or not path.exists():
        try:
            df = pd.read_parquet(PBP_URL.format(season=season))
        except Exception as e:  # noqa: BLE001 - a season that hasn't started has no file
            log.warning("could not download the WNBA play-by-play of %d (%s)", season, e)
            return None
        df.to_parquet(path)
    return path


def load_pbp(season: int, raw_dir: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq
    path = raw_dir / f"pbp_{season}.parquet"
    available = set(pq.read_schema(path).names)
    return numpy_dtypes(pd.read_parquet(path, columns=[c for c in COLUMNS if c in available]))


def team_key(team_id) -> str | None:
    """The franchise key of an ESPN team id; None for All-Star, Olympic and exhibition teams."""
    if _nan(team_id):
        return None
    return WNBA_TEAM_IDS.get(int(team_id))


# ---------------------------------------------------------------- one game

def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _actions(g: pd.DataFrame) -> list[dict]:
    """Rows that matter, in game order, tagged with what kind of action each is."""
    home_id, away_id = g.home_team_id.iloc[0], g.away_team_id.iloc[0]
    order = "game_play_number" if "game_play_number" in g else "sequence_number"
    g = g.assign(clock_s=np.ceil(g.start_quarter_seconds_remaining.astype(float)))
    g = g.sort_values(["period_number", "clock_s", order], ascending=[True, False, True], kind="stable")
    out = []
    for r in g.itertuples(index=False):
        if _nan(r.period_number) or _nan(r.clock_s):
            continue
        tt = str(r.type_text or "").replace("_", " ")
        text = str(r.text or "").lower()
        side = HOME if r.team_id == home_id else AWAY if r.team_id == away_id else None
        a = None
        if tt.startswith("Free Throw"):
            m = FT_OF.search(tt)
            made = bool(r.scoring_play) or "makes" in text
            if "Technical" in tt or m is None:
                a = dict(kind="tech", made=made)
            else:
                a = dict(kind="ft", k=int(m.group(1)), n=int(m.group(2)), made=made,
                         keep="Flagrant" in tt or "Clear Path" in tt)
        elif "Turnover" in tt and tt != "No Turnover":
            a = dict(kind="to")
        elif tt in TIMEOUTS:
            a = dict(kind="timeout")
        elif tt in ("Jumpball", "Jump Ball"):
            a = dict(kind="jump")
        elif tt in ("Offensive Rebound", "Defensive Rebound"):
            a = dict(kind="reb", off=tt.startswith("Offensive"))
        elif tt in PERSONAL_FOULS:
            a = dict(kind="pf", foul=tt)
        elif tt in SHOOTING_FOULS:
            a = dict(kind="sf")
        elif bool(r.shooting_play) and "Foul" not in tt:
            three = "three point" in text or (not _nan(r.score_value) and int(r.score_value) == 3)
            made = bool(r.scoring_play) and not _nan(r.score_value) and int(r.score_value) > 0
            a = dict(kind="shot", pts=3 if three else 2, made=made)
        if a is None or side is None:
            continue
        a.update(side=side, qtr=int(r.period_number), clock=int(r.clock_s),
                 post=(int(r.home_score), int(r.away_score)))
        out.append(a)
    return out


def _same_moment(a: dict, b: dict) -> bool:
    return a["qtr"] == b["qtr"] and a["clock"] == b["clock"]


def _events(acts: list[dict], lg: League = WNBA) -> list[dict]:
    """Merge the rows into events in the college reader's format: shots (with the free throws that make them
    fouled), fouls (with the trip they sent a team to the line for, or none), technical-style free throws
    (technicals, and the shots of a foul that keeps the ball), turnovers, timeouts, jump balls, and a tip-off
    at the start of the game and of each overtime and a run-out at the end of each period."""
    events: list[dict] = []
    trip_kind: dict[tuple, str] = {}  # (side, qtr, clock) -> "ft" or "tech": what the trip's later shots are
    for i, a in enumerate(acts):
        k = a["kind"]
        if k == "reb":
            if events:
                events[-1].setdefault("rebounds", []).append(a)
            continue
        if k == "pf":
            # a personal foul and a turnover by one team at one clock: an offensive foul (not a team foul)
            if any(b["kind"] == "to" and b["side"] == a["side"] and _same_moment(a, b) for b in acts[i + 1:i + 6]):
                continue
            events.append(dict(kind="foul", side=1 - a["side"], fouler=a["side"], n=0, foul=a["foul"],
                               qtr=a["qtr"], clock=a["clock"], post=a["post"]))
            continue
        if k == "sf":
            events.append(dict(kind="sfoul", fouler=a["side"], qtr=a["qtr"], clock=a["clock"]))
            continue
        if k == "ft":
            side, key = a["side"], (a["side"], a["qtr"], a["clock"])
            if a["k"] == 1:
                back = [e for e in reversed(events) if _same_moment(e, a)]
                shot = next((e for e in back if e["kind"] == "shot" and e["side"] == side and "fouled" not in e), None)
                sfoul = next((e for e in back if e["kind"] == "sfoul" and e["fouler"] == 1 - side), None)
                foul = next((e for e in back if e["kind"] == "foul" and e["fouler"] == 1 - side and e["n"] == 0
                             and not e.get("used")), None)
                fits = shot is not None and ((shot["made"] and a["n"] == 1) or (not shot["made"] and a["n"] == shot["pts"]))
                keep = a["keep"] or (foul is not None and foul["foul"] in ONE_SHOT_AND_BALL and a["n"] == 1)
                if fits and not keep and (sfoul is not None or foul is None):
                    shot["fouled"] = a["n"]
                    trip_kind[key] = "ft"
                else:
                    if foul is None:
                        foul = dict(kind="foul", side=side, fouler=1 - side, n=0, foul="", qtr=a["qtr"],
                                    clock=a["clock"], extra=True)
                        events.append(foul)
                    foul["used"] = True
                    if not keep:
                        foul["n"] = a["n"]
                        foul["first_made"] = a["made"]
                    trip_kind[key] = "tech" if keep else "ft"
            if trip_kind.get(key) == "tech":
                events.append(dict(kind="tech", side=side, made=a["made"], qtr=a["qtr"], clock=a["clock"],
                                   post=a["post"]))
            else:
                events.append(dict(a))
            continue
        events.append(a)
    events = [e for e in events if e["kind"] != "sfoul"]
    # a jump ball's row carries the first-named jumper's team, not the team that gained possession: that is
    # the team that acts next in the period
    for i, e in enumerate(events):
        if e["kind"] == "jump":
            nxt = next((b for b in events[i + 1:] if b["qtr"] == e["qtr"] and b["kind"] in POSSESSION_KINDS
                        and b["kind"] != "jump"), None)
            if nxt is not None:
                e["side"] = nxt["side"]
    out: list[dict] = []
    for q in sorted({e["qtr"] for e in events}):
        ev = [e for e in events if e["qtr"] == q]
        if q == 1 or lg.is_overtime(q):
            first_jump = next((i for i, e in enumerate(ev) if e["kind"] == "jump"), None)
            first_poss = next((i for i, e in enumerate(ev) if e["kind"] in POSSESSION_KINDS), None)
            if first_jump is not None and first_jump == first_poss:
                ev[first_jump]["tip"] = True
                ev = [ev[first_jump]] + [e for i, e in enumerate(ev) if i != first_jump and
                                         not (e["kind"] == "timeout" and i < first_jump)]
            elif first_poss is not None:
                tip = dict(kind="jump", tip=True, side=ev[first_poss]["side"], qtr=q, clock=lg.period_length(q))
                ev = [tip] + [e for i, e in enumerate(ev) if e["kind"] != "timeout" or i > first_poss]
        if ev and ev[-1]["clock"] > 0:
            ev.append(dict(kind="run_out", side=None, qtr=q, clock=0))
        out += ev
    return out


def process_game(g: pd.DataFrame, lg: League = WNBA) -> tuple[dict, list[dict]] | None:
    """One game's meta and event rows (the college reader's format, with team fouls), or None for a game of
    an All-Star or exhibition team, a preseason game or too few events."""
    first = g.iloc[0]
    home, away = team_key(first.home_team_id), team_key(first.away_team_id)
    if home is None or away is None or home == away:
        return None
    return college.process_game(g, lg, (home, away, False), events=_events(_actions(g), lg))


replay_agreement = college.replay_agreement


# ---------------------------------------------------------------- a season, all seasons

def _season(args) -> tuple[list[dict], list[dict], dict, int, int]:
    season, raw_dir, min_agreement = args
    lg = WNBA
    df = load_pbp(season, Path(raw_dir))
    metas, rows_out, misses = [], [], {}
    ok_all = tot_all = n_games = 0
    for _, g in df.groupby("game_id", sort=False):
        n_games += 1
        res = process_game(g, lg)
        if res is None:
            continue
        meta, rows = res
        ok, total, m = replay_agreement(meta, rows, lg)
        for k, v in m.items():
            misses[k] = misses.get(k, 0) + v
        ok_all += ok
        tot_all += total
        meta["agreement"] = ok / max(total, 1)
        if meta["agreement"] < min_agreement:
            continue
        meta["n_events"] = len(rows)
        for i, r in enumerate(rows):
            r["idx"] = i
        metas.append(meta)
        rows_out.extend(rows)
    log.info("wnba season %d: kept %d/%d games, replay agreement %.4f", season, len(metas), n_games,
             ok_all / max(tot_all, 1))
    return metas, rows_out, misses, ok_all, tot_all


def preprocess(seasons: list[int], raw_dir: Path, out_dir: Path, min_agreement: float = 0.9,
               jobs: int = 1) -> dict:
    """Process `seasons` to games / plays / ratings parquet files in `out_dir`, and return the replay
    statistics (``replay.json`` has the same): agreement per season and the commonest mismatches."""
    lg = WNBA
    out_dir.mkdir(parents=True, exist_ok=True)
    present = [s for s in seasons if (raw_dir / f"pbp_{s}.parquet").exists()]
    for s in sorted(set(seasons) - set(present)):
        log.warning("season %d: %s not found (download it first); skipped", s, raw_dir / f"pbp_{s}.parquet")
    args = [(s, str(raw_dir), min_agreement) for s in present]
    if jobs > 1 and len(args) > 1:
        with ProcessPoolExecutor(min(jobs, len(args))) as pool:
            results = list(pool.map(_season, args))
    else:
        results = [_season(a) for a in args]
    games = pd.DataFrame([m for r in results for m in r[0]])
    plays = pd.DataFrame([row for r in results for row in r[1]])
    misses: dict[str, int] = {}
    for r in results:
        for k, v in r[2].items():
            misses[k] = misses.get(k, 0) + v
    games.to_parquet(out_dir / "games.parquet")
    plays.to_parquet(out_dir / "plays.parquet")
    from .ratings import build_ratings
    build_ratings(games, out_dir, lg)
    stats = college.replay_stats(lg, present, results, games, misses)
    (out_dir / "replay.json").write_text(json.dumps(stats, indent=1))
    log.info("replay agreement %.4f over %d transitions", stats["agreement"], stats["transitions"])
    return stats
