"""Turn ESPN college football play-by-play (``espn_cfb_pbp``) into (state, outcome) sequences for the
college rules engine and the play transformer.

The output has the NFL's schema (``runitback.nfl.data.preprocess``: ``games.parquet``,
``plays.parquet``, ``ratings.parquet``), so the NFL model, its dataset encoding and the fold code
run on it unchanged. As there, outcomes are derived from the *next* play's state where possible (net
field position, possession, clock), so replaying a real game through the rules engine reproduces
it, and a game whose replay agrees on less than ``min_agreement`` of its transitions is dropped.

What the ESPN feed needs on top of the NFL's preprocess:

- **Tries.** Most seasons fold the try into the touchdown's row (its end score includes it, and
  ``pointAfterAttempt`` says which kind it was; before 2014 only the text does). A try row is
  inserted after each such touchdown, from the 3-yard line. Seasons and games that have their own
  try rows ("Extra Point Good", "Two Point Pass", ...) keep them.
- **Kickoffs** are listed under the receiving team; the kicking team is ``kicking_team``.
- **The game clock** of many games only changes every few plays (sometimes once a drive): in the
  median game of 2018-2023, three plays in four show no time elapsed. A game where more than
  ``COARSE_CLOCK`` of its plays run no clock is flagged ``clock_known = False``: its quarter clock
  is interpolated between the times it does show (so the state features stay roughly right) and
  its plays carry ``elapsed_known = False``, which the dataset encoding turns into an elapsed
  target with no loss. Every other head trains on those plays as usual.
- **Overtime** is untimed (``qsec`` 0, ``elapsed`` 0) and the feed's overtime rows are often
  incomplete (sometimes only the scoring plays). A game keeps its overtime plays only if every
  overtime transition replays; otherwise its sequence stops at the end of regulation. ``ot_first``
  and ``ot_half`` record who had the period's first possession and which possession a play is in.
- **Repeats.** Some games list a stretch of plays twice under new play numbers: a play identical to
  an earlier one (text, period, down, distance, spot and team) is dropped.
- **Openings.** Many 2004-2013 games start after the opening kickoff; their sequence starts with the
  first snap (the opening kicker is the team without the ball).
- **Teams** are keyed by ESPN team id (``vocab``); ``home_name`` / ``away_name`` keep the schools.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..engine.rules import AWAY, HOME, CFBState, Outcome, apply_outcome
from ..vocab import KICKOFF, MAX_ELAPSED, OT_SPOT, PAT, PAT_SPOT, SCRIMMAGE, TIMEOUTS_PER_HALF, TIMEOUTS_PER_OT
from .download import load_raw

log = logging.getLogger(__name__)

# a game whose share of consecutive same-quarter plays with no clock change exceeds this has a
# coarse clock (per-play clocks run ~5-15%, coarse ones ~75-80%)
COARSE_CLOCK = 0.3
POST_WEEK = 20  # bowl and playoff games sort after the regular season: week = POST_WEEK + ESPN's week

KICKOFF_TYPES = {"Kickoff", "Kickoff Return (Offense)", "Kickoff Return Touchdown", "Kickoff Team Fumble Recovery",
                 "Kickoff Team Fumble Recovery Touchdown", "Kickoff (Safety)"}
PUNT_TYPES = {"Punt", "Punt Return", "Blocked Punt", "Punt Return Touchdown", "Blocked Punt Touchdown",
              "Punt Team Fumble Recovery", "Punt Team Fumble Recovery Touchdown", "Punt (Safety)"}
FG_TYPES = {"Field Goal Good", "Field Goal Missed", "Blocked Field Goal", "Blocked Field Goal Touchdown",
            "Missed Field Goal Return", "Missed Field Goal Return Touchdown"}
XP_TYPES = {"Extra Point Good": True, "Extra Point Missed": False}
TWO_POINT_TYPES = {"Two-Point Conversion Good": True, "Two Point Pass": True, "Two Point Rush": True,
                   "Two-Point Conversion Missed": False, "Defensive 2pt Conversion": False}
PASS_TYPES = {"Pass Reception", "Pass Incompletion", "Passing Touchdown", "Pass Completion", "Pass", "Sack",
              "Interception Return", "Interception Return Touchdown"}
RUN_TYPES = {"Rush", "Rushing Touchdown"}
FUMBLE_TYPES = {"Fumble Recovery (Own)", "Fumble Recovery (Opponent)", "Fumble Return Touchdown",
                "Fumble Recovery (Opponent) Touchdown", "Fumble Recovery (Own) Touchdown", "Fumble", "Safety"}
LOST_FUMBLE_TYPES = {"Fumble Recovery (Opponent)", "Fumble Return Touchdown", "Fumble Recovery (Opponent) Touchdown"}
COMPLETE_TYPES = {"Pass Reception", "Passing Touchdown", "Pass Completion"}

STATE_COLS = ["qtr", "qsec", "phase", "pos", "down", "ytg", "yl", "score_h", "score_a", "to_h", "to_a",
              "ot_first", "ot_half"]
OUTCOME_COLS = ["play_type", "result", "yards", "elapsed"]
COMPARE = ["qtr", "phase", "pos", "down", "ytg", "yl", "score_h", "score_a"]


def _nan(x) -> bool:
    return x is None or x is pd.NA or (isinstance(x, float) and math.isnan(x))


def _int(x, default: int = 0) -> int:
    return default if _nan(x) else int(x)


def _flag(x) -> bool:
    return not _nan(x) and bool(x)


def _scrimmage_type(tt: str, r) -> str | None:
    text = str(r.text or "").lower()
    if tt in PUNT_TYPES:
        return "punt"
    if tt in FG_TYPES:
        return "field_goal"
    if tt == "Penalty":
        return "penalty"
    if tt in PASS_TYPES or (tt in FUMBLE_TYPES and _flag(r.pass_) and not _flag(r.rush)):
        return "qb_spike" if "spike" in text else "pass"
    if tt in RUN_TYPES or tt in FUMBLE_TYPES:
        return "qb_kneel" if _flag(r.kneel_down) else "run"
    return None


def _pass_result(tt: str, r) -> str:
    if tt.startswith("Interception") or _flag(r.int_):
        return "interception"
    if tt in LOST_FUMBLE_TYPES or _flag(r.fumble_lost):
        return "fumble_lost"
    if tt == "Sack" or _flag(r.sack):
        return "sack"
    return "complete" if (tt in COMPLETE_TYPES or _flag(r.completion)) else "incomplete"


def _try_kind(r, points: int) -> tuple[str, bool] | None:
    """The try folded into a touchdown row: (play type, good), from ESPN's PAT fields, the text and
    the points it added. None when no try was attempted."""
    pat = getattr(r, "pointAfterAttempt_text", None)
    pat = "" if _nan(pat) else str(pat)
    if points == 1:
        return "extra_point", True
    if points == 2:
        return "two_point", True
    if pat in ("Extra Point Missed", "Blocked PAT"):
        return "extra_point", False
    if pat in ("Two Point Pass", "Two Point Rush"):
        return "two_point", False
    if not _nan(r.two_point_conv_result):
        return "two_point", False
    if not _nan(r.extra_point_result):
        return "extra_point", False
    text = str(r.text or "").lower()
    if "two-point" in text or "two point" in text or "2-pt" in text or "2pt" in text:
        return "two_point", False
    if "kick" in text or "pat" in text or "extra point" in text:
        return "extra_point", False
    return None


def _clock(rows: list[dict]) -> bool:
    """Fill each row's ``qsec`` from its raw clock; interpolate a coarse clock. Returns whether the
    game's clock is known play by play."""
    same = zero = 0
    for a, b in zip(rows, rows[1:]):
        if a["qtr"] == b["qtr"] and a["qtr"] <= 4 and a["kind"] == "play" and b["kind"] == "play":
            same += 1
            zero += a["raw_qsec"] == b["raw_qsec"]
    known = same == 0 or zero / same <= COARSE_CLOCK
    for q in sorted({r["qtr"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["qtr"] == q]
        if q >= 5:
            for i in idx:
                rows[i]["qsec"] = 0
            continue
        vals = np.minimum.accumulate(np.array([rows[i]["raw_qsec"] for i in idx], dtype=float))
        if not known:
            out, a = vals.copy(), 0
            while a < len(vals):
                b = a
                while b < len(vals) and vals[b] == vals[a]:
                    b += 1
                nxt = vals[b] if b < len(vals) else 0.0
                out[a:b] = vals[a] - (vals[a] - nxt) * np.arange(b - a) / (b - a)
                a = b
            vals = np.round(out)
        # the period ends on the quarter's last row, as in the data: earlier rows keep at least a
        # second (a stale 0:00 or an untimed down would end it early) and the last row runs out the
        # clock within one play's elapsed (the feed often stamps it 0:00, or skips the plays before)
        vals = np.maximum(vals, 1)
        vals[-1] = min(vals[-1], MAX_ELAPSED)
        for i, v in zip(idx, vals):
            rows[i]["qsec"] = int(v)
    return known


def _candidates(g: pd.DataFrame) -> list[dict]:
    """First pass: keep real plays, tries and timeouts, each tagged with a phase."""
    first = g.iloc[0]
    home_id, away_id = int(first.homeTeamId), int(first.awayTeamId)

    def side(team_id):
        t = _int(team_id, -1)
        return HOME if t == home_id else AWAY if t == away_id else None

    rows: list[dict] = []
    last_td = None
    for r in g.itertuples(index=False):
        if _nan(r.period):
            continue
        tt = "" if _nan(r.type_text) else str(r.type_text)
        pre = (_int(r.start_homeScore), _int(r.start_awayScore))
        post = (_int(r.end_homeScore, pre[0]), _int(r.end_awayScore, pre[1]))
        row = None
        if tt == "Timeout":
            team = HOME if _flag(r.homeTimeoutCalled) else AWAY if _flag(r.awayTimeoutCalled) else None
            if team is not None:
                row = dict(kind="timeout", team=team)
        elif tt in KICKOFF_TYPES:
            kicker = side(r.kicking_team)
            if kicker is None:
                rec = side(r.pos_team_id)
                kicker = None if rec is None else 1 - rec
            if kicker is not None:
                row = dict(kind="play", phase=KICKOFF, pos=kicker, play_type="kickoff")
        elif tt in XP_TYPES or tt in TWO_POINT_TYPES:
            team = last_td if last_td is not None else side(r.pos_team_id)
            if team is not None:
                good = XP_TYPES.get(tt, TWO_POINT_TYPES.get(tt))
                row = dict(kind="play", phase=PAT, pos=team, play_type="extra_point" if tt in XP_TYPES else "two_point",
                           try_good=good, down=0, ytg=PAT_SPOT, yl=PAT_SPOT)
        else:
            pt = _scrimmage_type(tt, r)
            pos, down, yl = side(r.pos_team_id), _int(r.start_down), _int(r.start_yardsToEndzone, -1)
            if pt is not None and pos is not None and 1 <= down <= 4 and 1 <= yl <= 99:
                row = dict(kind="play", phase=SCRIMMAGE, pos=pos, play_type=pt, down=down,
                           ytg=max(1, min(_int(r.start_distance, 10), yl)), yl=yl, tt=tt)
        if row is None:
            continue
        clock = _int(r.clock_minutes) * 60 + _int(r.clock_seconds)
        row.update(qtr=int(r.period), raw_qsec=min(max(clock, 0), 900), pre=pre, post=post, raw=r)
        row.setdefault("down", 0), row.setdefault("ytg", 0), row.setdefault("yl", 0)
        rows.append(row)
        if row["kind"] == "play" and row["phase"] != PAT:
            for s in (HOME, AWAY):
                if post[s] - pre[s] >= 6:
                    last_td = s
    return _kick_distance(_fill_presnap_penalties(_drop_dead_ball_penalties(_insert_tries(rows))))


def _kick_distance(rows: list[dict]) -> list[dict]:
    """A field goal row's distance to go is usually a yard short in the feed (its spot is right):
    where the spot is where the team's previous play ended, take the distance that play left."""
    prev = None
    for row in rows:
        if row["kind"] != "play":
            continue
        if (row.get("play_type") == "field_goal" and prev is not None and prev["phase"] == SCRIMMAGE
                and prev["pos"] == row["pos"] and prev["qtr"] == row["qtr"]
                and prev.get("play_type") in ("run", "pass", "penalty")):
            gain = prev["yl"] - row["yl"]
            if prev.get("play_type") == "penalty":
                down, ytg = prev["down"], prev["ytg"] - gain
            else:
                down, ytg = (1, min(10, row["yl"])) if gain >= prev["ytg"] else (prev["down"] + 1, prev["ytg"] - gain)
            if down == row["down"] and ytg >= 1 and abs(ytg - row["ytg"]) <= 2:
                row["ytg"] = ytg
        prev = row
    return rows


def _fill_presnap_penalties(rows: list[dict]) -> list[dict]:
    """The feed drops some pre-snap fouls (false starts, offside). Where a play ends at spot ``e``
    (``end.yardsToEndzone``) and the same team's next snap is ``k`` yards from it on the down it
    should be, with the line to gain moved by the same ``k``, the only explanation is a foul in
    between: add it as a penalty row from ``e`` (so the play keeps its own yards)."""
    out = []
    plays = [i for i, r in enumerate(rows) if r["kind"] == "play"]
    after = {a: b for a, b in zip(plays, plays[1:])}
    for i, a in enumerate(rows):
        out.append(a)
        j = after.get(i)
        if j is None:
            continue
        b = rows[j]
        if b["phase"] != SCRIMMAGE or b["qtr"] != a["qtr"] or a["post"] != a["pre"] or b["pre"] != a["post"]:
            continue
        e = _int(a["raw"].end_yardsToEndzone, -1)
        if not 1 <= e <= 99 or e == b["yl"]:
            continue
        if a["phase"] == KICKOFF or a.get("play_type") == "punt":
            if b["pos"] == a["pos"]:
                continue
            down, ytg = 1, min(10, e)
        elif a["phase"] == SCRIMMAGE and a.get("play_type") in ("run", "pass") and b["pos"] == a["pos"]:
            gain = a["yl"] - e
            down, ytg = (1, min(10, e)) if gain >= a["ytg"] else (a["down"] + 1, a["ytg"] - gain)
            if down > 4:
                continue
        else:
            continue
        k = b["yl"] - e
        if abs(k) > 15 or b["down"] != down or b["ytg"] - ytg != k or ytg + k < 1:
            continue
        out.append(dict(kind="play", phase=SCRIMMAGE, pos=b["pos"], play_type="penalty", down=down, ytg=ytg, yl=e,
                        qtr=b["qtr"], raw_qsec=b["raw_qsec"], pre=b["pre"], post=b["pre"], raw=b["raw"],
                        tt="Penalty", inserted=True))
    return out


def _drop_dead_ball_penalties(rows: list[dict]) -> list[dict]:
    """Penalties between a score (or its try) and the kickoff that follows come as made-up
    1st-and-10 scrimmage rows; the kickoff's spot already carries their enforcement."""
    out = []
    for i, row in enumerate(rows):
        if row["kind"] == "play" and row.get("play_type") == "penalty":
            prev = next((x for x in reversed(out) if x["kind"] == "play"), None)
            nxt = next((x for x in rows[i + 1:] if x["kind"] == "play" and x.get("play_type") != "penalty"), None)
            after_score = prev is not None and (prev["phase"] == PAT or prev["post"] != prev["pre"])
            if after_score and nxt is not None and nxt["phase"] == KICKOFF:
                continue
        out.append(row)
    return out


def _insert_tries(rows: list[dict]) -> list[dict]:
    """After a touchdown whose row carries its try, add the try as its own row."""
    out = []
    for i, row in enumerate(rows):
        out.append(row)
        if row["kind"] != "play" or row["phase"] == PAT:
            continue
        delta = [row["post"][s] - row["pre"][s] for s in (HOME, AWAY)]
        scorer = next((s for s in (HOME, AWAY) if delta[s] >= 6), None)
        if scorer is None:
            continue
        nxt = next((x for x in rows[i + 1:] if x["kind"] == "play"), None)
        extra = delta[scorer] - 6
        if nxt is not None and nxt["phase"] == PAT:  # the feed lists the try itself
            if extra in (1, 2) and nxt["pre"] == nxt["post"] == row["post"]:
                # ... but already counted it in the touchdown's score (2004-2013)
                td_post = list(row["post"])
                td_post[scorer] -= extra
                row["post"] = nxt["pre"] = tuple(td_post)
            continue
        if extra > 2:
            continue  # not a touchdown and its try (bad scores): left to the replay check
        kind = _try_kind(row["raw"], extra)
        if kind is None:
            continue
        td_post = list(row["pre"])
        td_post[scorer] += 6
        td_post[1 - scorer] = row["post"][1 - scorer]
        pat = dict(kind="play", phase=PAT, pos=scorer, play_type=kind[0], try_good=kind[1], down=0,
                   ytg=PAT_SPOT, yl=PAT_SPOT, qtr=row["qtr"], raw_qsec=row["raw_qsec"], pre=tuple(td_post),
                   post=row["post"], raw=row["raw"], inserted=True)
        row["post"] = tuple(td_post)
        out.append(pat)
    return out


def _outcome(row: dict, nxt: dict | None) -> tuple[str, int]:
    """Derive (result, yards) for a non-timeout play (the NFL preprocess's rules)."""
    r = row["raw"]
    off, dfn = row["pos"], 1 - row["pos"]
    pts = (row["post"][0] - row["pre"][0], row["post"][1] - row["pre"][1])
    t = row["play_type"]
    q = row["qtr"]
    same_half = nxt is not None and (nxt["qtr"] == q or (q, nxt["qtr"]) in ((1, 2), (3, 4)) or (q >= 5 and nxt["qtr"] >= 5))
    usable = nxt is not None and same_half and nxt["phase"] == SCRIMMAGE

    def spot_for(holder: int) -> int:
        if pts[holder] >= 6:
            return 0
        if usable and nxt["pos"] == holder:
            return nxt["yl"]
        return 75 if t == "kickoff" else 80

    if t == "kickoff":
        res = "kick_recovered" if (usable and nxt["pos"] == off) or pts[off] >= 6 else "kick_received"
        return res, spot_for(off if res == "kick_recovered" else dfn)
    if row["phase"] == PAT:
        return ("good" if row["try_good"] else "no_good"), 0
    if t == "punt":
        res = "kick_recovered" if (usable and nxt["pos"] == off and pts[dfn] < 6) or pts[off] >= 6 else "kick_received"
        return res, spot_for(off if res == "kick_recovered" else dfn)
    if t == "field_goal":
        if pts[off] == 3:
            return "good", 0
        return "no_good", spot_for(dfn)

    tt = row.get("tt", "")
    if t == "pass":
        res = _pass_result(tt, r)
    elif t in ("run", "qb_kneel"):
        res = "fumble_lost" if (tt in LOST_FUMBLE_TYPES or _flag(r.fumble_lost)) else ("kneel" if t == "qb_kneel" else "run")
    elif t == "qb_spike":
        res = "spike"
    else:
        gain = row["yl"] - nxt["yl"] if usable else 0
        res = "penalty_first_down" if (
            usable and nxt["pos"] == off and nxt["down"] == 1 and gain < row["ytg"]
            and nxt["ytg"] == min(10, nxt["yl"])
            and not (row["down"] == 1 and nxt["ytg"] == row["ytg"] - gain)) else "penalty"
    if res in ("interception", "fumble_lost"):
        return res, spot_for(dfn)

    yl = row["yl"]
    if pts[off] >= 6:
        return res, yl
    if pts[dfn] == 2:
        return res, yl - 100
    if usable and nxt["pos"] == off and not (q >= 5 and nxt["qtr"] != q):
        gain = yl - nxt["yl"]
    elif usable and q <= 4:
        gain = yl - (100 - nxt["yl"])
    else:  # overtime possession over (the next state is the other team's 25) or nothing to go by
        gain = _int(r.statYardage)
    return res, int(min(max(gain, yl - 99), yl - 1))


def _penalty_as_played(row: dict, nxt: dict | None, res: str, yards: int) -> tuple[str, str]:
    """(play type, result) of a scrimmage play the feed folds a penalty into, from the down that
    followed (the NFL's feed marks these as no-plays itself): a run or pass whose down is replayed
    (an accepted foul that negated it) is a penalty; one that gives a first down short of the line
    to gain is a penalty first down; a "Penalty" that costs a down (intentional grounding) is an
    incomplete pass."""
    t = row["play_type"]
    if (nxt is None or nxt["phase"] != SCRIMMAGE or nxt["pos"] != row["pos"] or nxt["qtr"] != row["qtr"]
            or res in ("interception", "fumble_lost") or yards >= row["yl"]):
        return t, res
    if t == "penalty":
        if nxt["down"] == row["down"] + 1 and res == "penalty" and row["down"] < 4:
            return "pass", "incomplete"
        return t, res
    if t not in ("run", "pass", "qb_kneel", "qb_spike") or yards >= row["ytg"]:
        return t, res
    if nxt["down"] == row["down"] and nxt["down"] > 1:
        return "penalty", "penalty"
    if nxt["down"] == 1 and nxt["ytg"] == min(10, nxt["yl"]):
        return "penalty", "penalty_first_down"
    return t, res


def _week(r) -> int:
    w = _int(r.week, 0)
    return w + POST_WEEK if _int(r.seasonType, 2) == 3 else w


def _drop_repeats(g: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Some games list a stretch of scrimmage plays twice, under new play numbers: drop a play
    identical to one of the `window` rows before it."""
    seen: dict[tuple, int] = {}
    keep = []
    for i, key in enumerate(g[DUPLICATE_KEY].itertuples(index=False, name=None)):
        tt = key[-1]
        drop = (isinstance(key[0], str) and key[0] != "" and tt not in KICKOFF_TYPES and tt != "Timeout"
                and tt not in XP_TYPES and tt not in TWO_POINT_TYPES and i - seen.get(key, -window - 1) <= window)
        keep.append(not drop)
        seen[key] = i
    return g[keep]


DUPLICATE_KEY = ["text", "period", "start_down", "start_distance", "start_yardsToEndzone", "pos_team_id", "type_text"]


def process_game(g: pd.DataFrame) -> tuple[dict, list[dict]] | None:
    g = g.sort_values("game_play_number")
    g = _drop_repeats(g)
    if g.homeTeamId.isna().any() or g.awayTeamId.isna().any():
        return None
    rows = _candidates(g)
    while rows and rows[0]["kind"] == "timeout":
        rows.pop(0)
    plays = [x for x in rows if x["kind"] == "play"]
    # the feed of many 2004-2013 games starts after the opening kickoff: the sequence then starts
    # with the first snap, and the opening kicker is the team without the ball
    if len(plays) < 50 or plays[0]["qtr"] != 1 or plays[0]["phase"] == PAT:
        return None
    opening_kicker = plays[0]["pos"] if plays[0]["phase"] == KICKOFF else 1 - plays[0]["pos"]
    clock_known = _clock(rows)
    first = g.iloc[0]
    meta = dict(
        game_id=str(int(first.game_id)), season=int(first.season),
        game_type="POST" if _int(first.seasonType, 2) == 3 else "REG", week=_week(first),
        home=str(int(first.homeTeamId)), away=str(int(first.awayTeamId)),
        home_name=str(first.homeTeamName), away_name=str(first.awayTeamName),
        home_final=_int(first.homeFinalScore), away_final=_int(first.awayFinalScore),
        opening_kicker=opening_kicker, clock_known=clock_known,
    )
    ot_first: dict[int, int] = {}
    for x in plays:
        if x["qtr"] >= 5:
            ot_first.setdefault(x["qtr"], x["pos"])
    out = []
    timeouts = [TIMEOUTS_PER_HALF] * 2
    cur = 1
    for i, row in enumerate(rows):
        period = 1 if row["qtr"] <= 2 else 2 if row["qtr"] <= 4 else row["qtr"]
        if period != cur:
            timeouts = [TIMEOUTS_PER_HALF] * 2 if period == 2 else [TIMEOUTS_PER_OT] * 2
            cur = period
        nxt_any = rows[i + 1] if i + 1 < len(rows) else None
        nxt_play = next((x for x in rows[i + 1:] if x["kind"] == "play"), None)
        if row["kind"] == "timeout":
            if nxt_play is None or timeouts[row["team"]] <= 0:
                continue
            base = nxt_play
            st = dict(phase=base["phase"], pos=base["pos"], down=base["down"], ytg=base["ytg"], yl=base["yl"])
            oc = dict(play_type="timeout", result="timeout_offense" if row["team"] == base["pos"] else "timeout_defense",
                      yards=0)
        else:
            res, yards = _outcome(row, nxt_play)
            pt = row["play_type"]
            if row["phase"] == SCRIMMAGE:
                pt, res = _penalty_as_played(row, nxt_play, res, yards)
            st = dict(phase=row["phase"], pos=row["pos"], down=row["down"], ytg=row["ytg"], yl=row["yl"])
            oc = dict(play_type=pt, result=res, yards=yards)
        st.update(qtr=row["qtr"], qsec=row["qsec"], score_h=row["pre"][0], score_a=row["pre"][1])
        if row["qtr"] >= 5:
            el = 0
        elif nxt_any is not None and nxt_any["qtr"] == row["qtr"]:
            el = row["qsec"] - nxt_any["qsec"]
        else:
            el = row["qsec"]
        oc["elapsed"] = int(min(max(el, 0), MAX_ELAPSED))
        f = ot_first.get(row["qtr"], -1)
        st.update(to_h=timeouts[0], to_a=timeouts[1], ot_first=f, ot_half=-1 if f < 0 else int(st["pos"] != f))
        play_no = row["raw"].game_play_number
        out.append({"game_id": meta["game_id"], "play_id": _int(play_no, -1), **st, **oc,
                    "elapsed_known": clock_known})
        if row["kind"] == "timeout":
            timeouts[row["team"]] -= 1
    return meta, out


def state_from_row(meta: dict, row) -> CFBState:
    get = row.get if isinstance(row, dict) else (lambda k: getattr(row, k))
    f = int(get("ot_first"))
    return CFBState(
        season=meta["season"], game_type=meta["game_type"], home=meta["home"], away=meta["away"],
        qtr=int(get("qtr")), qsec=int(get("qsec")), phase=int(get("phase")), pos=int(get("pos")),
        down=int(get("down")), ytg=int(get("ytg")), yl=int(get("yl")),
        score=[int(get("score_h")), int(get("score_a"))], timeouts=[int(get("to_h")), int(get("to_a"))],
        opening_kicker=int(meta["opening_kicker"]), ot_first=None if f < 0 else f,
        ot_half=max(0, int(get("ot_half"))),
    )


def outcome_from_row(row) -> Outcome:
    get = row.get if isinstance(row, dict) else (lambda k: getattr(row, k))
    return Outcome(get("play_type"), get("result"), int(get("yards")), int(get("elapsed")))


def replay(meta: dict, plays: list[dict]) -> dict:
    """Apply each real outcome to the real state and compare with the real next state.

    Returns {"ok", "total"} over regulation transitions, {"ot_ok", "ot_total"} over overtime ones
    (including the one into overtime), the mismatch counts by play and field (``misses``), and
    whether the final score agrees with the engine (``final_ok``: the last state after its outcome;
    None when the game was not played to the end in the data)."""
    out = dict(ok=0, total=0, ot_ok=0, ot_total=0, misses={}, final_ok=None)
    for a, b in zip(plays, plays[1:]):
        s = state_from_row(meta, a)
        n = apply_outcome(s, outcome_from_row(a), ot_first=b["pos"] if b["qtr"] >= 5 else None)
        if n.over:
            continue
        got = dict(qtr=n.qtr, phase=n.phase, pos=n.pos, down=n.down, ytg=n.ytg, yl=n.yl,
                   score_h=n.score[0], score_a=n.score[1])
        bad = [k for k in COMPARE if got[k] != b[k] and not (k in ("down", "ytg", "yl") and b["phase"] != SCRIMMAGE)]
        ot = b["qtr"] >= 5
        out["ot_total" if ot else "total"] += 1
        if not bad:
            out["ot_ok" if ot else "ok"] += 1
        else:
            key = f"{'ot ' if ot else ''}{a['play_type']}/{a['result']}:{','.join(bad)}"
            out["misses"][key] = out["misses"].get(key, 0) + 1
    if plays:
        last = plays[-1]
        n = apply_outcome(state_from_row(meta, last), outcome_from_row(last))
        if n.over:
            out["final_ok"] = (n.score[HOME], n.score[AWAY]) == (meta["home_final"], meta["away_final"])
    return out


def _keep_overtime(meta: dict, rows: list[dict], rep: dict) -> list[dict]:
    """A game's plays, without its overtime unless every overtime transition replays."""
    if rep["ot_total"] and rep["ot_ok"] == rep["ot_total"] and rep["final_ok"] is not False:
        return rows
    return [r for r in rows if r["qtr"] <= 4]


def game_stats(df: pd.DataFrame, metas: pd.DataFrame) -> pd.DataFrame:
    """One row per kept game for the ratings: teams, final score and each team's offensive EPA/play
    (scrimmage runs and passes)."""
    plays = df[(df["rush"].fillna(False).astype(bool) | df["pass"].fillna(False).astype(bool)) & df["EPA"].notna()]
    epa = plays.groupby([plays.game_id.astype("int64").astype(str), plays.pos_team_id.astype("int64").astype(str)]).EPA.mean()
    g = metas[["game_id", "season", "week", "home", "away", "home_final", "away_final"]].rename(
        columns={"home_final": "home_score", "away_final": "away_score"}).copy()
    g["game_date"] = [f"{s}-{w:02d}" for s, w in zip(g.season, g.week)]
    g["home_epa"] = [epa.get((i, h), np.nan) for i, h in zip(g.game_id, g.home)]
    g["away_epa"] = [epa.get((i, a), np.nan) for i, a in zip(g.game_id, g.away)]
    return g


def _columns(df: pd.DataFrame) -> pd.DataFrame:
    """itertuples-friendly column names (``type.text`` -> ``type_text``; ``pass`` / ``int`` are keywords)."""
    ren = {c: c.replace(".", "_") for c in df.columns}
    ren |= {"pass": "pass_", "int": "int_"}
    return df.rename(columns=ren)


def preprocess(seasons: list[int], raw_dir: Path, out_dir: Path, min_agreement: float = 0.95) -> dict:
    """Write games.parquet, plays.parquet and ratings.parquet under `out_dir`; returns the replay
    summary (also logged)."""
    from ...nfl.data.ratings import compute_ratings

    out_dir.mkdir(parents=True, exist_ok=True)
    metas, all_rows, stats = [], [], []
    tot = dict(ok=0, total=0, ot_ok=0, ot_total=0, games=0, kept=0, final_ok=0, final_n=0, clock_known=0,
               kept_ok=0, kept_total=0)
    misses: dict[str, int] = {}
    for season in seasons:
        path = raw_dir / f"pbp_{season}.parquet"
        if not path.exists():
            log.warning("cfb %d: no raw play-by-play (runitback cfb download)", season)
            continue
        raw = load_raw(season, raw_dir)
        if "status_type_completed" in raw and raw.status_type_completed.notna().any():
            raw = raw[raw.status_type_completed.fillna(True).astype(bool)]
        df = _columns(raw)
        kept_meta = []
        n_games = 0
        for _, g in df.groupby("game_id", sort=False):
            n_games += 1
            res = process_game(g)
            if res is None:
                continue
            meta, rows = res
            rep = replay(meta, rows)
            for k, v in rep["misses"].items():
                misses[k] = misses.get(k, 0) + v
            for k in ("ok", "total", "ot_ok", "ot_total"):
                tot[k] += rep[k]
            meta["agreement"] = rep["ok"] / max(rep["total"], 1)
            meta["ot_agreement"] = rep["ot_ok"] / rep["ot_total"] if rep["ot_total"] else np.nan
            if rep["final_ok"] is not None:
                tot["final_n"] += 1
                tot["final_ok"] += rep["final_ok"]
            if meta["agreement"] < min_agreement:
                continue
            tot["kept_ok"] += rep["ok"]
            tot["kept_total"] += rep["total"]
            rows = _keep_overtime(meta, rows, rep)
            meta["overtime_kept"] = any(r["qtr"] >= 5 for r in rows)
            for i, r in enumerate(rows):
                r["idx"] = i
            metas.append(meta)
            kept_meta.append(meta)
            all_rows.extend(rows)
            tot["clock_known"] += meta["clock_known"]
        tot["games"] += n_games
        tot["kept"] += len(kept_meta)
        if kept_meta:
            stats.append(game_stats(raw, pd.DataFrame(kept_meta)))
        log.info("cfb %d: kept %d/%d games", season, len(kept_meta), n_games)
    games = pd.DataFrame(metas)
    plays = pd.DataFrame(all_rows)
    games.to_parquet(out_dir / "games.parquet")
    plays.to_parquet(out_dir / "plays.parquet")
    if stats:
        compute_ratings(pd.concat(stats, ignore_index=True)).to_parquet(out_dir / "ratings.parquet")
    summary = dict(tot, agreement=tot["ok"] / max(tot["total"], 1),
                   kept_agreement=tot["kept_ok"] / max(tot["kept_total"], 1),
                   ot_agreement=tot["ot_ok"] / max(tot["ot_total"], 1),
                   final_agreement=tot["final_ok"] / max(tot["final_n"], 1),
                   misses=dict(sorted(misses.items(), key=lambda kv: -kv[1])[:25]))
    log.info("engine replay agreement: %.4f over %d regulation transitions (%.4f in the games kept); "
             "overtime %.4f over %d; final score %.4f of %d games; kept %d/%d games (%d with a per-play clock)",
             summary["agreement"], tot["total"], summary["kept_agreement"], summary["ot_agreement"], tot["ot_total"],
             summary["final_agreement"], tot["final_n"], tot["kept"], tot["games"], tot["clock_known"])
    for k, v in summary["misses"].items():
        log.info("  mismatch %-50s %d", k, v)
    return summary
