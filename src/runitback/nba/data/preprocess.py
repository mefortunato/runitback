"""Turn ESPN NBA play-by-play into (state, outcome) event sequences for the engine and model.

Each event's outcome is derived from what happens *next* where possible: a missed shot is an
offensive or defensive rebound depending on which team acts next, a free throw that follows a
shot by the same team at the same clock turns that shot into an and-one (or a missed shot that
drew a foul), and free throws with no shot before them get a `foul` event. Replaying a real
game through the rules engine should then reproduce it.
"""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..engine.rules import AWAY, HOME, GameState, Outcome, apply_outcome, period_length
from ..vocab import (
    FREE_THROW, LIVE, MAX_Q4_TIMEOUTS, MAX_ELAPSED, OT_TIMEOUTS, TEAMS, TIMEOUTS_PER_GAME, TIP, canon_team,
)
from .download import load_raw

log = logging.getLogger(__name__)

STATE_COLS = ["qtr", "qsec", "phase", "pos", "fts_left", "score_h", "score_a", "to_h", "to_a"]
OUTCOME_COLS = ["play_type", "result", "elapsed"]
GAME_TYPES = {2: "REG", 3: "POST", 5: "POST"}  # regular season, playoffs, play-in
FT_OF = re.compile(r"(\d)\s*of\s*(\d)")
POSSESSION_KINDS = ("shot", "ft", "to", "foul", "jump")


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
        tt = str(r.type_text or "")
        text = str(r.text or "").lower()
        side = HOME if r.team_id == home_id else AWAY if r.team_id == away_id else None
        a = None
        if tt.startswith("Free Throw"):
            m = FT_OF.search(tt)
            made = bool(r.scoring_play) or "makes" in text
            if "Technical" in tt or m is None:
                a = dict(kind="tech", made=made)
            else:
                a = dict(kind="ft", k=int(m.group(1)), n=int(m.group(2)), made=made)
        elif "Turnover" in tt and tt != "No Turnover":
            a = dict(kind="to")
        elif tt == "Full Timeout" or tt == "Timeout":
            a = dict(kind="timeout")
        elif tt in ("Jumpball", "Jump Ball"):
            a = dict(kind="jump")
        elif tt in ("Offensive Rebound", "Defensive Rebound"):
            a = dict(kind="reb", off=tt.startswith("Offensive"))
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


def _events(actions: list[dict]) -> list[dict]:
    """Attach shooting fouls to shots, add `foul` events before other free-throw trips, and add a
    tip-off at the start of the game and of each overtime and a run-out at the end of each period."""
    events: list[dict] = []
    for a in actions:
        if a["kind"] == "reb":
            if events:
                events[-1].setdefault("rebounds", []).append(a)
            continue
        if a["kind"] == "ft" and a["k"] == 1:
            prev = next((e for e in reversed(events) if e["kind"] not in ("timeout", "tech")), None)
            if prev is not None and prev["kind"] == "shot" and prev["side"] == a["side"] \
                    and prev["qtr"] == a["qtr"] and prev["clock"] == a["clock"] and "fouled" not in prev \
                    and ((prev["made"] and a["n"] == 1) or (not prev["made"] and a["n"] == prev["pts"])):
                prev["fouled"] = a["n"]
            else:
                events.append(dict(kind="foul", side=a["side"], n=a["n"], qtr=a["qtr"], clock=a["clock"]))
        events.append(a)
    out: list[dict] = []
    for q in sorted({e["qtr"] for e in events}):
        ev = [e for e in events if e["qtr"] == q]
        if q == 1 or q >= 5:
            first_jump = next((i for i, e in enumerate(ev) if e["kind"] == "jump"), None)
            first_poss = next((i for i, e in enumerate(ev) if e["kind"] in POSSESSION_KINDS), None)
            if first_jump is not None and first_jump == first_poss:
                ev[first_jump]["tip"] = True
                ev = [ev[first_jump]] + [e for i, e in enumerate(ev) if i != first_jump and
                                         not (e["kind"] == "timeout" and i < first_jump)]
            elif first_poss is not None:
                tip = dict(kind="jump", tip=True, side=ev[first_poss]["side"], qtr=q, clock=period_length(q))
                ev = [tip] + [e for i, e in enumerate(ev) if e["kind"] != "timeout" or i > first_poss]
        if ev and ev[-1]["clock"] > 0:
            ev.append(dict(kind="run_out", side=None, qtr=q, clock=0))
        out += ev
    return out


def _next_actor(events: list[dict], i: int) -> int | None:
    q = events[i]["qtr"]
    for e in events[i + 1:]:
        if e["qtr"] != q:
            return None
        if e["kind"] in POSSESSION_KINDS:
            return e["side"]
    return None


def _rebound_side(e: dict) -> int | None:
    rebs = e.get("rebounds") or []
    if not rebs:
        return None
    r = rebs[0]
    return r["side"]


def process_game(g: pd.DataFrame) -> tuple[dict, list[dict]] | None:
    first = g.iloc[0]
    home, away = canon_team(str(first.home_team_abbrev)), canon_team(str(first.away_team_abbrev))
    game_type = GAME_TYPES.get(int(first.season_type))
    if home not in TEAMS or away not in TEAMS or game_type is None:
        return None
    events = _events(_actions(g))
    if len(events) < 150 or events[0]["kind"] != "jump":
        return None
    last = g.loc[g.period_number == g.period_number.max()]
    meta = dict(game_id=str(first.game_id), season=int(first.season), game_type=game_type,
                game_date=str(first.game_date)[:10], home=home, away=away,
                home_final=int(last.home_score.max()), away_final=int(last.away_score.max()),
                tip_winner=int(events[0]["side"]))
    spread = getattr(first, "home_team_spread", None)
    meta["home_spread"] = None if _nan(spread) else float(spread)

    rows: list[dict] = []
    score = [0, 0]
    timeouts = [TIMEOUTS_PER_GAME, TIMEOUTS_PER_GAME]
    trip_left = 0
    eng: GameState | None = None
    cur_q, qsec = 0, 0
    poss = [0, 0]
    for i, e in enumerate(events):
        q = e["qtr"]
        if q != cur_q:
            if eng is not None and not eng.over:
                poss[eng.pos] += 1
            cur_q, qsec, trip_left = q, period_length(q), 0
            if q == 4:
                timeouts = [min(t, MAX_Q4_TIMEOUTS) for t in timeouts]
            elif q >= 5:
                timeouts = [OT_TIMEOUTS, OT_TIMEOUTS]
        prev_pos = eng.pos if eng is not None else HOME  # who the engine says has the ball
        nxt = _next_actor(events, i)
        k = e["kind"]
        phase, pos, fts_left = (FREE_THROW if trip_left > 0 else LIVE), prev_pos, trip_left
        if k == "jump":
            if e.get("tip"):
                phase, pos, res = TIP, HOME, ("won" if e["side"] == HOME else "lost")
            else:
                res = "won" if e["side"] == prev_pos else "lost"
            oc = ("jump_ball", res)
        elif k == "shot":
            pos, phase, fts_left = e["side"], LIVE, 0
            if "fouled" in e:
                res = "made_fouled" if e["made"] else "missed_fouled"
            elif e["made"]:
                res = "made"
            else:
                after = nxt if nxt is not None else _rebound_side(e)
                res = "missed_off" if after == e["side"] else "missed_def"
            oc = (f"shot{e['pts']}", res)
        elif k == "ft":
            pos, phase, fts_left = e["side"], FREE_THROW, e["n"] - e["k"] + 1
            if e["made"]:
                res = "made"
            elif e["k"] < e["n"]:
                res = "missed"
            else:
                after = nxt if nxt is not None else _rebound_side(e)
                res = "missed_off" if after == e["side"] else "missed_def"
            oc = ("free_throw", res)
        elif k == "tech":
            pos = prev_pos if trip_left > 0 or nxt is None else nxt
            res = "tech_missed" if not e["made"] else "tech_offense" if e["side"] == pos else "tech_defense"
            oc = ("technical", res)
        elif k == "to":
            pos, phase, fts_left = e["side"], LIVE, 0
            oc = ("turnover", "turnover")
        elif k == "foul":
            pos, phase, fts_left = e["side"], LIVE, 0
            oc = ("foul", f"fts_{min(e['n'], 3)}")
        elif k == "timeout":
            pos = prev_pos if trip_left > 0 or nxt is None else nxt
            if timeouts[e["side"]] <= 0:
                continue
            oc = ("timeout", "timeout_offense" if e["side"] == pos else "timeout_defense")
        else:  # run_out
            oc = ("run_out", "end")
        elapsed = qsec if k == "run_out" else qsec - e["clock"]
        row = dict(game_id=meta["game_id"], qtr=q, qsec=qsec, phase=phase, pos=pos, fts_left=fts_left,
                   score_h=score[0], score_a=score[1], to_h=timeouts[0], to_a=timeouts[1],
                   play_type=oc[0], result=oc[1], elapsed=int(min(max(elapsed, 0), MAX_ELAPSED)))
        rows.append(row)
        # bookkeeping for the next event
        state = state_from_row(meta, row)
        eng = apply_outcome(state, outcome_from_row(row))
        if eng.qtr == q and eng.pos != pos and not eng.over:
            poss[pos] += 1
        if k == "timeout":
            timeouts[e["side"]] -= 1
        if "post" in e:
            score = list(e["post"])
        if k == "shot" and "fouled" in e:
            trip_left = e["fouled"]
        elif k == "foul":
            trip_left = e["n"]
        elif k == "ft":
            trip_left = e["n"] - e["k"]
        elif k in ("to", "jump"):
            trip_left = 0
        qsec = 0 if k == "run_out" else e["clock"]
    if eng is not None and not eng.over:
        poss[eng.pos] += 1
    meta.update(poss_h=poss[0], poss_a=poss[1])
    return meta, rows


def state_from_row(meta: dict, row) -> GameState:
    get = row.get if isinstance(row, dict) else (lambda k: getattr(row, k))
    return GameState(
        season=meta["season"], game_type=meta["game_type"], home=meta["home"], away=meta["away"],
        qtr=int(get("qtr")), qsec=int(get("qsec")), phase=int(get("phase")), pos=int(get("pos")),
        fts_left=int(get("fts_left")), fts_total=int(get("fts_left")),
        score=[int(get("score_h")), int(get("score_a"))], timeouts=[int(get("to_h")), int(get("to_a"))],
        tip_winner=int(meta["tip_winner"]),
    )


def outcome_from_row(row) -> Outcome:
    get = row.get if isinstance(row, dict) else (lambda k: getattr(row, k))
    return Outcome(get("play_type"), get("result"), int(get("elapsed")))


COMPARE = ["qtr", "phase", "pos", "fts_left", "score_h", "score_a"]


def replay_agreement(meta: dict, rows: list[dict]) -> tuple[int, int, dict]:
    """Apply each real outcome to the real state; compare with the real next state."""
    ok = total = 0
    misses: dict[str, int] = {}
    for a, b in zip(rows, rows[1:]):
        n = apply_outcome(state_from_row(meta, a), outcome_from_row(a))
        if n.over:
            continue
        got = dict(qtr=n.qtr, phase=n.phase, pos=n.pos, fts_left=n.fts_left, score_h=n.score[0],
                   score_a=n.score[1])
        total += 1
        bad = [k for k in COMPARE if got[k] != b[k] and not (k == "pos" and b["phase"] == TIP)]
        if not bad:
            ok += 1
        else:
            key = f"{a['play_type']}/{a['result']}:{','.join(bad)}"
            misses[key] = misses.get(key, 0) + 1
    return ok, total, misses


def preprocess(seasons: list[int], raw_dir: Path, out_dir: Path, min_agreement: float = 0.9) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metas, all_rows = [], []
    tot_ok = tot = 0
    misses: dict[str, int] = {}
    for season in seasons:
        path = raw_dir / f"pbp_{season}.parquet"
        if not path.exists():
            log.warning("season %d: %s not found (download it first); skipped", season, path)
            continue
        df = load_raw(season, raw_dir)
        n_games = n_kept = 0
        for _, g in df.groupby("game_id", sort=False):
            n_games += 1
            res = process_game(g)
            if res is None:
                continue
            meta, rows = res
            ok, total, m = replay_agreement(meta, rows)
            for k, v in m.items():
                misses[k] = misses.get(k, 0) + v
            tot_ok += ok
            tot += total
            meta["agreement"] = ok / max(total, 1)
            if meta["agreement"] < min_agreement:
                continue
            n_kept += 1
            metas.append(meta)
            for i, r in enumerate(rows):
                r["idx"] = i
            all_rows.extend(rows)
        log.info("season %d: kept %d/%d games", season, n_kept, n_games)
    games = pd.DataFrame(metas)
    plays = pd.DataFrame(all_rows)
    games.to_parquet(out_dir / "games.parquet")
    plays.to_parquet(out_dir / "plays.parquet")
    from .ratings import build_ratings
    build_ratings(games, out_dir)
    top = sorted(misses.items(), key=lambda kv: -kv[1])[:25]
    log.info("engine replay agreement: %.4f over %d transitions", tot_ok / max(tot, 1), tot)
    for k, v in top:
        log.info("  mismatch %-50s %d", k, v)
