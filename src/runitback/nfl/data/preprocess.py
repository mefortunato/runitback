"""Turn nflverse play-by-play into (state, outcome) sequences for the engine and model.

Outcomes are derived from the *next* play's state where possible (net field position,
possession, clock), so that replaying a real game through the rules engine reproduces it.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

from ..engine.rules import AWAY, HOME, GameState, Outcome, apply_outcome
from ..vocab import KICKOFF, MAX_ELAPSED, PAT, SCRIMMAGE, TEAM_ALIASES, yards_kind
from .download import load_raw

log = logging.getLogger(__name__)

STATE_COLS = ["qtr", "qsec", "phase", "pos", "down", "ytg", "yl", "score_h", "score_a", "to_h", "to_a"]
OUTCOME_COLS = ["play_type", "result", "yards", "elapsed"]


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _team(x):
    return TEAM_ALIASES.get(x, x)


def _candidates(g: pd.DataFrame) -> list[dict]:
    """First pass: keep real plays and timeouts, tag each with a phase."""
    home = _team(g.home_team.iloc[0])
    rows = []
    pre_h = pre_a = 0
    for r in g.itertuples(index=False):
        post_h = 0 if _nan(r.total_home_score) else int(r.total_home_score)
        post_a = 0 if _nan(r.total_away_score) else int(r.total_away_score)
        pt = r.play_type
        desc = r.desc or ""
        row = None
        if r.timeout == 1 and (_nan(pt) or pt == "no_play") and desc.lower().startswith("timeout") \
                and not _nan(r.timeout_team):
            row = dict(kind="timeout", team=HOME if _team(r.timeout_team) == home else AWAY)
        elif not _nan(pt) and not _nan(r.posteam):
            side = HOME if _team(r.posteam) == home else AWAY
            if pt == "kickoff":
                row = dict(kind="play", phase=KICKOFF, pos=1 - side, play_type="kickoff")
            elif pt == "extra_point" or r.two_point_attempt == 1:
                row = dict(kind="play", phase=PAT, pos=side,
                           play_type="extra_point" if pt == "extra_point" else "two_point")
            elif not _nan(r.down) and pt in ("run", "pass", "punt", "field_goal", "qb_kneel",
                                             "qb_spike", "no_play"):
                row = dict(kind="play", phase=SCRIMMAGE, pos=side,
                           play_type="penalty" if pt == "no_play" else pt,
                           down=int(r.down), ytg=int(r.ydstogo), yl=int(r.yardline_100))
        if row is not None and not _nan(r.qtr) and not _nan(r.quarter_seconds_remaining):
            row.update(qtr=int(r.qtr), qsec=int(r.quarter_seconds_remaining),
                       pre=(pre_h, pre_a), post=(post_h, post_a), raw=r)
            row.setdefault("down", 0), row.setdefault("ytg", 0), row.setdefault("yl", 0)
            rows.append(row)
        pre_h, pre_a = post_h, post_a
    # drop kicks / PATs that were re-done (penalty) and plays after the game-ending play
    out = []
    for i, row in enumerate(rows):
        if row["kind"] == "play" and row["phase"] in (KICKOFF, PAT):
            nxt = next((x for x in rows[i + 1:] if x["kind"] == "play"), None)
            if nxt is not None and nxt["phase"] == row["phase"] and nxt["post"] == row["post"] \
                    and nxt["qtr"] == row["qtr"]:
                continue
        out.append(row)
    return out


def _outcome(row: dict, nxt: dict | None) -> tuple[str, str, int]:
    """Derive (result, yards) for a non-timeout play."""
    r = row["raw"]
    off, dfn = row["pos"], 1 - row["pos"]
    pts = (row["post"][0] - row["pre"][0], row["post"][1] - row["pre"][1])
    t = row["play_type"]
    same_half = nxt is not None and (nxt["qtr"] == row["qtr"] or (row["qtr"], nxt["qtr"]) in ((1, 2), (3, 4)))
    usable = nxt is not None and same_half and nxt["phase"] == SCRIMMAGE

    def spot_for(holder: int) -> int:
        if pts[holder] >= 6:
            return 0
        if usable and nxt["pos"] == holder:
            return nxt["yl"]
        return 75 if t == "kickoff" else 80  # fallback: touchback-ish

    if t == "kickoff":
        res = "kick_recovered" if (usable and nxt["pos"] == off) or pts[off] >= 6 else "kick_received"
        return res, spot_for(off if res == "kick_recovered" else dfn)
    if t in ("extra_point", "two_point"):
        ok = (r.extra_point_result == "good") if t == "extra_point" else (r.two_point_conv_result == "success")
        return ("good" if ok else "no_good"), 0
    if t == "punt":
        res = "kick_recovered" if (usable and nxt["pos"] == off and pts[dfn] < 6) or pts[off] >= 6 else "kick_received"
        return res, spot_for(off if res == "kick_recovered" else dfn)
    if t == "field_goal":
        if pts[off] == 3:
            return "good", 0
        return "no_good", spot_for(dfn)

    if r.interception == 1:
        res = "interception"
    elif r.fumble_lost == 1:
        res = "fumble_lost"
    elif t == "run":
        res = "run"
    elif t == "pass":
        res = "sack" if r.sack == 1 else "complete" if r.complete_pass == 1 else "incomplete"
    elif t == "qb_kneel":
        res = "kneel"
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
        gain = yl
    elif pts[dfn] == 2:
        gain = yl - 100
    elif usable and nxt["pos"] == off:
        gain = yl - nxt["yl"]
    elif usable:
        gain = yl - (100 - nxt["yl"])
    else:
        gain = 0 if _nan(r.yards_gained) else int(r.yards_gained)
    return res, int(min(max(gain, yl - 99), yl - 1)) if pts[off] < 6 and pts[dfn] != 2 else int(gain)


def process_game(g: pd.DataFrame) -> tuple[dict, list[dict]] | None:
    g = g.sort_index()
    rows = _candidates(g)
    plays = [x for x in rows if x["kind"] == "play"]
    if len(plays) < 50 or plays[0]["phase"] != KICKOFF:
        return None
    first = g.iloc[0]
    meta = dict(
        game_id=first.game_id, season=int(first.season), game_type=first.season_type,
        week=int(first.week), home=_team(first.home_team), away=_team(first.away_team),
        home_final=int(first.home_score), away_final=int(first.away_score),
        opening_kicker=plays[0]["pos"],
    )
    out = []
    timeouts = [3, 3]
    cur_half = 1
    for i, row in enumerate(rows):
        half = 1 if row["qtr"] <= 2 else 2 if row["qtr"] <= 4 else row["qtr"]
        if half != cur_half:
            timeouts = [3, 3] if half == 2 else [2, 2]
            cur_half = half
        nxt_any = rows[i + 1] if i + 1 < len(rows) else None
        nxt_play = next((x for x in rows[i + 1:] if x["kind"] == "play"), None)
        if row["kind"] == "timeout":
            if nxt_play is None or timeouts[row["team"]] <= 0:
                continue
            base = nxt_play
            res = "timeout_offense" if row["team"] == base["pos"] else "timeout_defense"
            st = dict(qtr=row["qtr"], qsec=row["qsec"], phase=base["phase"], pos=base["pos"],
                      down=base["down"], ytg=base["ytg"], yl=base["yl"],
                      score_h=row["pre"][0], score_a=row["pre"][1])
            oc = dict(play_type="timeout", result=res, yards=0)
        else:
            res, yards = _outcome(row, nxt_play)
            st = dict(qtr=row["qtr"], qsec=row["qsec"], phase=row["phase"], pos=row["pos"],
                      down=row["down"], ytg=row["ytg"], yl=row["yl"],
                      score_h=row["pre"][0], score_a=row["pre"][1])
            oc = dict(play_type=row["play_type"], result=res, yards=yards)
        if nxt_any is not None and nxt_any["qtr"] == row["qtr"]:
            el = row["qsec"] - nxt_any["qsec"]
        else:
            el = row["qsec"]
        oc["elapsed"] = int(min(max(el, 0), MAX_ELAPSED))
        st.update(to_h=timeouts[0], to_a=timeouts[1])
        out.append({"game_id": meta["game_id"], **st, **oc})
        if row["kind"] == "timeout":
            timeouts[row["team"]] -= 1
    return meta, out


def state_from_row(meta: dict, row) -> GameState:
    get = row.get if isinstance(row, dict) else (lambda k: getattr(row, k))
    return GameState(
        season=meta["season"], game_type=meta["game_type"], home=meta["home"], away=meta["away"],
        qtr=int(get("qtr")), qsec=int(get("qsec")), phase=int(get("phase")), pos=int(get("pos")),
        down=int(get("down")), ytg=int(get("ytg")), yl=int(get("yl")),
        score=[int(get("score_h")), int(get("score_a"))], timeouts=[int(get("to_h")), int(get("to_a"))],
        opening_kicker=int(meta["opening_kicker"]),
    )


def outcome_from_row(row) -> Outcome:
    get = row.get if isinstance(row, dict) else (lambda k: getattr(row, k))
    return Outcome(get("play_type"), get("result"), int(get("yards")), int(get("elapsed")))


COMPARE = ["qtr", "phase", "pos", "down", "ytg", "yl", "score_h", "score_a"]


def replay_agreement(meta: dict, plays: list[dict]) -> tuple[int, int, dict]:
    """Apply each real outcome to the real state; compare with the real next state."""
    ok = total = 0
    misses: dict[str, int] = {}
    for a, b in zip(plays, plays[1:]):
        if a["qtr"] >= 5:
            continue  # OT bookkeeping is not reconstructible mid-game; covered by unit tests
        s = state_from_row(meta, a)
        n = apply_outcome(s, outcome_from_row(a))
        got = dict(qtr=n.qtr, phase=n.phase, pos=n.pos, down=n.down, ytg=n.ytg, yl=n.yl,
                   score_h=n.score[0], score_a=n.score[1])
        if n.over:
            continue
        total += 1
        bad = [k for k in COMPARE if got[k] != b[k] and not (k in ("down", "ytg", "yl") and b["phase"] != SCRIMMAGE)]
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
    build_ratings(seasons, raw_dir, out_dir)
    top = sorted(misses.items(), key=lambda kv: -kv[1])[:25]
    log.info("engine replay agreement: %.4f over %d transitions", tot_ok / max(tot, 1), tot)
    for k, v in top:
        log.info("  mismatch %-50s %d", k, v)
