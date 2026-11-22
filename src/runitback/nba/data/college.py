"""College basketball (men's ``cbb``, women's ``wcbb``) play-by-play as engine events (C2).

ESPN's college play-by-play is the same source as the NBA's (``runitback.nba.data.preprocess``) with
different habits, and this module turns it into the same ``(state, outcome)`` event tables for
:class:`~runitback.nba.league.League` ``CBB`` / ``WCBB``:

- **Periods.** Men play two 20-minute halves, women four 10-minute quarters, both with 5-minute
  overtimes. The release names the clock ``start_period_seconds_remaining`` or
  ``start_quarter_seconds_remaining`` by season and gender.
- **Free throws** carry no "1 of 2". Consecutive free throws by one team at one clock are a trip
  (a missed first shot's dead-ball team rebound inside it is dropped), and what started it comes from
  the rows around it: a shot at the same clock (and-one, or a shooting foul), a technical foul, or a
  personal foul. A foul in the bonus is recorded as ``fts_2``: the engine decides whether that is a
  1-and-1 (men, 7th to 9th team foul in a half) or two shots (double bonus, women's 5th foul).
- **Fouls are events.** A personal foul by the defense that sends nobody to the line is a ``foul``
  with result ``fts_0``, so the team-foul count (and the bonus it triggers) is part of the game
  state. Offensive fouls (a foul and a turnover by one team at one clock) are turnovers, and
  technical fouls are not team fouls.
- **Possession arrow.** Most games have no opening tip row, and the rows at a period's start or end
  are the arrow's possession; the tip is the first possession's team (as in the NBA), later periods
  and overtimes follow the arrow (``engine.rules``), and a held ball (``Jump Ball won by X`` inside a
  period) is a jump ball that moves the arrow.
- **Media timeouts** (``OfficialTVTimeOut``, no team) are not timeouts of either team and are dropped.
- **Teams.** ESPN lists every opponent of a Division I team. A team is Division I when ESPN files it
  under a conference in the schedule; its key is its school's latest name (the key
  ``runitback.wager.cbb`` uses, so a game's embedding joins the wager model's rows), and every other
  opponent is the pooled team ``Non-D1``. A game of two non-Division I teams is dropped.

``replay_agreement`` is the check the NFL's 99.55% is: apply each real outcome to the real state and
compare with the real next state.
"""
from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..engine.rules import AWAY, HOME, GameState, Outcome, apply_outcome, timeouts_at_period_start
from ..league import CBB, WCBB, League
from ..vocab import FREE_THROW, LIVE, TIP
from .download import numpy_dtypes

log = logging.getLogger(__name__)

RELEASES = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
SOURCES = {  # league -> (play-by-play release, schedule release and file prefix)
    "cbb": ("espn_mens_college_basketball_pbp", "espn_mens_college_basketball_schedules", "mbb_schedule"),
    "wcbb": ("espn_womens_college_basketball_pbp", "espn_womens_college_basketball_schedules", "wbb_schedule"),
}
NON_D1 = "Non-D1"  # the pooled team of every non-Division I opponent (runitback.wager.cbb.data.download)
GAME_TYPES = {2: "REG", 3: "POST"}  # 1 (preseason exhibitions) is dropped
SHOTS = ("JumpShot", "LayUpShot", "DunkShot", "TipShot")
STATE_EXTRA = ["foul_h", "foul_a", "oo", "arrow"]
POSSESSION_KINDS = ("shot", "ft", "to", "foul", "jump")
PBP_COLUMNS = [
    "game_id", "game_play_number", "sequence_number", "season", "season_type", "game_date", "period_number",
    "start_period_seconds_remaining", "start_quarter_seconds_remaining", "type_text", "text", "team_id",
    "home_team_id", "away_team_id", "home_score", "away_score", "score_value", "scoring_play",
    "shooting_play", "home_team_spread",
]
SCHEDULE_COLUMNS = ["game_id", "game_date", "season", "neutral_site", "home_id", "away_id", "home_location",
                    "away_location", "home_conference_id", "away_conference_id"]


# ---------------------------------------------------------------- download

def pbp_url(league: League, season: int) -> str:
    return f"{RELEASES}/{SOURCES[league.name][0]}/play_by_play_{season}.parquet"


def schedule_url(league: League, season: int) -> str:
    _, release, prefix = SOURCES[league.name]
    return f"{RELEASES}/{release}/{prefix}_{season}.parquet"


def download_season(league: League, season: int, raw_dir: Path, force: bool = False) -> tuple[Path, Path] | None:
    """One season's play-by-play and schedule (``pbp_<season>.parquet``, ``schedule_<season>.parquet``).
    `season` is the year it ends in (2025 = 2024-25). Returns None for a season that isn't published."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, url in (("pbp", pbp_url(league, season)), ("schedule", schedule_url(league, season))):
        path = raw_dir / f"{name}_{season}.parquet"
        if force or not path.exists():
            try:
                df = pd.read_parquet(url)
            except Exception as e:  # noqa: BLE001 - a season that hasn't started has no file
                log.warning("could not download the %s %s of %d (%s)", league.name, name, season, e)
                return None
            df.to_parquet(path)
        paths.append(path)
    return paths[0], paths[1]


def load_pbp(season: int, raw_dir: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq
    path = raw_dir / f"pbp_{season}.parquet"
    available = set(pq.read_schema(path).names)
    return numpy_dtypes(pd.read_parquet(path, columns=[c for c in PBP_COLUMNS if c in available]))


def load_schedules(seasons: list[int], raw_dir: Path) -> pd.DataFrame:
    frames = []
    for s in seasons:
        path = raw_dir / f"schedule_{s}.parquet"
        if path.exists():
            d = pd.read_parquet(path)
            d["season"] = s
            frames.append(d[[c for c in SCHEDULE_COLUMNS if c in d.columns]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SCHEDULE_COLUMNS)


# ---------------------------------------------------------------- teams

def team_table(schedules: pd.DataFrame) -> pd.DataFrame:
    """Per game, each side's team key: its school's latest name when ESPN files it under a conference
    that season (Division I), else :data:`NON_D1`. Columns: game_id, home, away, home_id, away_id, neutral."""
    from ...wager import espn  # the wager model's naming, so keys join its rows

    s = schedules.copy()
    s["day"] = pd.to_datetime(s["game_date"], errors="coerce", utc=True).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    sides = pd.concat([
        s.rename(columns={"home_id": "id", "home_location": "school", "home_conference_id": "conference_id"})
         [["id", "school", "conference_id", "season", "day"]],
        s.rename(columns={"away_id": "id", "away_location": "school", "away_conference_id": "conference_id"})
         [["id", "school", "conference_id", "season", "day"]],
    ], ignore_index=True)
    names = espn.latest_names(sides[sides.conference_id.notna()], "school")

    def key(ids: pd.Series, conference: pd.Series) -> pd.Series:
        return ids.map(names).where(conference.notna(), NON_D1).fillna(NON_D1)

    out = pd.DataFrame({
        "game_id": s["game_id"].astype(str),
        "home": key(s["home_id"], s["home_conference_id"]), "away": key(s["away_id"], s["away_conference_id"]),
        "home_id": s["home_id"], "away_id": s["away_id"],
        "neutral": s["neutral_site"].fillna(False).astype(bool) if "neutral_site" in s else False,
    })
    return out.drop_duplicates("game_id")


# ---------------------------------------------------------------- one game

def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _clock_column(g: pd.DataFrame) -> str:
    return "start_period_seconds_remaining" if "start_period_seconds_remaining" in g else "start_quarter_seconds_remaining"


def _actions(g: pd.DataFrame, lg: League) -> list[dict]:
    """Rows that matter, in game order, tagged with what kind of action each is."""
    home_id, away_id = g.home_team_id.iloc[0], g.away_team_id.iloc[0]
    order = "game_play_number" if "game_play_number" in g else "sequence_number"
    g = g.assign(clock_s=np.ceil(g[_clock_column(g)].astype(float)))
    g = g.sort_values(["period_number", "clock_s", order], ascending=[True, False, True], kind="stable")
    out = []
    last = [0, 0]  # the score before each row: a basket the rows don't flag as scoring still moves it
    for r in g.itertuples(index=False):
        before, last = last, [int(r.home_score), int(r.away_score)]
        if _nan(r.period_number) or _nan(r.clock_s):
            continue
        tt = str(r.type_text or "")
        text = str(r.text or "").lower()
        side = HOME if r.team_id == home_id else AWAY if r.team_id == away_id else None
        gained = last[side] - before[side] if side is not None else 0
        a = None
        if "FreeThrow" in tt or tt.startswith("Free Throw"):
            a = dict(kind="ft", made=bool(r.scoring_play) or " made " in f" {text} " or gained == 1)
        elif "Turnover" in tt and tt != "No Turnover":
            a = dict(kind="to")
        elif tt in ("RegularTimeOut", "ShortTimeOut", "Full Timeout", "Timeout"):
            a = dict(kind="timeout")  # OfficialTVTimeOut has no team and is dropped below
        elif tt == "Jumpball":
            if text.startswith("jump ball won by") or text.startswith("jumpball won by") or "possession, arrow" in text:
                a = dict(kind="jump")
        elif tt in ("Offensive Rebound", "Defensive Rebound"):
            a = dict(kind="reb", off=tt.startswith("Offensive"))
        elif tt == "PersonalFoul":
            a = dict(kind="pf")
        elif tt == "Technical Foul":
            a = dict(kind="techfoul")
        elif bool(r.shooting_play) and "Foul" not in tt:
            three = "three point" in text or (not _nan(r.score_value) and int(r.score_value) == 3)
            made = (bool(r.scoring_play) and not _nan(r.score_value) and int(r.score_value) > 0) or gained in (2, 3)
            a = dict(kind="shot", pts=3 if three else 2, made=made)
        if a is None or side is None:
            continue
        a.update(side=side, qtr=int(r.period_number), clock=int(r.clock_s),
                 post=(int(r.home_score), int(r.away_score)))
        out.append(a)
    return out


def _same_moment(a: dict, b: dict) -> bool:
    return a["qtr"] == b["qtr"] and a["clock"] == b["clock"]


def _trip(acts: list[dict], i: int) -> tuple[list[dict], int]:
    """The free throws of the trip that starts at acts[i] and the index after it. A trip is
    consecutive free throws by one team at one clock; team rebounds between them (the dead ball after
    a missed first shot) belong to it."""
    a = acts[i]
    trip, j, k = [a], i + 1, i + 1
    while k < len(acts):
        b = acts[k]
        if b["kind"] == "reb" and _same_moment(a, b):
            k += 1
        elif b["kind"] == "ft" and b["side"] == a["side"] and _same_moment(a, b):
            trip.append(b)
            k += 1
            j = k
        else:
            break
    return trip, j


def _events(acts: list[dict], lg: League) -> list[dict]:
    """Merge the rows into events: shots (with the free throws that make them fouled), `foul` events
    (with the trip they sent a team to the line for), technical free throws, turnovers, timeouts, held
    balls, and a tip-off at the start and a run-out at the end of each period."""
    events: list[dict] = []
    i = 0
    while i < len(acts):
        a = acts[i]
        i += 1
        k = a["kind"]
        if k == "reb":
            if events:
                events[-1].setdefault("rebounds", []).append(a)
            continue
        if k == "pf":
            # a foul by one team and a turnover by the same team at one clock: an offensive foul
            if any(b["kind"] == "to" and b["side"] == a["side"] and _same_moment(a, b) for b in acts[i:i + 6]):
                events.append(dict(kind="offfoul", side=a["side"], qtr=a["qtr"], clock=a["clock"]))
            else:
                events.append(dict(kind="foul", side=1 - a["side"], fouler=a["side"], n=0, qtr=a["qtr"],
                                   clock=a["clock"], post=a["post"]))
            continue
        if k == "techfoul":
            events.append(dict(kind="techfoul", side=a["side"], qtr=a["qtr"], clock=a["clock"]))
            continue
        if k == "ft":
            trip, i = _trip(acts, i - 1)
            side = a["side"]
            # what the trip follows, among the events at this clock
            back = [e for e in reversed(events) if e["qtr"] == a["qtr"] and e["clock"] == a["clock"]]
            tech = next((e for e in back if e["kind"] == "techfoul"), None)
            shot = next((e for e in back if e["kind"] == "shot" and e["side"] == side and "fouled" not in e), None)
            n = len(trip)
            if tech is not None:
                events.remove(tech)
                events += [dict(kind="tech", side=side, made=t["made"], qtr=t["qtr"], clock=t["clock"],
                                post=t["post"]) for t in trip]
                continue
            if shot is not None and ((shot["made"] and n == 1) or (not shot["made"] and n == shot["pts"])):
                shot["fouled"] = n
                for e in back:  # the foul that sent them to the line is part of the shot
                    if e["kind"] == "foul" and e["fouler"] == 1 - side and e["n"] == 0:
                        e["absorbed"] = True
                        break
                foul = None
            else:
                foul = next((e for e in back if e["kind"] == "foul" and e["fouler"] == 1 - side and e["n"] == 0
                             and not e.get("absorbed")), None)
                if foul is None:
                    foul = dict(kind="foul", side=side, fouler=1 - side, n=0, qtr=a["qtr"], clock=a["clock"],
                                extra=True)
                    events.append(foul)
                foul["n"] = n
                foul["first_made"] = trip[0]["made"]
            for kk, t in enumerate(trip, 1):
                events.append(dict(kind="ft", side=side, k=kk, n=n, made=t["made"], qtr=t["qtr"],
                                   clock=t["clock"], post=t["post"], shot=shot is not None and "fouled" in shot))
            continue
        events.append(a)
    events = [e for e in events if not e.get("absorbed") and e["kind"] not in ("techfoul", "offfoul")]
    # a toss at a period's end is the next period's: the arrow's possession, shown at its start
    quarters = {e["qtr"] for e in events}
    for e in events:
        if e["kind"] == "jump" and e["clock"] <= 0 and e["qtr"] + 1 in quarters:
            e["qtr"], e["clock"] = e["qtr"] + 1, lg.period_length(e["qtr"] + 1)
    # every period starts with a toss (the opening tip, or the possession arrow's); most rows don't show
    # it, and the team that has the ball first is the one the toss gave it to
    out: list[dict] = []
    for q in sorted(quarters):
        ev = [e for e in events if e["qtr"] == q]
        length = lg.period_length(q)
        first_jump = next((i for i, e in enumerate(ev) if e["kind"] == "jump"), None)
        first_poss = next((i for i, e in enumerate(ev) if e["kind"] in POSSESSION_KINDS), None)
        if first_jump is not None and first_jump == first_poss:
            ev[first_jump]["tip"] = True
            ev = [ev[first_jump]] + [e for i, e in enumerate(ev) if i != first_jump and
                                     not (e["kind"] == "timeout" and i < first_jump)]
        elif first_poss is not None:
            tip = dict(kind="jump", tip=True, side=ev[first_poss]["side"], qtr=q, clock=length)
            ev = [tip] + [e for i, e in enumerate(ev) if e["kind"] != "timeout" or i > first_poss]
        # other tosses at a period's start or end are not events
        ev = [e for e in ev if e["kind"] != "jump" or e.get("tip") or 0 < e["clock"] < length]
        if ev and ev[-1]["clock"] > 0:
            ev.append(dict(kind="run_out", side=None, qtr=q, clock=0))
        out += ev
    return _keep_the_ball(out)


def _keep_the_ball(events: list[dict]) -> list[dict]:
    """Intentional and flagrant fouls give two shots *and the ball*: a trip after a foul whose last shot
    is made and whose team acts next is a foul with no free throws followed by two technical-style shots
    (which keep possession). The foul still counts toward the bonus."""
    for i, e in enumerate(events):
        if e["kind"] != "foul" or e["n"] == 0:
            continue
        fts = events[i + 1:i + 1 + e["n"]]
        if len(fts) != e["n"] or any(f["kind"] != "ft" or f["side"] != e["side"] for f in fts) or not fts[-1]["made"]:
            continue
        if _next_actor(events, i + e["n"]) == e["side"]:
            for f in fts:
                f.update(kind="tech")
            e["n"] = 0
    return events


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
    return rebs[0]["side"] if rebs else None


def state_from_row(meta: dict, row, lg: League) -> GameState:
    get = row.get if isinstance(row, dict) else (lambda k: getattr(row, k))
    s = GameState(
        season=meta["season"], game_type=meta["game_type"], home=meta["home"], away=meta["away"], league=lg,
        qtr=int(get("qtr")), qsec=int(get("qsec")), phase=int(get("phase")), pos=int(get("pos")),
        fts_left=int(get("fts_left")), fts_total=int(get("fts_left")),
        score=[int(get("score_h")), int(get("score_a"))], timeouts=[int(get("to_h")), int(get("to_a"))],
        tip_winner=int(meta["tip_winner"]),
    )
    s.fouls = [int(get("foul_h")), int(get("foul_a"))]
    s.one_and_one = bool(get("oo"))
    arrow = int(get("arrow"))
    s.arrow = None if arrow < 0 else arrow
    return s


def outcome_from_row(row) -> Outcome:
    get = row.get if isinstance(row, dict) else (lambda k: getattr(row, k))
    return Outcome(get("play_type"), get("result"), int(get("elapsed")))


def process_game(g: pd.DataFrame, lg: League, keys: tuple[str, str, bool] | None = None,
                 events: list[dict] | None = None) -> tuple[dict, list[dict]] | None:
    """One game's meta and event rows, or None when it can't be used (a non-Division I pair, a
    preseason game, too few events). `keys` = (home key, away key, neutral site). `events` are the game's
    events when another reader made them (the WNBA's, ``runitback.nba.data.wnba``), else this module's
    from the college rows. Any league with a bonus works: the rows track team fouls with the league's
    rules (``Bonus``) independently of the engine, which the replay check compares."""
    first = g.iloc[0]
    home, away, neutral = keys if keys is not None else (str(first.home_team_id), str(first.away_team_id), False)
    game_type = GAME_TYPES.get(int(first.season_type))
    if (home == NON_D1 and away == NON_D1) or game_type is None:
        return None
    if events is None:
        events = _events(_actions(g, lg), lg)
    if len(events) < lg.min_events or events[0]["kind"] != "jump":
        return None
    last = g.loc[g.period_number == g.period_number.max()]
    meta = dict(game_id=str(first.game_id), season=int(first.season), game_type=game_type,
                game_date=str(first.game_date)[:10], home=home, away=away, neutral=bool(neutral),
                home_final=int(last.home_score.max()), away_final=int(last.away_score.max()),
                tip_winner=int(events[0]["side"]))
    spread = getattr(first, "home_team_spread", None)
    meta["home_spread"] = None if _nan(spread) else float(spread)

    bonus = lg.bonus
    rows: list[dict] = []
    score = [0, 0]
    timeouts = [lg.timeouts, lg.timeouts]
    fouls = [0, 0]
    trip_left = 0
    one_and_one = False
    eng: GameState | None = None
    cur_q, qsec = 0, 0
    poss = [0, 0]
    for i, e in enumerate(events):
        q = e["qtr"]
        if q != cur_q:
            if eng is not None and not eng.over:
                poss[eng.pos] += 1
            cur_q, qsec, trip_left, one_and_one = q, lg.period_length(q), 0, False
            if q > 1:
                timeouts = timeouts_at_period_start(timeouts, q, lg)
                if bonus.resets(q, lg.regulation_periods):
                    fouls = [0, 0]
        ot = lg.is_overtime(q)
        prev_pos = eng.pos if eng is not None else HOME  # who the engine says has the ball
        nxt = _next_actor(events, i)
        k = e["kind"]
        phase, pos, fts_left = (FREE_THROW if trip_left > 0 else LIVE), prev_pos, trip_left
        oo = False
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
            pos, phase = e["side"], FREE_THROW
            oo = bool(e.get("oo")) and e["k"] == 1
            fts_left = 2 if oo else e["n"] - e["k"] + 1
            if e["made"]:
                res = "made"
            elif e["k"] < e["n"] and not (e.get("oo") and e["k"] == 1):
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
            n = e["n"]
            counted = fouls[e["fouler"]] + 1
            if n == 0 and e["fouler"] == prev_pos:
                # a foul by the team with the ball and no free throws: it loses the ball, as on an offensive foul
                e["control"] = True
                pos = e["fouler"]
                oc = ("turnover", "turnover")
                res = None
            elif n == 0:
                res = "fts_0"
            elif (bonus.level(counted, ot) == 1 and ((n == 1 and not e["first_made"])
                                                  or (n == 2 and e["first_made"]))):
                res = "fts_2"  # a 1-and-1: the first shot made is followed by the second, a miss is not
                e["oo"] = True
            else:
                res = f"fts_{min(n, 3)}"
            if res is not None:
                oc = ("foul", res)
        elif k == "timeout":
            pos = prev_pos if trip_left > 0 or nxt is None else nxt
            if timeouts[e["side"]] <= 0:
                continue
            oc = ("timeout", "timeout_offense" if e["side"] == pos else "timeout_defense")
        else:  # run_out
            oc = ("run_out", "end")
        elapsed = qsec if k == "run_out" else qsec - e["clock"]
        arrow = -1 if eng is None or eng.arrow is None else int(eng.arrow)
        row = dict(game_id=meta["game_id"], qtr=q, qsec=qsec, phase=phase, pos=pos, fts_left=fts_left,
                   score_h=score[0], score_a=score[1], to_h=timeouts[0], to_a=timeouts[1],
                   foul_h=fouls[0], foul_a=fouls[1], oo=int(oo), arrow=arrow,
                   play_type=oc[0], result=oc[1], elapsed=int(min(max(elapsed, 0), lg.max_elapsed)))
        rows.append(row)
        # bookkeeping for the next event
        state = state_from_row(meta, row, lg)
        eng = apply_outcome(state, outcome_from_row(row))
        if eng.qtr == q and eng.pos != pos and not eng.over:
            poss[pos] += 1
        if k == "timeout":
            timeouts[e["side"]] -= 1
        if k == "foul" and not e.get("control"):
            fouls[e["fouler"]] = bonus.counted(fouls[e["fouler"]] + 1, e["clock"], ot)
            # the free throws that follow know whether they are a 1-and-1
            for later in events[i + 1:i + 1 + e["n"]]:
                if later["kind"] == "ft":
                    later["oo"] = bool(e.get("oo"))
        elif k == "shot" and "fouled" in e:
            fouls[1 - e["side"]] = bonus.counted(fouls[1 - e["side"]] + 1, e["clock"], ot)
        if "post" in e:
            score = list(e["post"])
        if k == "shot" and "fouled" in e:
            trip_left = e["fouled"]
        elif k == "foul":
            trip_left = 2 if e.get("oo") else e["n"]
            if e.get("control"):
                trip_left = 0
        elif k == "ft":
            trip_left = e["n"] - e["k"]
            if e.get("oo") and e["k"] == 1 and not e["made"]:
                trip_left = 0
        elif k in ("to", "jump"):
            trip_left = 0
        qsec = 0 if k == "run_out" else e["clock"]
    if eng is not None and not eng.over:
        poss[eng.pos] += 1
    meta.update(poss_h=poss[0], poss_a=poss[1])
    return meta, rows


COMPARE = ["qtr", "phase", "pos", "fts_left", "score_h", "score_a"]


def replay_agreement(meta: dict, rows: list[dict], lg: League) -> tuple[int, int, dict]:
    """Apply each real outcome to the real state; compare with the real next state (period, phase,
    possession, free throws left, score, and the team fouls)."""
    cols = COMPARE + ["foul_h", "foul_a"]
    ok = total = 0
    misses: dict[str, int] = {}
    for a, b in zip(rows, rows[1:]):
        n = apply_outcome(state_from_row(meta, a, lg), outcome_from_row(a))
        if n.over:
            continue
        got = dict(qtr=n.qtr, phase=n.phase, pos=n.pos, fts_left=n.fts_left, score_h=n.score[0],
                   score_a=n.score[1], foul_h=n.fouls[0], foul_a=n.fouls[1])
        total += 1
        bad = [c for c in cols if got[c] != b[c] and not (c == "pos" and b["phase"] == TIP)]
        if not bad:
            ok += 1
        else:
            key = f"{a['play_type']}/{a['result']}:{','.join(bad)}"
            misses[key] = misses.get(key, 0) + 1
    return ok, total, misses


# ---------------------------------------------------------------- a season, all seasons

def _season(args) -> tuple[list[dict], list[dict], dict, int, int]:
    league_name, season, raw_dir, teams, min_agreement = args
    lg = {"cbb": CBB, "wcbb": WCBB}[league_name]
    raw_dir = Path(raw_dir)
    df = load_pbp(season, raw_dir)
    keyed = teams.set_index("game_id")
    metas, rows_out, misses = [], [], {}
    ok_all = tot_all = n_games = 0
    for gid, g in df.groupby("game_id", sort=False):
        n_games += 1
        gid = str(gid)
        if gid not in keyed.index:
            continue
        t = keyed.loc[gid]
        res = process_game(g, lg, (t.home, t.away, bool(t.neutral)))
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
    log.info("%s season %d: kept %d/%d games, replay agreement %.4f", lg.name, season, len(metas), n_games,
             ok_all / max(tot_all, 1))
    return metas, rows_out, misses, ok_all, tot_all


def preprocess(league: League, seasons: list[int], raw_dir: Path, out_dir: Path, min_agreement: float = 0.9,
               jobs: int = 1) -> dict:
    """Process `seasons` to games / plays / ratings parquet files in `out_dir`, and return the replay
    statistics (``replay.json`` has the same): agreement per season and the commonest mismatches."""
    out_dir.mkdir(parents=True, exist_ok=True)
    present = [s for s in seasons if (raw_dir / f"pbp_{s}.parquet").exists() and (raw_dir / f"schedule_{s}.parquet").exists()]
    for s in set(seasons) - set(present):
        log.warning("season %d: play-by-play or schedule not found (download it first); skipped", s)
    teams = team_table(load_schedules(present, raw_dir))
    args = [(league.name, s, str(raw_dir), teams, min_agreement) for s in present]
    if jobs > 1 and len(args) > 1:
        with ProcessPoolExecutor(min(jobs, len(args))) as pool:
            results = list(pool.map(_season, args))
    else:
        results = [_season(a) for a in args]
    metas = [m for r in results for m in r[0]]
    plays = pd.DataFrame([row for r in results for row in r[1]])
    games = pd.DataFrame(metas)
    misses: dict[str, int] = {}
    for r in results:
        for k, v in r[2].items():
            misses[k] = misses.get(k, 0) + v
    games.to_parquet(out_dir / "games.parquet")
    plays.to_parquet(out_dir / "plays.parquet")
    teams.to_parquet(out_dir / "teams.parquet")
    from .ratings import build_ratings
    build_ratings(games, out_dir, league)
    stats = replay_stats(league, present, results, games, misses)
    (out_dir / "replay.json").write_text(json.dumps(stats, indent=1))
    log.info("replay agreement %.4f over %d transitions", stats["agreement"], stats["transitions"])
    return stats


def replay_stats(league: League, seasons: list[int], results: list, games: pd.DataFrame, misses: dict) -> dict:
    ok = sum(r[3] for r in results)
    total = sum(r[4] for r in results)
    per_season = {int(s): dict(agreement=round(r[3] / max(r[4], 1), 5), transitions=int(r[4]),
                               games=int(len(r[0]))) for s, r in zip(seasons, results)}
    return dict(
        league=league.name, agreement=round(ok / max(total, 1), 5), transitions=int(total), games=int(len(games)),
        events_per_game=dict(mean=float(games.n_events.mean()), p99=float(games.n_events.quantile(0.99)),
                             max=int(games.n_events.max())) if len(games) else {},
        per_season=per_season,
        top_mismatches=dict(sorted(misses.items(), key=lambda kv: -kv[1])[:25]),
    )
