"""Replay real NHL games through the rules engine.

`game_events` turns one game of sportsdataverse `nhl_pbp_full` play-by-play into engine outcomes:

- faceoffs, shot attempts (goal / saved / missed / blocked, credited to the shooting team), hits,
  giveaways, takeaways, delayed-penalty calls and stoppages (by reason) map one to one;
- goalie pulls and returns come from the line changes: the goalie leaves or rejoins the ice while
  the period is in progress (a goalie swap at a single change is not a pull);
- penalties are labeled by what they do to the strength. Penalties assessed at the same stoppage
  cancel by kind (majors, then double minors, then minors) and the cancelled ones are
  `coincidental`, except one minor each at full 5-on-5 strength with nothing else on the penalty
  clocks, which is played 4-on-4 (NHL rule 19.1). Misconducts never change the strength. A
  penalty-shot call (`ps-...`, 0 minutes) is moved before the attempt it awards, which the
  data lists first;
- shootout attempts are listed with each team's goals after its misses, so the order is rebuilt:
  the teams alternate from the first listed shooter, and each team's goals are placed so the
  shootout is decided exactly on the last attempt. When several orders fit (equal goals in the
  first three rounds), the first found is used: the data cannot tell them apart.

`replay_game` then applies each outcome to the engine and compares the engine's next state with
what the data says independently: the score after the event (from the goal rows), the period,
the skaters on the ice for each team and whether each net is empty (both from the shift-based
on-ice columns at the next event), and whether the next real event is legal in the engine's state.
The penalty box is never reset from the data, so an engine error persists until the penalty
would have ended; only the score is re-synced after a mismatch.
"""
from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from ..vocab import FACEOFF, LIVE, SHOOTOUT, TEAMS, canon_team
from .rules import (
    AWAY,
    HOME,
    GameState,
    Outcome,
    apply_outcome,
    is_legal,
    new_game,
    period_length,
    run_clock,
    shootout_decided,
)

log = logging.getLogger(__name__)

RELEASE_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
               "nhl_pbp_full/play_by_play_{file}.parquet")
COLUMNS = [
    "event_type", "secondary_type", "event_team_type", "period", "period_type",
    "period_seconds", "home_score", "away_score", "penalty_severity", "penalty_minutes",
    "home_skaters", "away_skaters", "home_goalie", "away_goalie", "game_id", "season",
    "season_type", "home_abbr", "away_abbr", "event_idx", "reason", "event_player_1_id",
    "home_on_1_id", "away_on_1_id",
]
GAME_TYPES = {"R": "REG", "P": "POST"}
SHOTS = {"GOAL": "goal", "SHOT": "saved", "MISSED_SHOT": "missed", "BLOCKED_SHOT": "blocked",
         "FAILED_SHOT_ATTEMPT": "missed"}
PLAIN = {"HIT": "hit", "GIVEAWAY": "giveaway", "TAKEAWAY": "takeaway",
         "DELAYED_PENALTY": "delayed_penalty"}
SKIP = ("PERIOD_START", "GAME_END", "SHOOTOUT_COMPLETE", "EARLY_INT_START", "EARLY_INT_END")
STOP_REASONS = {
    "goalie_freeze": ("goalie-stopped", "puck-frozen", "goalie-puck-frozen", "skater-puck-frozen"),
    "icing": ("icing",),
    "offside": ("offside",),
    "out_of_play": ("puck-in-netting", "puck-in-crowd", "puck-in-benches", "puck-in-penalty-benches"),
    "timeout": ("tv-timeout", "home-timeout", "visitor-timeout"),
}
STRENGTH = ("major", "double_minor", "minor")
MAX_GOALIE_MISSING = 0.15  # share of a game's events without a team's goalie on the ice  # cancellation order for coincidental penalties


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or x is pd.NA


def season_of(value) -> int:
    """`season` column value ("20242025" or 20242025) -> the year the season ends in."""
    return int(str(int(float(value)) if not isinstance(value, str) else value)[-4:])


def stop_reason(reason) -> str:
    text = "" if _nan(reason) else str(reason)
    for key, needles in STOP_REASONS.items():
        if any(n in text for n in needles):
            return key
    return "other"


def penalty_kind(severity, minutes, secondary) -> str:
    sec = "" if _nan(secondary) else str(secondary).lower()
    mins = 0.0 if _nan(minutes) else float(minutes)
    sev = "" if _nan(severity) else str(severity).lower()
    if sec.startswith("ps-") or (mins == 0 and "penalty shot" in sec):
        return "penalty_shot"
    if sev in ("major", "match") or mins == 5:
        return "major"
    if sev == "misconduct" or mins >= 10:
        return "misconduct"
    if mins == 4:
        return "double_minor"
    if mins == 0:
        return "misconduct"  # no time served: no effect on the strength
    return "minor"


# ---------------------------------------------------------------- data loading

REQUIRED = ("home_skaters", "away_skaters", "home_goalie", "away_goalie")


def load_season_file(path: Path) -> pd.DataFrame:
    """The columns the replay needs. Files without on-ice columns (2009-10) are rejected."""
    import pyarrow.parquet as pq
    available = set(pq.read_schema(path).names)
    missing = [c for c in REQUIRED if c not in available]
    if missing:
        raise ValueError(f"{path.name} has no on-ice columns ({', '.join(missing)})")
    return pd.read_parquet(path, columns=[c for c in COLUMNS if c in available])


def download_file(file: int, raw_dir: Path, force: bool = False) -> Path:
    """Download one `nhl_pbp_full` release file. Its season is in the `season` column: the file
    names do not follow one convention (play_by_play_2011 is 2010-11, play_by_play_2016 is 2016-17)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"play_by_play_{file}.parquet"
    if not path.exists() or force:
        pd.read_parquet(RELEASE_URL.format(file=file)).to_parquet(path)
    return path


def split_games(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a season file into games. Line-change rows carry no game id, so games are split where
    `event_idx` starts again, and the game id comes from the game's other rows."""
    key = (df.event_idx.diff().fillna(-1) < 0).cumsum()
    return [g for _, g in df.groupby(key, sort=False)]


# ---------------------------------------------------------------- events

@dataclass
class Event:
    play_type: str
    result: str
    side: int | None
    period: int
    t: int                                  # seconds into the period
    score: tuple[int, int] | None = None    # score after the event (data)
    skaters: tuple[int, int] | None = None  # on ice when the event happened (data)
    empty_net: tuple[bool, bool] | None = None
    player: int | None = None               # the penalized player (penalties)
    implied: bool = False                   # added by `repair_events`, not in the data


def _side(r) -> int | None:
    return HOME if r.event_team_type == "home" else AWAY if r.event_team_type == "away" else None


def game_events(g: pd.DataFrame) -> tuple[dict, list[Event]] | None:
    first = g[g.game_id.notna()].iloc[0] if g.game_id.notna().any() else None
    if first is None:
        return None
    home, away = canon_team(str(first.home_abbr)), canon_team(str(first.away_abbr))
    game_type = GAME_TYPES.get(str(first.season_type))
    if home not in TEAMS or away not in TEAMS or game_type is None:
        return None
    meta = dict(game_id=str(int(float(first.game_id))), season=season_of(first.season),
                game_type=game_type, home=home, away=away)
    # The on-ice columns come from shift charts. Some games have none (no line changes, no on-ice
    # ids: 174 games of 2019-20), and in some a goalie is missing from them for much of the game
    # (a real empty net is a few percent of a game's events). Those games' on-ice columns say
    # nothing reliable about the strength or empty nets: they are not compared, and no goalie
    # pulls are read from them.
    plays = g[~g.event_type.isin(("CHANGE", "PERIOD_END", *SKIP)) & (g.period_type != "SHOOTOUT")]
    on_ice = "home_on_1_id" not in g or (plays.home_on_1_id.notna().mean() >= 0.5
                                          and plays.away_on_1_id.notna().mean() >= 0.5)
    on_ice = on_ice and max(plays.home_goalie.isna().mean(), plays.away_goalie.isna().mean()) \
        <= MAX_GOALIE_MISSING
    meta["on_ice"] = bool(on_ice)

    events: list[Event] = []
    shootout: list[tuple[int, str]] = []
    score = (0, 0)
    in_play = False           # between a period's first faceoff and its end
    goalie_in = [True, True]
    for r in g.itertuples(index=False):
        et, period, t = r.event_type, int(r.period), int(r.period_seconds)
        if et in SKIP:
            continue
        if r.period_type == "SHOOTOUT":
            if et in SHOTS and _side(r) is not None:
                shootout.append((_side(r), "goal" if et == "GOAL" else
                                 "saved" if et == "SHOT" else "missed"))
            continue
        if et == "CHANGE":
            if not on_ice or not in_play or t >= period_length(period, game_type):
                continue
            for side, col in ((HOME, r.home_goalie), (AWAY, r.away_goalie)):
                now_in = not _nan(col)
                if now_in != goalie_in[side]:
                    goalie_in[side] = now_in
                    kind = "goalie_return" if now_in else "goalie_pull"
                    events.append(Event(kind, kind, side, period, t, score))
            continue
        known = on_ice and not (_nan(r.home_skaters) or _nan(r.away_skaters))
        obs = dict(skaters=(int(r.home_skaters), int(r.away_skaters)) if known else None,
                   empty_net=(_nan(r.home_goalie), _nan(r.away_goalie)) if known else None)
        if et == "PERIOD_END":
            if not any(e.period == period for e in events):
                continue  # a period end listed before the period has started
            events.append(Event("period_end", "end", None, period, t, score))
            in_play, goalie_in = False, [True, True]
            continue
        side = _side(r)
        if et == "FACEOFF":
            if not in_play:
                in_play, goalie_in = True, [True, True]
            ev = Event("faceoff", "won", side, period, t)
        elif et in SHOTS:
            if et == "GOAL" and not (_nan(r.home_score) or _nan(r.away_score)):
                score = (int(r.home_score), int(r.away_score))
            if et == "GOAL" and period > 3:
                in_play = False  # sudden death: everyone leaves the ice
            ev = Event("shot", SHOTS[et], side, period, t)
        elif et in PLAIN:
            ev = Event(PLAIN[et], PLAIN[et], side, period, t)
        elif et == "PENALTY":
            ev = Event("penalty", penalty_kind(r.penalty_severity, r.penalty_minutes, r.secondary_type),
                       side, period, t)
            pid = getattr(r, "event_player_1_id", None)
            ev.player = None if _nan(pid) else int(float(pid))
        elif et == "STOP":
            ev = Event("stoppage", stop_reason(getattr(r, "reason", None)), None, period, t)
        else:
            continue
        if ev.side is None and ev.play_type not in ("stoppage",):
            continue
        ev.score, ev.skaters, ev.empty_net = score, obs["skaters"], obs["empty_net"]
        events.append(ev)

    events = _drop_goalie_swaps(events)
    events = _merge_same_player_minors(events)
    events = _order_penalty_shots(events)
    if shootout:
        ordered = order_shootout(shootout)
        if ordered is None:
            meta["shootout_unordered"] = True
        else:
            p = events[-1].period + 1 if events else 5
            events += [Event("shootout", res, side, p, 0, score) for side, res in ordered]
    meta["home_goals"], meta["away_goals"] = score  # before any shootout
    return meta, events


def _drop_goalie_swaps(events: list[Event]) -> list[Event]:
    """A goalie leaving and a goalie coming on at the same moment is a swap, not an empty net."""
    out: list[Event] = []
    for e in events:
        if out and e.play_type == "goalie_return" and out[-1].play_type == "goalie_pull" \
                and out[-1].side == e.side and out[-1].period == e.period and out[-1].t == e.t:
            out.pop()
            continue
        out.append(e)
    return out


def _merge_same_player_minors(events: list[Event]) -> list[Event]:
    """Two minors to one player at one stoppage are served back to back: a double minor."""
    out: list[Event] = []
    for e in events:
        twin = next((x for x in reversed(out) if x.play_type == "penalty" and x.result == "minor"
                     and (x.period, x.t, x.side) == (e.period, e.t, e.side) and x.player is not None
                     and x.player == e.player), None) \
            if e.play_type == "penalty" and e.result == "minor" else None
        if twin is not None:
            twin.result = "double_minor"
            continue
        out.append(e)
    return out


def _order_penalty_shots(events: list[Event]) -> list[Event]:
    """Put each penalty-shot call before the attempt it awards (listed first, at the same time).
    A call with no attempt by the other team at that time is dropped."""
    out = list(events)
    for e in [e for e in events if e.play_type == "penalty" and e.result == "penalty_shot"]:
        i = out.index(e)
        same = [j for j, x in enumerate(out) if x.play_type == "shot" and x.result != "blocked"
                and x.side == 1 - e.side and x.period == e.period and x.t == e.t]
        out.pop(i)
        if not same:
            continue
        j = max([k for k in same if k < i], default=min(same) - 1 if min(same) > i else min(same))
        e.score = out[j - 1].score if j > 0 else (0, 0)  # the call comes before the attempt
        out.insert(j, e)
    return out


def order_shootout(attempts: list[tuple[int, str]]) -> list[tuple[int, str]] | None:
    """Rebuild the shootout order: the teams alternate (from the first listed shooter if the
    counts allow it), and each team's goals are placed so the shootout is decided exactly on the
    last attempt. None if no order fits."""
    for first in dict.fromkeys((attempts[0][0], 1 - attempts[0][0])):
        order = _order_shootout(attempts, first)
        if order is not None:
            return order
    return None


def _order_shootout(attempts: list[tuple[int, str]], first: int) -> list[tuple[int, str]] | None:
    per = {s: [res for side, res in attempts if side == s] for s in (HOME, AWAY)}
    n = {s: len(per[s]) for s in per}
    goals = {s: sum(r == "goal" for r in per[s]) for s in per}
    if not (n[first] == n[1 - first] or n[first] == n[1 - first] + 1):
        return None

    def sequences(side):
        misses = [r for r in per[side] if r != "goal"]
        for pos in itertools.combinations(range(n[side]), goals[side]):
            it = iter(misses)
            yield [("goal" if i in pos else next(it)) for i in range(n[side])]

    def listed_first(side):  # try the listed order first
        seqs = list(sequences(side))
        return sorted(seqs, key=lambda q: q != per[side])

    for sh in listed_first(HOME):
        for sa in listed_first(AWAY):
            seq = {HOME: sh, AWAY: sa}
            order = []
            for k in range(max(n.values())):
                for s in (first, 1 - first):
                    if k < n[s]:
                        order.append((s, seq[s][k]))
            shots, gl = [0, 0], [0, 0]
            ok = True
            for i, (s, res) in enumerate(order):
                shots[s] += 1
                gl[s] += res == "goal"
                if shootout_decided(shots, gl) != (i == len(order) - 1):
                    ok = False
                    break
            if ok:
                return order
    return None


# ---------------------------------------------------------------- penalty labels

def label_penalties(group: list[Event], s: GameState) -> None:
    """Label penalties assessed at one stoppage by their effect on the strength (in place), per
    NHL rules 19.1 and 19.5: majors cancel in pairs; one minor each at full 5-on-5 strength with
    no other penalties on the clocks is played 4-on-4 (`matched_minor`); otherwise minor time
    cancels by length (a minor against half a double minor, leaving a two-minute minor)."""
    def of(side, kind):
        return [e for e in group if e.side == side and e.result == kind]
    maj_h, maj_a = of(HOME, "major"), of(AWAY, "major")
    for e in maj_h[:len(maj_a)] + maj_a[:len(maj_h)]:
        e.result = "coincidental"
    minors = {side: of(side, "minor") for side in (HOME, AWAY)}
    doubles = {side: of(side, "double_minor") for side in (HOME, AWAY)}
    if len(group) == 2 and all(len(minors[side]) == 1 for side in (HOME, AWAY)) \
            and s.base_skaters == 5 and not s.pens[HOME] and not s.pens[AWAY]:
        for side in (HOME, AWAY):
            minors[side][0].result = "matched_minor"
        return

    def cancel(a, b):
        k = min(len(a), len(b))
        for e in a[:k] + b[:k]:
            e.result = "coincidental"
        return a[k:], b[k:]
    doubles[HOME], doubles[AWAY] = cancel(doubles[HOME], doubles[AWAY])
    minors[HOME], minors[AWAY] = cancel(minors[HOME], minors[AWAY])
    for d_side in (HOME, AWAY):  # at most one team has double minors left over against minors
        d, m = doubles[d_side], minors[1 - d_side]
        while d and m:
            dbl = d.pop(0)
            if len(m) >= 2:
                for e in (dbl, m.pop(0), m.pop(0)):
                    e.result = "coincidental"
            else:
                dbl.result, m.pop(0).result = "minor", "coincidental"


# ---------------------------------------------------------------- repairs

LIVE_PLAY = ("shot", "hit", "giveaway", "takeaway", "delayed_penalty")
LATE_FACEOFF = 2  # seconds: a faceoff listed this soon after the play it started is out of order


def repair_events(meta: dict, events: list[Event]) -> tuple[list[Event], int]:
    """Fill gaps in the feed (mostly before 2019) that make a real event illegal in the engine:
    a faceoff listed up to LATE_FACEOFF seconds after the play it started is moved to its place;
    a faceoff during play gets an implied stoppage before it; live play during a stoppage with no
    faceoff gets an implied faceoff, credited to the team that acts; a new period with no end of
    the last one gets an implied period end. Returns the events and the number of implied events
    added."""
    out = list(events)
    s = new_game(meta["home"], meta["away"], meta["season"], meta["game_type"])
    added = i = 0
    while i < len(out):
        e = out[i]
        if e.period > s.period and s.phase in (FACEOFF, LIVE) and e.play_type != "shootout":
            prev = out[i - 1] if i else e
            out.insert(i, Event("period_end", "end", None, s.period, prev.t, prev.score, implied=True))
            added += 1
        elif s.phase == LIVE and e.play_type == "faceoff":
            out.insert(i, Event("stoppage", "other", None, e.period, e.t, e.score, implied=True))
            added += 1
        elif s.phase == FACEOFF and e.play_type in LIVE_PLAY and s.psec > 0:
            j = next((k for k in range(i + 1, len(out))
                      if out[k].period != e.period or out[k].t > e.t + LATE_FACEOFF
                      or out[k].play_type in ("faceoff", "stoppage", "penalty", "period_end")
                      or (out[k].play_type == "shot" and out[k].result == "goal")), None)
            if j is not None and out[j].play_type == "faceoff" and out[j].period == e.period \
                    and out[j].t <= e.t + LATE_FACEOFF:
                fo = out.pop(j)
                fo.t = min(fo.t, e.t)
                out.insert(i, fo)
            else:
                out.insert(i, Event("faceoff", "won", e.side, e.period, e.t, implied=True,
                                    skaters=e.skaters, empty_net=e.empty_net))
                added += 1
        e = out[i]
        o = Outcome(e.play_type, e.result, e.side, 0)
        s = apply_outcome(s, o)
        if s.over:
            break
        i += 1
    return out, added


# ---------------------------------------------------------------- replay

@dataclass
class Replay:
    rows: list[dict]
    ok: int = 0
    total: int = 0
    field_misses: dict = field(default_factory=dict)
    misses: dict = field(default_factory=dict)
    bad_at: list = field(default_factory=list)   # (event index, mismatched fields)


def replay_game(meta: dict, events: list[Event]) -> Replay:
    """Apply each real outcome to the engine; compare the engine's next state with the data."""
    s = new_game(meta["home"], meta["away"], meta["season"], meta["game_type"])
    rep = Replay(rows=[])
    labeled: set[int] = set()
    prev_t, prev_period = 0, 1
    for i, e in enumerate(events):
        if e.play_type == "penalty" and id(e) not in labeled:
            group = [x for x in events[i:] if x.play_type == "penalty" and x.period == e.period
                     and x.t == e.t and x.result in STRENGTH]
            label_penalties(group, s)
            labeled.update(id(x) for x in events[i:] if x.play_type == "penalty"
                           and x.period == e.period and x.t == e.t)
        if e.period != prev_period:
            prev_t, prev_period = 0, e.period
        elapsed = max(0, e.t - prev_t)
        o = Outcome(e.play_type, e.result, e.side, elapsed)
        legal = is_legal(s, o)
        rep.rows.append(dict(_state_row(s), play_type=o.play_type, result=o.result,
                             side=-1 if o.side is None else o.side, elapsed=elapsed, legal=legal))
        n = apply_outcome(s, o)
        prev_t = e.t
        bad = [] if legal else ["legal"]
        rest = events[i + 1:]
        if n.over:
            # no more play may follow the deciding event (a stoppage or penalty call may)
            if any(x.play_type not in ("period_end", "stoppage", "penalty") for x in rest):
                bad.append("over")
        elif not rest:
            bad.append("not_over")
        else:
            nxt = rest[0]
            if n.phase != SHOOTOUT and e.score is not None and tuple(n.score) != tuple(e.score):
                bad.append("score")
            if n.period != nxt.period:
                bad.append("period")
            # On-ice columns: compare at the last event of each second (several rows can share
            # one second at a stoppage, and the columns show the end of it), with the engine's
            # clock run up to that event, and not at a period's end (players leave the ice).
            settled = len(rest) < 2 or (rest[1].period, rest[1].t) != (nxt.period, nxt.t)
            if n.phase != SHOOTOUT and nxt.skaters is not None and settled \
                    and nxt.play_type not in ("period_end", "shootout"):
                # The data has one-second resolution: an event in the second a penalty ends may
                # be before or after it, so penalty clocks one second behind also match.
                dt = nxt.t - e.t if nxt.period == e.period else 0
                lag = apply_outcome(s, Outcome(o.play_type, o.result, o.side, max(elapsed - 1, 0)))
                m = [run_clock(n, dt), run_clock(lag, dt) if dt or elapsed else lag]
                if all(tuple(x.skaters) != tuple(nxt.skaters) for x in m):
                    bad.append("skaters")
                if tuple(m[0].empty_net) != tuple(nxt.empty_net):
                    bad.append("empty_net")
        rep.total += 1
        if not bad:
            rep.ok += 1
        else:
            for b in bad:
                rep.field_misses[b] = rep.field_misses.get(b, 0) + 1
            rep.bad_at.append((i, bad))
            key = f"{e.play_type}/{e.result}:{','.join(bad)}"
            rep.misses[key] = rep.misses.get(key, 0) + 1
        if n.over:
            break
        if e.score is not None and n.phase != SHOOTOUT:
            n.score = list(e.score)  # re-sync the score only: a miss is counted once
        s = n
    return rep


def _state_row(s: GameState) -> dict:
    sk = s.skaters
    pen = [sorted(p[1] for p in s.serving(side)) for side in (HOME, AWAY)]
    return dict(period=s.period, psec=s.psec, phase=s.phase, score_h=s.score[HOME], score_a=s.score[AWAY],
                sk_h=sk[HOME], sk_a=sk[AWAY], en_h=s.empty_net[HOME], en_a=s.empty_net[AWAY],
                pen_h=len(s.pens[HOME]), pen_a=len(s.pens[AWAY]),
                pen_left_h=pen[HOME][0] if pen[HOME] else 0, pen_left_a=pen[AWAY][0] if pen[AWAY] else 0)


def replay_season(path: Path, max_games: int | None = None) -> dict:
    """Replay every game in one release file; return agreement and the largest mismatch kinds."""
    df = load_season_file(path)
    ok = total = n_games = n_skipped = n_so_unordered = raw_ok = raw_total = n_implied = n_no_ice = 0
    fields: dict[str, int] = {}
    misses: dict[str, int] = {}
    per_game = []
    for g in split_games(df):
        res = game_events(g)
        if res is None:
            n_skipped += 1
            continue
        meta, events = res
        n_so_unordered += bool(meta.get("shootout_unordered"))
        n_no_ice += not meta["on_ice"]
        raw = replay_game(meta, [_copy(e) for e in events])
        raw_ok, raw_total = raw_ok + raw.ok, raw_total + raw.total
        events, added = repair_events(meta, events)
        n_implied += added
        rep = replay_game(meta, events)
        n_games += 1
        ok, total = ok + rep.ok, total + rep.total
        for k, v in rep.field_misses.items():
            fields[k] = fields.get(k, 0) + v
        for k, v in rep.misses.items():
            misses[k] = misses.get(k, 0) + v
        per_game.append(rep.ok / max(rep.total, 1))
        if max_games and n_games >= max_games:
            break
    season = season_of(df.season.dropna().iloc[0])
    return dict(season=season, games=n_games, skipped=n_skipped, shootouts_unordered=n_so_unordered,
                games_without_on_ice=n_no_ice,
                transitions=total, agreement=ok / max(total, 1), implied_events=n_implied,
                agreement_unrepaired=raw_ok / max(raw_total, 1), field_misses=fields,
                games_below_99=sum(a < 0.99 for a in per_game), misses=misses)


def _copy(e: Event) -> Event:
    return replace(e)


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Replay nhl_pbp_full games through the NHL rules engine.")
    ap.add_argument("files", nargs="+", type=int, help="release file numbers (play_by_play_<n>.parquet)")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/nhl/raw"))
    ap.add_argument("--max-games", type=int)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args(argv)
    tot_ok = tot = 0
    for f in args.files:
        out = replay_season(download_file(f, args.raw_dir), args.max_games)
        tot += out["transitions"]
        tot_ok += round(out["agreement"] * out["transitions"])
        top = sorted(out.pop("misses").items(), key=lambda kv: -kv[1])[:args.top]
        print(json.dumps(out))
        for k, v in top:
            print(f"  {k:<55} {v}")
    print(f"overall agreement {tot_ok / max(tot, 1):.4f} over {tot} transitions")


if __name__ == "__main__":
    main()
