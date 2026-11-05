"""Deterministic NBA rules engine.

The model only predicts *what happened* on an event (an Outcome); this module owns the game
state and turns outcomes into the next state, so every simulated game is legal: scores follow
from made baskets, possession from rebounds and turnovers, free-throw trips run to completion,
quarters and overtimes end on the clock, and a game never ends tied.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..vocab import (
    FREE_THROW, LIVE, MAX_ELAPSED, MAX_Q4_TIMEOUTS, OT_SECONDS, OT_TIMEOUTS, PHASES, QUARTER_SECONDS,
    RESULTS_FOR_TYPE, TIMEOUTS_PER_GAME, TIP, TYPES_FOR_PHASE,
)

HOME, AWAY = 0, 1


@dataclass(frozen=True)
class Outcome:
    play_type: str
    result: str
    elapsed: int = 0


@dataclass
class GameState:
    season: int
    game_type: str  # "REG" | "POST"
    home: str
    away: str
    qtr: int = 1
    qsec: int = QUARTER_SECONDS
    phase: int = TIP
    pos: int = HOME           # team with the ball (at a tip-off: home, by convention)
    fts_left: int = 0         # free throws left in the current trip
    fts_total: int = 0        # size of the current trip (display only)
    score: list = field(default_factory=lambda: [0, 0])
    timeouts: list = field(default_factory=lambda: [TIMEOUTS_PER_GAME, TIMEOUTS_PER_GAME])
    tip_winner: int | None = None
    over: bool = False

    def copy(self) -> "GameState":
        return replace(self, score=list(self.score), timeouts=list(self.timeouts))

    @property
    def off_team(self) -> str:
        return self.home if self.pos == HOME else self.away

    @property
    def def_team(self) -> str:
        return self.away if self.pos == HOME else self.home

    def to_dict(self) -> dict:
        return {
            "season": self.season, "game_type": self.game_type, "home": self.home, "away": self.away,
            "qtr": self.qtr, "qsec": self.qsec, "phase": PHASES[self.phase], "pos": self.off_team,
            "fts_left": self.fts_left, "fts_total": self.fts_total,
            "home_score": self.score[HOME], "away_score": self.score[AWAY],
            "home_timeouts": self.timeouts[HOME], "away_timeouts": self.timeouts[AWAY],
            "over": self.over,
        }


def new_game(home: str, away: str, season: int, game_type: str = "REG") -> GameState:
    """A game at the opening tip. Who wins the tip is the first sampled event."""
    return GameState(season=season, game_type=game_type, home=home, away=away)


def period_length(qtr: int) -> int:
    return QUARTER_SECONDS if qtr <= 4 else OT_SECONDS


# ---------------------------------------------------------------- legality

def legal_types(s: GameState) -> list[str]:
    types = list(TYPES_FOR_PHASE[s.phase])
    if "timeout" in types and not (s.timeouts[0] > 0 or s.timeouts[1] > 0):
        types.remove("timeout")
    if "run_out" in types and s.qsec > MAX_ELAPSED:
        types.remove("run_out")
    return types


def legal_results(s: GameState, play_type: str) -> list[str]:
    res = list(RESULTS_FOR_TYPE[play_type])
    if play_type == "timeout":
        res = [r for r, side in (("timeout_offense", s.pos), ("timeout_defense", 1 - s.pos))
               if s.timeouts[side] > 0]
    elif play_type == "free_throw":
        res = ["made", "missed"] if s.fts_left > 1 else ["made", "missed_off", "missed_def"]
    return res


def elapsed_range(s: GameState, play_type: str) -> tuple[int, int]:
    hi = min(MAX_ELAPSED, max(s.qsec, 0))
    if play_type == "run_out":
        return hi, hi
    return 0, hi


# ---------------------------------------------------------------- transitions

def _start_trip(n: GameState, fts: int) -> None:
    n.phase, n.fts_left, n.fts_total = FREE_THROW, fts, fts


def _end_trip(n: GameState) -> None:
    n.phase, n.fts_left, n.fts_total = LIVE, 0, 0


def apply_outcome(s: GameState, o: Outcome) -> GameState:
    """Return the state after outcome `o` is applied to state `s`."""
    if s.over:
        return s.copy()
    n = s.copy()
    off, dfn = s.pos, 1 - s.pos
    t, r = o.play_type, o.result

    if t == "timeout":
        n.timeouts[off if r == "timeout_offense" else dfn] -= 1
    elif t == "jump_ball":
        n.pos = off if r == "won" else dfn
        if s.phase == TIP:
            n.phase = LIVE
            if s.qtr == 1:
                n.tip_winner = n.pos
    elif t in ("shot2", "shot3"):
        pts = 2 if t == "shot2" else 3
        if r in ("made", "made_fouled"):
            n.score[off] += pts
        if r == "made" or r == "missed_def":
            n.pos = dfn
        elif r == "made_fouled":
            _start_trip(n, 1)
        elif r == "missed_fouled":
            _start_trip(n, pts)
    elif t == "free_throw":
        if r == "made":
            n.score[off] += 1
        n.fts_left -= 1
        if n.fts_left <= 0:
            _end_trip(n)
            if r != "missed_off":
                n.pos = dfn
    elif t == "turnover":
        n.pos = dfn
    elif t == "foul":
        _start_trip(n, int(r[-1]))
    elif t == "technical":
        if r == "tech_offense":
            n.score[off] += 1
        elif r == "tech_defense":
            n.score[dfn] += 1

    n.qsec = 0 if t == "run_out" else max(0, s.qsec - max(0, int(o.elapsed)))
    if n.qsec <= 0 and n.phase != FREE_THROW:
        _end_period(n)
    return n


def _end_period(n: GameState) -> None:
    diff = n.score[HOME] - n.score[AWAY]
    tip = HOME if n.tip_winner is None else n.tip_winner
    if n.qtr < 4:
        n.qtr += 1
        n.qsec = QUARTER_SECONDS
        n.phase = LIVE
        # The team that lost the opening tip starts the 2nd and 3rd quarters; the winner the 4th.
        n.pos = tip if n.qtr == 4 else 1 - tip
        if n.qtr == 4:
            n.timeouts = [min(t, MAX_Q4_TIMEOUTS) for t in n.timeouts]
    elif diff != 0:
        n.over = True
    else:
        n.qtr += 1
        n.qsec = OT_SECONDS
        n.phase, n.pos = TIP, HOME
        n.timeouts = [OT_TIMEOUTS, OT_TIMEOUTS]
    n.fts_left = n.fts_total = 0
