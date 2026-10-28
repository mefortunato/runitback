"""Deterministic NFL rules engine.

The model only predicts *what happened* on a play (an Outcome); this module owns the game
state and turns outcomes into the next state, so every simulated game is legal.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

from ..vocab import (
    KICKOFF, PAT, SCRIMMAGE, RESULTS_FOR_TYPE, TYPES_FOR_PHASE, YARDS_MAX, yards_kind,
)

HOME, AWAY = 0, 1


@dataclass(frozen=True)
class Outcome:
    play_type: str
    result: str
    yards: int = 0
    elapsed: int = 0


@dataclass
class GameState:
    season: int
    game_type: str  # "REG" | "POST"
    home: str
    away: str
    qtr: int = 1
    qsec: int = 900
    phase: int = KICKOFF
    pos: int = HOME           # team "with the ball"; on kickoffs, the kicking team
    down: int = 0
    ytg: int = 0
    yl: int = 0               # yards from opponent end zone (scrimmage only)
    score: list = field(default_factory=lambda: [0, 0])
    timeouts: list = field(default_factory=lambda: [3, 3])
    opening_kicker: int = HOME
    ot_first: int | None = None
    ot_poss: int = 0
    ot_last_pos: int | None = None
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
            "season": self.season, "game_type": self.game_type, "home": self.home,
            "away": self.away, "qtr": self.qtr, "qsec": self.qsec,
            "phase": ["kickoff", "pat", "scrimmage"][self.phase],
            "pos": self.off_team, "down": self.down, "ytg": self.ytg, "yl": self.yl,
            "home_score": self.score[HOME], "away_score": self.score[AWAY],
            "home_timeouts": self.timeouts[HOME], "away_timeouts": self.timeouts[AWAY],
            "over": self.over,
        }


def new_game(home: str, away: str, season: int, game_type: str = "REG",
             rng: random.Random | None = None, opening_kicker: int | None = None) -> GameState:
    if opening_kicker is None:
        opening_kicker = (rng or random).choice([HOME, AWAY])
    return GameState(season=season, game_type=game_type, home=home, away=away,
                     pos=opening_kicker, opening_kicker=opening_kicker)


def ot_rule(season: int, game_type: str) -> str:
    if game_type == "POST":
        return "sudden" if season < 2010 else "modified" if season < 2022 else "both"
    return "sudden" if season < 2012 else "modified" if season < 2025 else "both"


def ot_length(season: int, game_type: str) -> int:
    return 600 if (game_type == "REG" and season >= 2017) else 900


# ---------------------------------------------------------------- legality

def legal_types(s: GameState) -> list[str]:
    types = list(TYPES_FOR_PHASE[s.phase])
    if "timeout" in types and not legal_results(s, "timeout"):
        types.remove("timeout")
    return types


def legal_results(s: GameState, play_type: str) -> list[str]:
    res = list(RESULTS_FOR_TYPE[play_type])
    if play_type == "timeout":
        res = [r for r, side in (("timeout_offense", s.pos), ("timeout_defense", 1 - s.pos))
               if s.timeouts[side] > 0]
    return res


def yards_range(s: GameState, play_type: str, result: str) -> tuple[int, int]:
    kind = yards_kind(play_type, result)
    if kind == "none":
        return 0, 0
    if kind == "spot":
        return 0, 99
    return s.yl - 100, min(s.yl, YARDS_MAX)


# ---------------------------------------------------------------- transitions

def _first_down(n: GameState, pos: int, yl: int) -> None:
    n.phase, n.pos, n.yl, n.down, n.ytg = SCRIMMAGE, pos, yl, 1, min(10, yl)


def _touchdown(n: GameState, team: int) -> None:
    n.score[team] += 6
    n.phase, n.pos, n.down, n.ytg, n.yl = PAT, team, 0, 0, 0


def _kickoff_by(n: GameState, team: int) -> None:
    n.phase, n.pos, n.down, n.ytg, n.yl = KICKOFF, team, 0, 0, 0


def apply_outcome(s: GameState, o: Outcome, rng: random.Random | None = None) -> GameState:
    """Return the state after outcome `o` is applied to state `s`."""
    if s.over:
        return s.copy()
    n = s.copy()
    off, dfn = s.pos, 1 - s.pos
    t, r, y = o.play_type, o.result, int(o.yards)

    if t == "timeout":
        n.timeouts[off if r == "timeout_offense" else dfn] -= 1
    elif s.phase == KICKOFF:
        holder = dfn if r == "kick_received" else off
        if y <= 0:
            _touchdown(n, holder)
        else:
            _first_down(n, holder, y)
    elif s.phase == PAT:
        if r == "good":
            n.score[off] += 1 if t == "extra_point" else 2
        _kickoff_by(n, off)
    elif t == "field_goal" and r == "good":
        n.score[off] += 3
        _kickoff_by(n, off)
    elif yards_kind(t, r) == "spot":
        holder = off if r == "kick_recovered" else dfn
        if y <= 0:
            _touchdown(n, holder)
        else:
            _first_down(n, holder, y)
    else:  # gain semantics
        if y >= s.yl:
            _touchdown(n, off)
        elif s.yl - y >= 100:
            n.score[dfn] += 2  # safety; conceding team free-kicks
            _kickoff_by(n, off)
        else:
            new_yl = s.yl - y
            if r == "penalty_first_down" or y >= s.ytg:
                _first_down(n, off, new_yl)
            elif r == "penalty":
                n.yl, n.ytg = new_yl, s.ytg - y
            elif s.down >= 4:
                _first_down(n, dfn, 100 - new_yl)  # turnover on downs
            else:
                n.yl, n.down, n.ytg = new_yl, s.down + 1, s.ytg - y

    n.qsec = max(0, s.qsec - max(0, int(o.elapsed)))

    if n.qtr >= 5:
        if n.phase in (SCRIMMAGE, PAT) and n.pos != n.ot_last_pos:
            n.ot_poss += 1
            n.ot_last_pos = n.pos
            if n.ot_first is None:
                n.ot_first = n.pos
        if _ot_over(n):
            n.over = True
            return n

    if n.qsec <= 0 and n.phase != PAT:
        _end_period(n, rng)
    return n


def _ot_over(n: GameState) -> bool:
    diff = n.score[HOME] - n.score[AWAY]
    if diff == 0:
        return False
    rule = ot_rule(n.season, n.game_type)
    if rule == "sudden":
        return True
    leader, lead = (HOME if diff > 0 else AWAY), abs(diff)
    t1 = n.ot_first
    if n.ot_poss <= 1 and leader == t1:
        return rule == "modified" and lead >= 6
    if n.ot_poss == 2 and leader == t1 and n.pos != t1:
        return False  # second team still has its possession
    return True


def _end_period(n: GameState, rng: random.Random | None) -> None:
    diff = n.score[HOME] - n.score[AWAY]
    if n.qtr in (1, 3):
        n.qtr, n.qsec = n.qtr + 1, 900
    elif n.qtr == 2:
        n.qtr, n.qsec = 3, 900
        n.timeouts = [3, 3]
        _kickoff_by(n, 1 - n.opening_kicker)
    elif diff != 0:
        n.over = True
    elif n.qtr == 4:
        n.qtr, n.qsec = 5, ot_length(n.season, n.game_type)
        n.timeouts = [2, 2]
        _kickoff_by(n, (rng or random).choice([HOME, AWAY]))
    elif n.game_type == "REG":
        n.over = True  # tie
    else:
        n.qtr, n.qsec = n.qtr + 1, 900
        if n.qtr % 2 == 1:
            n.timeouts = [2, 2]
