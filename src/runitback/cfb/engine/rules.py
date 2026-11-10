"""Deterministic college football rules engine.

The NFL engine (``runitback.nfl.engine.rules``) with college rules. As there, the model only predicts
*what happened* on a play (an :class:`Outcome`); this module owns the game state and turns outcomes
into the next state, so every simulated game is legal. What differs from the NFL:

- **Tries (PAT).** Snapped from the 3-yard line (``PAT_SPOT``): the state of a try has ``yl`` 3.
  Like the NFL's, a try is untimed and a period cannot end on one.
- **Overtime** (``ot_format``). No game clock and no kickoffs. Each overtime period gives each team
  one possession, 1st and 10 at the opponent's 25 (``OT_SPOT``). A possession ends with a score, a
  turnover, a missed field goal or a turnover on downs. After both possessions, the leader wins;
  otherwise another period starts, and the team that went first goes second. A touchdown that puts
  the second team ahead ends the game without a try, and a defensive score (return touchdown or
  safety) ends it at once. Two-point tries become mandatory and, later, every period is a
  two-point shootout (one try each), by season:

  ============  ===========================  ==========================
  seasons       two-point try required from  two-point shootout from
  ============  ===========================  ==========================
  2004-2018     3rd overtime                 never
  2019-2020     3rd overtime                 5th overtime
  2021-         2nd overtime                 3rd overtime
  ============  ===========================  ==========================

  Each team gets one timeout per overtime period. Every game is decided (no ties, regular season or
  post-season).
- **Clock** (``clock_rules``). Four 15-minute quarters with 3 timeouts per half, as in the NFL; overtime
  is untimed (``qsec`` stays 0, so the model's elapsed mask only allows 0). How much clock a play
  runs off is the model's prediction, not the engine's, and its rules changed: until 2022 the
  clock stopped on every first down; from 2023 only inside the last two minutes of a half; from
  2024 there is a two-minute timeout. The model sees the era through its season token.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ...nfl.engine.rules import AWAY, HOME, GameState, Outcome  # noqa: F401  (re-exported)
from ..vocab import (
    KICKOFF, OT_SPOT, PAT, PAT_SPOT, QUARTER_SECONDS, RESULTS_FOR_TYPE, SCRIMMAGE, TIMEOUTS_PER_HALF,
    TIMEOUTS_PER_OT, TYPES_FOR_PHASE, YARDS_MAX, yards_kind,
)


@dataclass
class CFBState(GameState):
    """The NFL's game state plus where a college overtime period stands. In overtime ``ot_first``
    (inherited) is the side with the period's first possession and ``ot_half`` is 0 during that
    possession, 1 during the second one."""

    ot_half: int = 0

    def to_dict(self) -> dict:
        d = super().to_dict()
        if self.qtr >= 5:
            d |= {"ot_period": self.qtr - 4, "ot_half": self.ot_half,
                  "ot_first": None if self.ot_first is None else (self.home, self.away)[self.ot_first]}
        return d


def new_game(home: str, away: str, season: int, game_type: str = "REG",
             rng: random.Random | None = None, opening_kicker: int | None = None) -> CFBState:
    if opening_kicker is None:
        opening_kicker = (rng or random).choice([HOME, AWAY])
    return CFBState(season=season, game_type=game_type, home=home, away=away,
                    pos=opening_kicker, opening_kicker=opening_kicker)


# ---------------------------------------------------------------- rules by season

def ot_format(season: int) -> dict[str, int | None]:
    """The overtime period from which a touchdown must be followed by a two-point try
    (``two_point``) and from which each period is a two-point shootout (``shootout``, None = never)."""
    if season < 2019:
        return {"two_point": 3, "shootout": None}
    if season < 2021:
        return {"two_point": 3, "shootout": 5}
    return {"two_point": 2, "shootout": 3}


def ot_period(s: GameState) -> int:
    """1 for the first overtime period, 0 in regulation."""
    return max(0, s.qtr - 4)


def is_shootout(season: int, period: int) -> bool:
    so = ot_format(season)["shootout"]
    return so is not None and period >= so


def two_point_only(s: GameState) -> bool:
    """Whether a try must be a two-point attempt (overtime rules of the season)."""
    p = ot_period(s)
    return p > 0 and p >= ot_format(s.season)["two_point"]


def clock_rules(season: int) -> dict[str, bool]:
    """The game-clock rules of a season (for reference: run-off is predicted by the model, which
    sees the era through its season token). ``first_down_stops``: the clock stops on every first
    down (from 2023 only inside the last two minutes of a half); ``two_minute_timeout``: an
    automatic stoppage at two minutes of each half (2024 on)."""
    return {"first_down_stops": season < 2023, "two_minute_timeout": season >= 2024,
            "untimed_overtime": True}


# ---------------------------------------------------------------- legality

def legal_types(s: GameState) -> list[str]:
    types = list(TYPES_FOR_PHASE[s.phase])
    if s.phase == PAT and two_point_only(s):
        types.remove("extra_point")
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


def _try(n: GameState, team: int) -> None:
    n.phase, n.pos, n.down, n.ytg, n.yl = PAT, team, 0, PAT_SPOT, PAT_SPOT


def _touchdown(n: GameState, team: int) -> None:
    n.score[team] += 6
    _try(n, team)


def _kickoff_by(n: GameState, team: int) -> None:
    n.phase, n.pos, n.down, n.ytg, n.yl = KICKOFF, team, 0, 0, 0


def _ot_possession(n: CFBState, team: int) -> None:
    """Start `team`'s possession of the current overtime period."""
    if is_shootout(n.season, ot_period(n)):
        _try(n, team)
    else:
        _first_down(n, team, OT_SPOT)


def start_overtime(n: CFBState, first: int) -> None:
    """Regulation ended tied: period 1 of overtime, `first` with the ball."""
    n.qtr, n.qsec, n.timeouts = 5, 0, [TIMEOUTS_PER_OT, TIMEOUTS_PER_OT]
    n.ot_first, n.ot_half = first, 0
    _ot_possession(n, first)


def _end_ot_possession(n: CFBState) -> None:
    if n.ot_half == 0:
        n.ot_half = 1
        _ot_possession(n, 1 - n.ot_first)
        return
    if n.score[HOME] != n.score[AWAY]:
        n.over = True
        return
    n.qtr += 1
    n.ot_first, n.ot_half = 1 - n.ot_first, 0
    n.timeouts = [TIMEOUTS_PER_OT, TIMEOUTS_PER_OT]
    _ot_possession(n, n.ot_first)


def _apply_ot(s: CFBState, n: CFBState, o: Outcome) -> None:
    off, dfn = s.pos, 1 - s.pos
    t, r, y = o.play_type, o.result, int(o.yards)
    if s.phase == PAT:
        if r == "good":
            n.score[off] += 1 if t == "extra_point" else 2
        _end_ot_possession(n)
        return
    if t == "field_goal" and r == "good":
        n.score[off] += 3
        _end_ot_possession(n)
    elif yards_kind(t, r) == "spot":
        holder = off if r == "kick_recovered" else dfn
        if holder == off:
            _first_down(n, off, max(y, 1))  # the kicking team kept the ball (muffed punt): play on
        elif y <= 0:
            n.score[dfn] += 6  # defensive touchdown: the game is over
            n.over = True
        else:
            _end_ot_possession(n)
    elif y >= s.yl:
        n.score[off] += 6
        if s.ot_half == 1 and n.score[off] > n.score[dfn]:
            n.over = True  # the second team went ahead: no try
        else:
            _try(n, off)
    elif s.yl - y >= 100:
        n.score[dfn] += 2  # safety: a defensive score ends the game
        n.over = True
    else:
        new_yl = s.yl - y
        if r == "penalty_first_down" or y >= s.ytg:
            _first_down(n, off, new_yl)
        elif r == "penalty":
            n.yl, n.ytg = new_yl, s.ytg - y
        elif s.down >= 4:
            _end_ot_possession(n)  # turnover on downs
        else:
            n.yl, n.down, n.ytg = new_yl, s.down + 1, s.ytg - y


def apply_outcome(s: GameState, o: Outcome, rng: random.Random | None = None,
                  ot_first: int | None = None) -> CFBState:
    """Return the state after outcome `o` is applied to state `s`. ``ot_first``: the side with the
    first possession of overtime if this play ends regulation tied (default: a coin flip)."""
    if not isinstance(s, CFBState):
        s = CFBState(**vars(s))
    if s.over:
        return s.copy()
    n = s.copy()
    off, dfn = s.pos, 1 - s.pos
    t, r, y = o.play_type, o.result, int(o.yards)

    if t == "timeout":
        n.timeouts[off if r == "timeout_offense" else dfn] -= 1
        return n
    if s.qtr >= 5:
        _apply_ot(s, n, o)
        return n

    if s.phase == KICKOFF:
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
            n.score[dfn] += 2  # safety; the conceding team free-kicks
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
    if n.qsec <= 0 and n.phase != PAT:
        _end_period(n, rng, ot_first)
    return n


def _end_period(n: CFBState, rng: random.Random | None, ot_first: int | None) -> None:
    if n.qtr in (1, 3):
        n.qtr, n.qsec = n.qtr + 1, QUARTER_SECONDS
    elif n.qtr == 2:
        n.qtr, n.qsec = 3, QUARTER_SECONDS
        n.timeouts = [TIMEOUTS_PER_HALF, TIMEOUTS_PER_HALF]
        _kickoff_by(n, 1 - n.opening_kicker)
    elif n.score[HOME] != n.score[AWAY]:
        n.over = True
    else:
        start_overtime(n, (rng or random).choice([HOME, AWAY]) if ot_first is None else ot_first)
