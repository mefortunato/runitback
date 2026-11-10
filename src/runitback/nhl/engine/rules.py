"""Deterministic NHL rules engine.

The model only predicts *what happened* on an event (an Outcome); this module owns the game state
and turns outcomes into the next state, so every simulated game is legal:

- play restarts with a faceoff after every stoppage, goal and penalty;
- penalties put players in the box: minors (2:00), double minors (4:00) and majors (5:00) leave the
  team a skater short, at most two at a time (a third waits until one expires) and never fewer
  than three skaters; a power-play goal ends the shortest minor (a double minor's current two
  minutes) of a team with more penalties being served; majors and matched minors (one each at
  5-on-5, played 4-on-4) are served in full; coincidental penalties and misconducts leave the
  strength alone; a penalty shot is one attempt by the fouled team, then a faceoff;
- a pulled goalie adds an extra attacker until the goalie returns or the period ends;
- regular-season overtime is 5 minutes of 3-on-3 (4-on-4 before 2015-16), where a penalty gives
  the other team an extra skater, and a player leaving the box joins a 4-on-4 that lasts until
  the next stoppage; playoff overtime is 20-minute 5-on-5 periods until a goal; an
  overtime goal ends the game;
- a regular-season game tied after overtime goes to a shootout: three rounds, then sudden death,
  ending as soon as one team cannot be caught; the winner gets one goal.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..vocab import (
    FACEOFF,
    FIRST_3V3_SEASON,
    FIRST_SHOOTOUT_SEASON,
    LIVE,
    MAX_ELAPSED,
    MAX_SERVING,
    MIN_SKATERS,
    MINOR_SECONDS,
    PENALTY_SHOT,
    PERIOD_SECONDS,
    PHASES,
    REG_OT_SECONDS,
    RELEASABLE,
    RESULTS_FOR_TYPE,
    SHOOTOUT,
    SHOOTOUT_ROUNDS,
    SIDELESS,
    SKATERS,
    STRENGTH_PENALTIES,
    TYPES_FOR_PHASE,
)

HOME, AWAY = 0, 1
# What can still happen once the period clock reads 0:00 (end-of-period scrums are penalized).
AT_ZERO = ("period_end", "penalty", "stoppage")


@dataclass(frozen=True)
class Outcome:
    play_type: str
    result: str
    side: int | None = None   # HOME / AWAY: the team that did it (the penalized team for a penalty)
    elapsed: int = 0          # clock seconds since the previous event


@dataclass
class GameState:
    season: int
    game_type: str  # "REG" | "POST"
    home: str
    away: str
    period: int = 1
    psec: int = PERIOD_SECONDS                       # seconds left in the period
    phase: int = FACEOFF
    score: list = field(default_factory=lambda: [0, 0])
    # Penalty box per team, in the order assessed: [kind, seconds left]. The first MAX_SERVING
    # are being served (their clocks run); the rest wait.
    pens: list = field(default_factory=lambda: [[], []])
    empty_net: list = field(default_factory=lambda: [False, False])
    ps_side: int | None = None                       # the team taking a penalty shot
    so_shots: list = field(default_factory=lambda: [0, 0])
    so_goals: list = field(default_factory=lambda: [0, 0])
    so_first: int | None = None                      # the team that shot first in the shootout
    # Regular-season overtime: a player leaving the box joins play and the other team keeps its
    # extra skater (4-on-3 becomes 4-on-4) until the next stoppage. Skaters held until then.
    held: list = field(default_factory=lambda: [0, 0])
    over: bool = False

    def copy(self) -> GameState:
        return replace(self, score=list(self.score), pens=[[list(p) for p in side] for side in self.pens],
                       empty_net=list(self.empty_net), so_shots=list(self.so_shots),
                       so_goals=list(self.so_goals), held=list(self.held))

    # ------------------------------------------------------------ derived state

    def serving(self, side: int) -> list:
        return self.pens[side][:MAX_SERVING]

    @property
    def base_skaters(self) -> int:
        if self.period <= 3 or self.game_type != "REG":
            return SKATERS
        return 3 if self.season >= FIRST_3V3_SEASON else 4

    @property
    def skaters(self) -> tuple[int, int]:
        """Skaters on the ice per team (home, away), counting an extra attacker for a pulled goalie.
        In a penalty shot or shootout: 1 for the shooting team, 0 for the other."""
        if self.phase in (PENALTY_SHOT, SHOOTOUT):
            shooter = self.ps_side if self.phase == PENALTY_SHOT else shootout_shooter(self)
            if self.over or shooter is None:
                return 0, 0
            return (1, 0) if shooter == HOME else (0, 1)
        return tuple(k + int(self.empty_net[s]) for s, k in enumerate(self.strength))

    @property
    def strength(self) -> tuple[int, int]:
        """Skaters per team from the penalty box alone (no extra attackers)."""
        sk = [self.base_skaters - len(self.serving(s)) for s in (HOME, AWAY)]
        low = min(sk)
        if low < MIN_SKATERS:  # never below three: the other team gets the extra skater instead
            sk = [k + MIN_SKATERS - low for k in sk]
        return tuple(max(k, h) for k, h in zip(sk, self.held))

    @property
    def shorthanded(self) -> tuple[int, int]:
        """Penalties being served per team (home, away)."""
        return len(self.serving(HOME)), len(self.serving(AWAY))

    def to_dict(self) -> dict:
        sk = self.skaters
        return {
            "season": self.season, "game_type": self.game_type, "home": self.home, "away": self.away,
            "period": self.period, "psec": self.psec, "phase": PHASES[self.phase],
            "home_score": self.score[HOME], "away_score": self.score[AWAY],
            "home_skaters": sk[HOME], "away_skaters": sk[AWAY],
            "home_penalties": [list(p) for p in self.pens[HOME]],
            "away_penalties": [list(p) for p in self.pens[AWAY]],
            "home_empty_net": self.empty_net[HOME], "away_empty_net": self.empty_net[AWAY],
            "home_so_goals": self.so_goals[HOME], "away_so_goals": self.so_goals[AWAY],
            "home_so_shots": self.so_shots[HOME], "away_so_shots": self.so_shots[AWAY],
            "over": self.over,
        }


def new_game(home: str, away: str, season: int, game_type: str = "REG") -> GameState:
    """A game at the opening faceoff."""
    return GameState(season=season, game_type=game_type, home=home, away=away)


def period_length(period: int, game_type: str) -> int:
    return PERIOD_SECONDS if period <= 3 or game_type != "REG" else REG_OT_SECONDS


def shootout_shooter(s: GameState) -> int | None:
    """The team that shoots next in the shootout (None: either, before the first attempt)."""
    if s.so_first is None:
        return None
    return s.so_first if s.so_shots[HOME] == s.so_shots[AWAY] else 1 - s.so_first


def shootout_decided(shots: list, goals: list) -> bool:
    """Whether a shootout with these attempts and goals per team (home, away) is over."""
    nh, na = shots
    gh, ga = goals
    if nh == na and nh >= SHOOTOUT_ROUNDS:
        return gh != ga
    if max(nh, na) <= SHOOTOUT_ROUNDS:
        return gh > ga + SHOOTOUT_ROUNDS - na or ga > gh + SHOOTOUT_ROUNDS - nh
    return False  # sudden death: both teams shoot in every round


# ---------------------------------------------------------------- legality

def legal_types(s: GameState) -> list[str]:
    if s.over:
        return []
    types = list(TYPES_FOR_PHASE[s.phase])
    if s.phase in (FACEOFF, LIVE) and s.psec <= 0:
        return [t for t in types if t in AT_ZERO]  # the clock has run out: end the period
    if "period_end" in types and s.psec > (0 if s.phase == FACEOFF else MAX_ELAPSED):
        types.remove("period_end")
    if "goalie_pull" in types and all(s.empty_net):
        types.remove("goalie_pull")
    if "goalie_return" in types and not any(s.empty_net):
        types.remove("goalie_return")
    return types


def legal_sides(s: GameState, play_type: str) -> list[int | None]:
    if play_type in SIDELESS:
        return [None]
    if play_type == "shootout":
        nxt = shootout_shooter(s)
        return [HOME, AWAY] if nxt is None else [nxt]
    if play_type == "shot" and s.phase == PENALTY_SHOT:
        return [s.ps_side]
    if play_type == "goalie_pull":
        return [side for side in (HOME, AWAY) if not s.empty_net[side]]
    if play_type == "goalie_return":
        return [side for side in (HOME, AWAY) if s.empty_net[side]]
    return [HOME, AWAY]


def legal_results(s: GameState, play_type: str) -> list[str]:
    res = list(RESULTS_FOR_TYPE[play_type])
    if play_type == "shot" and s.phase == PENALTY_SHOT:
        res.remove("blocked")
    elif play_type == "penalty" and s.phase != LIVE:
        res.remove("penalty_shot")  # a penalty shot is awarded for a foul during play
    return res


def is_legal(s: GameState, o: Outcome) -> bool:
    return (o.play_type in legal_types(s) and o.side in legal_sides(s, o.play_type)
            and o.result in legal_results(s, o.play_type))


def elapsed_range(s: GameState, play_type: str) -> tuple[int, int]:
    """Clock seconds an event may take. The clock is stopped for faceoffs, penalty shots and the
    shootout; a period end runs out the clock."""
    if play_type == "period_end":
        return s.psec, s.psec
    if s.phase != LIVE:
        return 0, 0
    return 0, min(MAX_ELAPSED, max(s.psec, 0))


# ---------------------------------------------------------------- transitions

def _run_clock(n: GameState, seconds: int) -> None:
    """Run the game clock: the period clock and the penalties being served."""
    n.psec = max(0, n.psec - seconds)
    before = n.strength
    expired = [0, 0]
    for side in (HOME, AWAY):
        left, box = seconds, n.pens[side]
        while left > 0 and box:
            serving = box[:MAX_SERVING]
            step = min(left, min(p[1] for p in serving))
            for p in serving:
                p[1] -= step
            expired[side] += sum(p[1] <= 0 for p in serving)
            box[:] = [p for p in box if p[1] > 0]
            left -= step
    if any(expired) and n.phase == LIVE and n.base_skaters < SKATERS:
        n.held = [min(SKATERS, b + x) for b, x in zip(before, expired)]


def run_clock(s: GameState, seconds: int) -> GameState:
    """The state `seconds` of game clock later with nothing else happening."""
    n = s.copy()
    if n.phase in (FACEOFF, LIVE) and not n.over:
        _run_clock(n, max(0, int(seconds)))
    return n


def _power_play_goal(n: GameState, scorer: int) -> None:
    """A goal by a team with more players: the shortest minor on the other team ends. Matched
    minors (4-on-4) do not count toward being short-handed."""
    other = 1 - scorer

    def short(side):
        return sum(p[0] != "matched_minor" for p in n.serving(side))
    if short(other) <= short(scorer):
        return
    minors = [p for p in n.serving(other) if p[0] in RELEASABLE]
    if not minors:
        return
    p = min(minors, key=lambda q: q[1])
    if p[0] == "double_minor" and p[1] > MINOR_SECONDS:
        p[1] = MINOR_SECONDS
    else:
        n.pens[other].remove(p)


def apply_outcome(s: GameState, o: Outcome) -> GameState:
    """Return the state after outcome `o` is applied to state `s`."""
    if s.over:
        return s.copy()
    n = s.copy()
    t, r, side = o.play_type, o.result, o.side
    if n.phase in (FACEOFF, LIVE):
        _run_clock(n, s.psec if t == "period_end" else max(0, int(o.elapsed)))

    if t == "faceoff":
        n.phase = LIVE
    elif t == "shot":
        if r == "goal":
            n.score[side] += 1
            if s.phase == LIVE:
                _power_play_goal(n, side)
            if n.period > 3:
                n.over = True  # sudden-death overtime
        if s.phase == PENALTY_SHOT:
            n.phase, n.ps_side = FACEOFF, None
        elif r == "goal":
            n.phase = FACEOFF
    elif t == "penalty":
        if r in STRENGTH_PENALTIES:
            n.pens[side].append([r, STRENGTH_PENALTIES[r]])
        if r == "penalty_shot":
            n.phase, n.ps_side = PENALTY_SHOT, 1 - side
        else:
            n.phase = FACEOFF
    elif t == "stoppage":
        n.phase = FACEOFF
    elif t == "goalie_pull":
        n.empty_net[side] = True
    elif t == "goalie_return":
        n.empty_net[side] = False
    elif t == "shootout":
        if n.so_first is None:
            n.so_first = side
        n.so_shots[side] += 1
        if r == "goal":
            n.so_goals[side] += 1
        if shootout_decided(n.so_shots, n.so_goals):
            n.score[0 if n.so_goals[HOME] > n.so_goals[AWAY] else 1] += 1
            n.over = True
    # hit, giveaway, takeaway, delayed_penalty: nothing but the clock

    if t == "period_end":
        _end_period(n)
    if n.phase != LIVE:
        n.held = [0, 0]  # a stoppage: back to the strength the penalty box gives
    return n


def _end_period(n: GameState) -> None:
    tied = n.score[HOME] == n.score[AWAY]
    if n.period >= 3 and not tied:
        n.over = True
        return
    n.empty_net = [False, False]
    if n.period >= 4 and n.game_type == "REG":
        if n.season >= FIRST_SHOOTOUT_SEASON:
            n.period += 1
            n.phase, n.psec, n.pens = SHOOTOUT, 0, [[], []]
        else:
            n.over = True  # a tie
        return
    n.period += 1
    n.psec = period_length(n.period, n.game_type)
    n.phase = FACEOFF
