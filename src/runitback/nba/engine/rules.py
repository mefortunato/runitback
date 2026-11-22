"""Deterministic NBA rules engine.

The model only predicts *what happened* on an event (an Outcome); this module owns the game
state and turns outcomes into the next state, so every simulated game is legal: scores follow
from made baskets, possession from rebounds and turnovers, free-throw trips run to completion,
quarters and overtimes end on the clock, and a game never ends tied.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..league import NBA, League
from ..vocab import FREE_THROW, LIVE, PHASES, TIP, TYPES_FOR_PHASE

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
    qtr: int = 1              # the period: a quarter, or a half in the men's college game
    qsec: int | None = None   # seconds left in the period (a full period when None)
    phase: int = TIP
    pos: int = HOME           # team with the ball (at a tip-off: home, by convention)
    fts_left: int = 0         # free throws left in the current trip
    fts_total: int = 0        # size of the current trip (display only)
    score: list = field(default_factory=lambda: [0, 0])
    timeouts: list | None = None  # per team (the league's opening count when None)
    tip_winner: int | None = None
    over: bool = False
    league: League = field(default=NBA, repr=False)
    fouls: list = field(default_factory=lambda: [0, 0])  # team fouls committed in the bonus period
    one_and_one: bool = False  # the current trip is a 1-and-1: its first shot is the only one if missed
    arrow: int | None = None   # possession arrow: who gets the next held ball / period start (college)

    def __post_init__(self):
        if self.qsec is None:
            self.qsec = self.league.period_length(self.qtr)
        if self.timeouts is None:
            self.timeouts = [self.league.timeouts, self.league.timeouts]

    def copy(self) -> "GameState":
        return replace(self, score=list(self.score), timeouts=list(self.timeouts), fouls=list(self.fouls))

    @property
    def off_team(self) -> str:
        return self.home if self.pos == HOME else self.away

    @property
    def def_team(self) -> str:
        return self.away if self.pos == HOME else self.home

    def to_dict(self) -> dict:
        d = {
            "season": self.season, "game_type": self.game_type, "home": self.home, "away": self.away,
            "qtr": self.qtr, "qsec": self.qsec, "phase": PHASES[self.phase], "pos": self.off_team,
            "fts_left": self.fts_left, "fts_total": self.fts_total,
            "home_score": self.score[HOME], "away_score": self.score[AWAY],
            "home_timeouts": self.timeouts[HOME], "away_timeouts": self.timeouts[AWAY],
            "over": self.over,
        }
        if self.league.bonus:
            d |= {"home_fouls": self.fouls[HOME], "away_fouls": self.fouls[AWAY],
                  "one_and_one": self.one_and_one}
        return d


def new_game(home: str, away: str, season: int, game_type: str = "REG", league: League = NBA) -> GameState:
    """A game at the opening tip. Who wins the tip is the first sampled event."""
    return GameState(season=season, game_type=game_type, home=home, away=away, league=league)


def period_length(qtr: int, league: League = NBA) -> int:
    return league.period_length(qtr)


# ---------------------------------------------------------------- legality

def foul_results(s: GameState) -> list[str]:
    """What a foul on the defense can end in. With a bonus, a foul in the bonus always sends the
    shooter to the line, so "no free throws" is legal only before it."""
    lg = s.league
    res = list(lg.results_for_type["foul"])
    if lg.bonus and lg.bonus.level(s.fouls[1 - s.pos] + 1, lg.is_overtime(s.qtr)) > 0:
        res.remove("fts_0")
    return res


def legal_types(s: GameState) -> list[str]:
    types = list(TYPES_FOR_PHASE[s.phase])
    if "timeout" in types and not (s.timeouts[0] > 0 or s.timeouts[1] > 0):
        types.remove("timeout")
    if "run_out" in types and s.qsec > s.league.max_elapsed:
        types.remove("run_out")
    return types


def legal_results(s: GameState, play_type: str) -> list[str]:
    res = list(s.league.results_for_type[play_type])
    if play_type == "timeout":
        res = [r for r, side in (("timeout_offense", s.pos), ("timeout_defense", 1 - s.pos))
               if s.timeouts[side] > 0]
    elif play_type == "free_throw":
        more = s.fts_left > 1 and not s.one_and_one
        res = ["made", "missed"] if more else ["made", "missed_off", "missed_def"]
    elif play_type == "foul" and s.league.bonus:
        res = foul_results(s)
    return res


def elapsed_range(s: GameState, play_type: str) -> tuple[int, int]:
    hi = min(s.league.max_elapsed, max(s.qsec, 0))
    if play_type == "run_out":
        return hi, hi
    return 0, hi


# ---------------------------------------------------------------- transitions

def _start_trip(n: GameState, fts: int) -> None:
    n.phase, n.fts_left, n.fts_total = FREE_THROW, fts, fts


def _end_trip(n: GameState) -> None:
    n.phase, n.fts_left, n.fts_total = LIVE, 0, 0
    n.one_and_one = False


def apply_outcome(s: GameState, o: Outcome) -> GameState:
    """Return the state after outcome `o` is applied to state `s`."""
    if s.over:
        return s.copy()
    n = s.copy()
    off, dfn = s.pos, 1 - s.pos
    t, r = o.play_type, o.result
    bonus, ot = s.league.bonus, s.league.is_overtime(s.qtr)
    fouler = None  # the team charged with a team foul on this event

    if t == "timeout":
        n.timeouts[off if r == "timeout_offense" else dfn] -= 1
    elif t == "jump_ball":
        n.pos = off if r == "won" else dfn
        if s.phase == TIP:
            n.phase = LIVE
            if s.qtr == 1:
                n.tip_winner = n.pos
        if s.league.arrow:
            n.arrow = 1 - n.pos  # the other team gets the next held ball and the next period
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
        if bonus and r in ("made_fouled", "missed_fouled"):
            n.fouls[dfn] += 1
            fouler = dfn
    elif t == "free_throw":
        if r == "made":
            n.score[off] += 1
        n.fts_left -= 1
        if s.one_and_one:
            n.one_and_one = False
            if r != "made":      # missing the first shot of a 1-and-1 ends the trip
                n.fts_left = 0
        if n.fts_left <= 0:
            _end_trip(n)
            if r != "missed_off":
                n.pos = dfn
    elif t == "turnover":
        n.pos = dfn
    elif t == "foul":
        k = int(r[-1])
        if bonus:
            n.fouls[dfn] += 1
            fouler = dfn
        if k:
            _start_trip(n, k)
            n.one_and_one = bool(bonus and k == 2 and bonus.level(n.fouls[dfn], ot) == 1)
    elif t == "technical":
        if r == "tech_offense":
            n.score[off] += 1
        elif r == "tech_defense":
            n.score[dfn] += 1

    n.qsec = 0 if t == "run_out" else max(0, s.qsec - max(0, int(o.elapsed)))
    if fouler is not None:  # the "last two minutes" penalty rule (WNBA), at the clock the foul came
        n.fouls[fouler] = bonus.counted(n.fouls[fouler], n.qsec, ot)
    if n.qsec <= 0 and n.phase != FREE_THROW:
        _end_period(n)
    return n


def timeouts_at_period_start(timeouts: list, qtr: int, lg: League) -> list:
    """Each team's timeouts when period `qtr` (2 on) starts: capped (the NBA's 4 in the 4th quarter,
    the college game's 3 into the second half) or, in overtime, reset (NBA) or added to (college)."""
    if qtr > lg.regulation_periods:
        return [t + lg.ot_timeouts for t in timeouts] if lg.ot_adds_timeouts else [lg.ot_timeouts] * 2
    for period, cap in lg.timeout_caps:
        if qtr == period:
            return [min(t, cap) for t in timeouts]
    return list(timeouts)


def _arrow_start(n: GameState) -> int:
    """Who the possession arrow gives a new period to. Until a toss has set it, it points to the loser
    of the opening tip."""
    tip = HOME if n.tip_winner is None else n.tip_winner
    return (1 - tip) if n.arrow is None else n.arrow


def _end_period(n: GameState) -> None:
    lg = n.league
    diff = n.score[HOME] - n.score[AWAY]
    tip = HOME if n.tip_winner is None else n.tip_winner
    if n.qtr < lg.regulation_periods:
        n.qtr += 1
        n.qsec = lg.period_seconds
        if lg.arrow:
            # A toss follows the arrow: the arrow's team has the ball unless the toss says otherwise
            # (a game's rows do not show every held ball, so the arrow is only a first guess).
            n.phase, n.pos = TIP, _arrow_start(n)
        else:
            n.phase = LIVE
            # The team that lost the opening tip starts the 2nd and 3rd quarters; the winner the 4th.
            n.pos = tip if n.qtr == 4 else 1 - tip
        n.timeouts = timeouts_at_period_start(n.timeouts, n.qtr, lg)
        if lg.bonus and lg.bonus.resets(n.qtr, lg.regulation_periods):
            n.fouls = [0, 0]
    elif diff != 0:
        n.over = True
    else:
        n.qtr += 1
        n.qsec = lg.ot_seconds
        n.phase = TIP
        n.pos = _arrow_start(n) if lg.arrow else HOME
        n.timeouts = timeouts_at_period_start(n.timeouts, n.qtr, lg)
        if lg.bonus and lg.bonus.resets(n.qtr, lg.regulation_periods):
            n.fouls = [0, 0]
    n.fts_left = n.fts_total = 0
    n.one_and_one = False
