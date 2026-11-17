"""Basketball leagues: the rules and sizes the engine, features and model are parameterized by.

The NBA event generator (``runitback.nba``) models one event at a time and lets a deterministic rules
engine own the game state (``engine.rules``). Everything that differs between leagues lives in a
:class:`League`, so another league (the college game here, the WNBA in W1) is a new value, not a
fork of the engine:

- the clock: period length, regulation periods (quarters or halves), overtime, the shot clock;
- timeouts: how many, caps when a period starts, how overtime adds;
- fouls: the bonus (1-and-1 / double bonus) and when team fouls reset;
- how a period or an overtime starts: a jump ball (NBA) or the possession arrow (college);
- the teams: a fixed list (NBA) or the teams found in the data (college: 360+ Division I schools and
  one pooled non-Division I opponent);
- the vocabulary of results and the model's input features, which grow with the rules (college adds
  a "foul, no free throws" result and the team-foul counts).

``NBA`` reproduces the original engine exactly (``tests/nba/test_league.py`` pins it); the registry
:data:`LEAGUES` maps a checkpoint's ``league`` name back to its rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from .vocab import (
    MAX_ELAPSED, MAX_Q4_TIMEOUTS, OT_SECONDS, OT_TIMEOUTS, QUARTER_SECONDS, RESULTS, RESULTS_FOR_TYPE, TEAMS,
    TIMEOUTS_PER_GAME,
)

# What the college "foul" event can end in: no free throws, or a trip of 1 to 3 (a bonus foul is
# "fts_2": a 1-and-1 when the fouling team is in the single bonus, two shots otherwise).
FOUL_RESULTS_COLLEGE = ["fts_0", "fts_1", "fts_2", "fts_3"]

BASE_FEATURES = [
    "qtr", "qsec", "phase", "fts_left", "diff", "to_off", "to_def", "is_home",
    "off_team", "def_team", "off_ts", "def_ts", "season", "game_type",
    "prev_type", "prev_result", "prev_elapsed",
]
FOUL_FEATURES = ["fouls_off", "fouls_def"]


@dataclass(frozen=True)
class Bonus:
    """The free-throw bonus: team fouls committed in the current bonus period.

    ``one_and_one_at`` is the foul that starts the 1-and-1 (men: the 7th in a half), ``double_at`` the
    foul that starts two shots (men: the 10th; women: the 5th in a quarter, with no 1-and-1).
    ``reset_periods`` are the periods whose start resets the counts (halves: 2; quarters: 2, 3, 4);
    overtime keeps the count of the last regulation period.
    """

    double_at: int
    one_and_one_at: int | None = None
    reset_periods: tuple[int, ...] = ()

    def level(self, fouls: int) -> int:
        """0 = no bonus, 1 = 1-and-1, 2 = two shots, for a team that has committed ``fouls``."""
        if fouls >= self.double_at:
            return 2
        if self.one_and_one_at is not None and fouls >= self.one_and_one_at:
            return 1
        return 0


@dataclass(frozen=True)
class League:
    name: str                       # "nba" | "cbb" | "wcbb": checkpoints store it
    label: str
    period_seconds: int             # a regulation period: a quarter or a half
    regulation_periods: int         # 4 quarters or 2 halves
    ot_seconds: int
    timeouts: int                   # per team at the opening tip
    ot_timeouts: int                # NBA: set to this at each overtime. College: added at each overtime
    ot_adds_timeouts: bool
    timeout_caps: tuple[tuple[int, int], ...]   # (period, cap on timeouts when that period starts)
    shot_clock: int                 # seconds
    max_elapsed: int                # longest clock gap one event can span (two shot clocks, NBA: 60)
    bonus: Bonus | None             # None: team fouls are not tracked
    arrow: bool                     # periods and held balls follow the possession arrow
    teams: tuple[str, ...] | None   # None: the teams found in the training data
    first_season: int
    min_events: int = 150           # a processed game needs at least this many events
    train_seasons: int | None = None  # seasons of games a fold's generator trains on (None: all)
    # pre-game ratings (data.ratings): EWMA weight of the newest game, share carried to a new season
    rating_alpha: float = 0.08
    rating_carry: float = 0.75

    def __post_init__(self):
        if self.max_elapsed > MAX_ELAPSED:
            raise ValueError(f"max_elapsed {self.max_elapsed} exceeds the vocabulary's {MAX_ELAPSED}")

    # ------------------------------------------------------------ clock
    def period_length(self, qtr: int) -> int:
        return self.period_seconds if qtr <= self.regulation_periods else self.ot_seconds

    @property
    def n_qsec(self) -> int:
        """Clock buckets: 15-second steps over a regulation period."""
        return self.period_seconds // 15 + 1

    def bucket_qsec(self, qsec: float) -> int:
        return int(min(max(qsec, 0), self.period_seconds) // 15)

    # ------------------------------------------------------------ vocabulary
    @cached_property
    def results(self) -> list[str]:
        return RESULTS + ["fts_0"] if self.bonus else RESULTS

    @cached_property
    def rs(self) -> dict[str, int]:
        return {r: i for i, r in enumerate(self.results)}

    @property
    def n_results(self) -> int:
        return len(self.results)

    @cached_property
    def results_for_type(self) -> dict[str, list[str]]:
        if not self.bonus:
            return RESULTS_FOR_TYPE
        return RESULTS_FOR_TYPE | {"foul": FOUL_RESULTS_COLLEGE}

    @cached_property
    def features(self) -> list[str]:
        return BASE_FEATURES + FOUL_FEATURES if self.bonus else BASE_FEATURES

    @property
    def n_fts(self) -> int:
        """free-throws-left buckets 0..3, plus "first shot of a 1-and-1" for leagues with one."""
        return 5 if self.bonus and self.bonus.one_and_one_at is not None else 4

    @property
    def n_fouls(self) -> int:
        """Team-foul buckets: 0 .. double_at + 1 (that many or more)."""
        return self.bonus.double_at + 2 if self.bonus else 1

    @property
    def state_cols(self) -> list[str]:
        """Columns of ``plays.parquet`` that rebuild a GameState."""
        base = ["qtr", "qsec", "phase", "pos", "fts_left", "score_h", "score_a", "to_h", "to_a"]
        return base + ["foul_h", "foul_a", "oo", "arrow"] if self.bonus else base

    @property
    def dynamic_teams(self) -> bool:
        return self.teams is None


NBA = League(
    name="nba", label="NBA", period_seconds=QUARTER_SECONDS, regulation_periods=4, ot_seconds=OT_SECONDS,
    timeouts=TIMEOUTS_PER_GAME, ot_timeouts=OT_TIMEOUTS, ot_adds_timeouts=False,
    timeout_caps=((4, MAX_Q4_TIMEOUTS),), shot_clock=24, max_elapsed=MAX_ELAPSED, bonus=None, arrow=False,
    teams=tuple(TEAMS), first_season=2002,
)

# College basketball (ESPN's data starts in 2006; the sportsdataverse releases are dense from about 2017
# for men and 2021 for women). Four timeouts a team, one more per overtime. The 30-second shot clock
# (2015-16 on) makes two clocks 60 seconds, the vocabulary's longest clock gap. Fold generators train on
# the five seasons before the test season: the rules (timeouts, media timeouts, quarters for women) changed
# over the years, and older games add training time without adding the current game.
CBB = League(
    name="cbb", label="Men's college basketball", period_seconds=1200, regulation_periods=2, ot_seconds=300,
    timeouts=4, ot_timeouts=1, ot_adds_timeouts=True,
    # no more than three timeouts carry into the second half
    timeout_caps=((2, 3),), shot_clock=30, max_elapsed=60,
    bonus=Bonus(double_at=10, one_and_one_at=7, reset_periods=(2,)),
    arrow=True, teams=None, first_season=2017, min_events=120, train_seasons=5,
    rating_alpha=0.10, rating_carry=0.70,
)
WCBB = League(
    name="wcbb", label="Women's college basketball", period_seconds=600, regulation_periods=4, ot_seconds=300,
    timeouts=4, ot_timeouts=1, ot_adds_timeouts=True, timeout_caps=(), shot_clock=30, max_elapsed=60,
    # quarters: the 5th team foul of a quarter is two shots (no 1-and-1); counts reset each quarter
    bonus=Bonus(double_at=5, one_and_one_at=None, reset_periods=(2, 3, 4)),
    arrow=True, teams=None, first_season=2017, min_events=120, train_seasons=5,
    rating_alpha=0.10, rating_carry=0.70,
)

LEAGUES = {lg.name: lg for lg in (NBA, CBB, WCBB)}


def get_league(league: "str | League | None") -> League:
    """A League by name (a checkpoint's ``league``); None is the NBA, which older checkpoints predate."""
    if isinstance(league, League):
        return league
    try:
        return LEAGUES[league or "nba"]
    except KeyError:
        raise ValueError(f"unknown league {league!r}: one of {sorted(LEAGUES)}") from None
