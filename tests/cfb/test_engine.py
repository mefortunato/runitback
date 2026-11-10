"""College rules engine: tries from the 3, overtime by season, untimed overtime, no ties."""
import random

import pytest

from runitback.cfb.engine.rules import (
    AWAY, HOME, CFBState, Outcome, apply_outcome, clock_rules, is_shootout, legal_types, new_game,
    ot_format, two_point_only,
)
from runitback.cfb.vocab import KICKOFF, OT_SPOT, PAT, PAT_SPOT, SCRIMMAGE


def scrim(**kw):
    base = dict(season=2023, game_type="REG", home="2", away="333", phase=SCRIMMAGE, pos=HOME,
                down=1, ytg=10, yl=75, qtr=1, qsec=600)
    base.update(kw)
    return CFBState(**base)


def ot(period=1, half=0, first=HOME, season=2023, **kw):
    """A state in overtime period `period`, possession `half` (0 = the first team's)."""
    pos = first if half == 0 else 1 - first
    base = dict(season=season, game_type="REG", home="2", away="333", qtr=4 + period, qsec=0, phase=SCRIMMAGE,
                pos=pos, down=1, ytg=10, yl=OT_SPOT, ot_first=first, ot_half=half, timeouts=[1, 1])
    base.update(kw)
    return CFBState(**base)


def test_touchdown_leads_to_a_try_from_the_3():
    n = apply_outcome(scrim(yl=20), Outcome("pass", "complete", 20, 7))
    assert (n.phase, n.pos, n.yl, n.ytg, n.score) == (PAT, HOME, PAT_SPOT, PAT_SPOT, [6, 0])
    assert legal_types(n) == ["extra_point", "two_point", "timeout"]
    k = apply_outcome(n, Outcome("extra_point", "good", 0, 0))
    assert (k.phase, k.pos, k.score) == (KICKOFF, HOME, [7, 0])
    k = apply_outcome(n, Outcome("two_point", "good", 0, 0))
    assert k.score == [8, 0]


def test_regulation_is_the_nfls():
    s = new_game("2", "333", 2023, opening_kicker=HOME)
    n = apply_outcome(s, Outcome("kickoff", "kick_received", 75, 5))
    assert (n.phase, n.pos, n.down, n.ytg, n.yl, n.qsec) == (SCRIMMAGE, AWAY, 1, 10, 75, 895)
    n = apply_outcome(scrim(down=4, ytg=2, yl=40), Outcome("run", "run", 1, 5))
    assert (n.pos, n.down, n.yl) == (AWAY, 1, 61)  # turnover on downs
    n = apply_outcome(scrim(yl=98), Outcome("run", "run", -2, 5))
    assert n.score == [0, 2] and n.phase == KICKOFF and n.pos == HOME  # safety, free kick
    n = apply_outcome(scrim(qtr=2, qsec=10), Outcome("pass", "incomplete", 0, 10))
    assert (n.qtr, n.qsec, n.phase, n.pos, n.timeouts) == (3, 900, KICKOFF, AWAY, [3, 3])


def test_a_try_is_untimed():
    s = scrim(qtr=4, qsec=3, yl=10, score=[10, 14])
    n = apply_outcome(s, Outcome("pass", "complete", 10, 5))
    assert n.qsec == 0 and n.phase == PAT and not n.over
    n = apply_outcome(n, Outcome("extra_point", "good", 0, 0))
    assert n.over and n.score == [17, 14]


@pytest.mark.parametrize("season,two_point,shootout", [(2004, 3, None), (2018, 3, None), (2019, 3, 5),
                                                       (2020, 3, 5), (2021, 2, 3), (2025, 2, 3)])
def test_ot_format_by_season(season, two_point, shootout):
    assert ot_format(season) == {"two_point": two_point, "shootout": shootout}


def test_tied_regulation_goes_to_overtime_at_the_25_without_a_clock():
    s = scrim(qtr=4, qsec=5, score=[21, 21], timeouts=[0, 2])
    n = apply_outcome(s, Outcome("run", "run", 3, 5), ot_first=AWAY)
    assert (n.qtr, n.qsec, n.phase, n.pos, n.down, n.ytg, n.yl) == (5, 0, SCRIMMAGE, AWAY, 1, 10, OT_SPOT)
    assert (n.ot_first, n.ot_half, n.timeouts, n.over) == (AWAY, 0, [1, 1], False)
    # a postseason game too: never a tie
    n = apply_outcome(scrim(qtr=4, qsec=5, score=[21, 21], game_type="POST"), Outcome("run", "run", 3, 5),
                      rng=random.Random(0))
    assert n.qtr == 5 and not n.over
    # untimed: plays run no clock
    k = apply_outcome(n, Outcome("run", "run", 4, 30))
    assert (k.qtr, k.qsec, k.down, k.yl) == (5, 0, 2, 21)


def test_first_possession_score_then_the_other_team_gets_the_ball():
    s = ot(score=[21, 21])
    n = apply_outcome(s, Outcome("field_goal", "good", 0, 0))
    assert (n.score, n.pos, n.ot_half, n.phase, n.yl, n.down, n.over) == ([24, 21], AWAY, 1, SCRIMMAGE, OT_SPOT, 1, False)
    # touchdown: a try, then the other team's possession
    n = apply_outcome(s, Outcome("run", "run", 25, 0))
    assert (n.phase, n.pos, n.score, n.yl) == (PAT, HOME, [27, 21], PAT_SPOT)
    n = apply_outcome(n, Outcome("extra_point", "good", 0, 0))
    assert (n.score, n.pos, n.ot_half, n.phase, n.yl) == ([28, 21], AWAY, 1, SCRIMMAGE, OT_SPOT)


def test_possession_ends_on_turnover_missed_fg_or_downs():
    for o, kw in ((Outcome("pass", "interception", 40, 0), {}), (Outcome("field_goal", "no_good", 80, 0), {}),
                  (Outcome("run", "run", 1, 0), dict(down=4, ytg=3, yl=12))):
        n = apply_outcome(ot(score=[21, 21], **kw), o)
        assert (n.pos, n.ot_half, n.yl, n.down, n.qtr, n.over) == (AWAY, 1, OT_SPOT, 1, 5, False), o


def test_second_possession_decides_or_starts_the_next_period():
    # first team kicked a field goal; the second team's touchdown wins without a try
    n = apply_outcome(ot(half=1, score=[24, 21]), Outcome("pass", "complete", 25, 0))
    assert n.over and n.score == [24, 27]
    # a field goal to tie: period 2, the team that went second goes first
    n = apply_outcome(ot(half=1, score=[24, 21]), Outcome("field_goal", "good", 0, 0))
    assert (n.qtr, n.pos, n.ot_first, n.ot_half, n.yl, n.score, n.over) == (6, AWAY, AWAY, 0, OT_SPOT, [24, 24], False)
    assert n.timeouts == [1, 1]
    # nothing: the first team's field goal wins
    n = apply_outcome(ot(half=1, score=[24, 21]), Outcome("pass", "interception", 30, 0))
    assert n.over and n.score == [24, 21]
    # the second team's touchdown trails by one: it tries
    n = apply_outcome(ot(half=1, score=[28, 21]), Outcome("run", "run", 25, 0))
    assert (n.phase, n.pos, n.score, n.over) == (PAT, AWAY, [28, 27], False)
    k = apply_outcome(n, Outcome("extra_point", "no_good", 0, 0))
    assert k.over and k.score == [28, 27]
    k = apply_outcome(n, Outcome("extra_point", "good", 0, 0))
    assert (k.qtr, k.pos, k.over) == (6, AWAY, False)


def test_a_defensive_score_ends_overtime():
    n = apply_outcome(ot(score=[21, 21]), Outcome("pass", "interception", 0, 0))
    assert n.over and n.score == [21, 27]
    n = apply_outcome(ot(score=[21, 21], yl=98), Outcome("run", "run", -2, 0))
    assert n.over and n.score == [21, 23]


@pytest.mark.parametrize("season,forced", [(2018, [False, False, True, True, True]),
                                           (2021, [False, True, True, True, True])])
def test_two_point_tries_required_by_period(season, forced):
    for period, must in enumerate(forced, start=1):
        s = ot(period=period, season=season, phase=PAT, yl=PAT_SPOT, ytg=PAT_SPOT, down=0)
        assert two_point_only(s) is must
        assert ("extra_point" in legal_types(s)) is not must


@pytest.mark.parametrize("season,first_shootout", [(2018, None), (2019, 5), (2023, 3)])
def test_shootout_periods_are_one_two_point_try_each(season, first_shootout):
    assert [is_shootout(season, p) for p in range(1, 7)] == \
        [first_shootout is not None and p >= first_shootout for p in range(1, 7)]
    if first_shootout is None:
        return
    # the period before the shootout ends tied: the next one starts with a try, not a possession
    s = ot(period=first_shootout - 1, half=1, season=season, score=[35, 35])
    n = apply_outcome(s, Outcome("pass", "interception", 30, 0))
    assert (n.qtr, n.phase, n.pos, n.yl, n.over) == (4 + first_shootout, PAT, n.ot_first, PAT_SPOT, False)
    assert legal_types(n) == ["two_point", "timeout"]
    a = apply_outcome(n, Outcome("two_point", "good", 0, 0))
    assert (a.phase, a.pos, a.ot_half, a.score[n.pos]) == (PAT, 1 - n.pos, 1, 37)
    b = apply_outcome(a, Outcome("two_point", "no_good", 0, 0))
    assert b.over
    b = apply_outcome(a, Outcome("two_point", "good", 0, 0))
    assert (b.qtr, b.phase, b.pos, b.score, b.over) == (5 + first_shootout, PAT, a.pos, [37, 37], False)


def test_overtime_timeouts_and_state_dict():
    s = ot(score=[21, 21])
    n = apply_outcome(s, Outcome("timeout", "timeout_defense", 0, 0))
    assert n.timeouts == [1, 0] and n.qsec == 0
    assert n.to_dict()["ot_period"] == 1 and n.to_dict()["ot_first"] == "2"
    assert legal_types(CFBState(**{**vars(n), "timeouts": [0, 0]})) == \
        ["run", "pass", "punt", "field_goal", "qb_kneel", "qb_spike", "penalty"]


def test_clock_rules_by_season():
    assert clock_rules(2022) == {"first_down_stops": True, "two_minute_timeout": False, "untimed_overtime": True}
    assert clock_rules(2023)["first_down_stops"] is False
    assert clock_rules(2024)["two_minute_timeout"] is True


def test_random_games_always_finish_with_a_winner():
    """Random legal outcomes: every game ends, decided, through any number of overtimes."""
    from runitback.cfb.engine.rules import legal_results, yards_range
    rng = random.Random(7)
    for i in range(300):
        s = new_game("2", "333", rng.choice([2010, 2019, 2023]), rng=rng)
        for _ in range(5000):
            if s.over:
                break
            t = rng.choice(legal_types(s))
            r = rng.choice(legal_results(s, t))
            lo, hi = yards_range(s, t, r)
            s = apply_outcome(s, Outcome(t, r, rng.randint(lo, hi), rng.randint(0, 40)), rng=rng)
        assert s.over and s.score[HOME] != s.score[AWAY], i
