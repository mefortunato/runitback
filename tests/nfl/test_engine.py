import random

from runitback.nfl.engine.rules import AWAY, HOME, GameState, Outcome, apply_outcome, legal_types, new_game
from runitback.nfl.vocab import KICKOFF, PAT, SCRIMMAGE


def scrim(**kw):
    base = dict(season=2023, game_type="REG", home="KC", away="BUF", phase=SCRIMMAGE, pos=HOME,
                down=1, ytg=10, yl=75, qtr=1, qsec=600)
    base.update(kw)
    return GameState(**base)


def test_kickoff_received_and_return_td():
    s = new_game("KC", "BUF", 2023, opening_kicker=HOME)
    n = apply_outcome(s, Outcome("kickoff", "kick_received", 75, 5))
    assert (n.phase, n.pos, n.down, n.ytg, n.yl, n.qsec) == (SCRIMMAGE, AWAY, 1, 10, 75, 895)
    n = apply_outcome(s, Outcome("kickoff", "kick_received", 0, 12))
    assert n.phase == PAT and n.pos == AWAY and n.score == [0, 6]
    n = apply_outcome(s, Outcome("kickoff", "kick_recovered", 52, 5))
    assert n.pos == HOME and n.yl == 52


def test_downs_first_down_and_turnover_on_downs():
    n = apply_outcome(scrim(), Outcome("run", "run", 4, 30))
    assert (n.down, n.ytg, n.yl) == (2, 6, 71)
    n = apply_outcome(scrim(down=3, ytg=2), Outcome("pass", "complete", 2, 6))
    assert (n.down, n.ytg, n.yl) == (1, 10, 73)
    n = apply_outcome(scrim(down=4, ytg=2, yl=40), Outcome("run", "run", 1, 5))
    assert (n.pos, n.down, n.yl) == (AWAY, 1, 61)
    n = apply_outcome(scrim(yl=8, ytg=8), Outcome("run", "run", 3, 5))
    assert (n.down, n.ytg, n.yl) == (2, 5, 5)
    n = apply_outcome(scrim(yl=15), Outcome("pass", "complete", 10, 5))
    assert (n.down, n.ytg, n.yl) == (1, 5, 5)  # goal to go


def test_penalties():
    n = apply_outcome(scrim(down=3, ytg=7), Outcome("penalty", "penalty", -5, 0))
    assert (n.down, n.ytg, n.yl) == (3, 12, 80)
    n = apply_outcome(scrim(down=3, ytg=7), Outcome("penalty", "penalty_first_down", 5, 0))
    assert (n.down, n.ytg, n.yl) == (1, 10, 70)


def test_scoring():
    n = apply_outcome(scrim(yl=20), Outcome("pass", "complete", 20, 5))
    assert n.score == [6, 0] and n.phase == PAT and n.pos == HOME
    n = apply_outcome(n, Outcome("extra_point", "good"))
    assert n.score == [7, 0] and n.phase == KICKOFF and n.pos == HOME
    n = apply_outcome(scrim(yl=25, down=4), Outcome("field_goal", "good", 0, 5))
    assert n.score == [3, 0] and n.phase == KICKOFF and n.pos == HOME
    n = apply_outcome(scrim(yl=35, down=4), Outcome("field_goal", "no_good", 65, 5))
    assert n.pos == AWAY and n.yl == 65 and n.phase == SCRIMMAGE
    n = apply_outcome(scrim(yl=98), Outcome("pass", "sack", -2, 5))
    assert n.score == [0, 2] and n.phase == KICKOFF and n.pos == HOME
    n = apply_outcome(scrim(yl=60), Outcome("pass", "interception", 0, 8))
    assert n.score == [0, 6] and n.phase == PAT and n.pos == AWAY


def test_halftime_and_end_of_regulation():
    s = scrim(qtr=2, qsec=5)
    s.opening_kicker = HOME
    n = apply_outcome(s, Outcome("pass", "incomplete", 0, 5))
    assert (n.qtr, n.qsec, n.phase, n.pos, n.timeouts) == (3, 900, KICKOFF, AWAY, [3, 3])
    s = scrim(qtr=4, qsec=3, score=[10, 7])
    assert apply_outcome(s, Outcome("qb_kneel", "kneel", -1, 3)).over
    s = scrim(qtr=4, qsec=3, score=[10, 10])
    n = apply_outcome(s, Outcome("qb_kneel", "kneel", -1, 3), random.Random(0))
    assert (n.qtr, n.qsec, n.phase, n.over) == (5, 600, KICKOFF, False)


def test_pat_after_clock_expires():
    n = apply_outcome(scrim(qtr=4, qsec=4, yl=10, score=[0, 6]), Outcome("pass", "complete", 10, 4))
    assert not n.over and n.phase == PAT and n.qsec == 0
    n = apply_outcome(n, Outcome("extra_point", "good"))
    assert n.over and n.score == [7, 6]


def _ot(season, game_type="REG"):
    s = GameState(season=season, game_type=game_type, home="KC", away="BUF", qtr=5, qsec=600,
                  phase=KICKOFF, pos=AWAY, score=[20, 20], timeouts=[2, 2])
    return apply_outcome(s, Outcome("kickoff", "kick_received", 75, 5))  # KC first possession


def test_ot_modified_rules():
    n = _ot(2020)
    assert n.ot_first == HOME and n.ot_poss == 1
    fg = apply_outcome(n, Outcome("field_goal", "good", 0, 5))
    assert not fg.over  # FG on first possession: other team gets the ball
    k = apply_outcome(fg, Outcome("kickoff", "kick_received", 75, 5))
    assert not k.over and k.pos == AWAY and k.ot_poss == 2
    punt = apply_outcome(k.copy(), Outcome("punt", "kick_received", 80, 5))
    assert punt.over  # BUF's possession ended trailing
    td = apply_outcome(n, Outcome("run", "run", 75, 5))
    assert td.over and td.score == [26, 20]  # TD on first possession wins, no PAT


def test_ot_both_possess_rules_2025():
    n = _ot(2025)
    td = apply_outcome(n, Outcome("run", "run", 75, 5))
    assert not td.over  # BUF still gets a possession
    td = apply_outcome(td, Outcome("extra_point", "good"))
    k = apply_outcome(td, Outcome("kickoff", "kick_received", 75, 5))
    assert not k.over and k.pos == AWAY
    td2 = apply_outcome(k, Outcome("pass", "complete", 75, 5))
    assert not td2.over and td2.phase == PAT  # down 1, may tie with XP
    two = apply_outcome(td2, Outcome("two_point", "good"))
    assert two.over and two.score == [27, 28]


def test_ot_sudden_death_old_and_reg_tie():
    n = _ot(2005)
    assert apply_outcome(n, Outcome("field_goal", "good", 0, 5)).over
    n = _ot(2023)
    n.qsec = 4
    end = apply_outcome(n, Outcome("pass", "incomplete", 0, 5))
    assert end.over and end.score == [20, 20]


def test_timeouts_legality():
    s = scrim(timeouts=[0, 0])
    assert "timeout" not in legal_types(s)
    s = scrim(timeouts=[1, 0])
    n = apply_outcome(s, Outcome("timeout", "timeout_offense"))
    assert n.timeouts == [0, 0]
