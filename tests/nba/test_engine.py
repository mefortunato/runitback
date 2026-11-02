import pytest

from runitback.nba.engine.rules import AWAY, HOME, GameState, Outcome, apply_outcome, legal_results, legal_types, new_game
from runitback.nba.vocab import FREE_THROW, LIVE, MAX_ELAPSED, OT_SECONDS, QUARTER_SECONDS, TIP


def live(**kw) -> GameState:
    s = new_game("BOS", "NYK", 2026)
    s.phase, s.tip_winner = LIVE, HOME
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_tip_off_sets_possession_and_tip_winner():
    s = new_game("BOS", "NYK", 2026)
    assert s.phase == TIP and legal_types(s) == ["jump_ball"]
    n = apply_outcome(s, Outcome("jump_ball", "lost"))
    assert n.phase == LIVE and n.pos == AWAY and n.tip_winner == AWAY


@pytest.mark.parametrize("t,r,pts,pos", [
    ("shot2", "made", 2, AWAY), ("shot3", "made", 3, AWAY), ("shot2", "missed_off", 0, HOME),
    ("shot3", "missed_def", 0, AWAY),
])
def test_shots(t, r, pts, pos):
    n = apply_outcome(live(), Outcome(t, r, 10))
    assert n.score == [pts, 0] and n.pos == pos and n.qsec == QUARTER_SECONDS - 10 and n.phase == LIVE


def test_and_one_then_free_throw_switches_possession():
    n = apply_outcome(live(), Outcome("shot2", "made_fouled", 5))
    assert n.score == [2, 0] and n.phase == FREE_THROW and n.fts_left == 1 and n.pos == HOME
    assert legal_results(n, "free_throw") == ["made", "missed_off", "missed_def"]
    n = apply_outcome(n, Outcome("free_throw", "made"))
    assert n.score == [3, 0] and n.phase == LIVE and n.pos == AWAY


def test_missed_three_fouled_gives_three_free_throws():
    n = apply_outcome(live(), Outcome("shot3", "missed_fouled", 5))
    assert n.fts_left == 3 and legal_results(n, "free_throw") == ["made", "missed"]
    assert set(legal_types(n)) == {"free_throw", "technical", "timeout"}
    n = apply_outcome(n, Outcome("free_throw", "missed"))
    n = apply_outcome(n, Outcome("free_throw", "made"))
    n = apply_outcome(n, Outcome("free_throw", "missed_off"))
    assert n.score == [1, 0] and n.phase == LIVE and n.pos == HOME


def test_foul_trip_and_turnover_and_technical():
    n = apply_outcome(live(pos=AWAY), Outcome("foul", "fts_2", 8))
    assert n.phase == FREE_THROW and n.fts_left == 2 and n.pos == AWAY
    n = apply_outcome(live(), Outcome("turnover", "turnover", 3))
    assert n.pos == AWAY
    n = apply_outcome(live(), Outcome("technical", "tech_defense"))
    assert n.score == [0, 1] and n.pos == HOME


def test_timeouts():
    s = live(timeouts=[1, 0])
    assert legal_results(s, "timeout") == ["timeout_offense"]
    n = apply_outcome(s, Outcome("timeout", "timeout_offense"))
    assert n.timeouts == [0, 0] and "timeout" not in legal_types(n)


def test_quarter_possession_follows_the_opening_tip():
    s = live(tip_winner=AWAY, pos=HOME, qsec=10)
    n = apply_outcome(s, Outcome("shot2", "missed_def", 10))
    assert n.qtr == 2 and n.qsec == QUARTER_SECONDS and n.pos == HOME  # tip loser starts Q2
    n = apply_outcome(live(tip_winner=AWAY, qtr=2, qsec=5), Outcome("run_out", "end", 5))
    assert n.qtr == 3 and n.pos == HOME
    n = apply_outcome(live(tip_winner=AWAY, qtr=3, qsec=5, timeouts=[6, 2]), Outcome("run_out", "end", 5))
    assert n.qtr == 4 and n.pos == AWAY and n.timeouts == [4, 2]


def test_free_throws_after_the_buzzer_are_shot_before_the_period_ends():
    n = apply_outcome(live(qsec=1), Outcome("shot2", "missed_fouled", 1))
    assert n.qsec == 0 and n.qtr == 1 and n.phase == FREE_THROW
    n = apply_outcome(n, Outcome("free_throw", "made"))
    n = apply_outcome(n, Outcome("free_throw", "made"))
    assert n.qtr == 2 and n.score == [2, 0]


def test_tied_game_goes_to_overtime_and_ends_without_a_tie():
    s = live(qtr=4, qsec=3, score=[100, 100])
    n = apply_outcome(s, Outcome("run_out", "end", 3))
    assert not n.over and n.qtr == 5 and n.qsec == OT_SECONDS and n.phase == TIP and n.timeouts == [2, 2]
    s = live(qtr=5, qsec=2, score=[110, 108])
    n = apply_outcome(s, Outcome("shot3", "missed_def", 2))
    assert n.over and n.score == [110, 108]


def test_run_out_only_near_the_end_of_a_period():
    assert "run_out" not in legal_types(live(qsec=MAX_ELAPSED + 1))
    assert "run_out" in legal_types(live(qsec=MAX_ELAPSED))
