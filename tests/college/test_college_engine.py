"""The basketball engine under college rules (C2): halves or quarters, the possession arrow, timeouts,
team fouls and the 1-and-1 / double bonus."""
from __future__ import annotations

import numpy as np
import pytest

from runitback.nba.engine.rules import (
    AWAY, HOME, GameState, Outcome, apply_outcome, elapsed_range, legal_results, legal_types, new_game,
)
from runitback.nba.league import CBB, WCBB
from runitback.nba.model.features import TeamSeasonVocab, encode_state, result_mask
from runitback.nba.vocab import FREE_THROW, LIVE, TIP


def live(league=CBB, **kw) -> GameState:
    s = new_game("Duke", "UNC", 2026, league=league)
    s.phase, s.tip_winner, s.arrow = LIVE, HOME, AWAY
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def play(s: GameState, *events: tuple) -> GameState:
    for t, r, *e in events:
        s = apply_outcome(s, Outcome(t, r, e[0] if e else 0))
    return s


def test_new_college_game_has_the_leagues_clock_and_timeouts():
    m, w = new_game("Duke", "UNC", 2026, league=CBB), new_game("UConn", "SC", 2026, league=WCBB)
    assert (m.qsec, m.timeouts, m.fouls) == (1200, [4, 4], [0, 0])
    assert (w.qsec, w.timeouts) == (600, [4, 4])
    assert m.phase == TIP and legal_types(m) == ["jump_ball"]
    assert m.to_dict()["home_fouls"] == 0 and "home_fouls" not in new_game("BOS", "NYK", 2026).to_dict()


def test_halves_and_the_second_half_toss():
    s = new_game("Duke", "UNC", 2026, league=CBB)
    s = play(s, ("jump_ball", "won"))                       # Duke wins the opening tip
    assert s.tip_winner == HOME and s.pos == HOME and s.arrow == AWAY
    s.qsec = 4
    s = play(s, ("shot2", "made", 4))                        # the first half ends
    assert (s.qtr, s.qsec, s.phase, s.pos) == (2, 1200, TIP, AWAY)  # the arrow's team has the ball
    assert legal_types(s) == ["jump_ball"]
    s = play(s, ("jump_ball", "lost"))                       # the toss says UNC doesn't: Duke gets it
    assert (s.phase, s.pos, s.arrow) == (LIVE, HOME, AWAY)
    s.qsec, s.score = 3, [30, 30]
    s = play(s, ("run_out", "end", 3))                       # a tie at the end of the second half: overtime
    assert (s.qtr, s.qsec, s.phase, s.over) == (3, 300, TIP, False)
    s.qsec, s.score = 2, [60, 58]
    assert play(s, ("run_out", "end", 2)).over


def test_women_play_four_quarters_with_an_overtime_after():
    s = live(WCBB, qtr=3, qsec=5)
    s = play(s, ("run_out", "end", 5))
    assert (s.qtr, s.qsec, s.phase) == (4, 600, TIP)
    s.phase, s.qsec, s.score = LIVE, 2, [50, 50]
    s = play(s, ("run_out", "end", 2))
    assert (s.qtr, s.qsec, s.phase) == (5, 300, TIP)


def test_held_ball_moves_the_arrow():
    s = live(pos=HOME, arrow=AWAY)
    s = play(s, ("jump_ball", "lost", 1))                    # the arrow gives it to the other team
    assert (s.pos, s.arrow) == (AWAY, HOME)
    s = play(s, ("jump_ball", "won", 1))                     # the next one stays with the team that has it
    assert (s.pos, s.arrow) == (AWAY, HOME)


def test_timeouts_cap_into_the_second_half_and_add_in_overtime():
    s = live(qsec=3, timeouts=[4, 1])
    s = play(s, ("run_out", "end", 3))
    assert s.timeouts == [3, 1]                              # no more than three carry into the second half
    s.phase, s.qsec, s.score = LIVE, 2, [10, 10]
    s = play(s, ("run_out", "end", 2))
    assert s.timeouts == [4, 2]                              # one more for each team in overtime
    w = live(WCBB, qtr=4, qsec=2, timeouts=[3, 0], score=[1, 1])
    assert play(w, ("run_out", "end", 2)).timeouts == [4, 1]


def test_run_out_follows_the_leagues_clock_gap():
    assert "run_out" not in legal_types(live(qsec=61)) and "run_out" in legal_types(live(qsec=60))
    assert elapsed_range(live(qsec=1200), "shot2") == (0, 60)


def foul_on_defense(s: GameState, result: str, elapsed: int = 5) -> GameState:
    return apply_outcome(s, Outcome("foul", result, elapsed))


def test_team_fouls_count_and_a_foul_before_the_bonus_may_have_no_free_throws():
    s = live(fouls=[0, 3])                                   # Duke has the ball; UNC has 3 fouls
    assert legal_results(s, "foul") == ["fts_0", "fts_1", "fts_2", "fts_3"]
    n = foul_on_defense(s, "fts_0")
    assert n.fouls == [0, 4] and n.phase == LIVE and n.pos == HOME
    # a shooting foul is a team foul too
    assert apply_outcome(s, Outcome("shot2", "missed_fouled", 5)).fouls == [0, 4]
    assert apply_outcome(s, Outcome("shot3", "made_fouled", 5)).fouls == [0, 4]
    # offensive fouls and technicals are not
    assert apply_outcome(s, Outcome("turnover", "turnover", 5)).fouls == [0, 3]
    assert apply_outcome(s, Outcome("technical", "tech_offense", 0)).fouls == [0, 3]


def test_seventh_foul_of_a_half_is_a_one_and_one():
    s = live(fouls=[0, 6])
    assert "fts_0" not in legal_results(s, "foul")           # the 7th foul always sends the shooter to the line
    n = foul_on_defense(s, "fts_2")
    assert n.fouls == [0, 7] and n.phase == FREE_THROW and n.fts_left == 2 and n.one_and_one
    assert legal_results(n, "free_throw") == ["made", "missed_off", "missed_def"]
    made = apply_outcome(n, Outcome("free_throw", "made"))   # the first shot made: a second one
    assert made.score == [1, 0] and made.phase == FREE_THROW and made.fts_left == 1 and not made.one_and_one
    assert legal_results(made, "free_throw") == ["made", "missed_off", "missed_def"]
    done = apply_outcome(made, Outcome("free_throw", "made"))
    assert done.score == [2, 0] and done.phase == LIVE and done.pos == AWAY
    for miss, pos in (("missed_def", AWAY), ("missed_off", HOME)):   # the first shot missed: the trip is over
        out = apply_outcome(n, Outcome("free_throw", miss))
        assert out.score == [0, 0] and out.phase == LIVE and out.fts_left == 0 and out.pos == pos
        assert not out.one_and_one


def test_tenth_foul_of_a_half_is_two_shots_and_nothing_is_a_one_and_one_after():
    s = live(fouls=[0, 9])
    n = foul_on_defense(s, "fts_2")
    assert n.fouls == [0, 10] and n.fts_left == 2 and not n.one_and_one
    assert legal_results(n, "free_throw") == ["made", "missed"]
    assert not foul_on_defense(live(fouls=[0, 12]), "fts_2").one_and_one
    # a shooting foul in the single bonus is the shot's own free throws, never a 1-and-1
    f = apply_outcome(live(fouls=[0, 7]), Outcome("shot2", "missed_fouled", 5))
    assert f.fts_left == 2 and not f.one_and_one


def test_team_fouls_reset_in_the_second_half_but_not_in_overtime():
    s = live(qsec=2, fouls=[8, 11])
    s = play(s, ("run_out", "end", 2))
    assert (s.qtr, s.fouls) == (2, [0, 0])
    s.phase, s.qsec, s.score, s.fouls = LIVE, 2, [5, 5], [3, 12]
    s = play(s, ("run_out", "end", 2))
    assert (s.qtr, s.fouls) == (3, [3, 12])                  # overtime carries the second half's count


def test_women_get_two_shots_from_the_fifth_foul_of_a_quarter():
    s = live(WCBB, fouls=[4, 4])
    assert legal_results(s, "foul") == ["fts_1", "fts_2", "fts_3"]  # the 5th foul sends a team to the line
    n = foul_on_defense(live(WCBB, fouls=[0, 4]), "fts_2")
    assert n.fts_left == 2 and not n.one_and_one and legal_results(n, "free_throw") == ["made", "missed"]
    q = live(WCBB, qsec=2, fouls=[6, 7])
    q = play(q, ("run_out", "end", 2))
    assert (q.qtr, q.fouls) == (2, [0, 0])
    q.phase, q.qsec = LIVE, 2
    q.fouls = [2, 3]
    assert play(q, ("run_out", "end", 2)).fouls == [0, 0]    # every quarter's start resets


def test_model_inputs_carry_the_fouls_and_the_one_and_one():
    ts = TeamSeasonVocab(["Duke_2026"], CBB, ["Duke", "UNC"])
    n = foul_on_defense(live(fouls=[2, 6]), "fts_2")
    f = encode_state(n, None, ts)
    assert f["fouls_off"] == 2 and f["fouls_def"] == 7 and f["fts_left"] == 4 and f["qsec"] == 1195 // 15
    assert f["off_team"] == 1 and f["def_team"] == 2
    assert result_mask(live(fouls=[0, 7]), "foul").tolist() == [r in ("fts_1", "fts_2", "fts_3") for r in CBB.results]


@pytest.mark.parametrize("league", [CBB, WCBB])
def test_random_legal_games_always_end_without_a_tie(league):
    rng = np.random.default_rng(7)
    for g in range(30):
        s = new_game("Duke", "UNC", 2026, league=league)
        for _ in range(1500):
            if s.over:
                break
            t = rng.choice(legal_types(s))
            r = rng.choice(legal_results(s, t))
            lo, hi = elapsed_range(s, t)
            e = int(hi if t == "run_out" else rng.integers(lo, hi + 1))
            s = apply_outcome(s, Outcome(str(t), str(r), e))
            assert 0 <= s.qsec <= league.period_length(s.qtr) and min(s.timeouts) >= 0 or s.over
            assert s.fouls[0] >= 0 and s.fouls[1] >= 0
        assert s.over and s.score[HOME] != s.score[AWAY]
