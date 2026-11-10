import random

import pytest

from runitback.nhl.engine.rules import (
    AWAY,
    HOME,
    GameState,
    Outcome,
    apply_outcome,
    elapsed_range,
    is_legal,
    legal_results,
    legal_sides,
    legal_types,
    new_game,
    run_clock,
    shootout_decided,
)
from runitback.nhl.vocab import (
    FACEOFF,
    LIVE,
    MAX_ELAPSED,
    PENALTY_SHOT,
    PERIOD_SECONDS,
    SHOOTOUT,
)


def live(**kw) -> GameState:
    s = new_game("BOS", "TOR", 2025)
    s.phase = LIVE
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def pen(s, side, kind, elapsed=0):
    return apply_outcome(s, Outcome("penalty", kind, side, elapsed))


def goal(s, side, elapsed=0):
    if s.phase == FACEOFF:
        s = apply_outcome(s, Outcome("faceoff", "won", side))
    o = Outcome("shot", "goal", side, elapsed)
    assert is_legal(s, o)
    return apply_outcome(s, o)


def test_faceoff_starts_play_and_stoppages_need_a_faceoff():
    s = new_game("BOS", "TOR", 2025)
    assert s.phase == FACEOFF and "shot" not in legal_types(s) and "faceoff" in legal_types(s)
    n = apply_outcome(s, Outcome("faceoff", "won", AWAY))
    assert n.phase == LIVE and "faceoff" not in legal_types(n)
    for o in (Outcome("stoppage", "icing"), Outcome("shot", "goal", HOME, 30),
              Outcome("penalty", "minor", AWAY, 30)):
        assert apply_outcome(n, o).phase == FACEOFF
    n = apply_outcome(n, Outcome("shot", "saved", HOME, 12))
    assert n.phase == LIVE and n.psec == PERIOD_SECONDS - 12 and n.score == [0, 0]


def test_minor_power_play_ends_on_a_goal_or_the_clock():
    s = pen(live(), AWAY, "minor")
    assert s.skaters == (5, 4) and s.phase == FACEOFF
    n = goal(s, HOME, 30)
    assert n.score == [1, 0] and n.skaters == (5, 5) and n.pens == [[], []]
    n = run_clock(apply_outcome(s, Outcome("faceoff", "won", HOME)), 119)
    assert n.skaters == (5, 4)
    assert run_clock(n, 1).skaters == (5, 5)
    # a short-handed goal does not end the penalty
    n = goal(s, AWAY, 30)
    assert n.skaters == (5, 4)


def test_double_minor_major_and_misconducts():
    s = pen(live(), AWAY, "double_minor")
    n = goal(s, HOME, 30)
    assert n.pens[AWAY] == [["double_minor", 120]] and n.skaters == (5, 4)  # first half ends
    assert goal(n, HOME).skaters == (5, 5)
    s = pen(live(), AWAY, "major")
    assert goal(s, HOME, 10).skaters == (5, 4)  # majors are served in full
    assert run_clock(s, 300).skaters == (5, 5)
    for kind in ("coincidental", "misconduct"):
        n = pen(live(), AWAY, kind)
        assert n.skaters == (5, 5) and n.phase == FACEOFF


def test_matched_minors_are_four_on_four_and_not_released():
    s = pen(pen(live(), HOME, "matched_minor"), AWAY, "matched_minor")
    assert s.skaters == (4, 4)
    assert goal(s, HOME).skaters == (4, 4)
    # a third minor makes it 4-on-3; a goal by the team with more players ends only that one
    s = pen(s, HOME, "minor")
    assert s.skaters == (3, 4)
    n = goal(s, AWAY)
    assert n.skaters == (4, 4) and [p[0] for p in n.pens[HOME]] == ["matched_minor"]


def test_stacked_penalties_and_three_skater_minimum():
    s = pen(pen(pen(live(), AWAY, "minor"), AWAY, "minor", 30), AWAY, "minor", 30)
    assert s.skaters == (5, 3) and s.pens[AWAY] == [["minor", 60], ["minor", 90], ["minor", 120]]
    # the first expires; the third, which was waiting, then starts its full two minutes
    n = run_clock(s, 60)
    assert n.skaters == (5, 3) and n.pens[AWAY] == [["minor", 30], ["minor", 120]]
    n = run_clock(n, 30)
    assert n.pens[AWAY] == [["minor", 90]] and n.skaters == (5, 4)
    assert run_clock(n, 90).skaters == (5, 5)
    # five penalties on one team: still three skaters, and the other team stays at five
    s = live()
    for _ in range(5):
        s = pen(s, AWAY, "minor")
    assert s.skaters == (5, 3)


def test_penalties_carry_over_periods():
    s = pen(live(psec=30), AWAY, "minor")
    s = apply_outcome(s, Outcome("faceoff", "won", HOME))
    n = apply_outcome(s, Outcome("period_end", "end", None, 30))
    assert n.period == 2 and n.phase == FACEOFF and n.pens[AWAY] == [["minor", 90]] and n.skaters == (5, 4)


def test_empty_net_adds_an_attacker_until_the_goalie_returns_or_the_period_ends():
    s = apply_outcome(live(psec=90), Outcome("goalie_pull", "goalie_pull", HOME, 5))
    assert s.skaters == (6, 5) and legal_sides(s, "goalie_pull") == [AWAY]
    assert legal_sides(s, "goalie_return") == [HOME]
    n = pen(s, AWAY, "minor")
    assert n.skaters == (6, 4)
    assert apply_outcome(s, Outcome("goalie_return", "goalie_return", HOME)).skaters == (5, 5)
    n = apply_outcome(s, Outcome("period_end", "end", None))
    assert n.period == 2 and n.empty_net == [False, False]


def test_the_period_ends_only_when_the_clock_runs_out():
    assert "period_end" not in legal_types(live(psec=MAX_ELAPSED + 1))
    s = live(psec=40)
    assert "period_end" in legal_types(s) and elapsed_range(s, "period_end") == (40, 40)
    assert "period_end" not in legal_types(apply_outcome(s, Outcome("stoppage", "icing")))
    zero = live(psec=0)
    assert set(legal_types(zero)) == {"period_end", "penalty", "stoppage"}
    assert elapsed_range(new_game("BOS", "TOR", 2025), "faceoff") == (0, 0)


def test_regulation_end_and_overtime_rules():
    n = apply_outcome(live(period=3, psec=10, score=[3, 2]), Outcome("period_end", "end", None, 10))
    assert n.over and n.score == [3, 2]
    n = apply_outcome(live(period=3, psec=10, score=[2, 2]), Outcome("period_end", "end", None, 10))
    assert (n.period, n.psec, n.phase, n.skaters) == (4, 300, FACEOFF, (3, 3))
    old = live(period=3, psec=10, score=[2, 2], season=2015)
    assert apply_outcome(old, Outcome("period_end", "end", None, 10)).skaters == (4, 4)
    playoff = live(period=3, psec=10, score=[2, 2], game_type="POST")
    n = apply_outcome(playoff, Outcome("period_end", "end", None, 10))
    assert (n.period, n.psec, n.skaters) == (4, PERIOD_SECONDS, (5, 5))
    n = apply_outcome(live(period=4, psec=PERIOD_SECONDS, game_type="POST", score=[2, 2]),
                      Outcome("period_end", "end", None))
    assert n.period == 5 and not n.over
    n = goal(live(period=6, psec=500, game_type="POST", score=[2, 2]), AWAY, 30)
    assert n.over and n.score == [2, 3]


def test_three_on_three_penalties_add_skaters():
    s = live(period=4, psec=300, score=[1, 1])
    assert s.skaters == (3, 3)
    s = pen(s, AWAY, "minor")
    assert s.skaters == (4, 3)
    assert pen(s, AWAY, "minor").skaters == (5, 3)
    assert pen(s, HOME, "minor").skaters == (3, 3)
    # the penalty ends during play: 4-on-4 until the next stoppage, then 3-on-3
    s = apply_outcome(s, Outcome("faceoff", "won", HOME))
    n = apply_outcome(s, Outcome("shot", "saved", HOME, 125))
    assert n.pens == [[], []] and n.skaters == (4, 4)
    assert apply_outcome(n, Outcome("stoppage", "goalie_freeze", None, 3)).skaters == (3, 3)


def test_a_penalty_carried_into_overtime():
    s = pen(live(period=3, psec=30, score=[2, 2]), AWAY, "minor")
    s = apply_outcome(s, Outcome("faceoff", "won", HOME))
    n = apply_outcome(s, Outcome("period_end", "end", None, 30))
    assert n.period == 4 and n.skaters == (4, 3) and n.pens[AWAY] == [["minor", 90]]


def test_penalty_shot():
    s = pen(live(), AWAY, "penalty_shot")
    assert s.phase == PENALTY_SHOT and s.skaters == (1, 0)
    assert legal_types(s) == ["shot"] and legal_sides(s, "shot") == [HOME]
    assert "blocked" not in legal_results(s, "shot")
    n = apply_outcome(s, Outcome("shot", "goal", HOME))
    assert n.phase == FACEOFF and n.score == [1, 0] and n.pens == [[], []]
    n = apply_outcome(s, Outcome("shot", "saved", HOME))
    assert n.phase == FACEOFF and n.score == [0, 0]
    assert "penalty_shot" not in legal_results(new_game("BOS", "TOR", 2025), "penalty")


def so(s, side, result):
    assert is_legal(s, Outcome("shootout", result, side)), (side, result)
    return apply_outcome(s, Outcome("shootout", result, side))


def test_tie_after_overtime_goes_to_a_shootout():
    s = live(period=4, psec=10, score=[2, 2])
    s = apply_outcome(s, Outcome("period_end", "end", None, 10))
    assert s.phase == SHOOTOUT and legal_types(s) == ["shootout"]
    assert legal_sides(s, "shootout") == [HOME, AWAY]
    s = so(s, AWAY, "goal")
    assert legal_sides(s, "shootout") == [HOME] and s.skaters == (1, 0)
    s = so(s, HOME, "saved")
    s = so(s, AWAY, "goal")
    assert not s.over
    s = so(s, HOME, "missed")   # 0-2 with one attempt left: home cannot catch up
    assert s.over and s.score == [2, 3]


def test_shootout_sudden_death():
    s = live(period=5, phase=SHOOTOUT, psec=0, score=[1, 1])
    for side, res in [(HOME, "goal"), (AWAY, "goal"), (HOME, "saved"), (AWAY, "saved"),
                      (HOME, "saved"), (AWAY, "saved"), (HOME, "goal")]:
        s = so(s, side, res)
        assert not s.over
    s = so(s, AWAY, "goal")      # round 4: both score
    assert not s.over
    s = so(s, HOME, "saved")
    s = so(s, AWAY, "goal")
    assert s.over and s.score == [1, 2]


@pytest.mark.parametrize("shots,goals,decided", [
    ([3, 3], [1, 1], False), ([3, 3], [2, 1], True), ([2, 1], [2, 0], False), ([2, 2], [2, 0], True),
    ([4, 3], [2, 1], False), ([4, 4], [2, 1], True), ([1, 0], [1, 0], False),
])
def test_shootout_decided(shots, goals, decided):
    assert shootout_decided(shots, goals) == decided


def random_outcome(s: GameState, rng: random.Random) -> Outcome:
    t = rng.choice(legal_types(s))
    lo, hi = elapsed_range(s, t)
    return Outcome(t, rng.choice(legal_results(s, t)), rng.choice(legal_sides(s, t)), rng.randint(lo, hi))


@pytest.mark.parametrize("game_type,season", [("REG", 2025), ("REG", 2012), ("POST", 2025)])
def test_random_legal_play_always_finishes_a_legal_game(game_type, season):
    rng = random.Random(0)
    for _ in range(40):
        s = new_game("BOS", "TOR", season, game_type)
        for _ in range(20000):
            if s.over:
                break
            o = random_outcome(s, rng)
            assert is_legal(s, o)
            s = apply_outcome(s, o)
            h, a = s.skaters
            if s.phase in (FACEOFF, LIVE):
                assert 3 <= h <= 6 and 3 <= a <= 6, s.to_dict()
            assert s.psec >= 0
        assert s.over and s.score[HOME] != s.score[AWAY]
        assert s.period >= 3 and (game_type == "REG" or s.phase != SHOOTOUT)
