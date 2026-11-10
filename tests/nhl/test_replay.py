from pathlib import Path

import pytest

from runitback.nhl.engine.replay import (
    Event,
    _merge_same_player_minors,
    _order_penalty_shots,
    game_events,
    label_penalties,
    load_season_file,
    order_shootout,
    penalty_kind,
    repair_events,
    replay_game,
    replay_season,
    season_of,
    split_games,
    stop_reason,
)
from runitback.nhl.engine.rules import AWAY, HOME, new_game, shootout_decided
from runitback.nhl.vocab import LIVE

FIXTURE = Path(__file__).parent / "fixtures" / "nhl_pbp_2025_games.parquet"
RAW = Path(__file__).resolve().parents[2] / "data" / "nhl" / "raw"


def test_season_names_and_labels():
    assert season_of("20242025") == 2025 and season_of(20102011) == 2011
    assert stop_reason("goalie-stopped-after-sog") == "goalie_freeze"
    assert stop_reason("tv-timeout,icing") == "icing"
    assert stop_reason("puck-in-netting") == "out_of_play" and stop_reason(None) == "other"
    assert penalty_kind("Minor", 2.0, "tripping") == "minor"
    assert penalty_kind("Minor", 4.0, "high-sticking-double-minor") == "double_minor"
    assert penalty_kind("Major", 5.0, "fighting") == "major"
    assert penalty_kind("Match", 10.0, "match-penalty") == "major"
    assert penalty_kind("Misconduct", 10.0, "misconduct") == "misconduct"
    assert penalty_kind("Minor", 0.0, "ps-hooking-on-breakaway") == "penalty_shot"


def pens(*spec):
    return [Event("penalty", kind, side, 1, 100) for side, kind in spec]


def test_one_minor_each_at_full_strength_is_four_on_four():
    group = pens((HOME, "minor"), (AWAY, "minor"))
    label_penalties(group, new_game("BOS", "TOR", 2025))
    assert [e.result for e in group] == ["matched_minor", "matched_minor"]
    # with a penalty already on the clock, they cancel instead
    s = new_game("BOS", "TOR", 2025)
    s.pens[HOME].append(["minor", 50])
    group = pens((HOME, "minor"), (AWAY, "minor"))
    label_penalties(group, s)
    assert [e.result for e in group] == ["coincidental", "coincidental"]


def test_fights_cancel_and_the_extra_minor_is_a_power_play():
    group = pens((HOME, "major"), (AWAY, "major"), (HOME, "minor"))
    label_penalties(group, new_game("BOS", "TOR", 2025))
    assert [e.result for e in group] == ["coincidental", "coincidental", "minor"]
    # fighting majors plus one roughing minor each: 5-on-5
    group = pens((HOME, "major"), (AWAY, "major"), (HOME, "minor"), (AWAY, "minor"))
    label_penalties(group, new_game("BOS", "TOR", 2025))
    assert all(e.result == "coincidental" for e in group)


def test_a_minor_cancels_half_a_double_minor():
    group = pens((HOME, "double_minor"), (AWAY, "minor"))
    label_penalties(group, new_game("BOS", "TOR", 2025))
    assert [e.result for e in group] == ["minor", "coincidental"]
    group = pens((HOME, "double_minor"), (AWAY, "minor"), (AWAY, "minor"))
    label_penalties(group, new_game("BOS", "TOR", 2025))
    assert all(e.result == "coincidental" for e in group)


def test_shootout_order_is_rebuilt():
    # as listed in the data: each team's goals after its misses
    listed = [(HOME, "saved"), (AWAY, "missed"), (HOME, "saved"), (AWAY, "saved"), (HOME, "missed"),
              (AWAY, "saved"), (HOME, "saved"), (HOME, "goal"), (AWAY, "goal"), (AWAY, "goal")]
    order = order_shootout(listed)
    assert [s for s, _ in order] == [HOME, AWAY] * 5
    for side in (HOME, AWAY):  # each team keeps its results
        assert sorted(r for s, r in order if s == side) == sorted(r for s, r in listed if s == side)
    shots, goals = [0, 0], [0, 0]
    for i, (side, res) in enumerate(order):  # decided exactly on the last attempt
        shots[side] += 1
        goals[side] += res == "goal"
        assert shootout_decided(shots, goals) == (i == len(order) - 1)
    assert order_shootout([(HOME, "goal"), (AWAY, "saved"), (HOME, "goal"), (AWAY, "saved")]) == \
        [(HOME, "goal"), (AWAY, "saved"), (HOME, "goal"), (AWAY, "saved")]
    assert order_shootout([(HOME, "goal"), (HOME, "goal")]) is None


def test_repairs_fill_missing_faceoffs_and_stoppages():
    meta = dict(home="BOS", away="TOR", season=2017, game_type="REG")
    events = [Event("faceoff", "won", HOME, 1, 0), Event("shot", "saved", HOME, 1, 10),
              Event("stoppage", "goalie_freeze", None, 1, 11), Event("shot", "saved", AWAY, 1, 12),
              Event("faceoff", "won", AWAY, 1, 12), Event("hit", "hit", AWAY, 1, 30),
              Event("faceoff", "won", HOME, 1, 40)]
    out, added = repair_events(meta, events)
    assert [(e.play_type, e.implied) for e in out] == [
        ("faceoff", False), ("shot", False), ("stoppage", False), ("faceoff", False), ("shot", False),
        ("hit", False), ("stoppage", True), ("faceoff", False)]
    assert added == 1
    out, added = repair_events(meta, events[:3] + events[5:6])
    assert out[3].play_type == "faceoff" and out[3].implied and out[3].side == AWAY and added == 1


def test_same_player_minors_merge_and_penalty_shots_come_before_the_attempt():
    a = Event("penalty", "minor", AWAY, 1, 100, player=7)
    b = Event("penalty", "minor", AWAY, 1, 100, player=7)
    c = Event("penalty", "minor", AWAY, 1, 100, player=8)
    out = _merge_same_player_minors([a, b, c])
    assert [e.result for e in out] == ["double_minor", "minor"]
    shot = Event("shot", "goal", HOME, 2, 500, score=(1, 0))
    call = Event("penalty", "penalty_shot", AWAY, 2, 500, score=(1, 0))
    before = Event("hit", "hit", AWAY, 2, 490, score=(0, 0))
    out = _order_penalty_shots([before, shot, call, Event("faceoff", "won", HOME, 2, 500)])
    assert [e.play_type for e in out] == ["hit", "penalty", "shot", "faceoff"]
    assert out[1].score == (0, 0)  # the score before the attempt
    # a call with no attempt is dropped
    assert _order_penalty_shots([before, call]) == [before]


def fixture_games():
    df = load_season_file(FIXTURE)
    return [game_events(g) for g in split_games(df)]


def test_fixture_games_replay_through_the_engine():
    games = fixture_games()
    assert len(games) == 4
    kinds = {m["game_id"]: m for m, _ in games}
    assert set(kinds) == {"2024020009", "2024020012", "2024020022", "2024030143"}
    seen = set()
    for meta, events in games:
        events, _ = repair_events(meta, events)
        rep = replay_game(meta, events)
        assert rep.ok / rep.total >= 0.99, (meta["game_id"], rep.misses)
        seen |= {(e.play_type, e.result) for e in events}
        rows = rep.rows
        assert rows[0]["phase"] != LIVE and all(r["legal"] for r in rows)
    # the four games cover the shootout, a penalty shot, empty nets and overtime
    for k in [("shootout", "goal"), ("penalty", "penalty_shot"), ("goalie_pull", "goalie_pull"),
              ("period_end", "end"), ("penalty", "minor")]:
        assert k in seen, k


@pytest.mark.skipif(not (RAW / "play_by_play_2025.parquet").exists(), reason="raw NHL data not downloaded")
def test_engine_replays_a_full_season():
    out = replay_season(RAW / "play_by_play_2025.parquet", max_games=300)
    assert out["games"] == 300 and out["agreement"] >= 0.99, out["field_misses"]
