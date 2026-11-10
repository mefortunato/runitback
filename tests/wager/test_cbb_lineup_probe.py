"""The C1 probe's per-game transforms (scripts/cbb_lineup_probe.py)."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("requests")

PATH = Path(__file__).resolve().parents[2] / "scripts" / "cbb_lineup_probe.py"
spec = importlib.util.spec_from_file_location("cbb_lineup_probe", PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def floor(home, away):
    return [{"id": i, "team": "H"} for i in home] + [{"id": i, "team": "A"} for i in away]


def play(id_, period, remaining, home, away, actor=None, type_="JumpShot"):
    return {
        "id": id_, "period": period, "secondsRemaining": remaining, "playType": type_,
        "onFloor": floor(home, away),
        "participants": [{"id": actor, "name": ""}] if actor is not None else [],
    }


HOME, AWAY = [1, 2, 3, 4, 5], [11, 12, 13, 14, 15]


def test_floor_seconds_splits_the_clock_between_plays():
    plays = [
        play(1, 1, 1100, HOME, AWAY),
        play(3, 1, 600, [6, 2, 3, 4, 5], AWAY),  # 6 for 1 at 10:00 of the first half
        play(2, 1, 900, HOME, AWAY),
        play(4, 2, 1200, HOME, AWAY),
    ]
    seconds, first = probe.floor_seconds(plays)
    assert first == set(HOME) | set(AWAY)
    assert seconds[1] == 600 + 1200  # 20:00-10:00, then the whole second half
    assert seconds[6] == 600
    assert seconds[11] == 2400


def test_cbbd_game_skips_substitutions_in_the_credited_check():
    plays = [
        play(1, 1, 1100, HOME, AWAY, actor=1),
        play(2, 1, 1000, HOME, AWAY, actor=6, type_="Substitution"),
        play(3, 1, 900, [6, 2, 3, 4, 5], AWAY, actor=6),
        play(4, 1, 800, [7, 2, 3, 4, 5], AWAY, actor=99),
    ]
    out = probe.cbbd_game(plays)
    assert out["subs"] == 1
    assert out["five_share"] == 1.0
    assert out["min_distinct_fives"] == 1  # the away five never changes
    assert out["credited_on_floor"] == pytest.approx(2 / 3)


def test_espn_games_counts_substitutions_per_side():
    rows = []
    for i in range(12):
        rows.append({"game_id": 1, "type_text": "Substitution", "text": "", "team_id": 10,
                     "athlete_id_1": i, "athlete_id_2": None, "home_team_id": 10,
                     "away_team_id": 20, "game_date": "2026-01-01"})
    rows.append({"game_id": 2, "type_text": "JumpShot", "text": "", "team_id": 10,
                 "athlete_id_1": 1, "athlete_id_2": None, "home_team_id": 10,
                 "away_team_id": 20, "game_date": "2026-01-01"})
    games = probe.espn_games(pd.DataFrame(rows), min_subs=10).set_index("game_id")
    assert games.loc[1, "home_subs"] == 12 and games.loc[1, "away_subs"] == 0
    assert games.loc[1, "has_subs"] and not games.loc[1, "rebuildable"]
    assert not games.loc[2, "has_subs"]
