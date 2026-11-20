import numpy as np
import pandas as pd
import pytest

from runitback.epl.data import players as fpl

TEAMS = pd.DataFrame({"id": [1, 2, 3], "short_name": ["ARS", "CHE", "MCI"]})
# fixture 10: ARS (home) - CHE; 11: MCI - ARS; 12: CHE - MCI (postponed, replayed)
FIXTURES = pd.DataFrame({"id": [10, 11, 12], "team_h": [1, 3, 2], "team_a": [2, 1, 3]})
ELEMENTS = pd.DataFrame(
    {
        "id": [100, 200, 300],
        "code": [9001, 9002, 9003],
        "first_name": ["Bukayo", "HÃ©ctor", "Erling"],
        "second_name": ["Saka", "BellerÃ­n", "Haaland"],
        "element_type": [3, 2, 4],
    }
)


def hist(element, fixture, round_, was_home, minutes, **extra):
    row = {"element": element, "fixture": fixture, "round": round_,
           "was_home": was_home, "minutes": minutes, "kickoff_time": f"2025-0{round_}-01T14:00:00Z",
           "goals_scored": 0, "own_goals": 0, "total_points": 2, "value": 105,
           "selected": 1000}  # fmt: skip
    return row | extra


HISTORY = pd.DataFrame(
    [
        hist(100, 10, 1, True, 90, goals_scored=1, total_points=9),
        hist(100, 11, 2, False, 60),  # away at City
        hist(200, 10, 1, False, 0),  # unused: dropped
        hist(200, 12, 3, True, 45),
        hist(200, 12, 3, True, 45),  # a repeated line
        hist(300, 12, 2, False, 0),  # the postponed game at its first gameweek ...
        hist(300, 12, 5, False, 90, own_goals=1),  # ... played at the replay
        hist(999, 10, 1, True, 90),  # unknown player: dropped
    ]
)


@pytest.fixture
def out():
    return fpl.build_players(HISTORY, ELEMENTS, TEAMS, FIXTURES, 2026)


def row(out, player_id, game_id):
    rows = out[(out.player_id == player_id) & (out.game_id == game_id)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_appearances_only_deduplicated_and_known_players(out):
    assert len(out) == 4 and set(out.player_id) == {9001, 9002, 9003}
    assert list(out.columns) == fpl.COLUMNS
    assert row(out, 9003, "2026_CHE_MCI").gameweek == 5  # the replay, not round 2


def test_team_and_location_come_from_the_fixture(out):
    saka_home, saka_away = (
        row(out, 9001, "2026_ARS_CHE"),
        row(out, 9001, "2026_MCI_ARS"),
    )
    assert (saka_home.team, saka_home.opp_team, saka_home.location) == ("ARS", "CHE", 1)
    assert (saka_away.team, saka_away.opp_team, saka_away.location) == (
        "ARS",
        "MCI",
        -1,
    )
    haaland = row(out, 9003, "2026_CHE_MCI")
    assert (haaland.team, haaland.opp_team, haaland.location) == ("MCI", "CHE", -1)


def test_stats_position_price_and_missing_columns(out):
    saka = row(out, 9001, "2026_ARS_CHE")
    assert (saka.minutes, saka.goals, saka.fpl_points, saka.position) == (
        90,
        1,
        9,
        "MID",
    )
    assert saka.price == 10.5 and saka.selected == 1000
    assert np.isnan(saka.xg) and np.isnan(saka.starts)  # not in this season's data
    assert row(out, 9003, "2026_CHE_MCI").own_goals == 1


def test_text_encoding_repaired(out):
    assert set(out.player) == {"Bukayo Saka", "Héctor Bellerín", "Erling Haaland"}


def test_expected_stats_kept_when_present():
    history = HISTORY.assign(expected_goals="0.45", starts=1)
    out = fpl.build_players(history, ELEMENTS, TEAMS, FIXTURES, 2026)
    assert (out.xg == 0.45).all() and (out.starts == 1).all()


def test_unknown_team_code_raises():
    teams = TEAMS.assign(short_name=["ARS", "CHE", "XXX"])
    with pytest.raises(ValueError, match="XXX"):
        fpl.build_players(HISTORY, ELEMENTS, teams, FIXTURES, 2026)


def test_check_against_games(out):
    games = pd.DataFrame(
        {
            "season": 2026,
            "game_id": ["2026_ARS_CHE"] * 2
            + ["2026_MCI_ARS"] * 2
            + ["2026_CHE_MCI"] * 2,
            "team": ["ARS", "CHE", "MCI", "ARS", "CHE", "MCI"],
            "opp_team": ["CHE", "ARS", "ARS", "MCI", "MCI", "CHE"],
            "goals": [1, 0, 0, 0, 1, 0],  # CHE-MCI: City's player's own goal
            "result": 1.0,
        }
    )
    report = fpl.check(out, games).iloc[0]
    assert report.rows_without_game == 0 and report.games_without_players == 0
    assert report.goal_mismatch == 0
    wrong = games.assign(goals=[2, 0, 0, 0, 1, 0])
    assert fpl.check(out, wrong).iloc[0].goal_mismatch == pytest.approx(1 / 6)


def test_archive_season_name():
    assert fpl.archive_season_name(2026) == "2025-26"
    assert fpl.archive_season_name(2001) == "2000-01"
