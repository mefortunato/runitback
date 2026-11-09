import numpy as np
import pandas as pd
import pytest

from runitback.wager.nfl.data.download import build_games
from runitback.wager.nfl.data.features import SCHEMA, STATS_COLUMNS
from runitback.wager.nfl.data.names import odds_map


def nflverse_frames():
    """Two games shaped like nflverse; SD's stats are missing from game 2."""
    schedule = pd.DataFrame(
        {
            "game_id": ["2010_01_OAK_KC", "2010_02_SD_STL"],
            "season": [2010, 2010],
            "week": [1, 2],
            "game_type": ["REG", "REG"],
            "location": ["Home", "Neutral"],
            "home_team": ["KC", "STL"],
            "away_team": ["OAK", "SD"],
            "home_score": [21, 10],
            "away_score": [14, 13],
            "result": [7, -3],
            "spread_line": [3.5, -1.0],
            "home_rest": [7, 7],
            "away_rest": [7, 6],
        }
    )
    stats = {col: 0.0 for col in STATS_COLUMNS}
    team_stats = pd.DataFrame(
        [
            {**stats, "season": 2010, "week": 1, "team": "KC", "passing_yards": 250},
            {**stats, "season": 2010, "week": 1, "team": "OAK", "passing_yards": 180},
            {**stats, "season": 2010, "week": 2, "team": "STL", "passing_yards": 300},
            {**stats, "season": 2010, "week": 2, "team": None, "passing_yards": 1},
        ]
    )
    return schedule, team_stats


def test_each_game_has_two_correctly_paired_rows():
    games = build_games(*nflverse_frames())
    assert games.groupby("game_id").size().eq(2).all()
    assert not games.duplicated(["season", "week", "team"]).any()

    kc = games[games.team == "KC"].iloc[0]
    assert (kc.opp_team, kc.location, kc.result, kc.rest, kc.opp_rest) == (
        "LV",
        1,
        7,
        7,
        7,
    )
    assert (kc.passing_yards, kc.opp_passing_yards) == (250, 180)
    lv = games[games.team == "LV"].iloc[0]
    assert (lv.location, lv.result, lv.spread_line) == (-1, -7, -3.5)


def test_missing_stats_are_missing_not_self_paired():
    """Regression: a row without a counterpart used to be paired with itself."""
    games = build_games(*nflverse_frames())
    la = games[games.team == "LA"].iloc[0]
    lac = games[games.team == "LAC"].iloc[0]
    assert la.location == 0 and lac.location == 0
    assert la.passing_yards == 300 and np.isnan(la.opp_passing_yards)
    assert np.isnan(lac.passing_yards) and lac.opp_passing_yards == 300
    assert lac.result == 3


def test_schema_columns_exist_in_built_games():
    games = build_games(*nflverse_frames())
    needed = set(SCHEMA.stats) | {SCHEMA.target, SCHEMA.location, *SCHEMA.key_columns}
    assert needed <= set(games.columns)


def test_odds_names_map_to_current_teams():
    current = {
        "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET",
        "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO",
        "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
    }  # fmt: skip
    assert set(odds_map.values()) == current
    assert odds_map["Las Vegas Raiders"] == "LV"


def injury_frames():
    """KC's QB and a corner across six weeks; snap counts for weeks 1-5."""
    games = pd.DataFrame(
        {"season": 2023, "week": list(range(1, 7)), "team": "KC", "opp_team": "BUF"}
    )
    snaps = pd.DataFrame(
        [
            # the QB plays every snap; the corner half the snaps in weeks 1-4
            {"season": 2023, "week": w, "team": "KC", "pfr_player_id": "qb1",
             "offense_pct": 1.0, "defense_pct": 0.0}
            for w in range(1, 6)
        ]
        + [
            {"season": 2023, "week": w, "team": "KC", "pfr_player_id": "cb1",
             "offense_pct": 0.0, "defense_pct": 0.5}
            for w in range(1, 5)
        ]
    )  # fmt: skip
    players = pd.DataFrame({"gsis_id": ["g-qb", "g-cb"], "pfr_id": ["qb1", "cb1"]})

    def report(week, gsis_id, position, status, modified="2023-01-01"):
        return {"season": 2023, "week": week, "team": "KC", "gsis_id": gsis_id,
                "position": position, "report_status": status,
                "date_modified": modified}  # fmt: skip

    injuries = pd.DataFrame(
        [
            report(3, "g-qb", "QB", "Questionable"),
            report(5, "g-cb", "CB", "Out"),
            # the later report wins
            report(6, "g-qb", "QB", "Questionable", "2023-01-01"),
            report(6, "g-qb", "QB", "Out", "2023-01-02"),
            report(6, "g-cb", "CB", None),
        ]
    )
    return games, injuries, snaps, players


def test_injury_features_weigh_status_by_recent_snap_share():
    from runitback.wager.nfl.data.injuries import INJURY_COLUMNS, injury_features

    features = injury_features(*injury_frames()).set_index("week")
    assert list(features.columns) == ["season", "team", *INJURY_COLUMNS]
    # week 1 has no earlier games: unknown, not "healthy"
    assert features.loc[1, INJURY_COLUMNS].isna().all()
    assert features.loc[2, "inj_qb"] == 0.0
    # questionable (0.33) x every snap of weeks 1-2
    assert features.loc[3, "inj_qb"] == pytest.approx(0.33)
    # out x half the snaps over weeks 1-4
    assert features.loc[5, "inj_secondary"] == pytest.approx(0.5)
    assert features.loc[5, "inj_qb"] == 0.0
    # out (latest report) x the QB's share over weeks 2-5; the corner was
    # listed without a game status and missed week 5 already
    assert features.loc[6, "inj_qb"] == pytest.approx(1.0)
    assert features.loc[6, "inj_secondary"] == 0.0


def test_build_games_adds_both_sides_injury_features():
    from runitback.wager.nfl.data.injuries import INJURY_COLUMNS, OPP_INJURY_COLUMNS

    schedule, team_stats = nflverse_frames()
    healthy = dict.fromkeys(INJURY_COLUMNS, 0.0)
    injuries = pd.DataFrame(
        [
            {"season": 2010, "week": 1, "team": "KC", **healthy, "inj_qb": 1.0},
            {"season": 2010, "week": 1, "team": "LV", **healthy, "inj_ol": 2.0},
        ]
    )
    games = build_games(schedule, team_stats, injuries).set_index("team")
    assert (games.loc["KC", "inj_qb"], games.loc["KC", "opp_inj_ol"]) == (1.0, 2.0)
    assert (games.loc["LV", "opp_inj_qb"], games.loc["LV", "inj_ol"]) == (1.0, 2.0)
    assert games.loc["LA", OPP_INJURY_COLUMNS].isna().all()
