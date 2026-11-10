import datetime

import pandas as pd
import pytest

from runitback.wager import espn
from runitback.wager.cbb.data import cbbd
from runitback.wager.cbb.data import download as college

# CBBD's /games (its own id -> ESPN's sourceId) and /lines, as its API spec
# describes them
GAMES = [
    {"id": 1, "sourceId": "401700001"},
    {"id": 2, "sourceId": "401700002"},
    {"id": 3, "sourceId": "401700003"},
]
LINES = [
    {
        "gameId": 1, "homeScore": 80, "awayScore": 70,
        "lines": [
            {"provider": "teamrankings", "spread": -20, "overUnder": 100,
             "homeMoneyline": -5000, "awayMoneyline": 2000},
            {"provider": "Bovada", "spread": -7.5, "overUnder": 141.5,
             "homeMoneyline": -320, "awayMoneyline": 255},
            {"provider": "ESPN Bet", "spread": -7, "overUnder": None,
             "homeMoneyline": None, "awayMoneyline": None},
        ],
    },
    # CBBD has home and away the other way around: left out
    {"gameId": 2, "homeScore": 60, "awayScore": 75,
     "lines": [{"provider": "Bovada", "spread": 3}]},
    # listed without lines (ESPN may have them): left out
    {"gameId": 3, "homeScore": 66, "awayScore": 64, "lines": []},
]  # fmt: skip


def schedule(*rows):
    return pd.DataFrame(
        rows, columns=["game_id", "start", "home_score", "away_score"]
    ).assign(start=lambda f: pd.to_datetime(f["start"], utc=True))


@pytest.fixture
def api(monkeypatch):
    calls = []

    def rows(path, key, **params):
        calls.append((path, params))
        assert key == "secret"
        return {"games": GAMES, "lines": LINES}[path]

    monkeypatch.setattr(cbbd, "_rows", rows)
    return calls


def test_parse_lines_prefers_books_and_skips_prediction_sites():
    assert cbbd.parse_lines(LINES[0]["lines"]) == {
        "spread_line": 7.0,  # the home team's margin: ESPN Bet's -7
        "spread_provider": "ESPN Bet",
        "total_line": 141.5,
        "total_provider": "Bovada",
        "moneyline": -320.0,
        "opp_moneyline": 255.0,
        "moneyline_provider": "Bovada",
    }
    assert cbbd.parse_lines([]) == {}


def test_season_lines_by_espn_id(api):
    games = schedule(
        ("401700001", "2025-03-01T20:00Z", 80, 70),
        ("401700002", "2025-03-02T20:00Z", 75, 60),
        ("401700003", "2025-03-03T20:00Z", 66, 64),  # no lines in CBBD
        ("401700004", "2025-03-19T20:00Z", 66, 64),  # not in CBBD
    )
    lines = cbbd.season_lines(2025, games, "secret")
    assert set(lines) == {"401700001"}
    assert lines["401700001"]["spread_line"] == 7.0
    # only the windows holding these games' dates: March, two endpoints
    assert [path for path, _ in api] == ["games", "lines"]
    assert api[0][1]["startDateRange"] == "2025-02-28T20:00:00Z"


def test_full_windows_are_split(monkeypatch):
    sizes = iter([cbbd.LIMIT, 2, 3])
    calls = []

    def rows(path, key, **params):
        calls.append(params)
        return [{"gameId": len(calls) * 10 + i} for i in range(next(sizes))]

    monkeypatch.setattr(cbbd, "_rows", rows)
    start = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2025, 2, 1, tzinfo=datetime.timezone.utc)
    assert len(cbbd._fetch("lines", "k", 2025, start, end)) == 5
    assert len(calls) == 3


def test_season_lines_cache_bulk_finished_games(api, tmp_path):
    cache = espn.Cache(str(tmp_path))
    games = schedule(
        ("401700001", "2025-03-01T20:00Z", 80, 70),
        ("401700002", "2025-03-02T20:00Z", 75, 60),
        ("401700003", "2025-03-03T20:00Z", 66, 64),
    )
    finished = list(games.game_id)
    fetched = []

    def fetch(game_id):
        fetched.append(game_id)
        return {} if game_id == "401700003" else {"spread_line": 1.0}

    def bulk(ids):
        return cbbd.season_lines(2025, games[games.game_id.isin(ids)], "secret")

    lines = espn.season_lines(
        2025, finished, ["401700009"], cache, fetch, workers=1, bulk=bulk
    )
    # finished games in bulk; the ones CBBD has no lines for (swapped, or
    # listed without any) and upcoming ones from ESPN
    assert lines["401700001"]["spread_line"] == 7.0
    assert lines["401700002"] == {"spread_line": 1.0}
    assert lines["401700003"] == {}
    assert sorted(fetched) == ["401700002", "401700003", "401700009"]
    assert set(cache.read_json("odds_2025.json")) == set(finished)
    # all cached, ESPN's answers too: no second request anywhere
    fetched.clear()
    espn.season_lines(2025, finished, [], cache, fetch, workers=1, bulk=bulk)
    assert len(api) == 2
    assert fetched == []


def test_bulk_lines_only_for_men_with_a_key(monkeypatch):
    monkeypatch.delenv(cbbd.KEY_ENV, raising=False)
    assert college.bulk_lines(college.MEN) is None
    monkeypatch.setenv(cbbd.KEY_ENV, "secret")
    assert college.bulk_lines(college.MEN) is not None
    assert college.bulk_lines(college.WOMEN) is None
