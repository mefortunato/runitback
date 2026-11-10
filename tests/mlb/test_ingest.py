"""MLB Stats API client (cache, retries, throttle), run values and season tables."""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from runitback.mlb.data import ingest, run_values, statsapi

FIXTURES = Path(__file__).parent / "fixtures"
FEEDS = {745444: 2024, 718723: 2023, 715769: 2022}


def fixture(pk: int) -> dict:
    return statsapi.read_cached(FIXTURES / f"feed_{pk}.json.gz")


class FakeClient:
    """Serves the fixture feeds and a schedule; counts requests."""

    def __init__(self, final: bool = True):
        self.calls: list[str] = []
        self.final = final

    def get(self, path: str, **params) -> dict:
        self.calls.append(path)
        if path == "/v1/schedule":
            season = params["season"]
            games = [{"gamePk": pk, "gameType": fixture(pk)["gameData"]["game"]["type"], "season": str(s),
                      "officialDate": f"{s}-06-01", "status": {"abstractGameState": "Final",
                                                               "detailedState": "Final"}}
                     for pk, s in FEEDS.items() if s == season]
            games.append({"gamePk": 1, "gameType": "R", "season": str(season), "officialDate": f"{season}-06-02",
                          "status": {"abstractGameState": "Final", "detailedState": "Postponed"}})
            games.append({"gamePk": 2, "gameType": "S", "season": str(season), "officialDate": f"{season}-03-01",
                          "status": {"abstractGameState": "Final", "detailedState": "Final"}})
            return {"dates": [{"date": f"{season}-06-01", "games": games}]}
        pk = int(path.split("/")[3])
        feed = json.loads(json.dumps(fixture(pk)))
        if not self.final:
            feed["gameData"]["status"] = {"abstractGameState": "Live", "detailedState": "In Progress"}
        return feed


def test_schedule_drops_postponed_and_spring_games():
    games = statsapi.schedule(2024, FakeClient())
    assert [g["game_pk"] for g in games] == [745444]
    assert games[0]["final"] and games[0]["game_type"] == "R"


def test_finished_games_are_cached(tmp_path):
    client = FakeClient()
    first = statsapi.feed(745444, 2024, tmp_path, client)
    again = statsapi.feed(745444, 2024, tmp_path, client)
    assert first == again and len(client.calls) == 1
    assert statsapi.cache_path(tmp_path, 2024, 745444).exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_unfinished_games_are_not_cached(tmp_path):
    client = FakeClient(final=False)
    statsapi.feed(745444, 2024, tmp_path, client)
    statsapi.feed(745444, 2024, tmp_path, client)
    assert len(client.calls) == 2
    assert not statsapi.cache_path(tmp_path, 2024, 745444).exists()


def test_client_retries_rate_limits_and_server_errors(monkeypatch):
    responses = iter([httpx.Response(429, headers={"retry-after": "0"}), httpx.Response(503),
                      httpx.Response(200, json={"ok": True})])
    client = statsapi.Client(min_interval=0.0)
    client._http = httpx.Client(transport=httpx.MockTransport(lambda request: next(responses)))
    monkeypatch.setattr(statsapi.time, "sleep", lambda s: None)
    assert client.get("/v1/schedule") == {"ok": True}
    assert client.requests == 3


def test_client_gives_up_after_retries(monkeypatch):
    client = statsapi.Client(min_interval=0.0, retries=2)
    client._http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    monkeypatch.setattr(statsapi.time, "sleep", lambda s: None)
    with pytest.raises(httpx.HTTPStatusError):
        client.get("/v1/schedule")
    assert client.requests == 3


def test_client_throttles_across_requests():
    client = statsapi.Client(min_interval=0.05)
    client._http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    t0 = time.monotonic()
    for _ in range(5):
        client.get("/v1/x")
    assert time.monotonic() - t0 >= 0.19


def test_probables():
    class Client:
        def get(self, path, **params):
            assert params["hydrate"] == "probablePitcher,lineups"
            return {"dates": [{"date": "2025-07-04", "games": [{
                "gamePk": 7, "gameType": "R", "officialDate": "2025-07-04", "status": {"detailedState": "Scheduled"},
                "teams": {"home": {"team": {"id": 120}, "probablePitcher": {"id": 11}},
                          "away": {"team": {"id": 111}}},
                "lineups": {"homePlayers": [{"id": i} for i in range(1, 10)]}}]}]}

    home, away = statsapi.probables("2025-07-04", Client())
    assert (home["side"], home["team_id"], home["opp_id"], home["probable_sp_id"]) == ("home", 120, 111, 11)
    assert home["lineup"] == list(range(1, 10))
    assert away["probable_sp_id"] is None and away["lineup"] == []


# --------------------------------------------------------------- run values
def synthetic_pa(n: int, runs: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "start_bases": rng.integers(0, 8, n), "start_outs": rng.integers(0, 3, n),
        "runs_rest": rng.poisson(runs, n), "half_complete": rng.random(n) > 0.05,
        "pre_bases": rng.integers(0, 8, n), "pre_outs": rng.integers(0, 3, n),
        "end_bases": rng.integers(0, 8, n), "end_outs": rng.integers(0, 4, n),
        "runs_pre": 0, "runs": rng.integers(0, 2, n),
    })


def test_run_expectancy_table():
    pa = synthetic_pa(5000, 0.5, 0)
    table = run_values.run_expectancy(pa)
    assert len(table) == 24 and table["re"].notna().all()
    keep = pa[pa["half_complete"]]
    want = keep[(keep["start_bases"] == 3) & (keep["start_outs"] == 1)]["runs_rest"].mean()
    assert table.loc[(3, 1), "re"] == pytest.approx(want)


def test_run_value_formula():
    table = run_values.run_expectancy(synthetic_pa(5000, 0.5, 0))
    pa = pd.DataFrame({"start_bases": [0, 1], "start_outs": [0, 2], "pre_bases": [0, 2], "pre_outs": [0, 2],
                       "end_bases": [1, 0], "end_outs": [0, 3], "runs_pre": [0, 0], "runs": [0, 0]})
    out = run_values.add_run_values(pa, table)
    re = table["re"]
    assert out["run_value"][0] == pytest.approx(re[(1, 0)] - re[(0, 0)])
    assert out["re_end"][1] == 0.0  # the inning is over
    assert out["run_value"][1] == pytest.approx(-re[(2, 2)])
    assert out["run_value_pa"][1] == pytest.approx(-re[(1, 2)])


def test_run_values_use_only_earlier_seasons():
    seasons = {2022: synthetic_pa(3000, 0.4, 1), 2023: synthetic_pa(3000, 0.6, 2)}
    values, tables = run_values.season_run_values(seasons)
    assert 2022 not in tables and values[2022]["run_value"].isna().all()
    pd.testing.assert_frame_equal(tables[2023], run_values.run_expectancy(seasons[2022]))
    # changing season 2023 itself does not change the values' run expectancy
    changed = {2022: seasons[2022], 2023: synthetic_pa(3000, 2.0, 3)}
    pd.testing.assert_frame_equal(run_values.season_run_values(changed)[1][2023], tables[2023])


# ------------------------------------------------------------ season tables
def test_build_seasons(tmp_path):
    client = FakeClient()
    cache, out = tmp_path / "feeds", tmp_path / "processed"
    summary = ingest.build_seasons([2022, 2023, 2024], cache, out, client, workers=2)
    for season in (2022, 2023, 2024):
        assert summary[season]["games"] == 1
        assert all(summary[season].get(k, 0) == 0 for k in ("base_mismatch", "post_mismatch", "runs_mismatch"))
        for name in ingest.TABLES:
            assert (out / f"{name}_{season}.parquet").exists()
    assert not summary[2022]["run_values"] and summary[2023]["run_values"]
    pa = pd.read_parquet(out / "pa_2024.parquet")
    # one earlier game leaves some of the 24 states unseen (NaN); a full season sees them all
    assert len(pa) == 79 and pa["run_value"].notna().mean() > 0.8
    re = pd.read_parquet(out / "re_2024.parquet")
    expected = run_values.run_expectancy(pd.read_parquet(out / "pa_2023.parquet")).reset_index()
    pd.testing.assert_frame_equal(re, expected)
    # a second build reads every feed from the cache
    calls = len(client.calls)
    ingest.build_seasons([2024], cache, out, client)
    assert [c for c in client.calls[calls:] if "feed" in c] == []
