import threading
from collections import Counter

import pytest
from fastapi.testclient import TestClient

from runitback import server as combined
from runitback.nba.engine.rules import AWAY, HOME
from runitback.nba.server import app as nba_server
from runitback.nba.sim.season import (
    DIVISIONS, GAMES_PER_TEAM, REGULAR_SEASON_GAMES, TEAM_DIVISION, generate_schedule, opening_day,
    simulate_season, standings, total_game_slots,
)
from runitback.nba.vocab import TEAMS


class HomeWinsSimulator:
    """Every game ends 110-100 to the home team; records the batches it was given."""
    ratings = None
    strength_scale = 1.0

    def __init__(self):
        self.batches = []

    def rollout_games(self, starts, ratings, seed=None, temperature=1.0, batch_size=128,
                      inference_context=None, on_batch=None):
        self.batches.append(list(zip(starts, ratings)))
        finals = []
        for start in starts:
            final = start.copy()
            final.score[HOME], final.score[AWAY] = 110, 100
            final.over = True
            finals.append(final)
        if on_batch:
            on_batch(len(starts))
        return finals


def pairs(schedule):
    return Counter(frozenset((g["home"], g["away"])) for g in schedule)


def test_schedule_follows_the_nba_formula():
    schedule = generate_schedule(2027, seed=4)
    assert len(schedule) == REGULAR_SEASON_GAMES
    games = Counter(t for g in schedule for t in (g["home"], g["away"]))
    homes = Counter(g["home"] for g in schedule)
    assert set(games.values()) == {GAMES_PER_TEAM} and set(homes.values()) == {41}
    meetings = pairs(schedule)
    for team in TEAMS:
        conference, division = TEAM_DIVISION[team]
        counts = Counter()
        for other in TEAMS:
            if other == team:
                continue
            n = meetings[frozenset((team, other))]
            if TEAM_DIVISION[other] == (conference, division):
                assert n == 4
            elif TEAM_DIVISION[other][0] == conference:
                counts[n] += 1
            else:
                assert n == 2
        assert counts == {4: 6, 3: 4}
    # nobody plays twice on one day; the season starts on opening night
    per_day = Counter((g["day"], t) for g in schedule for t in (g["home"], g["away"]))
    assert max(per_day.values()) == 1
    assert schedule[0]["date"] == opening_day(2027).isoformat() == "2026-10-27"
    assert schedule[-1]["day"] < 200
    assert generate_schedule(2027, seed=4) == schedule
    assert pairs(generate_schedule(2028, seed=4)) != meetings  # the 3-game opponents rotate


def test_full_season_with_play_in_and_playoffs():
    sim = HomeWinsSimulator()
    progress = []
    r = simulate_season(sim, 2027, seed=1, strength_mode="neutral",
                        progress=lambda *a: progress.append(a))
    assert r["regular_game_count"] == REGULAR_SEASON_GAMES and r["play_in"]
    assert progress[-1] == ("Complete", total_game_slots(2027), total_game_slots(2027))
    # all regular-season games go to the sampler together; play-in rounds and series follow
    assert len(sim.batches[0]) == REGULAR_SEASON_GAMES
    assert [len(b) for b in sim.batches[1:]] == [4, 2, 56, 28, 14, 7]
    for conference in ("East", "West"):
        seeds = r["standings"][conference]["seeds"]
        assert [row["seed"] for row in seeds] == list(range(1, 16))
        assert [row["status"] for row in seeds] == ["playoffs"] * 6 + ["play_in"] * 4 + ["out"] * 5
        assert all(row["wins"] + row["losses"] == 82 for row in seeds)
        assert sum(len(d) for d in r["standings"][conference]["divisions"].values()) == 15
    play_in = r["play_in_rounds"][0]["games"]
    assert len(play_in) == 6 and all(g["home"] == g["winner"] for g in play_in)
    # home teams always win: every series goes 7 and the higher seed wins it
    rounds = r["playoffs"]
    assert [rd["round"] for rd in rounds] == ["First Round", "Conference Semifinals", "Conference Finals", "NBA Finals"]
    assert [len(rd["series"]) for rd in rounds] == [8, 4, 2, 1]
    for rd in rounds:
        for s in rd["series"]:
            assert len(s["games"]) == 7 and s["winner"] == s["high"]["team"] and s["summary"].endswith("4-3")
            assert [g["home"] == s["high"]["team"] for g in s["games"]] == [True, True, False, False, True, False, True]
    first_round = {(s["high"]["seed"], s["low"]["seed"]) for s in rounds[0]["series"]}
    assert first_round == {(1, 8), (2, 7), (3, 6), (4, 5)}
    east_top = r["standings"]["East"]["seeds"][0]["team"]
    west_top = r["standings"]["West"]["seeds"][0]["team"]
    assert {r["finals"]["high"]["team"], r["finals"]["low"]["team"]} == {east_top, west_top}
    assert r["champion"] == r["finals"]["winner"]
    assert r["playoff_game_count"] == 6 + 15 * 7


def test_pre_play_in_seasons_seed_eight_teams():
    r = simulate_season(HomeWinsSimulator(), 2015, strength_mode="neutral")
    assert not r["play_in"] and r["play_in_rounds"] == []
    statuses = [row["status"] for row in r["standings"]["West"]["seeds"]]
    assert statuses == ["playoffs"] * 8 + ["out"] * 7
    with pytest.raises(ValueError):
        simulate_season(HomeWinsSimulator(), 2004)


def test_strength_modes_use_preseason_ratings(nba_sim):
    sim = HomeWinsSimulator()
    sim.ratings, sim.strength_scale = nba_sim.ratings, 2.0
    simulate_season(sim, 2027, strength_mode="checkpoint")
    ratings = {s.home: r[0] for batch in sim.batches for s, r in batch}
    neutral = ratings["ATL"]
    assert ratings["BOS"] != neutral and ratings["BOS"]["elo"] - 1505 == pytest.approx(
        2.0 * (nba_sim.ratings.latest("BOS", 2027)["elo"] - 1505))
    with pytest.raises(ValueError):
        simulate_season(HomeWinsSimulator(), 2027, strength_mode="randomized")


def game(home, away, home_score, away_score, n):
    return {"game_id": f"g{n}", "home": home, "away": away, "home_score": home_score,
            "away_score": away_score, "winner": home if home_score > away_score else away}


def test_tiebreakers_division_winner_then_head_to_head():
    # BOS and NYK finish level; NYK won the head-to-head meeting.
    games = [game("NYK", "BOS", 100, 90, 1), game("BOS", "MIA", 100, 90, 2), game("MIA", "NYK", 100, 90, 3)]
    table = standings(games, 2027)
    order = [row["team"] for row in table["conferences"]["East"]["divisions"]["Atlantic"]]
    assert order.index("NYK") < order.index("BOS")
    # Teams with no games tie at .000; every team still gets a unique seed.
    seeds = [row["seed"] for row in table["conferences"]["East"]["seeds"]]
    assert sorted(seeds) == list(range(1, 16))
    # BOS, NYK and MIA are all 1-1. NYK and MIA win their divisions, so BOS drops out of the
    # three-way tie first; MIA then beats NYK head-to-head, and NYK beats BOS head-to-head.
    assert [row["team"] for row in table["conferences"]["East"]["seeds"][:3]] == ["MIA", "NYK", "BOS"]
    assert {d for d in DIVISIONS["East"]} == set(table["conferences"]["East"]["divisions"])


def test_season_job_endpoint(monkeypatch, nba_sim):
    nba_server.set_sim(nba_sim)
    nba_server.SEASON_JOBS.clear()
    done, release = threading.Event(), threading.Event()
    seen = {}

    def fake_simulate(sim, season, **kwargs):
        seen.update(kwargs)
        release.wait(2)
        kwargs["progress"]("NBA Finals", 10, 10)
        done.set()
        return {"season": season, "champion": "BOS"}

    monkeypatch.setattr(nba_server, "simulate_season", fake_simulate)
    client = TestClient(combined.app)
    response = client.post("/api/nba/seasons", json={"season": 2027, "seed": 2, "strength_mode": "neutral"})
    assert response.status_code == 202
    job = response.json()
    assert job["progress"]["total"] == total_game_slots(2027)
    assert client.post("/api/nba/seasons", json={"season": 2027}).status_code == 429  # one at a time
    release.set()
    assert done.wait(2)
    status = client.get(f"/api/nba/seasons/{job['id']}").json()
    assert status["status"] == "complete" and status["result"]["champion"] == "BOS"
    assert len(seen["schedule"]) == REGULAR_SEASON_GAMES and seen["seed"] == 2
    assert seen["inference_context"] == nba_server._model_scheduler.background
    assert client.get("/api/nba/seasons/nope").status_code == 404
    assert client.post("/api/nba/seasons", json={"season": 2004}).status_code == 422


def test_real_rollouts_finish_a_short_batch(nba_sim):
    from runitback.nba.engine.rules import new_game
    starts = [new_game("BOS", "NYK", 2026), new_game("LAL", "DEN", 2026, "POST")]
    finals = nba_sim.rollout_games(starts, [nba_sim.ratings_for(s.home, s.away, 2026) for s in starts],
                                   seed=0, batch_size=2)
    assert [f.home for f in finals] == ["BOS", "LAL"]
    assert all(f.over and f.score[0] != f.score[1] for f in finals)
