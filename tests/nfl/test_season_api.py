import threading

from fastapi.testclient import TestClient

from runitback.nfl.server import app as server
from runitback.nfl.vocab import TEAMS


def full_schedule():
    rotating = list(TEAMS)
    schedule = []
    for week in range(17):
        schedule.extend({"week": week + 1, "home": rotating[index], "away": rotating[-index - 1]}
                        for index in range(16))
        rotating = [rotating[0], rotating[-1], *rotating[1:-1]]
    return schedule


def test_season_job_endpoint(monkeypatch, tiny_sim):
    server.set_sim(tiny_sim)
    server.SEASON_JOBS.clear()
    done = threading.Event()

    def fake_simulate(sim, schedule, season, **kwargs):
        kwargs["progress"]("Super Bowl", 285, 285)
        done.set()
        return {"season": season, "champion": "KC", "games": [], "playoffs": []}

    monkeypatch.setattr(server, "generate_schedule",
                        lambda sim, season, seed=0: {"games": full_schedule(), "bye_weeks": {},
                                                     "position_source": "test results"})
    monkeypatch.setattr(server, "simulate_season", fake_simulate)
    client = TestClient(server.app)
    response = client.post("/api/seasons", json={"season": 2026, "strength_mode": "neutral"})

    assert response.status_code == 202
    job_id = response.json()["id"]
    assert response.json()["progress"]["total"] == 285
    assert response.json()["position_source"] == "test results"
    assert "result" in response.json()
    assert done.wait(2)
    status = client.get(f"/api/seasons/{job_id}").json()
    assert status["status"] == "complete"
    assert status["progress"] == {"stage": "Super Bowl", "completed": 285, "total": 285}
    assert status["result"]["champion"] == "KC"


def test_season_job_rejects_schedule_generation_failure(monkeypatch, tiny_sim):
    server.set_sim(tiny_sim)

    def fail_schedule(*args, **kwargs):
        raise ValueError("could not generate schedule")

    monkeypatch.setattr(server, "generate_schedule", fail_schedule)
    client = TestClient(server.app)
    response = client.post("/api/seasons", json={"season": 2026, "strength_mode": "neutral"})

    assert response.status_code == 422
    assert response.json()["detail"] == "could not generate schedule"