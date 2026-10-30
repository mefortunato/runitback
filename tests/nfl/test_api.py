import threading

from fastapi.testclient import TestClient

from runitback.nfl.server import app as server


def test_model_scheduler_prioritizes_interactive_requests():
    scheduler = server.ModelScheduler()
    background_inside = threading.Event()
    release_background = threading.Event()
    interactive_inside = threading.Event()
    release_interactive = threading.Event()
    background_resumed = threading.Event()

    def background_work():
        with scheduler.background():
            background_inside.set()
            assert release_background.wait(2)
        with scheduler.background():
            background_resumed.set()

    def interactive_work():
        assert background_inside.wait(2)
        with scheduler.interactive():
            interactive_inside.set()
            assert release_interactive.wait(2)

    background = threading.Thread(target=background_work)
    interactive = threading.Thread(target=interactive_work)
    background.start()
    interactive.start()
    try:
        assert background_inside.wait(2)
        with scheduler._condition:
            assert scheduler._condition.wait_for(lambda: scheduler._interactive_waiters > 0, timeout=2)
        release_background.set()
        assert interactive_inside.wait(2)
        assert not background_resumed.is_set()
        release_interactive.set()
        assert background_resumed.wait(2)
    finally:
        release_background.set()
        release_interactive.set()
        background.join(2)
        interactive.join(2)
    assert not background.is_alive()
    assert not interactive.is_alive()


def test_interactive_session(tiny_sim):
    server.set_sim(tiny_sim)
    c = TestClient(server.app)
    assert c.get("/api/teams").json()["seasons"] == [2025]
    g = c.post("/api/games", json={"home": "KC", "away": "BUF", "season": 2025, "seed": 3}).json()
    assert g["state"]["phase"] == "kickoff" and g["legal_play_types"][0] == "kickoff"
    r = c.post(f"/api/games/{g['id']}/step", json={}).json()
    assert len(r["new_plays"]) == 1 and r["plays"][0]["outcome"]["play_type"] == "kickoff"
    assert c.post(f"/api/games/{g['id']}/step", json={"play_type": "field_goal"}).status_code in (200, 422)
    r = c.post(f"/api/games/{g['id']}/step", json={"count": 5}).json()
    n = len(r["plays"])
    r = c.post(f"/api/games/{g['id']}/undo").json()
    assert len(r["plays"]) == n - 1
    wp = c.get(f"/api/games/{g['id']}/winprob", params={"n": 10}).json()
    assert abs(wp["home_win"] + wp["away_win"] + wp["tie"] - 1) < 1e-9
    r = c.post(f"/api/games/{g['id']}/step", json={"count": 400}).json()
    assert r["state"]["over"]


def test_forced_play_call(tiny_sim):
    server.set_sim(tiny_sim)
    c = TestClient(server.app)
    g = c.post("/api/games", json={"home": "KC", "away": "BUF", "season": 2025, "seed": 1}).json()
    c.post(f"/api/games/{g['id']}/step", json={})
    while True:
        r = c.get(f"/api/games/{g['id']}").json()
        if "pass" in r["legal_play_types"]:
            break
        c.post(f"/api/games/{g['id']}/step", json={})
    r = c.post(f"/api/games/{g['id']}/step", json={"play_type": "pass"}).json()
    assert r["new_plays"][0]["outcome"]["play_type"] == "pass"


def test_simulate_endpoint(tiny_sim):
    server.set_sim(tiny_sim)
    c = TestClient(server.app)
    r = c.post("/api/simulate", json={"home": "KC", "away": "BUF", "season": 2025, "n": 4, "seed": 0})
    assert r.status_code == 200 and r.json()["n"] == 4
    assert c.post("/api/simulate", json={"home": "XXX", "away": "BUF"}).status_code == 422
