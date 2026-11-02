from fastapi.testclient import TestClient

from runitback import server as combined
from runitback.nba.server import app as nba_server


def test_interactive_session(nba_sim):
    nba_server.set_sim(nba_sim)
    c = TestClient(combined.app)
    t = c.get("/api/nba/teams").json()
    assert t["model_loaded"] and t["seasons"] == [2026] and "BOS" in t["teams"]
    g = c.post("/api/nba/games", json={"home": "BOS", "away": "NYK", "season": 2026, "seed": 3}).json()
    assert g["state"]["phase"] == "tip" and g["legal_play_types"] == ["jump_ball"]
    r = c.post(f"/api/nba/games/{g['id']}/step", json={}).json()
    assert len(r["new_plays"]) == 1 and r["plays"][0]["outcome"]["play_type"] == "jump_ball"
    assert c.post(f"/api/nba/games/{g['id']}/step", json={"play_type": "free_throw"}).status_code == 422
    r = c.post(f"/api/nba/games/{g['id']}/step", json={"play_type": "shot3"}).json()
    assert r["new_plays"][0]["outcome"]["play_type"] == "shot3"
    pos = r["state"]["pos"]
    r = c.post(f"/api/nba/games/{g['id']}/step", json={"count": 50, "until": "possession"}).json()
    assert r["state"]["pos"] != pos or r["state"]["qtr"] > 1 or len(r["new_plays"]) == 50
    n = len(r["plays"])
    r = c.post(f"/api/nba/games/{g['id']}/undo").json()
    assert len(r["plays"]) == n - 1
    wp = c.get(f"/api/nba/games/{g['id']}/winprob", params={"n": 10}).json()
    assert abs(wp["home_win"] + wp["away_win"] + wp["tie"] - 1) < 1e-9
    r = c.post(f"/api/nba/games/{g['id']}/step", json={"count": 800, "until": "period"}).json()
    assert r["state"]["qtr"] == 2 or r["state"]["over"]
    r = c.post(f"/api/nba/games/{g['id']}/step", json={"count": 800}).json()
    assert r["state"]["over"] and r["state"]["home_score"] != r["state"]["away_score"]


def test_simulate_endpoint(nba_sim):
    nba_server.set_sim(nba_sim)
    c = TestClient(combined.app)
    r = c.post("/api/nba/simulate", json={"home": "BOS", "away": "NYK", "season": 2026, "n": 4, "seed": 0})
    assert r.status_code == 200 and r.json()["n"] == 4
    assert c.post("/api/nba/simulate", json={"home": "XXX", "away": "BOS"}).status_code == 422
    assert c.post("/api/nba/simulate", json={"home": "BOS", "away": "BOS"}).status_code == 422


def test_teams_without_a_model_explain_how_to_get_one(monkeypatch, tmp_path):
    nba_server.set_sim(None)
    monkeypatch.setenv("RUNITBACK_NBA_CKPT", str(tmp_path / "missing.pt"))
    c = TestClient(combined.app)
    t = c.get("/api/nba/teams").json()
    assert not t["model_loaded"] and len(t["teams"]) == 30 and "runitback nba train" in t["error"]
    assert c.post("/api/nba/games", json={"home": "BOS", "away": "NYK"}).status_code == 503


def test_combined_app_serves_both_sports_and_legacy_nfl_paths():
    from runitback.nfl.data.ratings import RatingsTable
    from runitback.nfl.model.features import TeamSeasonVocab, feature_sizes
    from runitback.nfl.model.transformer import ModelConfig, PlayTransformer
    from runitback.nfl.server import app as nfl_server
    from runitback.nfl.sim.sampler import Simulator
    from ..nfl.conftest import toy_games
    from runitback.nfl.data.ratings import compute_ratings
    ts = TeamSeasonVocab(["KC_2025", "BUF_2025"])
    model = PlayTransformer(ModelConfig(feature_sizes(len(ts)), d_model=16, n_layers=1, n_heads=2))
    nfl_server.set_sim(Simulator(model, ts, ratings=RatingsTable(compute_ratings(toy_games()))))
    c = TestClient(combined.app)
    assert c.get("/api/nfl/teams").json()["seasons"] == [2025]
    assert c.get("/api/teams").json()["seasons"] == [2025]  # pre-NBA clients
    sports = {s["id"] for s in c.get("/api/sports").json()["sports"]}
    assert sports == {"nfl", "nba"}
