"""The college football simulator and its API: the NFL sampler under the college rules engine."""
from __future__ import annotations

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from runitback.cfb.data import preprocess as P
from runitback.cfb.engine import rules as cfb_rules
from runitback.cfb.foundation import CFBGenerator
from runitback.cfb.server import app as server
from runitback.cfb.sim.api import simulate_games
from runitback.cfb.sim.sampler import Simulator, attach_team_names
from runitback.nfl.engine import rules as nfl_rules
from runitback.nfl.sim.sampler import Simulator as NFLSimulator

from .test_preprocess import _raw


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cfbsim")
    P.preprocess([2021, 2023], _raw(tmp), tmp / "processed")
    out = tmp / "prod"
    path = CFBGenerator(epochs=1, d_model=16, n_layers=1, n_heads=2).train(tmp / "processed", out, None)
    attach_team_names(path, tmp / "processed")
    return path


@pytest.fixture(scope="module")
def sim(ckpt):
    return Simulator.load(ckpt)


def test_engines_are_per_sport():
    assert NFLSimulator.engine is nfl_rules and Simulator.engine is cfb_rules


def test_checkpoint_carries_schools(sim):
    names = sim.team_names()
    assert names and all(isinstance(k, str) and isinstance(v, str) for k, v in names.items())
    assert set(sim.teams()) <= set(names)  # every team in the latest season has a school name
    assert sim.seasons[-1] == 2023


def test_rollout_plays_legal_college_games(sim):
    a, b = sim.teams()[:2]
    start = cfb_rules.new_game(a, b, 2023, opening_kicker=0)
    finals, logs = sim.rollout(start, 6, seed=1, keep_plays=6, ratings=None, lineups=None)
    assert len(finals) == 6
    for f in finals:
        assert isinstance(f, cfb_rules.CFBState) and f.score[0] >= 0 and f.score[1] >= 0
    for log in logs:
        for s, o in log:  # every sampled outcome is legal under the college rules
            assert o.play_type in cfb_rules.legal_types(s)
            assert o.result in cfb_rules.legal_results(s, o.play_type)
            if s.phase == 1:  # a try is snapped from the 3
                assert s.yl == cfb_rules.PAT_SPOT


def test_simulate_games_summarizes(sim):
    a, b = sim.teams()[:2]
    r = simulate_games(sim, a, b, 2023, n=4, seed=3)
    assert r.n == 4 and r.home == a and abs(r.home_win + r.away_win + r.tie - 1) < 1e-9


@pytest.fixture
def client(sim):
    server.set_sim(sim)
    server.SESSIONS.clear()
    api = FastAPI()
    api.include_router(server.router, prefix="/api")
    yield TestClient(api)
    server.set_sim(None)


def test_api_game_session(client, sim):
    teams = client.get("/api/teams").json()
    assert teams["seasons"][-1] == 2023 and all({"id", "name"} == set(t) for t in teams["teams"])
    a, b = teams["teams"][0], teams["teams"][1]
    g = client.post("/api/games", json={"home": a["id"], "away": b["id"], "season": 2023, "seed": 5}).json()
    assert g["state"]["home"] == a["name"] and g["state"]["away"] == b["name"]  # the API speaks school names
    s = client.post(f"/api/games/{g['id']}/step", json={"count": 5}).json()
    assert len(s["new_plays"]) == 5 and all(a["name"] in p["text"] or b["name"] in p["text"] for p in s["new_plays"])
    u = client.post(f"/api/games/{g['id']}/undo?count=2").json()
    assert len(u["plays"]) == 3
    wp = client.get(f"/api/games/{g['id']}/winprob?n=10").json()
    assert 0 <= wp["home_win"] <= 1
    r = client.post("/api/simulate", json={"home": a["id"], "away": b["id"], "season": 2023, "n": 4, "seed": 1}).json()
    assert r["home"] == a["name"] and r["n"] == 4


def test_api_rejects_unknown_team_and_missing_model(client, monkeypatch, tmp_path):
    assert client.post("/api/games", json={"home": "nope", "away": "also", "season": 2023}).status_code == 422
    server.set_sim(None)
    monkeypatch.setenv("RUNITBACK_CFB_CKPT", str(tmp_path / "missing.pt"))
    r = client.get("/api/teams")
    assert r.status_code == 503 and "runitback cfb fetch-model" in r.json()["detail"]


def test_release_registry(tmp_path):
    from runitback.cfb import release

    assert release.LATEST is None and release.MODELS == {}  # nothing released: the image skips the model
    with pytest.raises(ValueError, match="no college football model"):
        release.fetch_model(tmp_path / "m.pt")
    path, ck = tmp_path / "releases.json", tmp_path / "model.pt"
    path.write_text('{"latest": null, "models": {}}')
    ck.write_bytes(b"model")
    rel = release.register_release("v0.1.0", ck, path)
    assert rel["latest"] == "v0.1.0"
    assert rel["models"]["v0.1.0"] == {"tag": "cfb-v0.1.0", "asset": "runitback-cfb-model-v0.1.0.pt",
                                       "sha256": release.sha256(ck)}
    with pytest.raises(ValueError, match="already registered"):
        release.register_release("v0.1.0", ck, path)
    with pytest.raises(ValueError, match="not a vMAJOR"):
        release.register_release("0.2", ck, path)
    got = release.fetch_release(tmp_path / "dl.pt", "v0.1.0", {"v0.1.0": ("t", "a", release.sha256(ck))},
                                url=ck.as_uri())
    assert got.read_bytes() == b"model"


def test_server_reports_the_generator_without_a_model(monkeypatch, tmp_path):
    import runitback.server as main

    monkeypatch.setenv("RUNITBACK_CFB_CKPT", str(tmp_path / "none.pt"))
    cfb = {s["id"]: s for s in TestClient(main.app).get("/api/sports").json()["sports"]}["cfb"]
    assert cfb["generator"] is True and cfb["model_available"] is False
