from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from runitback.nfl.data.players import build_coaches, coach_record
from runitback.nfl.data.ratings import RatingsTable, compute_ratings
from runitback.nfl.engine.rules import new_game
from runitback.nfl.model.features import CONT_FEATURES, TeamSeasonVocab, feature_sizes
from runitback.nfl.model.lineups import (
    N_GROUPS, N_SLOTS, NEW_GAMES, SLOT_IDX, PersonnelSampler, PlayerVocab, new_token, other_token, signature,
)
from runitback.nfl.model.transformer import ModelConfig, PlayTransformer
from runitback.nfl.server import app as server
from runitback.nfl.sim.api import simulate_games
from runitback.nfl.sim.sampler import Simulator

from conftest import toy_games

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "nfl" / "processed"

# 11 offensive and 11 defensive slots: QB1 RB1 WR1-3 TE1 OL1-5 / DL1-4 LB1-3 DB1-4
OFF_SIG = [SLOT_IDX[s] for s in [("QB", 0), ("RB", 0), ("WR", 0), ("WR", 1), ("WR", 2), ("TE", 0)] +
           [("OL", i) for i in range(5)]]
DEF_SIG = [SLOT_IDX[s] for s in [("DL", i) for i in range(4)] + [("LB", i) for i in range(3)] +
           [("DB", i) for i in range(4)]]


def _chart(prefix: str) -> dict:
    return {"QB": [f"{prefix}QB1", f"{prefix}QB2"], "RB": [f"{prefix}RB1"], "WR": [f"{prefix}WR{i}" for i in range(3)],
            "TE": [f"{prefix}TE1"], "OL": [f"{prefix}OL{i}" for i in range(5)],
            "DL": [f"{prefix}DL{i}" for i in range(4)], "LB": [f"{prefix}LB{i}" for i in range(3)],
            "DB": [f"{prefix}DB{i}" for i in range(4)], "K": [f"{prefix}K"], "P": [f"{prefix}P"], "LS": [f"{prefix}LS"]}


def tiny_lineup_meta() -> dict:
    players, ids = {}, []
    for team, prefix in (("KC", "kc_"), ("BUF", "bf_")):
        for g, pids in _chart(prefix).items():
            for pid in pids:
                players[pid] = [pid, g, g, team, 2025, 50]
                ids.append(pid)
    players["kc_QB2"][5] = 1  # a rookie backup: still NEW
    sampler = {}
    for key in ("run", "pass", "kickoff", "punt", "field_goal", "extra_point", "two_point", "timeout",
                "penalty", "qb_kneel", "qb_spike"):
        sampler[f"off|{key}"] = [[sorted(OFF_SIG)], [1.0]]
        sampler[f"def|{key}"] = [[sorted(DEF_SIG)], [1.0]]
    return {"lineups": {t: {"game_id": "g3", "season": 2025, "week": 1, "depth_chart": _chart(p),
                            "coach": c} for t, p, c in (("KC", "kc_", "Andy Reid"), ("BUF", "bf_", "Sean McDermott"))},
            "players": players, "player_vocab": sorted(ids),
            "coach_history": {"Andy Reid": [[2023, "KC", 11, 6, 0], [2024, "KC", 15, 2, 0]],
                              "Sean McDermott": [[2024, "BUF", 13, 4, 0]]},
            "coach_vocab": ["Andy Reid", "Sean McDermott"], "veteran_coaches": [], "personnel_sampler": sampler}


@pytest.fixture(scope="module")
def lineup_sim():
    torch.manual_seed(0)
    ts = TeamSeasonVocab(["KC_2025", "BUF_2025"])
    meta = tiny_lineup_meta()
    model = PlayTransformer(ModelConfig(feature_sizes(len(ts)), d_model=32, n_layers=1, n_heads=2,
                                        n_cont=len(CONT_FEATURES), n_players=len(PlayerVocab(meta["player_vocab"])),
                                        n_coaches=2 + 3))
    return Simulator(model, ts, {"lineup_meta": meta}, ratings=RatingsTable(compute_ratings(toy_games())))


def test_player_tokens_follow_new_player_rule():
    v = PlayerVocab(["a", "b"])
    assert v.token("a", "QB", 0) == new_token("QB")                 # first career game
    assert v.token("a", "QB", NEW_GAMES - 1) == new_token("QB")
    assert v.token("a", "QB", NEW_GAMES) == v.idx["a"]               # has his own embedding afterwards
    assert v.token("zzz", "WR", 40) == other_token("WR")             # veteran without an embedding
    assert v.token("zzz", "WR", None) == new_token("WR")             # never seen: NEW
    assert v.token("NEW", "DB", None) == new_token("DB")
    assert len({new_token(g) for g in range(N_GROUPS)} | {other_token(g) for g in range(N_GROUPS)}) == 2 * N_GROUPS


def test_personnel_sampler_fills_from_depth_chart():
    sig = signature([0, 3, -1], [1, 2, 3])
    s = PersonnelSampler.fit({"off|pass": __import__("collections").Counter({sig: 1})})
    lineup = np.zeros(N_SLOTS, dtype=np.int64)
    lineup[0] = 99  # QB1
    toks, grps = s.sample("off", "pass", 1, 10, 75, lineup, np.random.default_rng(0))
    assert toks[0] == 99 and grps[0] == 1                            # QB1 from the depth chart
    assert toks[1] == other_token(1)                                 # empty RB slot: OTHER_RB
    assert toks[2] == other_token(2) and toks[3:] == [0] * 8         # off-chart WR, then padding


def test_coach_records_are_leak_free():
    sched = pd.DataFrame({"game_id": ["a", "b", "c"], "season": [2020, 2020, 2021], "gameday": ["1", "2", "3"],
                          "home": ["KC", "KC", "KC"], "away": ["BUF", "BUF", "BUF"],
                          "home_coach": ["X", "X", "X"], "away_coach": ["Y", "Y", "Z"],
                          "home_score": [20, 30, 10], "away_score": [10, 3, 7]})
    c = build_coaches(sched).set_index(["game_id", "side"])
    assert c.loc[("a", "home"), "career_games"] == 0                 # nothing known before his first game
    assert c.loc[("b", "home"), "career_games"] == 1 and c.loc[("b", "home"), "career_win_pct"] > 0.5
    assert c.loc[("c", "home"), "hc_seasons"] == 1 and c.loc[("c", "home"), "team_seasons"] == 1
    assert c.loc[("c", "away"), "career_games"] == 0                 # a new coach
    assert coach_record([(2020, "KC", 2, 0, 0)], "X", "KC", 2020)["career_games"] == 0


def test_lineup_overrides_and_defaults(lineup_sim):
    spec = lineup_sim.lineup_spec("KC", "BUF", {"KC": {"depth_chart": {"QB": ["NEW"]}, "coach": "Someone New"}})
    assert spec["home"]["depth_chart"]["QB"] == ["NEW"] and spec["home"]["coach"] == "Someone New"
    assert spec["home"]["depth_chart"]["WR"] == _chart("kc_")["WR"]  # other groups keep the default
    assert spec["away"]["coach"] == "Sean McDermott"
    lu = lineup_sim.lineups_for("KC", "BUF", 2026, {"KC": {"depth_chart": {"QB": ["NEW"]}, "coach": "Someone New"}})
    assert lu.tokens[0, SLOT_IDX[("QB", 0)]] == new_token("QB")
    assert lu.coach[0][0] == 1                                       # NEW_COACH
    default = lineup_sim.lineups_for("KC", "BUF", 2026)
    r = lineup_sim.lineups
    assert default.tokens[0, SLOT_IDX[("QB", 0)]] == r.pvocab.idx["kc_QB1"]
    assert default.tokens[0, SLOT_IDX[("QB", 1)]] == new_token("QB")  # rookie backup with 1 game


def test_lineups_change_model_inputs(lineup_sim):
    m = lineup_sim.model.eval()
    base = lineup_sim.lineups_for("KC", "BUF", 2026)
    swapped = lineup_sim.lineups_for("KC", "BUF", 2026, {"KC": {"depth_chart": {"QB": ["kc_QB2"]}}})
    with torch.no_grad():
        a, b = m.lineup.teams(torch.as_tensor(base.tokens)[None]), m.lineup.teams(torch.as_tensor(swapped.tokens)[None])
        unk = m.lineup.teams(torch.zeros(1, 2, N_SLOTS, dtype=torch.long))
    assert not torch.allclose(a[0, 0], b[0, 0]) and torch.allclose(a[0, 1], b[0, 1])
    assert torch.allclose(unk[0, 0], m.lineup.unk_team)


def test_lineup_rollouts_finish(lineup_sim):
    finals, logs = lineup_sim.rollout(new_game("KC", "BUF", 2025, opening_kicker=0), 4, seed=0, keep_plays=1)
    assert all(s.over for s in finals)
    r = simulate_games(lineup_sim, "KC", "BUF", 2026, n=6, seed=1,
                       lineups={"KC": {"depth_chart": {"QB": ["NEW"], "K": []}}})
    assert r.n == 6


def test_lineup_endpoints(lineup_sim):
    server.set_sim(lineup_sim)
    c = TestClient(server.app)
    assert c.get("/api/teams").json()["lineups"] is True
    d = c.get("/api/teams/KC/lineup").json()
    assert d["coach"] == "Andy Reid" and d["coach_info"]["career_games"] == 34
    assert [p["id"] for p in d["depth_chart"]["QB"]] == ["kc_QB1", "kc_QB2"]
    assert d["depth_chart"]["QB"][1]["status"] == "new"
    assert "QB" in d["pool"] and d["max_depth"]["QB"] == 3
    assert c.get("/api/players", params={"q": "bf_qb"}).json()["players"][0]["id"] == "bf_QB1"
    assert any(x["name"] == "Andy Reid" for x in c.get("/api/coaches").json()["coaches"])
    g = c.post("/api/games", json={"home": "KC", "away": "BUF", "season": 2026, "seed": 3,
                                   "lineups": {"KC": {"depth_chart": {"QB": ["NEW", "kc_QB1"]}}}}).json()
    assert g["lineups"]["home"]["depth_chart"]["QB"] == ["NEW", "kc_QB1"]
    g = c.post(f"/api/games/{g['id']}/step", json={"count": 5}).json()
    assert len(g["plays"]) == 5
    bad = c.post("/api/simulate", json={"home": "KC", "away": "BUF", "n": 4,
                                        "lineups": {"KC": {"depth_chart": {"QBX": ["a"]}}}})
    assert bad.status_code == 422
    ok = c.post("/api/simulate", json={"home": "KC", "away": "BUF", "n": 4, "seed": 0,
                                       "lineups": {"KC": {"coach": "Brand New", "depth_chart": {"QB": ["NEW"]}}}})
    assert ok.status_code == 200 and ok.json()["n"] == 4


def test_team_only_model_rejects_lineups(tiny_sim):
    server.set_sim(tiny_sim)
    c = TestClient(server.app)
    assert c.get("/api/teams").json()["lineups"] is False
    assert c.get("/api/teams/KC/lineup").status_code == 404
    r = c.post("/api/simulate", json={"home": "KC", "away": "BUF", "n": 2, "lineups": {"KC": {"coach": "X"}}})
    assert r.status_code == 422


@pytest.mark.skipif(not (PROCESSED / "player_games.parquet").exists(), reason="player data not built")
def test_real_lineups_and_personnel_encode():
    from runitback.nfl.model.dataset import build_team_seasons, collate, encode_games
    from runitback.nfl.model.lineups import LineupData
    games = pd.read_parquet(PROCESSED / "games.parquet").query("season == 2024").head(3)
    plays = pd.read_parquet(PROCESSED / "plays.parquet")
    plays = plays[plays.game_id.isin(games.game_id)]
    ld = LineupData.load(PROCESSED, games.game_id)
    items = encode_games(plays, games, build_team_seasons(games), 400, None, ld)
    for it in items:
        assert (it["lineup"] > 0).sum(1).min() >= 22                 # both teams have full depth charts
        scrimmage = np.isin(it["type"], [0, 1])  # noqa: F841 (types vary by vocab order)
        has = (it["pers"] > 0).all(1)
        assert has.mean() > 0.8                                      # most plays have all 22 players
    batch = collate(items)
    assert batch["feats"]["pers"].shape[-1] == 22 and batch["feats"]["lineup"].shape[1:] == (2, N_SLOTS)
