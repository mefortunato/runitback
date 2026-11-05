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

from .conftest import toy_games

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
    # the deployed, combined server serves the same routes under /api/nfl
    from runitback.server import app as combined
    cc = TestClient(combined)
    assert cc.get("/api/nfl/teams/KC/lineup").json()["coach"] == "Andy Reid"
    assert cc.get("/api/nfl/players", params={"q": "kc_qb"}).status_code == 200
    assert cc.post("/api/nfl/simulate", json={"home": "KC", "away": "BUF", "n": 2, "seed": 0,
                                              "lineups": {"KC": {"depth_chart": {"QB": ["NEW"]}}}}).status_code == 200


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


def test_box_score_loss_is_differentiable_and_exact_for_real_outcomes(lineup_sim):
    from runitback.nfl.model.boxscore import GAME_STATS, game_box, true_play_stats
    from runitback.nfl.vocab import PT, RS
    # one game, 3 plays: home 12-yard TD run from the 12, away 30-yard completion, away interception
    t = torch.tensor([[PT["run"], PT["pass"], PT["pass"]]])
    r = torch.tensor([[RS["run"], RS["complete"], RS["interception"]]])
    y = torch.tensor([[12, 30, 0]])           # interception returned for a TD (spot 0)
    yl = torch.tensor([[12, 70, 40]])
    off, dfn = true_play_stats(t, r, y, yl)
    g = dict(zip(GAME_STATS, game_box(off, dfn, torch.tensor([[1, 0, 0]]), torch.ones(1, 3))[0].tolist()))
    assert g["home_points"] == 12 and g["home_rush_yds"] == 12 and g["away_pass_yds"] == 30
    assert g["away_turnovers"] == 1 and g["margin"] == 12
    # the model's expected box score carries gradients back to the play heads
    from runitback.nfl.model.boxscore import batch_box, box_loss
    m = lineup_sim.model.train()
    B, T = 1, 3
    feats = {f: torch.zeros(B, T, dtype=torch.long) for f in m.emb}
    feats |= {"cont": torch.zeros(B, T, 5), "yl": yl, "is_home": torch.tensor([[1, 0, 0]]), "phase": torch.full((B, T), 2)}
    feats |= lineup_sim._lineup_feats([[new_game("KC", "BUF", 2025)] * T], lineup_sim.lineups_for("KC", "BUF", 2025))
    batch = {"feats": feats, "valid": torch.ones(B, T), "type": t, "result": r, "yards": y + 99,
             "type_mask": torch.ones(B, T, 11, dtype=torch.bool)}
    pred, real = batch_box(m, m.encode(feats), m.personnel(feats), batch)
    loss, logs = box_loss(pred, real, torch.ones(len(GAME_STATS)) * 10)
    loss.backward()
    assert m.yards_head[-1].weight.grad.abs().sum() > 0 and m.type_head[-1].weight.grad.abs().sum() > 0
    m.eval()


@pytest.fixture(scope="module")
def who_sim():
    torch.manual_seed(0)
    ts = TeamSeasonVocab(["KC_2025", "BUF_2025"])
    meta = tiny_lineup_meta()
    model = PlayTransformer(ModelConfig(feature_sizes(len(ts)), d_model=32, n_layers=1, n_heads=2,
                                        n_cont=len(CONT_FEATURES), n_players=len(PlayerVocab(meta["player_vocab"])),
                                        n_coaches=5, player_dim=8, who_head=True))
    return Simulator(model, ts, {"lineup_meta": meta}, ratings=RatingsTable(compute_ratings(toy_games())))


def test_who_result_mask():
    from runitback.nfl.model.lineups import WHO_NONE
    from runitback.nfl.vocab import PT, RESULTS, RS
    t = torch.tensor([PT["pass"], PT["pass"], PT["run"]])
    who = torch.tensor([WHO_NONE, 3, 2])
    m = PlayTransformer.who_result_mask(t, who, len(RESULTS))
    assert not m[0, RS["complete"]] and m[0, RS["sack"]] and m[0, RS["incomplete"]]   # no target: no catch
    assert m[1, RS["complete"]] and not m[1, RS["sack"]]                              # a target: no sack
    assert m[2].all()


def test_who_head_attributes_runs_and_targets(who_sim):
    from runitback.nfl.model.lineups import N_SLOTS, SLOTS
    finals, logs = who_sim.rollout(new_game("KC", "BUF", 2025, opening_kicker=0), 3, seed=0, keep_plays=3)
    plays = [o for lg in logs for _, o in lg]
    runs = [o for o in plays if o.play_type == "run"]
    assert runs and all(o.who is not None for o in runs)                 # every run has a ball carrier
    assert all(o.who is None for o in plays if o.play_type not in ("run", "pass"))
    assert all(o.who is not None for o in plays if o.play_type == "pass" and o.result == "complete")
    assert all(o.who is None for o in plays if o.play_type == "pass" and o.result == "sack")
    # the tiny sampler only puts QB1 RB1 WR1-3 TE1 OL1-5 on the field
    on_field = {SLOT_IDX[s] for s in [("QB", 0), ("RB", 0), ("WR", 0), ("WR", 1), ("WR", 2), ("TE", 0)]
                + [("OL", i) for i in range(5)]}
    assert {o.who for o in plays if o.who is not None} <= on_field and all(o.who < N_SLOTS for o in runs)
    assert SLOTS[runs[0].who][0] in ("QB", "RB", "WR", "TE", "OL")


def test_who_loss_term(who_sim):
    from runitback.nfl.model.lineups import WHO_NONE
    from runitback.nfl.vocab import PT, RESULTS, RS
    m = who_sim.model.train()
    B, T = 1, 2
    st = [new_game("KC", "BUF", 2025)] * T
    feats = {f: torch.zeros(B, T, dtype=torch.long) for f in m.emb} | {"cont": torch.zeros(B, T, 5)}
    feats["phase"] = torch.full((B, T), 2)
    feats |= who_sim._lineup_feats([st], who_sim.lineups_for("KC", "BUF", 2025))
    feats["pers"] = torch.arange(1, 23).repeat(B, T, 1) + 30
    feats["pers_grp"] = torch.ones(B, T, 22, dtype=torch.long)
    feats["who"] = torch.tensor([[4, WHO_NONE]])
    t = torch.tensor([[PT["pass"], PT["pass"]]]); r = torch.tensor([[RS["complete"], RS["sack"]]])
    batch = {"feats": feats, "valid": torch.ones(B, T), "type": t, "result": r,
             "yards": torch.full((B, T), 105), "elapsed": torch.full((B, T), 5),
             "type_mask": torch.ones(B, T, 11, dtype=torch.bool), "result_mask": torch.ones(B, T, len(RESULTS), dtype=torch.bool),
             "yards_mask": torch.ones(B, T, 200, dtype=torch.bool), "elapsed_mask": torch.ones(B, T, 61, dtype=torch.bool)}
    loss, logs = m.loss(batch)
    assert "who" in logs and torch.isfinite(loss)
    loss.backward()
    assert m.who_ctx.weight.grad.abs().sum() > 0
    m.eval()


def test_inclusion_sample_hits_target_probabilities():
    from runitback.nfl.model.lineups import inclusion_sample
    rng = np.random.default_rng(0)
    r, k, n = np.array([0.9, 0.5, 0.3, 0.1]), 2, 20000
    cnt = np.zeros(4)
    for _ in range(n):
        pick = inclusion_sample(r, k, rng)
        assert len(set(pick.tolist())) == k
        cnt[pick] += 1
    # capped at 1, the rest proportional: [1, .556, .333, .111]
    assert np.allclose(cnt / n, [1.0, 5 / 9, 3 / 9, 1 / 9], atol=0.015)


def test_usage_mode_follows_player_snap_shares(lineup_sim):
    r = lineup_sim.lineups
    role = {g: [0.6, 0.3, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05][:n] for g, n in
            (("QB", 3), ("RB", 4), ("WR", 6), ("TE", 4), ("OL", 9), ("DL", 9), ("LB", 7), ("DB", 9))}
    role |= {"K": [1.0], "P": [1.0], "LS": [1.0]}
    from runitback.nfl.model.lineups import PersonnelSampler
    sampler = PersonnelSampler(r.sampler.table, role)
    chart = {"RB": ["kc_RB1", "bf_RB1"]}  # two backs; the package holds one
    spec = lineup_sim.lineup_spec("KC", "BUF", {"KC": {"depth_chart": chart}})["home"]["depth_chart"]
    toks = r.tokens(spec)
    rng = np.random.default_rng(0)

    def rb1_rate(usage):
        on = 0
        for _ in range(3000):
            _, _, c = sampler.sample("off", "run", 1, 10, 60, toks, rng, codes=True, usage=usage)
            on += SLOT_IDX[("RB", 0)] in c
        return on / 3000
    even = np.where(toks > 0, 0.5, 0.0)
    heavy = even.copy(); heavy[SLOT_IDX[("RB", 0)]], heavy[SLOT_IDX[("RB", 1)]] = 0.9, 0.1
    assert abs(rb1_rate(even) - 0.5) < 0.05 and abs(rb1_rate(heavy) - 0.9) < 0.04
    # snap_shares overrides reach the per-slot usage; checkpoints without a role table keep league rotation
    r.role_share = role
    u = r.usage(spec, toks, {"kc_RB1": 0.95})
    assert u[SLOT_IDX[("RB", 0)]] == 0.95 and 0 < u[SLOT_IDX[("RB", 1)]] < 0.95
    r.role_share = None
    assert r.usage(spec, toks) is None


def test_usage_features_are_leak_free():
    from runitback.nfl.data.players import USAGE_FEATURES, usage_features_vector
    assert usage_features_vector(None) == [0.0] * len(USAGE_FEATURES)
    st = {"carry_share": 0.6, "target_share": 0.1, "snap_share": 0.7, "carries": 15, "rush_yds": 70,
          "targets": 3, "rec_yds": 20, "n": 5}
    v = usage_features_vector(st)
    assert v[0] == 0.6 and abs(v[3] - (74.3 / 16) / 10) < 1e-9


@pytest.mark.skipif(not (PROCESSED / "usage.parquet").exists(), reason="usage features not built")
def test_usage_parquet_uses_only_earlier_games():
    u = pd.read_parquet(PROCESSED / "usage.parquet")
    pg = pd.read_parquet(PROCESSED / "player_games.parquet", columns=["game_id", "player_id", "season", "week"])
    d = u.merge(pg, on=["game_id", "player_id"]).sort_values(["season", "week"])
    first = d.groupby("player_id").head(1)
    assert (first.history == 0).all() and (first.carry_share == 0).all()  # nothing known before a debut


def test_who_head_uses_usage_features():
    torch.manual_seed(0)
    ts = TeamSeasonVocab(["KC_2025", "BUF_2025"])
    meta = tiny_lineup_meta()
    m = PlayTransformer(ModelConfig(feature_sizes(len(ts)), d_model=32, n_layers=1, n_heads=2,
                                    n_cont=len(CONT_FEATURES), n_players=len(PlayerVocab(meta["player_vocab"])),
                                    n_coaches=5, player_dim=8, who_head=True, n_usage=6)).eval()
    pers = torch.arange(1, 23)[None] + 30
    grp = torch.ones(1, 22, dtype=torch.long)
    u0 = torch.zeros(1, 11, 6)
    u1 = u0.clone(); u1[0, 3, 0] = 0.8  # player 4 now has an 80% carry share
    with torch.no_grad():
        c0, c1 = m.who_candidates(pers, grp, u0), m.who_candidates(pers, grp, u1)
    assert torch.allclose(c0[0, :3], c1[0, :3]) and not torch.allclose(c0[0, 3], c1[0, 3])


RAW = Path(__file__).resolve().parents[2] / "data" / "nfl" / "raw"


@pytest.mark.skipif(not (RAW / "pbp_2025.parquet").exists() or not (RAW / "players" / "players.parquet").exists(),
                    reason="raw data not downloaded")
def test_refresh_lineups_rebuilds_data_but_keeps_weights(lineup_sim, tmp_path):
    from runitback.nfl.refresh import refresh_checkpoint
    from runitback.nfl.release import sha256
    m = lineup_sim.model
    meta = lineup_sim.meta["lineup_meta"]
    ck = tmp_path / "model.pt"
    torch.save({"state_dict": m.state_dict(), "config": m.cfg.to_dict(), "team_seasons": lineup_sim.ts.keys,
                "lineup_meta": meta, "ratings": None}, ck)
    out = tmp_path / "refreshed.pt"
    s = refresh_checkpoint(ck, out, RAW, tmp_path / "processed", 2025, 2025, download=False)
    assert s["sha256"] == sha256(out) and s["ratings_through"].startswith("2025")
    new = torch.load(out, weights_only=False)
    assert all(torch.equal(new["state_dict"][k], v) for k, v in m.state_dict().items())      # weights untouched
    lm = new["lineup_meta"]
    assert lm["player_vocab"] == meta["player_vocab"] and lm["personnel_sampler"] == meta["personnel_sampler"]
    assert len(lm["lineups"]) == 32 and lm["lineups"]["KC"]["season"] == 2025   # rebuilt from the data
    assert new["ratings"] and new["refreshed"]["seasons"] == [2025, 2025]
    sim = Simulator.load(out)
    assert sim.lineups.default("KC")["depth_chart"]["QB"]
    assert simulate_games(sim, "KC", "BUF", 2025, n=2, seed=0).n == 2
