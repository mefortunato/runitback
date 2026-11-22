"""League parameterization (C2): the NBA is exactly what it was, and the leagues' sizes and rules."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from runitback.nba.engine.rules import Outcome, apply_outcome, elapsed_range, legal_results, legal_types, new_game
from runitback.nba.league import CBB, LEAGUES, NBA, WCBB, Bonus, get_league
from runitback.nba.model.dataset import encode_games
from runitback.nba.model.features import TeamSeasonVocab, encode_state, feature_sizes
from runitback.nba.model.transformer import EventTransformer, ModelConfig

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "golden_nba.json").read_text())


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _game(rng, gid, home, away, season):
    s = new_game(home, away, season, "REG" if gid % 2 else "POST")
    rows, states = [], []
    while not s.over and len(rows) < 700:
        t = rng.choice(legal_types(s))
        r = rng.choice(legal_results(s, t))
        lo, hi = elapsed_range(s, t)
        e = int(hi if t == "run_out" else rng.integers(min(lo + 4, hi), hi + 1))
        states.append(s.to_dict())
        rows.append(dict(game_id=f"g{gid}", idx=len(rows), qtr=s.qtr, qsec=s.qsec, phase=s.phase, pos=s.pos,
                         fts_left=s.fts_left, score_h=s.score[0], score_a=s.score[1], to_h=s.timeouts[0],
                         to_a=s.timeouts[1], play_type=str(t), result=str(r), elapsed=e))
        s = apply_outcome(s, Outcome(str(t), str(r), e))
    meta = dict(game_id=f"g{gid}", season=season, game_type="REG" if gid % 2 else "POST",
                game_date=f"{season}-01-{gid % 28 + 1:02d}", home=home, away=away, home_final=s.score[0],
                away_final=s.score[1], tip_winner=0, home_spread=None, poss_h=100, poss_a=100)
    return meta, rows, states, s.to_dict()


def test_the_nba_engine_and_features_are_unchanged():
    """Digests recorded from the engine before it was parameterized by league (seeded random legal games:
    every state, every encoded tensor, the embedding-table sizes)."""
    rng = np.random.default_rng(123)
    teams = ["BOS", "NYK", "LAL", "GSW"]
    metas, plays = [], []
    for i in range(8):
        m, r, st, fin = _game(rng, i, teams[i % 4], teams[(i + 1) % 4], 2024 + i % 2)
        metas.append(m)
        plays += r
        assert GOLDEN[f"game{i}"] == dict(n=len(r), rows=_digest(r), states=_digest(st), final=_digest(fin))
    games, p = pd.DataFrame(metas), pd.DataFrame(plays)
    ts = TeamSeasonVocab([f"{t}_{s}" for t in teams for s in (2024, 2025)])
    h = hashlib.sha256()
    for it in encode_games(p, games, ts, 512):
        for k in sorted(it):
            v = it[k]
            h.update(k.encode())
            h.update(np.ascontiguousarray(v).tobytes() if isinstance(v, np.ndarray) else str(v).encode())
    assert h.hexdigest() == GOLDEN["encoded"]
    assert feature_sizes(len(ts)) == GOLDEN["feature_sizes"]
    assert _digest(encode_state(new_game("BOS", "NYK", 2025), None, ts)) == GOLDEN["encode_state"]


def test_an_old_checkpoint_config_is_the_nba():
    """Configs saved before leagues have no `league` key; they must still build the NBA model."""
    sizes = feature_sizes(5)
    cfg = {k: v for k, v in ModelConfig(sizes, d_model=16, n_layers=1, n_heads=2).to_dict().items() if k != "league"}
    model = EventTransformer(ModelConfig(**cfg))
    assert model.features == NBA.features and model.res_head[-1].out_features == 18
    assert ts_checkpoint_roundtrip().league is NBA


def ts_checkpoint_roundtrip() -> TeamSeasonVocab:
    return TeamSeasonVocab.from_checkpoint({"team_seasons": ["BOS_2025"]})


def test_league_registry_and_sizes():
    assert set(LEAGUES) == {"nba", "cbb", "wcbb", "wnba"} and get_league(None) is NBA and get_league("wcbb") is WCBB
    with pytest.raises(ValueError):
        get_league("nhl")
    assert (NBA.period_seconds, NBA.regulation_periods, NBA.n_qsec, NBA.n_results, NBA.n_fts) == (720, 4, 49, 18, 4)
    assert (CBB.period_seconds, CBB.regulation_periods, CBB.n_qsec, CBB.n_results, CBB.n_fts) == (1200, 2, 81, 19, 5)
    assert (WCBB.period_seconds, WCBB.regulation_periods, WCBB.n_qsec, WCBB.n_fouls, WCBB.n_fts) == (600, 4, 41, 7, 4)
    assert CBB.shot_clock == WCBB.shot_clock == 30 and NBA.shot_clock == 24
    assert CBB.dynamic_teams and not NBA.dynamic_teams and len(NBA.teams) == 30
    assert "fouls_def" in CBB.features and "fouls_def" not in NBA.features
    # the college vocabulary extends the NBA's: shared results keep their index
    assert CBB.results[:18] == NBA.results and CBB.rs["fts_0"] == 18
    assert CBB.results_for_type["foul"] == ["fts_0", "fts_1", "fts_2", "fts_3"]


def test_bonus_levels():
    men, women = CBB.bonus, WCBB.bonus
    assert [men.level(n) for n in (0, 6, 7, 9, 10, 15)] == [0, 0, 1, 1, 2, 2]
    assert [women.level(n) for n in (0, 4, 5, 9)] == [0, 0, 2, 2]  # no 1-and-1 for women
    assert Bonus(double_at=4).level(4) == 2


def test_college_checkpoint_roundtrip_keeps_league_and_teams():
    ts = TeamSeasonVocab(["Duke_2025", "Non-D1_2025"], CBB, ["Duke", "Non-D1"])
    back = TeamSeasonVocab.from_checkpoint({**ts.meta()})
    assert back.league is CBB and back.teams == ["Duke", "Non-D1"] and back.team("Non-D1") == 2 and back.team("X") == 0
    assert back.get("Duke", 2025) == 1 and back.get("Duke", 2026) == 0
    assert back.sizes()["off_team"] == 3 and back.sizes()["qsec"] == 81


def test_college_model_has_the_college_heads():
    ts = TeamSeasonVocab(["A_2025", "B_2025"], WCBB, ["A", "B"])
    m = EventTransformer(ModelConfig(ts.sizes(), d_model=16, n_layers=1, n_heads=2, league="wcbb"))
    assert m.res_head[-1].out_features == 19 and set(m.features) >= {"fouls_off", "fouls_def"}
    assert m.emb["qsec"].num_embeddings == 41
    feats = {f: torch.zeros(1, 3, dtype=torch.long) for f in m.features}
    assert m.encode(feats).shape == (1, 3, 16)
