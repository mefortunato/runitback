import numpy as np
import pandas as pd
import pytest
import torch

from runitback.core.embedding import pool_by_possession, team_vectors
from runitback.nfl.data.ratings import compute_ratings
from runitback.nfl.embed import KEY_COLUMNS, game_embeddings, trained_on
from runitback.nfl.engine.rules import new_game

from .conftest import toy_games


def test_pool_by_possession_averages_each_sides_plays():
    h = torch.tensor([[[1.0], [3.0], [10.0], [99.0]]])  # last position is padding
    valid = torch.tensor([[1, 1, 1, 0]])
    home_ball = torch.tensor([[1, 1, 0, 1]])
    home_off, away_off = pool_by_possession(h, valid, home_ball)
    assert home_off.item() == 2.0 and away_off.item() == 10.0
    home, away = team_vectors(home_off, away_off)
    assert home.tolist() == [[2.0, 10.0]] and away.tolist() == [[10.0, 2.0]]


def test_trained_on_uses_last_training_game_then_validation_split():
    games = pd.DataFrame({"season": [2023, 2024, 2024, 2025], "week": [5, 9, 10, 1]})
    assert trained_on(games, {"train_last": [2024, 9]}).tolist() == [True, True, False, False]
    assert trained_on(games, {"val_season": 2024, "val_week": 10}).tolist() == [True, True, False, False]
    assert trained_on(games, {"val_season": None}).all()


def _play_rows(game_id, log):
    rows = []
    for i, (s, o) in enumerate(log):
        rows.append(dict(game_id=game_id, idx=i, qtr=s.qtr, qsec=s.qsec, phase=s.phase, pos=s.pos,
                         down=s.down, ytg=s.ytg, yl=s.yl, score_h=s.score[0], score_a=s.score[1],
                         to_h=s.timeouts[0], to_a=s.timeouts[1], play_type=o.play_type,
                         result=o.result, yards=o.yards, elapsed=o.elapsed))
    return rows


@pytest.fixture(scope="module")
def processed(tiny_sim, tmp_path_factory):
    """A processed-data directory of simulated (so legal) games, and a checkpoint."""
    games = toy_games()
    games["game_type"] = "REG"
    plays = []
    for g in games.itertuples():
        _, logs = tiny_sim.rollout(new_game(g.home, g.away, int(g.season), opening_kicker=0), 1,
                                   seed=0, keep_plays=1)
        plays += _play_rows(g.game_id, logs[0])
    root = tmp_path_factory.mktemp("nfl")
    data = root / "processed"
    data.mkdir()
    games.to_parquet(data / "games.parquet")
    pd.DataFrame(plays).to_parquet(data / "plays.parquet")
    compute_ratings(toy_games()).to_parquet(data / "ratings.parquet")
    ckpt = root / "model.pt"
    torch.save({"state_dict": tiny_sim.model.state_dict(), "config": tiny_sim.model.cfg.to_dict(),
                "team_seasons": tiny_sim.ts.keys, "val_season": 2025, "val_week": 1,
                "train_last": [2024, 2]}, ckpt)
    return data, ckpt


def test_game_embeddings_one_row_per_team_per_game(processed, tiny_sim):
    data, ckpt = processed
    df = game_embeddings(ckpt, data)
    d = tiny_sim.model.cfg.d_model
    emb = [f"emb_{i}" for i in range(2 * d)]
    assert list(df.columns) == KEY_COLUMNS + emb
    assert len(df) == 6 and df.groupby("game_id").size().eq(2).all()
    assert np.isfinite(df[emb].to_numpy()).all()
    # in_sample follows the checkpoint's training range (through 2024 week 2)
    assert df.set_index("game_id").in_sample.groupby(level=0).first().to_dict() == \
        {"g1": True, "g2": True, "g3": False}
    assert df.checkpoint.nunique() == 1 and len(df.checkpoint.iloc[0]) == 64
    # a team's defense half is its opponent's offense half
    for _, g in df.groupby("game_id"):
        home, away = g[g.is_home].iloc[0], g[~g.is_home].iloc[0]
        assert home.team == away.opp and away.team == home.opp
        h, a = home[emb].to_numpy(float), away[emb].to_numpy(float)
        assert np.allclose(h[:d], a[d:]) and np.allclose(h[d:], a[:d])


def test_embeddings_do_not_depend_on_batching(processed, tiny_sim):
    data, ckpt = processed
    together = game_embeddings(ckpt, data).set_index(["game_id", "team"])
    alone = game_embeddings(ckpt, data, seasons=[2025]).set_index(["game_id", "team"])
    emb = [c for c in alone.columns if c.startswith("emb_")]
    assert np.allclose(alone[emb].to_numpy(float), together.loc[alone.index, emb].to_numpy(float), atol=1e-5)
