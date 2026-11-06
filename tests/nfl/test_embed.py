import numpy as np
import pandas as pd
import torch

from runitback.nfl.embed import pool_sides, reduce
from runitback.wager.models.transformer.data import PBP_MISSING, pbp_columns, pbp_features
from runitback.wager.nfl import SCHEMA


def _raw(seasons):
    rng = np.random.default_rng(0)
    n = len(seasons)
    cols = {f"off_{j}": rng.normal(size=n) for j in range(3)} | {f"def_{j}": rng.normal(size=n) for j in range(3)}
    return pd.DataFrame({"game_id": [f"g{i}" for i in range(n)], "team": "A", "season": seasons, **cols})


def test_projection_is_fit_only_on_earlier_seasons():
    raw = _raw([2020] * 50 + [2021] * 50)
    later = raw.copy()
    later.loc[later.season == 2021, "off_0"] += 100.0  # the test season changes...
    a, _ = reduce(raw, 2021, dim=2)
    b, _ = reduce(later, 2021, dim=2)
    # ...but not the projection of the games before it
    pd.testing.assert_frame_equal(a[a.season == 2020], b[b.season == 2020])


def test_pool_sides_splits_offense_and_defense():
    h = torch.tensor([[[1.0], [3.0], [10.0], [0.0]]])           # 3 plays + padding
    is_home = torch.tensor([[1, 1, 0, 0]])
    valid = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    v = pool_sides(h, is_home, valid)
    assert v[0, 0].tolist() == [2.0, 10.0]   # home: own offense, own defense
    assert v[0, 1].tolist() == [10.0, 2.0]   # away: mirrored


def test_pbp_tokens_join_by_game_and_team(tmp_path):
    table = pd.DataFrame({"game_id": ["g1", "g1"], "team": ["A", "B"], "_pbp_0": [1.0, 2.0], "_pbp_1": [3.0, 4.0]})
    path = tmp_path / "pbp.parquet"
    table.to_parquet(path)
    games = pd.DataFrame({SCHEMA.game_id: ["g1", "g1", "g0"], SCHEMA.team: ["B", "A", "A"]})
    out = pbp_features(games, SCHEMA, str(path), 2)
    assert list(out.columns) == pbp_columns(2)
    assert out["_pbp_0"].tolist()[:2] == [2.0, 1.0]
    assert out[PBP_MISSING].tolist() == [0.0, 0.0, 1.0]
