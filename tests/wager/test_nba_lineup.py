"""N5: the NBA's player-value lineup layout on its Sport, and the per-team-game tables (foundation
embeddings, lineup values: string ESPN ids) joining a games table whose ids were read back from CSV
as integers."""
import numpy as np
import pandas as pd

from runitback.wager.core.sport import get_sport
from runitback.wager.models.transformer.data import (
    additive_spec,
    game_keys,
    lineup_columns,
    lineup_features,
    pbp_features,
)
from runitback.wager.nba.lineup import LINEUP


def test_game_keys():
    assert game_keys(pd.Series([401585601, 7])).tolist() == ["401585601", "7"]
    f = game_keys(pd.Series([401585601.0, np.nan]))
    assert f.iloc[0] == "401585601" and pd.isna(f.iloc[1])
    s = pd.Series(["2021_01_KC_CLE", None], dtype=object)
    assert game_keys(s) is s          # the NFL's string ids are untouched


def test_nba_sport_has_the_lineup_layout():
    sport = get_sport("nba")
    assert sport.lineup is LINEUP
    assert len(lineup_columns("pvc", "nba")) == 14 + 1
    spec = additive_spec(3, "pvc", "nba")
    # only the six changes vs the last 10 rotations enter, offense up and defense down on the total
    assert [i for i, _ in spec["additive_pairs"]] == [3 + 8 + g for g in range(6)]
    assert spec["additive_total_sign"] == [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]


def _games():
    # as pd.read_csv returns data/nba/games.csv: integer game ids
    return pd.DataFrame({"game_id": [401585601, 401585601, 401585602], "team": ["BOS", "NYK", "BOS"],
                         "opp_team": ["NYK", "BOS", "LAL"]})


def test_embeddings_and_lineup_values_join_integer_game_ids(tmp_path):
    schema = get_sport("nba").schema
    games = _games()
    emb = pd.DataFrame({"game_id": ["401585601", "401585601"], "team": ["BOS", "NYK"], "season": 2024,
                        **{f"_pbp_{i}": [1.0, 2.0] for i in range(4)}})
    emb.to_parquet(tmp_path / "pbp.parquet")
    out = pbp_features(games, schema, str(tmp_path / "pbp.parquet"), 4)
    assert out["_pbp_0"].tolist()[:2] == [1.0, 2.0] and np.isnan(out["_pbp_0"].iloc[2])
    assert out["_pbp_missing"].tolist() == [0.0, 0.0, 1.0]

    pv = pd.DataFrame({"game_id": ["401585601", "401585601", "401585602"], "team": ["BOS", "NYK", "BOS"],
                       **{f"_pv_{i}": np.float32([0.5, -0.5, 1.5]) for i in range(14)}})
    pv.to_parquet(tmp_path / "pv.parquet")
    lu = lineup_features(games, schema, str(tmp_path / "pv.parquet"), "pvc", "nba")
    assert lu["_pv_8"].tolist() == [0.5, -0.5, 1.5]
    assert lu["_opp_pv_8"].tolist()[:2] == [-0.5, 0.5] and lu["_opp_lu_missing"].tolist() == [0.0, 0.0, 1.0]
    assert lu["_lu_missing"].tolist() == [0.0, 0.0, 0.0]


def test_player_value_terms_follow_the_environment(tmp_path):
    import importlib, sys
    import runitback.wager.nba as nba
    try:
        assert "lineup" not in nba.SPORT.transformer
        import os
        os.environ["RUNITBACK_NBA_PLAYER_VALUES"] = str(tmp_path) + "/"
        mod = importlib.reload(nba)
        t = mod.SPORT.transformer
        assert t["lineup"] == f"{tmp_path}/pv.parquet" and t["lineup_set"] == "pvc" and t["ensemble"] == 4
    finally:
        os.environ.pop("RUNITBACK_NBA_PLAYER_VALUES", None)
        importlib.reload(nba)
