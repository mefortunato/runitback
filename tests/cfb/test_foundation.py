"""College foundation folds through the sport-agnostic fold code (runitback.core.foundation)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from runitback.cfb.data import preprocess as P
from runitback.cfb.foundation import CFB, TRAIN, CFBGenerator
from runitback.core import foundation as core

from .test_preprocess import _raw


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cfbfold")
    P.preprocess([2005, 2010, 2021, 2023], _raw(tmp), tmp / "processed")
    return tmp / "processed"


def test_cfb_is_a_fold_generator(data):
    assert isinstance(CFB, core.Generator) and CFB.sport == "cfb"
    assert TRAIN["use_lineups"] is False and TRAIN["min_season"] == 2004
    assert CFB.current_season(data) == 2023
    assert CFB.played_games(data) == {"401551789", "401525564", "401282614", "310102483", "252460344"}
    assert core.default_seasons(CFB, data) == list(range(2018, 2024))


def test_fold_trains_before_its_season_and_embeds_every_game(data, tmp_path):
    gen = CFBGenerator(epochs=1, d_model=16, n_layers=1, n_heads=2)
    root = tmp_path / "folds"
    done = core.prepare(gen, root, data, [2023], dim=4)
    assert done == {2023: "trained, embedded"}
    d = core.fold_dir(root, 2023)
    for f in core.fold_files(4):
        assert (d / f).exists(), f
    ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
    assert ck["val_season"] == 2022 and ck["config"]["n_players"] == 0 and ck["config"]["n_cont"] == 5
    assert not any(k.endswith("_2023") for k in ck["team_seasons"])  # never saw the fold's season
    assert ck["team_seasons"] and all(k.split("_")[0].isdigit() for k in ck["team_seasons"])
    emb = pd.read_parquet(d / "pbp_4.parquet")
    assert len(emb) == 10 and set(emb.team) >= {"52", "87"}
    assert list(emb.columns) == core.KEYS + [f"{core.PREFIX}{j}" for j in range(4)]
    proj = core.load_projection(d, 4)
    # the projection is fit on games before 2023 only: standardized there, those games average 0
    raw = gen.embed(d / "model.pt", data)
    before = raw[raw.season < 2023].filter(regex=core.RAW_COLUMNS).to_numpy(np.float64)
    np.testing.assert_allclose(proj["mean"], before.mean(0), rtol=1e-6, atol=1e-6)
    # a second prepare only adds new games (none here) with the stored projection
    assert core.prepare(gen, root, data, [2023], dim=4) == {2023: "0 new games"}
