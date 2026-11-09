import tempfile

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.loader import DataLoader

from runitback.wager.core.predictor import load_predictor
from runitback.wager.core.schema import played
from runitback.wager.core.splits import split_seasons
from runitback.wager.models.gnn import default_model, fit_gnn
from runitback.wager.models.gnn.datamodule import GraphDataModule

from .conftest import TOY_SCHEMA


@pytest.fixture(scope="module")
def datamodule(toy_games):
    games = played(toy_games, TOY_SCHEMA)
    split = split_seasons(games.season.unique())
    return GraphDataModule(games, TOY_SCHEMA, split, batch_size=8)


def quick_trainer(max_epochs=15):
    return pl.Trainer(
        max_epochs=max_epochs,
        default_root_dir=tempfile.mkdtemp(),  # checkpoints out of the repo
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min")],
    )


@pytest.fixture(scope="module")
def fitted(toy_games, tmp_path_factory):
    torch.manual_seed(0)
    save_dir = tmp_path_factory.mktemp("models") / "gnn"
    res = fit_gnn(
        toy_games,
        "toy",
        model=default_model("toy"),
        trainer=quick_trainer(),
        save_dir=str(save_dir),
        print_metrics=False,
    )
    return res, save_dir


def test_every_target_game_is_trained_on_each_epoch(datamodule):
    """Regression: the loss once covered only the batch's latest date."""
    graph, ds = datamodule.graph, datamodule.datasets["train"]
    in_train = torch.isin(graph.season, torch.tensor(datamodule.split.train))
    expected = int((graph.valid & in_train).sum())

    seen = 0
    for batch in DataLoader(ds, batch_size=8, shuffle=True):
        seen += int(batch.target.sum())
        # each sample's own date is its target date, wherever it sits in a batch
        for sample in batch.to_data_list():
            assert sample.target.any()
            assert (sample.date[sample.target] == sample.date.max()).all()
            assert not (sample.on_date & ~(sample.date == sample.date.max())).any()
    assert seen == expected


def test_early_season_games_with_history_are_targets(datamodule):
    """Regression: a date was dropped if any team on it had no earlier game."""
    week2 = [s for s in datamodule.datasets["train"].snapshots if s[1] == 2]
    assert len(week2) == len(datamodule.split.train)


def test_target_scaling_comes_from_training_data(fitted, toy_games):
    model = fitted[0]["model"]
    train = toy_games[toy_games.season.isin(fitted[0]["split"].train)]
    assert model.target_std.item() == pytest.approx(train.margin.std(), rel=1e-5)


def test_serving_does_not_touch_batchnorm_stats(fitted, toy_games):
    """Regression: embeddings were computed in train mode."""
    predictor = fitted[0]["predictor"]
    module = predictor.model.module
    before = module.encoder.edge_norm.running_mean.clone()
    predictor.update_state(toy_games)
    assert not module.training
    assert torch.equal(before, module.encoder.edge_norm.running_mean)
    assert not predictor.model.embeddings.requires_grad


def test_gnn_learns_and_serves(fitted, toy_games):
    res, save_dir = fitted
    assert res["metrics"]["test"]["accuracy"] > 0.6

    loaded = load_predictor(str(save_dir))
    loaded.update_state(toy_games)
    matchups = pd.DataFrame(
        {"team": ["T00", "T01"], "opp": ["T01", "T00"], "location": [1, -1]}
    )
    out = loaded.predict(matchups)
    assert out.pred_margin[0] == pytest.approx(-out.pred_margin[1])
    assert np.isfinite(out.win_prob).all()


def test_mismatched_model_dims_are_reported(toy_games):
    model = default_model("toy", context=[])
    with pytest.raises(ValueError, match="context_dim"):
        fit_gnn(toy_games, "toy", model=model, trainer=quick_trainer(1))


def test_seeded_fits_are_reproducible(toy_games):
    def fit():
        res = fit_gnn(
            toy_games,
            "toy",
            model=default_model("toy"),
            trainer=quick_trainer(2),
            print_metrics=False,
        )
        return res["metrics"]["val"]

    assert fit() == fit()


def test_refresh_skips_teams_unseen_in_training(fitted, toy_games):
    predictor = fitted[0]["predictor"]
    games = toy_games.copy()
    newcomer = (games.season == 2024) & (games.week == 1) & (games.team == "T00")
    games.loc[newcomer, "team"] = "NEW"
    games.loc[
        (games.season == 2024) & (games.week == 1) & (games.opp == "T00"), "opp"
    ] = "NEW"
    predictor.update_state(games)
    assert "NEW" not in predictor.model.teams and "T00" in predictor.model.teams
    with pytest.raises(ValueError, match="no state"):
        predictor.predict(pd.DataFrame({"team": ["NEW"], "opp": ["T01"]}))


def test_team_embeddings_are_the_current_teams_node_states(fitted, toy_games):
    loaded = load_predictor(str(fitted[1]))
    loaded.update_state(toy_games)
    embeddings = loaded.model.team_embeddings()
    assert list(embeddings.index) == loaded.model.teams
    assert np.isfinite(embeddings.to_numpy()).all()
