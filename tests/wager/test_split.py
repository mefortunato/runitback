"""Training an ensemble's members separately and assembling them gives the in-process ensemble
(``runitback.wager.web.split``)."""
import numpy as np
import pandas as pd
import pytest
import torch

from runitback.wager.models.lightning import default_trainer
from runitback.wager.models.transformer import fit_transformer
from runitback.wager.web import split

from .conftest import make_games


@pytest.fixture(scope="module")
def games():
    return make_games(persistence=0.7)


def _trainer():
    return default_trainer(max_epochs=2, enable_progress_bar=False, enable_model_summary=False)


def _predictions(fitted, games):
    predictor = fitted["predictor"]
    upcoming = games[games.margin.isna()] if "margin" in games else games.tail(20)
    return predictor.predict(upcoming if len(upcoming) else games.tail(20))


def test_members_trained_apart_make_the_in_process_ensemble(games, tmp_path):
    together = fit_transformer(games, "toy", trainer=_trainer, ensemble=2, seed=0, print_metrics=False)
    ckpts = [split.fit_member(games, "toy", seed, tmp_path / f"s{seed}", max_epochs=2) for seed in range(2)]
    apart = split.assemble(games, "toy", ckpts, print_metrics=False)

    # the members' weights are the same (trained in another process of the same machine)
    for a, b in zip(together["model"].members, apart["model"].members):
        for (name, x), (_, y) in zip(a.state_dict().items(), b.state_dict().items()):
            assert torch.equal(x, y), name
    # and so is everything calibrated from them
    assert together["metrics"] == apart["metrics"]
    tail = games.tail(40)
    pd.testing.assert_frame_equal(
        together["predictor"].predict(tail).reset_index(drop=True),
        apart["predictor"].predict(tail).reset_index(drop=True),
        check_exact=True,
    )


def test_assemble_takes_the_checkpoints_as_the_members(games, tmp_path):
    ckpts = [split.fit_member(games, "toy", seed, tmp_path / f"s{seed}", max_epochs=1) for seed in range(2)]
    with pytest.raises(TypeError):
        split.assemble(games, "toy", ckpts, ensemble=3)
    one = split.assemble(games, "toy", ckpts[:1], print_metrics=False)
    assert not hasattr(one["model"], "members")
