import dataclasses

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
import torch

from runitback.wager import backtest as bt
from runitback.wager.core.sport import register_sport
from runitback.wager.models.baselines import fit_elo
from runitback.wager.models.transformer import default_model, fit_transformer
from runitback.wager.models.transformer.data import sequence
from runitback.wager.models.transformer.fit import fit_settings, transformer_config

from .football import DRAWS_SCHEMA, football_games

ELO = {"k": 15.0, "home": 80.0, "regress": 0.05, "points_per_elo": 1 / 200, "mov": "football", "new_team": 1405.0}  # fmt: skip
TRANSFORMER = {"history": 8, "d_model": 16, "nhead": 2, "num_layers": 1, "distribution": True,
               "min_history": 0, "drop_older_than": 2}  # fmt: skip


def quick_trainer():
    return pl.Trainer(max_epochs=1, accelerator="cpu", logger=False, enable_progress_bar=False,
                      enable_checkpointing=False, enable_model_summary=False,
                      default_root_dir=__import__("tempfile").mkdtemp())  # fmt: skip


@pytest.fixture(scope="module")
def sports(toy_sport):
    """The football toy league as a sport without and with cold start."""
    base = dataclasses.replace(
        toy_sport, schema=DRAWS_SCHEMA, elo=ELO, transformer=TRANSFORMER
    )
    cold = register_sport(dataclasses.replace(base, name="toy_cold", cold_start=True))
    warm = register_sport(dataclasses.replace(base, name="toy_warm", cold_start=False))
    return warm, cold


@pytest.fixture(scope="module")
def games():
    """A club ('NEW') that appears only from the last season: no state when it first plays."""
    games = football_games()
    late = games["season"] >= 2023
    for column in ("team", "opp"):
        games.loc[late & (games[column] == "T11"), column] = "NEW"
    games["game_id"] = games["game_id"].where(
        ~late, games["game_id"].str.replace("_11_", "_99_")
    )
    return games


SPLIT = {
    "train_seasons": [2021],
    "val_seasons": [2022],
    "test_seasons": [],
}  # as walk_forward's


def matchup(team="NEW", opp="T00"):
    return pd.DataFrame(
        {"team": [team], "opp": [opp], "location": [1], "season": [2023]}
    )


def test_elo_refuses_an_unknown_club_unless_cold_start_is_on(games, sports):
    warm, cold = sports
    history = games[games.season < 2023]
    unseen = games[
        (games.season == 2023) & ((games.team == "NEW") | (games.opp == "NEW"))
    ]
    assert len(unseen)
    with pytest.raises(ValueError, match="no state for teams"):
        fit_elo(history, warm.name, print_metrics=False, **SPLIT)["predictor"].predict(
            matchup("T00", "NEW")
        )
    model = fit_elo(history, cold.name, print_metrics=False, **SPLIT)["predictor"]
    assert model.model.cold_start and "NEW" not in model.model.teams
    # a first rating of 1405 against the established club's, home edge 80
    out = model.predict(matchup("T00", "NEW"))
    rating = model.model.ratings["T00"]
    assert (
        out["pred_margin"][0]
        > (rating - 1405.0 + 80.0) * ELO["points_per_elo"] * 0.9 - 0.2
    )


def test_walk_forward_predicts_every_game_with_cold_start(games, sports):
    warm, cold = sports
    results = {
        sport.name: bt.walk_forward(
            games, sport.name, bt.baseline_fitter(sport.name, "elo"), [2023]
        )
        for sport in sports
    }
    played = games[(games.season == 2023) & games.margin.notna()]
    n = played.game_id.nunique()
    assert len(results[cold.name]) == n and results[cold.name].game_id.is_unique
    # without cold start a club is left out until it has played: its first game of the season
    involved = played[(played.team == "NEW") | (played.opp == "NEW")]
    skipped = involved[involved.week == involved.week.min()].game_id.nunique()
    assert skipped == 1 and len(results[warm.name]) == n - skipped
    assert (
        results[cold.name].win_prob.notna().all()
        and results[cold.name].draw_prob.notna().all()
    )
    # a game both clubs have state for is predicted the same either way
    both = results[cold.name].merge(
        results[warm.name], on="game_id", suffixes=("_cold", "_warm")
    )
    np.testing.assert_allclose(both.win_prob_cold, both.win_prob_warm)


def test_other_families_ignore_cold_start(games, sports):
    from runitback.wager.models.baselines import fit_home

    fit = fit_home(
        games[games.season < 2023], sports[1].name, print_metrics=False, **SPLIT
    )
    assert not fit[
        "predictor"
    ].model.cold_start  # the home baseline has no team state to start from


def test_drop_older_than_pads_games_seasons_back():
    tokens = torch.arange(6.0).reshape(3, 2)
    seasons = torch.tensor([2020.0, 2022.0, 2023.0])
    index = torch.tensor([[2, 1, 0]])
    season = torch.tensor([2023.0])
    kept = sequence(tokens, seasons, index, season)
    assert not kept["pad"].any()
    dropped = sequence(tokens, seasons, index, season, drop_older_than=2)
    assert dropped["pad"].tolist() == [[False, False, True]]  # 2020 is 3 seasons back
    assert (
        dropped["tokens"][0, 2].tolist() == [0.0, 0.0]
        and dropped["seasons_ago"][0, 2] == 0
    )
    assert sequence(tokens, seasons, index, season, drop_older_than=3)["pad"].sum() == 0


def test_sport_settings_reach_the_fit_and_not_the_encoder(sports):
    _warm, cold = sports
    assert (
        fit_settings(cold.name)["min_history"] == 0
        and fit_settings(cold.name)["drop_older_than"] == 2
    )
    config = transformer_config(cold.name)
    assert (
        "min_history" not in config["encoder"]
        and "drop_older_than" not in config["encoder"]
    )


@pytest.fixture(scope="module")
def fitted(games, sports):
    history = games[games.season < 2023]
    return {
        sport.name: fit_transformer(
            history, sport.name, trainer=quick_trainer(), print_metrics=False, **SPLIT
        )
        for sport in sports
    }


def test_the_transformer_is_fit_with_the_sports_cold_start_settings(fitted):
    cold = fitted["toy_cold"]["predictor"].model
    assert cold.features.drop_older_than == 2 and cold.cold_start
    assert not fitted["toy_warm"]["predictor"].model.cold_start


def test_a_transformer_predicts_a_club_it_has_no_state_for(fitted):
    cold = fitted["toy_cold"]["predictor"]
    out = cold.predict(
        pd.concat([matchup("T00", "NEW"), matchup("NEW", "T00"), matchup("T00", "T01")])
    )
    assert np.isfinite(out[["pred_margin", "win_prob", "draw_prob"]].to_numpy()).all()
    assert (out["win_prob"] + out["draw_prob"] < 1).all()
    # the same club on the other side: the antisymmetric margin, to the model's tolerance
    assert out["pred_margin"][0] == pytest.approx(-out["pred_margin"][1], abs=0.2)
    with pytest.raises(ValueError, match="no state for teams"):
        fitted["toy_warm"]["predictor"].predict(matchup("T00", "NEW"))


def test_embeddings_leave_out_the_empty_history_row(fitted):
    model = fitted["toy_cold"]["predictor"].model
    embeddings = model.team_embeddings()
    assert len(embeddings) == len(model.teams) == len(model.team_index)
    assert len(model.history) == len(model.teams) + 1  # the last row is padding only
    assert (model.history[-1] == -1).all()


def test_a_team_with_no_history_does_not_make_the_encoder_nan(games):
    model = default_model("toy_cold", history=6, season_decay=True)
    batch = sequence(
        torch.zeros(1, model.encoder.token_dim),
        torch.zeros(1),
        torch.full((2, 6), -1),
        torch.tensor([2023.0, 2023.0]),
    )
    out = model.encoder(**batch)
    assert torch.isfinite(out).all()


def test_a_transformers_score_distribution_takes_the_sports_ranges(sports):
    model = default_model(
        "toy_cold", distribution=True, margin_range=(-10, 10), total_range=(0, 12)
    )
    assert model.hparams["margin_range"] == (-10, 10)
    assert (
        len(model.margin_dist.values) == 21
    )  # goal differences, not the NFL's -70..70
    assert len(default_model("toy_cold", distribution=True).margin_dist.values) == 141
