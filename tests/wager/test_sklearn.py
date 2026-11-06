import numpy as np
import pandas as pd
import pytest
import sklearn.linear_model

from runitback.wager.core.metrics import antisymmetrize
from runitback.wager.core.predictor import load_predictor
from runitback.wager.models.sklearn import fit_sklearn
from runitback.wager.models.sklearn.features import latest_team_state, training_frame

from .conftest import TOY_SCHEMA


@pytest.fixture(scope="module")
def fitted(toy_games, tmp_path_factory):
    save_dir = tmp_path_factory.mktemp("models") / "sk"
    res = fit_sklearn(
        toy_games,
        "toy",
        model=sklearn.linear_model.Ridge(alpha=1.0),
        windows=[3],
        save_dir=str(save_dir),
        print_metrics=False,
    )
    return res, save_dir


def test_served_predictions_match_training_features(toy_games, fitted):
    """Regression: serving once fed the opponent's stats in a different order."""
    res, _ = fitted
    predictor = res["predictor"]
    week = toy_games[(toy_games.season == 2024) & (toy_games.week == 8)]
    predictor.update_state(toy_games[(toy_games.season < 2024) | (toy_games.week < 8)])
    served = predictor.predict(week[["team", "opp", *TOY_SCHEMA.pre_game_columns]])

    X = training_frame(toy_games, TOY_SCHEMA, (3,)).loc[week.index]
    expected = antisymmetrize(
        week, predictor.model.pipeline.predict(X), TOY_SCHEMA.game_id
    )
    np.testing.assert_allclose(served.pred_margin, expected, atol=1e-9)


def test_predicts_known_strengths(fitted):
    res, _ = fitted
    assert res["metrics"]["test"]["accuracy"] > 0.65
    assert res["metrics"]["test"]["pearson"] > 0.5


def test_predictions_are_antisymmetric_and_batch_independent(toy_games, fitted):
    predictor = fitted[0]["predictor"]
    predictor.update_state(toy_games)
    pair = pd.DataFrame({"team": ["T00"], "opp": ["T01"], "location": [1]})
    swapped = pd.DataFrame({"team": ["T01"], "opp": ["T00"], "location": [-1]})
    alone = predictor.predict(pair)
    others = pd.DataFrame({"team": ["T02", "T04"], "opp": ["T03", "T05"]})
    batch = predictor.predict(pd.concat([pair, others], ignore_index=True))

    assert alone.pred_margin[0] == pytest.approx(
        -predictor.predict(swapped).pred_margin[0]
    )
    assert np.isfinite(alone.win_prob[0])
    assert alone.win_prob[0] == pytest.approx(batch.win_prob[0])


def test_unknown_team_is_an_error(fitted):
    with pytest.raises(ValueError, match="no state"):
        fitted[0]["predictor"].predict(pd.DataFrame({"team": ["XXX"], "opp": ["T01"]}))


def test_train_seasons_are_honored(toy_games):
    res = fit_sklearn(
        toy_games,
        "toy",
        model=sklearn.linear_model.Ridge(),
        train_seasons=[2022],
        print_metrics=False,
    )
    assert res["split"].train == [2022]
    assert res["metrics"]["train"]["n"] < (toy_games.season == 2022).sum()


def test_latest_state_includes_most_recent_game(toy_games):
    state = latest_team_state(toy_games, TOY_SCHEMA)
    season = toy_games[(toy_games.season == 2024) & toy_games.margin.notna()]
    team = season[season.team == "T00"]
    assert state.loc["T00", "n_games"] == len(team)
    assert state.loc["T00", "points_avg"] == pytest.approx(team.points.mean())


def test_saved_model_round_trips_and_refuses_overwrite(toy_games, fitted):
    res, save_dir = fitted
    loaded = load_predictor(str(save_dir))
    for predictor in (loaded, res["predictor"]):
        predictor.update_state(toy_games)
    matchups = pd.DataFrame({"team": ["T00", "T03"], "opp": ["T01", "T02"]})
    pd.testing.assert_frame_equal(
        loaded.predict(matchups), res["predictor"].predict(matchups)
    )
    with pytest.raises(FileExistsError):
        fit_sklearn(toy_games, "toy", save_dir=str(save_dir), print_metrics=False)
