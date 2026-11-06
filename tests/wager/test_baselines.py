import math

import numpy as np
import pandas as pd
import pytest

from runitback.wager.models.baselines import (
    EloModel,
    EloParams,
    fit_elo,
    fit_home,
    fit_point_diff,
)

from .conftest import TOY_SCHEMA


def two_games():
    """A wins by 7 at home in 2023, then the two meet again in 2024."""
    rows = []
    for season, game, margin in [(2023, "g1", 7), (2024, "g2", np.nan)]:
        rows += [
            {"season": season, "week": 1, "game_id": game, "team": "A", "opp": "B",
             "location": 1, "margin": margin},
            {"season": season, "week": 1, "game_id": game, "team": "B", "opp": "A",
             "location": -1, "margin": -margin},
        ]  # fmt: skip
    return pd.DataFrame(rows)


def test_elo_update_matches_hand_computation():
    model = EloModel(TOY_SCHEMA)
    diffs = model.replay(two_games())
    p = EloParams()
    expected = 1 / (1 + 10 ** (-p.home / 400))
    multiplier = math.log(8) * 2.2 / (p.home * 0.001 + 2.2)
    shift = p.k * multiplier * (1 - expected)
    assert shift == pytest.approx(16.46, abs=0.01)  # 20 * 2.0198 * 0.4075
    assert model.ratings["A"] == pytest.approx(p.mean + shift)
    assert model.ratings["B"] == pytest.approx(p.mean - shift)
    assert list(diffs) == [p.home, -p.home]  # pre-game, both perspectives


def test_elo_regresses_ratings_in_a_new_season():
    model = EloModel(TOY_SCHEMA)
    model.replay(two_games())
    p = EloParams()
    gap = model.ratings["A"] - model.ratings["B"]
    same = pd.DataFrame(
        {"team": ["A"], "opp": ["B"], "location": [0], "season": [2023]}
    )
    new = same.assign(season=2024)
    assert model.predict_margin(same)[0] == pytest.approx(gap * p.points_per_elo)
    assert model.predict_margin(new)[0] == pytest.approx(
        gap * (1 - p.regress) * p.points_per_elo
    )


@pytest.mark.parametrize("fit", [fit_home, fit_point_diff, fit_elo])
def test_baselines_serve_like_any_model(fit, toy_games):
    res = fit(toy_games, "toy", print_metrics=False)
    predictor = res["predictor"]
    out = predictor.predict(
        pd.DataFrame(
            {"team": ["T00", "T01"], "opp": ["T01", "T00"], "location": [1, -1]}
        )
    )
    assert out.pred_margin[0] == pytest.approx(-out.pred_margin[1])
    assert np.isfinite(out.win_prob).all()
    assert set(res["metrics"]) == {"train", "val", "test"}


def test_home_field_baseline(toy_games):
    res = fit_home(toy_games, "toy", print_metrics=False)
    home = res["predictor"].model.home_advantage
    train = toy_games[toy_games.season.isin(res["split"].train)]
    assert home == pytest.approx(train[train.location == 1].margin.mean())
    neutral = pd.DataFrame({"team": ["T00"], "opp": ["T01"], "location": [0]})
    assert res["predictor"].predict(neutral).pred_margin[0] == 0


def test_rating_baselines_beat_home_field(toy_games):
    accuracy = {
        fit.__name__: fit(toy_games, "toy", print_metrics=False)["metrics"]["test"][
            "accuracy"
        ]
        for fit in (fit_home, fit_point_diff, fit_elo)
    }
    assert accuracy["fit_point_diff"] > accuracy["fit_home"]
    assert accuracy["fit_elo"] > accuracy["fit_home"]
