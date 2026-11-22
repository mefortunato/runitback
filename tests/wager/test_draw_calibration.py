import dataclasses
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from runitback.wager.core.evaluation import calibrate, calibrate_and_evaluate
from runitback.wager.core.metrics import (
    fit_outcome_calibration,
    outcome_class,
    outcome_probabilities,
    win_probability,
)
from runitback.wager.core.predictor import MatchupModel, Predictor
from runitback.wager.core.splits import split_seasons
from runitback.wager.core.sport import register_sport
from runitback.wager.models.baselines import fit_elo, fit_home

from .conftest import TOY_SCHEMA
from .football import DRAWS_SCHEMA, football_games


@pytest.fixture(scope="module")
def draws_sport(toy_sport):
    return register_sport(
        dataclasses.replace(toy_sport, name="toy_draws", schema=DRAWS_SCHEMA)
    )


@pytest.fixture(scope="module")
def games():
    return football_games()


def test_no_cut_is_exactly_the_two_outcome_probability():
    margin = np.array([-2.0, -0.3, 0.0, 0.8, 3.0])
    win, draw = outcome_probabilities(margin, 1.7)
    assert np.array_equal(win, win_probability(margin, 1.7))
    assert (draw == 0).all()
    assert np.array_equal(outcome_probabilities(margin, 1.7, 0.0)[0], win)


def test_ordered_probit_probabilities_are_a_distribution():
    margin = np.linspace(-3, 3, 13)
    win, draw = outcome_probabilities(margin, 1.5, 0.6)
    loss = outcome_probabilities(-margin, 1.5, 0.6)[0]
    np.testing.assert_allclose(win + draw + loss, 1.0)
    assert ((win >= 0) & (draw >= 0) & (loss >= 0)).all()
    # the draw is likeliest between the teams, and a wider cut gives more of it
    assert draw.argmax() == len(margin) // 2
    assert outcome_probabilities(margin, 1.5, 1.0)[1].sum() > draw.sum()


def test_fitted_cut_recovers_the_draw_rate():
    rng = np.random.default_rng(0)
    n = 40_000
    home, away = rng.uniform(0.8, 2.2, n), rng.uniform(0.6, 1.8, n)
    y = rng.poisson(home) - rng.poisson(away)  # goal differences: a Skellam
    sigma, cut = fit_outcome_calibration(y, home - away)
    win, draw = outcome_probabilities(home - away, sigma, cut)
    actual = float(np.mean(y == 0))
    assert 0.2 < actual < 0.3
    assert draw.mean() == pytest.approx(actual, abs=0.02)
    assert win.mean() == pytest.approx(np.mean(y > 0), abs=0.02)
    assert sigma > 0 and cut > 0


def test_no_draws_in_the_data_fits_no_cut():
    rng = np.random.default_rng(1)
    margin = rng.normal(0, 2, 500)
    y = np.where(margin + rng.normal(0, 1, 500) > 0, 1.0, -1.0)
    _, cut = fit_outcome_calibration(y, margin)
    assert cut == pytest.approx(0.0, abs=0.05)


def test_calibrate_scores_every_game_when_there_are_draws(games):
    rows = games.dropna(subset=["margin"]).reset_index(drop=True)
    split = split_seasons(rows.season.unique())
    result = calibrate(rows, rows["margin"] * 0.0, DRAWS_SCHEMA, split)
    assert result.draw_cut and result.draw_cut > 0
    metrics = result.metrics["val"]
    assert {"rps", "brier3", "log_loss3", "accuracy3"} <= set(metrics)
    assert metrics["n"] == (rows.season.isin(split.val)).sum()  # draws included
    # a sport with two outcomes is unchanged and has no cut
    two_way = calibrate(rows, rows["margin"] * 0.0, TOY_SCHEMA, split)
    assert two_way.draw_cut is None and "brier" in two_way.metrics["val"]
    # the three-value tuple a two-outcome caller unpacks, and a refusal for a sport with draws
    assert (
        len(calibrate_and_evaluate(rows, rows["margin"] * 0.0, TOY_SCHEMA, split)) == 3
    )
    with pytest.raises(ValueError, match="draw cut"):
        calibrate_and_evaluate(rows, rows["margin"] * 0.0, DRAWS_SCHEMA, split)


def test_predictor_gives_win_and_draw_probabilities(games, draws_sport):
    fit = fit_home(games, "toy_draws", print_metrics=False)
    predictor = fit["predictor"]
    assert predictor.draw_cut and predictor.draw_cut > 0
    matchup = pd.DataFrame(
        {"team": ["T00", "T01"], "opp": ["T01", "T00"], "location": [1, -1]}
    )
    out = predictor.predict(matchup)
    assert {"pred_margin", "win_prob", "draw_prob"} <= set(out)
    assert "cover_prob" not in out  # the handicap is out of scope for draws
    loss = 1 - out["win_prob"] - out["draw_prob"]
    assert ((out["draw_prob"] > 0.1) & (loss > 0)).all()
    # the home side is likelier to win than the same side away
    assert out.loc[0, "win_prob"] > out.loc[1, "win_prob"]
    # calibrated on its validation season: the model's draw probability is that season's draw rate
    played = games.dropna(subset=["margin"]).drop_duplicates("game_id")
    val = split_seasons(played.season.unique()).val
    rate = float(np.mean(played[played.season.isin(val)]["margin"] == 0))
    average = predictor.predict(played.iloc[:400].assign(location=0))[
        "draw_prob"
    ].mean()
    assert average == pytest.approx(rate, abs=0.03)


def test_two_outcome_predictors_are_unchanged(toy_games):
    predictor = fit_elo(toy_games, "toy", print_metrics=False)["predictor"]
    assert predictor.draw_cut is None
    out = predictor.predict(
        pd.DataFrame({"team": ["T00"], "opp": ["T01"], "location": [1]})
    )
    assert (
        "draw_prob" not in out and "cover_prob" not in out
    )  # no spread in the matchup
    assert out.loc[0, "win_prob"] == win_probability(
        out.loc[0, "pred_margin"], predictor.sigma
    )


def test_a_model_saved_before_draws_still_predicts(games, draws_sport):
    predictor = fit_elo(games, "toy_draws", print_metrics=False)["predictor"]
    del predictor.__dict__["draw_cut"]  # as unpickled from before the attribute existed
    out = predictor.predict(
        pd.DataFrame({"team": ["T00"], "opp": ["T01"], "location": [1]})
    )
    assert out.loc[0, "draw_prob"] == 0  # no cut: no draws, as such a model was fit


class PmfModel(MatchupModel):
    """A model that predicts a fixed score distribution over margins -3..3."""

    schema = DRAWS_SCHEMA
    teams: ClassVar[list] = ["A", "B"]
    values = np.arange(-3.0, 4.0)

    def update_state(self, games):
        pass

    def predict_margin(self, matchups):
        return np.zeros(len(matchups))

    def predict_outputs(self, matchups):
        pmf = np.tile([0.05, 0.1, 0.15, 0.3, 0.2, 0.15, 0.05], (len(matchups), 1))
        return {
            "margin": self.predict_margin(matchups),
            "margin_pmf": pmf,
            "margin_values": self.values,
        }


def test_distribution_models_give_unconditional_win_and_draw_probabilities():
    predictor = Predictor(PmfModel(), sigma=1.0, sport="toy_draws")
    out = predictor.predict(
        pd.DataFrame({"team": ["A"], "opp": ["B"], "location": [0]})
    )
    # the opponent's distribution is mirrored onto this team's side and averaged with it
    pmf = np.array([0.05, 0.1, 0.15, 0.3, 0.2, 0.15, 0.05])
    pmf = (pmf + pmf[::-1]) / 2
    assert out.loc[0, "draw_prob"] == pytest.approx(pmf[3])
    assert out.loc[0, "win_prob"] == pytest.approx(
        pmf[4:].sum()
    )  # not divided by 1 - draw
    assert out.loc[0, "win_prob"] + out.loc[0, "draw_prob"] < 1
    assert out.loc[0, "win_prob"] == pytest.approx(out.loc[0, "win_prob"])


def test_outcome_class_matches_the_probit_ordering():
    assert scipy.stats.norm.cdf(0) == 0.5  # the cut-free probability of an even game
    assert outcome_class([2, 0, -2]).tolist() == [2, 1, 0]


def test_two_outcome_distribution_models_keep_the_conditional_win_probability():
    class TwoWay(PmfModel):
        schema = TOY_SCHEMA

    out = Predictor(TwoWay(), sigma=1.0, sport="toy").predict(
        pd.DataFrame({"team": ["A"], "opp": ["B"], "location": [0]})
    )
    pmf = np.array([0.05, 0.1, 0.15, 0.3, 0.2, 0.15, 0.05])
    pmf = (pmf + pmf[::-1]) / 2
    # as before: the ties are left out, P(win) = P(> 0) / (P(> 0) + P(< 0))
    assert out.loc[0, "win_prob"] == pytest.approx(pmf[4:].sum() / (1 - pmf[3]))
    assert "draw_prob" not in out
