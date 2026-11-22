import numpy as np
import pytest

from runitback.wager.backtest import (
    DEVIG_METHODS,
    no_vig_probabilities,
    no_vig_probability,
)
from runitback.wager.core.metrics import evaluate, outcome_class, three_way_scores
from runitback.wager.core.schema import GameSchema


def old_no_vig_probability(moneyline, opp_moneyline):
    """The two-way formula as it was before the k-way version."""

    def implied(line):
        line = np.asarray(line, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(line > 0, 100 / (line + 100), -line / (100 - line))

    team, opp = implied(moneyline), implied(opp_moneyline)
    return team / (team + opp)


def test_outcome_class_orders_loss_draw_win():
    assert outcome_class([-3, 0, 2, -1, 1]).tolist() == [0, 1, 2, 0, 2]


@pytest.mark.parametrize(
    "margin, rps, brier3, log_loss3",
    [
        # forecast win .5, draw .3, loss .2
        (2, 0.5 * (0.2**2 + 0.5**2), 0.2**2 + 0.3**2 + 0.5**2, -np.log(0.5)),
        (0, 0.5 * (0.2**2 + 0.5**2), 0.5**2 + 0.7**2 + 0.2**2, -np.log(0.3)),
        (-1, 0.5 * (0.8**2 + 0.5**2), 0.5**2 + 0.3**2 + 0.8**2, -np.log(0.2)),
    ],
)
def test_three_way_scores_by_hand(margin, rps, brier3, log_loss3):
    scores = three_way_scores([margin], [0.5], [0.3])
    assert scores["rps"][0] == pytest.approx(rps)
    assert scores["brier3"][0] == pytest.approx(brier3)
    assert scores["log_loss3"][0] == pytest.approx(log_loss3)
    assert scores["correct3"][0] == (margin > 0)


def test_rps_rewards_a_near_miss():
    # a win happened: forecasting the draw is closer than forecasting the loss
    draw_forecast = three_way_scores([1], [0.1], [0.8])
    loss_forecast = three_way_scores([1], [0.1], [0.1])
    assert draw_forecast["rps"][0] < loss_forecast["rps"][0]
    # the multiclass Brier score cannot tell the two apart
    assert draw_forecast["brier3"][0] == pytest.approx(loss_forecast["brier3"][0])


def test_evaluate_scores_every_game_when_there_are_draws():
    y = np.array([2.0, 0.0, -1.0, 0.0])
    margin = np.array([1.0, 0.1, -0.5, -0.2])
    win, draw = np.array([0.6, 0.3, 0.2, 0.3]), np.array([0.25, 0.4, 0.3, 0.4])
    metrics = evaluate(y, margin, win, draw)
    assert metrics["n"] == 4  # the draws count
    assert {"rps", "brier3", "log_loss3", "accuracy3"} <= set(metrics)
    assert not {"brier", "log_loss", "accuracy"} & set(metrics)
    scores = three_way_scores(y, win, draw)
    assert metrics["rps"] == pytest.approx(scores["rps"].mean())
    # without draw probabilities the two-outcome metrics are scored on decided games, as before
    two_way = evaluate(y, margin, win)
    assert {"brier", "log_loss", "accuracy"} <= set(two_way) and "rps" not in two_way


def test_two_way_no_vig_probability_is_unchanged_bit_for_bit():
    rng = np.random.default_rng(0)
    a = np.r_[
        rng.integers(-400, -100, 50), rng.integers(100, 600, 50), [np.nan, -110, 100]
    ]
    b = np.r_[
        rng.integers(100, 600, 50), rng.integers(-400, -100, 50), [-110, np.nan, 100]
    ]
    new = no_vig_probability(a, b)
    assert np.array_equal(new, old_no_vig_probability(a, b), equal_nan=True)
    assert np.array_equal(no_vig_probabilities(a, b)[:, 0], new, equal_nan=True)
    assert float(no_vig_probability(-150, 130)) == float(
        old_no_vig_probability(-150, 130)
    )


@pytest.mark.parametrize("method", DEVIG_METHODS)
def test_three_way_probabilities_sum_to_one(method):
    home = np.array([-150.0, 120.0, 250.0, 700.0, np.nan])
    draw = np.array([280.0, 250.0, 260.0, 900.0, 300.0])
    away = np.array([400.0, 210.0, -105.0, -3000.0, 250.0])
    q = no_vig_probabilities(home, draw, away, method=method)
    assert q.shape == (5, 3)
    np.testing.assert_allclose(q[:4].sum(axis=1), 1.0)
    assert np.isnan(q[4]).all()  # one price missing: no probabilities
    assert (q[:4] > 0).all()


def test_power_and_shin_take_more_margin_off_longshots():
    # a clear favourite: the draw and the away win are the longshots
    prices = (np.array([-400.0]), np.array([500.0]), np.array([900.0]))
    proportional, power, shin = (
        no_vig_probabilities(*prices, method=m)[0] for m in DEVIG_METHODS
    )
    assert power[0] > proportional[0] and shin[0] > proportional[0]
    assert power[2] < proportional[2] and shin[2] < proportional[2]


def test_no_margin_to_remove_stays_proportional():
    # fair prices (the implied probabilities sum to 1): every method agrees
    prices = (np.array([100.0]), np.array([100.0]))
    for method in DEVIG_METHODS:
        np.testing.assert_allclose(
            no_vig_probabilities(*prices, method=method), [[0.5, 0.5]]
        )


def test_devig_input_checks():
    with pytest.raises(ValueError, match="at least two"):
        no_vig_probabilities([-110.0])
    with pytest.raises(ValueError, match="unknown method"):
        no_vig_probabilities([-110.0], [-110.0], method="mean")


def test_schema_defaults_leave_two_outcome_sports_alone():
    fields = GameSchema.__dataclass_fields__
    assert fields["draws"].default is False and fields["market_draw"].default is None
    assert fields["devig"].default == "proportional"


def test_epl_schema_is_a_three_outcome_schema():
    from runitback.wager.epl.data.features import SCHEMA

    assert SCHEMA.draws and SCHEMA.market_draw == "draw_moneyline"
    assert SCHEMA.devig in DEVIG_METHODS
