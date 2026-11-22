import numpy as np
import pandas as pd
import scipy.stats

from runitback.wager.backtest import no_vig_probabilities, no_vig_probability
from runitback.wager.core.efficiency import EXTRA_EFFICIENCY, epl_extra
from runitback.wager.models.transformer.data import (
    MARKET_TOKEN_COLUMNS,
    MARKET_TOKEN_COLUMNS_DRAWS,
    SequenceFeatures,
    eff2_columns,
    extra_columns,
    market_features,
)

from .conftest import TOY_SCHEMA
from .football import DRAWS_SCHEMA, football_games

SCHEMA = DRAWS_SCHEMA.__class__(
    **{**DRAWS_SCHEMA.__dict__, "market_closing": "closing"}
)


def three_games() -> pd.DataFrame:
    """A home win, a draw and an away win, with the prices each was quoted at."""
    return pd.DataFrame(
        {
            "margin": [2.0, 0.0, -1.0],
            "total": [3.0, 2.0, 1.0],
            "spread": [0.5, 0.25, -0.5],
            "total_line": [2.5, 2.5, 2.5],
            "moneyline": [-120.0, 250.0, 300.0],
            "draw_moneyline": [260.0, 240.0, 270.0],
            "opp_moneyline": [330.0, 110.0, -135.0],
            "over_odds": [-125.0, 100.0, -105.0],
            "under_odds": [105.0, -120.0, -115.0],
            "closing": [1.0, 0.0, 1.0],
        }
    )


def test_a_sport_with_draws_has_its_own_market_tokens():
    tokens = market_features(three_games(), SCHEMA)
    assert list(tokens) == MARKET_TOKEN_COLUMNS_DRAWS
    assert "_mkt_total" not in tokens  # the constant 2.5 line is gone
    q = no_vig_probabilities(
        *(three_games()[c] for c in ("moneyline", "draw_moneyline", "opp_moneyline"))
    )
    np.testing.assert_allclose(tokens["_mkt_win_minus_loss"], q[:, 0] - q[:, 2])
    np.testing.assert_allclose(tokens["_mkt_draw_prob"], q[:, 1])
    # points won (3 a win, 1 a draw, 0 a loss) minus the points the prices expected
    points = np.array([3.0, 1.0, 0.0])
    np.testing.assert_allclose(
        tokens["_mkt_points_surprise"], points - (3 * q[:, 0] + q[:, 1])
    )
    over = no_vig_probability(three_games()["over_odds"], three_games()["under_odds"])
    np.testing.assert_allclose(tokens["_mkt_over_z"], scipy.stats.norm.ppf(over))
    np.testing.assert_allclose(tokens["_mkt_total_resid"], [0.5, -0.5, -1.5])
    np.testing.assert_allclose(tokens["_mkt_spread"], [0.5, 0.25, -0.5])
    np.testing.assert_allclose(tokens["_mkt_cover"], [1.5, -0.25, -0.5])
    assert tokens["_mkt_closing"].tolist() == [1.0, 0.0, 1.0]
    assert (tokens["_mkt_has_line"] == 1).all()


def test_the_surprise_is_zero_on_average_for_fair_prices():
    # prices that are exactly the true probabilities, results drawn from them
    rng = np.random.default_rng(0)
    p = np.array([0.45, 0.27, 0.28])
    n = 20000
    outcome = rng.choice(3, size=n, p=p)
    odds = lambda x: -100 * x / (1 - x) if x >= 0.5 else 100 * (1 - x) / x
    games = pd.DataFrame(
        {
            "margin": np.array([1.0, 0.0, -1.0])[outcome],
            "moneyline": odds(p[0]),
            "draw_moneyline": odds(p[1]),
            "opp_moneyline": odds(p[2]),
            "spread": 0.0,
            "total": 2.0,
            "total_line": 2.5,
            "over_odds": -110.0,
            "under_odds": -110.0,
        }
    )
    tokens = market_features(games, DRAWS_SCHEMA)
    assert abs(tokens["_mkt_points_surprise"].mean()) < 0.03
    assert (
        tokens["_mkt_closing"].isna().all()
    )  # a schema without the column: no flag, no error


def test_missing_prices_give_missing_tokens():
    games = three_games()
    games.loc[1, "draw_moneyline"] = np.nan
    tokens = market_features(games, SCHEMA)
    assert (
        tokens.loc[1, ["_mkt_win_minus_loss", "_mkt_draw_prob", "_mkt_points_surprise"]]
        .isna()
        .all()
    )
    assert tokens.loc[[0, 2], "_mkt_draw_prob"].notna().all()


def test_two_outcome_market_tokens_are_the_ones_they_were():
    assert extra_columns(False, False, market_tokens=True) == MARKET_TOKEN_COLUMNS
    assert (
        extra_columns(False, False, market_tokens=True, draws=True)
        == MARKET_TOKEN_COLUMNS_DRAWS
    )
    games = football_games().head(6)
    assert list(market_features(games, TOY_SCHEMA)) == MARKET_TOKEN_COLUMNS


def test_a_constant_token_column_is_standardized_to_zero_not_divided_by_zero():
    games = football_games()
    played = games.dropna(subset=["margin"])
    features = SequenceFeatures.fit(played, DRAWS_SCHEMA, history=8, market_tokens=True)
    assert (
        features.token_columns[-len(MARKET_TOKEN_COLUMNS_DRAWS) :]
        == MARKET_TOKEN_COLUMNS_DRAWS
    )
    # the over/under is -110 / -110 in every game: its probit is 0 everywhere
    assert played["over_odds"].nunique() == 1
    assert (
        features.std["_mkt_over_z"] == 1.0
    )  # a zero spread is replaced, not divided by
    tokens = features.tokens(played)
    assert np.isfinite(tokens).all()
    assert (tokens[:, features.token_columns.index("_mkt_over_z")] == 0).all()


def test_eff2_adds_second_half_goals_and_red_cards():
    games = pd.DataFrame(
        {
            "goals": [3.0, 1.0, np.nan],
            "half_time_goals": [1.0, 1.0, np.nan],
            "red_cards": [0.0, 1.0, np.nan],
            "opp_goals": [1.0, 3.0, np.nan],
            "opp_half_time_goals": [1.0, 0.0, np.nan],
            "opp_red_cards": [1.0, 0.0, np.nan],
        }
    )
    out = epl_extra(games)
    assert out["_eff_second_half_goals"].tolist()[:2] == [2.0, 0.0]
    assert out["_eff_opp_second_half_goals"].tolist()[:2] == [0.0, 3.0]
    assert out["_eff_red_card"].tolist()[:2] == [0.0, 1.0] and out[
        "_eff_opp_red_card"
    ].tolist()[:2] == [1.0, 0.0]
    assert out.iloc[2].isna().all()  # no result, no columns
    assert EXTRA_EFFICIENCY["epl"] is epl_extra
    columns = eff2_columns("epl")
    assert columns == list(out) and "_eff_second_half_goals" in columns
    assert eff2_columns("nfl") == [] and eff2_columns(None) == []
    assert extra_columns(False, False, sport="epl", eff2=True) == columns
    assert extra_columns(False, False, sport="epl") == []  # an option of its own
