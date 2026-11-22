import numpy as np
import pandas as pd
import pytest

from runitback.wager.models.baselines import EloModel, EloParams

from .football import DRAWS_SCHEMA

K, HOME = 20.0, 65.0
FOOTBALL = EloParams(k=K, home=HOME, mov="football")


def one_game(margin, schema=DRAWS_SCHEMA):
    """A (home) vs B (away), both perspectives."""
    rows = []
    for team, opp, loc, m in (("A", "B", 1, margin), ("B", "A", -1, -margin)):
        rows.append(
            {
                "season": 2023,
                "week": 1,
                "game_id": "g",
                "team": team,
                "opp": opp,
                "location": loc,
                "margin": float(m),
            }
        )
    return pd.DataFrame(rows)


def shift(params, margin):
    """How far the home team's rating moves after one game, from level ratings."""
    model = EloModel(DRAWS_SCHEMA, params)
    model.replay(one_game(margin))
    return model.ratings["A"] - params.start


def expected_home():
    return 1 / (1 + 10 ** (-HOME / 400))


@pytest.mark.parametrize(
    "margin, actual, multiplier",
    [(0, 0.5, 1.0), (1, 1.0, 1.0), (-1, 0.0, 1.0), (2, 1.0, 1.5), (-2, 0.0, 1.5),
     (3, 1.0, 14 / 8), (5, 1.0, 16 / 8), (-4, 0.0, 15 / 8)],
)  # fmt: skip
def test_football_goal_difference_multiplier(margin, actual, multiplier):
    assert shift(FOOTBALL, margin) == pytest.approx(
        K * multiplier * (actual - expected_home())
    )


def test_a_draw_moves_ratings_in_football_but_not_with_the_nfl_rule():
    assert shift(FOOTBALL, 0) == pytest.approx(K * (0.5 - expected_home()))
    assert shift(FOOTBALL, 0) < 0  # the favoured home side drops a point on a draw
    assert (
        shift(EloParams(k=K, home=HOME), 0) == 0
    )  # log(0 + 1) = 0: as FiveThirtyEight's


def test_an_away_draw_lifts_the_underdog():
    model = EloModel(DRAWS_SCHEMA, FOOTBALL)
    model.replay(one_game(0))
    assert model.ratings["B"] > FOOTBALL.mean > model.ratings["A"]
    assert model.ratings["A"] + model.ratings["B"] == pytest.approx(2 * FOOTBALL.mean)


def test_new_teams_start_at_their_own_rating():
    params = EloParams(k=K, home=HOME, mov="football", new_team=1405.0)
    model = EloModel(DRAWS_SCHEMA, params)
    ratings = model.pre_game_ratings(one_game(1))
    assert ratings[["team_elo", "opp_elo"]].iloc[0].tolist() == [1405.0, 1405.0]
    assert params.start == 1405.0 and EloParams().start == EloParams().mean
    # a team the model has not seen predicts from that rating too
    matchups = pd.DataFrame(
        {
            "team": ["A", "Z"],
            "opp": ["Z", "B"],
            "location": [0, 0],
            "season": [2023, 2023],
        }
    )
    margin = model.predict_margin(matchups)
    assert margin[0] == pytest.approx(
        (model.ratings["A"] - 1405.0) * params.points_per_elo
    )
    assert margin[1] == pytest.approx(
        (1405.0 - model.ratings["B"]) * params.points_per_elo
    )


def test_defaults_are_the_nfl_settings_and_the_rule_is_checked():
    params = EloParams()
    assert (params.mov, params.new_team, params.k, params.home) == (
        "nfl",
        None,
        20.0,
        65.0,
    )
    assert EloParams(k=6.0, home=30.0, points_per_elo=1 / 120) == EloParams(
        k=6.0, home=30.0, points_per_elo=1 / 120
    )
    with pytest.raises(ValueError, match="football"):
        EloParams(mov="soccer")


def test_football_elo_over_a_season_keeps_ratings_zero_sum():
    rng = np.random.default_rng(0)
    teams = [f"T{i}" for i in range(8)]
    rows = []
    for week in range(1, 31):
        order = rng.permutation(teams).reshape(-1, 2)
        for a, b in order:
            m = int(rng.poisson(1.4) - rng.poisson(1.1))
            for t, o, loc, mm in ((a, b, 1, m), (b, a, -1, -m)):
                rows.append({"season": 2023, "week": week, "game_id": f"{week}{a}", "team": t,
                             "opp": o, "location": loc, "margin": float(mm)})  # fmt: skip
    model = EloModel(DRAWS_SCHEMA, FOOTBALL)
    model.replay(pd.DataFrame(rows))
    assert sum(model.ratings.values()) == pytest.approx(8 * FOOTBALL.mean)
    assert max(model.ratings.values()) > FOOTBALL.mean > min(model.ratings.values())
