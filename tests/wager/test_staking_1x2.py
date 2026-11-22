import dataclasses

import numpy as np
import pandas as pd
import pytest

from runitback import wager  # noqa: F401
from runitback.wager import backtest as bt
from runitback.wager import staking
from runitback.wager.backtest import no_vig_probabilities, payout
from runitback.wager.core.sport import register_sport

from .conftest import TOY_SCHEMA
from .football import DRAWS_SCHEMA

SPORT = "toy_staking_1x2"


def results_frame(rows):
    """Walk-forward-like results: (game, p win, p draw, goal margin, odds home / draw / away)."""
    return pd.DataFrame(
        [
            {
                "season": 2023, "week": i + 1, "game_id": game, "team": "A", "opp": "B",
                "margin": margin, "win_prob": win, "draw_prob": draw,
                "moneyline": home, "draw_moneyline": d, "opp_moneyline": away,
            }
            for i, (game, win, draw, margin, home, d, away) in enumerate(rows)
        ]
    )  # fmt: skip


# game, P(win), P(draw), margin, American odds of the home win / draw / away win.
# g1: EV home .5 * 2.2 - 1 = .10, draw .3 * 3.5 - 1 = .05, away .2 * 5 - 1 = 0 -> home
# g2: EV home -.10, draw .4 * 4 - 1 = .60, away .3 * 3.5 - 1 = .05 -> draw
# g3: EV home .2 * 6 - 1 = .20, draw -.05, away .55 * 1.909 - 1 = .05 -> home
# g4: home -.20, draw -.10, away -.10: every outcome loses money at its price -> no bet
GAMES = [
    ("g1", 0.5, 0.3, 0, 120, 250, 400),  # ends level
    ("g2", 0.3, 0.4, 0, 200, 300, 250),  # ends level
    ("g3", 0.2, 0.25, -1, 500, 280, -110),  # away win
    ("g4", 0.4, 0.3, 2, 100, 200, 200),
]


@pytest.fixture
def results():
    return results_frame(GAMES)


def test_picks_are_the_outcome_with_the_best_expected_value(results):
    picks = bt.one_x_two_picks(results, DRAWS_SCHEMA)
    assert picks["pick"][:3].tolist() == ["home", "draw", "home"]
    np.testing.assert_allclose(picks["edge"][:3], [0.10, 0.60, 0.20])
    assert picks["odds"][:3].tolist() == [120, 300, 500]
    np.testing.assert_allclose(picks["prob"][:3], [0.5, 0.4, 0.2])
    assert picks["edge"][3] < 0  # nothing worth betting on g4
    # the probability edge is the model's minus the market's, margin removed
    market = no_vig_probabilities(120.0, 250.0, 400.0)
    assert picks["prob_edge"][0] == pytest.approx(0.5 - market[0])


def test_a_draw_loses_the_side_bets_and_wins_the_draw_bets(results):
    picks = bt.one_x_two_picks(results, DRAWS_SCHEMA)
    # g1: a home pick, level: lost; g2: a draw pick, level: won; g3: a home pick, away win: lost
    assert picks["outcome"][:3].tolist() == [-1.0, 1.0, -1.0]
    bets = staking.market_bets(results, DRAWS_SCHEMA, "1x2")
    assert bets["game_id"].tolist() == ["g1", "g2", "g3"]  # only the bets worth making
    assert bets["pick"].tolist() == ["home", "draw", "home"]
    np.testing.assert_allclose(staking.returns(bets), [-1.0, payout(300), -1.0])
    assert not (bets["outcome"] == 0).any()  # no push in 1X2
    assert {"season", "date", "edge", "odds", "outcome", "prob", "prob_edge"} <= set(
        bets
    )


def test_the_record_counts_bets_above_each_expected_value(results):
    record = bt.one_x_two_record(results, DRAWS_SCHEMA, thresholds=(0, 0.15)).set_index(
        "min_edge"
    )
    assert record.loc[0, ["bets", "wins", "losses", "pushes"]].tolist() == [3, 1, 2, 0]
    assert record.loc[0, "units"] == pytest.approx(payout(300) - 2)
    assert record.loc[0.15, ["bets", "wins", "losses"]].tolist() == [
        2,
        1,
        1,
    ]  # g2 and g3
    assert record.loc[0.15, "units"] == pytest.approx(payout(300) - 1)
    # a sport with draws has no two-way moneyline record: it is the 1X2 one
    assert bt.moneyline_record(results, DRAWS_SCHEMA, (0, 0.15)).equals(
        record.reset_index()
    )


def test_missing_prices_or_probabilities_skip_the_game(results):
    results.loc[1, "draw_moneyline"] = np.nan
    results.loc[2, "win_prob"] = np.nan
    assert bt.one_x_two_picks(results, DRAWS_SCHEMA).index.tolist() == [0, 3]


def test_markets_of_a_sport_with_draws():
    assert staking.default_markets(DRAWS_SCHEMA, pd.DataFrame()) == ["1x2"]
    assert staking.default_markets(
        DRAWS_SCHEMA, pd.DataFrame({"pred_total": [1.0]})
    ) == ["1x2", "total"]
    assert staking.default_markets(TOY_SCHEMA, pd.DataFrame()) == [
        "spread",
        "moneyline",
    ]
    assert staking.MARKETS == (
        "spread",
        "moneyline",
        "total",
    )  # unchanged for the others


def test_the_wrong_markets_are_refused(results):
    for market in ("spread", "moneyline"):
        with pytest.raises(ValueError, match="1x2"):
            staking.market_bets(results, DRAWS_SCHEMA, market)
    with pytest.raises(ValueError, match="needs a sport with draws"):
        staking.market_bets(results, TOY_SCHEMA, "1x2")


def test_the_summary_of_a_backtest_has_the_1x2_record(results):
    football = results.assign(pred_margin=0.0, market_prob=0.4, market_draw_prob=0.3)
    summary = bt.summarize(football, DRAWS_SCHEMA)
    assert summary["one_x_two"][0]["bets"] == 3
    assert "1X2" in bt.format_summary(summary)


def synthetic_season_results(seed=0, seasons=range(2018, 2024), games=300):
    """Model probabilities with a little real signal, priced by a 5% margin on noisy fair odds."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for i in range(games):
            truth = rng.dirichlet([4.5, 2.7, 3.2])
            model = np.clip(truth + rng.normal(0, 0.04, 3), 0.02, None)
            model /= model.sum()
            fair = np.clip(truth + rng.normal(0, 0.05, 3), 0.02, None)
            fair /= fair.sum()
            outcome = rng.choice(3, p=truth)  # 0 home win, 1 draw, 2 away win
            price = bt.implied_probability  # noqa: F841
            odds = np.where(fair * 1.05 >= 0.5, -100 * fair * 1.05 / (1 - fair * 1.05), 100 * (1 - fair * 1.05) / (fair * 1.05))  # fmt: skip
            rows.append(
                {"season": season, "week": i // 10 + 1, "game_id": f"{season}_{i}", "team": "A", "opp": "B",
                 "margin": [1.0, 0.0, -1.0][outcome], "win_prob": model[0], "draw_prob": model[1],
                 "moneyline": odds[0], "draw_moneyline": odds[1], "opp_moneyline": odds[2]}
            )  # fmt: skip
    return pd.DataFrame(rows)


def test_staking_policies_run_on_three_way_picks(toy_sport):
    register_sport(dataclasses.replace(toy_sport, name=SPORT, schema=DRAWS_SCHEMA))
    results = synthetic_season_results()
    report = staking.staking(results, SPORT, print_summary=False)
    assert list(report) == ["1x2"]
    stakes, summary = report["1x2"]["stakes"], report["1x2"]["summary"]
    assert {"flat", "learned", "model_kelly"} <= set(summary.index)
    assert stakes["season"].min() == 2021  # three earlier seasons to learn from
    bets = staking.market_bets(results, DRAWS_SCHEMA, "1x2")
    assert summary.loc["flat", "bets"] == (bets["season"] >= 2021).sum()
    assert set(stakes["pick"]) <= {"home", "draw", "away"}
    assert np.isfinite(summary["roi"]).any()
    assert "learned" in report["1x2"]["fits"] and "1x2" in staking.format_staking(
        report
    )
