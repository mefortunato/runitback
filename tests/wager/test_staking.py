import json

import numpy as np
import pandas as pd
import pytest

from runitback.wager.staking import (
    Flat,
    LearnedKelly,
    Proportional,
    Threshold,
    market_bets,
    max_drawdown,
    returns,
    staking,
    summarize_stakes,
    walk_forward_stakes,
)

from .conftest import TOY_SCHEMA


def test_market_bets_take_the_models_side_of_each_market():
    results = pd.DataFrame(
        {
            "season": 2023, "week": 1, "game_id": ["a", "b", "c"],
            # model margin, spread, result
            "pred_margin": [5.0, -2.0, 3.0], "market_margin": [2.0, 1.0, 3.0],
            "margin": [7.0, 4.0, -3.0],
            "win_prob": [0.70, 0.40, 0.55], "market_prob": [0.60, 0.45, 0.55],
            "moneyline": [-150.0, 120.0, -120.0],
            "opp_moneyline": [130.0, -140.0, 100.0],
            "pred_total": [48.0, 40.0, 45.0], "total_line": [45.0, 44.0, 44.0],
            "total": [50.0, 44.0, 40.0],
            "over_odds": [-105.0, -110.0, -110.0],
            "under_odds": [-115.0, -120.0, -110.0],
        }
    )  # fmt: skip
    spread = market_bets(results, TOY_SCHEMA, "spread")
    # game c sits on the line: no pick
    assert spread["game_id"].tolist() == ["a", "b"]
    assert spread["edge"].tolist() == [3.0, 3.0]
    # a: team side covers (7 > 2); b: opponent side, team covered (4 > 1)
    assert spread["outcome"].tolist() == [1.0, -1.0]

    ml = market_bets(results, TOY_SCHEMA, "moneyline")
    assert ml["edge"].round(6).tolist() == [0.1, 0.05]
    assert ml["odds"].tolist() == [-150.0, -140.0]  # b: the opponent's line
    assert ml["outcome"].tolist() == [1.0, -1.0]

    total = market_bets(results, TOY_SCHEMA, "total")
    # a: over at -105 wins; b: under at -120 pushes; c: over loses
    assert total["odds"].tolist() == [-105.0, -120.0, -110.0]
    assert total["outcome"].tolist() == [1.0, 0.0, -1.0]
    np.testing.assert_allclose(returns(total), [100 / 105, 0.0, -1.0])


def synthetic_bets(seasons=range(2016, 2024), informative=True, n=400, seed=0):
    """Spread picks at -110 whose win rate rises with the edge (or not)."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        edge = rng.exponential(2.5, n)
        p = 0.45 + 0.03 * edge if informative else np.full(n, 0.5)
        win = rng.random(n) < np.clip(p, 0, 0.9)
        rows.append(
            pd.DataFrame(
                {
                    "season": season,
                    "date": rng.integers(1, 18, n),
                    "game_id": [f"{season}_{i}" for i in range(n)],
                    "edge": edge,
                    "odds": -110.0,
                    "outcome": np.where(win, 1.0, -1.0),
                }
            )  # fmt: skip
        )
    return pd.concat(rows, ignore_index=True)


def test_simple_policies():
    bets = pd.DataFrame({"edge": [1.0, 3.0, 5.0]})
    assert Flat().stakes(bets).tolist() == [1, 1, 1]
    assert Threshold(2.0).stakes(bets).tolist() == [0, 1, 1]
    assert Threshold(2.0).name == "edge>2"
    proportional = Proportional().fit(bets)
    assert proportional.stakes(bets).mean() == pytest.approx(1.0)
    assert proportional.stakes(pd.DataFrame({"edge": [6.0]})).tolist() == [2.0]


def test_learned_stakes_grow_with_an_edge_that_has_paid():
    bets = synthetic_bets()
    policy = LearnedKelly().fit(bets)
    small, large = pd.DataFrame({"edge": [0.5, 8.0], "odds": [-110.0, -110.0]}).pipe(
        lambda b: policy.stakes(b)
    )
    # below break-even near the line, a sizeable stake far from it
    assert small == 0 and large > 1
    assert policy.describe()["edge_slope_per_sd"] > 0
    train = policy.stakes(bets)
    # one unit on average, less a little where the cap binds
    assert train[train > 0].mean() == pytest.approx(1.0, abs=0.01)
    assert train.max() <= policy.max_units


def test_learned_stakes_are_zero_when_the_edge_has_not_paid():
    policy = LearnedKelly().fit(synthetic_bets(informative=False))
    stakes = policy.stakes(synthetic_bets(informative=False, seed=1))
    # a 50% pick at -110 loses money: only noise in the fit leaves a few
    # large-edge picks with a stake
    assert (stakes > 0).mean() < 0.1


def test_learned_stakes_account_for_the_odds():
    """Underdogs win less often: the fit must not read that as the edge
    failing, or stake the long odds for an edge that has not paid."""
    rng = np.random.default_rng(0)
    rows = []
    for season in range(2016, 2024):
        n = 600
        fair = rng.uniform(0.2, 0.8, n)
        priced = fair * 1.045  # the bookmaker's margin
        odds = np.where(
            priced >= 0.5, -100 * priced / (1 - priced), 100 * (1 - priced) / priced
        )
        edge = rng.uniform(0.01, 0.15, n)
        # picks win as often as the fair odds say, whatever the edge
        win = rng.random(n) < fair
        rows.append(
            pd.DataFrame(
                {
                    "season": season,
                    "date": 1,
                    "game_id": range(n),
                    "edge": edge,
                    "odds": odds,
                    "outcome": np.where(win, 1.0, -1.0),
                }
            )  # fmt: skip
        )
    bets = pd.concat(rows, ignore_index=True)
    stakes, _ = walk_forward_stakes(bets, [LearnedKelly()], 3)
    assert (stakes["stake_learned"] > 0).mean() < 0.1


def test_walk_forward_stakes_learn_from_earlier_seasons_only():
    bets = synthetic_bets()
    stakes, fits = walk_forward_stakes(bets, [Flat(), LearnedKelly()], 3)
    assert sorted(stakes["season"].unique()) == list(range(2019, 2024))
    # wrecking every season from 2020 on cannot change what 2019 learned
    wrecked = bets.copy()
    later = wrecked["season"] >= 2020
    wrecked.loc[later, "outcome"] = -wrecked.loc[later, "outcome"]
    _, wrecked_fits = walk_forward_stakes(wrecked, [LearnedKelly()], 3)
    assert wrecked_fits["learned"][2019] == fits["learned"][2019]
    assert wrecked_fits["learned"][2023] != fits["learned"][2023]
    with pytest.raises(ValueError, match="more than 3 seasons"):
        walk_forward_stakes(bets[bets.season < 2019], [Flat()], 3)


def test_summary_weighs_profit_by_stake():
    stakes = pd.DataFrame(
        {
            "season": 2020, "date": [1, 2, 3, 4],
            "outcome": [1.0, -1.0, 1.0, 0.0], "odds": -110.0,
            "stake_flat": 1.0, "stake_big": [3.0, 1.0, 0.0, 2.0],
        }
    )  # fmt: skip
    stakes["return"] = returns(stakes)
    summary = summarize_stakes(stakes, ["flat", "big"], n_boot=200)
    win = 100 / 110
    assert summary.loc["flat", "profit"] == pytest.approx(2 * win - 1)
    assert summary.loc["big", "profit"] == pytest.approx(3 * win - 1)
    assert summary.loc["big", "staked"] == 6
    assert summary.loc["big", "roi"] == pytest.approx((3 * win - 1) / 6)
    assert summary.loc["big", "bets"] == 3
    assert summary.loc["big", "weighted_win_pct"] == pytest.approx(3 / 4)
    assert summary.loc["flat", "roi_vs_flat"] == 0
    assert max_drawdown(np.array([1.0, -2.0, 0.5, -1.0, 3.0])) == pytest.approx(2.5)
    # a subset keeps its original row labels
    subset = summarize_stakes(stakes.set_index(np.arange(10, 14)), ["big"], n_boot=50)
    assert subset.loc["big", "profit"] == summary.loc["big", "profit"]


def test_staking_on_walk_forward_results(toy_games, tmp_path):
    from runitback.wager.backtest import transformer_fitter, walk_forward

    fitter = transformer_fitter("toy", max_epochs=2)
    results = walk_forward(toy_games, "toy", fitter, [2023, 2024])
    report = staking(
        results, "toy", min_train_seasons=1, output_dir=str(tmp_path),
        print_summary=False,
    )  # fmt: skip
    assert set(report) == {"spread", "moneyline", "total"}
    summary = report["spread"]["summary"]
    # walk-forward results carry the model's probabilities: its own Kelly too
    assert {
        "flat", "edge>2", "edge>4", "proportional", "learned", "model_kelly"
    } == set(summary.index)  # fmt: skip
    assert summary.loc["flat", "bets"] == len(report["spread"]["stakes"])
    saved = json.loads((tmp_path / "staking_summary.json").read_text())
    assert set(saved) == {"spread", "moneyline", "total"}
    assert 2024 in {int(k) for k in saved["spread"]["fits"]["learned"]}
    assert (tmp_path / "staking_bets.csv").exists()


def toy_results(seasons=range(2018, 2024), n=300, seed=1):
    """Walk-forward results whose spread picks win more often at big edges."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        market = rng.normal(0, 6, n).round(1)
        model = market + rng.normal(0, 3, n)
        # the game finishes toward the model by a fraction of its edge
        margin = (market + 0.4 * (model - market) + rng.normal(0, 12, n)).round()
        rows.append(
            pd.DataFrame(
                {
                    "season": season, "week": np.arange(n) % 18,
                    "game_id": [f"{season}-{i}" for i in range(n)],
                    "pred_margin": model, "market_margin": market, "margin": margin,
                }
            )
        )  # fmt: skip
    return pd.concat(rows, ignore_index=True)


def test_performance_staking_scores_stakes_that_grow_with_the_edge():
    from runitback.wager.staking import performance_staking

    out = performance_staking(toy_results(), TOY_SCHEMA)
    spread = out["spread"]
    assert spread["seasons"] == [2021, 2022, 2023]
    by_policy = {p["policy"]: p for p in spread["policies"]}
    assert {"flat", "proportional", "learned"} <= set(by_policy)
    # flat stakes one unit per pick
    assert by_policy["flat"]["staked"] == pytest.approx(spread["n"])
    curve = spread["curve"]
    edges = [c["edge"] for c in curve]
    units = [c["units"] for c in curve]
    assert edges == sorted(edges) and edges[0] > 0
    assert units == sorted(units) and units[-1] > units[0]
    assert max(units) <= spread["max_units"]


def test_performance_staking_is_empty_without_enough_seasons():
    from runitback.wager.staking import performance_staking

    assert performance_staking(toy_results(seasons=[2022, 2023]), TOY_SCHEMA) == {}
