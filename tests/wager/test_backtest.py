import json

import numpy as np
import pandas as pd
import pytest
import scipy.stats
import sklearn.linear_model

from runitback.wager.backtest import (
    BREAKEVEN,
    ats_record,
    backtest,
    no_vig_probability,
    sklearn_fitter,
    walk_forward,
)
from runitback.wager.cli import run

from .conftest import TOY_SCHEMA, make_games


def test_walk_forward_only_sees_the_past(toy_games):
    fits, states = [], []
    base = sklearn_fitter("toy", model=sklearn.linear_model.Ridge())

    def fitter(games, split):
        fits.append((games.season.max(), split))
        predictor = base(games, split)
        update = predictor.update_state

        def spy(known):
            states.append(known)
            update(known)

        predictor.update_state = spy
        return predictor

    results = walk_forward(toy_games, "toy", fitter, seasons=[2023, 2024])

    assert [(season, split.val) for season, split in fits] == [
        (2022, [2022]),
        (2023, [2023]),
    ]
    predicted = results[["season", "week"]].drop_duplicates().itertuples(index=False)
    for (season, week), known in zip(predicted, states, strict=True):
        assert known.season.max() <= season
        assert (known[known.season == season].week < week).all()


def test_results_have_one_row_per_game_with_market(toy_games):
    results = walk_forward(
        toy_games,
        "toy",
        sklearn_fitter("toy", model=sklearn.linear_model.Ridge()),
        seasons=[2024],
    )
    played = toy_games[(toy_games.season == 2024) & toy_games.margin.notna()]
    assert len(results) == played.game_id.nunique() == results.game_id.nunique()
    assert (results.location >= 0).all()
    home = played.set_index(["game_id", "team"]).spread
    assert np.allclose(
        results.market_margin, home.loc[list(zip(results.game_id, results.team))]
    )
    assert results.market_prob.between(0, 1).all()


def test_ats_record_counts_wins_losses_and_pushes():
    results = pd.DataFrame(
        {
            # model     spread   result
            "pred_margin": [5.0, -1.0, 3.0, 4.0],
            "market_margin": [2.0, 2.0, 3.0, 3.0],
            "margin": [4.0, 10.0, 0.0, 3.0],
        }
    )
    # edge +3 covers (win); edge -3 but team covered (loss); edge 0 (no bet);
    # edge +1 lands on the number (push)
    ats = ats_record(results, TOY_SCHEMA, thresholds=(0, 1)).set_index("min_edge")
    assert ats.loc[0, ["bets", "wins", "losses", "pushes"]].tolist() == [3, 1, 1, 1]
    assert ats.loc[0, "win_pct"] == 0.5
    assert ats.loc[0, "roi"] == pytest.approx((100 / 110 - 1) / 3)
    assert ats.loc[1, ["bets", "wins", "losses"]].tolist() == [2, 1, 1]
    assert BREAKEVEN == pytest.approx(0.5238, abs=1e-4)


def test_moneyline_record_bets_the_value_side_at_the_posted_odds():
    from runitback.wager.backtest import moneyline_record

    results = pd.DataFrame(
        {
            "win_prob": [0.70, 0.30, 0.55, 0.62],
            "market_prob": [0.60, 0.40, 0.54, 0.60],
            "moneyline": [-150.0, 120.0, -120.0, -150.0],
            "opp_moneyline": [130.0, -140.0, 100.0, 130.0],
            "margin": [7.0, 3.0, -3.0, 0.0],
        }
    )
    # team at -150 wins (+2/3); opponent at -140 loses (team won: -1);
    # team at -120 loses (-1); team at -150 ties (push)
    ml = moneyline_record(results, TOY_SCHEMA, thresholds=(0, 0.05))
    ml = ml.set_index("min_edge")
    assert ml.loc[0, ["bets", "wins", "losses", "pushes"]].tolist() == [4, 1, 2, 1]
    assert ml.loc[0, "units"] == pytest.approx(100 / 150 - 2)
    assert ml.loc[0.05, ["bets", "wins", "losses"]].tolist() == [2, 1, 1]


def test_totals_record_and_summary():
    from runitback.wager.backtest import totals_record, totals_summary

    results = pd.DataFrame(
        {
            "pred_total": [50.0, 40.0, 45.0, 44.0],
            "total_line": [45.0, 44.0, 45.0, 42.0],
            "total": [51.0, 47.0, 30.0, 42.0],
            "over_odds": [-105.0, -110.0, -110.0, -110.0],
            "under_odds": [-115.0, -110.0, -110.0, -110.0],
            "over_prob": [0.7, 0.4, 0.5, 0.55],
        }
    )
    # over at -105 wins; under loses; no bet on the line; over pushes
    record = totals_record(results, TOY_SCHEMA, thresholds=(0, 3)).set_index("min_edge")
    assert record.loc[0, ["bets", "wins", "losses", "pushes"]].tolist() == [3, 1, 1, 1]
    assert record.loc[0, "units"] == pytest.approx(100 / 105 - 1)
    assert record.loc[3, ["bets", "wins", "losses"]].tolist() == [2, 1, 1]

    summary = totals_summary(results, TOY_SCHEMA)
    assert summary["n"] == 4
    assert summary["rmse"]["market"] == pytest.approx(np.sqrt(np.mean([36, 9, 225, 0])))
    # the push is left out of the over/under Brier
    assert summary["brier"]["model"] == pytest.approx(
        np.mean([(0.7 - 1) ** 2, (0.4 - 1) ** 2, 0.5**2])
    )


def test_no_vig_probability():
    assert no_vig_probability(-110, -110) == pytest.approx(0.5)
    fav, dog = no_vig_probability(-200, 170), no_vig_probability(170, -200)
    assert fav + dog == pytest.approx(1.0)
    assert fav == pytest.approx((2 / 3) / (2 / 3 + 100 / 270))
    assert np.isnan(no_vig_probability(np.nan, -110))


def test_backtest_writes_outputs_and_scores_the_market(toy_games, tmp_path):
    out = backtest(
        toy_games,
        "toy",
        model=sklearn.linear_model.Ridge(),
        output_dir=str(tmp_path),
        print_summary=False,
    )
    summary = out["summary"]
    assert summary["seasons"] == [2023]  # 2024 is unfinished
    # the toy market knows each game's true expected margin
    overall = summary["overall"]
    assert overall["market"]["rmse"] < overall["model"]["rmse"]
    assert {"first 4 dates", "rest"} == set(summary["by_phase"])
    assert len(summary["ats"]) == 6

    saved = json.loads((tmp_path / "sklearn_summary.json").read_text())
    assert saved["overall"]["model"]["n"] == overall["model"]["n"]
    games = pd.read_csv(tmp_path / "sklearn_games.csv")
    assert {"pred_margin", "win_prob", "market_margin", "market_prob"} <= set(games)


def test_backtest_from_cli(toy_games, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    toy_games.to_csv("games.csv", index=False)
    out = run(
        [
            "toy", "backtest",
            "--games", "games.csv",
            "--model", "sklearn.linear_model.Ridge",
            "--seasons", "[2023, 2024]",
            "--print_summary", "false",
        ]
    )  # fmt: skip
    assert out["summary"]["seasons"] == [2023, 2024]
    assert (tmp_path / "backtests/toy/sklearn_summary.json").exists()


def test_benchmark_compares_families_on_common_games(toy_games, tmp_path):
    from runitback.wager.backtest import benchmark

    out = benchmark(
        toy_games,
        "toy",
        families=["home", "elo", "gnn", "transformer"],
        seasons=[2023],
        max_epochs=2,
        output_dir=str(tmp_path),
        print_summary=False,
    )
    table = out["table"]
    assert set(table.index) == {"home", "elo", "gnn", "transformer", "market"}
    assert table.brier.is_monotonic_increasing
    assert table.loc["market", "brier"] < table.loc["home", "brier"]
    assert np.isnan(table.loc["market", "brier_vs_market"])
    assert (table.drop("market").brier_vs_market_ci95 > 0).all()
    assert len(out["games"]) == table.n.iloc[0]
    for name in ("benchmark.csv", "benchmark.json", "benchmark_games.csv"):
        assert (tmp_path / name).exists()
    assert {"elo_margin", "transformer_prob"} <= set(out["games"])


def test_implied_sigma_recovers_the_residual_spread():
    from runitback.wager.backtest import implied_sigma

    rng = np.random.default_rng(0)
    mu = rng.normal(0, 1.0, 20000)
    y = mu + rng.normal(0, 2.5, len(mu))
    prob = scipy.stats.norm.cdf(mu / 2.5)
    assert implied_sigma(y, prob) == pytest.approx(2.5, rel=0.03)


def test_moneyline_only_markets_are_scored_without_a_spread(toy_games, monkeypatch):
    """Hockey and baseball: no point spread, the market is its moneylines."""
    import dataclasses

    from runitback.wager.backtest import summarize
    from runitback.wager.core import sport

    schema = dataclasses.replace(TOY_SCHEMA, market_spread=None)
    toy_ml = sport.Sport(name="toy_ml", schema=schema, download=make_games)
    monkeypatch.setitem(sport._registry, "toy_ml", toy_ml)
    results = walk_forward(
        toy_games,
        "toy_ml",
        sklearn_fitter("toy_ml", model=sklearn.linear_model.Ridge()),
        seasons=[2024],
    )
    no_vig = no_vig_probability(results.moneyline, results.opp_moneyline)
    assert np.allclose(results.market_prob, no_vig)
    # the implied margin has the moneyline's sign and roughly the spread's size
    assert (np.sign(results.market_margin) == np.sign(results.market_prob - 0.5)).all()
    spread = toy_games.set_index(["game_id", "team"]).spread
    spread = spread.loc[list(zip(results.game_id, results.team))].to_numpy()
    assert np.corrcoef(results.market_margin, spread)[0, 1] > 0.95
    summary = summarize(results, schema)
    assert "ats" not in summary and "moneyline" in summary
    assert summary["overall"]["market"]["n"] == len(results)
