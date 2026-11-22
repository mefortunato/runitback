import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from runitback.wager import backtest as bt
from runitback.wager import experiments, tune
from runitback.wager.backtest import no_vig_probabilities
from runitback.wager.core.metrics import three_way_scores
from runitback.wager.core.sport import register_sport

from .football import DRAWS_SCHEMA, football_games, true_probabilities

SPORT = "toy_draw_backtest"


@pytest.fixture(scope="module")
def sport(toy_sport):
    return register_sport(
        dataclasses.replace(toy_sport, name=SPORT, schema=DRAWS_SCHEMA)
    )


@pytest.fixture(scope="module")
def games():
    return football_games()


@pytest.fixture(scope="module")
def results(games, sport):
    return {
        name: bt.walk_forward(games, SPORT, bt.baseline_fitter(SPORT, name), [2023])
        for name in ("home", "elo")
    }


def test_market_arm_is_the_average_price_without_its_margin(games, sport):
    rows = games[games.groupby("game_id").cumcount() == 0].head(5)
    market = bt.market_predictions(rows, DRAWS_SCHEMA, sigma=1.0)
    win, draw, _ = true_probabilities()
    # quoted at 3% margin, removed proportionally: the true probabilities come back
    np.testing.assert_allclose(market["market_prob"], win, atol=2e-3)
    np.testing.assert_allclose(market["market_draw_prob"], draw, atol=2e-3)
    expected = no_vig_probabilities(*(rows[c] for c in DRAWS_SCHEMA.market_average))
    np.testing.assert_allclose(market["market_prob"], expected[:, 0])
    # the preferred bookmaker is its own arm
    book = no_vig_probabilities(rows.moneyline, rows.draw_moneyline, rows.opp_moneyline)
    np.testing.assert_allclose(market["book_prob"], book[:, 0])
    assert (market["market_prob"] + market["market_draw_prob"] < 1).all()
    np.testing.assert_allclose(market["market_margin"], rows["spread"])


def test_market_arm_falls_back_to_the_preferred_prices_without_an_average(games):
    schema = dataclasses.replace(DRAWS_SCHEMA, market_average=None)
    rows = games[games.groupby("game_id").cumcount() == 0].head(3)
    market = bt.market_predictions(rows, schema, sigma=1.0)
    assert "book_prob" not in market  # the same prices: no second arm
    book = no_vig_probabilities(rows.moneyline, rows.draw_moneyline, rows.opp_moneyline)
    np.testing.assert_allclose(market["market_prob"], book[:, 0])


def test_market_is_missing_where_its_prices_are(games):
    rows = games[games.groupby("game_id").cumcount() == 0].head(3).copy()
    rows.loc[rows.index[0], "avg_draw_moneyline"] = np.nan
    market = bt.market_predictions(rows, DRAWS_SCHEMA, sigma=1.0)
    assert (
        np.isnan(market["market_prob"].iloc[0])
        and market["market_prob"].iloc[1:].notna().all()
    )


def test_walk_forward_carries_win_and_draw_probabilities(results):
    frame = results["elo"]
    assert {"win_prob", "draw_prob", "market_prob", "market_draw_prob"} <= set(frame)
    assert {"book_prob", "book_draw_prob", "avg_moneyline", "draw_moneyline"} <= set(
        frame
    )
    assert "cover_prob" not in frame
    assert (frame["win_prob"] + frame["draw_prob"] <= 1).all()
    assert frame["draw_prob"].between(0.05, 0.6).all()


def test_summary_scores_draws_and_pairs_the_model_with_each_market_arm(results):
    schema = DRAWS_SCHEMA
    summary = bt.summarize(results["elo"], schema)
    assert not {"ats", "edge_correlation", "moneyline"} & set(summary)
    overall = summary["overall"]
    assert set(overall) == {"model", "market", "book"}
    assert {"rps", "brier3", "log_loss3", "accuracy3"} <= set(overall["model"])
    assert overall["model"]["n"] == len(results["elo"])  # the draws count
    # model and market arm are scored on the same games
    assert overall["market"]["n"] == overall["model"]["n"]
    frame = results["elo"]
    scores = three_way_scores(frame[schema.target], frame.win_prob, frame.draw_prob)
    assert overall["model"]["rps"] == pytest.approx(scores["rps"].mean())
    for arm in ("market", "book"):
        paired = summary["paired"][arm]
        assert {"d_rps", "d_rps_ci95", "d_log_loss3", "n"} <= set(paired)
        arm_rps = overall[arm]["rps"]
        assert paired["d_rps"] == pytest.approx(overall["model"]["rps"] - arm_rps)
    assert json.dumps(summary, default=float)  # what backtest() writes
    assert set(summary["by_season"]) == {2023}
    text = bt.format_summary(summary)
    assert "rps" in text and "Model minus market" in text and "Model minus book" in text


def test_a_market_that_knows_the_odds_is_not_beaten(results):
    # the games' true probabilities never change, and the market quotes them
    summary = bt.summarize(results["home"], DRAWS_SCHEMA)
    paired = summary["paired"]["market"]
    assert abs(paired["d_rps"]) < 3 * paired["d_rps_ci95"] + 1e-3


def test_compare_ranks_families_and_market_by_rps(results):
    table, per_game = bt.compare(results, DRAWS_SCHEMA)
    assert set(table.index) == {"home", "elo", "market", "book"}
    assert table["rps"].is_monotonic_increasing
    assert {"rps_vs_market", "rps_vs_market_ci95", "early_rps"} <= set(table)
    assert "rps_vs_market" not in table.loc["market"].dropna()
    assert {"elo_draw_prob", "market_draw_prob"} <= set(per_game)
    assert table.loc["market", "n"] == len(per_game)


def test_ablation_and_season_tables_use_rps(results):
    table = experiments.ablation_table(results, SPORT, baseline="home")
    assert set(table.index) == {"home", "elo", "vegas", "book"}
    assert table.loc["home", "d_rps"] == 0
    assert {"rps", "brier3", "log_loss3", "vs_market", "vs_market_ci95"} <= set(table)
    assert "brier" not in table
    seasons = experiments.season_table(results, SPORT, baseline="home")
    assert list(seasons.index) == [2023] and (seasons["home"] == 0).all()
    markdown = experiments.format_table(table)
    assert "RPS" in markdown and "| elo |" in markdown


def test_tuning_score_is_the_mean_rps(results):
    frame = results["elo"]
    scores = three_way_scores(frame["margin"], frame.win_prob, frame.draw_prob)
    assert tune.season_brier(frame, SPORT) == pytest.approx(scores["rps"].mean())


def test_backtest_runs_end_to_end_on_a_draws_sport(games, sport, tmp_path):
    out = bt.backtest(
        games,
        SPORT,
        family="elo",
        seasons=[2023],
        output_dir=str(tmp_path),
        print_summary=False,
    )
    assert out["summary"]["paired"]["market"]["n"] == len(out["results"])
    assert (tmp_path / "elo_games.csv").exists()
    saved = json.loads((tmp_path / "elo_summary.json").read_text())
    assert saved["overall"]["model"]["rps"] == pytest.approx(
        out["summary"]["overall"]["model"]["rps"]
    )
    # a season's games are predicted once each (no game left out for a missing state)
    assert out["results"][DRAWS_SCHEMA.game_id].is_unique


def test_benchmark_prints_rps(games, sport, capsys):
    out = bt.benchmark(
        games, SPORT, families=["home", "elo"], seasons=[2023], print_summary=True
    )
    assert "sorted by RPS" in capsys.readouterr().out
    assert isinstance(out["table"], pd.DataFrame) and "rps" in out["table"]
