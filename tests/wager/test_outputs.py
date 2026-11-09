"""Neural families with extra outputs: the total and every box-score stat."""

import numpy as np
import pandas as pd
import pytest
import torch

from runitback.wager.core.predictor import load_predictor
from runitback.wager.core.schema import mirror, played
from runitback.wager.models.lightning import output_targets

from .conftest import TOY_SCHEMA
from .test_gnn import quick_trainer as gnn_trainer
from .test_transformer import quick_trainer as transformer_trainer


def test_output_targets_weight_the_total_and_share_the_stats_weight():
    assert output_targets(TOY_SCHEMA) == ((), ())
    names, weights = output_targets(TOY_SCHEMA, total_weight=0.5, stats_weight=1.0)
    assert names == ("total", *TOY_SCHEMA.stats)
    assert weights[0] == 0.5
    assert sum(weights[1:]) == pytest.approx(1.0)


def fit(family, games, tmp_path, **weights):
    if family == "transformer":
        from runitback.wager.models.transformer import fit_transformer

        return fit_transformer(
            games, "toy", trainer=transformer_trainer(3),
            save_dir=str(tmp_path / family), print_metrics=False, **weights,
        )  # fmt: skip
    from runitback.wager.models.gnn import default_model, fit_gnn

    return fit_gnn(
        games, "toy", model=default_model("toy", **weights), trainer=gnn_trainer(3),
        save_dir=str(tmp_path / family), print_metrics=False,
    )  # fmt: skip


@pytest.mark.parametrize("family", ["transformer", "gnn"])
def test_total_and_stats_outputs_train_and_serve(family, toy_games, tmp_path):
    res = fit(family, toy_games, tmp_path, total_weight=1.0, stats_weight=1.0)
    predictor = res["predictor"]
    assert res["model"].n_outputs == 2 + len(TOY_SCHEMA.stats)
    assert predictor.total_sigma > 0
    assert {"total_rmse", "total_mae"} <= set(res["metrics"]["val"])

    games = played(toy_games, TOY_SCHEMA)
    last = games[games.season == 2024]
    matchups = last.drop_duplicates("game_id").head(6)
    out = predictor.predict(matchups)
    # a symmetric total, an antisymmetric margin
    flipped = predictor.predict(mirror(matchups, TOY_SCHEMA))
    np.testing.assert_allclose(out.pred_total, flipped.pred_total, atol=1e-4)
    np.testing.assert_allclose(out.pred_margin, -flipped.pred_margin, atol=1e-4)
    # toy totals are ~42: a few epochs land in the right range
    assert out.pred_total.between(20, 70).all()
    over = 1 - torch.distributions.Normal(
        torch.tensor(out.pred_total.to_numpy()), predictor.total_sigma
    ).cdf(torch.tensor(matchups["total_line"].to_numpy()))
    np.testing.assert_allclose(out.over_prob, over.numpy(), atol=1e-5)

    loaded = load_predictor(str(tmp_path / family))
    loaded.update_state(toy_games)
    predictor.update_state(toy_games)
    np.testing.assert_allclose(
        loaded.predict(matchups).pred_total, predictor.predict(matchups).pred_total,
        atol=1e-4,
    )  # fmt: skip


@pytest.mark.parametrize("family", ["transformer", "gnn"])
def test_margin_only_models_serve_no_total(family, toy_games, tmp_path):
    # the toy sport's transformer predicts the total by default
    res = fit(family, toy_games, tmp_path, total_weight=0.0)
    out = res["predictor"].predict(
        played(toy_games, TOY_SCHEMA).drop_duplicates("game_id").head(3)
    )
    assert "pred_total" not in out and "over_prob" not in out
    assert res["predictor"].total_sigma is None


def test_missing_stats_are_left_out_of_the_loss(toy_games, tmp_path):
    games = toy_games.copy()
    early = games.season == 2021
    games.loc[early, "yards"] = np.nan
    res = fit("transformer", games, tmp_path, total_weight=1.0, stats_weight=1.0)
    assert np.isfinite(res["metrics"]["train"]["rmse"])
    assert np.isfinite(res["model"].target_mean.numpy()).all()


def test_predictors_saved_before_totals_still_serve(toy_games, tmp_path):
    """A pickled margin-only module lacks the new attributes."""
    res = fit("transformer", toy_games, tmp_path, total_weight=0.0)
    predictor = res["predictor"]
    module = predictor.model.module
    del module.extra_targets
    del module._buffers["loss_weights"]
    del predictor.total_sigma
    del module.head.out_dim
    matchups = pd.DataFrame({"team": ["T00"], "opp": ["T01"], "location": [1]})
    assert np.isfinite(predictor.predict(matchups).pred_margin).all()


def test_walk_forward_scores_totals_and_moneylines(toy_games):
    from runitback.wager.backtest import (
        format_summary,
        summarize,
        transformer_fitter,
        walk_forward,
    )

    fitter = transformer_fitter("toy", max_epochs=2, model={"total_weight": 1.0})
    results = walk_forward(toy_games, "toy", fitter, [2023])
    assert {"pred_total", "over_prob", "market_over_prob", "total"} <= set(results)
    assert results.market_over_prob.round(6).eq(0.5).all()  # -110 both ways

    summary = summarize(results, TOY_SCHEMA)
    totals = summary["totals"]
    assert totals["n"] == len(results)
    assert {"model", "market"} == set(totals["brier"])
    # the toy market knows the expected total
    assert totals["rmse"]["market"] < totals["rmse"]["model"]
    assert len(summary["moneyline"]) == 4
    text = format_summary(summary)
    assert "Over/under" in text and "Moneyline" in text


def test_score_distribution_learns_values_a_bell_curve_misses():
    from runitback.wager.models.lightning import ScoreDistribution

    dist = ScoreDistribution(-20, 20, symmetric=True)
    assert len(dist.bias) == 21  # |k| for k in -20..20
    log_prob = dist.log_prob(torch.tensor([2.0]), torch.tensor([np.log(6.0)]))
    assert torch.exp(log_prob).sum().item() == pytest.approx(1.0)

    # margins of about +-3 far more often than a bell curve allows
    rng = np.random.default_rng(0)
    y = np.where(
        rng.random(4000) < 0.3, rng.choice([-3, 3], 4000), rng.normal(0, 8, 4000)
    )
    y = torch.tensor(np.clip(np.round(y), -20, 20), dtype=torch.float32)
    loc = torch.zeros(1, requires_grad=True)
    log_scale = torch.full((1,), np.log(8.0), requires_grad=True)
    opt = torch.optim.Adam([loc, log_scale, *dist.parameters()], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        lp = dist.log_prob(loc.expand(len(y)), log_scale.expand(len(y)))
        dist.nll(lp, y).backward()
        opt.step()
    p = torch.exp(dist.log_prob(loc, log_scale))[0].detach().numpy()
    at = {k: p[k + 20] for k in (2, 3, 4, -3)}
    assert at[3] > 2 * at[4] and at[3] > 2 * at[2]
    assert at[3] == pytest.approx(at[-3], rel=0.05)  # tied weights
    # games without an outcome add nothing to the loss
    assert dist.nll(lp, torch.full_like(y, np.nan)).item() == 0


def test_distribution_model_prices_every_market_consistently(toy_games, tmp_path):
    from runitback.wager.models.transformer import default_model, fit_transformer

    model = default_model("toy", total_weight=1.0, distribution=True)
    assert model.n_outputs == 4 and model.head.out_dim == 4
    res = fit_transformer(
        toy_games, "toy", model=model, trainer=transformer_trainer(3),
        save_dir=str(tmp_path / "dist"), print_metrics=False,
    )  # fmt: skip
    predictor = res["predictor"]
    games = played(toy_games, TOY_SCHEMA)
    matchups = games[games.season == 2024].drop_duplicates("game_id").head(8)
    # integer lines, so pushes are possible and left out
    matchups = matchups.assign(spread=matchups["spread"].round(), total_line=44.0)
    out = predictor.predict(matchups)
    assert {"pred_margin", "win_prob", "cover_prob", "pred_total", "over_prob"} <= set(
        out
    )
    assert out[["win_prob", "cover_prob", "over_prob"]].stack().between(0, 1).all()
    flipped = predictor.predict(mirror(matchups, TOY_SCHEMA))
    np.testing.assert_allclose(out.pred_margin, -flipped.pred_margin, atol=1e-4)
    np.testing.assert_allclose(out.win_prob, 1 - flipped.win_prob, atol=1e-5)
    np.testing.assert_allclose(out.cover_prob, 1 - flipped.cover_prob, atol=1e-5)
    np.testing.assert_allclose(out.over_prob, flipped.over_prob, atol=1e-5)
    np.testing.assert_allclose(out.pred_total, flipped.pred_total, atol=1e-4)
    # covering a 0-point spread is winning
    even = predictor.predict(matchups.assign(spread=0.0))
    np.testing.assert_allclose(even.cover_prob, even.win_prob, atol=1e-6)

    loaded = load_predictor(str(tmp_path / "dist"))
    loaded.update_state(toy_games)
    predictor.update_state(toy_games)
    np.testing.assert_allclose(
        loaded.predict(matchups).cover_prob, predictor.predict(matchups).cover_prob,
        atol=1e-5,
    )  # fmt: skip


def test_normal_models_price_the_cover_too(toy_games, tmp_path):
    res = fit("transformer", toy_games, tmp_path, total_weight=1.0)
    predictor = res["predictor"]
    games = played(toy_games, TOY_SCHEMA).drop_duplicates("game_id").head(4)
    out = predictor.predict(games)
    expected = 1 - torch.distributions.Normal(0.0, predictor.sigma).cdf(
        torch.tensor((out.spread - out.pred_margin).to_numpy())
    )
    np.testing.assert_allclose(out.cover_prob, expected.numpy(), atol=1e-5)
