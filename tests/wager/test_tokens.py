"""Market and efficiency history tokens, the antisymmetric head, seed
ensembles, the ablation report and the tuning search space."""

import dataclasses
import tempfile

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
import torch

from runitback.wager.core import efficiency as eff
from runitback.wager.core.predictor import load_predictor
from runitback.wager.core.schema import mirror, played
from runitback.wager.core.sport import get_sport, register_sport
from runitback.wager.experiments import ablation_table, blend_weight, format_table
from runitback.wager.models.transformer import fit_transformer
from runitback.wager.models.transformer.data import (
    MARKET_TOKEN_COLUMNS,
    SequenceFeatures,
    efficiency_columns,
    market_features,
    opponent_rows,
)
from runitback.wager.models.transformer.fit import transformer_config, transformer_dims
from runitback.wager.models.transformer.model import PairHead, TransformerEnsemble

from .conftest import TOY_SCHEMA, make_games


def toy_efficiency(games):
    return eff.both_sides(
        games,
        lambda own, opp: {"yards_point": eff.ratio(own("yards"), own("points"))},
        lambda own, opp: {"yard_share": own("yards") / (own("yards") + opp("yards"))},
    )


@pytest.fixture(scope="module", autouse=True)
def toy_eff_sport():
    toy = get_sport("toy")
    return register_sport(
        dataclasses.replace(toy, name="toy_eff", efficiency=toy_efficiency)
    )


@pytest.fixture(scope="module")
def games():
    return make_games(persistence=0.7)


def quick_trainer(max_epochs=4):
    return pl.Trainer(
        max_epochs=max_epochs,
        default_root_dir=tempfile.mkdtemp(),
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min")],
    )


def test_market_features_describe_the_game_against_its_lines(games):
    rows = played(games, TOY_SCHEMA)
    m = market_features(rows, TOY_SCHEMA)
    assert list(m.columns) == MARKET_TOKEN_COLUMNS
    np.testing.assert_allclose(m["_mkt_cover"], rows["margin"] - rows["spread"])
    np.testing.assert_allclose(
        m["_mkt_total_resid"], rows["total"] - rows["total_line"]
    )
    # favorites by the spread are favorites by the moneyline
    clear = rows["spread"].abs() >= 3
    assert (np.sign(m["_mkt_prob_z"][clear]) == np.sign(rows["spread"][clear])).all()
    assert (m["_mkt_has_line"] == 1).all()
    # a game without lines has NaN features and no line flag
    none = rows.head(2).assign(spread=np.nan, moneyline=np.nan, opp_moneyline=np.nan)
    m = market_features(none, TOY_SCHEMA)
    assert m["_mkt_cover"].isna().all() and (m["_mkt_has_line"] == 0).all()


def test_market_ratings_reproduce_the_next_line(games):
    from runitback.wager.models.transformer.data import histories, market_ratings

    rows = played(games, TOY_SCHEMA).reset_index(drop=True)
    ratings = market_ratings(rows, TOY_SCHEMA)
    opp = opponent_rows(rows, TOY_SCHEMA)
    np.testing.assert_allclose(
        ratings["_mkt_rating"], ratings["_mkt_opp_rating"].to_numpy()[opp]
    )
    # each team's rating after its previous game predicts the next spread
    last = histories(rows, TOY_SCHEMA, 1)[:, 0]
    ok = (last >= 0) & (last[opp] >= 0) & (rows["season"] > 2021).to_numpy()
    r = ratings["_mkt_rating"].to_numpy()
    estimate = r[last] - r[last[opp]]
    assert np.corrcoef(estimate[ok], rows["spread"][ok])[0, 1] > 0.9


def test_efficiency_columns_mirror_between_the_two_rows_of_a_game(games):
    rows = played(games, TOY_SCHEMA).reset_index(drop=True)
    frame = toy_efficiency(rows)
    assert list(frame.columns) == efficiency_columns("toy_eff")
    opp = opponent_rows(rows, TOY_SCHEMA)
    np.testing.assert_allclose(
        frame["_eff_yards_point"], frame["_eff_opp_yards_point"].to_numpy()[opp]
    )
    assert efficiency_columns("toy") == []


def test_basketball_four_factors():
    row = pd.DataFrame(
        {
            "points": [110], "fgm": [40], "fga": [85], "fg3m": [12], "fg3a": [33],
            "ftm": [18], "fta": [22], "oreb": [10], "dreb": [34], "tov": [13],
            "ast": [25], "stl": [7], "blk": [5],
            "opp_points": [100], "opp_fgm": [38], "opp_fga": [88],
            "opp_fg3m": [10], "opp_fg3a": [30], "opp_ftm": [14], "opp_fta": [18],
            "opp_oreb": [12], "opp_dreb": [32], "opp_tov": [15], "opp_ast": [22],
            "opp_stl": [8], "opp_blk": [4],
        }
    )  # fmt: skip
    out = eff.basketball_efficiency(row).iloc[0]
    poss = ((85 - 10 + 13 + 0.44 * 22) + (88 - 12 + 15 + 0.44 * 18)) / 2
    assert out["_eff_pace"] == pytest.approx(poss)
    assert out["_eff_rating"] == pytest.approx(100 * 110 / poss)
    assert out["_eff_opp_rating"] == pytest.approx(100 * 100 / poss)
    assert out["_eff_net_rating"] == pytest.approx(100 * 10 / poss)
    assert out["_eff_efg"] == pytest.approx((40 + 6) / 85)
    assert out["_eff_orb_pct"] == pytest.approx(10 / (10 + 32))


def test_nhl_shares_and_mlb_rates():
    nhl = pd.DataFrame(
        {
            "goals": [3], "shots": [30], "missed_shots": [10], "blocked_shots": [12],
            "empty_net_goals": [1], "pp_goals": [1], "pp_opportunities": [4],
            "opp_goals": [2], "opp_shots": [25], "opp_missed_shots": [8],
            "opp_blocked_shots": [15], "opp_empty_net_goals": [0],
        }
    )  # fmt: skip
    out = eff.nhl_efficiency(nhl).iloc[0]
    cf, ca = 30 + 10 + 15, 25 + 8 + 12
    assert out["_eff_corsi_pct"] == pytest.approx(cf / (cf + ca))
    assert out["_eff_shooting_pct"] == pytest.approx(2 / 30)
    assert out["_eff_save_pct"] == pytest.approx(1 - 2 / 25)
    mlb = pd.DataFrame(
        {
            "runs": [5], "hits": [9], "walks": [3], "hit_by_pitch": [1],
            "at_bats": [34], "sac_flies": [1], "total_bases": [15],
            "plate_appearances": [39], "strikeouts": [8], "home_runs": [2],
        }
    )  # fmt: skip
    out = eff.mlb_efficiency(mlb).iloc[0]
    assert out["_eff_obp"] == pytest.approx(13 / 39)
    assert out["_eff_slg"] == pytest.approx(15 / 34)
    assert out["_eff_babip"] == pytest.approx(7 / 25)


def test_every_sport_derives_its_efficiency_columns_from_an_empty_table():
    from runitback.wager.core.sport import BUILTIN_SPORTS

    for name in BUILTIN_SPORTS:
        columns = efficiency_columns(name)
        assert columns and all(c.startswith("_eff_") for c in columns), name


def test_nfl_play_by_play_efficiency():
    from runitback.wager.nfl.data.pbp import PBP_COLUMNS, game_efficiency

    pbp = pd.DataFrame(
        {
            "game_id": "g", "season": 2024, "week": 1,
            "posteam": ["A", "A", "A", "A", "B", "A"],
            "defteam": ["B", "B", "B", "B", "A", "B"],
            "play_type": ["pass", "run", "pass", "run", "pass", "punt"],
            "epa": [1.0, -0.5, -1.0, 0.5, 0.2, -0.3],
            "success": [1, 0, 0, 1, 1, 0],
            "down": [1, 2, 3, 1, 1, 4],
            "qb_dropback": [1, 0, 1, 0, 1, 0],
            "rush": [0, 1, 0, 1, 0, 0],
            "yards_gained": [25, 2, -5, 12, 6, 0],
            "wp": [0.5, 0.5, 0.9, 0.5, 0.5, 0.5],
            "interception": [0, 0, 1, 0, 0, 0],
            "fumble_lost": 0,
            "special_teams_play": [0, 0, 0, 0, 0, 1],
            "drive": [1, 1, 1, 2, 3, 2],
            "fixed_drive_result": ["Punt", "Punt", "Punt", "Touchdown",
                                   "Field goal", "Touchdown"],
        }
    )  # fmt: skip
    out = game_efficiency(pbp).set_index("team")
    a = out.loc["A"]
    assert set(PBP_COLUMNS) <= set(out.columns)
    assert a["pbp_plays"] == 4
    assert a["pbp_epa_play"] == pytest.approx(0.0)
    assert a["pbp_success"] == pytest.approx(0.5)
    assert a["pbp_pass_epa"] == pytest.approx(0.0)
    assert a["pbp_early_epa"] == pytest.approx((1.0 - 0.5 + 0.5) / 3)
    assert a["pbp_explosive"] == pytest.approx(2 / 4)
    assert a["pbp_turnover_epa"] == pytest.approx(-1.0)
    assert a["pbp_st_epa"] == pytest.approx(-0.3)
    assert a["pbp_drive_score"] == pytest.approx(0.5)


def test_token_options_set_the_token_width(games):
    schema = get_sport("toy_eff").schema
    plain = transformer_dims(schema)["token_dim"]
    wide = transformer_dims(
        schema, market_tokens=True, efficiency=True, sport="toy_eff"
    )["token_dim"]
    assert wide == plain + len(MARKET_TOKEN_COLUMNS) + 3
    config = transformer_config("toy_eff", market_tokens=True, efficiency=True)
    assert config["encoder"]["token_dim"] == wide
    assert config["token_features"]["market_tokens"]
    features = SequenceFeatures.fit(
        played(games, schema), schema, 8, market_tokens=True, efficiency=True,
        sport="toy_eff",
    )  # fmt: skip
    assert features.token_dim == wide
    tokens = features.tokens(played(games, schema))
    assert np.isfinite(tokens).all()


def test_antisymmetric_head_is_exactly_antisymmetric():
    torch.manual_seed(0)
    head = PairHead(8, context_dim=3, out_dim=3, antisymmetric=True,
                    symmetric_outputs=(1,)).eval()  # fmt: skip
    a, b = torch.randn(5, 8), torch.randn(5, 8)
    ca, cb = torch.randn(5, 3), torch.randn(5, 3)
    forward, backward = head(a, b, ca, cb), head(b, a, cb, ca)
    torch.testing.assert_close(forward[:, 0], -backward[:, 0])
    torch.testing.assert_close(forward[:, 1], backward[:, 1])


def test_fit_with_every_option_serves_an_antisymmetric_ensemble(games, tmp_path):
    res = fit_transformer(
        games,
        "toy_eff",
        trainer=lambda: quick_trainer(3),
        batch_size=64,
        market_tokens=True,
        efficiency=True,
        model=None,
        ensemble=2,
        save_dir=str(tmp_path / "model"),
        print_metrics=False,
    )
    assert isinstance(res["model"], TransformerEnsemble)
    assert len(res["model"].members) == 2
    # the two members start from different seeds
    w0, w1 = (m.head.net[0].weight for m in res["model"].members)
    assert not torch.equal(w0, w1)
    predictor = load_predictor(str(tmp_path / "model"))
    upcoming = games[games["margin"].isna()].drop_duplicates("game_id")
    out = predictor.predict(upcoming)
    back = predictor.predict(mirror(upcoming, TOY_SCHEMA))
    np.testing.assert_allclose(out["pred_margin"], -back["pred_margin"], atol=1e-5)
    assert np.isfinite(out["pred_total"]).all()


def test_antisymmetric_model_from_config(games):
    from runitback.wager.models.transformer import default_model

    model = default_model("toy", antisymmetric=True)
    assert model.head.antisymmetric and model.head.symmetric_outputs == (1,)
    res = fit_transformer(
        games, "toy", model=model, trainer=quick_trainer(2), batch_size=64,
        print_metrics=False,
    )  # fmt: skip
    assert res["model"].target_mean[0] == 0  # margin centered for antisymmetry


def test_blend_weight_recovers_the_mixing_weight():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.2, 0.8, 20000)
    won = (rng.uniform(size=truth.size) < truth).astype(float)
    market = np.clip(truth + rng.normal(0, 0.08, truth.size), 0.02, 0.98)
    # a model that is the truth: the blend should lean on it fully
    out = blend_weight(won, market, truth)
    assert out["blend_w"] == pytest.approx(1.0, abs=0.15)
    # a model that is pure noise around the market gets no weight
    noise = np.clip(market + rng.normal(0, 0.08, truth.size), 0.02, 0.98)
    assert abs(blend_weight(won, market, noise)["blend_w"]) < 0.1


def test_ablation_table_compares_configs_on_common_games():
    rng = np.random.default_rng(1)
    n = 400
    y = rng.normal(0, 13, n).round()
    base = pd.DataFrame(
        {
            "season": 2024, "week": 1, "game_id": np.arange(n), "team": "A",
            "opp": "B", "margin": y, "market_margin": y * 0.3,
            "market_prob": 0.5 + 0.01 * y,
        }
    )  # fmt: skip
    base["market_prob"] = base["market_prob"].clip(0.05, 0.95)
    results = {
        "a": base.assign(pred_margin=y * 0.2, win_prob=0.5 + 0.005 * y),
        "b": base.assign(pred_margin=y * 0.1, win_prob=0.5 + 0.002 * y),
    }
    for r in results.values():
        r["win_prob"] = r["win_prob"].clip(0.05, 0.95)
    table = ablation_table(results, "toy", baseline="a")
    assert table.loc["a", "d_brier"] == 0
    assert table.loc["b", "d_brier"] > 0  # less informative than a
    assert "vegas" in table.index
    assert "| b |" in format_table(table)


def test_search_space_builds_a_runnable_config():
    import optuna

    from runitback.wager.tune import merge, params_to_config, search_space

    params = {
        "d_model": 32, "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1,
        "nhead": 2, "num_layers": 1, "hidden_dim": 32, "history": 8,
        "season_decay": False, "batch_size": 64,
    }  # fmt: skip
    config = merge({"model": {"market_tokens": True}}, search_space(
        optuna.trial.FixedTrial(params), "toy"
    ))  # fmt: skip
    assert config["model"]["market_tokens"] and config["fit"]["batch_size"] == 64
    assert params_to_config(params)["fit"] == {"batch_size": 64}
    transformer_config("toy", **config["model"])  # valid model settings
