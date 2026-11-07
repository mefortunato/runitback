import tempfile

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
import torch

from runitback.wager.core.metrics import antisymmetrize
from runitback.wager.core.predictor import load_predictor
from runitback.wager.core.schema import played
from runitback.wager.models.transformer import default_model, fit_transformer
from runitback.wager.models.transformer.data import (
    SequenceData,
    SequenceFeatures,
    histories,
    opponent_rows,
    strength,
)

from .conftest import TOY_SCHEMA, make_games


def quick_trainer(max_epochs=8):
    return pl.Trainer(
        max_epochs=max_epochs,
        default_root_dir=tempfile.mkdtemp(),
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min")],
    )


@pytest.fixture(scope="module")
def toy_games():
    """A league whose team strength carries over between seasons, as real
    ones do (the opponent-strength features rely on that)."""
    return make_games(persistence=0.7)


@pytest.fixture(scope="module")
def played_games(toy_games):
    return played(toy_games, TOY_SCHEMA).reset_index(drop=True)


@pytest.fixture(scope="module")
def fitted(toy_games, tmp_path_factory):
    save_dir = tmp_path_factory.mktemp("models") / "transformer"
    res = fit_transformer(
        toy_games,
        "toy",
        trainer=quick_trainer(20),
        batch_size=32,  # enough optimizer steps on a small toy league
        opponent_strength=True,
        save_dir=str(save_dir),
        print_metrics=False,
    )
    return res, save_dir


def test_histories_are_earlier_games_of_the_same_team(played_games):
    games = played_games
    hist = histories(games, TOY_SCHEMA, history=8)
    for row in [0, 100, 500, len(games) - 1]:
        idx = hist[row][hist[row] >= 0]
        earlier = games.iloc[idx]
        now = (games.season[row], games.week[row])
        assert (earlier.team == games.team[row]).all()
        when = list(zip(earlier.season, earlier.week))
        assert all(w < now for w in when)
        assert when == sorted(when, reverse=True)  # most recent first
        team = games[games.team == games.team[row]]
        n_before = sum(w < now for w in zip(team.season, team.week))
        assert len(idx) == min(8, n_before)


def test_histories_cross_seasons_and_opponent_rows_pair_up(played_games):
    hist = histories(played_games, TOY_SCHEMA, history=8)
    opener = played_games[(played_games.season == 2022) & (played_games.week == 1)]
    row = opener.index[0]
    assert (played_games.season.iloc[hist[row]] == 2021).all()  # all last season

    opp = opponent_rows(played_games, TOY_SCHEMA)
    assert (played_games.team.iloc[opp].to_numpy() == played_games.opp.to_numpy()).all()
    assert (
        played_games.game_id.iloc[opp].to_numpy() == played_games.game_id.to_numpy()
    ).all()


@pytest.mark.parametrize("week", [1, 6])
def test_served_predictions_match_training_path(fitted, toy_games, week):
    """Serving rebuilds the same sequences training saw, including openers."""
    res, _ = fitted
    predictor, module = res["predictor"], res["model"]
    games = played(toy_games, TOY_SCHEMA).reset_index(drop=True)
    target = games[(games.season == 2024) & (games.week == week)]

    predictor.update_state(games[(games.season < 2024) | (games.week < week)])
    served = predictor.predict(target)

    features = predictor.model.features
    data = SequenceData(games, TOY_SCHEMA, features).with_rows(target.index)
    with torch.no_grad():
        raw = module.predict_margin(data[list(range(len(target)))]).numpy()
    expected = antisymmetrize(target, raw, TOY_SCHEMA.game_id)
    np.testing.assert_allclose(served.pred_margin, expected, atol=1e-4)


def test_learns_and_round_trips(fitted, toy_games):
    res, save_dir = fitted
    assert res["metrics"]["test"]["accuracy"] > 0.6

    loaded = load_predictor(str(save_dir))
    loaded.update_state(toy_games)
    matchups = pd.DataFrame(
        {"team": ["T00", "T01"], "opp": ["T01", "T00"], "location": [1, -1]}
    )
    out = loaded.predict(matchups)
    assert out.pred_margin[0] == pytest.approx(-out.pred_margin[1])
    assert np.isfinite(out.win_prob).all()


def test_seeded_fits_are_reproducible(toy_games):
    def fit():
        res = fit_transformer(
            toy_games, "toy", trainer=quick_trainer(2), print_metrics=False
        )
        return res["metrics"]["val"]

    assert fit() == fit()


def test_mismatched_model_dims_are_reported(toy_games):
    with pytest.raises(ValueError, match="context_dim"):
        fit_transformer(
            toy_games,
            "toy",
            model=default_model("toy", context=[]),
            trainer=quick_trainer(1),
        )


def test_features_standardize_on_training_games(played_games):
    features = SequenceFeatures.fit(played_games, TOY_SCHEMA, history=8)
    tokens = features.tokens(played_games)
    assert tokens.shape == (len(played_games), features.token_dim)
    np.testing.assert_allclose(tokens.mean(axis=0), 0, atol=1e-4)


def test_strength_features_use_only_earlier_games(played_games):
    games = played_games
    full = strength(games, TOY_SCHEMA)
    # drop every game after 2024 week 5: features of the rest must not move
    early = games[(games.season < 2024) | (games.week < 6)]
    truncated = strength(early, TOY_SCHEMA)
    pd.testing.assert_frame_equal(full.loc[early.index], truncated)
    # Elo going into a game reflects earlier results: better teams rate higher
    gap = full["_team_elo"] - full["_opp_elo"]
    assert np.corrcoef(gap, games.margin)[0, 1] > 0.2


def test_season_decay_down_weights_earlier_seasons():
    from runitback.wager.models.transformer.model import SequenceEncoder

    torch.manual_seed(0)
    tokens = torch.randn(4, 6, 5)
    ago = torch.tensor([[0.0, 0, 0, 1, 1, 2]] * 4)
    pad = torch.zeros(4, 6, dtype=torch.bool)
    changed = tokens.clone()
    changed[:, 3:] += 5.0  # only last season's and older games differ

    def shift(**options):
        enc = SequenceEncoder(token_dim=5, history=6, d_model=16, nhead=2,
                              num_layers=2, dropout=0.0, **options).eval()  # fmt: skip
        with torch.no_grad():
            return (enc(tokens, ago, pad) - enc(changed, ago, pad)).abs().max()

    assert shift() > 0.1
    # a steep penalty per season leaves earlier seasons almost unseen
    assert shift(season_decay=True, season_embedding=True, decay_init=30.0) < 1e-3


def test_season_options_train_and_round_trip(toy_games, tmp_path):
    model = default_model("toy", history=10, season_embedding=True, season_decay=True)
    res = fit_transformer(
        toy_games,
        "toy",
        model=model,
        trainer=quick_trainer(2),
        save_dir=str(tmp_path / "model"),
        print_metrics=False,
    )
    encoder = res["model"].encoder
    assert encoder.history == 10 and encoder.decay is not None

    loaded = load_predictor(str(tmp_path / "model"))
    for predictor in (res["predictor"], loaded):
        predictor.update_state(toy_games)
    matchups = pd.DataFrame({"team": ["T00"], "opp": ["T03"], "location": [1]})
    np.testing.assert_allclose(
        loaded.predict(matchups).pred_margin,
        res["predictor"].predict(matchups).pred_margin,
        atol=1e-5,
    )


def test_elo_margin_is_the_margin_elo_did_not_expect(played_games, toy_games):
    from runitback.wager.models.baselines import EloModel

    extra = strength(played_games, TOY_SCHEMA, opponent_strength=False, elo_margin=True)
    assert list(extra) == ["_margin_vs_elo"]
    elo = EloModel(TOY_SCHEMA)
    expected = elo.replay(played_games) * elo.params.points_per_elo
    np.testing.assert_allclose(
        extra["_margin_vs_elo"], played_games["margin"] - expected
    )
    # a game's adjustment depends only on that game and the ones before it
    early = played_games[played_games.season < 2024]
    np.testing.assert_allclose(
        strength(early, TOY_SCHEMA, False, True)["_margin_vs_elo"],
        extra.loc[early.index, "_margin_vs_elo"],
    )

    res = fit_transformer(
        toy_games, "toy", elo_margin=True, trainer=quick_trainer(1),
        print_metrics=False,
    )  # fmt: skip
    assert res["predictor"].model.features.token_columns[-1] == "_margin_vs_elo"


def test_sport_setting_turns_on_opponent_strength(toy_sport):
    import dataclasses

    from runitback.wager.core.sport import register_sport
    from runitback.wager.models.transformer.fit import transformer_config

    base = transformer_config("toy")["encoder"]["token_dim"]
    strong = dataclasses.replace(
        toy_sport, name="toy_strength", transformer={"opponent_strength": True}
    )
    register_sport(strong)
    config = transformer_config("toy_strength")
    assert config["encoder"]["token_dim"] == base + 3
    assert "opponent_strength" not in config
    off = transformer_config("toy_strength", opponent_strength=False)
    assert off["encoder"]["token_dim"] == base


def test_team_waiting_for_its_first_game_keeps_state(toy_games):
    """Regression: a team on a bye in week 1 lost its state until it played."""
    fitted = fit_transformer(
        toy_games, "toy", trainer=pl.Trainer(max_epochs=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False), print_metrics=False,
    )  # fmt: skip
    model = fitted["predictor"].model
    games = played(toy_games, TOY_SCHEMA)
    last = games["season"].max()
    week1 = games[(games["season"] == last) & (games["week"] == 1)]
    resting = week1["team"].iloc[0]
    known = pd.concat(
        [games[games["season"] < last], week1[~week1["team"].isin([resting])]]
    )
    model.update_state(known)
    assert resting in model.teams


def test_team_embeddings_encode_each_teams_history(fitted, toy_games):
    loaded = load_predictor(str(fitted[1]))
    loaded.update_state(toy_games)
    embeddings = loaded.model.team_embeddings()
    assert list(embeddings.index) == loaded.model.teams
    assert embeddings.shape[1] == loaded.model.module.encoder.out_dim
    assert np.isfinite(embeddings.to_numpy()).all()


def test_additive_lineup_terms_are_monotone_and_hidden_from_the_mlp():
    import torch

    from runitback.wager.models.transformer.data import additive_spec, lineup_columns
    from runitback.wager.models.transformer.model import PairHead

    n_before = 3
    team = lineup_columns("pva")
    spec = additive_spec(n_before, "pva")
    assert spec["additive_pairs"][0] == (n_before, n_before + len(team))
    assert len(spec["additive_pairs"]) == 18 and len(spec["mask_context"]) == 40
    ctx_dim = n_before + 2 * len(team)
    torch.manual_seed(0)
    head = PairHead(8, ctx_dim, out_dim=2, antisymmetric=True, symmetric_outputs=(1,), **spec).eval()
    t, o = torch.randn(5, 8), torch.randn(5, 8)
    ctx = torch.randn(5, ctx_dim)
    opp_ctx = torch.cat([ctx[:, :n_before], ctx[:, n_before + len(team):], ctx[:, n_before:n_before + len(team)]], 1)
    with torch.no_grad():
        base = head(t, o, ctx, opp_ctx)
        for g, total_up in ((0, True), (5, False)):        # a better QB / a better pass rusher
            better = ctx.clone()
            better[:, n_before + g] += 1.0
            better_opp = opp_ctx.clone()
            better_opp[:, n_before + len(team) + g] += 1.0
            out = head(t, o, better, better_opp)
            assert (out[:, 0] > base[:, 0]).all()            # the team's margin goes up, every row
            assert ((out[:, 1] > base[:, 1]) if total_up else (out[:, 1] < base[:, 1])).all()
        # the MLP never sees the lineup columns: only the additive terms change
        m0, _ = head.additive(ctx)
        ctx2 = ctx.clone()
        ctx2[:, n_before + 1] += 3.0
        m1, _ = head.additive(ctx2)
        assert torch.allclose(head._side(t, o, ctx), head._side(t, o, ctx2))
        assert (m1 > m0).all()
        # in raw points, a point of value is worth the same at every position
        k = len(spec["additive_pairs"])
        head.set_additive_scale(torch.rand(k) + 0.5, torch.zeros(k), torch.rand(k) + 0.5, torch.zeros(k), [14.0, 13.0])
        std = head.add_std[:, 0]
        bumps = []
        for g in (0, 8):                                  # +1 point of QB value, +1 point of kicking
            x = ctx.clone()
            x[:, n_before + g] += 1.0 / std[g]
            bumps.append(head.additive(x)[0] - head.additive(ctx)[0])
        assert torch.allclose(bumps[0], bumps[1], atol=1e-5) and (bumps[0] > 0).all()
    assert additive_spec(n_before, "pvc")["additive_kind"] == [1] * 9
