"""Training an ensemble's members separately and assembling them gives the in-process ensemble
(``runitback.wager.web.split``)."""
import numpy as np
import pandas as pd
import pytest
import torch

from runitback.wager.models.lightning import default_trainer
from runitback.wager.models.transformer import fit_transformer
from runitback.wager.web import split

from .conftest import make_games


@pytest.fixture(scope="module")
def games():
    return make_games(persistence=0.7)


def _trainer():
    return default_trainer(max_epochs=2, enable_progress_bar=False, enable_model_summary=False)


def _predictions(fitted, games):
    predictor = fitted["predictor"]
    upcoming = games[games.margin.isna()] if "margin" in games else games.tail(20)
    return predictor.predict(upcoming if len(upcoming) else games.tail(20))


def test_members_trained_apart_make_the_in_process_ensemble(games, tmp_path):
    together = fit_transformer(games, "toy", trainer=_trainer, ensemble=2, seed=0, print_metrics=False)
    ckpts = [split.fit_member(games, "toy", seed, tmp_path / f"s{seed}", max_epochs=2) for seed in range(2)]
    apart = split.assemble(games, "toy", ckpts, print_metrics=False)

    # the members' weights are the same (trained in another process of the same machine)
    for a, b in zip(together["model"].members, apart["model"].members):
        for (name, x), (_, y) in zip(a.state_dict().items(), b.state_dict().items()):
            assert torch.equal(x, y), name
    # and so is everything calibrated from them
    assert together["metrics"] == apart["metrics"]
    tail = games.tail(40)
    pd.testing.assert_frame_equal(
        together["predictor"].predict(tail).reset_index(drop=True),
        apart["predictor"].predict(tail).reset_index(drop=True),
        check_exact=True,
    )


def test_assemble_takes_the_checkpoints_as_the_members(games, tmp_path):
    ckpts = [split.fit_member(games, "toy", seed, tmp_path / f"s{seed}", max_epochs=1) for seed in range(2)]
    with pytest.raises(TypeError):
        split.assemble(games, "toy", ckpts, ensemble=3)
    one = split.assemble(games, "toy", ckpts[:1], print_metrics=False)
    assert not hasattr(one["model"], "members")


def _flat(value, prefix=""):
    if isinstance(value, dict):
        return {k: v for key, item in value.items() for k, v in _flat(item, f"{prefix}{key}.").items()}
    if isinstance(value, list):
        return {k: v for i, item in enumerate(value) for k, v in _flat(item, f"{prefix}{i}.").items()}
    return {prefix: value}


@pytest.mark.parametrize("n_seasons", [1, 2])    # the main model stands in for the latest; an earlier one has its own fit
def test_the_split_pipeline_builds_what_build_builds(tmp_path, monkeypatch, n_seasons):
    """prep -> plan -> members (one call each) -> main + season -> ``build`` from the cache gives the
    same pool, rankings and metadata as one in-process ``build``."""
    from runitback.wager.core.sport import get_sport
    from runitback.wager.web.build import build

    games = make_games(seasons=(2020, 2021, 2022, 2023, 2024), persistence=0.7)
    monkeypatch.setitem(get_sport("toy").transformer, "ensemble", 2)
    kw = dict(backtest_seasons=n_seasons, max_epochs=2)

    whole = tmp_path / "whole"
    build(sports=["toy"], output_dir=str(whole), games=games, cache=str(tmp_path / "cache-whole"), **kw)

    cache, parts = str(tmp_path / "cache-split"), tmp_path / "parts"
    pkl = tmp_path / "games.pkl"
    games.to_pickle(pkl)
    split.web_plan("toy", str(pkl), out=str(tmp_path / "plan.json"), cache=cache, **kw)
    plan = __import__("json").loads((tmp_path / "plan.json").read_text())
    assert plan["fits"][0] == "main" and plan["ensemble"] == 2 and len(plan["seasons"]) == n_seasons
    assert len(plan["fits"]) == n_seasons
    for fit in plan["fits"]:
        for seed in plan["seeds"]:
            split.web_fit_member("toy", str(pkl), fit, seed, str(parts / fit), cache=cache, max_epochs=2)
    split.web_main("toy", str(pkl), str(parts / "main"), output_dir=str(tmp_path / "split"), cache=cache, max_epochs=2)
    for s in plan["seasons"]:
        from_main = s in plan["main_for"]
        split.web_season("toy", str(pkl), s, str(parts / ("main" if from_main else str(s))),
                         from_main=from_main, cache=cache, max_epochs=2)  # fmt: skip
    # a second call of a member is served from the cache
    split.web_fit_member("toy", str(pkl), "main", 0, str(parts / "again"), cache=cache, max_epochs=2)
    assert (parts / "again" / "member0.ckpt").read_bytes() == (parts / "main" / "member0.ckpt").read_bytes()

    build(sports=["toy"], output_dir=str(tmp_path / "split"), games=games, cache=cache,
          reuse_model_days=2, max_new_seasons=0, **kw)  # fmt: skip

    for name in ("pool.csv", "rankings.csv"):
        pd.testing.assert_frame_equal(
            pd.read_csv(whole / "toy" / name), pd.read_csv(tmp_path / "split" / "toy" / name),
            check_exact=False, rtol=1e-9, atol=0,
        )  # (a season read back from the cache is rounded to float32 text: 1e-16 off in a derived column)
    a, b = (__import__("json").loads((d / "toy" / "meta.json").read_text()) for d in (whole, tmp_path / "split"))
    # the summary of a season read back from the cache differs from the fresh one by float32 -> csv rounding
    # (also when a plain build reuses a cached season)
    assert a["backtest_cache"] == b["backtest_cache"] and a["model"] == b["model"]
    assert _flat(b["backtest"]) == pytest.approx(_flat(a["backtest"]), rel=1e-7)
