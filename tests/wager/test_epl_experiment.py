import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from runitback.wager import experiments
from runitback.wager.core.metrics import three_way_scores
from runitback.wager.models.transformer import default_model

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "epl" / "wager.yaml"
CONFIGS = yaml.safe_load(EXPERIMENT.read_text())


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    return module


matrix_script, arm_script, report_script = (
    script("epl_wager_matrix"),
    script("wager_arm"),
    script("epl_report"),
)


# ------------------------------------------------------------------ the configuration
def test_every_stage_names_configurations_of_the_file():
    configs = set(CONFIGS["configs"])
    for stage, spec in CONFIGS["stages"].items():
        assert set(spec["configs"]) <= configs, stage
    screen = CONFIGS["stages"]["screen"]
    assert screen["seasons"] == [
        2017,
        2018,
        2019,
        2020,
        2021,
    ]  # before the first test season
    assert min(CONFIGS["stages"]["final"]["seasons"]) == 2022
    # the arms are chosen on the screen, never on the test seasons
    assert not set(screen["seasons"]) & set(CONFIGS["stages"]["baselines"]["seasons"])


def transformer_model(name):
    """The model a transformer configuration builds, as ``transformer_fitter`` builds it."""
    config = CONFIGS["configs"][name]
    fit = config.get("fit") or {}
    return default_model(
        "epl",
        None,
        fit.get("opponent_strength"),
        fit.get("elo_margin"),
        **(config.get("model") or {}),
    )


TRANSFORMERS = [n for n, c in CONFIGS["configs"].items() if "family" not in c]


@pytest.mark.parametrize("name", TRANSFORMERS)
def test_every_transformer_arm_builds_a_model(name):
    model = transformer_model(name)
    assert (
        model.distribution and len(model.margin_dist.values) == 21
    )  # goal differences -10..10


def test_the_arms_differ_from_tf_by_the_token_columns_they_add():
    base = transformer_model("tf").encoder.token_dim
    extra = {n: transformer_model(n).encoder.token_dim - base for n in TRANSFORMERS}
    assert extra["tf"] == extra["tf_s1"] == extra["tf_h19"] == extra["tf_rps"] == 0
    assert extra["tf_mkt"] == 9  # the draws market tokens, with the closing flag
    assert extra["tf_eff"] > 0 and extra["tf_eff2"] == extra["tf_eff"] + 4
    assert extra["tf_opp"] == 3  # Elo, the opponent's Elo and its margin average
    assert extra["tf_all"] == extra["tf_mkt"] + extra["tf_eff2"] + extra["tf_opp"]
    assert transformer_model("tf_h19").encoder.history == 19
    assert transformer_model("tf_h57").encoder.history == 57
    assert (
        transformer_model("tf_rps").rps_weight == 0.5
        and transformer_model("tf").rps_weight == 0
    )


def test_the_seeds_of_tf_differ_only_in_the_seed():
    configs = CONFIGS["configs"]
    assert configs["tf_s1"]["model"] == configs["tf"]["model"]
    assert [configs[f"tf_s{i}"]["fit"]["seed"] for i in (1, 2, 3)] == [1, 2, 3]


# ------------------------------------------------------------------------- the matrix
def test_matrix_of_each_stage():
    screen = matrix_script.matrix(CONFIGS, "screen")["include"]
    assert len(screen) == 15 * 5 and screen[0] == {"config": "home", "season": 2017}
    baselines = matrix_script.matrix(CONFIGS, "baselines")["include"]
    assert len(baselines) == 8 * 5 and {j["season"] for j in baselines} == {
        2022,
        2023,
        2024,
        2025,
        2026,
    }
    final = matrix_script.matrix(CONFIGS, "final", "elo, tf_mkt_e4")["include"]
    assert [j["config"] for j in final][::5] == ["elo", "tf_mkt_e4"]


def test_matrix_mistakes_stop_before_the_jobs_start():
    with pytest.raises(ValueError, match="pass --arms"):
        matrix_script.matrix(CONFIGS, "final")
    with pytest.raises(ValueError, match="not configurations"):
        matrix_script.matrix(CONFIGS, "final", "tf_mkt_e4,tf_typo_e4")
    with pytest.raises(ValueError, match="twice"):
        matrix_script.matrix(CONFIGS, "final", "elo,elo")
    with pytest.raises(ValueError, match="unknown stage"):
        matrix_script.matrix(CONFIGS, "later")


def test_e4_is_the_same_configuration_averaging_four_seeds():
    configs = CONFIGS["configs"]
    resolved = arm_script.resolve(configs, "tf_opp_e4")
    assert resolved["fit"] == {"opponent_strength": True, "ensemble": 4}
    assert resolved["model"] == configs["tf_opp"]["model"]
    assert arm_script.resolve(configs, "tf_e4")["fit"] == {"ensemble": 4}
    assert (
        arm_script.resolve(configs, "nope") is None
        and arm_script.resolve(configs, "nope_e4") is None
    )


# ------------------------------------------------------------------- family arms in a run
def test_a_configuration_can_be_a_model_family(tmp_path, toy_sport):
    import dataclasses

    from runitback.wager.core.sport import register_sport

    from .football import DRAWS_SCHEMA, football_games

    register_sport(
        dataclasses.replace(toy_sport, name="toy_family", schema=DRAWS_SCHEMA)
    )
    path = tmp_path / "games.csv"
    football_games().to_csv(path, index=False)
    out = tmp_path / "elo" / "2023.csv"
    # a transformer is the default; a `family` names another model
    experiments._run_job(
        str(path), "toy_family", "elo", {"family": "elo"}, 2023, str(out), 1
    )
    result = pd.read_csv(out)
    assert {"win_prob", "draw_prob", "market_prob"} <= set(result) and result.season.eq(
        2023
    ).all()
    with pytest.raises(ValueError, match="unknown model family"):
        experiments._run_job(
            str(path),
            "toy_family",
            "x",
            {"family": "svm"},
            2023,
            str(tmp_path / "x.csv"),
            1,
        )


# ------------------------------------------------------------------------- the RPS loss
def test_the_auxiliary_rps_is_the_rps_of_the_distributions_win_draw_loss(toy_sport):
    import dataclasses

    from runitback.wager.core.sport import register_sport

    from .football import DRAWS_SCHEMA

    register_sport(dataclasses.replace(toy_sport, name="toy_rps", schema=DRAWS_SCHEMA))
    model = default_model(
        "toy_rps", distribution=True, margin_range=(-6, 6), rps_weight=0.5
    )
    assert model.rps_weight == 0.5 and model.hparams["rps_weight"] == 0.5
    assert default_model("toy_rps", distribution=True).rps_weight == 0
    loc = np.array([0.8, -0.3, 0.0, 1.5])
    log_prob = model.margin_dist.log_prob(
        torch_tensor(loc), torch_tensor(np.log([1.2, 1.0, 1.6, 0.9]))
    )
    margin = np.array([2.0, 0.0, -1.0, np.nan])
    got = float(model.ranked_probability_score(log_prob, torch_tensor(margin)))
    pmf = log_prob.exp().detach().numpy()
    values = model.margin_dist.values.numpy()
    win, draw = pmf[:, values > 0].sum(1), pmf[:, values == 0].sum(1)
    want = three_way_scores(margin[:3], win[:3], draw[:3])[
        "rps"
    ].mean()  # the game without a margin is left out
    assert got == pytest.approx(want, rel=1e-5)
    assert 0 < got < 1


def torch_tensor(values):
    import torch

    return torch.as_tensor(np.asarray(values, dtype=np.float32))


# ------------------------------------------------------------------------- the report
def synthetic_results(seed, strength, seasons, games=120):
    """Walk-forward results in the repo's format for the EPL schema; ``strength`` > 0 is a better model."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        truth = rng.dirichlet([4.5, 2.7, 3.2], games)
        market = np.clip(truth + rng.normal(0, 0.01, truth.shape), 0.02, None)
        market /= market.sum(1, keepdims=True)
        model = np.clip(
            truth + rng.normal(0, 0.08 / (1 + strength), truth.shape), 0.02, None
        )
        model /= model.sum(1, keepdims=True)
        outcome = np.array([rng.choice(3, p=p) for p in truth])
        for i in range(games):
            rows.append(
                {"season": season, "game_id": f"{season}_{i}", "location": 1, "day": i // 10, "team": f"A{i % 7}",
                 "opp_team": f"B{i % 5}", "result": [1.0, 0.0, -1.0][outcome[i]], "pred_margin": model[i, 0] - model[i, 2],
                 "win_prob": model[i, 0], "draw_prob": model[i, 1], "market_margin": 0.0,
                 "market_prob": market[i, 0], "market_draw_prob": market[i, 1],
                 "book_prob": market[i, 0], "book_draw_prob": market[i, 1]}
            )  # fmt: skip
    return pd.DataFrame(rows)


def write(root, quality, seasons):
    for i, (name, strength) in enumerate(quality.items()):
        frame = synthetic_results(
            7, strength, seasons
        )  # the same games, a better or worse model
        rng = np.random.default_rng(100 + i)
        p = frame[["win_prob", "draw_prob"]].to_numpy()
        frame[["win_prob", "draw_prob"]] = np.clip(
            p + rng.normal(0, 0.06 / (1 + strength), p.shape), 0.02, 0.9
        )
        (root / name).mkdir(parents=True)
        for season in seasons:
            frame[frame.season == season].to_csv(
                root / name / f"{season}.csv", index=False
            )


SCREEN_SEASONS = [2017, 2018, 2019]
QUALITY = {"home": 0, "elo": 2, "sklearn": 1, "gnn": 0.5, "tf": 3, "tf_mkt": 6, "tf_eff": 4, "tf_opp": 3.5,
           "tf_s1": 3, "tf_h19": 2.5}  # fmt: skip


def test_the_screen_picks_the_reference_and_the_two_best_transformer_variants(tmp_path):
    write(tmp_path, QUALITY, SCREEN_SEASONS)
    results = experiments.load_results(str(tmp_path), list(QUALITY), SCREEN_SEASONS)
    text, pick = report_script.screen_report(results)
    assert pick["reference"] in (
        "elo",
        "sklearn",
        "gnn",
        "tf",
    )  # never the market, home or a variant
    assert pick["reference"] == "tf"  # the best of the four by construction
    assert pick["top"] == ["tf_mkt", "tf_eff"]  # not the seeds of tf, not the reference
    assert pick["arms"] == "tf_e4,tf_mkt_e4,tf_eff_e4"
    assert (
        "Final stage `arms` input: `tf_e4,tf_mkt_e4,tf_eff_e4`" in text
        and "vegas" in text
    )


def test_a_non_transformer_reference_is_run_once_and_the_default_transformer_competes(
    tmp_path,
):
    quality = {**QUALITY, "elo": 9}
    write(tmp_path, quality, SCREEN_SEASONS)
    results = experiments.load_results(str(tmp_path), list(quality), SCREEN_SEASONS)
    pick = report_script.screen_report(results)[1]
    assert pick["reference"] == "elo" and pick["arms"].split(",")[0] == "elo"
    assert pick["top"][0] == "tf_mkt" and "tf" in pick["top"] + ["tf"]


def test_incomplete_configurations_are_left_out(tmp_path, capsys):
    write(tmp_path, {"elo": 2, "tf": 3}, SCREEN_SEASONS)
    (tmp_path / "tf" / "2019.csv").unlink()
    assert report_script.complete(tmp_path, SCREEN_SEASONS) == ["elo"]
    assert "left out" in capsys.readouterr().err


def test_noise_floor_is_the_spread_of_the_seeds(tmp_path):
    write(
        tmp_path,
        {"tf": 3, "tf_s1": 3, "tf_s2": 2.5, "tf_s3": 3.5, "elo": 2},
        SCREEN_SEASONS,
    )
    results = experiments.load_results(
        str(tmp_path), ["tf", "tf_s1", "tf_s2", "tf_s3", "elo"], SCREEN_SEASONS
    )
    text = report_script.noise_floor(results)
    assert (
        "4 seeds of `tf`" in text
        and "standard deviation across seeds" in text
        and "tf_s1 vs tf_s3" in text
    )
    assert "elo" not in text.split("Mean RPS by seed")[1].split("\n")[0]
    assert "needs two" in report_script.noise_floor({"tf": results["tf"]})


def test_final_report_gives_a_verdict_against_the_reference(tmp_path):
    write(tmp_path, {"elo": 2, "tf_mkt_e4": 8, "tf_eff_e4": 0.2}, SCREEN_SEASONS)
    results = experiments.load_results(
        str(tmp_path), ["elo", "tf_mkt_e4", "tf_eff_e4"], SCREEN_SEASONS
    )
    games = pd.DataFrame(
        {
            "season": 2017,
            "team": "A0",
            "game_id": [f"2017_{i}" for i in range(6)],
            "day": range(6),
            "promoted": 1,
        }
    )
    text = report_script.final_report(results, "elo", games)
    assert (
        "better: the whole CI is below 0" in text.split("`tf_mkt_e4`")[1].split("\n")[0]
    )
    assert (
        "worse: the whole CI is above 0" in text.split("`tf_eff_e4`")[1].split("\n")[0]
    )
    assert "promoted club's first 5 games" in text and "vegas" in text


def test_the_report_runs_from_the_command_line(tmp_path, monkeypatch, capsys):
    write(tmp_path / "runs", QUALITY, SCREEN_SEASONS)
    monkeypatch.setattr(sys, "argv", ["epl_report", "screen", str(tmp_path / "runs"), "--out", str(tmp_path / "r.md"),
                                      "--seasons", *map(str, SCREEN_SEASONS)])  # fmt: skip
    report_script.main()
    assert (
        "Reference arm" in (tmp_path / "r.md").read_text()
        and "Reference arm" in capsys.readouterr().out
    )
