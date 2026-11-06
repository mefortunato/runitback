import json

import pandas as pd
import pytest

from runitback.wager.cli import run

from .conftest import make_games


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_games().to_csv("raw.csv", index=False)
    return tmp_path


def test_help_works_before_any_data_exists(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit:
        run(["nfl", "fit", "sklearn", "--help"])
    assert exit.value.code == 0
    assert "--games" in capsys.readouterr().out


def test_process_fit_predict(workdir):
    run(["toy", "process", "--games", "raw.csv", "--output_file", "games.csv"])
    run(
        [
            "toy", "fit", "sklearn",
            "--games", "games.csv",
            "--model", "sklearn.linear_model.Ridge",
            "--save_dir", "models/sk",
            "--print_metrics", "false",
        ]
    )  # fmt: skip
    preds = run(
        [
            "toy", "predict",
            "--model_dir", "models/sk",
            "--games", "games.csv",
            "--output_file", "out/preds.json",
        ]
    )  # fmt: skip
    records = json.loads((workdir / "out/preds.json").read_text())
    assert len(records) == len(preds) == 6  # one row per upcoming game
    assert {"team", "opp", "pred_margin", "win_prob"} <= set(records[0])
    assert all(r["location"] >= 0 for r in records)


def test_fit_gnn_from_cli(workdir):
    run(["toy", "process", "--games", "raw.csv", "--output_file", "games.csv"])
    run(
        [
            "toy", "fit", "gnn",
            "--games", "games.csv",
            "--save_dir", "models/gnn",
            "--trainer.max_epochs", "2",
            "--trainer.accelerator", "cpu",
            "--trainer.enable_progress_bar", "false",
            "--print_metrics", "false",
        ]
    )  # fmt: skip
    matchups = pd.DataFrame({"team": ["T00"], "opp": ["T01"]})
    matchups.to_csv("matchups.csv", index=False)
    preds = run(
        [
            "toy", "predict",
            "--model_dir", "models/gnn",
            "--games", "games.csv",
            "--matchups", "matchups.csv",
        ]
    )  # fmt: skip
    assert preds.win_prob.between(0, 1).all()
    assert (workdir / "mlflow.db").exists()  # trainer default logs to MLflow
