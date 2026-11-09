"""Optuna search over the transformer's hyperparameters.

A trial is a walk-forward backtest of one configuration over the tuning
``seasons`` (each fit on the seasons before it and replayed date by date,
exactly as the reported backtests are), scored by the mean Brier score of
its win probabilities. Seasons run in order and the running mean is
reported after each, so Optuna's median pruner stops a configuration that is
already worse than most earlier trials were at that point. Pick tuning
seasons that come before the seasons you report, so the search never sees
them.

Trials run in ``n_jobs`` processes sharing one Optuna journal file
(``<output_dir>/study.log``), so a search can be stopped and resumed.
"""

from __future__ import annotations

import json
import multiprocessing
import pathlib
import time

import numpy as np

from .core.io import read_table
from .core.sport import get_sport


def search_space(trial, sport: str) -> dict:
    """``default_model`` overrides and fit settings for one trial."""
    default_history = get_sport(sport).transformer.get("history", 20)
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    model = {
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "d_model": d_model,
        "nhead": trial.suggest_categorical("nhead", [2, 4, 8]),
        "num_layers": trial.suggest_int("num_layers", 1, 4),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
        "history": trial.suggest_int(
            "history",
            max(4, default_history // 2),
            2 * default_history,
            step=max(1, default_history // 10),
        ),
        "season_decay": trial.suggest_categorical("season_decay", [False, True]),
    }
    fit = {"batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512])}
    return {"model": model, "fit": fit}


def merge(base: dict, extra: dict) -> dict:
    """``base`` config with ``extra``'s ``model`` / ``fit`` entries on top."""
    out = {k: v for k, v in base.items() if k not in ("model", "fit")}
    for key in ("model", "fit"):
        out[key] = {**(base.get(key) or {}), **(extra.get(key) or {})}
    return out


def season_brier(results, sport: str) -> float:
    schema = get_sport(sport).schema
    y = results[schema.target].to_numpy(dtype=float)
    decided = y != 0
    prob = results["win_prob"].to_numpy(dtype=float)[decided]
    return float(np.mean((prob - (y[decided] > 0)) ** 2))


def _worker(
    games_path: str,
    sport: str,
    seasons: list[int],
    base: dict,
    output_dir: str,
    n_trials: int,
    deadline: float,
    threads: int,
) -> None:
    import logging
    import warnings

    import optuna
    import torch

    from .backtest import transformer_fitter, walk_forward

    warnings.filterwarnings("ignore")
    logging.getLogger("lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    torch.set_num_threads(threads)
    games = read_table(games_path)
    study = load_study(output_dir)

    def objective(trial):
        config = merge(base, search_space(trial, sport))
        fitter = transformer_fitter(
            sport,
            max_epochs=config.get("max_epochs", 100),
            patience=config.get("patience", 10),
            model=config["model"],
            **config["fit"],
        )
        scores = []
        start = time.time()
        for step, season in enumerate(seasons):
            results = walk_forward(games, sport, fitter, [season])
            scores.append(season_brier(results, sport))
            trial.set_user_attr(f"brier_{season}", scores[-1])
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("seconds", time.time() - start)
        return float(np.mean(scores))

    def stop(study, trial):
        done = [t for t in study.trials if t.state.is_finished()]
        if time.time() > deadline or len(done) >= n_trials:
            study.stop()

    if time.time() < deadline:
        study.optimize(objective, callbacks=[stop])


def load_study(output_dir: str):
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    path = pathlib.Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(path / "study.log")))
    return optuna.create_study(
        study_name="transformer",
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=None, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1),
    )


def tune(
    games: str,
    sport: str,
    seasons: list[int],
    output_dir: str,
    base: dict | None = None,
    n_trials: int = 40,
    timeout_hours: float = 4.0,
    n_jobs: int = 4,
    threads: int = 1,
    enqueue_defaults: bool = True,
) -> dict:
    """Search hyperparameters; write ``best.json`` and ``trials.csv``.

    Args:
        games: Path of the processed games table.
        seasons: Tuning seasons (before the seasons you report).
        output_dir: Holds the study journal and the summary.
        base: Configuration every trial starts from (e.g. the token options
            and head that ablations turned on), as in
            :mod:`runitback.wager.experiments`.
        n_trials / timeout_hours: Stop after this many finished trials or
            this long, whichever comes first.
        n_jobs / threads: Worker processes and torch threads per process.
        enqueue_defaults: Make the sport's current settings the first trial,
            so the search is measured against them.

    Returns:
        ``{"best_params", "best_value", "default_value"}``
    """
    base = base or {}
    study = load_study(output_dir)
    if enqueue_defaults and not study.trials:
        spec = get_sport(sport).transformer
        study.enqueue_trial(
            {
                "lr": 3e-4,
                "weight_decay": 1e-4,
                "dropout": 0.1,
                "d_model": spec.get("d_model", 64),
                "nhead": spec.get("nhead", 4),
                "num_layers": spec.get("num_layers", 2),
                "hidden_dim": spec.get("d_model", 64),
                "history": spec.get("history", 20),
                "season_decay": False,
                "batch_size": 256,
            },
            skip_if_exists=True,
        )
    deadline = time.time() + 3600 * timeout_hours
    context = multiprocessing.get_context("spawn")
    workers = [
        context.Process(
            target=_worker,
            args=(games, sport, seasons, base, output_dir, n_trials, deadline, threads),
        )
        for _ in range(n_jobs)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    return summarize_study(output_dir)


def summarize_study(output_dir: str) -> dict:
    """Write ``trials.csv`` and ``best.json`` for the study in ``output_dir``."""
    import optuna

    study = load_study(output_dir)
    frame = study.trials_dataframe()
    path = pathlib.Path(output_dir)
    frame.to_csv(path / "trials.csv", index=False)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    summary = {"n_trials": len(study.trials), "n_complete": len(complete)}
    if complete:
        best = study.best_trial
        summary |= {
            "best_params": best.params,
            "best_value": best.value,
            "best_trial": best.number,
        }
        if complete and study.trials[0].state == optuna.trial.TrialState.COMPLETE:
            summary["default_value"] = study.trials[0].value
            summary["default_params"] = study.trials[0].params
    (path / "best.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    return summary


def params_to_config(params: dict) -> dict:
    """A trial's params as an :mod:`runitback.wager.experiments` configuration."""
    params = dict(params)
    fit = {"batch_size": params.pop("batch_size")} if "batch_size" in params else {}
    return {"model": params, "fit": fit}
