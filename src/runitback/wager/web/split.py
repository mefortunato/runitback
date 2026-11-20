"""Train an ensemble's members in separate processes, then assemble them.

``fit_transformer`` trains ``ensemble`` members one after another, each after
``seed_and_reset(member, seed + i)``, so member ``i`` depends only on its own seed, never on the
members trained before it. :func:`fit_member` trains one of them (``seed=i, ensemble=1``) and
keeps its best checkpoint; :func:`assemble` calls the same ``fit_transformer`` with the members'
checkpoints in place of training, so the ensemble, the validation predictions and the calibration
(``finish_fit``) are made by the same code as an in-process fit. Train predictions runs each
member in its own job (``.github/workflows/train-predictions.yml``).

Lives under ``web/`` (not hashed by :func:`runitback.wager.web.build.code_hash`), so the cached
backtest seasons stay valid; nothing here changes how a model trains.
"""

from __future__ import annotations

import pathlib
import shutil
import types

from ..core.io import Table
from ..models.lightning import default_trainer
from ..models.transformer import fit_transformer


def fit_member(
    games: Table,
    sport: str,
    seed: int,
    out: str | pathlib.Path,
    max_epochs: int = 100,
    **fit_kwargs,
) -> pathlib.Path:
    """Train the ensemble member with seed ``seed`` (the one an in-process fit with the same
    arguments trains ``seed`` places after its first) and copy its best checkpoint to
    ``out/member<seed>.ckpt``. ``fit_kwargs`` go to ``fit_transformer`` (the season split, and
    ``model=`` for a non-default model), as the in-process fit would get them."""
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    trainers = []

    def trainer():
        t = default_trainer(
            max_epochs=max_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainers.append(t)
        return t

    fit_transformer(
        games,
        sport,
        trainer=trainer,
        ensemble=1,
        seed=seed,
        print_metrics=False,
        **fit_kwargs,
    )
    best = trainers[-1].checkpoint_callback.best_model_path
    if not best:
        raise RuntimeError(f"{sport} member {seed}: training left no checkpoint")
    target = out / f"member{seed}.ckpt"
    shutil.copyfile(best, target)
    return target


def assemble(
    games: Table,
    sport: str,
    checkpoints: list[str | pathlib.Path],
    **fit_kwargs,
) -> dict:
    """The fit an in-process ``fit_transformer(..., ensemble=len(checkpoints))`` returns
    (``{"predictor", "metrics", "split", "model"}``), with member ``i`` loaded from
    ``checkpoints[i]`` instead of trained. ``fit_kwargs`` are the in-process fit's, except
    ``trainer``, ``ensemble`` and ``seed``."""
    for key in ("trainer", "ensemble", "seed"):
        if key in fit_kwargs:
            raise TypeError(f"assemble() takes no {key!r}: the checkpoints are the members")
    stubs = iter(
        types.SimpleNamespace(
            fit=lambda *args, **kwargs: None,
            logger=None,
            checkpoint_callback=types.SimpleNamespace(best_model_path=str(path)),
        )
        for path in checkpoints
    )
    return fit_transformer(
        games,
        sport,
        trainer=lambda: next(stubs),
        ensemble=len(checkpoints),
        seed=0,
        **fit_kwargs,
    )
