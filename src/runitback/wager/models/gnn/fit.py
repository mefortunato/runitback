from __future__ import annotations

import lightning.pytorch as pl

from ...core.evaluation import calibrate_and_evaluate, calibrate_total
from ...core.io import Table, read_table
from ...core.metrics import format_metrics
from ...core.predictor import Predictor, prepare_save_dir, save_predictor
from ...core.schema import played
from ...core.splits import split_seasons
from ...core.sport import get_sport
from ..lightning import best_model, default_trainer, output_targets, seed_and_reset
from .datamodule import GraphDataModule
from .model import GraphMatchupModel, GraphModel
from .nn.encoder.edge_conv_encoder import EdgeConvEncoder
from .nn.predictor.ffn import EdgeFFN


def graph_dims(schema) -> dict[str, int]:
    """Edge / context feature counts a GraphModel needs for ``schema``."""
    location = 1 if schema.location else 0
    return {
        "edge_dim": len(schema.stats) + location,
        "context_dim": len(schema.context) + location,
    }


ENCODER_KEYS = ("hidden_dim", "out_dim", "num_layers", "dropout", "aggr")


def gnn_config(
    sport: str,
    context: list[str] | None = None,
    total_weight: float | None = None,
    stats_weight: float | None = None,
) -> dict:
    """GraphModel init args for ``sport``: EdgeConvEncoder + EdgeFFN.

    Sizes and optimizer settings come from ``Sport.gnn``; feature counts from
    the sport's schema. Used by :func:`default_model` and the CLI defaults.
    ``total_weight`` / ``stats_weight`` add the total and every box-score
    stat as extra outputs (see
    :func:`~runitback.wager.models.lightning.output_targets`); None takes the
    sport's ``Sport.gnn`` setting (default 0, off).
    """
    spec = get_sport(sport)
    dims = graph_dims(spec.schema.with_features(context))
    settings = {"hidden_dim": 128, "out_dim": 64, "num_layers": 3, **spec.gnn}
    if total_weight is None:
        total_weight = settings.get("total_weight", 0.0)
    if stats_weight is None:
        stats_weight = settings.get("stats_weight", 0.0)
    settings = {
        k: v for k, v in settings.items() if k not in ("total_weight", "stats_weight")
    }
    extra, weights = output_targets(spec.schema, total_weight, stats_weight)
    encoder = {k: v for k, v in settings.items() if k in ENCODER_KEYS}
    return {
        "encoder": {"edge_dim": dims["edge_dim"], **encoder},
        "predictor": {
            "in_dim": encoder["out_dim"],
            "hidden_dim": encoder["out_dim"],
            "context_dim": dims["context_dim"],
            "out_dim": 1 + len(extra),
        },
        "extra_targets": extra,
        "extra_weights": weights,
        **{k: v for k, v in settings.items() if k not in ENCODER_KEYS},
    }


def default_model(
    sport: str,
    context: list[str] | None = None,
    total_weight: float | None = None,
    stats_weight: float | None = None,
) -> GraphModel:
    """EdgeConvEncoder + EdgeFFN configured for ``sport`` (see ``Sport.gnn``)."""
    config = gnn_config(sport, context, total_weight, stats_weight)
    return GraphModel(
        encoder=EdgeConvEncoder(**config.pop("encoder")),
        predictor=EdgeFFN(**config.pop("predictor")),
        **config,
    )


def fit_gnn(
    games: Table,
    sport: str,
    model: GraphModel,
    trainer: pl.Trainer | None = None,
    save_dir: str | None = None,
    context: list[str] | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
    train_seasons: list[int] | None = None,
    val_seasons: list[int] | None = None,
    test_seasons: list[int] | None = None,
    min_games: int = 1,
    seed: int | None = 0,
    overwrite: bool = False,
    print_metrics: bool = True,
) -> dict:
    """Train a graph model on season snapshots and save it.

    Args:
        games: Processed games table.
        sport: Sport name, e.g. ``"nfl"``.
        model: Untrained :class:`GraphModel`. Its encoder's ``edge_dim`` must
            equal the number of stats + 1 (location) and its predictor's
            ``context_dim`` the number of context columns + 1 (location).
        trainer: Lightning trainer; defaults to :func:`default_trainer`.
        context: Override the sport's numeric context columns.
        train_seasons / val_seasons / test_seasons: Season split; val seasons
            drive early stopping and win-probability calibration.
        min_games: Only predict games where both teams have played this many
            games that season.
        seed: Seed for weight initialization and batch order (None: random).

    Returns:
        ``{"predictor", "metrics", "split", "model"}``
    """
    prepare_save_dir(save_dir, overwrite)
    schema = get_sport(sport).schema.with_features(context, categorical=[])
    games = played(read_table(games), schema)
    split = split_seasons(
        games[schema.season].unique(), train_seasons, val_seasons, test_seasons
    )
    if not split.val:
        raise ValueError("GNN training needs validation seasons (early stopping)")

    extra = tuple(model.extra_targets)
    data = GraphDataModule(
        games, schema, split, batch_size, num_workers, min_games, extra
    )
    _check_dims(model, data)
    seed_and_reset(model, seed)
    model.set_target_scaling(data.target_mean, data.target_std)

    trainer = trainer or default_trainer()
    trainer.fit(model, datamodule=data)

    best = best_model(trainer, model)
    rows, raw = data.predict_targets(best)
    raw = raw.reshape(len(rows), -1)
    sigma, metrics, _ = calibrate_and_evaluate(rows, raw[:, 0], schema, split)
    total_sigma = None
    if schema.total and schema.total in extra:
        total = raw[:, 1 + extra.index(schema.total)]
        total_sigma = calibrate_total(rows, total, schema, split, metrics)
    if print_metrics:
        print(format_metrics(metrics))
    if trainer.logger is not None:
        trainer.logger.log_metrics(
            {f"final_{s}_{k}": v for s, m in metrics.items() for k, v in m.items()}
        )

    matchup_model = GraphMatchupModel(schema, best, data.features)
    matchup_model.update_state(games)
    predictor = Predictor(
        matchup_model,
        sigma=sigma,
        sport=sport,
        metrics=metrics,
        total_sigma=total_sigma,
    )
    if save_dir is not None:
        save_predictor(predictor, save_dir)
    return {"predictor": predictor, "metrics": metrics, "split": split, "model": best}


def _check_dims(model: GraphModel, data: GraphDataModule) -> None:
    expected = {
        "encoder edge_dim": (model.encoder.edge_dim, data.features.edge_dim),
        "predictor context_dim": (
            model.predictor.context_dim,
            data.features.context_dim,
        ),
        "predictor in_dim": (model.predictor.in_dim, model.encoder.out_dim),
    }
    wrong = {k: v for k, v in expected.items() if v[0] != v[1]}
    if wrong:
        detail = ", ".join(
            f"{k}={got} (expected {want})" for k, (got, want) in wrong.items()
        )
        raise ValueError(f"model does not match the data: {detail}")
