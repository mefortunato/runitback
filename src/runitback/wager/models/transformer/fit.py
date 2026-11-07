from __future__ import annotations

import copy
from collections.abc import Callable

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch

from ...core.evaluation import finish_fit
from ...core.io import Table, read_table
from ...core.predictor import prepare_save_dir
from ...core.schema import GameSchema, played
from ...core.splits import split_seasons
from ...core.sport import get_sport
from ..lightning import (
    best_model,
    default_trainer,
    output_targets,
    reset_parameters,
    seed_and_reset,
)
from .data import (
    SequenceData,
    SequenceFeatures,
    base_token_columns,
    extra_columns,
    additive_spec,
    lineup_context_columns,
    pbp_fold_path,
)
from .model import (
    PairHead,
    SequenceEncoder,
    TransformerEnsemble,
    TransformerMatchupModel,
    TransformerModel,
)

ENCODER_KEYS = (
    "history", "d_model", "nhead", "num_layers", "dropout",
    "season_embedding", "season_decay", "max_seasons_ago", "decay_init",
)  # fmt: skip
# Token options: which derived columns each history token carries.
TOKEN_KEYS = (
    "opponent_strength", "elo_margin", "market_tokens", "efficiency",
    "market_rating",
)  # fmt: skip
# Play-by-play embedding options (not booleans): ``pbp`` is the embedding table's path with a
# ``{season}`` placeholder for the walk-forward fold (the test season), ``pbp_dim`` its width and
# ``pbp_only`` drops the box-score stats from the tokens (runitback.nfl.embed).
PBP_KEYS = ("pbp", "pbp_dim", "pbp_only", "lineup", "lineup_set")
PBP_DEFAULTS = {"pbp": None, "pbp_dim": 32, "pbp_only": False, "lineup": None, "lineup_set": "lu"}
# Settings that are not model init args.
FIT_KEYS = ("ensemble", "batch_size", "patience", "max_epochs")


def transformer_dims(
    schema: GameSchema,
    opponent_strength: bool = False,
    elo_margin: bool = False,
    market_tokens: bool = False,
    efficiency: bool = False,
    sport: str | None = None,
    market_rating: bool = False,
    pbp: str | None = None,
    pbp_dim: int = 0,
    pbp_only: bool = False,
    lineup: str | None = None,
    lineup_set: str | None = None,
) -> dict[str, int]:
    """Token / context feature counts a TransformerModel needs for ``schema``."""
    location = [schema.location] if schema.location else []
    tokens = base_token_columns(schema, pbp_only)
    tokens += extra_columns(
        opponent_strength, elo_margin, market_tokens, efficiency, sport, market_rating,
        pbp_dim if pbp else 0, pbp_only,
    )
    return {
        "token_dim": len(tokens),
        "context_dim": len(schema.context) + len(location)
        + len(lineup_context_columns(lineup_set if lineup else None)),
    }


def sport_opponent_strength(sport: str, opponent_strength: bool | None) -> bool:
    """``opponent_strength``, or the sport's ``Sport.transformer`` setting."""
    return token_options(sport, opponent_strength=opponent_strength)[
        "opponent_strength"
    ]


def token_options(sport: str, model=None, **given) -> dict[str, bool]:
    """Which derived token columns to use: each of :data:`TOKEN_KEYS` from
    ``given`` where not None, else from ``model`` (a TransformerModel built
    with them), else the sport's ``Sport.transformer`` setting (default
    off)."""
    settings = get_sport(sport).transformer
    built = getattr(model, "token_features", None) or {}
    out = {}
    for key in TOKEN_KEYS:
        value = given.get(key)
        if value is None:
            value = built.get(key, settings.get(key, False))
        out[key] = bool(value)
    for key in PBP_KEYS:
        value = given.get(key)
        if value is None:
            value = built.get(key, settings.get(key, PBP_DEFAULTS[key]))
        out[key] = value
    out["pbp_dim"] = int(out["pbp_dim"] or 0)
    out["pbp_only"] = bool(out["pbp_only"])
    return out


def fit_settings(sport: str) -> dict:
    """The sport's ``Sport.transformer`` settings that are not model args
    (``ensemble``, ``batch_size``, ...)."""
    return {k: v for k, v in get_sport(sport).transformer.items() if k in FIT_KEYS}


def transformer_config(
    sport: str,
    context: list[str] | None = None,
    opponent_strength: bool | None = None,
    elo_margin: bool | None = None,
    total_weight: float | None = None,
    stats_weight: float | None = None,
    **overrides,
) -> dict:
    """TransformerModel init args for ``sport``: sizes from
    ``Sport.transformer``, then ``overrides`` (any encoder arg, e.g.
    ``history=34, season_decay=True``, ``lr``, or a token option).

    ``total_weight`` / ``stats_weight`` add the total and every box-score
    stat as extra outputs with those loss weights (see
    :func:`~runitback.wager.models.lightning.output_targets`); None takes the
    sport's ``Sport.transformer`` setting (default 0, off). With
    ``distribution=True`` (an override or ``Sport.transformer`` key) the
    model predicts a :class:`~runitback.wager.models.lightning.ScoreDistribution`
    of the margin (and the total) instead of point values.
    ``antisymmetric=True`` makes the head antisymmetric
    (:class:`PairHead`). Token options (:data:`TOKEN_KEYS`:
    ``opponent_strength``, ``elo_margin``, ``market_tokens``,
    ``efficiency``, ``market_rating``) None or absent take the sport's
    setting (default off).
    """
    spec = get_sport(sport)
    tokens = token_options(
        sport,
        opponent_strength=opponent_strength,
        elo_margin=elo_margin,
        **{k: overrides.pop(k) for k in (*TOKEN_KEYS, *PBP_KEYS) if k in overrides},
    )
    dims = transformer_dims(spec.schema.with_features(context), **tokens, sport=sport)
    settings = {
        "history": 20,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.1,
        **spec.transformer,
        **overrides,
    }
    if total_weight is None:
        total_weight = settings.get("total_weight", 0.0)
    if stats_weight is None:
        stats_weight = settings.get("stats_weight", 0.0)
    distribution = bool(settings.get("distribution", False))
    antisymmetric = bool(settings.get("antisymmetric", False))
    output_keys = (
        "total_weight", "stats_weight", "distribution", "antisymmetric",
        *TOKEN_KEYS, *PBP_KEYS, *FIT_KEYS,
    )  # fmt: skip
    settings = {k: v for k, v in settings.items() if k not in output_keys}
    extra, weights = output_targets(spec.schema, total_weight, stats_weight)
    outputs = 1 + len(extra)
    if distribution:
        # every output but the margin's location: log-scales and the total
        symmetric = tuple(range(1, 2 * outputs))
    else:
        symmetric = (1,) if spec.schema.total in extra[:1] else ()
    encoder = {k: v for k, v in settings.items() if k in ENCODER_KEYS}
    head = {
        "in_dim": encoder["d_model"],
        "context_dim": dims["context_dim"],
        "hidden_dim": settings.pop("hidden_dim", encoder["d_model"]),
        "dropout": encoder["dropout"],
        "out_dim": 2 * outputs if distribution else outputs,
    }
    if antisymmetric:
        head |= {"antisymmetric": True, "symmetric_outputs": symmetric}
    if tokens.get("lineup"):
        sch = spec.schema.with_features(context)
        n_before = len(sch.context) + (1 if sch.location else 0)
        head |= additive_spec(n_before, tokens.get("lineup_set"))
    return {
        "encoder": {"token_dim": dims["token_dim"], **encoder},
        "head": head,
        "extra_targets": extra,
        "extra_weights": weights,
        "distribution": distribution,
        "token_features": tokens,
        **{k: v for k, v in settings.items() if k not in ENCODER_KEYS},
    }


def default_model(
    sport: str,
    context: list[str] | None = None,
    opponent_strength: bool | None = None,
    elo_margin: bool | None = None,
    total_weight: float | None = None,
    stats_weight: float | None = None,
    **overrides,
) -> TransformerModel:
    config = transformer_config(
        sport,
        context,
        opponent_strength,
        elo_margin,
        total_weight,
        stats_weight,
        **overrides,
    )
    return TransformerModel(
        encoder=SequenceEncoder(**config.pop("encoder")),
        head=PairHead(**config.pop("head")),
        **config,
    )


def _loader(data: SequenceData, batch_size: int, shuffle: bool, num_workers: int):
    sampler = torch.utils.data.BatchSampler(
        torch.utils.data.RandomSampler(data)
        if shuffle
        else torch.utils.data.SequentialSampler(data),
        batch_size=batch_size,
        drop_last=False,
    )
    return torch.utils.data.DataLoader(
        data,
        sampler=sampler,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def fit_transformer(
    games: Table,
    sport: str,
    model: TransformerModel | None = None,
    trainer: pl.Trainer | Callable[[], pl.Trainer] | None = None,
    save_dir: str | None = None,
    context: list[str] | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
    train_seasons: list[int] | None = None,
    val_seasons: list[int] | None = None,
    test_seasons: list[int] | None = None,
    min_history: int = 1,
    opponent_strength: bool | None = None,
    elo_margin: bool | None = None,
    market_tokens: bool | None = None,
    efficiency: bool | None = None,
    market_rating: bool | None = None,
    total_weight: float | None = None,
    stats_weight: float | None = None,
    ensemble: int | None = None,
    seed: int | None = 0,
    overwrite: bool = False,
    print_metrics: bool = True,
    pbp_finetune: dict | None = None,
) -> dict:
    """Train a transformer over each team's recent games and save it.

    Args:
        games: Processed games table.
        sport: Sport name, e.g. ``"nfl"``.
        model: Untrained :class:`TransformerModel`; defaults to
            :func:`default_model`. Its sequence length is the encoder's
            ``history``.
        trainer: Lightning trainer, or a function returning a fresh one
            (each ensemble member needs its own); defaults to
            ``default_trainer()``. Members after the first get a
            ``default_trainer`` with this trainer's ``max_epochs`` when a
            trainer instance is given.
        context: Override the sport's numeric context columns.
        batch_size: Training batch size (None: the sport's ``batch_size``
            setting, else 256).
        train_seasons / val_seasons / test_seasons: Season split; val seasons
            drive early stopping and win-probability calibration.
        min_history: Only use games where both teams have this many earlier
            games (any season).
        opponent_strength: Add each history game's pre-game Elo of both teams
            and the opponent's season-to-date margin to its token. None
            takes the model's setting, else the sport's ``Sport.transformer``
            setting: off for the NFL, where on 2021-2025 it did not change
            the backtest Brier (+0.0002 +/- 0.0027) and makes state refreshes
            slower; on for college football, where schedule strength varies
            far more.
        elo_margin: Add each history game's margin minus the margin pre-game
            Elo expected, so a result is read against the opponent's rating.
        market_tokens: Add each history game's closing lines and how the
            game went against them (cover margin, win against the no-vig
            moneyline probability, total against the over/under).
        efficiency: Add each history game's rate stats from the sport's
            ``Sport.efficiency`` (EPA per play, four factors, Corsi, ...).
        market_rating: Add the market's running rating of both teams of
            each history game, fit to every closing line so far
            (:func:`~.data.market_ratings`): the latest token then says
            what the market thinks of the team now.
        total_weight / stats_weight: Also regress the total (served as
            ``pred_total``) and every box-score stat, with these loss
            weights (None: the sport's ``Sport.transformer`` setting). Used
            to build the default model; ignored with ``model``.
        ensemble: Train this many models (seeds ``seed``, ``seed + 1``, ...)
            and serve their average (None: the sport's ``ensemble``
            setting, else 1).
        seed: Seed for weight initialization and batch order (None: random).
        pbp_finetune: After fitting, fine-tune the play-by-play foundation model's top
            blocks together with the wager model (needs the ``pbp`` token option; settings
            in :mod:`.pbp_finetune`).

    Returns:
        ``{"predictor", "metrics", "split", "model"}``
    """
    prepare_save_dir(save_dir, overwrite)
    settings = fit_settings(sport)
    ensemble = int(ensemble or settings.get("ensemble", 1))
    batch_size = int(batch_size or settings.get("batch_size", 256))
    tokens = token_options(
        sport,
        model,
        opponent_strength=opponent_strength,
        elo_margin=elo_margin,
        market_tokens=market_tokens,
        efficiency=efficiency,
        market_rating=market_rating,
    )
    schema = get_sport(sport).schema.with_features(context, categorical=[])
    games = played(read_table(games), schema).reset_index(drop=True)
    split = split_seasons(
        games[schema.season].unique(), train_seasons, val_seasons, test_seasons
    )
    if not split.val:
        raise ValueError("transformer training needs validation seasons")

    model = model or default_model(
        sport,
        context,
        total_weight=total_weight,
        stats_weight=stats_weight,
        **tokens,
    )
    template = copy.deepcopy(model)
    train_games = games[games[schema.season].isin(split.train)]
    fold = max([*split.train, *split.val]) + 1
    if tokens.get("pbp"):
        # the foundation model of the season after the training data (never trained on it)
        tokens = tokens | {"pbp": pbp_fold_path(tokens["pbp"], fold)}
    if tokens.get("lineup"):
        tokens = tokens | {"lineup": pbp_fold_path(tokens["lineup"], fold)}
    features = SequenceFeatures.fit(
        train_games, schema, model.encoder.history, **tokens, sport=sport
    )
    _check_dims(model, features)
    extra = tuple(model.extra_targets)
    data = SequenceData(games, schema, features, extra_targets=extra)

    enough = np.ones(len(games), dtype=bool)
    if min_history > 0:
        enough &= (data.history[:, min_history - 1] >= 0).numpy()
        enough &= (data.opp_history[:, min_history - 1] >= 0).numpy()
    season = games[schema.season].to_numpy()
    rows = {
        name: np.flatnonzero(np.isin(season, seasons) & enough)
        for name in ("train", "val", "test")
        if (seasons := getattr(split, name))
    }

    y = games[[schema.target, *extra]].apply(pd.to_numeric, errors="coerce")
    y = y.iloc[rows["train"]]
    if extra:
        std = y.std().replace(0, 1).fillna(1.0)
        mean = y.mean().fillna(0.0).to_numpy().copy()
        if getattr(model.head, "antisymmetric", False):
            mean[0] = 0.0  # an antisymmetric margin is centered on 0
        scaling = (mean, std.to_numpy())
    else:
        y = y[schema.target].to_numpy(dtype=float)
        antisymmetric = getattr(model.head, "antisymmetric", False)
        mean = 0.0 if antisymmetric else float(y.mean())
        scaling = (mean, float(y.std()))

    def new_trainer(i: int) -> pl.Trainer:
        if callable(trainer) and not isinstance(trainer, pl.Trainer):
            return trainer()
        if trainer is not None and i == 0:
            return trainer
        if trainer is not None:
            return default_trainer(
                max_epochs=trainer.max_epochs,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
        return default_trainer()

    members = []
    for i in range(ensemble):
        member = model if i == 0 else copy.deepcopy(template)
        if seed is None:
            reset_parameters(member)
        else:
            seed_and_reset(member, seed + i)
        member.set_target_scaling(*scaling)
        fit_trainer = new_trainer(i)
        fit_trainer.fit(
            member,
            train_dataloaders=_loader(
                data.with_rows(rows["train"]), batch_size, True, num_workers
            ),
            val_dataloaders=_loader(
                data.with_rows(rows["val"]), batch_size, False, num_workers
            ),
        )
        members.append(best_model(fit_trainer, member).eval())
    best = members[0] if len(members) == 1 else TransformerEnsemble(members).eval()
    if pbp_finetune:
        if not features.pbp or len(members) > 1:
            raise ValueError("pbp_finetune needs the pbp token option and a single model")
        from .pbp_finetune import finetune

        best, data, features = finetune(
            best, data, features, games, schema, rows, pbp_finetune, fold, extra
        )

    predicted = np.concatenate(list(rows.values()))
    with torch.no_grad():
        raw = np.concatenate(
            [
                best.predict_outputs(batch).numpy()
                for batch in _loader(data.with_rows(predicted), 4096, False, 0)
            ]
        )
    raw_total = None
    if schema.total and schema.total in extra:
        raw_total = raw[:, 1 + extra.index(schema.total)]
    result = finish_fit(
        TransformerMatchupModel(schema, best, features),
        games,
        games.iloc[predicted],
        raw[:, 0],
        split,
        sport,
        save_dir,
        print_metrics,
        raw_total=raw_total,
    )
    if fit_trainer.logger is not None:
        fit_trainer.logger.log_metrics(
            {
                f"final_{s}_{k}": v
                for s, m in result["metrics"].items()
                for k, v in m.items()
            }
        )
    return {**result, "model": best}


def _check_dims(model: TransformerModel, features: SequenceFeatures) -> None:
    expected = {
        "encoder token_dim": (model.encoder.token_dim, features.token_dim),
        "head context_dim": (model.head.context_dim, features.context_dim),
        "head in_dim": (model.head.in_dim, model.encoder.out_dim),
    }
    wrong = {k: v for k, v in expected.items() if v[0] != v[1]}
    if wrong:
        detail = ", ".join(
            f"{k}={got} (expected {want})" for k, (got, want) in wrong.items()
        )
        raise ValueError(f"model does not match the data: {detail}")
