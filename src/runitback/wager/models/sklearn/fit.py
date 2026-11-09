from __future__ import annotations

import sklearn.base
import sklearn.ensemble

from ...core.evaluation import calibrate_and_evaluate
from ...core.io import Table, read_table
from ...core.metrics import format_metrics
from ...core.predictor import Predictor, prepare_save_dir, save_predictor
from ...core.schema import played
from ...core.splits import split_seasons
from ...core.sport import get_sport
from .features import training_frame
from .model import SklearnMatchupModel, make_pipeline


def fit_sklearn(
    games: Table,
    sport: str,
    model: sklearn.base.BaseEstimator | None = None,
    save_dir: str | None = None,
    windows: list[int] | None = None,
    context: list[str] | None = None,
    categorical: list[str] | None = None,
    train_seasons: list[int] | None = None,
    val_seasons: list[int] | None = None,
    test_seasons: list[int] | None = None,
    min_games: int = 1,
    refit: bool = False,
    overwrite: bool = False,
    print_metrics: bool = True,
) -> dict:
    """Fit an sklearn regressor on season-to-date averages and save it.

    Args:
        games: Processed games table.
        sport: Sport name, e.g. ``"nfl"``.
        model: Regressor; defaults to a random forest.
        save_dir: Where to save the MLflow model (skipped if None).
        windows: Also use averages of each team's last ``w`` games.
        context / categorical: Override the sport's pre-game feature columns.
        train_seasons / val_seasons / test_seasons: Season split; defaults to
            latest season = test, previous = val (win-probability calibration),
            the rest = train.
        min_games: Only train on games where both teams have played this many
            games that season.
        refit: After evaluating, refit on every season before saving.
        overwrite: Replace an existing ``save_dir``.

    Returns:
        ``{"predictor", "metrics", "split"}``
    """
    prepare_save_dir(save_dir, overwrite)
    schema = get_sport(sport).schema.with_features(context, categorical)
    windows = tuple(windows or ())
    if model is None:
        model = sklearn.ensemble.RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            max_features=0.3,
            n_jobs=-1,
            random_state=0,
        )

    games = played(read_table(games), schema)
    split = split_seasons(
        games[schema.season].unique(), train_seasons, val_seasons, test_seasons
    )
    X = training_frame(games, schema, windows)
    usable = (
        (X["team_n_games"] >= min_games)
        & (X["opp_n_games"] >= min_games)
        & games[schema.season].isin(split.all)
    )
    X, rows = X[usable], games[usable]
    y = rows[schema.target]
    in_train = rows[schema.season].isin(split.train)

    pipeline = make_pipeline(model, schema, list(X.columns))
    pipeline.fit(X[in_train], y[in_train])
    sigma, metrics, _ = calibrate_and_evaluate(rows, pipeline.predict(X), schema, split)
    if print_metrics:
        print(format_metrics(metrics))

    if refit:
        pipeline.fit(X, y)

    matchup_model = SklearnMatchupModel(schema, pipeline, windows)
    matchup_model.update_state(games)
    predictor = Predictor(matchup_model, sigma=sigma, sport=sport, metrics=metrics)
    if save_dir is not None:
        save_predictor(predictor, save_dir)
    return {"predictor": predictor, "metrics": metrics, "split": split}
