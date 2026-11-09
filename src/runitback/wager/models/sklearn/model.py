from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn.base
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ...core.predictor import MatchupModel
from ...core.schema import GameSchema
from .features import latest_team_state, matchup_features


def make_pipeline(
    estimator: sklearn.base.BaseEstimator, schema: GameSchema, columns: list[str]
) -> Pipeline:
    """Imputation + one-hot preprocessing in front of ``estimator``.

    Fitted as one object, so the exact preprocessing used in training is what
    the saved model applies when serving.
    """
    categorical = [c for c in columns if c in schema.categorical]
    numeric = [c for c in columns if c not in schema.categorical]
    transformers = [
        (
            "numeric",
            SimpleImputer(strategy="median", add_indicator=True),
            numeric,
        )
    ]
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=False,
                ),
                categorical,
            )
        )
    return Pipeline(
        [
            ("features", ColumnTransformer(transformers)),
            ("model", sklearn.base.clone(estimator)),
        ]
    )


class SklearnMatchupModel(MatchupModel):
    """Any sklearn regressor on season-to-date team averages."""

    def __init__(
        self, schema: GameSchema, pipeline: Pipeline, windows: tuple[int, ...] = ()
    ):
        self.schema = schema
        self.pipeline = pipeline
        self.windows = tuple(windows)
        self.state: pd.DataFrame | None = None

    def update_state(self, games: pd.DataFrame) -> None:
        self.state = latest_team_state(games, self.schema, self.windows)

    @property
    def teams(self) -> list:
        return [] if self.state is None else list(self.state.index)

    def predict_margin(self, matchups: pd.DataFrame) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("call update_state before predicting")
        team = self.state.reindex(matchups[self.schema.team].values)
        opp = self.state.reindex(matchups[self.schema.opp].values)
        X = matchup_features(team, opp, matchups, self.schema)
        return self.pipeline.predict(X)
