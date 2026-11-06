"""Model-family-agnostic prediction layer.

Every model family (sklearn, GNN, and future ones such as a transformer)
implements :class:`MatchupModel`. :class:`Predictor` wraps any of them with the
behavior every served model needs: antisymmetric margins, calibrated win
probabilities, a stable output format and MLflow persistence.
"""

from __future__ import annotations

import abc
import pathlib
import shutil
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

from .metrics import over_probability, win_probability
from .schema import GameSchema, mirror


class MatchupModel(abc.ABC):
    """A fitted model that predicts the point margin of ``team`` vs ``opp``.

    Implementations keep whatever per-team state they need (season averages,
    graph embeddings, sequence encodings, ...) and rebuild it from a games
    table in :meth:`update_state`, so a saved model can be refreshed with new
    results without retraining.
    """

    schema: GameSchema

    @abc.abstractmethod
    def update_state(self, games: pd.DataFrame) -> None:
        """Rebuild per-team state from the played games in ``games``."""

    @abc.abstractmethod
    def predict_margin(self, matchups: pd.DataFrame) -> np.ndarray:
        """Raw margin for each row's ``team`` vs ``opp``.

        Rows whose teams have no state should get NaN. Callers should use
        :class:`Predictor`, which also averages in the mirrored perspective.
        """

    def predict_outputs(self, matchups: pd.DataFrame) -> dict[str, np.ndarray]:
        """Raw ``margin`` and, for models that predict it, ``total``; a
        model that predicts score distributions also returns ``margin_pmf``
        over ``margin_values`` (symmetric around 0) and ``total_pmf`` over
        ``total_values``.

        Override when a model predicts more than the margin in one pass.
        """
        return {"margin": self.predict_margin(matchups)}

    @property
    @abc.abstractmethod
    def teams(self) -> list:
        """Teams that currently have state, i.e. can be predicted."""

    def team_embeddings(self) -> pd.DataFrame | None:
        """Each team's learned representation in the current state, one row
        per team (indexed by team), or ``None`` for models without one."""
        return None


class Predictor(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc model that serves any :class:`MatchupModel`.

    ``predict`` takes a DataFrame with the schema's ``team`` and ``opp``
    columns plus any pre-game columns (location, context, categorical) and
    returns it with ``pred_margin`` and ``win_prob`` appended. A model that
    predicts the total also gets ``pred_total`` and, where the matchup
    carries the market's total line, ``over_prob``; where it carries the
    market's spread, ``cover_prob`` (P(team covers)). Probabilities of
    covering and of the over leave out a push, and come from the model's
    score distribution when it predicts one (else a normal distribution
    around the prediction, with the residual spread fit at training).
    """

    def __init__(
        self,
        model: MatchupModel,
        sigma: float,
        sport: str,
        metrics: dict[str, dict[str, float]] | None = None,
        total_sigma: float | None = None,
    ):
        self.model = model
        self.sigma = float(sigma)
        self.sport = sport
        self.metrics = metrics or {}
        self.total_sigma = None if total_sigma is None else float(total_sigma)

    @property
    def schema(self) -> GameSchema:
        return self.model.schema

    def update_state(self, games: pd.DataFrame) -> None:
        self.model.update_state(games)

    def predict(self, model_input, params: dict[str, Any] | None = None):
        schema = self.schema
        matchups = pd.DataFrame(model_input).reset_index(drop=True)
        missing = [c for c in (schema.team, schema.opp) if c not in matchups]
        if missing:
            raise ValueError(f"matchups need columns {missing}")
        unknown = set(matchups[schema.team]) | set(matchups[schema.opp])
        unknown -= set(self.model.teams)
        if unknown:
            raise ValueError(f"no state for teams: {sorted(unknown, key=str)}")
        if schema.location and schema.location not in matchups:
            matchups[schema.location] = 0

        forward = self.model.predict_outputs(matchups)
        backward = self.model.predict_outputs(mirror(matchups, schema))
        if "margin_pmf" in forward:
            return self._from_distributions(matchups, forward, backward)
        margin = (np.asarray(forward["margin"]) - np.asarray(backward["margin"])) / 2
        out = matchups.assign(
            pred_margin=margin, win_prob=win_probability(margin, self.sigma)
        )
        if schema.market_spread and schema.market_spread in out:
            line = out[schema.market_spread].to_numpy(dtype=float)
            out["cover_prob"] = win_probability(margin - line, self.sigma)
        if "total" in forward:
            total = (np.asarray(forward["total"]) + np.asarray(backward["total"])) / 2
            out["pred_total"] = total
            # saved before totals existed: no total_sigma attribute
            sigma = getattr(self, "total_sigma", None)
            if sigma and schema.market_total and schema.market_total in out:
                out["over_prob"] = over_probability(
                    total, out[schema.market_total], sigma
                )
        return out

    def _from_distributions(self, matchups, forward, backward) -> pd.DataFrame:
        schema = self.schema
        values = forward["margin_values"]
        if not np.array_equal(values, -values[::-1]):
            raise ValueError("margin values must be symmetric around 0")
        # the opponent's margin distribution, mirrored onto this team's side
        pmf = (forward["margin_pmf"] + backward["margin_pmf"][:, ::-1]) / 2
        out = matchups.assign(
            pred_margin=pmf @ values, win_prob=beyond(pmf, values, 0.0)
        )
        if schema.market_spread and schema.market_spread in out:
            line = out[schema.market_spread].to_numpy(dtype=float)
            out["cover_prob"] = beyond(pmf, values, line)
        if "total_pmf" in forward:
            totals = forward["total_values"]
            total = (forward["total_pmf"] + backward["total_pmf"]) / 2
            out["pred_total"] = total @ totals
            if schema.market_total and schema.market_total in out:
                line = out[schema.market_total].to_numpy(dtype=float)
                out["over_prob"] = beyond(total, totals, line)
        return out


def beyond(pmf: np.ndarray, values: np.ndarray, line) -> np.ndarray:
    """P(outcome > line) among outcomes not on the line, per row of ``pmf``
    (``[n, len(values)]``); NaN where ``line`` is."""
    line = np.broadcast_to(np.asarray(line, dtype=float), (len(pmf),))
    above = (pmf * (values[None] > line[:, None])).sum(axis=1)
    below = (pmf * (values[None] < line[:, None])).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isnan(line), np.nan, above / (above + below))


def prepare_save_dir(path: str | None, overwrite: bool = False) -> None:
    """Fail before training, not after, if ``path`` already holds a model."""
    if path is None:
        return
    target = pathlib.Path(path)
    if target.exists() and any(target.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists; pass overwrite=True to replace it"
            )
        shutil.rmtree(target)


def save_predictor(predictor: Predictor, path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    mlflow.pyfunc.save_model(path, python_model=predictor)


def load_predictor(path: str) -> Predictor:
    """Load a saved model as a :class:`Predictor` (with update_state etc.)."""
    return mlflow.pyfunc.load_model(path).unwrap_python_model()
