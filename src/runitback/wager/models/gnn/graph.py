"""Games table -> PyG graph, with the encodings saved alongside a model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg

from ...core.schema import GameSchema


@dataclass
class GraphFeatures:
    """How games become graph tensors; fitted once and saved with the model.

    Attributes:
        teams: Node vocabulary; node ``i`` is ``teams[i]``.
        edge_columns: Edge features: the schema's stats plus location.
        context_columns: Pre-game features for the prediction head.
        fill_values: Median of each column on training games (imputes NaN).
        context_mean / context_std: Standardization of context features.
    """

    teams: list
    edge_columns: list[str]
    context_columns: list[str]
    fill_values: dict[str, float]
    context_mean: dict[str, float]
    context_std: dict[str, float]

    @classmethod
    def fit(
        cls, games: pd.DataFrame, train_games: pd.DataFrame, schema: GameSchema
    ) -> GraphFeatures:
        location = [schema.location] if schema.location else []
        edge_columns = list(schema.stats) + location
        context_columns = location + list(schema.context)
        columns = list(dict.fromkeys(edge_columns + list(schema.context)))
        numeric = train_games[columns].apply(pd.to_numeric, errors="coerce")
        fill = numeric.median().fillna(0.0)
        ctx = numeric[context_columns].fillna(fill)
        teams = pd.unique(pd.concat([games[schema.team], games[schema.opp]]))
        return cls(
            teams=sorted(teams.tolist(), key=str),
            edge_columns=edge_columns,
            context_columns=context_columns,
            fill_values=fill.to_dict(),
            context_mean=ctx.mean().to_dict(),
            context_std=ctx.std().replace(0, 1).fillna(1.0).to_dict(),
        )

    @property
    def edge_dim(self) -> int:
        return len(self.edge_columns)

    @property
    def context_dim(self) -> int:
        return len(self.context_columns)

    def index(self, keys) -> np.ndarray:
        lookup = {team: i for i, team in enumerate(self.teams)}
        return np.array([lookup.get(k, -1) for k in keys], dtype=np.int64)

    def _numeric(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        values = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
        return values.fillna({c: self.fill_values.get(c, 0.0) for c in columns})

    def edge_tensor(self, games: pd.DataFrame) -> torch.Tensor:
        values = self._numeric(games, self.edge_columns).to_numpy(dtype=np.float32)
        return torch.from_numpy(values)

    def context_tensor(self, matchups: pd.DataFrame) -> torch.Tensor:
        values = self._numeric(matchups, self.context_columns)
        mean = pd.Series(self.context_mean)[self.context_columns]
        std = pd.Series(self.context_std)[self.context_columns]
        values = ((values - mean) / std).to_numpy(dtype=np.float32)
        return torch.from_numpy(values.reshape(len(matchups), self.context_dim))


def prior_games(games: pd.DataFrame, schema: GameSchema) -> pd.DataFrame:
    """Games each row's team and opponent played earlier in the season."""
    ordered = games.sort_values([schema.season, schema.date])
    team = ordered.groupby([schema.season, schema.team]).cumcount().reindex(games.index)
    by_game = pd.Series(team.values, index=[games[schema.game_id], games[schema.team]])
    opp = by_game.reindex(pd.MultiIndex.from_frame(games[[schema.game_id, schema.opp]]))
    return pd.DataFrame({"team": team.values, "opp": opp.values}, index=games.index)


def build_graph(
    games: pd.DataFrame,
    schema: GameSchema,
    features: GraphFeatures,
    min_games: int = 1,
    extra_targets: tuple[str, ...] = (),
) -> pyg.data.Data:
    """One edge per played row: opponent -> team, carrying the team's stats.

    ``y`` is the margin, or ``[margin, *extra_targets]`` per edge.

    Edge-level tensors: ``edge_attr``, ``context``, ``y``, ``season``,
    ``date``, ``game`` (index into ``graph.game_ids``) and ``valid`` (both
    teams had ``min_games`` earlier games that season, so the edge can be a
    training target).
    """
    team = features.index(games[schema.team])
    opp = features.index(games[schema.opp])
    if (team < 0).any() or (opp < 0).any():
        raise ValueError("games contain teams missing from the vocabulary")
    prior = prior_games(games, schema)
    game_codes, game_ids = pd.factorize(games[schema.game_id])

    graph = pyg.data.Data(
        edge_index=torch.from_numpy(np.stack([opp, team])),
        edge_attr=features.edge_tensor(games),
        context=features.context_tensor(games),
        y=_targets(games, schema, extra_targets),
        season=torch.tensor(games[schema.season].to_numpy(dtype=np.int64)),
        date=torch.tensor(games[schema.date].to_numpy(dtype=np.float64)),
        game=torch.from_numpy(game_codes.astype(np.int64)),
        valid=torch.from_numpy(
            ((prior["team"] >= min_games) & (prior["opp"] >= min_games)).to_numpy()
        ),
        num_nodes=len(features.teams),
    )
    graph.game_ids = np.asarray(game_ids)
    return graph


def _targets(games, schema: GameSchema, extra: tuple[str, ...]) -> torch.Tensor:
    y = games[[schema.target, *extra]].apply(pd.to_numeric, errors="coerce")
    y = torch.tensor(y.to_numpy(dtype=np.float32))
    return y if extra else y[:, 0]


EDGE_KEYS = ("edge_attr", "context", "y", "season", "date", "game", "valid")


def edge_subgraph(graph: pyg.data.Data, mask: torch.Tensor) -> pyg.data.Data:
    """Keep the masked edges and every node (explicit, no attribute guessing)."""
    sub = pyg.data.Data(edge_index=graph.edge_index[:, mask], num_nodes=graph.num_nodes)
    for key in EDGE_KEYS:
        sub[key] = graph[key][mask]
    return sub
