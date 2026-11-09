from __future__ import annotations

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from ...core.schema import GameSchema, played
from ...core.splits import SeasonSplit
from .dataset import SnapshotDataset
from .graph import GraphFeatures, build_graph


class GraphDataModule(pl.LightningDataModule):
    """Season snapshots for train / val / test seasons of one games table."""

    def __init__(
        self,
        games: pd.DataFrame,
        schema: GameSchema,
        split: SeasonSplit,
        batch_size: int = 8,
        num_workers: int = 0,
        min_games: int = 1,
        extra_targets: tuple[str, ...] = (),
    ):
        super().__init__()
        self.schema = schema
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers

        games = played(games, schema)
        games = games[games[schema.season].isin(split.all)]
        train_games = games[games[schema.season].isin(split.train)]
        self.features = GraphFeatures.fit(games, train_games, schema)
        self.graph = build_graph(
            games, schema, self.features, min_games, tuple(extra_targets)
        )

        self.extra_targets = tuple(extra_targets)
        y = train_games[[schema.target, *self.extra_targets]].apply(
            pd.to_numeric, errors="coerce"
        )
        if self.extra_targets:
            self.target_mean = y.mean().fillna(0.0).to_numpy().copy()
            self.target_std = y.std().replace(0, 1).fillna(1.0).to_numpy()
        else:
            self.target_mean = float(y[schema.target].mean())
            self.target_std = float(y[schema.target].std())

        self.datasets = {
            name: SnapshotDataset(self.graph, seasons)
            for name in ("train", "val", "test")
            if (seasons := getattr(split, name))
        }

    def _loader(self, name: str, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            self.datasets[name],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("val")

    def test_dataloader(self):
        return self._loader("test")

    @torch.no_grad()
    def predict_targets(self, model) -> tuple[pd.DataFrame, np.ndarray]:
        """Each split's target edges with ``model``'s unscaled predictions.

        Returns played-game rows (season, game_id, target, and the total if
        it is an extra target) and predictions (``[n, n_outputs]``).
        """
        total = self.schema.total
        model.eval()
        rows, preds = [], []
        for name in self.datasets:
            for batch in self._loader(name):
                batch = batch.to(model.device)
                preds.append(model.predict_outputs(batch).cpu().numpy())
                t = batch.target
                y = batch.y[t].cpu().numpy().reshape(int(t.sum()), -1)
                frame = {
                    self.schema.season: batch.season[t].cpu().numpy(),
                    self.schema.game_id: self.graph.game_ids[
                        batch.game[t].cpu().numpy()
                    ],
                    self.schema.target: y[:, 0],
                }
                if total and total in self.extra_targets:
                    frame[total] = y[:, 1 + self.extra_targets.index(total)]
                rows.append(pd.DataFrame(frame))
        return pd.concat(rows, ignore_index=True), np.concatenate(preds)
