import torch
import torch_geometric as pyg

from .graph import edge_subgraph


class SnapshotDataset(torch.utils.data.Dataset):
    """One sample per (season, date): that season's games up to the date.

    Each sample marks its own targets, so samples can be batched freely:

    * ``on_date``: edges played on the snapshot date (excluded from the
      message-passing context).
    * ``target``: ``on_date`` edges whose teams both have earlier games.
    """

    def __init__(self, graph: pyg.data.Data, seasons: list[int]):
        self.graph = edge_subgraph(
            graph, torch.isin(graph.season, torch.tensor(seasons))
        )
        keys = torch.stack([self.graph.season.double(), self.graph.date], dim=1)
        self.snapshots = torch.unique(keys[self.graph.valid], dim=0).tolist()

    def __len__(self):
        return len(self.snapshots)

    def __getitem__(self, idx: int) -> pyg.data.Data:
        season, date = self.snapshots[idx]
        g = self.graph
        sub = edge_subgraph(g, (g.season == int(season)) & (g.date <= date))
        sub.on_date = sub.date == date
        sub.target = sub.on_date & sub.valid
        return sub
