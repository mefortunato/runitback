import torch
from torch_geometric.nn import MessagePassing

from .base import GraphEncoder


class EdgeMean(MessagePassing, GraphEncoder):
    """Embeds each team as the (batch-normalized) mean of its edge features."""

    def __init__(self, edge_dim: int):
        super().__init__(aggr="mean")
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__
        self.edge_dim = self.out_dim = edge_dim

        self.bn = torch.nn.BatchNorm1d(edge_dim)

    def forward(self, edge_index, edge_attr, num_nodes):
        aggr = self.propagate(
            edge_index, edge_attr=edge_attr, size=(num_nodes, num_nodes)
        )
        return self.bn(aggr)

    def message(self, edge_attr):
        return edge_attr
