import torch
from torch_geometric.nn import MessagePassing

from .base import GraphEncoder


class EdgeEncoder(MessagePassing, GraphEncoder):
    """Mean of linearly projected, batch-normalized edge features."""

    def __init__(self, edge_dim: int, out_dim: int):
        super().__init__(aggr="mean")
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__
        self.edge_dim = edge_dim
        self.out_dim = out_dim

        self.bn = torch.nn.BatchNorm1d(edge_dim)
        self.lin_edge = torch.nn.Linear(edge_dim, out_dim)
        self.act = torch.nn.ReLU()

    def forward(self, edge_index, edge_attr, num_nodes):
        """
        Args:
            edge_index: LongTensor with shape [2, num_edges]
            edge_attr: Tensor with shape [num_edges, edge_dim]
            num_nodes: number of teams
        Returns:
            Tensor of shape [num_nodes, out_dim]
        """
        edge_attr = self.lin_edge(self.bn(edge_attr))
        aggr = self.propagate(
            edge_index, edge_attr=edge_attr, size=(num_nodes, num_nodes)
        )
        return self.act(aggr)

    def message(self, edge_attr):
        # lin already applied in forward
        return edge_attr
