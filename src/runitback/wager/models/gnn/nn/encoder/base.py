from __future__ import annotations

from lightning.pytorch.core.mixins import HyperparametersMixin
from torch import Tensor, nn


class GraphEncoder(nn.Module, HyperparametersMixin):
    """Turns a graph of past games into one embedding per team.

    Edges point from opponent to team and carry that team's game stats.
    Subclasses must set ``edge_dim`` (edge features expected) and ``out_dim``
    (embedding size) and implement :meth:`forward`.
    """

    edge_dim: int
    out_dim: int

    def forward(self, edge_index: Tensor, edge_attr: Tensor, num_nodes: int) -> Tensor:
        """Return node embeddings of shape ``[num_nodes, out_dim]``."""
        raise NotImplementedError
