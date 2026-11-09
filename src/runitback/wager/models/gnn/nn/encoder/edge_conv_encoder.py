from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.utils import scatter

from .base import GraphEncoder


class EdgeConvEncoderLayer(MessagePassing):
    """Weight-shared edge-driven graph convolution layer.

    Because this layer is reused across all iterations of message passing in
    :class:`EdgeConvEncoder`, its input and output dimensions are both fixed to
    ``hidden_dim``.  The message is formed by concatenating the source-node
    hidden state with the edge attributes, so the edge MLP always sees a vector
    of size ``edge_dim + hidden_dim``.

    Args:
        edge_dim: Dimensionality of edge attributes (box score stats).
        hidden_dim: Fixed dimensionality for both input and output node states.
        aggr: Aggregation scheme ('mean', 'sum', or 'max').
        dropout: Dropout probability applied within the edge MLP.
    """

    def __init__(
        self,
        edge_dim: int,
        hidden_dim: int,
        aggr: str = "mean",
        dropout: float = 0.0,
    ) -> None:
        super().__init__(aggr=aggr)

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, edge_index: Adj, edge_attr: Tensor, x: Tensor) -> Tensor:
        """Run one weight-shared message passing step.

        Args:
            edge_index: Graph connectivity of shape [2, E].
            edge_attr: Edge feature matrix of shape [E, edge_dim].
            x: Current node embeddings of shape [N, hidden_dim].

        Returns:
            Updated node embeddings of shape [N, hidden_dim].
        """
        src = edge_index[0]
        # Message: opponent context (x_src) + game stats (edge_attr)
        msg_input = torch.cat([edge_attr, x[src]], dim=-1)
        n = x.size(0)
        agg = self.propagate(edge_index=edge_index, msg_input=msg_input, size=(n, n))
        return self.update_mlp(agg)

    def message(self, msg_input: Tensor) -> Tensor:
        return self.edge_mlp(msg_input)


class EdgeConvEncoder(GraphEncoder):
    """Recurrent-style edge-driven graph convolution encoder.

    A **single** :class:`EdgeConvEncoderLayer` is applied ``num_layers`` times,
    sharing weights across every iteration.  This keeps the parameter count
    constant regardless of depth and encourages learning a general update rule
    rather than layer-specific transformations.

    **Initialisation** — before any message passing, each team node is seeded
    with its season-average box score stats, computed by mean-pooling all edge
    attributes incident to that node (both as source and destination).  These
    raw averages are projected to ``hidden_dim`` via a learned linear map.

    **Iteration** — at each step the layer computes messages as
    ``concat(edge_attr, x_src)`` and aggregates them at the destination node.
    Because ``x_src`` carries the source team's current embedding, each round
    grows the receptive field by one hop and folds in contextual information
    about how opponents have played.

    Args:
        edge_dim: Dimensionality of edge attributes (box score stats).
        hidden_dim: Dimensionality of node embeddings throughout all iterations.
        out_dim: Dimensionality of the final output node embeddings.
        num_layers: Number of times the shared layer is applied (>= 1).
        aggr: Aggregation scheme ('mean', 'sum', or 'max').
        dropout: Dropout probability applied within the edge MLP.
    """

    def __init__(
        self,
        edge_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        aggr: str = "mean",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        self.num_layers = num_layers
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__

        # Normalises raw edge features (points, assists, etc.) to a common
        # scale before any projection or pooling is applied.
        self.edge_norm = nn.BatchNorm1d(edge_dim)

        # Projects mean-pooled edge attrs (season averages) → hidden_dim.
        self.node_init_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
        )

        # Single shared layer reused at every iteration.
        self.layer = EdgeConvEncoderLayer(
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            aggr=aggr,
            dropout=dropout,
        )

        # Maps final hidden state → desired output dimension.
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, edge_index: Adj, edge_attr: Tensor, num_nodes: int) -> Tensor:
        """Encode all team nodes from game edge attributes.

        Args:
            edge_index: Graph connectivity of shape [2, E].
            edge_attr: Edge feature matrix of shape [E, edge_dim].
            num_nodes: Number of team nodes, including teams without edges.

        Returns:
            Node embedding matrix of shape [N, out_dim].
        """
        # Normalise edge features once before any pooling or message passing.
        edge_attr = self.edge_norm(edge_attr)

        # --- Initialise node features as season averages ---
        # Pool every edge attribute to its destination (target) node so each
        # team gets the mean of all box score stats from games played against
        # them.  Since the graph is directed and each game is represented as
        # two opposing directed edges, pooling to targets only avoids
        # double-counting.
        x_init = scatter(
            edge_attr, edge_index[1], dim=0, dim_size=num_nodes, reduce="mean"
        )
        x = self.node_init_proj(x_init)  # [N, hidden_dim]

        # --- Iterative refinement with weight-shared convolution ---
        for _ in range(self.num_layers):
            x = self.layer(edge_index=edge_index, edge_attr=edge_attr, x=x)

        return self.output_proj(x)  # [N, out_dim]
