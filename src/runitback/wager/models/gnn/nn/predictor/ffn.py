import torch
from lightning.pytorch.core.mixins import HyperparametersMixin


class EdgeFFN(torch.nn.Module, HyperparametersMixin):
    """Predicts an edge's target from its two node embeddings and game context.

    Edges point from opponent (source) to team (target); the prediction is the
    target team's margin.

    Args:
        in_dim: dimensionality of input node features
        out_dim: dimensionality of output edge features / predictions
        hidden_dim: optional hidden dimension for a single hidden layer
        context_dim: number of pre-game context features per edge (location,
            rest, ...)
        activation: activation module class or instance (default: ReLU)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_dim: int | None = None,
        context_dim: int = 0,
        activation=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.context_dim = context_dim
        if activation is None:
            activation = torch.nn.ReLU()
        # source and target embeddings are concatenated with the context
        proj_in = in_dim * 2 + context_dim
        if hidden_dim is None:
            self.net = torch.nn.Linear(proj_in, out_dim)
        else:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(proj_in, hidden_dim),
                activation,
                torch.nn.Linear(hidden_dim, out_dim),
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute edge predictions from node features.

        Args:
                x: node feature tensor of shape [num_nodes, in_dim]
                edge_index: long tensor of shape [2, num_edges] with source,dest indices
                context: tensor of shape [num_edges, context_dim]

        Returns:
                Tensor of shape [num_edges, out_dim]
        """
        parts = [x[edge_index[0]], x[edge_index[1]]]
        if self.context_dim:
            parts.append(context)
        return self.net(torch.cat(parts, dim=-1))
