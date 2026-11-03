"""Sport-agnostic game embeddings: pool a play model's hidden states by which team had the ball."""
from __future__ import annotations

import torch


def pool_by_possession(h: torch.Tensor, valid: torch.Tensor, home_has_ball: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean hidden state over each team's possessions.

    Args:
        h: ``[B, T, D]`` hidden states, one per play/event.
        valid: ``[B, T]`` 1 where there is a real play (0 = padding).
        home_has_ball: ``[B, T]`` 1 where the home team is on offense.

    Returns:
        ``(home_off, away_off)``, each ``[B, D]``: the mean state over plays where the home
        (away) team had the ball; zeros for a side with no plays.
    """
    valid = valid.to(h.dtype)
    home = valid * home_has_ball.to(h.dtype)
    away = valid - home

    def mean(w):
        return (h * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1)

    return mean(home), mean(away)


def team_vectors(home_off: torch.Tensor, away_off: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each team's game vector: its own offense first, then its defense (the opponent's offense).

    Returns ``(home, away)``, each ``[B, 2D]``.
    """
    return torch.cat([home_off, away_off], -1), torch.cat([away_off, home_off], -1)
