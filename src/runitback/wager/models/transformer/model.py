from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from lightning.pytorch.core.mixins import HyperparametersMixin
from torch import nn

from ...core.predictor import MatchupModel
from ...core.schema import GameSchema, mirror, played
from ..lightning import MarginModule
from .data import SequenceFeatures, sequence


class SequenceEncoder(nn.Module, HyperparametersMixin):
    """Transformer over a team's recent games -> one team embedding.

    Each token is an earlier game (stats, result, location) plus how many
    seasons ago it was; a learned position embedding encodes recency (0 is
    the most recent game). The output is the final state of a [CLS] token.

    Args:
        token_dim: Features per game token.
        history: Maximum number of earlier games per sequence.
        d_model / nhead / num_layers / dropout: Transformer sizes.
        season_embedding: Also add a learned embedding of how many seasons
            ago each game was (0, 1, ..., ``max_seasons_ago`` and older), so
            the sequence is encoded by both game order and season.
        season_decay: Down-weight games from seasons before the most recent
            one in the sequence: each head adds ``-slope * (seasons_ago -
            newest)`` to every query's attention logit for that game, with a
            learned slope per head starting at ``decay_init`` per season. A
            season opener's sequence (no games yet this season) is therefore
            not penalized; mid-season, last season's games count for less
            than this season's.
    """

    def __init__(
        self,
        token_dim: int,
        history: int = 20,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        season_embedding: bool = False,
        season_decay: bool = False,
        max_seasons_ago: int = 3,
        decay_init: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__
        self.token_dim = token_dim
        self.history = history
        self.out_dim = d_model

        self.embed = nn.Linear(token_dim + 1, d_model)
        self.position = nn.Embedding(history + 1, d_model)
        self.nhead = nhead
        self.max_seasons_ago = max_seasons_ago
        self.season = (
            nn.Embedding(max_seasons_ago + 1, d_model) if season_embedding else None
        )
        # softplus(raw) is each head's penalty per season ago
        self.decay = (
            nn.Parameter(torch.full((nhead,), math.log(math.expm1(decay_init))))
            if season_decay
            else None
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, seasons_ago, pad) -> torch.Tensor:
        """
        Args:
            tokens: ``[B, K, token_dim]``; seasons_ago: ``[B, K]``;
            pad: ``[B, K]`` bool, True where there is no game.
        Returns:
            ``[B, d_model]`` team embeddings.
        """
        batch, length, _ = tokens.shape
        x = self.embed(torch.cat([tokens, seasons_ago.unsqueeze(-1)], dim=-1))
        if self.season is not None:
            ago = seasons_ago.clamp(0, self.max_seasons_ago).long()
            x = x + self.season(ago)
        x = torch.cat([self.cls_token.expand(batch, -1, -1), x], dim=1)
        x = x + self.position(torch.arange(length + 1, device=x.device))
        pad = torch.cat([pad.new_zeros(batch, 1), pad], dim=1)
        if self.decay is None:
            return self.norm(self.encoder(x, src_key_padding_mask=pad)[:, 0])
        return self.norm(
            self.encoder(x, mask=self._attention_bias(seasons_ago, pad))[:, 0]
        )

    def _attention_bias(self, seasons_ago, pad) -> torch.Tensor:
        """``[B * nhead, L, L]`` additive attention mask: ``-slope_h * (seasons
        older than the sequence's newest game)`` for each key game (0 for
        [CLS]), a large negative for padding."""
        batch = seasons_ago.shape[0]
        games = pad[:, 1:]
        newest = seasons_ago.masked_fill(games, float("inf")).amin(dim=1, keepdim=True)
        older = (seasons_ago - newest).masked_fill(games, 0.0).clamp(min=0)
        ago = torch.cat([older.new_zeros(batch, 1), older], dim=1)
        slope = nn.functional.softplus(self.decay)  # [nhead]
        bias = -slope[None, :, None] * ago[:, None, :]  # [B, nhead, L]
        bias = bias.masked_fill(pad[:, None, :], -1e9)
        length = ago.shape[1]
        return (
            bias[:, :, None, :]
            .expand(-1, -1, length, -1)
            .reshape(batch * self.nhead, length, length)
        )


class PairHead(nn.Module, HyperparametersMixin):
    """MLP on (team embedding, opponent embedding, context) -> margin, or
    ``out_dim`` outputs (margin first, then e.g. total and box-score stats).

    With ``antisymmetric`` the MLP ``g`` also reads the matchup from the
    opponent's side (embeddings swapped, ``opp_context`` the mirrored
    context) and output 0, the margin, is ``(g(A, B) - g(B, A)) / 2``, so
    the opponent's predicted margin is exactly its negative; the outputs in
    ``symmetric_outputs`` (the total, a distribution's log-scales) are
    averaged over both sides and the rest come from the team's side.
    """

    def __init__(
        self,
        in_dim: int,
        context_dim: int = 0,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        out_dim: int = 1,
        antisymmetric: bool = False,
        symmetric_outputs: tuple[int, ...] = (),
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__
        self.in_dim = in_dim
        self.context_dim = context_dim
        self.out_dim = out_dim
        self.antisymmetric = antisymmetric
        self.symmetric_outputs = tuple(symmetric_outputs)
        self.net = nn.Sequential(
            nn.Linear(2 * in_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _side(self, team, opp, context) -> torch.Tensor:
        parts = [team, opp] + ([context] if self.context_dim else [])
        return self.net(torch.cat(parts, dim=-1))

    def forward(self, team, opp, context, opp_context=None) -> torch.Tensor:
        out = self._side(team, opp, context)
        if getattr(self, "antisymmetric", False):
            back = self._side(
                opp, team, context if opp_context is None else opp_context
            )
            columns = list(out.unbind(-1))
            columns[0] = (out[:, 0] - back[:, 0]) / 2
            for i in self.symmetric_outputs:
                columns[i] = (out[:, i] + back[:, i]) / 2
            out = torch.stack(columns, dim=-1)
        return out.squeeze(-1) if getattr(self, "out_dim", 1) == 1 else out


class TransformerModel(MarginModule):
    """Shared :class:`SequenceEncoder` for both teams + :class:`PairHead`."""

    SUBMODULES = ("encoder", "head")

    def __init__(
        self,
        encoder: SequenceEncoder,
        head: PairHead,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        target_mean: float | list[float] = 0.0,
        target_std: float | list[float] = 1.0,
        extra_targets: tuple[str, ...] = (),
        extra_weights: tuple[float, ...] = (),
        distribution: bool = False,
        token_features: dict | None = None,
    ):
        super().__init__(
            lr,
            weight_decay,
            target_mean,
            target_std,
            extra_targets,
            extra_weights,
            distribution,
        )
        self.encoder = encoder
        self.head = head
        # the token options (TOKEN_KEYS) the model was built for
        self.token_features = dict(token_features or {})
        self.save_hyperparameters(ignore=list(self.SUBMODULES))
        self.save_submodules()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def forward(self, batch) -> torch.Tensor:
        team = self.encoder(**batch["team"])
        opp = self.encoder(**batch["opp"])
        return self.head(team, opp, batch["context"], batch.get("opp_context"))

    def targets(self, batch) -> torch.Tensor:
        return batch["y"]


class TransformerEnsemble(nn.Module):
    """Several :class:`TransformerModel` fits of the same data (different
    seeds) served as one: point outputs are the members' average, a score
    distribution the average of their probabilities."""

    def __init__(self, members: list[TransformerModel]):
        super().__init__()
        self.members = nn.ModuleList(members)

    @property
    def extra_targets(self) -> tuple[str, ...]:
        return tuple(getattr(self.members[0], "extra_targets", ()))

    @property
    def distribution(self) -> bool:
        return bool(getattr(self.members[0], "distribution", False))

    @property
    def margin_dist(self):
        return self.members[0].margin_dist

    @property
    def total_dist(self):
        return self.members[0].total_dist

    @property
    def encoder(self) -> SequenceEncoder:
        return self.members[0].encoder

    @property
    def head(self) -> PairHead:
        return self.members[0].head

    def predict_outputs(self, batch) -> torch.Tensor:
        return torch.stack([m.predict_outputs(batch) for m in self.members]).mean(0)

    def predict_margin(self, batch) -> torch.Tensor:
        return self.predict_outputs(batch)[:, 0]

    def log_probs(self, batch) -> dict[str, torch.Tensor]:
        per_member = [m.log_probs(batch) for m in self.members]
        n = math.log(len(per_member))
        return {
            key: torch.logsumexp(torch.stack([p[key] for p in per_member]), 0) - n
            for key in per_member[0]
        }


def distribution_outputs(module, batch) -> dict[str, np.ndarray]:
    """A distribution model's probability tables: ``margin_pmf`` over
    ``margin_values`` (and ``total_pmf`` over ``total_values``)."""
    probs = {k: v.exp().numpy() for k, v in module.log_probs(batch).items()}
    out = {
        "margin_pmf": probs["margin"],
        "margin_values": module.margin_dist.values.numpy(),
    }
    if "total" in probs:
        out |= {
            "total_pmf": probs["total"],
            "total_values": module.total_dist.values.numpy(),
        }
    return out


class TransformerMatchupModel(MatchupModel):
    """Serves a :class:`TransformerModel` from each team's latest games.

    State is each team's last ``history`` played games as raw tokens;
    embeddings are computed per request, with ``seasons_ago`` relative to the
    matchup's season (so a season opener sees last season's games as one
    season old, exactly as in training).
    """

    def __init__(
        self, schema: GameSchema, module: TransformerModel, features: SequenceFeatures
    ):
        self.schema = schema
        self.module = module.cpu().eval()
        self.features = features
        self.team_index: dict = {}
        self.state_season = None

    def update_state(self, games: pd.DataFrame) -> None:
        s = self.schema
        games = played(games, s).sort_values([s.season, s.date], kind="stable")
        self.state_season = games[s.season].max()
        tokens = self.features.tokens(games)
        seasons = games[s.season].to_numpy(dtype=np.float32)
        # teams of this season or the last: a team still waiting for its
        # first game of the season (a bye, a staggered start) keeps its state
        recent = games[s.season] >= self.state_season - 1
        current = set(games.loc[recent, s.team])

        k = self.features.history
        position = pd.Series(np.arange(len(games)), index=games.index)
        index = []
        teams = [t for t in pd.unique(games[s.team]) if t in current]
        for team in teams:
            rows = position[games[s.team] == team].to_numpy()[::-1][:k]
            index.append(np.pad(rows, (0, k - len(rows)), constant_values=-1))
        self.team_index = {team: i for i, team in enumerate(teams)}
        self.history = torch.from_numpy(np.array(index, dtype=np.int64).reshape(-1, k))
        self.tokens = torch.from_numpy(tokens)
        self.seasons = torch.from_numpy(seasons)

    @property
    def teams(self) -> list:
        return list(self.team_index)

    @torch.no_grad()
    def team_embeddings(self) -> pd.DataFrame | None:
        """Each team's encoded history as of the state's season (an
        ensemble's members side by side: their spaces differ)."""
        if not self.team_index:
            return None
        season = torch.full((len(self.history),), float(self.state_season))
        batch = sequence(self.tokens, self.seasons, self.history, season)
        members = getattr(self.module, "members", [self.module])
        parts = [member.encoder(**batch).numpy() for member in members]
        return pd.DataFrame(np.concatenate(parts, axis=1), index=self.teams)

    def predict_margin(self, matchups: pd.DataFrame) -> np.ndarray:
        return self.predict_outputs(matchups)["margin"]

    @torch.no_grad()
    def predict_outputs(self, matchups: pd.DataFrame) -> dict[str, np.ndarray]:
        s = self.schema
        season = (
            matchups[s.season].to_numpy(dtype=np.float32)
            if s.season in matchups
            else np.full(len(matchups), self.state_season, dtype=np.float32)
        )
        season = torch.from_numpy(season)

        def side(keys):
            index = self.history[[self.team_index[k] for k in keys]]
            return sequence(self.tokens, self.seasons, index, season)

        batch = {
            "team": side(matchups[s.team]),
            "opp": side(matchups[s.opp]),
            "context": torch.from_numpy(self.features.context(matchups)),
            "opp_context": torch.from_numpy(self.features.context(mirror(matchups, s))),
        }
        out = self.module.predict_outputs(batch).numpy()
        extra = getattr(self.module, "extra_targets", ())
        result = {"margin": out[:, 0]}
        if s.total and s.total in extra:
            result["total"] = out[:, 1 + extra.index(s.total)]
        if getattr(self.module, "distribution", False):
            result |= distribution_outputs(self.module, batch)
        return result
