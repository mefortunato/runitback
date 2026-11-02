"""NFL play transformer: shared causal trunk (core) with factorized next-play outcome heads."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.transformer import Block, KVCache, mlp
from ..vocab import N_ELAPSED, N_YARDS, PLAY_TYPES, RESULTS
from .features import FEATURES
from .lineups import N_COACH_CONT, N_GROUPS, N_SLOTS, SIDE, SLOT_GROUP

NEG = -1e9


@dataclass
class ModelConfig:
    feature_sizes: dict
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    max_len: int = 400
    dropout: float = 0.1
    n_cont: int = 0  # continuous team-strength inputs (0 = model without ratings)
    n_players: int = 0  # player tokens (0 = team-level model without lineups, coaches or personnel)
    n_coaches: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class LineupEncoder(nn.Module):
    """Player-level team strength.

    teams(): a team's depth chart (slot tokens) -> one vector. Each position group is pooled
    separately (learned weights per depth slot, so QB1 can dominate the QB channel), then the
    groups are concatenated and projected. An all-empty depth chart maps to a learned UNKNOWN vector.
    personnel(): the 22 players on the field for a play -> one vector (per-group sums per side, so
    the number of WRs, TEs, DBs... on the field is visible), or a learned UNKNOWN vector.
    """

    def __init__(self, n_players: int, d: int):
        super().__init__()
        self.player = nn.Embedding(n_players, d, padding_idx=0)
        self.slot = nn.Embedding(N_SLOTS, d)
        self.slot_w = nn.Parameter(torch.zeros(N_SLOTS))
        self.missing = nn.Parameter(torch.zeros(N_GROUPS, d))
        self.team_proj = nn.Sequential(nn.Linear(N_GROUPS * d, d), nn.GELU(), nn.LayerNorm(d))
        self.unk_team = nn.Parameter(torch.zeros(d))
        self.as_off, self.as_def = nn.Linear(d, d), nn.Linear(d, d)
        self.group = nn.Embedding(N_GROUPS + 1, d, padding_idx=0)
        self.pers_proj = nn.Sequential(nn.Linear(2 * N_GROUPS * d, d), nn.GELU(), nn.LayerNorm(d))
        self.unk_pers = nn.Parameter(torch.zeros(d))
        self.register_buffer("slot_group", F.one_hot(torch.as_tensor(SLOT_GROUP), N_GROUPS).float(),
                             persistent=False)

    def teams(self, lineup: torch.Tensor) -> torch.Tensor:
        """lineup (B, K, S) slot tokens -> (B, K, d)."""
        x = self.player(lineup) + self.slot.weight                      # (B,K,S,d)
        w = F.softplus(self.slot_w) * (lineup > 0)                       # (B,K,S)
        wg = w[..., None] * self.slot_group                              # (B,K,S,G)
        num = torch.einsum("bksg,bksd->bkgd", wg, x)
        den = wg.sum(2)[..., None]                                       # (B,K,G,1)
        pooled = torch.where(den > 0, num / den.clamp(min=1e-6), self.missing)
        v = self.team_proj(pooled.flatten(-2))
        known = (lineup > 0).any(-1, keepdim=True)
        return torch.where(known, v, self.unk_team)

    def personnel(self, pers: torch.Tensor, grp: torch.Tensor) -> torch.Tensor:
        """pers, grp (B, T, 22) tokens and 1-based groups -> (B, T, d)."""
        x = self.player(pers) + self.group(grp)                          # (B,T,22,d)
        oh = F.one_hot(grp, N_GROUPS + 1)[..., 1:].float()               # (B,T,22,G)
        B, T = pers.shape[:2]
        off = torch.einsum("btpg,btpd->btgd", oh[:, :, :SIDE], x[:, :, :SIDE]).reshape(B, T, -1)
        dfn = torch.einsum("btpg,btpd->btgd", oh[:, :, SIDE:], x[:, :, SIDE:]).reshape(B, T, -1)
        v = self.pers_proj(torch.cat([off, dfn], -1))
        known = (pers > 0).any(-1, keepdim=True)
        return torch.where(known, v, self.unk_pers)


class PlayTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.emb = nn.ModuleDict({f: nn.Embedding(cfg.feature_sizes[f], d) for f in FEATURES})
        self.lineup = LineupEncoder(cfg.n_players, d) if cfg.n_players else None
        if cfg.n_players:
            self.off_coach = nn.Embedding(max(cfg.n_coaches, 3), d)
            self.def_coach = nn.Embedding(max(cfg.n_coaches, 3), d)
            self.coach_cont = nn.Linear(N_COACH_CONT, d)
        self.pos = nn.Embedding(cfg.max_len, d)
        self.cont = nn.Linear(cfg.n_cont, d) if cfg.n_cont else None
        self.in_ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(d)
        nT, nR = len(PLAY_TYPES), len(RESULTS)
        self.e_type, self.e_res, self.e_yards = nn.Embedding(nT, d), nn.Embedding(nR, d), nn.Embedding(N_YARDS, d)
        self.type_head = mlp(d, d, nT)
        self.res_head = mlp(d, d, nR)
        self.yards_head = mlp(d, d, N_YARDS)
        self.el_head = mlp(d, d, N_ELAPSED)

    # -- trunk
    def encode(self, feats: dict[str, torch.Tensor], cache: KVCache | None = None) -> torch.Tensor:
        x = sum(self.emb[f](feats[f]) for f in FEATURES)
        if self.cont is not None:
            x = x + self.cont(feats["cont"])
        if self.lineup is not None:
            x = x + self.lineup_context(feats)
        T = x.shape[1]
        start = cache.len if cache is not None else 0
        x = x + self.pos(torch.arange(start, start + T, device=x.device).clamp(max=self.cfg.max_len - 1))
        x = self.drop(self.in_ln(x))
        for i, b in enumerate(self.blocks):
            x = b(x, cache, i)
        if cache is not None:
            cache.len += T
        return self.ln_f(x)

    def lineup_context(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        """Possession team's lineup (as offense) + opponent's lineup (as defense) + both coaches."""
        teams = self.lineup.teams(feats["lineup"])                      # (B, 2, d): home, away
        home = feats["is_home"].bool()[..., None]                       # (B, T, 1)
        off = torch.where(home, teams[:, :1], teams[:, 1:])
        dfn = torch.where(home, teams[:, 1:], teams[:, :1])
        return (self.lineup.as_off(off) + self.lineup.as_def(dfn) + self.off_coach(feats["off_coach"])
                + self.def_coach(feats["def_coach"]) + self.coach_cont(feats["coach_cont"]))

    def personnel(self, feats: dict[str, torch.Tensor]) -> torch.Tensor | int:
        """Per-play on-field vector added to the result/yards/clock heads (0 without lineups)."""
        if self.lineup is None or "pers" not in feats:
            return 0
        return self.lineup.personnel(feats["pers"], feats["pers_grp"])

    def new_cache(self, batch: int) -> KVCache:
        c = self.cfg
        return KVCache(c.n_layers, batch, c.n_heads, c.max_len, c.d_model // c.n_heads,
                       device=next(self.parameters()).device)

    # -- heads (each conditioned on the fields chosen before it)
    def type_logits(self, h):
        return self.type_head(h)

    # `p` is the per-play personnel vector: chosen after the play type, so on-field players shape
    # how the play turns out but do not decide what is called (e.g. punt unit -> punt).
    def result_logits(self, h, t, p=0):
        return self.res_head(h + self.e_type(t) + p)

    def yards_logits(self, h, t, r, p=0):
        return self.yards_head(h + self.e_type(t) + self.e_res(r) + p)

    def elapsed_logits(self, h, t, r, y, p=0):
        return self.el_head(h + self.e_type(t) + self.e_res(r) + self.e_yards(y) + p)

    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        h = self.encode(batch["feats"])
        p = self.personnel(batch["feats"])
        valid = batch["valid"]
        t, r, y, e = batch["type"], batch["result"], batch["yards"], batch["elapsed"]
        parts = {
            "type": (self.type_logits(h), t, batch["type_mask"]),
            "result": (self.result_logits(h, t, p), r, batch["result_mask"]),
            "yards": (self.yards_logits(h, t, r, p), y, batch["yards_mask"]),
            "elapsed": (self.elapsed_logits(h, t, r, y, p), e, batch["elapsed_mask"]),
        }
        total, logs = 0.0, {}
        n = valid.sum().clamp(min=1)
        for name, (logits, tgt, mask) in parts.items():
            logits = logits.masked_fill(~mask, NEG)
            nll = F.cross_entropy(logits.flatten(0, 1), tgt.flatten(), reduction="none").view_as(tgt)
            l = (nll * valid).sum() / n
            total = total + l
            logs[name] = l.item()
        return total, logs
