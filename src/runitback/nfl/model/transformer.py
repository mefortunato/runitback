"""NFL play transformer: shared causal trunk (core) with factorized next-play outcome heads."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.transformer import Block, KVCache, mlp
from ..vocab import N_ELAPSED, N_YARDS, PLAY_TYPES, RESULTS
from .features import FEATURES
from .lineups import GROUP_IDX, N_COACH_CONT, N_GROUPS, N_SLOTS, SIDE, SLOT_GROUP, WHO_NONE
from ..vocab import PT, RS

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
    player_dim: int = 0  # player/coach embedding width, projected to d_model (0 = d_model)
    lineup_dropout: float = 0.0  # dropout on the lineup/coach context and the personnel vector
    who_head: bool = False  # predict which offensive player gets the ball (needs lineups)
    n_usage: int = 0  # recent-usage features per who-head candidate (0 = embeddings only)
    # weight of the passer target: each pass play's result and yards predicted from the on-field
    # QB's player embedding alone, so the embedding has to carry how good he is (0 = off)
    qb_weight: float = 0.0

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

    def __init__(self, n_players: int, d: int, player_dim: int = 0):
        super().__init__()
        pd_ = player_dim or d
        emb = nn.Embedding(n_players, pd_, padding_idx=0)
        # a narrow table with a shared projection limits how much each player can memorize
        self.player = emb if pd_ == d else nn.Sequential(emb, nn.Linear(pd_, d, bias=False))
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
        self.lineup = LineupEncoder(cfg.n_players, d, cfg.player_dim) if cfg.n_players else None
        if cfg.n_players:
            cd = cfg.player_dim or d
            coach = lambda: nn.Embedding(max(cfg.n_coaches, 3), d) if cd == d else nn.Sequential(
                nn.Embedding(max(cfg.n_coaches, 3), cd), nn.Linear(cd, d, bias=False))
            self.off_coach, self.def_coach = coach(), coach()
            self.coach_cont = nn.Linear(N_COACH_CONT, d)
            self.lineup_drop = nn.Dropout(cfg.lineup_dropout)
        if cfg.n_players and cfg.who_head:
            self.who_ctx, self.who_cand = nn.Linear(d, d), nn.Linear(d, d)
            self.who_score = nn.Sequential(nn.GELU(), nn.Linear(d, 1))
            self.who_none = nn.Linear(d, 1)
            self.who_out = nn.Linear(d, d)
            self.who_none_emb = nn.Parameter(torch.zeros(d))
            self.who_unk = nn.Parameter(torch.zeros(d))
            self.who_usage = nn.Sequential(nn.Linear(cfg.n_usage, d), nn.GELU(), nn.Linear(d, d)) \
                if cfg.n_usage else None
        if cfg.n_players and cfg.qb_weight > 0:
            nR_, nY_ = len(RESULTS), N_YARDS
            self.qb_res = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, nR_))
            self.qb_yds = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, nY_))
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
        return self.lineup_drop(self.lineup.as_off(off) + self.lineup.as_def(dfn) + self.off_coach(feats["off_coach"])
                                + self.def_coach(feats["def_coach"]) + self.coach_cont(feats["coach_cont"]))

    def personnel(self, feats: dict[str, torch.Tensor]) -> torch.Tensor | int:
        """Per-play on-field vector added to the result/yards/clock heads (0 without lineups)."""
        if self.lineup is None or "pers" not in feats:
            return 0
        return self.lineup_drop(self.lineup.personnel(feats["pers"], feats["pers_grp"]))

    # -- who gets the ball (after the play type and personnel, before the result)
    @property
    def has_who(self) -> bool:
        return getattr(self, "who_ctx", None) is not None

    def who_candidates(self, pers: torch.Tensor, grp: torch.Tensor, usage: torch.Tensor | None = None) -> torch.Tensor:
        """(.., 22) personnel -> (.., 11, d): the possession team's players on the field, with their
        recent usage (.., 11, n_usage) when the model takes it."""
        c = self.lineup.player(pers[..., :SIDE]) + self.lineup.group(grp[..., :SIDE])
        if getattr(self, "who_usage", None) is not None and usage is not None:
            c = c + self.who_usage(usage)
        return c

    def who_mask(self, t: torch.Tensor, pers: torch.Tensor) -> torch.Tensor:
        """(.., 12) legal choices: runs need a ball carrier, passes may target nobody (sack, throwaway),
        other plays go to nobody; without known personnel only 'nobody' is legal."""
        cand = pers[..., :SIDE] > 0
        run, pas = t == PT["run"], t == PT["pass"]
        ok = cand & (run | pas)[..., None]
        none = ~run | ~cand.any(-1)
        return torch.cat([ok, none[..., None]], -1)

    def who_logits(self, h, t, p, cand):
        ctx = h + self.e_type(t) + p
        s = self.who_score(self.who_ctx(ctx)[..., None, :] + self.who_cand(cand)).squeeze(-1)
        return torch.cat([s, self.who_none(ctx)], -1)

    def who_vec(self, who: torch.Tensor, t: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """Vector added to the result/yards/clock heads: the chosen player, 'nobody', or unknown (-1)."""
        idx = who.clamp(0, SIDE - 1)
        chosen = torch.gather(cand, -2, idx[..., None, None].expand(*idx.shape, 1, cand.shape[-1])).squeeze(-2)
        v = torch.where((who >= 0)[..., None] & (who < SIDE)[..., None], self.who_out(chosen), self.who_unk)
        nobody = (who == WHO_NONE) | ((t != PT["run"]) & (t != PT["pass"]))
        return torch.where(nobody[..., None], self.who_none_emb, v)

    @staticmethod
    def who_result_mask(t: torch.Tensor, who: torch.Tensor, n_results: int) -> torch.Tensor:
        """Results consistent with who got the ball: no completion to nobody, no sack with a target."""
        m = torch.ones(*who.shape, n_results, dtype=torch.bool, device=who.device)
        pas = t == PT["pass"]
        m[..., RS["complete"]] &= ~(pas & (who == WHO_NONE))
        m[..., RS["sack"]] &= ~(pas & (who >= 0) & (who < SIDE))
        return m

    def passer_loss(self, f, t, r, y, valid, result_mask, yards_mask) -> torch.Tensor | None:
        """The passer target: NLL of each pass play's result, and of its yards given the result,
        from the possession team's on-field QB's embedding alone (no game state), so the
        embedding must encode the QB's completion / sack / interception / yardage profile."""
        off, grp = f["pers"][..., :SIDE], f["pers_grp"][..., :SIDE]
        is_qb = grp == GROUP_IDX["QB"] + 1
        qb = (off * is_qb).max(-1).values                                 # (B, T); 0 = no QB known
        m = (qb > 0) & (t == PT["pass"]) & (valid > 0)
        if not m.any():
            return None
        e = self.lineup.player(qb[m])                                     # (N, d)
        lr = self.qb_res(e).masked_fill(~result_mask[m], NEG)
        ly = self.qb_yds(e + self.e_res(r[m]).detach()).masked_fill(~yards_mask[m], NEG)
        return F.cross_entropy(lr, r[m]) + F.cross_entropy(ly, y[m])

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

    def loss(self, batch: dict[str, torch.Tensor], box_weight: float = 0.0,
             box_sigma: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        """Per-play NLL (mean over plays). With box_weight > 0, adds the game-level box-score loss:
        each game's expected box score, summed from the per-play predictions, against the real one."""
        h = self.encode(batch["feats"])
        p = self.personnel(batch["feats"])
        valid = batch["valid"]
        t, r, y, e = batch["type"], batch["result"], batch["yards"], batch["elapsed"]
        who_part = None
        if self.has_who and "who" in batch["feats"]:
            f = batch["feats"]
            cand = self.who_candidates(f["pers"], f["pers_grp"], f.get("pers_usage"))
            who = f["who"]
            logits = self.who_logits(h, t, p, cand).masked_fill(~self.who_mask(t, f["pers"]), NEG)
            m = (who >= 0) & ((t == PT["run"]) | (t == PT["pass"])) & (valid > 0)
            nll = F.cross_entropy(logits.flatten(0, 1), who.clamp(min=0).flatten(), reduction="none").view_as(who)
            who_part = (nll * m).sum() / m.sum().clamp(min=1)
            p = p + self.lineup_drop(self.who_vec(who, t, cand))
            rm = batch["result_mask"] & self.who_result_mask(t, who, batch["result_mask"].shape[-1])
            batch = dict(batch, result_mask=rm | F.one_hot(r, rm.shape[-1]).bool())
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
        if who_part is not None:
            total = total + who_part
            logs["who"] = who_part.item()
        if getattr(self, "qb_res", None) is not None and "pers" in batch["feats"]:
            q = self.passer_loss(batch["feats"], t, r, y, valid, batch["result_mask"], batch["yards_mask"])
            if q is not None:
                total = total + self.cfg.qb_weight * q
                logs["qb"] = q.item()
        if box_weight > 0:
            from .boxscore import batch_box, box_loss
            pred, real = batch_box(self, h, p, batch)
            bl, blogs = box_loss(pred, real, box_sigma)
            total = total + box_weight * bl
            logs |= {"box": bl.item()} | blogs
        return total, logs
