"""Game-level box-score loss built from the per-play predictions.

For every play the model's own distributions (play type -> result -> yards) give the *expected*
contribution to each box-score stat, computed exactly as the rules engine scores plays:

    gain results (run, complete, sack)   TD if yards >= yardline, safety if yards <= yardline - 100
    spot results (kicks, turnovers, missed FG)   return TD if the spot is 0
    field goal good 3, extra point good 1, two-point good 2

Summing those expectations over a game's plays (teacher forcing: the real game states) gives a
predicted box score for each team; the real box score uses the same definitions on the real
outcomes. The loss is the squared error of each stat, scaled by its spread across games.

Stats (per team): points, rushing yards, passing yards, sacks taken, turnovers; plus the margin.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..vocab import N_YARDS, PLAY_TYPES, PT, RESULTS, RESULTS_FOR_TYPE, RS, YARDS_MIN

TEAM_STATS = ["points", "rush_yds", "pass_yds", "sacks", "turnovers"]
GAME_STATS = [f"home_{s}" for s in TEAM_STATS] + [f"away_{s}" for s in TEAM_STATS] + ["margin"]
N_TEAM = len(TEAM_STATS)
# (type, result) pairs whose yards matter for a stat or for points; "gain" or "spot" yards
YARD_PAIRS = [("run", "run", "gain"), ("pass", "complete", "gain"), ("pass", "sack", "gain"),
              ("run", "fumble_lost", "spot"), ("pass", "interception", "spot"), ("pass", "fumble_lost", "spot"),
              ("qb_kneel", "fumble_lost", "spot"), ("kickoff", "kick_received", "spot"),
              ("kickoff", "kick_recovered", "spot"), ("punt", "kick_received", "spot"),
              ("punt", "kick_recovered", "spot"), ("field_goal", "no_good", "spot")]
RESULT_TYPES = ["kickoff", "run", "pass", "punt", "field_goal", "extra_point", "two_point", "qb_kneel"]
DEF_RETURN = {("run", "fumble_lost"), ("pass", "interception"), ("pass", "fumble_lost"), ("qb_kneel", "fumble_lost"),
              ("kickoff", "kick_received"), ("punt", "kick_received"), ("field_goal", "no_good")}
OFF_RETURN = {("kickoff", "kick_recovered"), ("punt", "kick_recovered")}
TURNOVERS = [("run", "fumble_lost"), ("pass", "interception"), ("pass", "fumble_lost"), ("qb_kneel", "fumble_lost")]
NEG = -1e9


def _pair_index(t, r, name_t, name_r):
    return (t == PT[name_t]) & (r == RS[name_r])


def true_play_stats(t: torch.Tensor, r: torch.Tensor, y: torch.Tensor, yl: torch.Tensor):
    """Real per-play contributions. t, r = type/result indices, y = yards (not index), yl = yardline.
    Returns (offense stats (..., 5), defense points (...))."""
    f = lambda m: m.float()
    gain_td = (_pair_index(t, r, "run", "run") | _pair_index(t, r, "pass", "complete")) & (y >= yl)
    safety = (_pair_index(t, r, "run", "run") | _pair_index(t, r, "pass", "complete")
              | _pair_index(t, r, "pass", "sack")) & (y <= yl - 100)
    ret = y <= 0
    off_ret = sum(_pair_index(t, r, a, b) for a, b in OFF_RETURN).bool() & ret
    def_ret = sum(_pair_index(t, r, a, b) for a, b in DEF_RETURN).bool() & ret
    good = r == RS["good"]
    off_pts = 6 * f(gain_td | off_ret) + 3 * f(good & (t == PT["field_goal"])) + \
        f(good & (t == PT["extra_point"])) + 2 * f(good & (t == PT["two_point"]))
    def_pts = 6 * f(def_ret) + 2 * f(safety)
    rush = torch.where(_pair_index(t, r, "run", "run"), y.float(), torch.zeros_like(off_pts))
    pas = torch.where(_pair_index(t, r, "pass", "complete"), y.float(), torch.zeros_like(off_pts))
    sacks = f(_pair_index(t, r, "pass", "sack"))
    tos = f(sum(_pair_index(t, r, a, b) for a, b in TURNOVERS).bool())
    return torch.stack([off_pts, rush, pas, sacks, tos], -1), def_pts


def expected_play_stats(model, h: torch.Tensor, p, type_mask: torch.Tensor, yl: torch.Tensor):
    """Expected per-play contributions under the model's distributions (differentiable)."""
    B, T, _ = h.shape
    pt = F.softmax(model.type_logits(h).masked_fill(~type_mask, NEG), -1)          # (B,T,nT)
    prob = {}  # (type, result) -> P(type, result)
    for name in RESULT_TYPES:
        ti = torch.full((B, T), PT[name], dtype=torch.long, device=h.device)
        legal = torch.zeros(len(RESULTS), dtype=torch.bool, device=h.device)
        legal[[RS[x] for x in RESULTS_FOR_TYPE[name]]] = True
        pr = F.softmax(model.result_logits(h, ti, p).masked_fill(~legal, NEG), -1)
        for res in RESULTS_FOR_TYPE[name]:
            prob[(name, res)] = pt[..., PT[name]] * pr[..., RS[res]]
    ar = torch.arange(N_YARDS, device=h.device) + YARDS_MIN                          # yards per index
    yl_ = yl[..., None]
    gain_mask = (ar >= yl_ - 100) & (ar <= yl_.clamp(max=N_YARDS + YARDS_MIN - 1))
    spot_mask = ((ar >= 0) & (ar <= 99)).expand_as(gain_mask)
    zero = torch.zeros(B, T, device=h.device)
    off_pts, def_pts, rush, pas = zero.clone(), zero.clone(), zero, zero
    for name, res, kind in YARD_PAIRS:
        ti = torch.full((B, T), PT[name], dtype=torch.long, device=h.device)
        ri = torch.full((B, T), RS[res], dtype=torch.long, device=h.device)
        py = F.softmax(model.yards_logits(h, ti, ri, p).masked_fill(~(gain_mask if kind == "gain" else spot_mask), NEG), -1)
        w = prob[(name, res)]
        if kind == "gain":
            td = (py * (ar >= yl_)).sum(-1)
            sf = (py * (ar <= yl_ - 100)).sum(-1)
            if res != "sack":
                off_pts = off_pts + 6 * w * td
            def_pts = def_pts + 2 * w * sf
            mean = (py * ar).sum(-1)
            if (name, res) == ("run", "run"):
                rush = w * mean
            elif (name, res) == ("pass", "complete"):
                pas = w * mean
        else:
            ret = py[..., -YARDS_MIN]  # spot 0
            if (name, res) in OFF_RETURN:
                off_pts = off_pts + 6 * w * ret
            else:
                def_pts = def_pts + 6 * w * ret
    off_pts = off_pts + 3 * prob[("field_goal", "good")] + prob[("extra_point", "good")] + 2 * prob[("two_point", "good")]
    sacks = prob[("pass", "sack")]
    tos = sum(prob[k] for k in TURNOVERS)
    return torch.stack([off_pts, rush, pas, sacks, tos], -1), def_pts


def game_box(off: torch.Tensor, def_pts: torch.Tensor, is_home: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Sum per-play contributions into (B, 11) game vectors: home stats, away stats, margin."""
    hm = (is_home.float() * valid)[..., None]
    aw = ((1 - is_home.float()) * valid)[..., None]
    home = (off * hm).sum(1)
    away = (off * aw).sum(1)
    home[:, 0] = home[:, 0] + (def_pts * aw[..., 0]).sum(1)  # points scored on the other team's plays
    away[:, 0] = away[:, 0] + (def_pts * hm[..., 0]).sum(1)
    return torch.cat([home, away, (home[:, :1] - away[:, :1])], -1)


def batch_box(model, h, p, batch) -> tuple[torch.Tensor, torch.Tensor]:
    """(predicted, real) game box scores for a batch, each (B, 11)."""
    f = batch["feats"]
    yl, is_home, valid = f["yl"], f["is_home"], batch["valid"]
    eo, ed = expected_play_stats(model, h, p, batch["type_mask"], yl)
    to, td = true_play_stats(batch["type"], batch["result"], batch["yards"] + YARDS_MIN, yl)
    return game_box(eo, ed, is_home, valid), game_box(to, td, is_home, valid)


def box_loss(pred: torch.Tensor, real: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Mean over games of the average squared standardized error across stats."""
    z = (pred - real) / sigma
    per_stat = (z ** 2).mean(0)
    logs = {f"box_{k}": float(v) ** 0.5 for k, v in zip(GAME_STATS, per_stat.detach())}  # RMS z per stat
    return 0.5 * per_stat.mean(), logs


def sim_box(log) -> list[float]:
    """Home/away rushing and passing yards of one simulated game's play log [(state, outcome), ...]."""
    out = [0.0, 0.0, 0.0, 0.0]  # home rush, home pass, away rush, away pass
    for s, o in log:
        side = 0 if s.pos == 0 else 2
        if o.play_type == "run" and o.result == "run":
            out[side] += o.yards
        elif o.play_type == "pass" and o.result == "complete":
            out[side + 1] += o.yards
    return out
