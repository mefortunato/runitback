"""Sport-agnostic player values: a weekly, leak-free ridge adjusted plus-minus.

Each event (an NFL play, an NBA possession, an NHL shift segment, ...) has a target (EPA, points,
expected goals, ...) and, for each side (e.g. ``off`` / ``def``), the ids of the players on the field.
The target is regressed on one coefficient per (side, player) plus situation controls, by ridge
regression with:

- **shrinkage per group**: a player's column is scaled by his group's shrink scale ``c``, so it is
  penalized ``c**2`` less (a wider prior; e.g. NFL quarterbacks, whose plays swing more);
- **recency**: an event from ``k`` seasons back is weighted ``decay ** k``, over ``window`` seasons
  back plus the current one up to the fit (times the event's own weight, if the config names a
  ``weight`` column: e.g. an NHL shift segment's length, for a target that is a rate);
- **replacement level**: players with fewer than ``min_events`` events (in the window; with
  ``count_weight``, that much of the ``weight`` column instead, e.g. minutes on ice) on a side
  share one replacement column per (side, group); a player's value is his coefficient minus his
  group's replacement coefficient, times the side's sign (``-1`` for defense, so that higher is
  better).

Every fit uses only events with ``t < before`` (``t`` is any increasing time key, e.g.
``season * 100 + week``), so values are leak-free; :func:`fit_weekly` runs one fit per cut-off.

:func:`shrunk_rates` is the companion for specialists valued per attempt (NFL kickers and punters):
recency-weighted target per attempt over the league mean, shrunk toward 0 by pseudo-attempts.

A sport supplies an adapter that builds :class:`Events` and an :class:`APMConfig`
(``runitback.nfl.player_values`` is the NFL's).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

__all__ = ["APMConfig", "Events", "codes_from_lists", "fit_apm", "fit_weekly", "shrunk_rates"]


@dataclass
class Events:
    """Events with known on-field personnel, players as integer codes.

    ``frame`` has one row per event, in the same order as the code arrays, with at least ``t`` (the
    time key fits cut on), ``season`` and the target column; any other columns are for the
    controls. ``sides`` maps each side to an ``(n, W)`` array of player codes (``-1`` = empty slot).
    ``ids`` maps code -> player id and ``group`` code -> index into the config's ``groups``
    (``-1`` = unknown: such a player's slots are ignored)."""
    frame: pd.DataFrame
    sides: dict[str, np.ndarray]
    ids: np.ndarray
    group: np.ndarray


@dataclass(frozen=True)
class APMConfig:
    """How a sport's adjusted plus-minus is fit (see the module docstring).

    ``controls(window_frame)`` returns the sparse control columns for the events in the fit's
    window (their rows of ``Events.frame``); controls are barely penalized if they are scaled up.
    ``sides`` lists (side, sign) in column order; a side's value is ``sign * (coef - replacement)``."""
    groups: tuple[str, ...]
    target: str
    controls: Callable[[pd.DataFrame], sp.spmatrix]
    sides: tuple[tuple[str, float], ...] = (("off", 1.0), ("def", -1.0))
    group_scale: dict[str, float] = field(default_factory=dict)   # shrink scale per group (default 1)
    min_events: int = 150          # events in the window for a player's own coefficient
    decay: float = 0.7             # weight of an event from k seasons back: decay ** k
    window: int = 4                # previous seasons in each fit (plus the current one)
    lam: float = 3000.0            # ridge penalty on (scaled) player coefficients
    target_clip: float | None = None
    weight: str | None = None      # column of per-event weights (default: every event weighs 1)
    count_weight: bool = False     # min_events (and the events column) count the weight, not events
    atol: float = 1e-6
    btol: float = 1e-6
    iter_lim: int = 3000


def codes_from_lists(lists: Sequence[Sequence], code: dict, width: int) -> np.ndarray:
    """Per-event lists of player ids -> an ``(n, width)`` code array (``-1`` = empty), keeping the
    first ``width`` ids of each list."""
    out = np.full((len(lists), width), -1, dtype=np.int32)
    for i, lst in enumerate(lists):
        c = [code[p] for p in lst[:width]]
        out[i, :len(c)] = c
    return out


def fit_apm(events: Events, cfg: APMConfig, before: int, season: int) -> pd.DataFrame:
    """Player values from the events with ``t < before`` in seasons ``season - window ..
    season`` (``season`` is the fit's current season, for the recency weights).

    Returns ``side, player_id, group, events, value``: one row per (side, player) with his own
    coefficient, ``events`` his count in the window and ``value`` his per-event value over his
    group's replacement level on that side."""
    fr = events.frame
    sel = ((fr.t < before) & (fr.season >= season - cfg.window)).to_numpy()
    win = fr[sel]
    w = cfg.decay ** (season - win.season.to_numpy())
    if cfg.weight is not None:
        w = w * win[cfg.weight].to_numpy(dtype=float)
    n_ids, G = len(events.ids), len(cfg.groups)
    groups = np.array(cfg.groups)
    scale = np.array([cfg.group_scale.get(g, 1.0) for g in cfg.groups])
    tabs, blocks, scales, signs = [], [], [], {}
    offset = 0
    for side, sign in cfg.sides:
        codes = events.sides[side][sel]
        valid = (codes >= 0)
        valid &= events.group[np.where(valid, codes, 0)] >= 0
        if cfg.count_weight:
            ew = np.broadcast_to(win[cfg.weight].to_numpy(dtype=float)[:, None], codes.shape)
            cnt = np.bincount(codes[valid], weights=ew[valid], minlength=n_ids)
        else:
            cnt = np.bincount(codes[valid], minlength=n_ids)
        own = (cnt >= cfg.min_events) & (events.group >= 0)
        own_col = np.full(n_ids, -1)
        own_col[own] = np.arange(own.sum())
        n_own = int(own.sum())
        # a player without his own column uses his side's replacement column for his group
        col_of = np.where(own, own_col, n_own + np.maximum(events.group, 0))
        r, c = np.nonzero(valid)
        cols = col_of[codes[r, c]]
        col_scale = np.concatenate([scale[events.group[own]], scale])
        blocks.append(sp.csr_matrix((col_scale[cols], (r, cols)), shape=(len(win), n_own + G)))
        scales.append(col_scale)
        tabs.append(pd.DataFrame({"side": side, "player_id": events.ids[own], "group": groups[events.group[own]],
                                  "events": cnt[own], "_col": offset + own_col[own],
                                  "_repl": offset + n_own + events.group[own]}))
        signs[side] = sign
        offset += n_own + G
    X = sp.hstack([*blocks, cfg.controls(win)]).tocsr()
    y = win[cfg.target].to_numpy()
    if cfg.target_clip is not None:
        y = np.clip(y, -cfg.target_clip, cfg.target_clip)
    sw = np.sqrt(w)
    A = sp.diags(sw) @ X
    b = lsqr(A, sw * (y - np.average(y, weights=w)), damp=np.sqrt(cfg.lam), atol=cfg.atol, btol=cfg.btol,
             iter_lim=cfg.iter_lim)[0]
    b = b[:offset] * np.concatenate(scales)          # back to target units per event
    tab = pd.concat(tabs, ignore_index=True)
    d = b[tab._col.to_numpy()] - b[tab._repl.to_numpy()]
    tab["value"] = d * tab.side.map(signs).to_numpy(dtype=float)   # exact for signs of +-1
    return tab.drop(columns=["_col", "_repl"])


def fit_weekly(events: Events, cfg: APMConfig, befores: Sequence[int], season_of: Callable[[int], int],
               jobs: int = 1) -> list[pd.DataFrame]:
    """:func:`fit_apm` at each cut-off in ``befores`` (``season_of(before)`` is its season), in
    parallel; each fit sees only the events before its cut-off."""
    from joblib import Parallel, delayed

    return Parallel(n_jobs=jobs)(delayed(fit_apm)(events, cfg, t, season_of(t)) for t in befores)


def shrunk_rates(df: pd.DataFrame, key: str, target: str, season: int, shrink: float, decay: float = 0.7
                 ) -> pd.DataFrame:
    """Per-attempt values for specialists: each ``key``'s recency-weighted (``decay ** seasons back``)
    sum of ``target`` over the league's weighted mean, divided by its weight plus ``shrink``
    pseudo-attempts at the mean. ``df`` holds the attempts already in the fit's window. Returns
    ``key, weight, value`` (one row per ``key``, sorted by it)."""
    sw = decay ** (season - df.season.to_numpy())
    mu = np.average(df[target], weights=sw)
    g = df.assign(w=sw, we=sw * (df[target] - mu)).groupby(key)[["w", "we"]].sum()
    v = g.we / (g.w + shrink)
    return pd.DataFrame({key: g.index, "weight": g.w.to_numpy(), "value": v.to_numpy()})
