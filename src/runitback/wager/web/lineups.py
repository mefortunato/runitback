"""The game pages' lineup editor: each upcoming game's projected lineups with every player's value,
and how much a point of lineup value moves the model's margin and total.

The layout (position groups, the editor's groups and labels) is the sport's ``Sport.lineup``
(:class:`~runitback.wager.core.lineup.LineupSpec`). The production NFL model reads player values
only through sign-constrained additive terms (the ``pvc`` lineup set, :class:`~runitback.wager.models.transformer.model.PairHead`): its predicted home
margin moves by ``margin_per_point[g] * (change of the home lineup's value in group g - the away
lineup's)`` and its total by ``total_per_point[g] * (the sum of both changes)``, exactly, whatever
the rest of the inputs. So the page recomputes the prediction for an edited lineup in the browser,
from the same values the model's lineup columns were summed from (``runitback nfl player-values``:
``lineups.parquet``, ``slots.json`` next to the model's ``pv.parquet``).
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pandas as pd

from ..core.lineup import PV_PREFIX, LineupSpec
from ..core.sport import lineup_spec

BENCH = 4   # backups listed beyond the starting slots


def _softplus(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _spec(predictor) -> LineupSpec:
    return lineup_spec(getattr(predictor.model.features, "sport", None))


def additive_weights(predictor) -> dict | None:
    """Points of predicted home margin / total per point of lineup value in each ``_pv_`` group
    column (``{"margin": [..], "total": [..]}``, one per ``Sport.lineup`` group), averaged over an
    ensemble's members; None for a model without the additive lineup terms."""
    model = getattr(predictor, "model", None)
    module, features = getattr(model, "module", None), getattr(model, "features", None)
    if module is None or features is None:
        return None
    members = list(getattr(module, "members", [module]))
    head = getattr(members[0], "head", None)
    pairs = tuple(getattr(head, "additive_pairs", ()))
    if not pairs:
        return None
    cols = list(features.context_columns)
    spec = _spec(predictor)
    margin, total = np.zeros(len(spec.groups)), np.zeros(len(spec.groups))
    for m in members:
        h = m.head
        w = _softplus(h.add_w.detach().numpy())
        u = _softplus(h.add_u.detach().numpy())
        kind = h._add_kind.numpy()
        sign = h._add_sign.numpy()
        for p, (team_col, _) in enumerate(pairs):
            g = spec.group_of_index(int(cols[team_col].removeprefix(PV_PREFIX)))
            margin[g] += w[kind[p]] / len(members)
            total[g] += sign[p] * u[kind[p]] / len(members)
    return {"margin": margin.round(5).tolist(), "total": total.round(5).tolist()}


class LineupEditor:
    """Per upcoming game, both teams' editable lineups (see the module docstring)."""

    def __init__(self, weights: dict, slots: dict, lineups: pd.DataFrame, spec: LineupSpec):
        self.spec = spec
        self.weights = weights
        self.slots = {g: [round(float(x), 3) for x in slots.get(g, [])] for g in spec.editor_groups}
        self.by_team = {k: d for k, d in lineups.groupby(["game_id", "team"])}

    @classmethod
    def load(cls, predictor) -> LineupEditor | None:
        """The editor for ``predictor``'s lineup table, or None (no lineup terms, or no editor files
        next to its ``pv.parquet``)."""
        weights = additive_weights(predictor)
        path = getattr(getattr(predictor.model, "features", None), "lineup", None) if weights else None
        if not path:
            return None
        root = pathlib.Path(path).parent
        if not (root / "lineups.parquet").exists() or not (root / "slots.json").exists():
            return None
        return cls(weights, json.loads((root / "slots.json").read_text()),
                   pd.read_parquet(root / "lineups.parquet"), _spec(predictor))

    def _team(self, game_id: str, team: str) -> dict | None:
        d = self.by_team.get((game_id, team))
        if d is None or d.empty:
            return None
        groups = {}
        for g in self.spec.editor_groups:
            rows = d[d.group == g]
            n = len(self.slots[g]) + BENCH
            active = rows[rows["rank"] >= 0].sort_values("rank").head(n)
            out = rows[rows["rank"] < 0]
            groups[g] = [_player(r, False) for r in active.itertuples()] + [_player(r, True) for r in out.itertuples()]
        return groups

    def game(self, game_id: str, home: str, away: str) -> dict | None:
        """``{"groups", "slots", "weights", "home", "away"}`` for the page, or None."""
        h, a = self._team(game_id, home), self._team(game_id, away)
        if h is None or a is None:
            return None
        return {
            "groups": [{"key": g, "label": self.spec.labels.get(g, g),
                        "column": "" if g in self.spec.rotation_groups else self.spec.column_of(g)}
                       for g in self.spec.editor_groups],
            "columns": list(self.spec.groups),
            "slots": self.slots,
            "weights": self.weights,
            "home": h,
            "away": a,
        }


def _player(r, out: bool) -> dict:
    pos = r.position if isinstance(r.position, str) else None
    name = r.name if isinstance(r.name, str) and r.name else f"Unnamed player ({pos or r.group})"
    value = float(r.value_off) + float(r.value_def) if hasattr(r, "value_off") else float(r.value_play)
    player = {"id": r.player_id, "name": name, "pos": pos,
              "value": 0.0 if not math.isfinite(value) else round(value, 5), "out": out}
    if hasattr(r, "value_off"):   # a rotation's player adds offense and defense into his position's columns
        player["values"] = {f"{r.pos_group}_OFF": round(float(r.value_off), 5),
                            f"{r.pos_group}_DEF": round(float(r.value_def), 5)}
    return player
