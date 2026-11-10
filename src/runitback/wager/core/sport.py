from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from .lineup import LineupSpec
from .schema import GameSchema

BUILTIN_SPORTS = ("nfl", "cfb", "nba", "cbb", "wcbb", "wnba", "nhl", "mlb")

_registry: dict[str, Sport] = {}


@dataclass(frozen=True)
class Sport:
    """Everything the CLI and models need to know about one sport.

    Attributes:
        name: CLI name, e.g. ``"nfl"``.
        schema: Column roles of the sport's games table.
        download: Fetches source data and returns a games table.
        gnn: Default GNN settings (``hidden_dim``, ``out_dim``, ``lr``, ...).
        transformer: Default transformer settings (``history``, ``d_model``,
            ``lr``, ...).
        elo: The Elo baseline's settings (``k``, ``home``, ``regress``; see
            :class:`~runitback.wager.models.baselines.EloParams`) where the NFL's do
            not suit the sport.
        efficiency: Derives per-game rate stats (``_eff_*`` columns) from
            a games table's box score, for the transformer's history tokens
            (see :mod:`runitback.wager.core.efficiency`).
        lineup: The position groups behind the sport's player-value lineup
            columns (``_pv_*``), their signs on the total and the lineup
            editor's groups; None for a sport without player values (the
            ``pv``/``pva``/``pvc`` lineup sets then fail).
    """

    name: str
    schema: GameSchema
    download: Callable[..., pd.DataFrame]
    gnn: dict = field(default_factory=dict)
    transformer: dict = field(default_factory=dict)
    elo: dict = field(default_factory=dict)
    efficiency: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    lineup: LineupSpec | None = None


def register_sport(sport: Sport) -> Sport:
    """Make a sport available to :func:`get_sport` (and so to every model)."""
    _registry[sport.name] = sport
    return sport


def get_sport(name: str) -> Sport:
    if name not in _registry and name in BUILTIN_SPORTS:
        register_sport(importlib.import_module(f"runitback.wager.{name}").SPORT)
    if name not in _registry:
        known = sorted(set(BUILTIN_SPORTS) | set(_registry))
        raise ValueError(f"unknown sport {name!r}; choose from {known}")
    return _registry[name]


def lineup_spec(sport: str | Sport | LineupSpec | None) -> LineupSpec:
    """``sport``'s player-value lineup layout (``Sport.lineup``); a spec is returned as is."""
    if isinstance(sport, LineupSpec):
        return sport
    spec = (sport if isinstance(sport, Sport) else get_sport(sport)).lineup if sport else None
    if spec is None:
        raise ValueError(f"sport {getattr(sport, 'name', sport)!r} has no player-value lineup groups (Sport.lineup)")
    return spec


def all_sports() -> dict[str, Sport]:
    """Built-in sports followed by any registered with :func:`register_sport`."""
    names = list(BUILTIN_SPORTS) + [n for n in _registry if n not in BUILTIN_SPORTS]
    return {name: get_sport(name) for name in names}
