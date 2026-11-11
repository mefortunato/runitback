"""``<sport>/players.json``: this season's players ranked by player value within each position group.

A sport's player-values build (``runitback <sport> player-values``) writes ``values_latest.csv``: the
latest weekly, leak-free fit of every player's value. That table also holds players who are no longer
playing, so each sport's adapter joins it with who played this season (and for which team), keeps
the players with a minimum of playing time and ranks them within their position group.

Only the NFL has an adapter so far; a sport is added to :data:`ADAPTERS` with a function from its
values directory and processed data directory to a :class:`Ranks`.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

NFL_GROUPS = [  # (group, label, side)
    ("QB", "Quarterbacks", "off"), ("RB", "Running backs", "off"), ("WR", "Wide receivers", "off"),
    ("TE", "Tight ends", "off"), ("OL", "Offensive line", "off"),
    ("DL", "Defensive line", "def"), ("LB", "Linebackers", "def"), ("DB", "Defensive backs", "def"),
    ("K", "Kickers", "special"), ("P", "Punters", "special"),
]
MIN_SNAPS = {"K": 30, "P": 30}   # kickers / punters play ~10 snaps a game
DEFAULT_MIN_SNAPS = 100          # about two games of a starter


@dataclass
class Ranks:
    season: int
    through: int                  # the last week played
    unit: str                     # what ``value`` is, for the page
    detail: str                   # what ``rate`` is
    positions: list[dict]

    def json(self) -> dict:
        return {"season": self.season, "through": self.through, "unit": self.unit,
                "detail": self.detail, "positions": self.positions}


def nfl_ranks(values_dir: pathlib.Path, processed_dir: pathlib.Path) -> Ranks:
    """The current NFL season's players by group, ranked by points per game as a starter over a
    replacement-level player (``values_latest.csv``'s ``pts_as_starter``)."""
    values = pd.read_csv(values_dir / "values_latest.csv")
    pg = pd.read_parquet(processed_dir / "player_games.parquet",
                         columns=["season", "week", "team", "player_id", "group", "main_snaps"])
    season = int(pg.season.max())
    cur = pg[pg.season == season].sort_values("week")
    played = cur.groupby("player_id").agg(team=("team", "last"), games=("week", "nunique"),
                                          snaps=("main_snaps", "sum"))
    table = values.merge(played, left_on="player_id", right_index=True)
    positions = []
    for group, label, side in NFL_GROUPS:
        rows = table[(table.group == group) & (table.snaps >= MIN_SNAPS.get(group, DEFAULT_MIN_SNAPS))]
        rows = rows.sort_values("pts_as_starter", ascending=False)
        positions.append({
            "key": group, "label": label, "side": side,
            "players": [{"id": r.player_id, "name": r.name, "team": r.team, "games": int(r.games),
                         "snaps": int(r.snaps), "value": round(float(r.pts_as_starter), 3),
                         "rate": round(float(r.value_play), 4), "rank": i + 1}
                        for i, r in enumerate(rows.itertuples())]})
    return Ranks(season, int(cur.week.max()), "points per game as a starter, over a replacement-level player",
                 "EPA per play, over replacement", [p for p in positions if p["players"]])


ADAPTERS: dict[str, Callable[[pathlib.Path, pathlib.Path], Ranks]] = {"nfl": nfl_ranks}


def player_ranks(sport: str, values_dir: str | pathlib.Path, processed_dir: str | pathlib.Path) -> dict | None:
    """The sport's ``players.json`` payload; None for a sport without an adapter."""
    adapter = ADAPTERS.get(sport)
    return adapter(pathlib.Path(values_dir), pathlib.Path(processed_dir)).json() if adapter else None
