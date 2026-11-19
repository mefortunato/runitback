"""The college football simulator: the NFL's play sampler under the college rules engine.

The model is the NFL play transformer trained on college plays (team-level: no personnel data), so
sampling is the NFL's; what changes is the engine that says which outcomes are legal and what state
they lead to (``runitback.cfb.engine.rules``: tries from the 3, untimed overtime from the 25).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from ...nfl.sim.sampler import Simulator as _NFLSimulator
from ..engine import rules as cfb_rules


class Simulator(_NFLSimulator):
    engine = cfb_rules

    def team_names(self) -> dict[str, str]:
        """{ESPN team id: school name} for the teams the checkpoint knows (``attach_team_names``)."""
        return dict(self.meta.get("team_names") or {})

    def teams(self, season: int | None = None) -> list[str]:
        """The ESPN ids of the teams with a team-season token in ``season`` (default: the latest)."""
        season = season if season is not None else (self.seasons[-1] if self.seasons else None)
        return sorted({k.rsplit("_", 1)[0] for k in self.ts.keys if int(k.rsplit("_", 1)[1]) == season})


def attach_team_names(ckpt: Path, data_dir: Path) -> dict[str, str]:
    """Add ``team_names`` ({ESPN team id: school}) from the processed games to a checkpoint, in place,
    so the server can list schools without the data. Later games win (schools rename)."""
    games = pd.read_parquet(Path(data_dir) / "games.parquet", columns=["season", "home", "away", "home_name", "away_name"])
    games = games.sort_values("season")
    names: dict[str, str] = {}
    for side in ("home", "away"):
        for tid, name in zip(games[side].astype(str), games[f"{side}_name"]):
            if isinstance(name, str) and name:
                names[tid] = name
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ck["team_names"] = names
    torch.save(ck, ckpt)
    return names
