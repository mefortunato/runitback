import os

from ..core.efficiency import basketball_efficiency
from ..core.sport import Sport
from .data.download import download
from .data.features import SCHEMA
from .lineup import LINEUP

# W2 (experiments/wnba): the player-value lineup terms (pvc set, 4 seeds), as the NBA's. On when
# RUNITBACK_WNBA_PLAYER_VALUES names the directory `runitback wnba player-values` writes (pv.parquet); off
# otherwise, so nothing changes until a pipeline builds that table and sets it.
PLAYER_VALUES = os.environ.get("RUNITBACK_WNBA_PLAYER_VALUES", "").rstrip("/")
PV = ({"lineup": f"{PLAYER_VALUES}/pv.parquet", "lineup_set": "pvc", "ensemble": 4}
      if PLAYER_VALUES else {})

SPORT = Sport(
    name="wnba",
    schema=SCHEMA,
    download=download,
    efficiency=basketball_efficiency,
    # the player-value lineup sets (pv / pva / pvc; runitback wnba player-values, experiments/wnba)
    lineup=LINEUP,
    gnn={"hidden_dim": 128, "out_dim": 64, "num_layers": 3},
    # 40-44 games a season: about a season of history, as in the NBA's 40
    # past-line tokens + efficiency: -0.0019 +/- 0.0041 Brier on 2024-2026
    # (experiments/wnba)
    transformer={
        "history": 40,
        "total_weight": 1.0,
        "market_tokens": True,
        "efficiency": True,
        **PV,
    },
    # tuned on 2008-2020 (Brier 0.2173 against the NFL settings' 0.2187):
    # rosters turn over more between seasons
    elo={"k": 14.0, "regress": 0.5},
)
