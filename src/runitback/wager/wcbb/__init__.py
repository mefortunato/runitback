import os

from ..cbb.data.features import SCHEMA
from ..cbb.lineup import LINEUP
from ..core.efficiency import basketball_efficiency
from ..core.sport import Sport
from .data.download import download

# C4 (experiments/wcbb): the box-score model plus the player-value lineup terms (prod_pvc_e4: pvc set, 4
# seeds, the lineup weight starting at 0.2 as in the experiment). On when RUNITBACK_WCBB_PLAYER_VALUES names the
# directory `runitback college player-values --gender wcbb` writes (pv.parquet); off otherwise, so nothing
# changes until a pipeline builds that table.
PLAYER_VALUES = os.environ.get("RUNITBACK_WCBB_PLAYER_VALUES", "").rstrip("/")
PV = ({"lineup": f"{PLAYER_VALUES}/pv.parquet", "lineup_set": "pvc", "ensemble": 4, "additive_init": 0.2}
      if PLAYER_VALUES else {})

SPORT = Sport(
    name="wcbb",
    schema=SCHEMA,
    download=download,
    efficiency=basketball_efficiency,
    # the player-value lineup sets (pv / pva / pvc; runitback college player-values, experiments/cbb)
    lineup=LINEUP,
    gnn={"hidden_dim": 128, "out_dim": 64, "num_layers": 3},
    # as men's college basketball
    # past-line tokens + efficiency as in the other sports (not backtested
    # here: they helped the NBA and WNBA, which share the basketball stats)
    transformer={
        "history": 32,
        "total_weight": 1.0,
        "opponent_strength": True,
        "market_tokens": True,
        "efficiency": True,
        **PV,
    },
)
