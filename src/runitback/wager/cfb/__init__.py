import os

from ..core.efficiency import cfb_efficiency
from ..core.sport import Sport
from .data.download import download
from .data.features import SCHEMA
from .lineup import LINEUP

# F3 (experiments/cfb): pbp32_qb, the box-score model plus the F1 foundation's frozen game embeddings
# and the starting quarterback's value change vs the team's last 7 starters (sign-constrained additive
# terms on margin and total, 4 seeds). Passes the non-inferiority bar on moneyline, spread and total
# (4 seeds vs 4 seeds, all within +-0.0003 Brier of prod) and moves the prediction the right way for
# every QB change; not better than prod. On only when BOTH variables are set; off otherwise, so nothing
# changes until a pipeline builds the tables:
#   RUNITBACK_CFB_FOUNDATION   the fold directory `runitback cfb foundation` keeps, with
#                              `runitback cfb wager-embeddings` run on it (fold<S>/pbp_32_wager.parquet)
#   RUNITBACK_CFB_PLAYER_VALUES  the directory `runitback cfb player-values` writes (pv.parquet)
FOUNDATION = os.environ.get("RUNITBACK_CFB_FOUNDATION", "").rstrip("/")
PLAYER_VALUES = os.environ.get("RUNITBACK_CFB_PLAYER_VALUES", "").rstrip("/")
PBP32_QB = ({"pbp": f"{FOUNDATION}/fold{{season}}/pbp_32_wager.parquet", "pbp_dim": 32,
             "lineup": f"{PLAYER_VALUES}/pv.parquet", "lineup_set": "pvc", "ensemble": 4}
            if FOUNDATION and PLAYER_VALUES else {})

SPORT = Sport(
    name="cfb",
    schema=SCHEMA,
    download=download,
    efficiency=cfb_efficiency,
    # the quarterback-value lineup sets (pv / pva / pvc; runitback cfb player-values, experiments/cfb)
    lineup=LINEUP,
    gnn={"hidden_dim": 128, "out_dim": 64, "num_layers": 3},
    # ~2 seasons of games, and each game's pre-game Elo of both teams: CFB
    # schedules differ far more in strength than the NFL's. On 2021-2025
    # walk-forward backtests (Brier, paired 95% CI) opponent strength gained
    # -0.0055 +/- 0.0026 over history 16 alone and history 28 another
    # -0.0008 +/- 0.0015; d_model 128 changed nothing (+0.0002)
    # past-line tokens + efficiency: -0.0014 +/- 0.0021 Brier on 2023-2025
    # (experiments/cfb; efficiency alone: -0.0020 +/- 0.0020), on as in
    # every sport
    transformer={
        "history": 28,
        "total_weight": 1.0,
        "opponent_strength": True,
        "market_tokens": True,
        "efficiency": True,
        **PBP32_QB,
    },
)
