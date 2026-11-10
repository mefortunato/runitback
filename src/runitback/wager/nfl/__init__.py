import os

from ..core.efficiency import nfl_efficiency
from ..core.sport import Sport
from .data.download import download
from .data.features import SCHEMA
from .lineup import LINEUP

# The play-by-play foundation's frozen game embeddings in the history tokens (experiments/nfl:
# pbp32, at least as good as the box-score model on every market): on when this names the fold
# directory `runitback nfl foundation` keeps (one embedding table per test season).
FOUNDATION = os.environ.get("RUNITBACK_NFL_FOUNDATION", "").rstrip("/")
PBP = {"pbp": f"{FOUNDATION}/fold{{season}}/pbp_32.parquet", "pbp_dim": 32} if FOUNDATION else {}
# Player values in the head (experiments/nfl: pbp32_pvc, 4 seeds): each position group's change in
# the projected lineup's value vs the team's last 8 lineups, through sign-constrained additive terms
# on the margin and total, so a better player at any position moves the prediction his team's way
# (-0.0018 Brier on moneylines over 2021-2025, -0.011 where the quarterback changes; spread and
# total unchanged). On with the foundation when this names the directory `runitback nfl
# player-values` writes (pv.parquet, rebuilt by Train and Refresh predictions).
PLAYER_VALUES = os.environ.get("RUNITBACK_NFL_PLAYER_VALUES", "").rstrip("/")
PV = ({"lineup": f"{PLAYER_VALUES}/pv.parquet", "lineup_set": "pvc", "ensemble": 4}
      if PLAYER_VALUES and FOUNDATION else {})

SPORT = Sport(
    name="nfl",
    schema=SCHEMA,
    download=download,
    efficiency=nfl_efficiency,
    lineup=LINEUP,
    # tuned on 2025 validation: early stopping otherwise hits at epoch 1
    gnn={
        "hidden_dim": 128,
        "out_dim": 64,
        "num_layers": 3,
        "dropout": 0.2,
        "lr": 3e-4,
        "weight_decay": 1e-4,
    },
    # about one season of games, so openers still see a full season; the
    # total as a second output serves the over/under and left the margin's
    # backtest Brier unchanged (+0.0001 +/- 0.0026 on 2021-2025), while adding
    # every box-score stat made it worse (+0.0018 +/- 0.0030)
    # past-line tokens, efficiency stats and the Optuna search's settings
    # (experiments/nfl): -0.0022 +/- 0.0022 Brier on 2016-2025 vs the old
    # 2-layer, 20-game default
    transformer={
        "history": 14,
        "total_weight": 1.0,
        "market_tokens": True,
        "efficiency": True,
        "d_model": 128,
        "nhead": 8,
        "num_layers": 1,
        "hidden_dim": 128,
        "dropout": 0.27,
        "lr": 1.4e-4,
        "weight_decay": 0.04,
        "batch_size": 128,
        **PBP,
        **PV,
    },
)
