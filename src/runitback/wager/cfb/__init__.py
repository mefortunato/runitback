from ..core.efficiency import cfb_efficiency
from ..core.sport import Sport
from .data.download import download
from .data.features import SCHEMA

SPORT = Sport(
    name="cfb",
    schema=SCHEMA,
    download=download,
    efficiency=cfb_efficiency,
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
    },
)
