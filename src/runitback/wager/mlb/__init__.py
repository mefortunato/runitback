from ..core.efficiency import mlb_efficiency
from ..core.sport import Sport
from .data.download import download
from .data.features import SCHEMA

SPORT = Sport(
    name="mlb",
    schema=SCHEMA,
    download=download,
    efficiency=mlb_efficiency,
    gnn={"hidden_dim": 128, "out_dim": 64, "num_layers": 3},
    # about a third of a season of games
    # past-line tokens + efficiency: -0.0006 +/- 0.0008 Brier on 2024-2026
    # (experiments/mlb)
    transformer={
        "history": 50,
        "total_weight": 1.0,
        "market_tokens": True,
        "efficiency": True,
    },
    # tuned on 2010-2018 (Brier 0.2446 against the NFL settings' 0.2538); a
    # run is about 90 Elo points (least squares on the same seasons)
    elo={"k": 4.0, "home": 24.0, "points_per_elo": 1 / 90},
)
