from ..core.efficiency import nhl_efficiency
from ..core.sport import Sport
from .data.download import download
from .data.features import SCHEMA

SPORT = Sport(
    name="nhl",
    schema=SCHEMA,
    download=download,
    efficiency=nhl_efficiency,
    gnn={"hidden_dim": 128, "out_dim": 64, "num_layers": 3},
    # about half a season of games, as in the NBA
    # past-line tokens + efficiency: -0.0010 +/- 0.0010 Brier on 2024-2026
    # (experiments/nhl)
    transformer={
        "history": 40,
        "total_weight": 1.0,
        "market_tokens": True,
        "efficiency": True,
    },
    # tuned on 2009-10 to 2020-21 (Brier 0.2419 against the NFL settings'
    # 0.2467): hockey results are noisy, so ratings move slowly; a goal is
    # about 120 Elo points (least squares on the same seasons)
    elo={"k": 6.0, "home": 30.0, "points_per_elo": 1 / 120},
)
