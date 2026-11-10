from ..core.efficiency import basketball_efficiency
from ..core.sport import Sport
from .data.download import download
from .data.features import SCHEMA

SPORT = Sport(
    name="wnba",
    schema=SCHEMA,
    download=download,
    efficiency=basketball_efficiency,
    gnn={"hidden_dim": 128, "out_dim": 64, "num_layers": 3},
    # 40-44 games a season: about a season of history, as in the NBA's 40
    # past-line tokens + efficiency: -0.0019 +/- 0.0041 Brier on 2024-2026
    # (experiments/wnba)
    transformer={
        "history": 40,
        "total_weight": 1.0,
        "market_tokens": True,
        "efficiency": True,
    },
    # tuned on 2008-2020 (Brier 0.2173 against the NFL settings' 0.2187):
    # rosters turn over more between seasons
    elo={"k": 14.0, "regress": 0.5},
)
