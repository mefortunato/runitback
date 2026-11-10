from ..core.efficiency import basketball_efficiency
from ..core.sport import Sport
from .data.download import download
from .data.features import SCHEMA

SPORT = Sport(
    name="nba",
    schema=SCHEMA,
    download=download,
    efficiency=basketball_efficiency,
    gnn={"hidden_dim": 128, "out_dim": 64, "num_layers": 3},
    # about half a season of games; the total as a second output serves the
    # over/under. On 2024-2026 walk-forward backtests (Brier, paired 95% CI)
    # history 20 was no better (+0.0009 +/- 0.0016) and opponent strength
    # changed nothing (-0.0007 +/- 0.0014)
    # past-line tokens + efficiency: -0.0028 +/- 0.0017 Brier on 2024-2026
    # (experiments/nba)
    transformer={
        "history": 40,
        "total_weight": 1.0,
        "market_tokens": True,
        "efficiency": True,
    },
)
