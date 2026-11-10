from ..cbb.data.features import SCHEMA
from ..core.efficiency import basketball_efficiency
from ..core.sport import Sport
from .data.download import download

SPORT = Sport(
    name="wcbb",
    schema=SCHEMA,
    download=download,
    efficiency=basketball_efficiency,
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
    },
)
