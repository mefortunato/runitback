from ..core.efficiency import basketball_efficiency
from ..core.sport import Sport
from .data.download import download
from .data.features import SCHEMA

SPORT = Sport(
    name="cbb",
    schema=SCHEMA,
    download=download,
    efficiency=basketball_efficiency,
    gnn={"hidden_dim": 128, "out_dim": 64, "num_layers": 3},
    # about a season of games, and each game's pre-game Elo of both teams:
    # like college football's, 360+ Division I schedules differ far more in
    # strength than a pro league's
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
