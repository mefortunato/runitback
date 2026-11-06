from ..core.efficiency import nfl_efficiency
from ..core.sport import Sport
from .data.download import download
from .data.features import SCHEMA

SPORT = Sport(
    name="nfl",
    schema=SCHEMA,
    download=download,
    efficiency=nfl_efficiency,
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
    },
)
