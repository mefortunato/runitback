"""Sport- and model-agnostic building blocks.

Kept import-light on purpose; the MLflow-backed serving layer lives in
:mod:`runitback.wager.core.predictor`.
"""

from .metrics import antisymmetrize, evaluate, fit_sigma, win_probability
from .schema import GameSchema, mirror, played, upcoming, validate
from .splits import SeasonSplit, split_seasons
from .sport import Sport, all_sports, get_sport, register_sport

__all__ = [
    "GameSchema",
    "SeasonSplit",
    "Sport",
    "all_sports",
    "antisymmetrize",
    "evaluate",
    "fit_sigma",
    "get_sport",
    "mirror",
    "played",
    "register_sport",
    "split_seasons",
    "upcoming",
    "validate",
    "win_probability",
]
