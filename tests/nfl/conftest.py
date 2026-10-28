import pandas as pd
import torch
import pytest

from runitback.nfl.data.ratings import RatingsTable, compute_ratings
from runitback.nfl.model.features import CONT_FEATURES, TeamSeasonVocab, feature_sizes
from runitback.nfl.model.transformer import ModelConfig, PlayTransformer
from runitback.nfl.sim.sampler import Simulator


def toy_games() -> pd.DataFrame:
    rows = [("g1", 2024, 1, "2024-09-08", "KC", "BUF", 30, 10, 0.25, -0.10),
            ("g2", 2024, 2, "2024-09-15", "BUF", "KC", 24, 21, 0.05, 0.02),
            ("g3", 2025, 1, "2025-09-07", "KC", "BUF", 17, 20, -0.05, 0.10)]
    return pd.DataFrame(rows, columns=["game_id", "season", "week", "game_date", "home", "away",
                                       "home_score", "away_score", "home_epa", "away_epa"])


@pytest.fixture(scope="session")
def tiny_sim():
    torch.manual_seed(0)  # untrained weights: fix them so test runs are repeatable
    ts = TeamSeasonVocab(["KC_2025", "BUF_2025"])
    model = PlayTransformer(ModelConfig(feature_sizes(len(ts)), d_model=32, n_layers=1, n_heads=2,
                                        n_cont=len(CONT_FEATURES)))
    return Simulator(model, ts, ratings=RatingsTable(compute_ratings(toy_games())))
