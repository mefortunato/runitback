import pandas as pd
import pytest
import torch

from runitback.nba.data.ratings import RatingsTable, compute_ratings
from runitback.nba.model.features import CONT_FEATURES, TeamSeasonVocab, feature_sizes
from runitback.nba.model.transformer import EventTransformer, ModelConfig
from runitback.nba.sim.sampler import Simulator


def toy_games() -> pd.DataFrame:
    rows = [("g1", 2025, "2024-10-22", "BOS", "NYK", 132, 109, 98, 97),
            ("g2", 2025, "2024-11-13", "NYK", "BOS", 113, 108, 95, 96),
            ("g3", 2026, "2025-10-22", "BOS", "NYK", 105, 111, 99, 100)]
    return pd.DataFrame(rows, columns=["game_id", "season", "game_date", "home", "away", "home_final",
                                       "away_final", "poss_h", "poss_a"])


@pytest.fixture(scope="session")
def nba_sim():
    torch.manual_seed(0)  # untrained weights: fix them so test runs are repeatable
    ts = TeamSeasonVocab(["BOS_2026", "NYK_2026"])
    model = EventTransformer(ModelConfig(feature_sizes(len(ts)), d_model=32, n_layers=1, n_heads=2,
                                         max_len=700, n_cont=len(CONT_FEATURES)))
    return Simulator(model, ts, ratings=RatingsTable(compute_ratings(toy_games())))
