import hashlib
import json

import pandas as pd
import pytest

from runitback.nfl.model.dataset import build_team_seasons, encode_games
from runitback.nfl.sim.season import generate_season


def test_generate_season_writes_training_data(tmp_path, tiny_sim):
    schedule = pd.DataFrame([{"week": 1, "home_team": "KC", "away_team": "BUF"}])
    out = tmp_path / "synthetic"
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")

    metadata = generate_season(tiny_sim, schedule, 2025, out, seed=17, checkpoint=str(checkpoint))

    games = pd.read_parquet(out / "games.parquet")
    plays = pd.read_parquet(out / "plays.parquet")
    ratings = pd.read_parquet(out / "ratings.parquet")
    assert metadata["games"] == len(games) == 1
    assert metadata["plays"] == len(plays) > 0
    assert metadata["ratings_included"]
    assert metadata["generator_checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert json.loads((out / "generation.json").read_text())["generator_checkpoint"] == str(checkpoint)
    assert plays.game_id.nunique() == len(games)
    assert plays.groupby("game_id").idx.apply(list).iloc[0] == list(range(len(plays)))
    assert len(ratings) == 2

    ts = build_team_seasons(games)
    items = encode_games(plays, games, ts, tiny_sim.model.cfg.max_len, ratings)
    assert len(items) == 1 and len(items[0]["type"]) == len(plays)

    second = generate_season(tiny_sim, schedule, 2025, tmp_path / "second", seed=18)
    second_games = pd.read_parquet(tmp_path / "second" / "games.parquet")
    assert second["games"] == 1 and games.game_id.iloc[0] != second_games.game_id.iloc[0]


def test_generate_season_rejects_existing_output(tmp_path, tiny_sim):
    schedule = pd.DataFrame([{"week": 1, "home": "KC", "away": "BUF"}])
    out = tmp_path / "synthetic"
    generate_season(tiny_sim, schedule, 2025, out, seed=2)

    with pytest.raises(FileExistsError, match="force=True"):
        generate_season(tiny_sim, schedule, 2025, out, seed=2)


def test_generate_season_rejects_invalid_team(tiny_sim, tmp_path):
    schedule = pd.DataFrame([{"week": 1, "home": "XXX", "away": "BUF"}])

    with pytest.raises(ValueError, match="unknown team"):
        generate_season(tiny_sim, schedule, 2025, tmp_path / "synthetic")


def test_randomized_strengths_shuffle_preseason_profiles(tiny_sim, tmp_path):
    schedule = pd.DataFrame([{"week": 1, "home": "KC", "away": "BUF"}])
    out = tmp_path / "randomized"

    metadata = generate_season(tiny_sim, schedule, 2025, out, seed=18, strength_mode="randomized")

    ratings = pd.read_parquet(out / "ratings.parquet").set_index("team")
    preseason_profiles = {
        team: tiny_sim.ratings.latest(team, 2025, week=1)
        for team in ("KC", "BUF")
    }
    generated_profiles = {
        team: {key: ratings.at[team, f"pre_{key}"] for key in ("off_epa", "def_epa", "elo")}
        for team in ("KC", "BUF")
    }
    assert metadata["strength_mode"] == "randomized"
    assert generated_profiles["KC"] == preseason_profiles["BUF"]
    assert generated_profiles["BUF"] == preseason_profiles["KC"]


def test_neutral_strengths_remove_rating_differentials(tiny_sim, tmp_path):
    schedule = pd.DataFrame([{"week": 1, "home": "KC", "away": "BUF"}])
    out = tmp_path / "neutral"

    metadata = generate_season(tiny_sim, schedule, 2025, out, strength_mode="neutral")

    ratings = pd.read_parquet(out / "ratings.parquet")
    assert metadata["strength_mode"] == "neutral"
    assert (ratings.pre_off_epa == 0).all()
    assert (ratings.pre_def_epa == 0).all()
    assert (ratings.pre_elo == 1505).all()