from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from runitback.nfl.data.preprocess import outcome_from_row, process_game, replay_agreement, state_from_row
from runitback.nfl.engine.rules import new_game
from runitback.nfl.model.dataset import build_team_seasons, collate, encode_games
from runitback.nfl.model.features import encode_state, result_mask, type_mask, yards_mask, elapsed_mask
from runitback.nfl.sim.api import simulate_games

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "nfl" / "processed"
RAW = Path(__file__).resolve().parents[2] / "data" / "nfl" / "raw"


@pytest.mark.skipif(not (RAW / "pbp_2024.parquet").exists(), reason="raw data not downloaded")
def test_engine_replays_real_games():
    df = pd.read_parquet(RAW / "pbp_2024.parquet")
    ok = tot = 0
    for _, g in list(df.groupby("game_id"))[:40]:
        meta, rows = process_game(g)
        a, b, _ = replay_agreement(meta, rows)
        ok, tot = ok + a, tot + b
    assert ok / tot >= 0.98


@pytest.mark.skipif(not (PROCESSED / "plays.parquet").exists(), reason="data not preprocessed")
def test_training_masks_match_engine():
    games = pd.read_parquet(PROCESSED / "games.parquet").query("season == 2024").head(5)
    plays = pd.read_parquet(PROCESSED / "plays.parquet")
    plays = plays[plays.game_id.isin(games.game_id)].sort_values(["game_id", "idx"])
    ts = build_team_seasons(games)
    ratings = pd.read_parquet(PROCESSED / "ratings.parquet")
    items = encode_games(plays, games, ts, 400, ratings)
    batch = collate(items)
    meta = games.set_index("game_id")
    for gi, item in enumerate(items):
        gp = plays[plays.game_id == item["game_id"]]
        m = meta.loc[item["game_id"]].to_dict() | {"game_id": item["game_id"]}
        for t, row in enumerate(gp.itertuples()):
            s = state_from_row(m, row)
            o = outcome_from_row(row)
            assert (batch["type_mask"][gi, t].numpy() == type_mask(s)).all()
            assert (batch["result_mask"][gi, t].numpy() == result_mask(s, o.play_type)).all()
            assert (batch["yards_mask"][gi, t].numpy() == yards_mask(s, o.play_type, o.result)).all()
            assert (batch["elapsed_mask"][gi, t].numpy() == elapsed_mask(s)).all()
            rg = ratings[ratings.game_id == item["game_id"]].set_index("side")
            pre = tuple({k: rg.at[sd, f"pre_{k}"] for k in ("off_epa", "def_epa", "elo")} for sd in ("home", "away"))
            cont = encode_state(s, None, ts, pre)["cont"]
            assert np.allclose(batch["feats"]["cont"][gi, t].numpy(), cont, atol=1e-4)


def test_sampler_produces_finished_legal_games(tiny_sim):
    finals, logs = tiny_sim.rollout(new_game("KC", "BUF", 2025, opening_kicker=0), 8, seed=0, keep_plays=2)
    assert len(finals) == 8 and all(s.over for s in finals)
    assert all(s.score[0] >= 0 and s.score[1] >= 0 for s in finals)
    first_play = next(o for _, o in logs[0] if o.play_type != "timeout")  # timeouts are legal before the kick
    assert first_play.play_type == "kickoff"


def test_ratings_override_changes_inputs(tiny_sim):
    base = tiny_sim.ratings_for("KC", "BUF", 2025)
    what_if = tiny_sim.ratings_for("KC", "BUF", 2025, override={"KC": {"elo": 1800}})
    assert what_if[0]["elo"] == 1800 and what_if[0]["off_epa"] == base[0]["off_epa"]


def test_simulate_games_summary(tiny_sim):
    r = simulate_games(tiny_sim, "KC", "BUF", 2025, n=6, seed=1)
    assert r.n == 6 and abs(r.home_win + r.away_win + r.tie - 1) < 1e-9
