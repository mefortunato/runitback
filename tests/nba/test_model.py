from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from runitback.nba.data.preprocess import outcome_from_row, process_game, replay_agreement, state_from_row
from runitback.nba.engine.rules import apply_outcome, new_game
from runitback.nba.model.dataset import build_team_seasons, collate, encode_games
from runitback.nba.model.features import elapsed_mask, encode_state, result_mask, type_mask
from runitback.nba.sim.api import simulate_games
from runitback.nba.sim.describe import describe

ROOT = Path(__file__).resolve().parents[2] / "data" / "nba"
RAW, PROCESSED = ROOT / "raw", ROOT / "processed"


@pytest.mark.skipif(not (RAW / "pbp_2024.parquet").exists(), reason="raw data not downloaded")
def test_engine_replays_real_games():
    from runitback.nba.data.download import load_raw
    df = load_raw(2024, RAW)
    ok = tot = finals = n = 0
    for _, g in list(df.groupby("game_id"))[:40]:
        meta, rows = process_game(g)
        a, b, _ = replay_agreement(meta, rows)
        ok, tot, n = ok + a, tot + b, n + 1
        end = apply_outcome(state_from_row(meta, rows[-1]), outcome_from_row(rows[-1]))
        finals += end.over and end.score == [meta["home_final"], meta["away_final"]]
    assert ok / tot >= 0.98
    assert finals == n


@pytest.mark.skipif(not (PROCESSED / "plays.parquet").exists(), reason="data not preprocessed")
def test_training_masks_match_engine():
    games = pd.read_parquet(PROCESSED / "games.parquet").head(4)
    plays = pd.read_parquet(PROCESSED / "plays.parquet")
    plays = plays[plays.game_id.isin(games.game_id)].sort_values(["game_id", "idx"])
    ts = build_team_seasons(games)
    ratings = pd.read_parquet(PROCESSED / "ratings.parquet")
    items = encode_games(plays, games, ts, 700, ratings)
    batch = collate(items)
    meta = games.set_index("game_id")
    for gi, item in enumerate(items):
        gp = plays[plays.game_id == item["game_id"]]
        m = meta.loc[item["game_id"]].to_dict() | {"game_id": item["game_id"]}
        rg = ratings[ratings.game_id == item["game_id"]].set_index("side")
        pre = tuple({k: rg.at[sd, f"pre_{k}"] for k in ("off_rtg", "def_rtg", "elo")} for sd in ("home", "away"))
        for t, row in enumerate(gp.itertuples()):
            s, o = state_from_row(m, row), outcome_from_row(row)
            if not (type_mask(s)[batch["type"][gi, t]] and result_mask(s, o.play_type)[batch["result"][gi, t]]
                    and elapsed_mask(s, o.play_type)[o.elapsed]):
                continue  # rare illegal real target: training widens the mask to keep it learnable
            assert (batch["type_mask"][gi, t].numpy() == type_mask(s)).all()
            assert (batch["result_mask"][gi, t].numpy() == result_mask(s, o.play_type)).all()
            assert (batch["elapsed_mask"][gi, t].numpy() == elapsed_mask(s, o.play_type)).all()
            assert np.allclose(batch["feats"]["cont"][gi, t].numpy(), encode_state(s, None, ts, pre)["cont"],
                               atol=1e-4)


def test_sampler_produces_finished_legal_games(nba_sim):
    finals, logs = nba_sim.rollout(new_game("BOS", "NYK", 2026), 6, seed=0, keep_plays=2)
    assert len(finals) == 6 and all(s.over for s in finals)
    assert all(s.score[0] != s.score[1] for s in finals)  # no ties in the NBA
    assert logs[0][0][1].play_type == "jump_ball"
    for state, outcome in logs[0]:
        assert describe(state, outcome, apply_outcome(state, outcome))


def test_ratings_override_changes_inputs(nba_sim):
    base = nba_sim.ratings_for("BOS", "NYK", 2026)
    what_if = nba_sim.ratings_for("BOS", "NYK", 2026, override={"BOS": {"elo": 1800}})
    assert what_if[0]["elo"] == 1800 and what_if[0]["off_rtg"] == base[0]["off_rtg"]


def test_simulate_games_summary(nba_sim):
    r = simulate_games(nba_sim, "BOS", "NYK", 2026, n=4, seed=1)
    assert r.n == 4 and r.tie == 0 and abs(r.home_win + r.away_win - 1) < 1e-9
