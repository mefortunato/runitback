"""NBA foundation folds (runitback.nba.foundation / runitback.nba.embed): the per-possession pooling,
and a tiny fold trained end to end on synthetic games through the generic fold code."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from runitback.core import foundation as core
from runitback.nba import embed as nba_embed
from runitback.nba.data.ratings import build_ratings
from runitback.nba.engine.rules import Outcome, apply_outcome, elapsed_range, legal_results, legal_types, new_game
from runitback.nba.foundation import NBA, NBAGenerator, prepare
from runitback.nba.model.dataset import collate, encode_games
from runitback.nba.model.features import TeamSeasonVocab

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "nba" / "processed"


def test_possession_ids_split_on_offense_and_period():
    is_home = np.array([1, 1, 1, 0, 0, 1, 1, 1])
    qtr = np.array([1, 1, 1, 1, 1, 1, 2, 2])
    assert possession_ids_list(is_home, qtr) == [0, 0, 0, 1, 1, 2, 3, 3]
    assert possession_ids_list(np.array([], int), np.array([], int)) == []


def possession_ids_list(is_home, qtr):
    return nba_embed.possession_ids(is_home, qtr).tolist()


def test_pooling_is_the_mean_over_possessions():
    # one game: home has possessions of 3 events and 1 event; away one of 2 events; then padding
    is_home = torch.tensor([[1, 1, 1, 0, 0, 1, 0]])
    qtr = torch.ones_like(is_home)
    valid = torch.tensor([[1, 1, 1, 1, 1, 1, 0]], dtype=torch.float32)
    h = torch.arange(7, dtype=torch.float32).view(1, 7, 1).repeat(1, 1, 2)
    h[0, 6] = 1000.0  # padding must not count
    w = nba_embed.possession_weights(is_home, qtr, valid)
    v = nba_embed.pool_sides(h, is_home, w)
    home_off = ((0 + 1 + 2) / 3 + 5) / 2          # possession means 1 and 5
    away_off = (3 + 4) / 2
    assert v.shape == (1, 2, 4)
    assert torch.allclose(v[0, 0], torch.tensor([home_off, home_off, away_off, away_off]))
    assert torch.allclose(v[0, 1], torch.tensor([away_off, away_off, home_off, home_off]))
    # per event (the NFL's pooling) would weight the 3-event possession three times
    assert home_off != pytest.approx((0 + 1 + 2 + 5) / 4)


def _random_game(rng: np.random.Generator, gid: str, season: int, home: str, away: str, date: str):
    s = new_game(home, away, season)
    rows = []
    while not s.over and len(rows) < 700:
        t = rng.choice(legal_types(s))
        r = rng.choice(legal_results(s, t))
        lo, hi = elapsed_range(s, t)
        e = int(hi if t == "run_out" else rng.integers(min(lo + 4, hi), hi + 1))
        rows.append(dict(game_id=gid, idx=len(rows), qtr=s.qtr, qsec=s.qsec, phase=s.phase, pos=s.pos,
                         fts_left=s.fts_left, score_h=s.score[0], score_a=s.score[1], to_h=s.timeouts[0],
                         to_a=s.timeouts[1], play_type=str(t), result=str(r), elapsed=e))
        s = apply_outcome(s, Outcome(str(t), str(r), e))
    meta = dict(game_id=gid, season=season, game_type="REG", game_date=date, home=home, away=away,
                home_final=s.score[0], away_final=s.score[1], tip_winner=0, home_spread=None,
                poss_h=100, poss_a=100)
    return meta, rows


TEAMS = ["BOS", "NYK", "LAL", "GSW"]


def _synthetic(tmp: Path, through: dict[int, int]) -> Path:
    """Processed NBA data with `through[season]` played games per season (round robin of 4 teams)."""
    rng = np.random.default_rng(0)
    metas, plays = [], []
    for season, n in through.items():
        for i in range(n):
            home, away = TEAMS[i % 4], TEAMS[(i + 1 + i // 4) % 4]
            if home == away:
                away = TEAMS[(i + 2) % 4]
            date = (pd.Timestamp(f"{season - 1}-10-20") + pd.Timedelta(days=2 * i)).strftime("%Y-%m-%d")
            m, r = _random_game(rng, f"{season}{i:04d}", season, home, away, date)
            metas.append(m)
            plays += r
    tmp.mkdir(parents=True, exist_ok=True)
    games = pd.DataFrame(metas)
    games.to_parquet(tmp / "games.parquet")
    pd.DataFrame(plays).to_parquet(tmp / "plays.parquet")
    build_ratings(games, tmp)
    return tmp


TINY = NBAGenerator(epochs=1, d_model=16, n_layers=1, n_heads=2, batch_size=8, min_season=2000)


def test_generator_satisfies_the_interface():
    assert isinstance(NBA, core.Generator)
    assert NBA.settings["val_fraction"] == 0.4 and NBA.settings["epochs"] == 6


def test_fold_trains_without_its_season_and_pca_fits_before_it(tmp_path):
    data = _synthetic(tmp_path / "d1", {2024: 20, 2025: 20, 2026: 6})
    root = tmp_path / "folds"
    assert TINY.current_season(data) == 2026
    done = prepare(root, data, [2026], gen=TINY)
    assert done == {2026: "trained, embedded"}
    d = core.fold_dir(root, 2026)
    assert sorted(p.name for p in d.iterdir()) == sorted(core.fold_files(32))

    ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
    assert ck["val_season"] == 2025
    assert all(int(k.split("_")[1]) <= 2025 for k in ck["team_seasons"])  # never trained on 2026

    table = pd.read_parquet(d / "pbp_32.parquet")
    games = pd.read_parquet(data / "games.parquet")
    assert len(table) == 2 * len(games)
    assert list(table.columns[:3]) == core.KEYS and table.game_id.map(type).eq(str).all()
    sides = table.groupby("game_id").team.apply(set)
    assert all(sides[g.game_id] == {g.home, g.away} for g in games.itertuples())

    # the stored projection is the PCA of the games before 2026 only
    raw = NBA.embed(d / "model.pt", data)
    red, proj = core.reduce(raw, 2026, 32)
    stored = core.load_projection(d, 32)
    np.testing.assert_allclose(stored["mean"], proj["mean"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(table.filter(like="_pbp_").to_numpy(), red.filter(like="_pbp_").to_numpy(),
                               rtol=1e-4, atol=1e-4)
    changed = _synthetic(tmp_path / "d1b", {2024: 20, 2025: 20, 2026: 12})  # more 2026 games: same PCA
    _, proj2 = core.reduce(NBA.embed(d / "model.pt", changed), 2026, 32)
    np.testing.assert_allclose(proj2["components"], proj["components"], rtol=1e-4, atol=1e-5)

    # later: only the new games are embedded, earlier rows stay as they were
    again = prepare(root, changed, [2026], train=False, gen=TINY)
    assert again == {2026: "6 new games"}
    after = pd.read_parquet(d / "pbp_32.parquet")
    pd.testing.assert_frame_equal(after.iloc[: len(table)], table)
    assert len(after) == len(table) + 12


def test_teacher_forced_embedding_only_depends_on_its_own_game(tmp_path):
    """A game's embedding is the same whether it is embedded alone or with the rest (no batch leakage)."""
    data = _synthetic(tmp_path / "d", {2025: 10, 2026: 4})
    ckpt = TINY.train(data, tmp_path / "w", 2025)
    every = NBA.embed(ckpt, data)
    one = NBA.embed(ckpt, data, game_ids=["20260002"])
    a = every[every.game_id == "20260002"].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, one, check_exact=False, atol=1e-5)


@pytest.mark.skipif(not (PROCESSED / "plays.parquet").exists(), reason="data not preprocessed")
def test_real_games_fit_the_model_and_have_both_sides():
    games = pd.read_parquet(PROCESSED / "games.parquet")
    plays = pd.read_parquet(PROCESSED / "plays.parquet")
    n = plays.groupby("game_id").size()
    assert (n <= 512).mean() > 0.999  # the trunk's max_len: nothing of a game is cut off
    g = games[games.game_id.isin(n.index)].head(8)
    items = encode_games(plays[plays.game_id.isin(g.game_id)], g, TeamSeasonVocab([]), 512)
    b = collate(items)
    f = b["feats"]
    w = nba_embed.possession_weights(f["is_home"], f["qtr"], b["valid"])
    for i in range(len(items)):
        k = int(b["valid"][i].sum())
        pid = nba_embed.possession_ids(f["is_home"][i, :k].numpy(), f["qtr"][i, :k].numpy())
        assert 150 <= pid.max() + 1 <= 300  # ~100 possessions a team
        assert float(w[i].sum()) == pytest.approx(pid.max() + 1)


RAW = PROCESSED.parent / "raw"


@pytest.mark.skipif(not (RAW / "pbp_2002.parquet").exists(), reason="raw 2002 data not downloaded")
def test_nullable_download_preprocesses_like_the_release_file():
    """sportsdataverse's loader can return nullable dtypes (pd.NA team ids in 2002); load_raw
    normalizes them so preprocessing gives the same games either way."""
    from runitback.nba.data.download import COLUMNS, numpy_dtypes
    from runitback.nba.data.preprocess import process_game

    plain = pd.read_parquet(RAW / "pbp_2002.parquet")
    plain = numpy_dtypes(plain[[c for c in COLUMNS if c in plain]])
    nullable = plain.convert_dtypes()
    assert nullable.team_id.isna().any() and isinstance(nullable.team_id.dtype, pd.Int64Dtype)
    ids = plain.game_id.drop_duplicates().head(5)
    for gid in ids:
        a = process_game(plain[plain.game_id == gid])
        b = process_game(numpy_dtypes(nullable[nullable.game_id == gid]))
        assert a == b
