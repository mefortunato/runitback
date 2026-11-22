"""College foundation folds (C2): a tiny generator per gender trained end to end on synthetic games, through
the generic fold code, with 360+ teams and the pooled non-Division I opponent."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from runitback.core import foundation as core
from runitback.nba.data.ratings import build_ratings
from runitback.nba.engine.rules import Outcome, apply_outcome, elapsed_range, legal_results, legal_types, new_game
from runitback.nba.foundation import GENERATORS, BasketballGenerator, prepare
from runitback.nba.league import CBB, WCBB
from runitback.nba.model.dataset import build_team_seasons, collate, encode_games
from runitback.nba.sim.sampler import Simulator

N_TEAMS = 362
TEAMS = [f"School {i}" for i in range(N_TEAMS)] + ["Non-D1"]


def _random_game(rng, lg, gid, season, home, away, date):
    s = new_game(home, away, season, league=lg)
    rows = []
    while not s.over and len(rows) < 500:
        t = rng.choice(legal_types(s))
        r = rng.choice(legal_results(s, t))
        lo, hi = elapsed_range(s, t)
        e = int(hi if t == "run_out" else rng.integers(min(lo + 4, hi), hi + 1))
        rows.append(dict(game_id=gid, idx=len(rows), qtr=s.qtr, qsec=s.qsec, phase=s.phase, pos=s.pos,
                         fts_left=s.fts_left, score_h=s.score[0], score_a=s.score[1], to_h=s.timeouts[0],
                         to_a=s.timeouts[1], foul_h=s.fouls[0], foul_a=s.fouls[1], oo=int(s.one_and_one),
                         arrow=-1 if s.arrow is None else s.arrow, play_type=str(t), result=str(r), elapsed=e))
        s = apply_outcome(s, Outcome(str(t), str(r), e))
    meta = dict(game_id=gid, season=season, game_type="REG", game_date=date, home=home, away=away,
                home_final=s.score[0], away_final=s.score[1], tip_winner=0, home_spread=None, poss_h=70, poss_a=70,
                neutral=False)
    return meta, rows


def _synthetic(tmp: Path, lg, through: dict[int, int]) -> Path:
    rng = np.random.default_rng(0)
    metas, plays = [], []
    for season, n in through.items():
        for i in range(n):
            home, away = TEAMS[(2 * i) % len(TEAMS)], TEAMS[(2 * i + 1 + i // 2) % len(TEAMS)]  # every team plays
            date = (pd.Timestamp(f"{season - 1}-11-10") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            m, r = _random_game(rng, lg, f"{season}{i:04d}", season, str(home), str(away), date)
            metas.append(m)
            plays += r
    tmp.mkdir(parents=True, exist_ok=True)
    games = pd.DataFrame(metas)
    games.to_parquet(tmp / "games.parquet")
    pd.DataFrame(plays).to_parquet(tmp / "plays.parquet")
    build_ratings(games, tmp, lg)
    return tmp


def tiny(lg, **kw) -> BasketballGenerator:
    return BasketballGenerator(lg, epochs=1, d_model=16, n_layers=1, n_heads=2, batch_size=8, min_season=2000, **kw)


def test_generators_by_league():
    assert set(GENERATORS) == {"nba", "cbb", "wcbb", "wnba"}
    assert isinstance(GENERATORS["cbb"], core.Generator) and GENERATORS["wcbb"].sport == "wcbb"
    assert GENERATORS["cbb"].settings["val_fraction"] == 0.4 and CBB.train_seasons == WCBB.train_seasons == 5


@pytest.mark.parametrize("lg", [CBB, WCBB], ids=["men", "women"])
def test_fold_never_sees_its_season_and_pca_fits_before_it(tmp_path, lg):
    data = _synthetic(tmp_path / "d", lg, {2024: 185, 2025: 16, 2026: 8})
    root = tmp_path / "folds"
    gen = tiny(lg)
    assert gen.current_season(data) == 2026
    assert prepare(root, data, [2026], gen=gen) == {2026: "trained, embedded"}
    d = core.fold_dir(root, 2026)
    ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
    assert ck["val_season"] == 2025 and ck["league"] == lg.name and ck["sport"] == lg.name
    assert all(int(k.rsplit("_", 1)[1]) <= 2025 for k in ck["team_seasons"])     # never trained on 2026
    assert len(ck["teams"]) > 250 and "Non-D1" in ck["teams"]                    # every school has its own embedding
    assert set(ck["teams"]) <= set(pd.read_parquet(data / "games.parquet").query("season <= 2025").home) | set(
        pd.read_parquet(data / "games.parquet").query("season <= 2025").away)
    assert ck["config"]["league"] == lg.name and ck["config"]["feature_sizes"]["qsec"] == lg.n_qsec

    table = pd.read_parquet(d / "pbp_32.parquet")
    games = pd.read_parquet(data / "games.parquet")
    assert len(table) == 2 * len(games) and list(table.columns[:3]) == core.KEYS
    sides = table.groupby("game_id").team.apply(set)
    assert all(sides[g.game_id] == {g.home, g.away} for g in games.itertuples())

    raw = gen.embed(d / "model.pt", data)
    red, proj = core.reduce(raw, 2026, 32)
    np.testing.assert_allclose(core.load_projection(d, 32)["mean"], proj["mean"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(table.filter(like="_pbp_").to_numpy(), red.filter(like="_pbp_").to_numpy(),
                               rtol=1e-4, atol=1e-4)
    more = _synthetic(tmp_path / "d2", lg, {2024: 185, 2025: 16, 2026: 14})      # 6 more games of 2026: same PCA
    again = prepare(root, more, [2026], train=False, gen=gen)
    assert again == {2026: "6 new games"}
    after = pd.read_parquet(d / "pbp_32.parquet")
    pd.testing.assert_frame_equal(after.iloc[: len(table)], table)


def test_fold_trains_on_the_leagues_window_of_seasons(tmp_path):
    lg = replace(CBB, train_seasons=2)
    data = _synthetic(tmp_path / "d", lg, {2022: 10, 2023: 10, 2024: 10, 2025: 10, 2026: 4})
    gen = tiny(lg)
    ckpt = gen.train(data, tmp_path / "w", 2025)                                  # fold 2026: seasons 2024 and 2025
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert sorted({int(k.rsplit("_", 1)[1]) for k in ck["team_seasons"]}) == [2024, 2025]


def test_unseen_teams_and_seasons_are_unk_and_the_batch_is_the_games_own(tmp_path):
    lg = CBB
    data = _synthetic(tmp_path / "d", lg, {2025: 12})
    games, plays = pd.read_parquet(data / "games.parquet"), pd.read_parquet(data / "plays.parquet")
    few = games.head(4)
    ts = build_team_seasons(few, lg)
    assert sorted(ts.teams) == sorted(set(few.home) | set(few.away)) and len(ts) == len(set(few.home) | set(few.away)) + 1
    items = encode_games(plays[plays.game_id.isin(games.game_id)], games, ts, 512)
    unseen = set(games.home) | set(games.away) - set(ts.teams)
    assert unseen and all(int(i["f_off_team"].min()) >= 0 for i in items)
    b = collate(items[:3])
    assert set(b["feats"]) == set(lg.features) | {"cont"} and b["result_mask"].shape[-1] == lg.n_results
    # the first team of every game is a known team (not UNK) only when the vocabulary has it
    for it, gid in zip(items, games.game_id):
        g = games.set_index("game_id").loc[gid]
        known = g.home in ts.team_idx
        first_home = it["f_off_team"][it["f_is_home"] == 1]
        assert (first_home == ts.team(g.home)).all() and (first_home > 0).all() == known if len(first_home) else True


@pytest.mark.parametrize("lg", [CBB, WCBB], ids=["men", "women"])
def test_sampler_plays_legal_college_games(tmp_path, lg):
    data = _synthetic(tmp_path / "d", lg, {2025: 12})
    ckpt = tiny(lg).train(data, tmp_path / "w", 2025)
    sim = Simulator.load(ckpt)
    assert sim.ts.league is lg and sim.model.cfg.league == lg.name
    start = new_game("School 1", "School 2", 2025, league=lg)
    finals, logs = sim.rollout(start, 4, seed=1, keep_plays=2, ratings=None)
    assert len(finals) == 4 and all(s.league is lg for s in finals)
    state, outcome = logs[0][0]
    assert outcome.play_type == "jump_ball" and state.phase == 0                   # the opening tip
    for s, o in logs[0]:
        assert o.play_type in legal_types(s) and o.result in legal_results(s, o.play_type)
        assert elapsed_range(s, o.play_type)[0] <= o.elapsed <= elapsed_range(s, o.play_type)[1]
    done = [f for f in finals if f.over]
    assert all(f.score[0] != f.score[1] for f in done)
