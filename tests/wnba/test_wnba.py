"""WNBA engine and generator (W1): the league's rules, the play-by-play reader, a replay of real games, and a
fold trained end to end (leak-free, PCA before the test season, keyed as the wager model keys teams)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from runitback.core import foundation as core
from runitback.nba.data import college, wnba
from runitback.nba.data.ratings import build_ratings
from runitback.nba.engine.rules import (
    AWAY, HOME, GameState, Outcome, apply_outcome, elapsed_range, legal_results, legal_types, new_game,
)
from runitback.nba.foundation import GENERATORS, BasketballGenerator, prepare
from runitback.nba.league import CBB, LEAGUES, NBA, WCBB, WNBA, WNBA_TEAM_IDS, Bonus, get_league
from runitback.nba.model.dataset import build_team_seasons, encode_games
from runitback.nba.sim.sampler import Simulator
from runitback.nba.vocab import FREE_THROW, LIVE, TIP

FIXTURE = Path(__file__).parent / "fixtures" / "wnba_pbp_games.parquet"
RAW = Path("data/wnba/raw")


# ---------------------------------------------------------------- the league

def test_wnba_league():
    assert get_league("wnba") is WNBA and LEAGUES["wnba"] is WNBA and GENERATORS["wnba"].league is WNBA
    assert (WNBA.period_seconds, WNBA.regulation_periods, WNBA.ot_seconds, WNBA.shot_clock) == (600, 4, 300, 24)
    assert (WNBA.n_qsec, WNBA.n_results, WNBA.n_fts, WNBA.n_fouls) == (41, 19, 4, 7)
    assert not WNBA.arrow and not WNBA.dynamic_teams and WNBA.train_seasons is None and WNBA.first_season == 2006
    assert "fouls_off" in WNBA.features and WNBA.results[:18] == NBA.results and WNBA.rs["fts_0"] == 18
    # the franchises, keyed as runitback.wager.wnba keys them (latest abbreviation by ESPN id)
    assert len(WNBA.teams) == 18 and set(WNBA.teams) == set(WNBA_TEAM_IDS.values())
    assert WNBA_TEAM_IDS[3] == "DAL" and WNBA_TEAM_IDS[17] == "LV" and WNBA_TEAM_IDS[18] == "CON"
    assert {"GS", "TOR", "POR"} <= set(WNBA.teams)


def test_bonus_rules():
    b = WNBA.bonus
    assert [b.level(n) for n in (0, 4, 5, 9)] == [0, 0, 2, 2]                  # the 5th foul of a quarter
    assert [b.level(n, overtime=True) for n in (3, 4)] == [0, 2]               # the 4th of an overtime
    assert b.counted(1, 300) == 1 and b.counted(1, 120) == 4 and b.counted(6, 30) == 6
    assert b.counted(1, 30, overtime=True) == 3
    assert [b.resets(q, 4) for q in (2, 3, 4, 5, 6)] == [True] * 5            # every period, overtimes too
    # the college bonuses are unchanged: overtime keeps the count, no late rule
    for lg in (CBB, WCBB):
        assert not lg.bonus.resets(lg.regulation_periods + 1, lg.regulation_periods)
        assert lg.bonus.counted(1, 10) == 1 and lg.bonus.limit(True) == lg.bonus.limit(False)
    assert Bonus(double_at=4).counted(0, 0) == 0


# ---------------------------------------------------------------- the engine

def _state(**kw) -> GameState:
    s = new_game("LV", "NY", 2026, league=WNBA)
    s.phase = LIVE
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_quarters_overtime_and_quarter_possession():
    s = new_game("LV", "NY", 2026, league=WNBA)
    assert s.qsec == 600 and s.timeouts == [7, 7] and s.fouls == [0, 0]
    s = apply_outcome(s, Outcome("jump_ball", "lost", 0))                       # the away team wins the tip
    assert s.pos == AWAY and s.tip_winner == AWAY
    starts = []
    while s.qtr < 4 or s.qsec > 0:
        q = s.qtr
        s = apply_outcome(s, Outcome("run_out", "end", s.qsec)) if s.qsec <= 60 else \
            apply_outcome(s, Outcome("turnover", "turnover", 60))
        if s.qtr != q and not s.over:
            starts.append((s.qtr, s.pos, s.phase))
        if s.over or s.qtr > 4:
            break
    # the tip's loser starts the 2nd and 3rd quarters, its winner the 4th (as in the NBA)
    assert starts[:3] == [(2, HOME, LIVE), (3, HOME, LIVE), (4, AWAY, LIVE)]
    assert s.qtr == 5 and s.qsec == 300 and s.phase == TIP and s.timeouts == [min(7, 5) + 2] * 2  # tied: overtime


def test_the_penalty_is_the_fifth_foul_and_free_throws_are_two():
    s = _state(fouls=[0, 3], qsec=400)                                         # home has the ball, away fouls
    assert "fts_0" in legal_results(s, "foul")
    s = apply_outcome(s, Outcome("foul", "fts_0", 5))
    assert s.fouls == [0, 4] and s.pos == HOME and s.phase == LIVE
    assert legal_results(s, "foul") == ["fts_1", "fts_2", "fts_3"]           # the 5th: no "no free throws"
    s = apply_outcome(s, Outcome("foul", "fts_2", 5))
    assert s.fouls == [0, 5] and s.phase == FREE_THROW and s.fts_left == 2 and not s.one_and_one


def test_last_two_minutes_second_foul_is_a_penalty():
    s = _state(fouls=[0, 1], qsec=150)
    s = apply_outcome(s, Outcome("foul", "fts_0", 40))                         # at 1:50: the first late foul
    assert s.fouls == [0, 4] and s.phase == LIVE                               # not a penalty, but the next one is
    assert "fts_0" not in legal_results(s, "foul")
    s = _state(fouls=[0, 1], qsec=150)
    s = apply_outcome(s, Outcome("foul", "fts_0", 10))                         # at 2:20: an ordinary foul
    assert s.fouls == [0, 2] and "fts_0" in legal_results(s, "foul")
    s = _state(fouls=[2, 0], qsec=100, pos=AWAY)                               # a shooting foul counts too
    s = apply_outcome(s, Outcome("shot2", "missed_fouled", 5))
    assert s.fouls == [4, 0] and s.fts_left == 2


def test_overtime_resets_the_count_and_the_fourth_foul_is_a_penalty():
    s = _state(qtr=4, qsec=10, fouls=[6, 6], score=[70, 70])
    s = apply_outcome(s, Outcome("run_out", "end", 10))
    assert s.qtr == 5 and s.fouls == [0, 0] and s.phase == TIP
    s = apply_outcome(s, Outcome("jump_ball", "won", 0))
    for _ in range(3):
        assert "fts_0" in legal_results(s, "foul")
        s = apply_outcome(s, Outcome("foul", "fts_0", 5))
    assert s.fouls == [0, 3] and "fts_0" not in legal_results(s, "foul")


def test_random_legal_wnba_games_finish_untied():
    rng = np.random.default_rng(7)
    for _ in range(30):
        s = new_game("LV", "NY", 2026, league=WNBA)
        n = 0
        while not s.over and n < 2000:
            t = str(rng.choice(legal_types(s)))
            r = str(rng.choice(legal_results(s, t)))
            lo, hi = elapsed_range(s, t)
            s = apply_outcome(s, Outcome(t, r, int(hi if t == "run_out" else rng.integers(min(lo + 8, hi), hi + 1))))
            assert min(s.timeouts) >= 0 and s.fts_left >= 0
            n += 1
        assert s.over and s.score[HOME] != s.score[AWAY] and s.qtr >= 4


# ---------------------------------------------------------------- the reader, on synthetic rows

def _rows(plays: list[tuple]) -> pd.DataFrame:
    """ESPN-format rows: (period, clock, type_text, team (H/A/None), home score, away score, scoring)."""
    out = []
    for i, (q, clock, tt, team, hs, as_, scoring) in enumerate(plays):
        tid = {"H": 17, "A": 9, None: np.nan}[team]
        out.append(dict(game_id=1, game_play_number=i, sequence_number=i, season=2026, season_type=2,
                        game_date="2026-06-01", period_number=q, start_quarter_seconds_remaining=float(clock),
                        type_text=tt, text="three point" if "Three" in tt else "", team_id=tid,
                        home_team_id=17, away_team_id=9, home_score=hs, away_score=as_,
                        score_value=(3 if "Three" in tt else 1 if "Free" in tt else 2) if scoring else 0,
                        scoring_play=scoring, shooting_play="Shot" in tt or "Free" in tt, home_team_spread=np.nan))
    return pd.DataFrame(out)


def _outcomes(df: pd.DataFrame) -> list[tuple]:
    """(kind, free throws of a foul, side, free throws a shot drew) per event."""
    events = wnba._events(wnba._actions(df))
    return [(e["kind"], e.get("n") if e["kind"] in ("foul", "ft") else None, e.get("side"), e.get("fouled"))
            for e in events]


def test_reader_fouls_trips_and_kept_balls():
    df = _rows([
        (1, 600, "Jumpball", "A", 0, 0, False),                # the first-named jumper is away; home gets it
        (1, 590, "Jump Shot", "H", 2, 0, True),
        (1, 580, "Personal Foul", "H", 2, 0, False),           # a foul, no free throws: fts_0 by home
        (1, 570, "Jump Shot", "A", 2, 0, False),
        (1, 570, "Defensive Rebound", "H", 2, 0, False),
        (1, 560, "Personal Foul", "A", 2, 0, False),           # a foul with its trip: one event, n = 2
        (1, 560, "Free Throw - 1 of 2", "H", 3, 0, True),
        (1, 560, "Free Throw - 2 of 2", "H", 4, 0, True),
        (1, 550, "Jump Shot", "A", 4, 0, False),
        (1, 550, "Shooting Foul", "H", 4, 0, False),           # a shooting foul: the shot drew it
        (1, 550, "Free Throw - 1 of 2", "A", 4, 1, True),
        (1, 550, "Free Throw - 2 of 2", "A", 4, 1, False),
        (1, 550, "Defensive Rebound", "H", 4, 1, False),
        (1, 540, "Offensive Foul", "H", 4, 1, False),          # not a team foul: the turnover is the event
        (1, 540, "Offensive Foul Turnover", "H", 4, 1, False),
        (1, 530, "Transition Take Foul", "H", 4, 1, False),   # one shot and the ball
        (1, 530, "Free Throw - 1 of 1", "A", 4, 2, True),
        (1, 520, "Flagrant Foul Type 1", "A", 4, 2, False),   # two shots and the ball
        (1, 520, "Free Throw - Flagrant 1 of 2", "H", 5, 2, True),
        (1, 520, "Free Throw - Flagrant 2 of 2", "H", 5, 2, False),
        (1, 510, "Three Point Jump Shot", "H", 8, 2, True),
    ])
    assert _outcomes(df) == [
        ("jump", None, HOME, None),
        ("shot", None, HOME, None),
        ("foul", 0, AWAY, None),
        ("shot", None, AWAY, None),
        ("foul", 2, HOME, None), ("ft", 2, HOME, None), ("ft", 2, HOME, None),
        ("shot", None, AWAY, 2), ("ft", 2, AWAY, None), ("ft", 2, AWAY, None),
        ("to", None, HOME, None),
        ("foul", 0, AWAY, None), ("tech", None, AWAY, None),
        ("foul", 0, HOME, None), ("tech", None, HOME, None), ("tech", None, HOME, None),
        ("shot", None, HOME, None),
        ("run_out", None, None, None),
    ]


def test_reader_rows_replay_and_count_team_fouls():
    df = _rows([
        (1, 600, "Jumpball", "H", 0, 0, False),
        *[(1, 590 - 10 * i, "Personal Foul", "A", 0, 0, False) for i in range(4)],   # four fouls, no trips
        (1, 540, "Personal Foul", "A", 0, 0, False),                                  # the 5th: two shots
        (1, 540, "Free Throw - 1 of 2", "H", 1, 0, True),
        (1, 540, "Free Throw - 2 of 2", "H", 2, 0, True),
        (1, 535, "Loose Ball Foul", "A", 2, 0, False),                                # away has the ball: a turnover
        (1, 100, "Personal Foul", "A", 2, 0, False),                                  # in the penalty: two shots
        (1, 100, "Free Throw - 1 of 2", "H", 3, 0, True),
        (1, 100, "Free Throw - 2 of 2", "H", 4, 0, True),
        (1, 90, "Jump Shot", "A", 4, 2, True),
    ])
    events = wnba._events(wnba._actions(df))
    assert college.process_game(df, WNBA, ("LV", "NY", False), events=events) is None  # too few events
    lg = replace(WNBA, min_events=1)
    meta, rows = college.process_game(df, lg, ("LV", "NY", False), events=events)
    kinds = [(r["play_type"], r["result"], r["foul_a"]) for r in rows]
    assert kinds[:6] == [("jump_ball", "won", 0)] + [("foul", "fts_0", n) for n in range(4)] + [("foul", "fts_2", 4)]
    assert kinds[6:] == [("free_throw", "made", 5)] * 2 + [("turnover", "turnover", 5), ("foul", "fts_2", 5)] + \
        [("free_throw", "made", 6)] * 2 + [("shot2", "made", 6), ("run_out", "end", 6)]
    ok, total, misses = college.replay_agreement(meta, rows, lg)
    assert ok == total, misses


# ---------------------------------------------------------------- real games

def _replay(df: pd.DataFrame) -> list[tuple[dict, list[dict], float]]:
    out = []
    for _, g in df.groupby("game_id", sort=False):
        res = wnba.process_game(g)
        if res is None:
            out.append((None, None, None))
            continue
        ok, total, _ = wnba.replay_agreement(*res, WNBA)
        out.append((*res, ok / total))
    return out


def test_real_games_replay():
    """Four real games (2006, 2015 and two of 2026, two with overtime, flagrant and take fouls) replay through
    the engine; the All-Star game is dropped. Every row's team-foul count follows the WNBA's rules."""
    df = wnba.numpy_dtypes(pd.read_parquet(FIXTURE))
    got = _replay(df)
    assert got[-1][0] is None                                             # the All-Star game
    penalty_no_fts = 0
    for meta, rows, agreement in got[:-1]:
        assert agreement >= 0.99 and meta["home"] in WNBA.teams and meta["away"] in WNBA.teams
        last = rows[-1]
        final = apply_outcome(college.state_from_row(meta, last, WNBA), college.outcome_from_row(last))
        assert final.over and final.score == [meta["home_final"], meta["away_final"]]
        r = pd.DataFrame(rows)
        assert (r.result == "fts_0").any() and r[["foul_h", "foul_a"]].to_numpy().max() >= 5
        # a foul in the penalty sends the shooter to the line
        fouls_def = np.where(r.pos == 0, r.foul_a, r.foul_h)
        pen = np.where(r.qtr > 4, fouls_def + 1 >= 4, fouls_def + 1 >= 5)
        penalty_no_fts += int(((r.result == "fts_0") & pen).sum())
    # but for one 2006 row: a defensive "PersonalFoul" then the offense's turnover at the same clock (a
    # mislabeled offensive foul; the older feed has more of these, experiments/wnba/README.md)
    assert penalty_no_fts <= 1
    assert sum(m["home_final"] + m["away_final"] > 0 for m, _, _ in got[:-1]) == 4
    assert {r["qtr"] for _, rows, _ in got[:-1] for r in rows} >= {5}    # overtime games are in the fixture


def test_real_games_encode_with_masks_that_hold_the_targets():
    df = wnba.numpy_dtypes(pd.read_parquet(FIXTURE))
    metas, plays = [], []
    for meta, rows, _ in _replay(df)[:-1]:
        for i, r in enumerate(rows):
            r["idx"] = i
        metas.append(meta)
        plays += rows
    games, plays = pd.DataFrame(metas), pd.DataFrame(plays)
    ts = build_team_seasons(games, WNBA)
    assert ts.teams == list(WNBA.teams)
    items = encode_games(plays, games, ts, 512)
    assert len(items) == 4 and all(it["result_mask"].shape[1] == WNBA.n_results for it in items)
    assert all(it["f_fouls_def"].max() <= WNBA.n_fouls - 1 for it in items)


@pytest.mark.skipif(not RAW.exists(), reason="data/wnba/raw not downloaded (runitback wnba download)")
def test_full_seasons_replay(tmp_path):
    seasons = sorted(int(p.stem.split("_")[1]) for p in RAW.glob("pbp_*.parquet"))[-2:]
    stats = wnba.preprocess(seasons, RAW, tmp_path)
    assert stats["agreement"] >= 0.99 and stats["games"] > 200


# ---------------------------------------------------------------- a fold

def _random_game(rng, gid, season, home, away, date):
    s = new_game(home, away, season, league=WNBA)
    rows = []
    while not s.over and len(rows) < 500:
        t = str(rng.choice(legal_types(s)))
        r = str(rng.choice(legal_results(s, t)))
        lo, hi = elapsed_range(s, t)
        e = int(hi if t == "run_out" else rng.integers(min(lo + 6, hi), hi + 1))
        rows.append(dict(game_id=gid, idx=len(rows), qtr=s.qtr, qsec=s.qsec, phase=s.phase, pos=s.pos,
                         fts_left=s.fts_left, score_h=s.score[0], score_a=s.score[1], to_h=s.timeouts[0],
                         to_a=s.timeouts[1], foul_h=s.fouls[0], foul_a=s.fouls[1], oo=int(s.one_and_one),
                         arrow=-1, play_type=t, result=r, elapsed=e))
        s = apply_outcome(s, Outcome(t, r, e))
    meta = dict(game_id=gid, season=season, game_type="REG", game_date=date, home=home, away=away,
                home_final=s.score[0], away_final=s.score[1], tip_winner=0, home_spread=None, poss_h=80, poss_a=80,
                neutral=False)
    return meta, rows


def _synthetic(tmp: Path, through: dict[int, int]) -> Path:
    rng = np.random.default_rng(0)
    teams = list(WNBA.teams)
    metas, plays = [], []
    for season, n in through.items():
        for i in range(n):
            home, away = teams[i % len(teams)], teams[(i + 1 + i // len(teams)) % len(teams)]
            date = (pd.Timestamp(f"{season}-05-15") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            m, r = _random_game(rng, f"{season}{i:04d}", season, home, away, date)
            metas.append(m)
            plays += r
    tmp.mkdir(parents=True, exist_ok=True)
    games = pd.DataFrame(metas)
    games.to_parquet(tmp / "games.parquet")
    pd.DataFrame(plays).to_parquet(tmp / "plays.parquet")
    build_ratings(games, tmp, WNBA)
    return tmp


def test_fold_never_sees_its_season_and_pca_fits_before_it(tmp_path):
    data = _synthetic(tmp_path / "d", {2024: 60, 2025: 20, 2026: 6})
    gen = BasketballGenerator(WNBA, epochs=1, d_model=16, n_layers=1, n_heads=2, batch_size=8, min_season=2000)
    root = tmp_path / "folds"
    assert prepare(root, data, [2026], gen=gen) == {2026: "trained, embedded"}
    d = core.fold_dir(root, 2026)
    ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
    assert ck["val_season"] == 2025 and ck["league"] == ck["sport"] == ck["config"]["league"] == "wnba"
    assert all(int(k.rsplit("_", 1)[1]) <= 2025 for k in ck["team_seasons"])
    assert ck["teams"] == list(WNBA.teams) and ck["config"]["feature_sizes"]["fouls_def"] == WNBA.n_fouls
    table = pd.read_parquet(d / "pbp_32.parquet")
    games = pd.read_parquet(data / "games.parquet")
    assert len(table) == 2 * len(games) and set(table.team) <= set(WNBA.teams)
    raw = gen.embed(d / "model.pt", data)
    _, proj = core.reduce(raw, 2026, 32)
    np.testing.assert_allclose(core.load_projection(d, 32)["mean"], proj["mean"], rtol=1e-5, atol=1e-6)
    # the sampler plays legal WNBA games from the fold's model
    sim = Simulator.load(d / "model.pt")
    assert sim.ts.league is WNBA
    finals, logs = sim.rollout(new_game("LV", "NY", 2026, league=WNBA), 3, seed=0, keep_plays=1, ratings=None)
    for s, o in logs[0]:
        assert o.play_type in legal_types(s) and o.result in legal_results(s, o.play_type)
    assert all(f.score[0] != f.score[1] for f in finals if f.over)
