"""ESPN college play-by-play -> engine sequences, and the replay check against real games.

The fixture holds five real games from sportsdataverse's espn_cfb_pbp (selected columns): 2023 with a
per-play clock, 2023 with a coarse one, a 2021 overtime game (Notre Dame 41, Florida State 38), 2010
(Oregon-Auburn, separate extra-point rows, no opening kickoff in the feed) and 2005.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from runitback.cfb.data import preprocess as P
from runitback.cfb.data.download import COLUMNS
from runitback.cfb.vocab import KICKOFF, PAT, PAT_SPOT, SCRIMMAGE

FIXTURE = Path(__file__).parent / "fixtures" / "espn_cfb_pbp_games.parquet"
DATA = Path("data/cfb")


def _raw(tmp: Path) -> Path:
    """The fixture as the download step leaves it (one file per season)."""
    raw = tmp / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FIXTURE)
    for s, g in df.groupby("season"):
        g.drop(columns=[c for c in g if g[c].isna().all() and c in ("pointAfterAttempt.text", "extra_point_result",
                                                                     "two_point_conv_result")]).to_parquet(raw / f"pbp_{s}.parquet")
    return raw


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cfb")
    summary = P.preprocess([2005, 2010, 2021, 2023], _raw(tmp), tmp / "processed")
    return tmp / "processed", summary


def _game(game_id: str):
    df = P._columns(pd.read_parquet(FIXTURE))
    return P.process_game(df[df.game_id == int(game_id)])


def test_every_fixture_game_replays_through_the_engine(processed):
    out, summary = processed
    games = pd.read_parquet(out / "games.parquet")
    assert len(games) == 5 and summary["kept"] == 5
    assert summary["agreement"] >= 0.99
    assert summary["final_agreement"] == 1.0  # the engine reaches every final score
    assert (games.home_final != games.away_final).all()
    plays = pd.read_parquet(out / "plays.parquet")
    assert set(P.STATE_COLS + P.OUTCOME_COLS + ["elapsed_known", "idx"]) <= set(plays.columns)
    assert plays.groupby("game_id").idx.apply(lambda s: (s.diff().dropna() == 1).all()).all()
    ratings = pd.read_parquet(out / "ratings.parquet")
    assert len(ratings) == 10 and {"pre_off_epa", "pre_def_epa", "pre_elo"} <= set(ratings.columns)


def test_teams_are_espn_ids_and_schools_are_kept(processed):
    games = pd.read_parquet(processed[0] / "games.parquet").set_index("game_id")
    nd = games.loc["401282614"]
    assert (nd.home, nd.away, nd.home_name, nd.away_name) == ("52", "87", "Florida State", "Notre Dame")
    assert (nd.home_final, nd.away_final, nd.game_type) == (38, 41, "REG")


def test_tries_are_rows_of_their_own_from_the_3():
    meta, rows = _game("401551789")  # 2023: the try is folded into the touchdown's row
    pats = [r for r in rows if r["phase"] == PAT]
    tds = [a for a, b in zip(rows, rows[1:]) if b["phase"] == PAT]
    assert len(pats) == 5 and all(r["yl"] == PAT_SPOT for r in pats)  # Michigan 34, Washington 13
    assert all(b["score_h"] + b["score_a"] - a["score_h"] - a["score_a"] == 6 for a, b in zip(tds, pats))
    assert {r["play_type"] for r in pats} == {"extra_point"}
    meta, rows = _game("310102483")  # 2010: the feed lists the try, but counts it in the touchdown too
    i = next(i for i, r in enumerate(rows) if r["phase"] == PAT)
    assert rows[i]["score_h"] + rows[i]["score_a"] - rows[i - 1]["score_h"] - rows[i - 1]["score_a"] == 6


def test_feed_without_an_opening_kickoff_starts_at_the_first_snap():
    meta, rows = _game("310102483")
    first = next(r for r in rows if r["play_type"] != "timeout")
    assert first["phase"] == SCRIMMAGE and meta["opening_kicker"] == 1 - first["pos"]
    meta, rows = _game("401551789")
    assert rows[0]["phase"] == KICKOFF and meta["opening_kicker"] == rows[0]["pos"]


def test_coarse_clock_is_interpolated_and_flagged():
    meta, rows = _game("401525564")
    assert not meta["clock_known"] and not any(r["elapsed_known"] for r in rows)
    reg = pd.DataFrame([r for r in rows if r["qtr"] <= 4])
    for _, q in reg.groupby("qtr"):
        assert (q.qsec.diff().dropna() <= 0).all()  # never runs backwards
        assert (q.qsec.iloc[:-1] >= 1).all() and q.qsec.iloc[-1] <= 60
        assert (q.elapsed == 0).mean() < 0.3  # spread out, not a jump once a drive
    meta, rows = _game("401551789")
    assert meta["clock_known"] and all(r["elapsed_known"] for r in rows)


def test_overtime_is_kept_when_it_replays(processed):
    games = pd.read_parquet(processed[0] / "games.parquet").set_index("game_id")
    assert games.loc["401282614", "overtime_kept"] and games.loc["401282614", "ot_agreement"] == 1.0
    plays = pd.read_parquet(processed[0] / "plays.parquet")
    ot = plays[(plays.game_id == "401282614") & (plays.qtr >= 5)]
    assert len(ot) and (ot.qsec == 0).all() and (ot.elapsed == 0).all()
    first = ot.iloc[0]
    assert (first.down, first.ytg, first.yl, first.ot_half) == (1, 10, 25, 0)
    assert set(ot.ot_half) == {0, 1} and (ot.ot_first == first.pos).all()


def test_folded_penalties_follow_the_down_that_came_next():
    row = dict(phase=SCRIMMAGE, pos=0, qtr=1, down=2, ytg=8, yl=60, play_type="run")
    nxt = dict(phase=SCRIMMAGE, pos=0, qtr=1, down=2, ytg=18, yl=70)
    assert P._penalty_as_played(row, nxt, "run", -10) == ("penalty", "penalty")  # holding: down replayed
    nxt = dict(phase=SCRIMMAGE, pos=0, qtr=1, down=1, ytg=10, yl=55)
    assert P._penalty_as_played(row, nxt, "run", 5) == ("penalty", "penalty_first_down")
    nxt = dict(phase=SCRIMMAGE, pos=0, qtr=1, down=3, ytg=4, yl=56)
    assert P._penalty_as_played(row, nxt, "run", 4) == ("run", "run")
    grounding = dict(row, play_type="penalty")
    nxt = dict(phase=SCRIMMAGE, pos=0, qtr=1, down=3, ytg=18, yl=70)
    assert P._penalty_as_played(grounding, nxt, "penalty", -10) == ("pass", "incomplete")


def test_unknown_elapsed_adds_no_loss_in_the_nfl_encoding(processed):
    from runitback.nfl.model.dataset import build_team_seasons, collate, encode_games
    out = processed[0]
    games, plays = pd.read_parquet(out / "games.parquet"), pd.read_parquet(out / "plays.parquet")
    items = encode_games(plays, games, build_team_seasons(games), 400)
    coarse = next(it for it in items if it["game_id"] == "401525564")
    fine = next(it for it in items if it["game_id"] == "401551789")
    assert (coarse["elapsed"] == 0).all() and (coarse["el_hi"] == 0).all()
    assert (fine["elapsed"] > 0).mean() > 0.5 and (fine["el_hi"] >= fine["elapsed"]).all()
    b = collate([coarse])
    assert b["elapsed_mask"][0, : len(coarse["type"])].sum(-1).eq(1).all()  # one legal value: no loss
    # without the column the encoding is the NFL's (elapsed targets as recorded)
    legacy = encode_games(plays.drop(columns="elapsed_known"), games, build_team_seasons(games), 400)
    old = next(it for it in legacy if it["game_id"] == "401525564")
    assert (old["elapsed"] > 0).any()


@pytest.mark.skipif(not (DATA / "raw").exists(), reason="no college play-by-play (runitback cfb download)")
def test_replay_agreement_on_downloaded_seasons(tmp_path):
    """The whole feed: the games kept (>= 95% of their transitions) replay at >= 98.5%."""
    seasons = sorted(int(p.stem.split("_")[1]) for p in (DATA / "raw").glob("pbp_*.parquet"))[-2:]
    summary = P.preprocess(seasons, DATA / "raw", tmp_path)
    assert summary["kept"] >= 0.9 * summary["games"]
    assert summary["kept_agreement"] >= 0.985
    assert summary["final_agreement"] >= 0.97


def test_download_keeps_only_the_preprocess_columns(tmp_path):
    from runitback.cfb.data.download import download_season, load_raw
    df = pd.read_parquet(FIXTURE)
    df["unused"] = 1
    path = download_season(2023, tmp_path, read=lambda url: df[df.season == 2023])
    assert path is not None and "unused" not in pd.read_parquet(path).columns
    raw = load_raw(2023, tmp_path)
    assert set(COLUMNS) <= set(raw.columns)
    assert download_season(2023, tmp_path / "x", read=lambda url: (_ for _ in ()).throw(OSError("404"))) is None
    assert np.isin(raw.game_id.unique(), [401551789, 401525564]).all()
