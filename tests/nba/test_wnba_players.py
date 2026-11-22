"""W2: the NBA's player tables, projected lineups and player values for the WNBA (``WNBA_PLAYERS``,
``WNBA_VALUES``), keyed as the WNBA wager model keys teams."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from runitback.nba import player_values as pv
from runitback.nba.data.lineups import projected_lineups
from runitback.nba.data.players import (
    NBA_PLAYERS, WNBA_PLAYERS, build_on_court, build_player_games, build_players,
)
from runitback.nba.league import WNBA_TEAM_IDS

from .test_lineups import toy
from .test_players import AWAY_T, HOME_T, box, pbp


def wnba_box(**kw) -> pd.DataFrame:
    """The NBA test's box score as a WNBA game: Dallas (ESPN id 3, which played as Detroit and Tulsa)
    at Las Vegas (17, which played as San Antonio)."""
    b = box(**kw)
    b["team_id"] = np.where(b.team_id == HOME_T, 17, 3)
    b["team_abbreviation"] = np.where(b.team_id == 17, "SA", "TUL")    # the names they played under
    b["opponent_team_id"] = np.where(b.team_id == 17, 3, 17)
    b["opponent_team_abbreviation"] = np.where(b.team_id == 17, "TUL", "SA")
    return b


def wnba_pbp(events) -> pd.DataFrame:
    p = pbp(events)
    ids = {HOME_T: 17, AWAY_T: 3}
    for c in ("team_id", "home_team_id", "away_team_id"):
        p[c] = p[c].map(ids)
    return p


def test_teams_are_keyed_by_espn_id_as_the_wager_model_keys_them():
    pg = build_player_games(wnba_box(), league=WNBA_PLAYERS)
    assert set(pg.team) == {"LV", "DAL"} and set(pg.opponent) == {"LV", "DAL"}
    assert WNBA_PLAYERS.team_key(3, "DET") == "DAL" and WNBA_PLAYERS.team_key(131247, "COL") is None  # All-Stars
    assert WNBA_PLAYERS.teams == frozenset(WNBA_TEAM_IDS.values())
    # an All-Star game has no franchise key and is dropped from the values
    star = wnba_box()
    star["game_id"] = 2
    star["team_id"], star["opponent_team_id"] = 131247, 131246
    both = build_player_games(pd.concat([wnba_box(), star]), league=WNBA_PLAYERS)
    assert pv.franchise_games(both, WNBA_PLAYERS.teams) == {"1"}
    # the NBA's default keys are unchanged (abbreviations, canonicalized)
    assert set(build_player_games(box()).team) == {"BOS", "NYK"} and NBA_PLAYERS.by_id is None


def test_wnba_rows_that_are_no_evidence():
    # a lowercase "ejection" row names a bench player: no evidence he was on the floor
    ev = [(1, 600, "Jump Shot", 101, None, 2, 0), (1, 590, "ejection", 107, None, 2, 0),
          (1, 580, "Jump Shot", 201, None, 2, 2), (1, 0, "End Period", None, None, 2, 2)]
    oc, st = build_on_court(wnba_pbp(ev), wnba_box(), WNBA_PLAYERS.not_evidence)
    assert st["added"] == 0 and oc.iloc[1][[f"home_{i}" for i in range(1, 6)]].tolist() == [101, 102, 103, 104, 105]
    assert "ejection" not in NBA_PLAYERS.not_evidence


def test_build_players_for_the_wnba(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    ev = [(1, 600, "Jump Ball", 105, 205, 0, 0), (1, 500, "Substitution", 106, 101, 0, 0),
          (1, 300, "Jump Shot", 106, None, 2, 0), (1, 0, "End Period", None, None, 2, 0)]
    wnba_pbp(ev).to_parquet(raw / "pbp_2025.parquet")
    wnba_box().to_parquet(raw / "player_box_2025.parquet")
    summary = build_players([2025], raw, tmp_path / "processed", WNBA_PLAYERS)
    assert summary[2025]["five_share"] == 1.0
    pg = pd.read_parquet(tmp_path / "processed" / "player_games.parquet")
    assert set(pg.team) == {"LV", "DAL"}


def test_wnba_projection_shares_out_200_minutes():
    proj = projected_lineups(toy(target_dnp={1: "LEFT KNEE"}), league=WNBA_PLAYERS)
    av = proj[(proj.game_id == "g4") & (proj.status == "available")]
    assert av.proj_minutes.sum() == pytest.approx(200.0, abs=0.05) and av.proj_minutes.max() <= 38.0
    nba = projected_lineups(toy(target_dnp={1: "LEFT KNEE"}))
    assert nba[(nba.game_id == "g4") & (nba.status == "available")].proj_minutes.sum() == pytest.approx(240.0, abs=0.05)


def test_value_settings():
    # the NBA's settings are the module constants (its pv.parquet is unchanged)
    assert pv.NBA_VALUES.config() == pv.apm_config()
    assert pv.NBA_VALUES.config().min_events == pv.MIN_POSS
    w = pv.WNBA_VALUES.config()
    assert w.lam == pv.WNBA_LAMBDA and w.min_events == round(pv.WNBA_MIN_MINUTES * pv.WNBA_POSS_PER_MIN)
    assert pv.WNBA_VALUES.config(lam=7.0, min_minutes=100).lam == 7.0
    assert pv.VALUE_SETTINGS["wnba"].teams == WNBA_PLAYERS.teams


def test_the_wnba_wager_sport_has_the_lineup_layout_and_no_values_by_default():
    import os

    from runitback.wager.core.sport import get_sport, lineup_spec
    from runitback.wager.wnba.lineup import LINEUP

    spec = lineup_spec("wnba")
    assert spec is LINEUP and spec.columns == pv.PV_COLUMNS and spec.recent == pv.RECENT
    assert spec.groups == pv.GROUP_COLUMNS
    if not os.environ.get("RUNITBACK_WNBA_PLAYER_VALUES"):
        assert "lineup" not in get_sport("wnba").transformer


RAW = Path("data/wnba/raw")


@pytest.mark.skipif(not any(RAW.glob("player_box_*.parquet")), reason="no WNBA play-by-play and box scores")
def test_real_wnba_season_has_five_on_court_for_99_percent_of_events(tmp_path):
    s = max(int(p.stem.split("_")[-1]) for p in RAW.glob("player_box_*.parquet")
            if (RAW / f"pbp_{p.stem.split('_')[-1]}.parquet").exists())
    summary = build_players([s], RAW, tmp_path, WNBA_PLAYERS)
    assert summary[s]["five_share"] >= 0.99, summary[s]
    pg = pd.read_parquet(tmp_path / "player_games.parquet")
    assert set(pg.team.dropna()) <= WNBA_PLAYERS.teams
