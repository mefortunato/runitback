from pathlib import Path

import pandas as pd
import pytest

from runitback.nba.data.players import (
    LINEUP_COLS, build_on_court, build_player_directory, build_player_games, build_players, credit_lineups,
    dnp_kind, five_share, game_on_court, on_court_totals,
)

HOME_T, AWAY_T = 1, 2
H = [101, 102, 103, 104, 105, 106, 107]  # home: 101-105 start
A = [201, 202, 203, 204, 205, 206, 207]  # away: 201-205 start


def box(minutes=None, dnp=(), reason=None) -> pd.DataFrame:
    rows = []
    for team, ids, ha, opp in ((HOME_T, H, "home", "NYK"), (AWAY_T, A, "away", "BOS")):
        for k, p in enumerate(ids):
            rows.append(dict(
                game_id=1, season=2025, season_type=2, game_date="2025-01-01", athlete_id=p,
                athlete_display_name=f"P{p}", athlete_short_name=f"P{p}", team_id=team,
                team_abbreviation="BOS" if team == HOME_T else "NY", opponent_team_abbreviation=opp,
                home_away=ha, minutes=(minutes or {}).get(p, 30.0 if k < 5 else 9.0), plus_minus="+0",
                points=0.0, starter=k < 5, did_not_play=p in dnp,
                reason=(reason if p in dnp else "COACH'S DECISION"), ejected=False,
                athlete_position_abbreviation=["PG", "SG", "SF", "PF", "C", "G", "F"][k]))
    return pd.DataFrame(rows)


def pbp(events) -> pd.DataFrame:
    """events: (period, clock, type_text, id1, id2, home_score, away_score)."""
    rows = []
    for n, (q, clock, tt, a1, a2, hs, as_) in enumerate(events):
        team = HOME_T if (a1 or 0) // 100 == 1 else AWAY_T
        rows.append(dict(game_id=1, id=n, game_play_number=n + 1, sequence_number=str(n + 1), period_number=q,
                         start_quarter_seconds_remaining=float(clock), type_text=tt, team_id=team,
                         athlete_id_1=a1, athlete_id_2=a2, athlete_id_3=None, home_team_id=HOME_T,
                         away_team_id=AWAY_T, home_score=hs, away_score=as_))
    return pd.DataFrame(rows)


def lineup(oc: pd.DataFrame, i: int, side: str) -> set[int]:
    r = oc.iloc[i]
    return {int(r[f"{side}_{k}"]) for k in range(1, 6) if pd.notna(r[f"{side}_{k}"])}


def test_starters_substitutions_and_period_start_inference():
    ev = [
        (1, 720, "Jump Ball", 105, 205, 0, 0),
        (1, 700, "Jump Shot", 101, None, 2, 0),
        (1, 600, "Substitution", 106, 101, 2, 0),          # 106 in for 101
        (1, 500, "Jump Shot", 106, None, 4, 0),
        (1, 0, "End Period", None, None, 4, 0),
        # Q2: no substitutions logged at the break. 101 came back in for 106 silently; 102 never
        # touches the ball but is subbed out later, so he started the period
        (2, 700, "Jump Shot", 101, None, 6, 0),
        (2, 650, "Defensive Rebound", 203, None, 6, 0),
        (2, 600, "Substitution", 107, 102, 6, 0),
        (2, 0, "End Period", None, None, 6, 0),
    ]
    oc, _ = game_on_court(pbp(ev), box())
    assert len(oc) == len(ev) and five_share(oc) == 1.0
    assert lineup(oc, 0, "home") == {101, 102, 103, 104, 105}
    assert lineup(oc, 2, "home") == {102, 103, 104, 105, 106}  # recorded after the substitution
    assert lineup(oc, 5, "home") == {101, 102, 103, 104, 105}  # inferred at the start of Q2
    assert lineup(oc, 7, "home") == {101, 103, 104, 105, 107}
    assert lineup(oc, 5, "away") == {201, 202, 203, 204, 205}  # filled from the end of Q1


def test_missing_substitution_is_repaired_with_look_ahead():
    ev = [
        (1, 720, "Jump Ball", 105, 205, 0, 0),
        (1, 700, "Jump Shot", 101, None, 0, 0),
        # missing: 106 in for 102. 106 shoots while not on court; 102 comes back in later, so he
        # is the one who must have gone off
        (1, 600, "Jump Shot", 106, None, 2, 0),
        (1, 550, "Jump Shot", 103, None, 2, 0),
        (1, 500, "Substitution", 102, 104, 2, 0),
        (1, 0, "End Period", None, None, 2, 0),
    ]
    oc, st = game_on_court(pbp(ev), box())
    assert five_share(oc) == 1.0
    assert lineup(oc, 2, "home") == {101, 103, 104, 105, 106}
    assert lineup(oc, 4, "home") == {101, 102, 103, 105, 106}
    assert st["added"] == 1 and st["removed"] == 1


def test_same_clock_event_after_a_substitution_is_not_a_missing_sub():
    ev = [
        (1, 720, "Jump Ball", 105, 205, 0, 0),
        (1, 430, "Substitution", 106, 101, 0, 0),
        (1, 430, "Lost Ball Turnover", 101, None, 0, 0),  # logged after the sub, same clock
        (1, 400, "Jump Shot", 106, None, 2, 0),
        (1, 0, "End Period", None, None, 2, 0),
    ]
    oc, st = game_on_court(pbp(ev), box())
    assert five_share(oc) == 1.0 and st["added"] == 0
    assert 101 not in lineup(oc, 2, "home")


def test_technical_fouls_are_not_evidence():
    ev = [
        (1, 720, "Jump Ball", 105, 205, 0, 0),
        (1, 500, "Technical Foul", 107, None, 0, 0),  # a bench player
        (1, 0, "End Period", None, None, 0, 0),
    ]
    oc, _ = game_on_court(pbp(ev), box())
    assert 107 not in lineup(oc, 1, "home") and five_share(oc) == 1.0


def test_points_at_a_substitution_are_credited_to_the_five_before_it():
    ev = [
        (1, 720, "Jump Ball", 105, 205, 0, 0),
        (1, 600, "Shooting Foul", 201, None, 0, 0),
        (1, 600, "Free Throw - 1 of 2", 101, None, 1, 0),
        (1, 600, "Substitution", 106, 101, 1, 0),
        (1, 600, "Free Throw - 2 of 2", 101, None, 2, 0),  # still 101's trip: credited to the old five
        (1, 500, "Jump Shot", 106, None, 4, 0),
        (1, 0, "End Period", None, None, 4, 0),
    ]
    oc, _ = game_on_court(pbp(ev), box())
    credit = credit_lineups(oc)
    assert set(credit.iloc[4][LINEUP_COLS[:5]].astype(int)) == {101, 102, 103, 104, 105}
    t = on_court_totals(oc).set_index("player_id")
    assert t.at[101, "pm"] == 2 and t.at[106, "pm"] == 2 and t.at[102, "pm"] == 4
    assert t.at[201, "pm"] == -4
    assert t.at[101, "sec"] == 120 and t.at[106, "sec"] == 600 and t.at[102, "sec"] == 720


def test_build_on_court_skips_games_without_a_box_score():
    p = pd.concat([pbp([(1, 720, "Jump Ball", 105, 205, 0, 0)]),
                   pbp([(1, 720, "Jump Ball", 105, 205, 0, 0)]).assign(game_id=2)])
    oc, st = build_on_court(p, box())
    assert st["games"] == 1 and st["skipped"] == 1 and set(oc.game_id) == {"1"}


def test_dnp_kinds():
    assert dnp_kind("RIGHT ANKLE SPRAIN") == "injury"
    assert dnp_kind("ILLNESS") == "injury"
    assert dnp_kind("COACH'S DECISION") == "coach"
    assert dnp_kind("REST") == "rest"
    assert dnp_kind("NOT WITH TEAM") == "personal"
    assert dnp_kind("SUSPENDED BY LEAGUE") == "suspension"
    assert dnp_kind(None) is None


def test_player_games_and_directory():
    b = box(dnp={107, 207}, reason="LEFT HAMSTRING STRAIN")
    b.loc[b.athlete_id.isin([107, 207]), "minutes"] = None
    pg = build_player_games(b)
    assert len(pg) == 14 and set(pg.team) == {"BOS", "NYK"}  # ESPN's "NY" is canonicalized
    r = pg.set_index("player_id")
    assert r.at[107, "dnp"] and r.at[107, "dnp_reason"] == "LEFT HAMSTRING STRAIN" and r.at[107, "dnp_kind"] == "injury"
    assert pd.isna(r.at[101, "dnp_reason"]) and pd.isna(r.at[101, "dnp_kind"])  # ESPN's reason for a player who played
    assert r.at[101, "starter"] and not r.at[106, "starter"] and r.at[101, "group"] == "G" and r.at[105, "group"] == "C"
    assert r.at[107, "minutes"] == 0 and r.at[101, "career_games"] == 0
    d = build_player_directory(pg).set_index("player_id")
    assert d.at[101, "games"] == 1 and d.at[107, "games"] == 0 and d.at[103, "group"] == "F"


def test_build_players_writes_the_tables(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    ev = [(1, 720, "Jump Ball", 105, 205, 0, 0), (1, 600, "Substitution", 106, 101, 0, 0),
          (1, 300, "Jump Shot", 106, None, 2, 0), (1, 0, "End Period", None, None, 2, 0)]
    pbp(ev).to_parquet(raw / "pbp_2025.parquet")
    box().to_parquet(raw / "player_box_2025.parquet")
    summary = build_players([2025], raw, tmp_path / "processed")
    assert summary[2025]["five_share"] == 1.0
    for f in ("on_court", "player_games", "players"):
        assert (tmp_path / "processed" / f"{f}.parquet").exists()
    pg = pd.read_parquet(tmp_path / "processed" / "player_games.parquet").set_index("player_id")
    assert pg.at[101, "oncourt_min"] == 2.0 and pg.at[106, "oncourt_pm"] == 2


RAW = Path("data/nba/raw")


@pytest.mark.skipif(not any(RAW.glob("player_box_*.parquet")), reason="no NBA play-by-play and box scores")
def test_real_seasons_have_five_on_court_for_99_percent_of_events(tmp_path):
    seasons = sorted(int(p.stem.split("_")[-1]) for p in RAW.glob("player_box_*.parquet")
                     if (RAW / f"pbp_{p.stem.split('_')[-1]}.parquet").exists())[-1:]
    summary = build_players(seasons, RAW, tmp_path)
    for s in seasons:
        assert summary[s]["five_share"] >= 0.99, summary[s]
    assert summary["all"]["minutes_within_1"] >= 0.9


def test_sub_ids_with_missing_text_and_ids_under_pandas_na():
    # pyarrow-backed parquet reads give pd.NA for missing cells; truth-testing one raised
    from runitback.nba.data.players import _sub_ids
    from types import SimpleNamespace as NS
    row = NS(athlete_id_1=5, athlete_id_2=pd.NA, text=pd.NA)
    assert _sub_ids(row) == (5, None)
    row = NS(athlete_id_1=7, athlete_id_2=pd.NA, text=" enters the game for Y")
    assert _sub_ids(row) == (None, 7)
    assert _sub_ids(NS(athlete_id_1=1, athlete_id_2=2, text=pd.NA)) == (1, 2)
