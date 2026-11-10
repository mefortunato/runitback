import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from runitback.nhl.data.download import (
    download_players, download_schedule, download_season, download_shifts, load_shifts, parse_clock,
    release_candidates, season_code, shifts_frame,
)
from runitback.nhl.data.events import (
    AWAY_SKATERS, HOME_SKATERS, assign_game_ids, build_events, coverage, fill_from_shifts, gap_games,
    normalize, personnel_agreement, personnel_from_shifts, shift_changes,
)

GID = 2024020001
HG, AG = 1, 2                                   # goalies
H = [11, 12, 13, 14, 15, 16, 17]                # home skaters: 11-15 start, 16-17 come on
A = [21, 22, 23, 24, 25, 26]                    # away skaters: 21-25 start, 26 comes on
GOALIES = {HG, AG}


def row(idx, etype, t, period=1, team=None, home=H[:5], away=A[:5], hg=HG, ag=AG, gid=GID, hs=0, as_=0,
        on=None, off=None, p1=None, xg=np.nan, hsk=None, ask=None, ptype="REGULAR"):
    r = dict(game_id=float(gid) if etype != "CHANGE" else np.nan, season="20242025", season_type="R",
             game_date="2024-10-04", event_idx=idx, event_type=etype, secondary_type=None,
             event_team_type=team, period=period, period_type=ptype, period_seconds=t,
             game_seconds=(period - 1) * 1200 + t, home_score=hs, away_score=as_, home_abbr="BUF",
             away_abbr="NJD", event_goalie_id=np.nan, penalty_severity=None, penalty_minutes=np.nan,
             empty_net=False, x_fixed=np.nan, y_fixed=np.nan, shot_distance=np.nan, shot_angle=np.nan, xg=xg,
             home_skaters=len(home) if hsk is None else hsk, away_skaters=len(away) if ask is None else ask,
             home_goalie_id=hg, away_goalie_id=ag, ids_on=on, ids_off=off,
             event_player_1_id=p1, event_player_1_type=None, event_player_2_id=np.nan, event_player_2_type=None,
             event_player_3_id=np.nan, event_player_3_type=None)
    for i in range(7):
        r[f"home_on_{i + 1}_id"] = home[i] if i < len(home) else np.nan
        r[f"away_on_{i + 1}_id"] = away[i] if i < len(away) else np.nan
    return r


def game_rows(gid=GID, personnel=True):
    """A short game: opening lines, a home change on the fly after a shot, a penalty (away 4 skaters),
    a power-play goal, then the away goalie pulled for a sixth skater."""
    h2 = [11, 12, 13, 16, 17]
    a4 = [21, 22, 23, 24]
    rows = [
        row(0, "CHANGE", 0, home=H[:5], away=[], on="11, 12, 13, 14, 15, 1", off="0"),
        row(1, "CHANGE", 0, on="21, 22, 23, 24, 25, 2", off="0"),
        row(2, "FACEOFF", 0, team="home", p1=13),
        row(3, "PERIOD_START", 0),
        row(4, "SHOT", 30, team="home", p1=14, xg=0.05),
        row(5, "CHANGE", 30, home=h2, on="16, 17", off="14, 15"),
        row(6, "PENALTY", 100, team="away", home=h2, p1=25),
        row(7, "CHANGE", 100, home=h2, away=a4, off="25"),
        row(8, "FACEOFF", 100, team="home", home=h2, away=a4, p1=13),
        row(9, "GOAL", 150, team="home", home=h2, away=a4, p1=16, xg=0.3, hs=1),
        row(10, "CHANGE", 150, home=h2, away=[21, 22, 23, 24, 26], ag=np.nan, on="26", off="2", hs=1),
        row(11, "FACEOFF", 150, team="away", home=h2, away=[21, 22, 23, 24, 26], ag=np.nan, p1=21, hs=1, ask=5),
        row(12, "SHOT", 200, team="away", home=h2, away=[21, 22, 23, 24, 25, 26], ag=np.nan, p1=26, xg=0.1, hs=1),
        row(13, "PERIOD_END", 1200, home=[], away=[], hg=np.nan, ag=np.nan, hs=1),
        row(14, "GAME_END", 1200, home=[], away=[], hg=np.nan, ag=np.nan, hs=1),
    ]
    for r in rows:
        r["game_id"] = float(gid) if r["event_type"] != "CHANGE" else np.nan
    df = pd.DataFrame(rows)
    if not personnel:
        df = df[df["event_type"] != "CHANGE"].copy()
        for c in [f"{s}_on_{i}_id" for s in ("home", "away") for i in range(1, 8)] + ["home_goalie_id",
                                                                                      "away_goalie_id"]:
            df[c] = np.nan
        df["event_idx"] = range(len(df))
    return df


def shifts_for_game(gid=GID) -> pd.DataFrame:
    """Shift charts matching game_rows (one period): the goalie is pulled at 150, after the goal, and 25
    comes back from the penalty box at 190 (a change the release rows leave out)."""
    rows = []
    add = lambda p, team, start, end: rows.append(dict(player_id=p, team_id=7 if team == "BUF" else 1,  # noqa: E731
                                                       team=team, period=1, start=start, end=end))
    add(HG, "BUF", 0, 1200)
    for p in (11, 12, 13):
        add(p, "BUF", 0, 1200)
    for p in (14, 15):
        add(p, "BUF", 0, 30)
    for p in (16, 17):
        add(p, "BUF", 30, 1200)
    add(AG, "NJD", 0, 150)
    for p in (21, 22, 23, 24):
        add(p, "NJD", 0, 1200)
    add(25, "NJD", 0, 100)
    add(26, "NJD", 150, 1200)
    add(25, "NJD", 190, 1200)
    return pd.DataFrame(rows)


def on_ice(ev, i, side):
    cols = HOME_SKATERS if side == "home" else AWAY_SKATERS
    r = ev.iloc[i]
    return {int(r[c]) for c in cols if pd.notna(r[c])}


def test_assign_game_ids_fills_changes_and_splits_games():
    df = pd.concat([game_rows(GID), game_rows(GID + 1)], ignore_index=True)
    gid = assign_game_ids(df)
    assert gid.notna().all()
    assert (gid.iloc[:15] == GID).all() and (gid.iloc[15:] == GID + 1).all()


def test_normalize_personnel_strength_and_change_sides():
    ev = normalize(game_rows(), 2025, GOALIES)
    assert list(ev["event_idx"]) == list(range(15))
    fo = ev.iloc[2]
    assert on_ice(ev, 2, "home") == set(H[:5]) and on_ice(ev, 2, "away") == set(A[:5])
    assert fo["home_goalie"] == HG and fo["away_goalie"] == AG and fo["n_home"] == 5
    # opening change rows: the goalie listed among ids_on is not a skater
    assert ev.iloc[0]["event_team"] == "home" and ev.iloc[1]["event_team"] == "away"
    assert ev.iloc[0]["ids_on"] == "11,12,13,14,15,1" and pd.isna(ev.iloc[0]["ids_off"])
    assert ev.iloc[5]["event_team"] == "home" and ev.iloc[7]["event_team"] == "away"
    # strength from the home side; the power-play goal
    assert ev.iloc[9]["strength_state"] == "5v4"
    assert ev.iloc[9]["xg"] == pytest.approx(0.3)
    # goalie pulled: six skaters, no goalie
    r = ev.iloc[12]
    assert pd.isna(r["away_goalie"]) and r["n_away"] == 6 and r["strength_state"] == "5v6"
    assert ev["game_type"].eq("REG").all()


def test_goalie_listed_among_skaters_is_moved():
    df = game_rows()
    df.loc[2, "home_on_6_id"] = HG
    df.loc[2, "home_goalie_id"] = np.nan
    ev = normalize(df, 2025, GOALIES)
    assert ev.iloc[2]["n_home"] == 5 and ev.iloc[2]["home_goalie"] == HG


def test_all_null_bool_columns_are_empty():
    df = game_rows()
    df["home_on_7_id"] = pd.Series([None] * len(df), dtype=object)
    ev = normalize(df, 2025, GOALIES)
    assert ev.iloc[2]["n_home"] == 5


def test_coverage_and_gaps():
    full = normalize(game_rows(GID), 2025, GOALIES)
    bare = normalize(game_rows(GID + 1, personnel=False), 2025, GOALIES)
    ev = pd.concat([full, bare], ignore_index=True)
    c = coverage(ev)
    assert c.loc[GID, "coverage"] == 1.0 and c.loc[GID + 1, "coverage"] == 0.0
    assert gap_games(ev) == [GID + 1]
    assert set(bare["personnel"]) == {"pbp", "none"}


def test_shift_changes_merge_restarted_shifts():
    s = pd.DataFrame([dict(player_id=5, team_id=7, team="BUF", period=1, start=0, end=40),
                      dict(player_id=5, team_id=7, team="BUF", period=1, start=40, end=90),
                      dict(player_id=6, team_id=1, team="NJD", period=1, start=0, end=90)])
    ch = shift_changes(s, {7: "home", 1: "away"})
    assert ch == [(1, 0.0, "home", [5], []), (1, 0.0, "away", [6], []), (1, 90.0, "home", [], [5]),
                  (1, 90.0, "away", [], [6])]


def test_personnel_from_shifts_matches_release():
    truth = normalize(game_rows(), 2025, GOALIES)
    new = personnel_from_shifts(truth, shifts_for_game(), GOALIES)
    assert new is not None and set(new["personnel"]) == {"shifts"}
    play = new[new["event_type"] != "CHANGE"].reset_index(drop=True)
    ref = truth[truth["event_type"] != "CHANGE"].reset_index(drop=True)
    for i in range(len(play)):
        if play.iloc[i]["event_type"] in ("PERIOD_END", "GAME_END"):
            continue
        for side in ("home", "away"):
            assert on_ice(play, i, side) == on_ice(ref, i, side), (i, side)
    # the shot at 30 still sees 14 and 15 (their shift ends at 30); the goal at 150 sees the away goalie
    assert {14, 15} <= on_ice(play, 2, "home")
    assert play.loc[play["event_type"] == "GOAL", "away_goalie"].iloc[0] == AG
    assert pd.isna(play.loc[play["event_type"] == "SHOT", "away_goalie"].iloc[-1])
    # changes come before the faceoff at their second, after the shot and the goal at it
    types = list(new["event_type"])
    assert types[:4] == ["CHANGE", "CHANGE", "FACEOFF", "PERIOD_START"]
    i_shot, i_goal = types.index("SHOT"), types.index("GOAL")
    assert types[i_shot + 1] == "CHANGE" and new.iloc[i_shot + 1]["event_team"] == "home"
    assert types[i_goal + 1] == "CHANGE" and new.iloc[i_goal + 1]["period_seconds"] == 150
    assert new.iloc[i_goal + 1]["ids_off"] == "2" and new.iloc[i_goal + 1]["ids_on"] == "26"
    assert personnel_agreement(truth, new) == dict(events=8, skaters=1.0, skaters_and_goalies=1.0)


def test_personnel_from_shifts_without_release_personnel():
    bare = normalize(game_rows(personnel=False), 2025, GOALIES)
    new = personnel_from_shifts(bare, shifts_for_game(), GOALIES)
    fo = new[new["event_type"] == "FACEOFF"].reset_index(drop=True)
    assert on_ice(fo, 1, "away") == {21, 22, 23, 24} and fo.iloc[1]["home_goalie"] == HG


def test_personnel_from_shifts_rejects_unusable_charts():
    bare = normalize(game_rows(personnel=False), 2025, GOALIES)
    one_team = shifts_for_game()
    assert personnel_from_shifts(bare, one_team[one_team["team"] == "BUF"], GOALIES) is None
    other_period = shifts_for_game().assign(period=2)
    assert personnel_from_shifts(bare, other_period, GOALIES) is None


def test_team_sides_by_players_when_abbreviations_differ():
    bare = normalize(game_rows(personnel=False), 2025, GOALIES)
    s = shifts_for_game().assign(team=lambda d: d["team"].map({"BUF": "XXX", "NJD": "YYY"}))
    new = personnel_from_shifts(bare, s, GOALIES)
    assert new is not None and on_ice(new[new["event_type"] == "FACEOFF"].reset_index(), 0, "home") == set(H[:5])


def test_fill_from_shifts_uses_cache(tmp_path):
    full = normalize(game_rows(GID), 2025, GOALIES)
    bare = normalize(game_rows(GID + 1, personnel=False), 2025, GOALIES)
    ev = pd.concat([full, bare], ignore_index=True)
    out, stats = fill_from_shifts(ev, tmp_path, GOALIES)
    assert stats == dict(gap_games=1, filled=0, no_shifts=1, unusable=0)
    (tmp_path / "shifts").mkdir()
    shifts_for_game(GID + 1).to_parquet(tmp_path / "shifts" / f"{GID + 1}.parquet")
    out, stats = fill_from_shifts(ev, tmp_path, GOALIES)
    assert stats["filled"] == 1
    g = out[out["game_id"] == GID + 1]
    assert set(g["personnel"]) == {"shifts"} and (g["event_type"] == "CHANGE").sum() > 0
    assert list(g["event_idx"]) == list(range(len(g)))
    assert coverage(out).loc[GID + 1, "coverage"] == 1.0
    kept = out[out["game_id"] == GID].reset_index(drop=True)
    assert kept[HOME_SKATERS + AWAY_SKATERS].equals(full[HOME_SKATERS + AWAY_SKATERS])


# ---------------------------------------------------------------- download (no network)

def test_season_helpers():
    assert season_code(2025) == "20242025"
    assert release_candidates(2016)[0] == 2015 and release_candidates(2025)[0] == 2025
    assert parse_clock("12:34") == 754 and np.isnan(parse_clock(None))


def test_download_season_finds_the_file_holding_the_season(tmp_path):
    files = {2015: "20152016", 2014: "20132014", 2016: "20162017"}  # as the release is (no 2014-15)

    def read(url):
        n = int(url.rsplit("_", 1)[1].split(".")[0])
        if n not in files:
            raise FileNotFoundError(url)
        return pd.DataFrame({"season": [files[n]] * 2, "x": [1, 2]})

    assert download_season(2016, tmp_path, read=read) == tmp_path / "pbp_2016.parquet"
    assert pd.read_parquet(tmp_path / "pbp_2016.parquet")["season"].eq("20152016").all()
    assert download_season(2015, tmp_path, read=read) is None


def test_schedule_players_and_shifts_from_api(tmp_path):
    def fetch(url):
        url = urllib.parse.unquote_plus(url)
        if "/game?" in url:
            return {"data": [dict(id=2024020001, gameDate="2024-10-04", gameType=2, homeTeamId=7, visitingTeamId=1,
                                  homeScore=1, visitingScore=0, gameStateId=7),
                             dict(id=2024010001, gameDate="2024-09-20", gameType=1, homeTeamId=7, visitingTeamId=1,
                                  homeScore=1, visitingScore=0, gameStateId=7)]}
        if url.endswith("/team"):
            return {"data": [dict(id=7, triCode="BUF"), dict(id=1, triCode="NJD")]}
        if "skater/summary" in url:
            return {"data": [dict(playerId=11, skaterFullName="A B", positionCode="C", shootsCatches="L",
                                  gamesPlayed=1)] if "gameTypeId=2" in url else []}
        if "goalie/summary" in url:
            return {"data": [dict(playerId=1, goalieFullName="G H", shootsCatches="L", gamesPlayed=1)]}
        if "shiftcharts" in url:
            return {"data": [dict(typeCode=517, playerId=11, teamId=7, teamAbbrev="BUF", period=1,
                                  startTime="00:00", endTime="00:45"),
                             dict(typeCode=505, playerId=11, teamId=7, teamAbbrev="BUF", period=1,
                                  startTime="00:40", endTime="00:40")]}
        raise AssertionError(url)

    g = pd.read_parquet(download_schedule(2025, tmp_path, fetch=fetch))
    assert list(g["game_id"]) == [2024020001] and g.iloc[0]["home"] == "BUF" and g.iloc[0]["game_type"] == "REG"
    p = pd.read_parquet(download_players(2025, tmp_path, fetch=fetch))
    assert set(zip(p["player_id"], p["position"])) == {(11, "C"), (1, "G")}
    assert download_shifts([GID], tmp_path, fetch=fetch) == 1
    assert download_shifts([GID], tmp_path, fetch=fetch) == 0  # cached
    s = load_shifts(GID, tmp_path)
    assert len(s) == 1 and s.iloc[0]["end"] == 45
    assert shifts_frame([]).empty


def test_build_events_end_to_end(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    pd.concat([game_rows(GID), game_rows(GID + 1, personnel=False)], ignore_index=True).to_parquet(
        raw / "pbp_2025.parquet")
    pd.DataFrame([dict(season=2025, player_id=p, name=f"P{p}", position="G" if p in GOALIES else "C",
                       shoots="L", games=1, game_type="REG") for p in [HG, AG] + H + A]).to_parquet(
        raw / "players_2025.parquet")
    (raw / "shifts").mkdir()
    shifts_for_game(GID + 1).to_parquet(raw / "shifts" / f"{GID + 1}.parquet")
    summary = build_events([2024, 2025], raw, out)
    assert summary[2024]["events"] == 0
    s = summary[2025]
    assert s["games"] == 2 and s["filled"] == 1 and s["coverage"] == 1.0 and s["release_coverage"] == 0.5
    games = pd.read_parquet(out / "games.parquet")
    assert dict(zip(games["game_id"], games["personnel"])) == {GID: "pbp", GID + 1: "shifts"}
    assert games["home_score"].eq(1).all() and games["home"].eq("BUF").all()
    players = pd.read_parquet(out / "players.parquet")
    assert set(players.loc[players["group"] == "G", "player_id"]) == GOALIES
    assert summary["players"]["on_ice_without_position"] == 0


# ---------------------------------------------------------------- real data (skipped without it)

RAW = Path("data/nhl/raw")


@pytest.mark.skipif(not (RAW / "pbp_2025.parquet").exists(), reason="needs data/nhl/raw (runitback nhl download)")
def test_real_season_normalizes_with_personnel():
    import pyarrow.parquet as pq
    from runitback.nhl.data.events import SOURCE_COLUMNS, goalie_ids, load_players
    path = RAW / "pbp_2025.parquet"
    cols = [c for c in SOURCE_COLUMNS if c in pq.read_schema(path).names]
    ev = normalize(pd.read_parquet(path, columns=cols), 2025, goalie_ids(load_players([2025], RAW)))
    assert ev["game_id"].nunique() > 1300
    c = coverage(ev)
    assert (c["coverage"] >= 0.99).mean() > 0.98
    assert ev.loc[ev["event_type"] == "CHANGE", "event_team"].notna().mean() > 0.99
