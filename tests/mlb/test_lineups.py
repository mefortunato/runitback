"""MLB projected lineups (runitback.mlb.data.lineups) on small synthetic seasons."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from runitback.mlb.data import lineups as L

NYY, BOS, NYM = 147, 111, 121
POS = ["6", "4", "8", "3", "10", "9", "5", "7", "2"]   # spots 1-9, DH 5th
NO_DH_POS = ["6", "4", "8", "3", "9", "5", "7", "2", "1"]   # the pitcher bats 9th


class Season:
    """Builds games / lineups / players tables one game at a time."""

    def __init__(self):
        self.games, self.lineups, self.players = [], [], {}

    def player(self, pid, throws="R", bats="R", position="SS"):
        self.players[pid] = dict(player_id=pid, name=str(pid), bats=bats, throws=throws, position=position)

    def game(self, pk, date, home, away, home_sp, away_sp, home_bat, away_bat, probable=None,
             season=None, pos=POS, home_pitchers=None, away_pitchers=None):
        season = season or int(date[:4])
        probable = probable or {}
        self.games.append(dict(
            game_pk=pk, season=season, game_type="R", date=date, start=f"{date}T23:05:00Z", doubleheader="N",
            game_number=1, home_id=home, away_id=away,
            home_probable_sp=probable.get("home", home_sp), away_probable_sp=probable.get("away", away_sp),
            home_sp=home_sp, away_sp=away_sp))
        for side, team, sp, bat, pitchers in (("home", home, home_sp, home_bat, home_pitchers),
                                              ("away", away, away_sp, away_bat, away_pitchers)):
            for spot, (pid, p) in enumerate(zip(bat, pos), start=1):
                pid = sp if p == "1" else pid
                self.lineups.append(dict(game_pk=pk, side=side, team_id=team, player_id=pid,
                                         batting_order=spot * 100, spot=spot, starter=True, start_position=p,
                                         is_sp=p == "1", pitcher_order=0 if p == "1" else None))
            for k, pid in enumerate(pitchers or ([] if "1" in pos else [sp])):
                self.lineups.append(dict(game_pk=pk, side=side, team_id=team, player_id=pid,
                                         batting_order=None, spot=None, starter=False, start_position="1",
                                         is_sp=k == 0, pitcher_order=k))
        return self

    def tables(self):
        return pd.DataFrame(self.games), pd.DataFrame(self.lineups), pd.DataFrame(self.players.values())


def bats(proj, pk, side="home"):
    d = proj[(proj.game_pk == pk) & (proj.side == side) & (proj.role == "B")].sort_values("spot")
    return [None if pd.isna(p) else int(p) for p in d.player_id]


def sp_row(proj, pk, side="home"):
    return proj[(proj.game_pk == pk) & (proj.side == side) & (proj.role == "SP")].iloc[0]


VS_R = list(range(1, 10))         # NYY's lineup against right-handers
VS_L = [1, 2, 3, 4, 11, 12, 7, 8, 9]   # against left-handers (two platoon bats)
OPP = list(range(201, 210))


def base():
    s = Season()
    for pid, throws in ((501, "R"), (502, "L"), (503, "R"), (601, "R"), (602, "L"), (603, "S")):
        s.player(pid, throws=throws, position="P")
    for pid in VS_R + VS_L + OPP + [21, 22]:
        s.player(pid, bats="L" if pid % 2 else "R")
    s.game(1, "2024-04-01", NYY, BOS, 501, 601, VS_R, OPP)
    s.game(2, "2024-04-02", NYY, BOS, 502, 602, VS_L, OPP)
    return s


def test_uses_dh():
    assert L.uses_dh(2019, NYY) and not L.uses_dh(2019, NYM)
    assert L.uses_dh(2020, NYM) and not L.uses_dh(2021, NYM) and L.uses_dh(2022, NYM)
    assert L.uses_dh(2013, 117) and not L.uses_dh(2012, 117)


def test_previous_lineup_against_the_same_hand():
    s = base()
    s.game(3, "2024-04-03", NYY, BOS, 503, 603, VS_R, OPP)     # BOS: a switch pitcher, counts as R
    s.game(4, "2024-04-04", NYY, BOS, 501, 602, VS_L, OPP)     # BOS left-hander again
    proj = L.projected_lineups(*s.tables())
    assert bats(proj, 3) == VS_R and set(proj[(proj.game_pk == 3) & (proj.side == "home")].source) == {"prev_hand"}
    assert int(proj[(proj.game_pk == 3) & (proj.side == "home")].ref_game_pk.iloc[0]) == 1
    assert bats(proj, 4) == VS_L
    assert sp_row(proj, 4).player_id == 501 and sp_row(proj, 4).opp_sp_throws == "L"
    # the first game has no history; the second faces a hand not seen yet: the previous lineup
    assert bats(proj, 1) == [] and sp_row(proj, 1).source == "none"
    assert bats(proj, 2) == VS_R and sp_row(proj, 2).source == "prev"


def test_projection_reads_only_earlier_games():
    s = base()
    s.game(3, "2024-04-03", NYY, BOS, 503, 601, VS_R, OPP)
    s.game(4, "2024-04-04", NYY, BOS, 501, 601, VS_R, OPP)
    before = L.projected_lineups(*s.tables())
    # change game 3's own lineup and everything after it: game 3's projection must not move
    s.lineups = [dict(r, player_id=r["player_id"] + 1000) if r["game_pk"] >= 3 and r["starter"] else r
                 for r in s.lineups]
    after = L.projected_lineups(*s.tables())
    cols = ["role", "spot", "player_id", "source"]
    pd.testing.assert_frame_equal(before[before.game_pk <= 3][cols].reset_index(drop=True),
                                  after[after.game_pk <= 3][cols].reset_index(drop=True))
    assert bats(after, 4) == [p + 1000 for p in VS_R]


def test_traded_batter_is_replaced_from_the_position_history():
    s = base()
    s.player(31)
    # 3 (CF) starts at CF for NYY vs a lefty on 04-02 ... then is traded and plays for BOS on 04-03;
    # 21 started at CF for NYY on 04-03 (vs L), so he fills the spot on 04-04 vs R.
    lineup = list(VS_L)
    lineup[2] = 21
    s.game(3, "2024-04-03", NYY, BOS, 503, 602, lineup, [3] + OPP[1:])
    s.game(4, "2024-04-04", NYY, BOS, 501, 601, VS_R, OPP)
    proj = L.projected_lineups(*s.tables())
    d = proj[(proj.game_pk == 4) & (proj.side == "home") & (proj.role == "B")].sort_values("spot")
    assert bats(proj, 4) == [1, 2, 21, 4, 5, 6, 7, 8, 9]
    assert d.source.tolist()[2] == "filled" and d.position.tolist()[2] == "8"


def test_no_dh_game_bats_the_projected_starter():
    s = Season()
    for pid in (501, 502, 601, 602):
        s.player(pid, position="P")
    for pid in VS_R + OPP:
        s.player(pid)
    s.game(1, "2019-04-01", NYM, BOS, 501, 601, VS_R, OPP, pos=NO_DH_POS)
    s.game(2, "2019-04-02", NYM, BOS, 502, 602, VS_R, OPP, pos=NO_DH_POS)
    proj = L.projected_lineups(*s.tables())
    assert not sp_row(proj, 2).dh
    assert bats(proj, 2) == VS_R[:8] + [502]
    assert sp_row(proj, 2).spot == 9
    # an NL team's lineup at an AL park (DH) after a no-DH game: the DH spot gets the team's DH
    s.game(3, "2019-04-03", NYY, NYM, 601, 501, OPP, VS_R)   # NYM away at NYY, DH: no NYM DH yet
    proj = L.projected_lineups(*s.tables())
    d = proj[(proj.game_pk == 3) & (proj.side == "away") & (proj.role == "B")].sort_values("spot")
    assert d.position.tolist()[8] == "10" and d.source.tolist()[8] == "vacant"


def test_posted_is_the_starting_nine():
    s = base()
    s.game(3, "2024-04-03", NYY, BOS, 503, 601, VS_L, OPP)
    proj = L.projected_lineups(*s.tables(), lineup="posted")
    assert bats(proj, 3) == VS_L and set(proj[proj.game_pk == 3].source) == {"posted"}
    assert bats(proj, 1) == VS_R


def test_rotation_starter_and_probable_fallback():
    s = Season()
    rotation = [501, 502, 503, 504, 505]
    for pid in rotation + [601]:
        s.player(pid, position="P")
    for pid in VS_R + OPP:
        s.player(pid)
    for k in range(7):
        s.game(k + 1, f"2024-04-{k + 1:02d}", NYY, BOS, rotation[k % 5], 601, VS_R, OPP,
               probable={"away": None} if k == 6 else None)
    proj = L.projected_lineups(*s.tables(), sp="rotation")
    assert sp_row(proj, 6).player_id == 501 and sp_row(proj, 6).sp_source == "rotation"
    assert sp_row(proj, 2).player_id == 502 and sp_row(proj, 2).sp_source == "probable"
    # BOS's 601 pitched every day: not rested, and no probable in game 7
    assert pd.isna(sp_row(proj, 7, "away").player_id)
    prob = L.projected_lineups(*s.tables())
    assert pd.isna(sp_row(prob, 7, "away").player_id) or sp_row(prob, 7, "away").sp_source == "rotation"


def test_evaluate_against_previous_game():
    s = base()
    s.game(3, "2024-04-03", NYY, BOS, 503, 601, VS_R, OPP)
    s.game(4, "2024-04-04", NYY, BOS, 501, 602, VS_L, OPP)
    g, lu, pl = s.tables()
    ev = L.evaluate(L.projected_lineups(g, lu, pl), g, lu)
    r = ev.loc[2024]
    # games 2-4 (game 1 has no previous game): NYY's 3 and 4 are right by hand, the previous game
    # gets 7 of 9 in each; BOS always fields OPP
    assert r.team_games == 6 and r.sp == 1.0
    assert r.batters == pytest.approx((7 / 9 + 1 + 1 + 3) / 6)
    assert r.prev_batters == pytest.approx((3 * 7 / 9 + 3) / 6) and r.hand_match == pytest.approx(4 / 6)


def test_live_uses_posted_lineup_when_nine_are_listed():
    s = base()
    g, lu, pl = s.tables()
    upcoming = [dict(game_pk=9, game_type="R", date="2024-04-05", state="Scheduled", side="home", team_id=NYY,
                     opp_id=BOS, probable_sp_id=503, lineup=VS_L[::-1]),
                dict(game_pk=9, game_type="R", date="2024-04-05", state="Scheduled", side="away", team_id=BOS,
                     opp_id=NYY, probable_sp_id=602, lineup=[])]
    live = L.project_live(g, lu, pl, upcoming)
    assert bats(live, 9) == VS_L[::-1] and sp_row(live, 9).source == "posted"
    assert sp_row(live, 9).player_id == 503
    assert bats(live, 9, "away") == OPP and sp_row(live, 9, "away").source == "prev_hand"


def test_build_writes_tables(tmp_path: Path):
    s = base()
    s.game(3, "2024-04-03", NYY, BOS, 503, 601, VS_R, OPP)
    g, lu, pl = s.tables()
    g.to_parquet(tmp_path / "games_2024.parquet")
    lu.to_parquet(tmp_path / "lineups_2024.parquet")
    pl.to_parquet(tmp_path / "players_2024.parquet")
    summary = L.build_lineups(tmp_path)
    assert set(summary) == {2024}
    proj = pd.read_parquet(tmp_path / "projected_lineups.parquet")
    assert list(proj.columns) == L.COLUMNS
    assert (tmp_path / "posted_lineups.parquet").exists()
