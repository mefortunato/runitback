"""C4: projected rotations of upcoming college games (``runitback.college.live``) and their pv rows."""
import numpy as np
import pandas as pd
import pytest

from runitback.college import data, lineups, live
from runitback.college import player_values as P
from runitback.wager.cbb.lineup import LINEUP
from tests.college.synthetic import league


@pytest.fixture(scope="module")
def tables():
    pg = data.build_player_games(league())
    return pg, data.team_games(pg)


def _event(eid, state, home, away, date="2026-11-20T00:30Z"):
    comp = {"competitors": [{"homeAway": "home", "team": {"id": str(home)}},
                            {"homeAway": "away", "team": {"id": str(away)}}]}
    return {"id": str(eid), "date": date, "status": {"type": {"state": state}}, "competitions": [comp]}


def test_parse_scoreboard_keeps_games_not_started_in_eastern_dates():
    d = live.parse_scoreboard({"leagues": [{"season": {"year": 2027}}],
                               "events": [_event(1, "pre", 10, 11), _event(2, "post", 12, 13),
                                          _event(3, "in", 14, 15)]})
    assert list(d.game_id) == ["1"] and d.season[0] == 2027
    assert (d.home_id[0], d.away_id[0]) == (10, 11)
    assert d.game_date[0] == "2026-11-19"         # 00:30Z is the evening before, US Eastern


def test_parse_injuries_and_roster():
    inj = {"injuries": [{"injuries": [{"status": "Out", "athlete": {"id": "5"}},
                                      {"status": "Day-To-Day", "athlete": {"id": "6"}},
                                      {"status": "Doubtful", "athlete": {"links": [{"href": "x/id/7/name"}]}}]}]}
    assert live.parse_injuries(inj) == {5, 7}
    r = live.parse_roster({"athletes": [{"id": "1", "position": {"abbreviation": "PG"}},
                                        {"id": "2", "position": {"abbreviation": "ATH"}}, {"id": "3"}]})
    assert r == [(1, "G"), (2, None), (3, None)]


def _upcoming(pg, days=3):
    day = (pd.to_datetime(pg.game_date).max() + pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    ids = sorted(pg.team_id.unique())[:2]
    games = pd.DataFrame([dict(game_id="999999", game_date=day, season=int(pg.season.max()),
                               home_id=ids[0], away_id=ids[1])])
    last = pg[pg.season == pg.season.max()]
    rosters = {t: [(int(p), "G") for p in sorted(last[last.team_id == t].player_id.unique())] + [(777, "F")]
               for t in ids}
    return games, rosters, ids, day


def test_project_live_candidates_and_minutes(tables):
    pg, _ = tables
    games, rosters, ids, day = _upcoming(pg)
    out_id = rosters[ids[0]][0][0]
    up = live.project_live(pg, games, rosters, {out_id})
    assert list(up.columns) == live.OUTPUT_COLUMNS
    assert set(up.team_id) == set(ids) and (up.source == "live").all()
    assert 777 not in set(up.player_id)                    # never played: no minutes to project
    assert out_id not in set(up.player_id)                 # ruled out
    assert up.groupby("team_id").proj_minutes.sum().round(6).eq(200.0).all()
    assert up.groupby("team_id").proj_starter.sum().eq(5).all()
    assert up.groupby("team_id").home.nunique().eq(1).all() and up.home.nunique() == 2
    assert (up.game_date == day).all() and up.game_id.eq("999999").all()


def test_project_live_uses_only_games_before(tables):
    pg, _ = tables
    games, rosters, _, day = _upcoming(pg)
    base = live.project_live(pg, games, rosters, set())
    # a game on (or after) the upcoming date cannot move the projection
    late = pg[~pg.dnp].tail(40).assign(game_id="late", game_date=day, minutes=39.0, starter=True)
    again = live.project_live(pd.concat([pg, late], ignore_index=True), games, rosters, set())
    pd.testing.assert_frame_equal(base, again)


def test_project_live_skips_teams_without_history(tables):
    pg, _ = tables
    games, rosters, ids, _ = _upcoming(pg)
    games = games.assign(away_id=424242)                   # a team never seen (non-Division I)
    assert live.project_live(pg, games, rosters, set()).empty


@pytest.mark.parametrize("days", [3, 10])     # the same week as the last game, and a new one (a week not yet fit)
def test_pv_rows_for_upcoming_games(tables, days):
    pg, tg = tables
    games, rosters, ids, day = _upcoming(pg, days)
    proj = lineups.projected_lineups(pg)
    up = live.project_live(pg, games, rosters, set())
    base = P.build(pg, tg, proj, first_season=2022)
    gv = P.build(pg, tg, proj, first_season=2022, upcoming=up)
    rows = gv.pv[gv.pv.game_id == "999999"]
    assert sorted(rows.team_id) == sorted(ids) and rows.season.eq(pg.season.max()).all()
    assert rows[list(LINEUP.columns)].notna().all().all() and rows[list(LINEUP.columns)].abs().to_numpy().sum() > 0
    # the played games' rows are exactly what they were without the upcoming ones
    old = gv.pv[gv.pv.game_id != "999999"].reset_index(drop=True)
    pd.testing.assert_frame_equal(old, base.pv.reset_index(drop=True))
    # the change vs the previous rotation is vs the team's last played game's
    assert np.isfinite(rows[LINEUP.columns].to_numpy()).all()
