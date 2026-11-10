import numpy as np
import pandas as pd

from runitback.college import data, lineups
from tests.college.synthetic import league


def _pg():
    return data.build_player_games(league())


def test_player_games_and_team_games():
    pg = _pg()
    assert set(pg.group) <= set(data.GROUPS)
    assert (pg.loc[pg.position.eq("ATH"), "group"] == data.DEFAULT_GROUP).all()
    assert (pg.loc[pg.dnp, "minutes"] == 0).all() and (pg.loc[pg.dnp, "pts"] == 0).all()
    tg = data.team_games(pg)
    assert (tg.groupby("game_id").size() == 2).all()
    # both teams share the game's possessions, from FGA - OREB + TOV + 0.475 FTA
    one = pg[pg.game_id == pg.game_id.iloc[0]]
    own = one.groupby("team_id").apply(lambda d: d.fga.sum() - d.oreb.sum() + d.tov.sum() + 0.475 * d.fta.sum(),
                                       include_groups=False)
    assert np.isclose(tg[tg.game_id == one.game_id.iloc[0]].poss.iloc[0], own.mean())
    assert np.allclose(tg.minutes, 200, atol=1e-6)


def test_projection_removes_dnps_and_scales_minutes():
    pg = _pg()
    proj = lineups.projected_lineups(pg)
    k = proj.groupby(["game_id", "team_id"])
    assert np.allclose(k.proj_minutes.sum(), 200)
    assert proj.proj_minutes.max() <= lineups.MAX_MINUTES + 1e-9
    assert (k.proj_starter.sum() == 5).all()
    out = pg[pg.dnp][["game_id", "team_id", "player_id"]]
    assert out.merge(proj, on=["game_id", "team_id", "player_id"]).empty
    # ranks follow projected minutes
    first = proj[proj["rank"] == 0].set_index(["game_id", "team_id"]).proj_minutes
    assert (first >= k.proj_minutes.max() - 1e-9).all()


def test_projection_is_leak_free_and_transfer_aware():
    pg = _pg()
    proj = lineups.projected_lineups(pg)
    g = pg[(pg.season == 2022) & (pg.team_id == 101) & ~pg.dnp].game_date.min()
    # changing every game from this date on leaves the projected minutes of this date unchanged
    pg2 = pg.copy()
    later = pg2.game_date > g
    pg2.loc[later, "minutes"] = pg2.loc[later, "minutes"] * 0.5
    a = proj[proj.game_date <= g].set_index(["game_id", "team_id", "player_id"]).exp_minutes
    b = lineups.projected_lineups(pg2)
    b = b[b.game_date <= g].set_index(["game_id", "team_id", "player_id"]).exp_minutes
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())
    # the transfer's first game for team 1 uses his games for team 0 (not a newcomer's minutes)
    t = proj[(proj.player_id == 1000) & (proj.team_id == 101)].sort_values("game_date")
    if len(t):
        assert t.recent_games.iloc[0] > 0
        prev = pg[(pg.player_id == 1000) & ~pg.dnp & (pg.team_id == 100)]
        assert t.recent_games.iloc[0] == len(prev)


def test_previous_rule_and_evaluate():
    pg = _pg()
    proj = lineups.projected_lineups(pg, rule="previous")
    assert (proj.source == "previous").all()
    ev = lineups.evaluate(proj, pg)
    assert set(ev.columns) >= {"season", "minutes_mae", "top5", "starters", "missed"}
    assert ((ev.top5 >= 0) & (ev.top5 <= 1)).all()
    ev_dnp = lineups.evaluate(lineups.projected_lineups(pg), pg)
    assert (ev_dnp.missed == 0).all()


def test_scale_minutes_caps():
    out = lineups.scale_minutes(np.array([60.0, 60, 10, 10, 10, 5, 5]), total=200, cap=38)
    assert np.isclose(out.sum(), 200) and out.max() <= 38 + 1e-9
    assert np.isclose(out[0], 38) and out[2] > out[5]
    # five players: 40 each, whatever the cap
    assert np.allclose(lineups.scale_minutes(np.array([60.0, 60, 10, 10, 10]), total=200, cap=38), 40)
