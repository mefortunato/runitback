import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from runitback.core.player_values import fit_apm
from runitback.mlb.player_values import (
    BULLPEN_GAMES,
    GROUPS,
    PV_COLUMNS,
    SP_SHRINK,
    Tables,
    apm_config,
    bullpens,
    day_number,
    events_from,
    expected_sp_bf,
    fit_values,
    game_table,
    holdout_score,
    league_bf,
    pitcher_games,
    prepare_games,
    prepare_pa,
    previous_lineup,
    slot_pa,
    starters,
    to_wager_keys,
    week_start,
)
from runitback.wager.mlb.lineup import LINEUP

TEAMS = (10, 20, 30, 40)
SEASONS = (2020, 2021)
FIXTURES = Path(__file__).parent / "fixtures"


def bat_ids(team):
    return [team * 100 + k for k in range(1, 11)]          # 9 regulars and a bench bat (10th)


def sp_ids(team):
    return [team * 100 + 50 + k for k in range(3)]


def rp_ids(team):
    return [team * 100 + 60 + k for k in range(4)]


def skill(pid):
    """Batters: + runs per PA; pitchers: runs prevented per PA."""
    return ((pid * 7919) % 13 - 6) / 60


def synthetic(games_per_season=36, seed=0) -> Tables:
    rng = np.random.default_rng(seed)
    games, pa, lineups = [], [], []
    pk = 1000
    rounds = [((10, 20), (30, 40)), ((30, 10), (40, 20)), ((10, 40), (20, 30)),
              ((20, 10), (40, 30)), ((30, 10), (20, 40)), ((40, 10), (30, 20))]
    for season in SEASONS:
        day = datetime.date(season, 4, 1)
        for i in range(games_per_season):
            for home, away in rounds[i % len(rounds)]:
                pk += 1
                sps = {t: sp_ids(t)[i % 3] for t in (home, away)}
                games.append(dict(game_pk=pk, season=season, game_type="R", date=day.isoformat(),
                                  start=f"{day.isoformat()}T19:00:00Z", final=True, venue_id=home,
                                  home_id=home, away_id=away, home_sp=sps[home], away_sp=sps[away]))
                ab = 0
                for side, bat, fld in (("away", away, home), ("home", home, away)):
                    nine = bat_ids(bat)[:9]
                    if i % 4 == 3:                     # the bench bat starts at spot 9 every 4th game
                        nine = nine[:8] + [bat_ids(bat)[9]]
                    for k, p in enumerate(nine):
                        lineups.append(dict(game_pk=pk, side=side, team_id=bat, player_id=p,
                                            batting_order=(k + 1) * 100, spot=k + 1, starter=True,
                                            start_position=str(k % 8 + 2), is_sp=False))
                    pitchers = [sps[fld]] + rp_ids(fld)[i % 2: i % 2 + 2]
                    for p in pitchers:
                        lineups.append(dict(game_pk=pk, side="home" if fld == home else "away", team_id=fld,
                                            player_id=p, batting_order=None, spot=None, starter=False,
                                            start_position="1", is_sp=p == sps[fld]))
                    for n in range(36):
                        pit = pitchers[0] if n < 24 else pitchers[1 + (n - 24) // 6]
                        b = nine[n % 9]
                        rv = skill(b) - skill(pit) + rng.normal(0, 0.25)
                        pa.append(dict(game_pk=pk, season=season, at_bat_index=ab, bat_team_id=bat,
                                       fld_team_id=fld, bat_home=int(bat == home), batter_id=b,
                                       bat_side="LR"[b % 2], spot=n % 9 + 1, pitcher_id=pit,
                                       pitch_hand="LR"[pit % 2], pre_bases=n % 8, pre_outs=n % 3,
                                       run_value=rv, **{f"fielder_{q}": None for q in range(2, 10)}))
                        ab += 1
            day += datetime.timedelta(days=1)
    players = pd.DataFrame([dict(player_id=p, name=f"p{p}", position="P" if p % 100 >= 50 else "RF")
                            for t in TEAMS for p in bat_ids(t) + sp_ids(t) + rp_ids(t)])
    g = prepare_games(pd.DataFrame(games))
    return Tables(prepare_pa(pd.DataFrame(pa), g), g, pd.DataFrame(lineups), players)


@pytest.fixture(scope="module")
def tb():
    return synthetic()


def test_layout():
    assert LINEUP.groups == ("BAT", "SP", "BP")
    assert LINEUP.width("pvc") == 8 and PV_COLUMNS == [f"_pv_{i}" for i in range(8)]
    assert LINEUP.total_sign == (1.0, -1.0, -1.0)


def test_week_start():
    t = day_number("2024-07-03")                # a Wednesday
    assert datetime.date(1970, 1, 1) + datetime.timedelta(days=week_start(t)) == datetime.date(2024, 7, 1)


def test_prepare_pa_roles(tb):
    pa = tb.pa
    g = pa[pa.game_pk == pa.game_pk.iloc[0]]
    first = g[g.bat_home == 0]
    # the starter's first 9 / next 9 / next 6 batters faced: times through the order 0, 1, 2; then relievers
    assert list(first.role) == [0] * 9 + [1] * 9 + [2] * 6 + [4] * 12
    assert first.is_sp.sum() == 24
    assert set(pa.hand) <= {0, 1, 2, 3} and pa.state.between(0, 23).all()
    assert (np.diff(pa.t.to_numpy()) >= 0).all()


def test_events_role_specific_codes(tb):
    players = tb.players.copy()
    two_way = sp_ids(10)[0]
    pa = tb.pa.copy()
    pa.loc[pa.index[:5], "batter_id"] = two_way          # a pitcher also bats (group PB: position P)
    ev = events_from(pa, players)
    a, b = ev.segments["bat"]
    c, d = ev.segments["pit"]
    assert b == c and d == len(ev.ids)
    assert two_way in ev.ids[a:b] and two_way in ev.ids[c:d]
    bat_code, pit_code = ev.codes("bat", [two_way])[0], ev.codes("pit", [two_way])[0]
    assert bat_code != pit_code
    assert GROUPS[ev.group[bat_code]] == "PB" and GROUPS[ev.group[pit_code]] == "P"
    assert ev.codes("bat", [999999])[0] == -1
    # a position player pitching (a blowout) has his own replacement group
    pa.loc[pa.index[5:8], "pitcher_id"] = bat_ids(20)[0]
    ev = events_from(pa, players)
    assert GROUPS[ev.group[ev.codes("pit", [bat_ids(20)[0]])[0]]] == "XP"
    assert (ev.sides["bat"][:5, 0] == bat_code).all()


def test_fit_recovers_skills(tb):
    ev = events_from(tb.pa, tb.players)
    cfg = apm_config(lam=50, min_pa=50)
    tab = fit_values(ev, int(tb.pa.t.max()) + 1, 2021, cfg)
    for side in ("bat", "pit"):
        t = tab[tab.side == side]
        r = np.corrcoef(t.value_pa, t.player_id.map(skill))[0, 1]
        assert r > 0.8, (side, r)              # higher is better on both sides
    score = holdout_score(ev, 2021, cfg)
    assert score["mse_gain"] > 0 and score["corr"] > 0.1


def test_fit_is_leak_free(tb):
    ev = events_from(tb.pa, tb.players)
    cfg = apm_config(lam=50, min_pa=50)
    cut = int(tb.pa.t[tb.pa.season == 2021].min()) + 7
    a = fit_apm(ev, cfg, cut, 2021)
    fr = ev.frame.copy()
    later = (fr.t >= cut).to_numpy()
    fr.loc[later, "run_value"] = 5.0
    ev2 = type(ev)(fr, {k: v.copy() for k, v in ev.sides.items()}, ev.ids, ev.group, ev.segments)
    ev2.sides["bat"][later] = ev.sides["bat"][0, 0]
    b = fit_apm(ev2, cfg, cut, 2021)
    pd.testing.assert_frame_equal(a, b)


def test_usage(tb):
    slots = slot_pa(tb.pa, tb.lineups)
    assert sorted(slots) == list(SEASONS) and slots[2020] == slots[2021]
    assert slots[2020] == [4.0] * 9                       # 36 PAs a side over nine spots
    pg = pitcher_games(tb.pa)
    league = league_bf(pg)
    assert league[2021] == (24.0, 12.0)
    exp = expected_sp_bf(pg, league)
    first = exp.groupby(level="pitcher_id").first()
    assert np.allclose(first, 24.0)                       # no starts yet: the league mean
    assert np.allclose(exp, 24.0)                         # every start was 24 batters


def test_expected_sp_bf_shrinks():
    pg = pd.DataFrame(dict(game_pk=[1, 2, 3], t=[1, 2, 3], season=2020, team_id=1, pitcher_id=7, is_sp=True,
                           bf=[30, 30, 30]))
    exp = expected_sp_bf(pg, {2020: (20.0, 15.0)})
    assert exp.tolist() == [20.0, (30 + SP_SHRINK * 20) / (1 + SP_SHRINK), (60 + SP_SHRINK * 20) / (2 + SP_SHRINK)]


def test_bullpens_use_earlier_games_only(tb):
    pg = pitcher_games(tb.pa)
    pens = bullpens(pg, tb.games)
    team = TEAMS[0]
    tg = tb.games[(tb.games.home_id == team) | (tb.games.away_id == team)].sort_values("order")
    first, second = int(tg.game_pk.iloc[0]), int(tg.game_pk.iloc[1])
    assert (first, team) not in pens
    used = set(pg[(pg.game_pk == first) & (pg.team_id == team) & ~pg.is_sp].pitcher_id)
    assert {p for p, _ in pens[(second, team)]} == used
    w = pens[(int(tg.game_pk.iloc[BULLPEN_GAMES + 3]), team)]
    assert abs(sum(s for _, s in w) - 1) < 1e-9 and {p for p, _ in w} <= set(rp_ids(team))


def test_previous_lineup(tb):
    prev = previous_lineup(tb.lineups, tb.games)
    team = TEAMS[1]
    tg = tb.games[(tb.games.home_id == team) | (tb.games.away_id == team)].sort_values("order").game_pk.tolist()
    st = starters(tb.lineups)
    got = prev[(prev.game_pk == tg[5]) & (prev.team_id == team)].sort_values("spot")
    want = st[(st.game_pk == tg[4]) & (st.team_id == team)].sort_values("spot")
    assert got.player_id.tolist() == want.player_id.tolist() and len(got) == 9
    assert prev[(prev.game_pk == tg[0]) & (prev.team_id == team)].empty


def test_game_table(tb):
    cfg = apm_config(lam=50, min_pa=50)
    gv = game_table(tb, first_season=2021, jobs=1, cfg=cfg)
    pv = gv.pv
    assert list(pv.columns) == ["game_pk", "team_id", *PV_COLUMNS]
    n2021 = tb.games[tb.games.season == 2021]
    assert len(pv) == 2 * len(n2021)                      # every 2021 team-game (each had a previous game)
    assert not pv[PV_COLUMNS].isna().any().any()
    # the stand-in projection is the previous game's nine, so the lineup change vs it is 0
    assert np.allclose(pv._pv_3, 0)
    # the starters rotate, so the pitching change is not
    assert (pv._pv_4.abs() > 0).mean() > 0.5
    # level columns follow the skills: the team with the best regulars has the best BAT value
    bat = pv.groupby("team_id")._pv_0.mean()
    true = {t: sum(skill(p) for p in bat_ids(t)[:9]) for t in TEAMS}
    assert bat.idxmax() == max(true, key=true.get)
    # SP: one starter's value times his expected batters faced (24)
    row = pv.iloc[-1]
    g = tb.games.set_index("game_pk").loc[row.game_pk]
    sp_id = g.home_sp if g.home_id == row.team_id else g.away_sp
    fit = gv.fits[week_start(int(g.t))]
    v = fit[(fit.side == "pit") & (fit.player_id == sp_id)].value_pa
    assert row._pv_1 == pytest.approx(float(v.iloc[0]) * 24 if len(v) else 0.0, rel=1e-5)
    keyed = to_wager_keys(pv, {t: f"T{t}" for t in TEAMS})
    assert keyed.game_id.iloc[0] == str(pv.game_pk.iloc[0]) and keyed.team.iloc[0] == f"T{pv.team_id.iloc[0]}"


def test_game_table_week_fits_are_leak_free(tb):
    """A game's values come from its week's fit, which sees no PA from that week on."""
    cfg = apm_config(lam=50, min_pa=50)
    gv = game_table(tb, first_season=2021, jobs=1, cfg=cfg)
    w = sorted(gv.fits)[2]
    ev = events_from(tb.pa, tb.players)
    pd.testing.assert_frame_equal(gv.fits[w], fit_values(ev, w, 2021, cfg))


def test_on_real_feeds():
    """The three M1 fixture feeds through the PA preparation (run values from their own run
    expectancy, only to have numbers)."""
    import gzip
    import json

    from runitback.mlb.data.ingest import parse_season
    from runitback.mlb.data.run_values import add_run_values, run_expectancy

    feeds = [json.load(gzip.open(p, "rt")) for p in sorted(FIXTURES.glob("feed_*.json.gz"))]
    tables, _ = parse_season(feeds)
    pa = add_run_values(tables["pa"], run_expectancy(tables["pa"]))
    games = prepare_games(tables["games"])
    p = prepare_pa(pa, games)
    assert len(p) == len(pa) and p.is_sp.any() and (p.role == 4).any()
    # each starter's PAs are the first ones his team faced
    for (g, team), d in p.groupby(["game_pk", "fld_team_id"]):
        sp = d.is_sp.to_numpy()
        assert sp[0] and (sp.all() or not sp[np.argmin(sp):].any())
    st = starters(tables["lineups"])
    assert (st.groupby(["game_pk", "team_id"]).size() == 9).all()
