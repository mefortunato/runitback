import datetime

import numpy as np
import pandas as pd
import pytest

from runitback.core.player_values import fit_apm
from runitback.nhl.data.events import normalize
from runitback.nhl.player_values import (
    MAX_SKATERS, SEG_COLUMNS, START_DZ, START_FLY, START_NZ, START_OZ, apm_config, day_number, events_from,
    fit_values, from_projected, game_table, goalie_values, player_games, player_groups, previous_lineup, season_segments, season_shots,
    segment_rows, to_wager_teams, week_start,
)
from runitback.wager.nhl.lineup import LINEUP

from .test_events import AG, GOALIES, HG, game_rows


def test_segments_and_shots_from_events():
    ev = normalize(game_rows(), 2025, GOALIES)
    seg = season_segments(ev)
    # opening lines 0-30 (home shot), home change 30-100, away penalty 100-150 (home goal); then the
    # away goalie is pulled (not fit) and the opening zero-length change is dropped
    assert seg.seconds.tolist() == [30, 70, 50]
    np.testing.assert_allclose(seg.xg_home, [0.05, 0.0, 0.3])
    np.testing.assert_allclose(seg.xg_away, [0.0, 0.0, 0.0])
    assert (seg.home_goalie == HG).all() and (seg.away_goalie == AG).all()
    assert seg[[f"home_{i}" for i in range(1, 6)]].iloc[1].tolist() == [11, 12, 13, 16, 17]
    assert seg[[f"away_{i}" for i in range(1, 6)]].iloc[2].tolist() == [21, 22, 23, 24, -1]
    assert seg.home_lead.tolist() == [0, 0, 0] and list(seg.columns) == SEG_COLUMNS
    frame, off, dfn = segment_rows(seg)
    # home attacking 5 on 4 in the third segment: strength (5-3)*3 + (4-3); away attacking: 4 on 5
    assert frame.strength.tolist() == [8, 8, 7, 8, 8, 5]
    assert frame.off_home.tolist() == [True] * 3 + [False] * 3
    np.testing.assert_allclose(frame.xg60[:3], [0.05 * 120, 0.0, 0.3 * 72])
    assert (off[3:] == dfn[:3]).all()
    sh = season_shots(ev)
    # the away shot while their own goalie is pulled is still at the home goalie
    assert sh.goalie.tolist() == [AG, AG, HG] and sh.goal.tolist() == [False, True, False]


def test_zone_starts_flip_for_the_away_side():
    seg = pd.DataFrame({c: [0] * 4 for c in SEG_COLUMNS})
    seg["start_x"] = [np.nan, 60.0, 0.0, -70.0]
    seg["seconds"] = 60
    for i in range(1, 6):
        seg[f"home_{i}"] = i
        seg[f"away_{i}"] = 10 + i
    frame, _, _ = segment_rows(seg)
    assert frame.start.tolist() == [START_FLY, START_OZ, START_NZ, START_DZ, START_FLY, START_DZ, START_NZ, START_OZ]


# -- synthetic league -------------------------------------------------------------------------------

def synthetic(seed=0, teams=4, seasons=(2022, 2023), games_per_team=30, segs_per_game=90):
    """Shift segments with planted skater offense and defense effects (xG per 60) and goalie effects
    (goals saved per shot), plus the games table and a player directory. Each team has 14 forwards,
    7 defensemen and 2 goalies; the first 12 / 6 / 1 play most."""
    rng = np.random.default_rng(seed)
    fw = {t: [10000 * (t + 1) + k for k in range(14)] for t in range(teams)}
    dm = {t: [10000 * (t + 1) + 100 + k for k in range(7)] for t in range(teams)}
    gk = {t: [10000 * (t + 1) + 900, 10000 * (t + 1) + 901] for t in range(teams)}
    skaters = [p for d in (fw, dm) for r in d.values() for p in r]
    off_eff = {p: rng.normal(0, 0.4) for p in skaters}
    def_eff = {p: rng.normal(0, 0.4) for p in skaters}
    g_eff = {g: rng.normal(0, 0.05) for r in gk.values() for g in r}
    wf = np.r_[np.full(12, 1.0), 0.05, 0.05]
    wd = np.r_[np.full(6, 1.0), 0.05]
    wf, wd = wf / wf.sum(), wd / wd.sum()
    segs, shots, games = [], [], []
    for s in seasons:
        day = datetime.date(s - 1, 10, 10)
        for gn in range(games_per_team * teams // 2):
            day += datetime.timedelta(days=1)
            h, a = (int(x) for x in rng.permutation(teams)[:2])
            gid = s * 1_000_000 + 20000 + gn
            games.append(dict(game_id=gid, season=s, game_date=day.isoformat(), home=f"T{h}", away=f"T{a}"))
            hg = gk[h][0] if rng.random() < 0.8 else gk[h][1]
            ag = gk[a][0] if rng.random() < 0.8 else gk[a][1]
            for k in range(segs_per_game):
                line = {t: sorted([*rng.choice(fw[t], 3, replace=False, p=wf),
                                   *rng.choice(dm[t], 2, replace=False, p=wd)]) for t in (h, a)}
                secs = 40
                xg = {}
                for o, d in ((h, a), (a, h)):
                    rate = 2.5 + 0.2 * (o == h) + sum(off_eff[p] for p in line[o]) - sum(def_eff[p] for p in line[d])
                    xg[o] = max(rate, 0.1) * secs / 3600 * rng.gamma(2.0, 0.5)
                row = dict(game_id=gid, season=s, game_type="REG", game_date=day.isoformat(), t=0, period=1 + k // 30,
                           seconds=secs, start_x=np.nan, home_lead=0, xg_home=xg[h], xg_away=xg[a],
                           home_goalie=hg, away_goalie=ag)
                for i in range(MAX_SKATERS):
                    row[f"home_{i + 1}"] = line[h][i]
                    row[f"away_{i + 1}"] = line[a][i]
                segs.append(row)
            for g in (hg, ag):
                x = rng.uniform(0.05, 0.3, 30)
                goal = rng.random(30) < np.clip(x - g_eff[g], 0, 1)
                shots += [dict(game_id=gid, season=s, t=day_number(day.isoformat()), goalie=g, xg=xi, goal=gi)
                          for xi, gi in zip(x, goal)]
    seg = pd.DataFrame(segs)
    seg["t"] = seg.game_date.map(day_number).astype(np.int32)
    players = pd.DataFrame({"player_id": [*skaters, *[g for r in gk.values() for g in r]]})
    players["group"] = np.where(players.player_id % 10000 >= 900, "G",
                                np.where(players.player_id % 10000 >= 100, "D", "F"))
    players["name"] = "P" + players.player_id.astype(str)
    players.loc[players.player_id == 10013, "group"] = None   # no NHL position: the default group
    return (seg[SEG_COLUMNS], pd.DataFrame(shots), pd.DataFrame(games), players, off_eff, def_eff, g_eff)


@pytest.fixture(scope="module")
def league():
    return synthetic()


def group_of(players):
    return player_groups(players)


def test_fit_recovers_offense_and_defense(league):
    seg, _, _, players, off_eff, def_eff, _ = league
    ev = events_from(seg, group_of(players))
    cfg = apm_config(lam=3, min_toi=200)
    tab = fit_values(ev, before=int(seg.t.max()) + 1, season=2023, cfg=cfg)
    assert set(tab.side) == {"off", "def"} and set(tab.group) == {"F", "D"}
    for side, eff in (("off", off_eff), ("def", def_eff)):
        t = tab[tab.side == side]
        assert np.corrcoef(t.value_60, t.player_id.map(eff))[0, 1] > 0.6   # defense: higher is better
    # the rarely used extra skaters share the replacement level; goalies are not skaters
    assert not tab.player_id.isin([10012, 10013, 10106, 20012]).any()
    assert not (tab.player_id % 10000 >= 900).any()


def test_fits_are_leak_free(league):
    seg, _, _, players, _, _, _ = league
    before = week_start(day_number("2023-01-02"))
    cfg = apm_config(lam=3, min_toi=200)
    a = fit_apm(events_from(seg, group_of(players)), cfg, before, 2023)
    later = (seg.t >= before).to_numpy()
    seg2 = seg.assign(xg_home=np.where(later, 5.0, seg.xg_home), seconds=np.where(later, 7, seg.seconds))
    b = fit_apm(events_from(seg2, group_of(players)), cfg, before, 2023)
    pd.testing.assert_frame_equal(a, b)


def test_goalie_values(league):
    _, shots, _, _, _, _, g_eff = league
    end = int(shots.t.max()) + 1
    tab = goalie_values(shots, end, 2023, shrink=200)
    assert set(tab.player_id) == set(g_eff)
    assert np.corrcoef(tab.value_shot, tab.player_id.map(g_eff))[0, 1] > 0.6
    # the backups (fewer shots) are shrunk harder toward the league
    starters = tab.player_id % 10 == 0
    raw = shots.assign(r=shots.xg - shots.goal).groupby("goalie").r.mean()
    ratio = tab.value_shot.abs() / (raw - raw.mean()).abs().reindex(tab.player_id).to_numpy()
    assert ratio[starters].mean() > ratio[~starters].mean()
    # leak-free: later shots don't move the values
    before = week_start(day_number("2023-01-02"))
    later = shots.t >= before
    a = goalie_values(shots, before, 2023, 200)
    b = goalie_values(shots.assign(goal=np.where(later, True, shots.goal)), before, 2023, 200)
    pd.testing.assert_frame_equal(a, b)


def test_player_games_and_previous_lineup(league):
    seg, _, games, players, _, _, _ = league
    pg = player_games(seg, games, group_of(players))
    one = pg[(pg.game_id == games.game_id.iloc[0]) & (pg.team == games.home.iloc[0])]
    assert set(one.group) == {"F", "D", "G"}
    f = one[one.group == "F"].sort_values("rank")
    assert f.toi.is_monotonic_decreasing and f["rank"].tolist() == list(range(len(f)))
    assert len(one[one.group == "G"]) == 1 and one[one.group == "G"].toi.iloc[0] == pytest.approx(60.0)
    pr = previous_lineup(pg)
    team = games.home.iloc[0]
    ids = pg[pg.team == team].drop_duplicates("game_id").sort_values("game_date").game_id.tolist()
    assert pr[(pr.game_id == ids[0]) & (pr.team == team)].empty   # no earlier game
    got = pr[(pr.game_id == ids[3]) & (pr.team == team)].sort_values(["group", "rank"])
    want = pg[(pg.game_id == ids[2]) & (pg.team == team) & ((pg.group != "G") | (pg["rank"] == 0))]
    assert got.player_id.tolist() == want.sort_values(["group", "rank"]).player_id.tolist()


def test_game_table_layout_and_leakage(league):
    seg, shots, games, players, _, _, _ = league
    cfg = apm_config(lam=3, min_toi=200)
    gv = game_table(seg, shots, games, players, first_season=2023, jobs=1, cfg=cfg, goalie_shrink=200)
    assert list(gv.pv.columns) == ["game_id", "team", *LINEUP.columns] and len(LINEUP.columns) == 13
    assert (gv.pv.game_id // 1_000_000 == 2023).all() and len(gv.pv) > 0
    assert not gv.pv.duplicated(["game_id", "team"]).any()
    # previous game's lineup as the projection: no skater change vs the previous game
    assert (gv.pv[["_pv_5", "_pv_6"]].abs() < 1e-5).all().all()
    # the starting goalie changes between some games, and so does the goalie column
    assert gv.pv._pv_4.nunique() > 1
    # each game uses the fit on the segments before its Monday
    g = games[games.season == 2023].sort_values("game_date").iloc[30]
    w = week_start(day_number(g.game_date))
    assert w in gv.fits
    want = fit_apm(events_from(seg, group_of(players)), cfg, w, 2023)
    np.testing.assert_allclose(gv.fits[w].value_60.to_numpy(), want.value.to_numpy())
    # usage: top-line minutes over the fourth line's, and 2023 uses 2022's
    assert gv.usage.toi[2023] == gv.usage.toi[2022] and gv.usage.toi[2023]["F"][0] > gv.usage.toi[2023]["F"][12]
    assert gv.usage.shots[2023] == pytest.approx(30.0)
    # a projected lineup with a better goalie raises the goalie column
    row = gv.pv.iloc[-1]
    gid, team = row.game_id, row.team
    lineup = previous_lineup(player_games(seg, games, group_of(players)))
    lu = lineup[(lineup.game_id == gid) & (lineup.team == team)]
    best = gv.goalie_fits[max(gv.goalie_fits)].sort_values("value_shot").player_id.iloc[-1]
    swapped = pd.concat([lu[lu.group != "G"], lu[lu.group == "G"].assign(player_id=best)])
    gv2 = game_table(seg, shots, games, players, first_season=2023, jobs=1, cfg=cfg, goalie_shrink=200,
                     lineups=swapped)
    assert gv2.pv.set_index(["game_id", "team"]).loc[(gid, team), "_pv_4"] >= row._pv_4


def test_to_wager_teams():
    pv = pd.DataFrame({"game_id": [2011020001, 2020020001], "team": ["ATL", "ARI"]})
    out = to_wager_teams(pv)
    assert out.game_id.tolist() == ["2011020001", "2020020001"] and out.team.tolist() == ["WPG", "UTA"]


def test_from_projected_picks_one_goalie():
    proj = pd.DataFrame({
        "game_id": [5] * 5, "team": ["BUF"] * 5, "player_id": [11, 12, 21, 1, 2], "group": ["F", "F", "D", "G", "G"],
        "slot": [2, 1, 1, 1, 2], "goalie_prev": [False, False, False, True, False],
        "goalie_actual": [False, False, False, False, True]})
    prev, act = from_projected(proj, "prev"), from_projected(proj, "actual")
    assert prev[prev.group != "G"][["player_id", "rank"]].values.tolist() == [[11, 1], [12, 0], [21, 0]]
    assert prev[prev.group == "G"][["player_id", "rank"]].values.tolist() == [[1, 0]]
    assert act[act.group == "G"][["player_id", "rank"]].values.tolist() == [[2, 0]]
