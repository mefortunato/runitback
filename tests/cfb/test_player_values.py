"""College football QB values (work package F2): id mapping, quarterback plays, starters, leak-free
fits and the per-game ``_pv_*`` columns, on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from runitback.cfb import player_values as pv
from runitback.cfb.data import players as pl
from runitback.wager.cfb.lineup import LINEUP

# -- id mapping -------------------------------------------------------------------------------------

def test_norm_name():
    assert pl.norm_name("C.J. Stroud") == pl.norm_name("CJ Stroud") == "cj stroud"
    assert pl.norm_name("Kenneth Walker III") == "kenneth walker"
    assert pl.norm_name("Ja'Marr Chase") == "jamarr chase"
    assert pl.norm_name("José Pérez Jr.") == "jose perez"
    assert pl.norm_name(None) == ""
    assert pl.norm_name("Ryan Griffin slant") == "ryan griffin"
    assert pl.norm_name("Kawann Short") == "kawann short"
    assert pl.norm_name("Plitt,Drew") == "drew plitt"


def _box(rows):
    return pd.DataFrame(rows, columns=["game_id", "team_id", "athlete_id", "athlete_name"]).assign(season=2024)


def test_map_ids_keeps_pbp_ids_and_fills_by_name():
    box = _box([
        (1, 10, 100, "C.J. Stroud"), (1, 10, 101, "Bob Smith"), (1, 20, 200, "Al Jones"),
        (1, 20, 201, "Al Jones"),            # two Al Jones on one team: ambiguous, never matched
        (1, 20, 202, "Carl Lastonly"),
        (2, 10, 103, "Dan Backup"),          # in game 2's box score only
    ])
    pbp = pd.DataFrame({
        "game_id": [1, 1, 1, 1, 1, 1, 1],
        "season": 2024,
        "pos_team_id": [10, 10, 20, 20, 10, 10, 10],
        "passer_player_name": ["CJ Stroud", "CJ Stroud", "Al Jones", "C. Lastonly", None, "Nobody Here",
                               "Dan Backup"],
        "passer_player_id": [100] + [np.nan] * 6,
        "rusher_player_name": [None, None, None, None, "Bob Smith", None, None],
        "rusher_player_id": [np.nan] * 7,
    })
    out, summary = pl.map_ids(pbp, box)
    ids = out.passer_id
    assert ids.iloc[0] == 100 and ids.iloc[1] == 100 and ids.iloc[3] == 202   # row 3: last name, unique
    assert ids.iloc[6] == 103                                                 # same team, same season
    assert ids.iloc[2:3].isna().all() and ids.iloc[4:6].isna().all()
    assert out.passer_src.tolist() == ["pbp", "name", "none", "name", "none", "none", "name"]
    assert out.rusher_id.iloc[4] == 101 and out.rusher_src.iloc[4] == "name"
    s = summary["passer"]
    assert s["rows_with_name"] == 6
    assert s["pbp_id"] == pytest.approx(1 / 6) and s["name_match"] == pytest.approx(3 / 6)
    assert s["unmapped"] == pytest.approx(2 / 6)
    assert s["name_match_agrees"] == 1.0           # row 0: the name match finds the pbp's own id


def test_map_ids_drops_the_team_placeholder():
    box = _box([(1, 10, 100, "Bob Smith"), (1, 10, 999, "TEAM")])
    pbp = pd.DataFrame({"game_id": [1, 1], "season": 2024, "pos_team_id": [10, 10],
                        "passer_player_name": ["TEAM", "Team"], "passer_player_id": [999, np.nan],
                        "rusher_player_name": [None, None], "rusher_player_id": [np.nan, np.nan]})
    out, summary = pl.map_ids(pbp, box)
    assert out.passer_id.isna().all() and (out.passer_src == "none").all()
    assert summary["passer"]["rows_with_name"] == 0


# -- quarterback plays and starters -------------------------------------------------------------------

def _pbp(rows):
    """rows: (game, play#, off, def, pass, rush, sack, kneel, passer, rusher, epa)."""
    df = pd.DataFrame(rows, columns=["game_id", "game_play_number", "pos_team_id", "def_pos_team_id", "pass",
                                     "rush", "sack", "kneel_down", "passer_id", "rusher_id", "EPA"])
    df["passer_id"] = df.passer_id.astype("Int64")
    df["rusher_id"] = df.rusher_id.astype("Int64")
    df = df.assign(game_id=df.game_id.astype(str), season=2024, scrimmage_play=True, homeTeamId=10, down=1,
                   distance=10, pos_score_diff_start=0, **{"start.yardsToEndzone": 75})
    return df


def test_qb_plays_dropbacks_and_qb_runs():
    rows = [("1", i, 10, 20, True, False, False, False, 7, None, 0.1) for i in range(pl.QB_MIN_DROPBACKS)]
    rows += [
        ("1", 100, 10, 20, True, False, True, False, 7, None, -2.0),    # sack: a dropback
        ("1", 101, 10, 20, False, True, False, False, None, 7, 0.5),    # QB run
        ("1", 102, 10, 20, False, True, False, True, None, 7, -0.1),    # kneel-down: excluded
        ("1", 103, 10, 20, False, True, False, False, None, 8, 0.3),    # RB run: excluded
        ("1", 104, 20, 10, True, False, False, False, 9, None, 0.2),    # the away team's passer
        ("2", 1, 10, 20, True, False, False, False, 7, None, 0.2),      # unscheduled game: dropped
    ]
    days = pd.DataFrame({"game_id": ["1"], "t": [19000], "neutral": [False]})
    p = pl.qb_plays(_pbp(rows), days)
    assert list(p.columns) == pl.PLAY_COLUMNS
    assert (p.game_id == "1").all()
    assert (p.kind == pl.DROPBACK).sum() == pl.QB_MIN_DROPBACKS + 2
    runs = p[p.kind == pl.RUSH]
    assert len(runs) == 1 and runs.qb.iloc[0] == 7 and runs.epa.iloc[0] == 0.5
    assert set(p[p.off == 10].home) == {1} and set(p[p.off == 20].home) == {-1}
    p_no = pl.qb_plays(_pbp(rows), days, rushes=False)
    assert (p_no.kind == pl.DROPBACK).all()
    neutral = pl.qb_plays(_pbp(rows), days.assign(neutral=True))
    assert (neutral.home == 0).all()
    assert (p.def_fbs == 1).all()
    lvl = pl.qb_plays(_pbp(rows), days, fbs={(2024, 20)})        # team 10 is FCS this season
    assert set(lvl[lvl["def"] == 20].def_fbs) == {1} and set(lvl[lvl["def"] == 10].def_fbs) == {0}


def test_team_games_starter_ignores_a_trick_play():
    rows = [("1", 0, 10, 20, True, False, False, False, 55, None, 3.0)]          # WR pass on the first snap
    rows += [("1", i, 10, 20, True, False, False, False, 7, None, 0.1) for i in range(1, 30)]
    rows += [("1", 30 + i, 10, 20, True, False, False, False, 8, None, 0.1) for i in range(40)]   # backup
    days = pd.DataFrame({"game_id": ["1"], "t": [19000], "neutral": [False]})
    tg = pl.team_games(pl.qb_plays(_pbp(rows), days, rushes=False))
    row = tg.iloc[0]
    assert row.starter == 7 and row.leader == 8 and row.dropbacks == 70 and row.starter_dropbacks == 29


def test_game_days_us_time():
    sched = pd.DataFrame({"game_id": [1, 2], "start_date": ["2024-09-01T03:30:00.000Z", "2024-08-31T19:00Z"],
                          "neutral_site": [False, True]})
    d = pl.game_days(sched)
    assert d.t.tolist() == [pl.day_number("2024-08-31")] * 2       # a late Saturday kickoff is Saturday
    assert d.neutral.tolist() == [False, True]


# -- fits and per-game columns ----------------------------------------------------------------------

def synthetic_plays(seed: int = 0) -> tuple[pd.DataFrame, dict[int, float]]:
    """Six teams, three seasons of weekly games; each team has a starter (and team 1 changes starter
    in the last season's week 6). Each play's EPA is its quarterback's quality plus noise."""
    rng = np.random.default_rng(seed)
    quality = {100 + i: q for i, q in enumerate([0.3, 0.15, 0.0, -0.1, -0.2, 0.05, -0.35])}
    starter_of = {t: 100 + t - 1 for t in range(1, 7)}
    rows = []
    for season in (2019, 2020, 2021):
        for week in range(10):
            t = pl.day_number(f"{season}-09-07") + 7 * week
            order = rng.permutation(6) + 1
            for k in range(3):
                home, away = int(order[2 * k]), int(order[2 * k + 1])
                gid = f"{season}{week:02d}{k}"
                for off, dfn in ((home, away), (away, home)):
                    qb = starter_of[off]
                    if off == 1 and season == 2021 and week >= 6:
                        qb = 106
                    for i in range(40):
                        rows.append((gid, season, t, i, off, dfn, 1 if off == home else -1, qb,
                                     int(i % 5 == 4), quality[qb] + rng.normal(0, 1.0), 1 + i % 4, 10,
                                     20 + i, 0, 1))
    plays = pd.DataFrame(rows, columns=pl.PLAY_COLUMNS)
    return plays.astype({"kind": np.int8, "home": np.int8}), quality


def test_fit_recovers_quarterback_order():
    plays, quality = synthetic_plays()
    ev = pv.events_from(plays)
    cfg = pv.apm_config(lam=50.0, min_plays=100)
    tab = pv.fit_values(ev, before=int(plays.t.max()) + 1, season=2021, cfg=cfg)
    v = dict(zip(tab.player_id, tab.value_play))
    regulars = [q for q in quality if q != 106]
    order = sorted(regulars, key=lambda q: -v[q])
    assert order[0] == 100 and order[-1] == 104
    assert np.corrcoef([v[q] for q in regulars], [quality[q] for q in regulars])[0, 1] > 0.9


def test_fit_is_leak_free():
    plays, _ = synthetic_plays()
    cut = int(plays[plays.season == 2021].t.min()) + 21
    cfg = pv.apm_config(lam=50.0, min_plays=100)
    a = pv.fit_values(pv.events_from(plays), cut, 2021, cfg)
    later = plays.t >= cut
    changed = plays.assign(epa=np.where(later, plays.epa + 5.0, plays.epa))
    b = pv.fit_values(pv.events_from(changed), cut, 2021, cfg)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_controls_with_and_without_defense():
    plays, _ = synthetic_plays()
    ev = pv.events_from(plays)
    fr = ev.frame
    X0 = pv.controls(0)(fr)
    X1 = pv.controls(2.0)(fr)
    n_def = fr.dkey.nunique()
    assert X1.shape[1] == X0.shape[1] + n_def
    assert np.allclose(X1[:, X0.shape[1]:].sum(1), 2.0)          # one defense column per play
    assert np.allclose(X0.sum(1), 8 * pv.CTRL_SCALE)              # one column per control family


def test_holdout_score_runs():
    plays, _ = synthetic_plays()
    r = pv.holdout_score(pv.events_from(plays), 2021, pv.apm_config(lam=50.0, min_plays=100))
    assert r["mse_gain"] > 0 and r["corr"] > 0


def _schedule_games(plays):
    g = plays.groupby(["game_id", "off"]).agg(season=("season", "first"), t=("t", "first")).reset_index()
    return g.rename(columns={"off": "team_id"}).assign(team=lambda d: "T" + d.team_id.astype(str))[
        ["game_id", "season", "t", "team_id", "team"]].sort_values(["t", "game_id", "team_id"]).reset_index(drop=True)


def test_project_starters_uses_only_earlier_games():
    plays, _ = synthetic_plays()
    tg = pl.team_games(plays)
    games = _schedule_games(plays)
    pr = pv.project_starters(tg, games)
    m = pr.merge(games, on=["game_id", "team_id"])
    # team 1's first game with the new starter (2021 week 6) is projected with the old one
    t6 = pl.day_number("2021-09-07") + 42
    assert m[(m.team_id == 1) & (m.t == t6)].qb.iloc[0] == 100
    assert m[(m.team_id == 1) & (m.t == t6 + 7)].qb.iloc[0] == 106
    first = games.groupby("team_id").t.min()
    assert not m.merge(first.rename("t0"), on="team_id").pipe(lambda d: (d.t == d.t0)).any()
    lead = pv.project_starters(tg, games, rule="prev_leader")
    assert len(lead) == len(pr)
    with pytest.raises(ValueError):
        pv.project_starters(tg, games, rule="depth_chart")


def test_game_table_columns_and_change_term():
    plays, _ = synthetic_plays()
    tg = pl.team_games(plays)
    games = _schedule_games(plays)
    games = games[games.season >= 2020]
    cfg = pv.apm_config(lam=50.0, min_plays=100)
    gv = pv.game_table(plays, tg, games, jobs=1, cfg=cfg, rule="prev_starter")
    assert list(gv.pv.columns) == ["game_id", "team", *LINEUP.columns]
    assert LINEUP.columns == ["_pv_0", "_pv_1", "_pv_2"]
    st = gv.starters
    # with the previous starter projected, the change vs the previous game is 0 by construction
    assert np.allclose(st._pv_1, 0.0, atol=1e-9)
    # recompute _pv_2 for team 1 just after the switch: new starter vs the mean of the last RECENT
    t7 = pl.day_number("2021-09-07") + 49
    row = st[(st.team_id == 1) & (st.t == t7)].iloc[0]
    before = pv.week_start(t7)
    vals = dict(zip(gv.fits[before].player_id, gv.fits[before].value_play))
    mult = gv.usage[2021]
    hist = tg[(tg.team_id == 1) & (tg.t < t7)].sort_values("t").starter.to_numpy()[-pv.RECENT:]
    exp_cur = vals.get(106, 0.0) * mult
    assert row._pv_0 == pytest.approx(exp_cur)
    assert row._pv_2 == pytest.approx(exp_cur - np.mean([vals.get(q, 0.0) for q in hist]) * mult)
    assert row._pv_2 < 0                                  # 106 is the worst quarterback
    # every game's values come from a fit before its week
    assert set(st.t.map(pv.week_start)) <= set(gv.fits)
    assert all(int(f.plays.sum()) <= int((plays.t < w).sum()) for w, f in gv.fits.items())


def test_usage_uses_the_previous_season():
    plays, _ = synthetic_plays()
    tg = pl.team_games(plays)
    u = pv.starter_plays(plays, tg)
    assert u[2019] == u[2020] == pytest.approx(40.0)


def test_fbs_teams():
    sched = pd.DataFrame({"season": [2024, 2024], "home_id": [10, 30], "home_division": ["fbs", "fcs"],
                          "away_id": [20, 10], "away_division": ["fcs", "fbs"]})
    assert pl.fbs_teams(sched) == {(2024, 10)}


def test_schedule_team_games_pools_fcs():
    sched = pd.DataFrame({
        "game_id": [1, 2, 3], "season": [2024] * 3,
        "start_date": ["2024-09-01T00:00Z"] * 3,
        "home_id": [10, 30, 40], "home_team": ["Ohio State", "Youngstown State", "Montana"],
        "home_division": ["fbs", "fcs", "fcs"],
        "away_id": [20, 10, 50], "away_team": ["Texas", "Ohio State", "Idaho"],
        "away_division": ["fbs", "fbs", "fcs"],
    })
    g = pv.schedule_team_games(sched, pl.game_days(sched), 2020)
    assert len(g) == 4                                      # the FCS-only game is not a wager game
    assert set(g.team) == {"Ohio State", "Texas", "FCS"}
    assert g[g.team_id == 30].team.iloc[0] == "FCS"
