import numpy as np
import pandas as pd
import pytest

from runitback.college import box_values as B
from runitback.college import data, lineups
from runitback.college import player_values as P
from runitback.wager.cbb.lineup import LINEUP
from tests.college.synthetic import league


@pytest.fixture(scope="module")
def tables():
    pg = data.build_player_games(league())
    return pg, data.team_games(pg)


def _first_monday(pg, season):
    t = pd.to_datetime(pg[pg.season == season].game_date).sub(pd.Timestamp("1970-01-01")).dt.days.min()
    return int(B.week_start(t))


def test_box_values_leak_free(tables):
    pg, tg = tables
    m = B.BoxModel.build(pg, tg)
    cut = _first_monday(pg, 2023) + 14
    v = m.values(cut, 2023)
    # rewrite every game on or after the cut-off: the values at the cut-off can't move
    pg2, tg2 = pg.copy(), tg.copy()
    later = pd.to_datetime(pg2.game_date).sub(pd.Timestamp("1970-01-01")).dt.days >= cut
    for c in data.STATS:
        pg2.loc[later & ~pg2.dnp, c] = pg2.loc[later & ~pg2.dnp, c] * 3 + 1
    tl = pd.to_datetime(tg2.game_date).sub(pd.Timestamp("1970-01-01")).dt.days >= cut
    tg2.loc[tl, "points"] = 200
    v2 = B.BoxModel.build(pg2, tg2).values(cut, 2023)
    pd.testing.assert_frame_equal(v.reset_index(drop=True), v2.reset_index(drop=True))


def test_box_values_rank_skill(tables):
    pg, tg = tables
    m = B.BoxModel.build(pg, tg)
    v = m.values(_first_monday(pg, 2023), 2023)
    tot = v.groupby("player_id").value.sum()
    # the synthetic star (skill 3) is the best player; values are per possession and centred near 0
    assert tot.idxmax() == 1000
    assert tot.abs().max() < 0.5
    # a player with no history is at replacement
    feat = m.features(_first_monday(pg, 2023), 2023, players=np.array([999999]))
    assert feat.empty


def test_apm_prior_moves_toward_truth():
    rng = np.random.default_rng(1)
    n, players = 60000, 40
    off = np.stack([rng.choice(20, 5, replace=False) for _ in range(n)])
    dfn = np.stack([20 + rng.choice(20, 5, replace=False) for _ in range(n)])
    true = np.zeros(players)
    true[3] = 0.15                                  # player 3 is much better on offense than his prior
    pts = 1.0 + true[off].sum(1) + rng.normal(0, 0.3, n)
    poss = pd.DataFrame({"game_id": np.repeat(np.arange(n // 100), 100).astype(str), "season": 2025,
                         "game_date": "2025-01-01", "t": np.repeat(np.arange(n // 100), 100), "period": 1,
                         "off_home": rng.random(n) < 0.5, "start": 0, "points": pts})
    for i in range(5):
        poss[f"off_{i + 1}"] = off[:, i] + 1
        poss[f"def_{i + 1}"] = dfn[:, i] + 1
    ev = P.events_from(poss, {})
    prior = pd.concat([pd.DataFrame({"side": s, "player_id": np.arange(1, players + 1), "group": "F",
                                     "minutes": 500.0, "value": 0.0}) for s in ("off", "def")], ignore_index=True)
    cfg = P.apm_config(lam=500.0, min_minutes=10)
    out = P.fit_apm_prior(ev, prior, before=10 ** 6, season=2025, cfg=cfg)
    o = out[out.side == "off"].set_index("player_id").value
    assert o[4] > 0.08 and o.drop(4).abs().max() < 0.05
    # the prior itself: a player whose prior already is the truth stays there
    prior.loc[(prior.side == "off") & (prior.player_id == 4), "value"] = 0.15
    out2 = P.fit_apm_prior(ev, prior, before=10 ** 6, season=2025, cfg=cfg)
    assert abs(out2[(out2.side == "off") & (out2.player_id == 4)].value.iloc[0] - 0.15) < 0.03


def test_game_table_layout_and_leak_free(tables):
    pg, tg = tables
    proj = lineups.projected_lineups(pg)
    gv = P.build(pg, tg, proj, first_season=2022)
    assert list(gv.pv.columns[4:]) == list(LINEUP.columns)
    assert len(LINEUP.columns) == 14
    assert gv.pv.groupby(["game_id", "team_id"]).size().max() == 1
    assert gv.pv[list(LINEUP.columns)].notna().all().all()
    # the star's minutes add to his team's group columns: his share of the level is his value * minutes
    star = gv.pv[gv.pv.team_id == 101].iloc[-1]
    assert star[[f"_pv_{i}" for i in range(6)]].abs().sum() > 0
    # rows before a cut-off don't change when later games change
    cut = "2022-12-01"
    pg2 = pg.copy()
    later = pg2.game_date >= cut
    pg2.loc[later & ~pg2.dnp, "pts"] += 10
    tg2 = data.team_games(pg2)
    tg2.loc[tg2.game_date >= cut, "points"] += 10
    gv2 = P.build(pg2, tg2, lineups.projected_lineups(pg2), first_season=2022)
    early = gv.pv.game_id.isin(set(pg[pg.game_date < cut].game_id))
    a = gv.pv[early].sort_values(["game_id", "team_id"]).reset_index(drop=True)
    b = gv2.pv[gv2.pv.game_id.isin(set(a.game_id))].sort_values(["game_id", "team_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_side_values_minutes_weighting():
    vals = {("off", 1): 0.1, ("def", 1): 0.05, ("off", 2): -0.02}
    out = P.side_values([(1, 40.0), (2, 20.0)], vals, {1: "G", 2: "C"}, ppg=70.0)
    assert np.isclose(out["G_OFF"], 0.1 * 70) and np.isclose(out["G_DEF"], 0.05 * 70)
    assert np.isclose(out["C_OFF"], -0.02 * 70 / 2) and out["F_OFF"] == 0
