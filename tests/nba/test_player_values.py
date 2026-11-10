import datetime

import numpy as np
import pandas as pd
import pytest

from runitback.core.player_values import fit_apm
from runitback.nba.data.players import game_on_court
from runitback.nba.player_values import (
    DREB, FT, OREB, OTHER, SHOT, START_DREB, START_MADE, START_PERIOD, START_TO, TO, apm_config,
    classify, day_number, events_from, fit_values, franchise_games, franchise_only, game_possessions,
    game_table, previous_rotation, rotations, season_possessions, week_start,
)
from runitback.wager.nba.lineup import LINEUP

HOME_T, AWAY_T = 1, 2


def rows(events):
    """events: (period, clock, type_text, team, athlete, made, value, shooting)."""
    out = []
    for n, (q, clock, tt, team, a1, made, value, shooting) in enumerate(events):
        out.append(dict(game_id=1, id=n, game_play_number=n + 1, sequence_number=str(n + 1), season=2025,
                        season_type=2, game_date="2025-01-01", period_number=q, start_quarter_seconds_remaining=clock,
                        type_text=tt, text="makes" if made else "misses", team_id=team, athlete_id_1=a1,
                        athlete_id_2=None, athlete_id_3=None, home_team_id=HOME_T, away_team_id=AWAY_T,
                        scoring_play=made, shooting_play=shooting, score_value=value if made else 0))
    return pd.DataFrame(rows_scores(out))


def rows_scores(out):
    h = a = 0
    for r in out:
        pts = r["score_value"] if r["scoring_play"] else 0
        if r["team_id"] == HOME_T:
            h += pts
        else:
            a += pts
        r["home_score"], r["away_score"] = h, a
    return out


GAME = [
    (1, 720, "Jump Ball", HOME_T, 105, False, 0, False),
    (1, 700, "Jump Shot", HOME_T, 101, True, 2, True),                    # home 2
    (1, 680, "Jump Shot", AWAY_T, 201, False, 0, True),
    (1, 678, "Offensive Rebound", AWAY_T, 202, False, 0, False),
    (1, 670, "Layup Shot", AWAY_T, 202, True, 2, True),                   # away 2 + and-one
    (1, 670, "Substitution", HOME_T, 106, False, 0, False),
    (1, 670, "Free Throw - 1 of 1", AWAY_T, 202, True, 1, True),
    (1, 650, "Lost Ball Turnover", HOME_T, 102, False, 0, False),          # home 0
    (1, 640, "Jump Shot", AWAY_T, 203, False, 0, True),                   # away 0
    (1, 638, "Defensive Rebound", HOME_T, 103, False, 0, False),
    (1, 630, "Shooting Foul", AWAY_T, 204, False, 0, False),
    (1, 630, "Free Throw - 1 of 2", HOME_T, 103, True, 1, True),
    (1, 630, "Free Throw - 2 of 2", HOME_T, 103, True, 1, True),          # home 2
    (1, 620, "Free Throw - Technical", HOME_T, 104, True, 1, True),       # not a possession's points
    (1, 610, "Jump Shot", AWAY_T, 205, True, 3, True),                    # away 3
    (1, 0, "End Period", None, None, False, 0, False),
    (2, 720, "Jump Shot", AWAY_T, 201, False, 0, True),                   # Q2: away 0, a new start
]


def test_classify_and_possessions():
    df = rows(GAME)
    cl = classify(df)
    assert list(cl.kind[:5]) == [OTHER, SHOT, SHOT, OREB, SHOT]
    assert cl.kind[7] == TO and cl.kind[9] == DREB and cl.kind[11] == FT and cl.kind[13] == OTHER
    assert cl.pts[13] == 0 and cl.last_ft[12] and not cl.last_ft[11]
    poss = game_possessions(df.period_number.to_numpy(), *(cl[c].to_numpy() for c in
                                                            ("side", "kind", "made", "last_ft", "pts")))
    got = [(s, st, p) for _, s, st, p in poss]
    assert got == [(0, START_PERIOD, 2), (1, START_MADE, 3), (0, START_MADE, 0), (1, START_TO, 0),
                   (0, START_DREB, 2), (1, START_MADE, 3), (1, START_PERIOD, 0)]


def box():
    out = []
    for team, base, ha in ((HOME_T, 100, "home"), (AWAY_T, 200, "away")):
        for k in range(1, 8):
            out.append(dict(game_id=1, season=2025, season_type=2, game_date="2025-01-01", athlete_id=base + k,
                            athlete_display_name=f"P{base + k}", athlete_short_name="", team_id=team,
                            team_abbreviation="BOS", opponent_team_abbreviation="NYK", home_away=ha,
                            minutes=30.0 if k <= 5 else 9.0, plus_minus="+0", points=0.0, starter=k <= 5,
                            did_not_play=False, reason=None, ejected=False, athlete_position_abbreviation="G"))
    return pd.DataFrame(out)


def test_season_possessions_credit_the_fouled_five():
    df = rows(GAME)
    oc, _ = game_on_court(df, box())
    poss = season_possessions(df, oc)
    assert len(poss) == 7 and poss.points.sum() == 2 + 3 + 2 + 3
    assert poss.t.iloc[0] == day_number("2025-01-01")
    # the and-one possession: 106 came in for 101 at the clock of the free throw, after the foul
    p = poss.iloc[2]   # home's turnover possession, after the substitution
    assert set(p[[f"off_{i}" for i in range(1, 6)]]) == {102, 103, 104, 105, 106}
    p = poss.iloc[1]   # away's and-one: home's defenders are the five before the substitution
    assert set(p[[f"def_{i}" for i in range(1, 6)]]) == {101, 102, 103, 104, 105}
    assert set(p[[f"off_{i}" for i in range(1, 6)]]) == {201, 202, 203, 204, 205}


def test_week_start_is_monday():
    for d in ("2025-01-06", "2025-01-08", "2025-01-12"):
        m = week_start(day_number(d))
        assert datetime.date(1970, 1, 1) + datetime.timedelta(days=m) == datetime.date(2025, 1, 6)


# -- synthetic league -------------------------------------------------------------------------------

def synthetic(seed=0, teams=4, seasons=(2022, 2023), games_per_team=24, poss_per_game=100):
    """Possessions with planted per-player offense and defense effects, plus player_games and a
    player directory. Each team has 9 players; the first 7 play most."""
    rng = np.random.default_rng(seed)
    roster = {t: [1000 * (t + 1) + k for k in range(9)] for t in range(teams)}
    groups = ["G", "G", "F", "F", "C", "G", "F", "C", "G"]
    off_eff = {p: rng.normal(0, 0.05) for r in roster.values() for p in r}
    def_eff = {p: rng.normal(0, 0.05) for r in roster.values() for p in r}
    weights = np.array([5, 5, 5, 5, 5, 3, 3, 2, 0.3])
    weights = weights / weights.sum()
    poss, pg = [], []
    for s in seasons:
        day = datetime.date(s - 1, 10, 20)
        for gn in range(games_per_team * teams // 2):
            day += datetime.timedelta(days=1)
            order = rng.permutation(teams)
            h, a = int(order[0]), int(order[1])
            gid = f"{s}{gn:04d}"
            mins = {}
            for k in range(poss_per_game):
                o, d = (h, a) if k % 2 == 0 else (a, h)
                op = rng.choice(roster[o], 5, replace=False, p=weights)
                dp = rng.choice(roster[d], 5, replace=False, p=weights)
                for p in [*op, *dp]:
                    mins[p] = mins.get(p, 0) + 0.24
                mu = 1.1 + 0.03 * (o == h) + sum(off_eff[p] for p in op) - sum(def_eff[p] for p in dp)
                pts = int(np.clip(rng.poisson(max(mu, 0.05)), 0, 4))
                poss.append((gid, s, day.isoformat(), 1, o == h, 1, pts, *op, *dp))
            for t in (h, a):
                for p in roster[t]:
                    pg.append(dict(game_id=gid, season=s, game_date=day.isoformat(), team=f"T{t}", player_id=p,
                                   minutes=mins.get(p, 0.0), dnp=p not in mins))
    cols = ["game_id", "season", "game_date", "period", "off_home", "start", "points",
            *[f"off_{i}" for i in range(1, 6)], *[f"def_{i}" for i in range(1, 6)]]
    poss = pd.DataFrame(poss, columns=cols)
    poss.insert(3, "t", poss.game_date.map(day_number))
    players = pd.DataFrame({"player_id": [p for r in roster.values() for p in r],
                            "group": [groups[k] for _ in roster for k in range(9)],
                            "name": [f"P{p}" for r in roster.values() for p in r]})
    players.loc[players.player_id == 1008, "group"] = None   # no ESPN position: the default group
    return poss, pd.DataFrame(pg), players, off_eff, def_eff


@pytest.fixture(scope="module")
def league():
    return synthetic()


def test_fit_recovers_offense_and_defense(league):
    poss, _, players, off_eff, def_eff = league
    ev = events_from(poss, dict(zip(players.player_id, players.group)))
    tab = fit_values(ev, before=int(poss.t.max()) + 1, season=2023, cfg=apm_config(lam=300, min_poss=200))
    assert set(tab.side) == {"off", "def"} and set(tab.group) <= {"G", "F", "C"}
    for side, eff in (("off", off_eff), ("def", def_eff)):
        t = tab[tab.side == side]
        assert np.corrcoef(t.value_poss, t.player_id.map(eff))[0, 1] > 0.6   # defense: higher is better
    # the rarely used ninth men share the replacement level
    assert not tab.player_id.isin([1008, 2008, 3008, 4008]).any()


def test_fits_are_leak_free(league):
    poss, _, players, _, _ = league
    group_of = dict(zip(players.player_id, players.group))
    before = week_start(day_number("2023-01-02"))
    cfg = apm_config(lam=300, min_poss=200)
    a = fit_apm(events_from(poss, group_of), cfg, before, 2023)
    later = poss.t >= before
    poss2 = poss.assign(points=np.where(later, 4, poss.points))
    b = fit_apm(events_from(poss2, group_of), cfg, before, 2023)
    pd.testing.assert_frame_equal(a, b)


def test_previous_rotation(league):
    _, pg, _, _, _ = league
    pr = previous_rotation(pg)
    rot = rotations(pg)
    games = pg[pg.team == "T0"].drop_duplicates("game_id").sort_values("game_date").game_id.tolist()
    first = pr[(pr.game_id == games[0]) & (pr.team == "T0")]
    assert first.empty   # no earlier game
    got = pr[(pr.game_id == games[3]) & (pr.team == "T0")].sort_values("rank")
    want = rot[(rot.game_id == games[2]) & (rot.team == "T0")].sort_values("rank")
    assert got.player_id.tolist() == want.player_id.tolist()


def test_game_table_layout_and_leakage(league):
    poss, pg, players, _, _ = league
    cfg = apm_config(lam=300, min_poss=200)
    gv = game_table(poss, pg, players, first_season=2023, jobs=1, cfg=cfg)
    assert list(gv.pv.columns) == ["game_id", "team", *LINEUP.columns] and len(LINEUP.columns) == 14
    assert gv.pv.game_id.str.startswith("2023").all() and len(gv.pv) > 0
    assert not gv.pv.duplicated(["game_id", "team"]).any()
    # previous game's rotation as the projection: no change vs the previous game
    assert (gv.pv[["_pv_6", "_pv_7"]].abs() < 1e-5).all().all()
    # each game uses the fit on the possessions before its Monday
    g = pg[pg.season == 2023].sort_values("game_date").iloc[200]
    w = week_start(day_number(g.game_date))
    assert w in gv.fits
    want = fit_apm(events_from(poss, dict(zip(players.player_id, players.group))), cfg, w, 2023)
    np.testing.assert_allclose(gv.fits[w].value_poss.to_numpy(), want.value.to_numpy())
    # slot usage: starters play more possessions than the bench, and 2023 uses 2022's
    assert gv.slots[2023] == gv.slots[2022] and gv.slots[2023][0] > gv.slots[2023][7]


def test_franchise_games_drop_all_star_games():
    pg = pd.DataFrame({"game_id": ["1", "1", "2", "2"], "team": ["BOS", "NYK", "EAST", "WEST"],
                       "opponent": ["NYK", "BOS", "WEST", "EAST"]})
    poss = pd.DataFrame({"game_id": ["1", "2", "2"]})
    assert franchise_games(pg) == {"1"}
    p, g = franchise_only(poss, pg)
    assert p.game_id.tolist() == ["1"] and set(g.team) == {"BOS", "NYK"}
