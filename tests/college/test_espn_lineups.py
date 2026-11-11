import numpy as np
import pandas as pd

from runitback.college import espn_lineups as E

HOME, AWAY = 10, 20
H = [101, 102, 103, 104, 105]
A = [201, 202, 203, 204, 205]


def _row(n, period, clock, kind, text, team=None, pid=None, scoring=False, shooting=False, value=0):
    return {"game_id": "1", "game_play_number": n, "period_number": period, "clock_minutes": clock // 60,
            "clock_seconds": clock % 60, "type_text": kind, "text": text, "team_id": team, "athlete_id_1": pid,
            "home_team_id": HOME, "away_team_id": AWAY, "scoring_play": scoring, "shooting_play": shooting,
            "score_value": value, "season": 2026}


def _game(old_ft=False):
    ft1 = "X made Free Throw." if old_ft else "X makes free throw 1 of 2"
    ft2 = "X made Free Throw." if old_ft else "X makes free throw 2 of 2"
    rows = [
        _row(1, 1, 1200, "Jumpball", "Start game"),
        _row(2, 1, 1180, "JumpShot", "makes three", HOME, 101, True, True, 3),            # home 3
        _row(3, 1, 1160, "JumpShot", "misses", AWAY, 201, False, True, 2),
        _row(4, 1, 1158, "Defensive Rebound", "reb", HOME, 102),
        _row(5, 1, 1150, "PersonalFoul", "Foul on 201", AWAY, 201),
        _row(6, 1, 1150, "MadeFreeThrow", ft1, HOME, 103, True, False, 1),                # home 1 of 2
        # a block of substitutions between the free throws: home 105 -> 106, away 205 -> 206
        _row(7, 1, 1150, "Substitution", "P105 subbing out for Home", HOME, 105),
        _row(8, 1, 1150, "Substitution", "P106 subbing in for Home", HOME, 106),
        _row(9, 1, 1150, "Substitution", "P206 subbing in for Away", AWAY, 206),
        _row(10, 1, 1150, "Substitution", "P205 subbing out for Away", AWAY, 205),
        _row(11, 1, 1150, "MadeFreeThrow", ft2, HOME, 103, True, False, 1),               # home 2 of 2
        _row(12, 1, 1100, "LayUpShot", "makes layup", AWAY, 206, True, True, 2),          # away 2
        _row(13, 1, 600, "Lost Ball Turnover", "turnover", HOME, 106),
        _row(14, 1, 0, "End Period", "End of 1st half"),
        _row(15, 2, 1200, "JumpShot", "makes", AWAY, 206, True, True, 2),                 # away 2
        _row(16, 2, 0, "End Period", "End of 2nd half"),
    ]
    # enough substitution events for the game to count as rebuildable
    pad = []
    for i in range(E.MIN_SUBS):
        for team, (out, inn) in ((HOME, (106, 105)), (AWAY, (206, 205))):
            pad.append(_row(0, 2, 1200, "Substitution", f"P{inn} subbing in for X", team, inn))
            pad.append(_row(0, 2, 1200, "Substitution", f"P{out} subbing out for X", team, out))
            out, inn = inn, out
    return pd.DataFrame(rows), pd.DataFrame(pad)


def _pg():
    rows = [{"game_id": "1", "season": 2026, "game_date": "2026-01-10", "team_id": t, "home": t == HOME,
             "player_id": p, "starter": p in H + A, "dnp": False, "minutes": 0.0}
            for t, ps in ((HOME, [*H, 106]), (AWAY, [*A, 206])) for p in ps]
    return pd.DataFrame(rows)


def test_game_on_floor_follows_substitutions():
    g, _ = _game()
    h, a = E.game_on_floor(g, {HOME: H, AWAY: A})
    assert list(h[1]) == H and list(a[1]) == A
    # inside the substitution block a side may not have five: marked -1
    assert (h[6] == -1).all()                       # after 105 went out, before 106 came in
    assert (a[8] == -1).all()                       # 206 in before 205 out: six
    assert list(h[11]) == [101, 102, 103, 104, 106] and list(a[11]) == [201, 202, 203, 204, 206]
    # the next period starts with the five that ended the previous one
    assert list(a[14]) == [201, 202, 203, 204, 206]


def test_late_logged_substitution_from_evidence():
    g, _ = _game()
    g = g[g.game_play_number != 8]                  # 106's sub-in is missing; his turnover puts him on
    h, _ = E.game_on_floor(g.reset_index(drop=True), {HOME: H, AWAY: A})
    i = int(np.flatnonzero(g.game_play_number.to_numpy() == 13)[0])
    assert list(h[i]) == [101, 102, 103, 104, 106]


def test_possessions_minutes_and_rebuildable():
    g, pad = _game()
    pbp = pd.concat([g, pad], ignore_index=True)
    assert E.rebuildable(pbp) == {"1"}
    assert E.rebuildable(g) == set()
    of = E.on_floor(g.assign(game_play_number=range(len(g))), _pg(), min_subs=0)
    poss, _ = E.season_possessions(of, _pg())
    # home 3 / away miss / home 2 FTs / away 2 / home TO / away 2 (period 2)
    assert poss.points.tolist() == [3, 0, 2, 2, 0, 2]
    assert poss.off_home.tolist() == [True, False, True, False, True, False]
    # the free-throw trip counts for the five on the floor at its first free throw
    assert sorted(poss.iloc[2][[f"off_{i}" for i in range(1, 6)]]) == H
    fm = E.floor_minutes(of).set_index("player_id").floor_minutes
    assert np.isclose(fm[105], 50 / 60) and np.isclose(fm[106], (1150 + 1200) / 60)
    assert np.isclose(fm[101], 40.0)


def test_old_free_throw_format():
    g, _ = _game(old_ft=True)
    cl = E.classify(g)
    ft = g.type_text.eq("MadeFreeThrow").to_numpy()
    # neither is technical; the first of the trip is not the last, the second is
    assert cl.pts[ft].tolist() == [1, 1]
    assert cl.last_ft[ft].tolist() == [False, True]
    of = E.on_floor(g.assign(game_play_number=range(len(g))), _pg(), min_subs=0)
    poss, _ = E.season_possessions(of, _pg())
    assert poss.points.tolist() == [3, 0, 2, 2, 0, 2]
