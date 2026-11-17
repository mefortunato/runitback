import pandas as pd

from runitback.college import cbbd_lineups as C


def _floor(home, away):
    return [{"id": i, "name": n, "team": "Home U"} for i, n in home] + \
        [{"id": i, "name": n, "team": "Away St"} for i, n in away]


HOME = [(1, "Al One"), (2, "Bo Two Jr."), (3, "Cy Three"), (4, "Di Four"), (5, "Ed Five")]
AWAY = [(11, "Fa Six"), (12, "Gu Seven"), (13, "Ha Eight"), (14, "Io Nine"), (15, "Jo Ten")]


def _play(pid, kind, text, home, scoring=False, shooting=False, value=0, sec=1000, period=1):
    return {"gameId": 9, "gameSourceId": "401", "id": pid, "period": period, "secondsRemaining": sec,
            "playType": kind, "playText": text, "isHomeTeam": home, "team": "Home U" if home else "Away St",
            "opponent": "Away St" if home else "Home U", "scoringPlay": scoring, "shootingPlay": shooting,
            "scoreValue": value, "onFloor": _floor(HOME, AWAY)}


def _game():
    return [
        _play(1, "JumpShot", "Al One makes three", True, True, True, 3, 1190),
        _play(2, "JumpShot", "Fa Six misses jumper", False, False, True, 2, 1170),
        _play(3, "Defensive Rebound", "Bo Two rebound", True, sec=1168),
        _play(4, "Lost Ball Turnover", "Bo Two turnover", True, sec=1150),
        _play(5, "LayUpShot", "Gu Seven makes layup", False, True, True, 2, 1140),
        _play(6, "MadeFreeThrow", "Gu Seven makes free throw 1 of 1", False, True, True, 1, 1140),
        _play(7, "JumpShot", "Cy Three misses", True, False, True, 2, 1120),
        _play(8, "Offensive Rebound", "Cy Three off reb", True, sec=1118),
        _play(9, "MadeFreeThrow", "Cy Three makes free throw 1 of 2", True, True, True, 1, 1110),
        _play(10, "MadeFreeThrow", "Cy Three misses free throw 2 of 2", True, False, True, 1, 1110),
        _play(11, "Defensive Rebound", "Ha Eight reb", False, sec=1105),
    ]


def _pg(swap=False):
    rows = []
    for (pid, name), home in [*((p, True) for p in HOME), *((p, False) for p in AWAY)]:
        side_home = (not home) if swap else home
        rows.append({"game_id": "401", "season": 2025, "game_date": "2025-01-04", "home": side_home,
                     "player_id": 5000 + pid, "name": name.replace(" Jr.", ""), "team_id": 1 if home else 2})
    return pd.DataFrame(rows)


def test_compact_and_possessions():
    plays = C.compact_plays(_game())
    assert list(plays.columns) == C.PLAY_COLUMNS
    assert plays.home_ids.iloc[0] == [1, 2, 3, 4, 5]
    poss, st = C.season_possessions(plays, _pg())
    assert st["unmatched"] == 0 and st["swapped_sides"] == 0
    # home 3 / away miss / home TO / away 2+1 / home miss, OREB, 1 of 2 FT, DREB by away
    assert poss.points.tolist() == [3, 0, 0, 3, 1, 0]
    assert poss.off_home.tolist() == [True, False, True, False, True, False]
    assert poss.off_1.iloc[0] == 5001 and poss.def_1.iloc[0] == 5011


def test_swapped_home_away_matches_by_roster():
    plays = C.compact_plays(_game())
    poss, st = C.season_possessions(plays, _pg(swap=True))
    assert st["swapped_sides"] == 2 and st["unmatched"] == 0
    # CBBD's home team is ESPN's away team: off_home follows ESPN
    assert poss.off_home.tolist() == [False, True, False, True, False, True]
    assert poss.off_1.iloc[0] == 5001


def test_name_matching():
    names = {C.norm_name(n): i for i, n in enumerate(["Michael Wilson Jr.", "D'Aundre Samuels", "José Pérez"])}
    assert C._match(names, "Michael Wilson") == 0
    assert C._match(names, "DAundre Samuels") == 1
    assert C._match(names, "Jose Perez") == 2
    assert C._match(names, "Nobody Here") is None


def test_fetch_days_caches_and_keeps_reserve(tmp_path, monkeypatch):
    calls = []
    left = iter([30, 29, 28, 27, 26, 25, 24])

    def fake_get(path, key, **params):
        calls.append(params["date"])
        return _game(), next(left)

    monkeypatch.setattr(C, "_get", fake_get)
    days = [f"2025-01-0{i}" for i in range(1, 10)]
    res = C.fetch_days(days, tmp_path, "k", reserve=27)
    assert calls == days[:4]              # stops once 27 calls are left
    assert res["fetched"] == 4 and len(res["missing"]) == 5
    res = C.fetch_days(days, tmp_path, "k", reserve=0, max_calls=2)
    assert calls[4:] == days[4:6]         # cached days are never fetched again
    assert len(C.load_days(tmp_path)) == 6 * len(_game())


def test_season_days_add_the_next_utc_day():
    pg = pd.DataFrame({"season": [2025, 2025, 2025, 2024], "game_date": ["2025-01-04", "2025-01-05", "2025-01-09",
                                                                         "2024-01-01"]})
    assert C.season_days(pg, 2025) == ["2025-01-04", "2025-01-05", "2025-01-06", "2025-01-09", "2025-01-10"]


def test_get_retries_after_429(monkeypatch):
    import requests

    class R:
        def __init__(self, code):
            self.status_code, self.headers = code, {"retry-after": "0", "x-calllimit-remaining": "41"}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code))

        def json(self):
            return []

    codes = iter([429, 429, 200])
    monkeypatch.setattr(requests, "get", lambda *a, **k: R(next(codes)))
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    assert C._get("plays/date", "k", date="2025-01-01") == ([], 41)
