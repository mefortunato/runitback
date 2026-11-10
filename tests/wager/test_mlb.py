import datetime

import numpy as np
import pandas as pd
import pytest

from runitback.wager import espn
from runitback.wager.core.process import process
from runitback.wager.core.schema import mirror, validate
from runitback.wager.mlb.data import download as mlb
from runitback.wager.mlb.data.features import SCHEMA

TODAY = datetime.datetime(2025, 5, 3, 12, tzinfo=datetime.timezone.utc)

TEAMS = [
    {"id": 147, "abbreviation": "NYY"},
    {"id": 111, "abbreviation": "BOS"},
    {"id": 114, "abbreviation": "CLE"},
]


def game(pk, date, home, away, scores=(None, None), state="Final", number=1,
         pitchers=(None, None), game_type="R", start="23:05"):  # fmt: skip
    sides = {}
    for side, team, score, pitcher in zip(("home", "away"), (home, away), scores,
                                          pitchers):  # fmt: skip
        sides[side] = {"team": {"id": team, "name": f"team {team}"}}
        if score is not None:
            sides[side]["score"] = score
        if pitcher:
            sides[side]["probablePitcher"] = {"id": pitcher, "fullName": f"P{pitcher}"}
    final = state in ("Final", "Postponed", "Cancelled", "Completed Early")
    return {"gamePk": pk, "gameType": game_type, "season": date[:4],
            "gameDate": f"{date}T{start}:00Z", "officialDate": date,
            "status": {"abstractGameState": "Final" if final else "Preview",
                       "detailedState": state},
            "teams": sides, "doubleHeader": "S" if number > 1 else "N",
            "gameNumber": number}  # fmt: skip


SCHEDULE = {
    "dates": [
        {"games": [
            game(1, "2024-09-28", 147, 111, (5, 3), pitchers=(10, 20)),
            # the Indians are the Guardians now: same team id
            game(2, "2025-04-01", 114, 147, (2, 1), pitchers=(30, 10)),
            # a doubleheader, then a postponement made up later (same id)
            game(3, "2025-04-02", 147, 111, (4, 6), pitchers=(11, 20),
                 start="17:05"),
            game(4, "2025-04-02", 147, 111, (7, 0), number=2, pitchers=(12, 21)),
            game(5, "2025-04-03", 111, 114, state="Postponed"),
            game(6, "2025-04-20", 111, 147, state="Cancelled"),
        ]},
        {"games": [
            game(5, "2025-04-25", 111, 114, (3, 2), pitchers=(20, 30)),
            game(7, "2025-05-04", 114, 147, state="Scheduled", pitchers=(31, 10)),
            game(8, "2025-04-30", 114, 111, state="Scheduled"),  # never played
        ]},
    ]
}  # fmt: skip


def log(team, splits):
    return {
        "stats": [
            {"group": {"displayName": group},
             "splits": [{"game": {"gamePk": pk}, "team": {"id": team},
                         "stat": stat} for pk, stat in splits.get(group, [])]}
            for group in ("hitting", "pitching", "fielding")
        ]
    }  # fmt: skip


LOGS = [
    log(147, {"hitting": [(3, {"runs": 4, "hits": 9, "homeRuns": 2})],
              "fielding": [(3, {"errors": 1})]}),
    log(111, {"hitting": [(3, {"runs": 6, "hits": 11, "homeRuns": 0})],
              "pitching": [(3, {"numberOfPitches": 150})]}),
    log(114, {"hitting": [(7, {"runs": 99})]}),  # a stray future log
]  # fmt: skip


def start(season, date, pk, outs, er, k, bb, bf):
    return {"season": str(season), "date": date, "game": {"gamePk": pk},
            "stat": {"gamesStarted": 1, "outs": outs, "earnedRuns": er,
                     "strikeOuts": k, "baseOnBalls": bb,
                     "battersFaced": bf}}  # fmt: skip


PEOPLE = [
    {"id": 10, "stats": [{"splits": [
        start(2024, "2024-09-28", 1, 18, 2, 7, 1, 24),
        start(2025, "2025-04-01", 2, 21, 0, 9, 0, 25),
    ]}]},
    {"id": 20, "stats": [{"splits": [
        start(2025, "2025-04-02", 3, 15, 3, 4, 2, 22),
        {"season": "2025", "date": "2025-04-10", "game": {"gamePk": 50},
         "stat": {"gamesStarted": 0, "outs": 3, "earnedRuns": 5}},  # relief
    ]}]},
]  # fmt: skip

LINES = {"3": {"moneyline": -140.0, "opp_moneyline": 120.0, "spread_line": 1.5,
               "spread_odds": 130.0, "opp_spread_odds": -150.0,
               "total_line": 8.5}}  # fmt: skip


@pytest.fixture
def games():
    return mlb.build_games([SCHEDULE], LOGS, PEOPLE, TEAMS, LINES, today=TODAY)


def row(games, game_id, team):
    rows = games[(games.game_id == str(game_id)) & (games.team == team)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_schema_columns_and_process(games):
    validate(games, SCHEMA)
    processed = process("mlb", games)
    assert set(processed.game_id) == {"1", "2", "3", "4", "5", "7"}


def test_postponed_games_are_played_later_cancelled_dropped(games):
    makeup = row(games, 5, "BOS")
    assert (makeup.game_date, makeup.result) == ("2025-04-25", 1)
    assert "6" not in set(games.game_id) and "8" not in set(games.game_id)
    assert pd.isna(row(games, 7, "CLE").result)


def test_team_logs_join_both_sides(games):
    yankees = row(games, 3, "NYY")
    assert (yankees.runs, yankees.hits, yankees.home_runs, yankees.errors) == (
        4,
        9,
        2,
        1,
    )
    assert (yankees.opp_hits, yankees.opp_pitches, yankees.result) == (11, 150, -2)
    assert np.isnan(row(games, 7, "CLE").hits)
    assert row(games, 4, "NYY").rest == 0  # second game of a doubleheader


def test_starters_come_from_earlier_starts_only(games):
    # P10's first 2025 start: last season's 6 innings, 2 earned runs
    first = row(games, 2, "NYY")
    assert first.sp_starts == 1
    expected = (9 * 2 + mlb.PRIOR_ERA * mlb.PRIOR_INNINGS) / (6 + mlb.PRIOR_INNINGS)
    assert first.sp_era == pytest.approx(expected)
    # upcoming: both 2024's and the 2025 start before it
    upcoming = row(games, 7, "NYY")
    assert upcoming.sp_starts == 2 and upcoming.opp_sp_name == "P31"
    era = (9 * 2 + mlb.PRIOR_ERA * mlb.PRIOR_INNINGS) / (13 + mlb.PRIOR_INNINGS)
    assert upcoming.sp_era == pytest.approx(era)
    # a debut gets the priors; relief appearances are not starts
    debut = row(games, 3, "NYY")
    assert debut.sp_starts == 0 and debut.sp_era == pytest.approx(mlb.PRIOR_ERA)
    later = row(games, 5, "BOS")
    assert later.sp_starts == 1
    assert row(games, 3, "BOS").opp_sp_starts == 0


def test_market_from_both_perspectives(games):
    yankees, red_sox = row(games, 3, "NYY"), row(games, 3, "BOS")
    assert (yankees.moneyline, red_sox.moneyline) == (-140, 120)
    assert (yankees.run_line, red_sox.run_line) == (1.5, -1.5)
    assert (red_sox.run_line_odds, red_sox.total_line) == (-150, 8.5)
    back = mirror(games[games.index == yankees.name], SCHEMA).iloc[0]
    assert (back.team, back.sp_id, back.opp_sp_id) == ("BOS", 20, 11)


def test_doubleheaders_match_espn_games_in_order():
    games = mlb.schedule_frame(SCHEDULE, mlb.team_keys(TEAMS))
    events = espn.events_frame(
        [
            {"id": eid, "date": date,
             "competitions": [{"competitors": [
                 {"homeAway": "home", "team": {"abbreviation": "NYY"}},
                 {"homeAway": "away", "team": {"abbreviation": "BOS"}}]}]}
            for eid, date in (("e2", "2025-04-02T23:05Z"), ("e1", "2025-04-02T17:05Z"))
        ]
    )  # fmt: skip
    games = games.assign(game_date=games["game_date"])
    matched = espn.match_events(games, events, mlb.espn_keys(events))
    ids = dict(zip(games.game_id, matched))
    assert (ids["3"], ids["4"]) == ("e1", "e2")


def test_innings_and_seasons():
    assert mlb.innings("5.2") == pytest.approx(5 + 2 / 3)
    assert mlb.season_of(datetime.date(2025, 10, 30)) == 2025
    assert not mlb.season_complete(2025, TODAY) and mlb.season_complete(2024, TODAY)
