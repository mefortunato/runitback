import datetime

import numpy as np
import pandas as pd
import pytest

from runitback.wager import espn
from runitback.wager.core.process import process
from runitback.wager.core.schema import mirror, validate
from runitback.wager.nhl.data import download as nhl
from runitback.wager.nhl.data.features import SCHEMA

TODAY = datetime.datetime(2025, 4, 2, 12, tzinfo=datetime.timezone.utc)

TEAMS = [
    {"id": 6, "triCode": "BOS", "fullName": "Boston Bruins"},
    {"id": 10, "triCode": "TOR", "fullName": "Toronto Maple Leafs"},
    {"id": 11, "triCode": "ATL", "fullName": "Atlanta Thrashers"},
    {"id": 52, "triCode": "WPG", "fullName": "Winnipeg Jets"},
    {"id": 53, "triCode": "ARI", "fullName": "Arizona Coyotes"},
    {"id": 59, "triCode": "UTA", "fullName": "Utah Hockey Club"},
]


def game(gid, start, home, away, scores=(0, 0), state=7, game_type=2, period=3):
    season = int(str(gid)[:4])
    return {"id": gid, "easternStartTime": start, "gameDate": start[:10],
            "gameStateId": state, "gameType": game_type, "homeScore": scores[0],
            "homeTeamId": home, "period": period, "season": f"{season}{season + 1}",
            "visitingScore": scores[1], "visitingTeamId": away}  # fmt: skip


SCHEDULE = [
    game(2010020001, "2010-10-08T19:00:00", 11, 6, (3, 2)),  # Thrashers
    game(2023020001, "2023-10-10T19:00:00", 53, 10, (4, 5), period=5),  # shootout
    game(2024020001, "2024-10-08T19:00:00", 59, 6, (2, 1)),
    game(2024020002, "2024-10-09T19:30:00", 10, 6, (4, 1)),
    game(2024020003, "2024-10-10T19:00:00", 6, 52, (3, 3), state=1),  # postponed
    game(2024010001, "2024-09-25T19:00:00", 6, 10, (1, 2), state=7, game_type=1),
    game(2024030111, "2025-04-03T19:00:00", 6, 10, state=1, game_type=3),
]


def report(gid, team, **stats):
    return {"gameId": gid, "teamId": team, **stats}


REPORTS = {
    "summary": [report(2024020001, t, shotsForPerGame=30.0, faceoffWinPct=0.55,
                       powerPlayPct=0.25, penaltyKillPct=0.8)
                for t in (59, 6)]
    + [report(2024030111, 6, shotsForPerGame=99.0)],  # a stray future report
    "realtime": [report(2024020001, t, hits=20, blockedShots=12, giveaways=8,
                        takeaways=6, missedShots=10, emptyNetGoals=0)
                 for t in (59, 6)],
    "powerplay": [report(2024020001, 59, ppOpportunities=4, powerPlayGoalsFor=1,
                         ppTimeOnIcePerGame=400.0, shGoalsAgainst=0)],
}  # fmt: skip

LINES = {"2024020002": {"moneyline": -150.0, "opp_moneyline": 130.0,
                        "spread_line": 1.5, "spread_odds": 160.0,
                        "opp_spread_odds": -190.0, "total_line": 6.0}}  # fmt: skip


@pytest.fixture
def games():
    return nhl.build_games(SCHEDULE, REPORTS, TEAMS, LINES, today=TODAY)


def row(games, game_id, team):
    rows = games[(games.game_id == str(game_id)) & (games.team == team)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_schema_columns_and_process(games):
    validate(games, SCHEMA)
    processed = process("nhl", games)
    assert set(processed.game_id) == {"2010020001", "2023020001", "2024020001",
                                      "2024020002", "2024030111"}  # fmt: skip


def test_relocated_franchises_keep_one_key(games):
    thrashers = row(games, 2010020001, "WPG")
    assert thrashers.team_name == "Atlanta Thrashers" and thrashers.result == 1
    coyotes = row(games, 2023020001, "UTA")
    assert coyotes.team_name == "Arizona Coyotes" and coyotes.periods == 5
    assert coyotes.result == -1 and coyotes.total == 9


def test_preseason_and_unplayed_games_are_dropped_upcoming_kept(games):
    assert "2024010001" not in set(games.game_id)
    assert "2024020003" not in set(games.game_id)
    playoff = row(games, 2024030111, "BOS")
    assert pd.isna(playoff.result) and playoff.game_type == "playoffs"
    assert np.isnan(playoff.shots)  # no stats leak into an upcoming game


def test_reports_join_both_sides(games):
    utah = row(games, 2024020001, "UTA")
    assert (utah.shots, utah.hits) == (30, 20)
    assert (utah.faceoff_pct, utah.pp_pct) == pytest.approx((55, 25))
    assert (utah.pp_opportunities, utah.opp_shots) == (4, 30)
    assert np.isnan(row(games, 2024020001, "BOS").pp_opportunities)
    assert row(games, 2024020002, "BOS").rest == 1


def test_moneylines_and_puck_line_from_both_perspectives(games):
    leafs, bruins = row(games, 2024020002, "TOR"), row(games, 2024020002, "BOS")
    assert (leafs.moneyline, leafs.opp_moneyline) == (-150, 130)
    assert (bruins.moneyline, bruins.opp_moneyline) == (130, -150)
    assert (leafs.puck_line, bruins.puck_line) == (1.5, -1.5)
    assert (bruins.puck_line_odds, bruins.total_line) == (-190, 6.0)
    back = mirror(games[games.index == leafs.name], SCHEMA).iloc[0]
    assert (back.team, back.moneyline, back.puck_line) == ("BOS", 130, -1.5)


def test_espn_games_match_on_date_and_teams():
    games = nhl.schedule_frame(SCHEDULE, nhl.team_codes(TEAMS))
    events = espn.events_frame(
        [
            {"id": "401", "date": "2024-10-09T23:30Z",
             "competitions": [{"competitors": [
                 {"homeAway": "home", "team": {"abbreviation": "TOR"}},
                 {"homeAway": "away", "team": {"abbreviation": "BOS"}}]}]},
            {"id": "402", "date": "2024-10-08T23:00Z",
             "competitions": [{"competitors": [
                 {"homeAway": "home", "team": {"abbreviation": "UTAH"}},
                 {"homeAway": "away", "team": {"abbreviation": "BOS"}}]}]},
        ]
    )  # fmt: skip
    matched = espn.match_events(games, events, nhl.espn_keys(events))
    ids = dict(zip(games.game_id, matched))
    assert ids["2024020002"] == "401" and ids["2024020001"] == "402"
    assert pd.isna(ids["2023020001"])


def test_seasons_and_months():
    assert nhl.season_of(datetime.date(2024, 10, 8)) == 2025
    assert nhl.season_id(2025) == "20242025"
    assert nhl.months(2025)[0] == "202409" and nhl.months(2025)[-1] == "202507"
    assert not nhl.season_complete(2025, TODAY) and nhl.season_complete(2024, TODAY)
