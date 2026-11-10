"""Source parsers fed with data shaped like each API's real responses."""

import pandas as pd

from runitback.wager.cbb.data.kaggle import TEAM_STATS_COLUMNS, format_games
from runitback.wager.odds.utils import process_odds


def test_kaggle_rows_become_two_perspectives_with_names():
    stats = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO"]
    stats += ["Stl", "Blk", "PF"]
    row = {"Season": 2026, "DayNum": 30, "NumOT": 0, "WLoc": "A"}
    row |= {"WTeamID": 1104, "WScore": 80, "LTeamID": 1101, "LScore": 70}
    row |= {f"W{s}": 10 for s in stats} | {f"L{s}": 5 for s in stats}
    teams = pd.DataFrame(
        {"TeamID": [1101, 1104], "TeamName": ["Abilene Chr", "Alabama"]}
    )
    games = format_games(pd.DataFrame([row]), teams)

    assert len(games) == 2 and games["_id"].nunique() == 1
    bama = games[games.TeamID == 1104].iloc[0]
    acu = games[games.TeamID == 1101].iloc[0]
    assert (bama.Team, bama.Loc, bama.PlusMinus, bama.FGM, bama.FGM_OPP) == (
        "Alabama",
        -1,
        10,
        10,
        5,
    )
    assert (acu.Team_OPP, acu.Loc, acu.PlusMinus, acu.Win) == ("Alabama", 1, -10, 0)
    assert set(TEAM_STATS_COLUMNS) <= set(games.columns)


def test_odds_api_payload_to_consensus_lines():
    odds = [
        {
            "id": "x",
            "commence_time": "2026-11-01T13:00:00-05:00",
            "home_team": "Buffalo Bills",
            "away_team": "Baltimore Ravens",
            "bookmakers": [
                {
                    "key": book,
                    "title": book,
                    "last_update": "2026-10-30T12:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-10-30T12:00:00Z",
                            "outcomes": [
                                {"name": "Buffalo Bills", "price": -150 + shift},
                                {"name": "Baltimore Ravens", "price": 130},
                            ],
                        },
                        {
                            "key": "spreads",
                            "last_update": "2026-10-30T12:00:00Z",
                            "outcomes": [
                                {"name": "Buffalo Bills", "price": -110, "point": -3.0},
                                {
                                    "name": "Baltimore Ravens",
                                    "price": -110,
                                    "point": 3.0,
                                },
                            ],
                        },
                        {
                            "key": "totals",
                            "last_update": "2026-10-30T12:00:00Z",
                            "outcomes": [
                                {"name": "Over", "price": -110, "point": 47.5},
                                {"name": "Under", "price": -110, "point": 47.5},
                            ],
                        },
                    ],
                }
                for book, shift in [("dk", 0), ("fd", 10)]
            ],
        }
    ]
    from runitback.wager.nfl.data.names import odds_map

    lines = process_odds(odds, odds_map).iloc[0]
    assert (lines.team, lines.opp_team) == ("BUF", "BAL")
    assert lines.moneyline == -145 and lines.spread_line == -3.0
    assert lines.total_line == 47.5
