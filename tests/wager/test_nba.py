import datetime
import io
import json

import numpy as np
import pandas as pd
import pytest

from runitback.wager import espn as shared
from runitback.wager.core.process import process
from runitback.wager.core.schema import mirror, validate
from runitback.wager.nba.data import download as espn
from runitback.wager.nba.data.download import build_games, parse_odds
from runitback.wager.nba.data.features import SCHEMA
from runitback.wager.nba.data.teams import teams_frame

TODAY = datetime.datetime(2025, 6, 20, 12, tzinfo=datetime.timezone.utc)

TEAMS = {
    2: ("BOS", "Boston Celtics"),
    17: ("NJ", "New Jersey Nets"),
    9: ("GS", "Golden State Warriors"),
    6: ("DAL", "Dallas Mavericks"),
    31: ("EAST", "Eastern Conf All-Stars"),
}


def game(gid, start, home, away, scores=(None, None), season_type=2, **extra):
    hs, as_ = scores
    return {
        "game_id": gid,
        "season": 2025,
        "season_type": season_type,
        "date": start,
        "neutral_site": False,
        "status_type_completed": hs is not None,
        "notes_headline": None,
        **{f"home_{k}": v for k, v in zip(("id", "abbreviation", "display_name"),
                                           (home, *TEAMS[home]))},
        **{f"away_{k}": v for k, v in zip(("id", "abbreviation", "display_name"),
                                           (away, *TEAMS[away]))},
        "home_score": hs,
        "away_score": as_,
        **extra,
    }  # fmt: skip


SCHEDULE = pd.DataFrame(
    [
        # Tuesday opener (2024-10-22 evening Eastern), then a back-to-back
        game(1, "2024-10-22T23:30Z", 2, 17, (110, 100)),
        game(2, "2024-10-26T02:00Z", 9, 6, (120, 125)),
        game(3, "2024-10-23T23:00Z", 17, 9, (99, 101)),
        game(4, "2024-10-24T23:00Z", 17, 2, (90, 95), neutral_site=True),
        game(5, "2024-10-20T23:00Z", 2, 9, (100, 90), season_type=1),  # preseason
        game(6, "2025-02-16T01:00Z", 31, 2, (150, 140)),  # All-Star team
        game(7, "2025-01-10T00:00Z", 6, 2),  # postponed, never played
        game(8, "2025-06-05T00:30Z", 2, 6, (107, 89), season_type=3),
        game(9, "2025-06-21T00:30Z", 6, 2),  # upcoming
    ]
)
SCHEDULE["game_id"] = SCHEDULE["game_id"].astype(str)


def box_row(gid, team_id, **stats):
    base = {"field_goals_made": 40, "field_goals_attempted": 88,
            "three_point_field_goals_made": 12, "three_point_field_goals_attempted": 35,
            "free_throws_made": 18, "free_throws_attempted": 22,
            "offensive_rebounds": 10, "defensive_rebounds": 33, "assists": 25,
            "steals": 8, "blocks": 5, "total_turnovers": 13, "fouls": 19,
            "points_in_paint": 44, "fast_break_points": 12,
            "turnover_points": 15}  # fmt: skip
    return {"game_id": gid, "team_id": team_id, **base, **stats}


BOX = pd.DataFrame(
    [box_row(g, t) for g, h, a in [(1, 2, 17), (2, 9, 6), (3, 17, 9), (4, 17, 2),
                                   (8, 2, 6)] for t in (h, a)]
    + [box_row(9, 6, assists=99)]  # a stray box score of an unplayed game
)  # fmt: skip

LINES = {
    "1": {"spread_line": 5.5, "spread_odds": -110.0, "opp_spread_odds": -110.0,
          "total_line": 215.5, "over_odds": -105.0, "under_odds": -115.0,
          "moneyline": -220.0, "opp_moneyline": 180.0},
    "9": {"spread_line": -2.5, "total_line": 208.0},
}  # fmt: skip


@pytest.fixture
def games():
    return build_games(SCHEDULE, BOX, LINES, today=TODAY)


def row(games, game_id, team):
    rows = games[(games.game_id == str(game_id)) & (games.team == team)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_schema_columns_and_two_rows_per_game(games):
    validate(games, SCHEMA)
    assert games.groupby("game_id").size().eq(2).all()
    assert len(process("nba", games)) == len(games)


def test_preseason_all_star_and_postponed_games_are_dropped(games):
    assert set(games.game_id) == {"1", "2", "3", "4", "8", "9"}


def test_franchises_keep_their_current_key_and_the_name_they_played_under(games):
    nets = row(games, 1, "BKN")
    assert (nets.team_name, nets.opp_team, nets.location) == (
        "New Jersey Nets",
        "BOS",
        -1,
    )
    assert row(games, 2, "GSW").opp_team == "DAL"


def test_results_locations_and_upcoming_games(games):
    bos = row(games, 1, "BOS")
    assert (bos.points, bos.opp_points, bos.result, bos.total) == (110, 100, 10, 210)
    assert row(games, 4, "BKN").location == 0 and row(games, 4, "BOS").location == 0
    upcoming = row(games, 9, "DAL")
    assert np.isnan(upcoming.result) and np.isnan(upcoming.ast)
    assert upcoming.spread_line == -2.5 and row(games, 9, "BOS").spread_line == 2.5
    assert row(games, 8, "BOS").game_type == "playoffs"
    assert row(games, 8, "BOS").postseason == 1 and bos.postseason == 0


def test_days_weeks_and_rest_follow_eastern_dates(games):
    day = games.drop_duplicates("game_id").set_index("game_id")
    # game 2 starts 02:00 UTC on the 26th: 10 pm on the 25th in New York
    assert day.loc["2", "game_date"] == "2024-10-25"
    assert list(day.loc[["1", "2", "3", "4"], "day"]) == [0, 3, 1, 2]
    assert list(day.loc[["1", "2", "8"], "week"]) == [1, 1, 33]
    nets = games[games.team == "BKN"].set_index("game_id")["rest"]
    assert (nets["1"], nets["3"], nets["4"]) == (espn.MAX_REST, 1, 1)
    assert row(games, 3, "BKN").opp_rest == espn.MAX_REST  # Golden State's opener
    assert row(games, 2, "GSW").rest == 2


def test_box_scores_join_both_sides(games):
    bos = row(games, 1, "BOS")
    assert (bos.fgm, bos.opp_fgm, bos.tov, bos.oreb) == (40, 40, 13, 10)
    assert set(SCHEMA.stats) <= set(games.columns)


def test_market_columns_from_both_perspectives(games):
    bos, nets = row(games, 1, "BOS"), row(games, 1, "BKN")
    assert (bos.spread_line, nets.spread_line) == (5.5, -5.5)
    assert (bos.moneyline, bos.opp_moneyline) == (-220, 180)
    assert (nets.moneyline, nets.opp_moneyline) == (180, -220)
    assert (bos.over_odds, nets.over_odds, bos.total_line) == (-105, -105, 215.5)
    mirrored = mirror(games[games.game_id == "1"], SCHEMA)
    assert sorted(mirrored.spread_line) == [-5.5, 5.5]


def item(name, spread=None, total=None, home_ml=None, away_ml=None, **kw):
    return {
        "provider": {"name": name},
        "spread": spread,
        "overUnder": total,
        "overOdds": kw.get("over"),
        "underOdds": kw.get("under"),
        "homeTeamOdds": {"moneyLine": home_ml, "spreadOdds": kw.get("home_price")},
        "awayTeamOdds": {"moneyLine": away_ml, "spreadOdds": kw.get("away_price")},
    }


def test_lines_prefer_books_and_skip_predictions_live_and_placeholders():
    lines = parse_odds(
        {
            "items": [
                item("numberfire", spread=-20, total=100),
                item("ESPN Bet - Live Odds", spread=9, total=250, home_ml=500,
                     away_ml=-900),
                item("ESPN BET", spread=-6.5, total=221.5, home_ml=-250, away_ml=205,
                     over=-110, under=-110, home_price=-108, away_price=-112),
                item("consensus", spread=-7, home_ml=-5, away_ml=3),
            ]
        }
    )  # fmt: skip
    # ESPN's spread is the home handicap: -7 means the home team by 7
    assert (lines["spread_line"], lines["spread_provider"]) == (7.0, "consensus")
    assert (lines["total_line"], lines["total_provider"]) == (221.5, "ESPN BET")
    assert (lines["moneyline"], lines["opp_moneyline"]) == (-250, 205)
    assert lines["moneyline_provider"] == "ESPN BET"
    assert lines["spread_odds"] is None  # consensus has no prices
    assert parse_odds({"items": []}) == {}
    away_favored = parse_odds({"items": [item("DraftKings", spread=5.5)]})
    assert away_favored["spread_line"] == -5.5


def test_finished_games_lines_are_cached_and_upcoming_refetched(monkeypatch, tmp_path):
    calls = []

    def fetch(game_id):
        calls.append(game_id)
        return {"spread_line": float(game_id)}

    monkeypatch.setattr(espn, "fetch_odds", fetch)
    cache = espn.Cache(str(tmp_path))
    first = espn.season_lines(2025, ["1", "2"], ["9"], cache)
    assert first == {g: {"spread_line": float(g)} for g in ("1", "2", "9")}
    assert json.loads((tmp_path / "odds_2025.json").read_text()).keys() == {"1", "2"}
    calls.clear()
    espn.season_lines(2025, ["1", "2", "3"], ["9"], cache)
    assert sorted(calls) == ["3", "9"]


def test_download_caches_finished_seasons_files(monkeypatch, tmp_path):
    requested = []

    class Response:
        status_code = 200

        def __init__(self, data):
            self.content = data

        def raise_for_status(self):
            pass

    def get(url):
        requested.append(url)
        buffer = io.BytesIO()
        frame = SCHEDULE if "schedules" in url else BOX
        frame.to_parquet(buffer)
        return Response(buffer.getvalue())

    monkeypatch.setattr(shared, "_get", get)
    monkeypatch.setattr(espn, "FIRST_LINES_SEASON", 3000)
    cache = espn.Cache(str(tmp_path))
    for _ in range(2):
        schedule, box, _ = espn.fetch_season(2024, TODAY, cache)
        espn.fetch_season(2025, TODAY, cache)  # not finished: never cached
    assert len(schedule) == len(SCHEDULE) and len(box) == len(BOX)
    assert sum("2024" in url for url in requested) == 2
    assert sum("2025" in url for url in requested) == 4


def test_season_of_and_completion():
    assert espn.season_of(datetime.date(2025, 10, 21)) == 2026
    assert espn.season_of(datetime.date(2026, 6, 10)) == 2026
    assert not espn.season_complete(2025, TODAY)
    assert espn.season_complete(2024, TODAY)


def test_team_metadata():
    teams = [
        {"team": {"id": "9", "abbreviation": "GS",
                  "displayName": "Golden State Warriors",
                  "shortDisplayName": "Warriors", "color": "1d428a",
                  "alternateColor": "ffc72c", "logos": [{"href": "gsw.png"}]}},
        {"team": {"id": "17", "abbreviation": "BKN", "displayName": "Brooklyn Nets",
                  "shortDisplayName": "Nets", "color": "000000", "logos": []}},
    ]  # fmt: skip
    groups = [
        {"abbreviation": "West",
         "children": [{"name": "Pacific", "teams": [{"id": "9"}]}]},
        {"abbreviation": "East",
         "children": [{"name": "Atlantic", "teams": [{"id": "17"}]}]},
    ]  # fmt: skip
    frame = teams_frame(teams, groups).set_index("team")
    assert frame.loc["GSW", "nickname"] == "Warriors"
    assert (
        frame.loc["GSW", "color"] == "#1d428a" and frame.loc["GSW", "logo"] == "gsw.png"
    )
    assert (frame.loc["BKN", "conference"], frame.loc["BKN", "division"]) == (
        "East",
        "Atlantic",
    )
    # missing (None, or NaN in pandas 3 string columns; both export as null)
    assert pd.isna(frame.loc["BKN", "logo"]) and pd.isna(frame.loc["BKN", "color2"])


def test_odds_names_map_onto_franchise_keys():
    from runitback.wager.nba.data.names import odds_map

    assert set(odds_map.values()) == set(espn.CURRENT.values())
