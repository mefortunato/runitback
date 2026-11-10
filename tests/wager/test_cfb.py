import datetime
import io
import json

import numpy as np
import pandas as pd
import pytest

from runitback.wager.cfb.data import download as cfb
from runitback.wager.cfb.data.download import FCS, build_games, parse_odds
from runitback.wager.cfb.data.features import SCHEMA
from runitback.wager.core.process import process
from runitback.wager.core.schema import mirror, validate

from .cfb_fixtures import BOX, ODDS, SCHEDULE, item

TODAY = datetime.datetime(2025, 3, 1, tzinfo=datetime.timezone.utc)
LINES = {gid: parse_odds({"items": items}) for gid, items in ODDS.items()}


@pytest.fixture
def games():
    return build_games(SCHEDULE, BOX, LINES, today=TODAY)


def row(games, game_id, team_name):
    rows = games[(games.game_id == str(game_id)) & (games.team_name == team_name)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_schema_columns_and_two_rows_per_game(games):
    validate(games, SCHEMA)
    assert games.groupby("game_id").size().eq(2).all()
    processed = process("cfb", games)
    assert len(processed) == len(games)


def test_fcs_opponents_are_pooled_and_fcs_only_games_dropped(games):
    assert "3" not in set(games.game_id)
    bama = row(games, 1, "Alabama")
    assert (bama.opp_team, bama.opp_name, bama.opp_conference) == (
        FCS,
        "Montana State",
        FCS,
    )
    msu = row(games, 1, "Montana State")
    assert (msu.team, msu.location, msu.result) == (FCS, -1, -42)
    assert np.isnan(msu.rest)


def test_neutral_sites_and_cancelled_and_upcoming_games(games):
    assert row(games, 2, "Georgia").location == 0
    assert row(games, 2, "Clemson").location == 0
    assert "5" not in set(games.game_id)  # cancelled
    sugar = row(games, 11, "Georgia")
    assert np.isnan(sugar.result) and np.isnan(sugar.totalYards)
    assert sugar.spread_line == 3.0


def test_bowl_weeks_follow_the_regular_season_in_order(games):
    week = games.drop_duplicates("game_id").set_index("game_id")["week"]
    assert list(week[["1", "4", "6"]]) == [1, 2, 3]
    # first round, a bowl a week later, semifinal, final: increasing weeks
    assert week["7"] == 4
    assert week["7"] < week["8"] < week["9"] < week["10"]
    types = games.drop_duplicates("game_id").set_index("game_id")["game_type"]
    assert (types["1"], types["7"], types["8"], types["10"]) == (
        "regular",
        "playoff",
        "bowl",
        "final",
    )


def test_lines_are_the_books_median_and_skip_predictions_and_placeholders():
    # BOVADA's moneylines are placeholders, teamrankings is a prediction site
    assert (LINES["1"]["spread_line"], LINES["1"]["total_line"]) == (38.5, 55.5)
    assert "moneyline" not in LINES["1"]
    assert (LINES["2"]["spread_line"], LINES["2"]["total_line"]) == (13.5, 48.5)
    assert (LINES["2"]["moneyline"], LINES["2"]["opp_moneyline"]) == (-500, 375)
    lines = parse_odds(
        {
            "items": [
                item("Opening", -1.0, 40.0),
                item("consensus", -1.0),  # ESPN's consensus: the opening line
                item("ESPN Bet - Live Odds", 9.0, 90.5, 500, -900),
                item("BOVADA.lv", -7.0, 52.0, -280, 230, home_price=-115),
                item("DraftKings", -6.5, 37.5, -250, 205, home_price=-110,
                     away_price=-110, over=-105, under=-115),  # a broken total
                item("5Dimes.eu", -6.5, 51.5, -260, 220, home_price=-105,
                     over=-110, under=-110),
            ]
        }
    )  # fmt: skip
    # ESPN's spread is the home handicap: -6.5 means the home team by 6.5
    assert (lines["spread_line"], lines["spread_odds"]) == (6.5, -110)
    assert (lines["total_line"], lines["over_odds"]) == (51.5, -110)
    assert (lines["moneyline"], lines["opp_moneyline"]) == (-250, 205)
    assert lines["moneyline_provider"] == "DraftKings"
    # an even count's median between two quotes: no book's prices
    between = parse_odds({"items": [item("DraftKings", -3.0), item("Westgate", -4.0)]})
    assert between == {"spread_line": 3.5, "spread_odds": None, "opp_spread_odds": None}
    # 2013: ESPN's consensus spread is the only line
    early = parse_odds({"items": [item("consensus", 5.5), item("numberfire", 3, 50)]})
    assert early == {"spread_line": -5.5, "spread_odds": None, "opp_spread_odds": None}
    assert parse_odds({"items": []}) == {}


def test_market_columns_from_both_perspectives(games):
    home, away = row(games, 4, "Alabama"), row(games, 4, "Georgia")
    assert (home.spread_line, away.spread_line) == (-1.5, 1.5)
    assert (home.moneyline, home.opp_moneyline) == (105, -125)
    assert (away.moneyline, away.opp_moneyline) == (-125, 105)
    assert (home.spread_odds, away.spread_odds) == (-108, -112)
    assert home.over_odds == away.over_odds == -110
    assert home.total_line == away.total_line == 50.5
    assert home.total == away.total == 75
    mirrored = mirror(games[games.game_id == "4"], SCHEMA)
    assert sorted(mirrored.spread_line) == [-1.5, 1.5]


def test_box_scores_join_both_sides(games):
    bama = row(games, 4, "Alabama")
    assert (bama.totalYards, bama.opp_totalYards) == (500, 480)
    assert (bama.passingCompletions, bama.passingAttempts) == (20, 31)
    assert (bama.thirdDownConv, bama.thirdDownAtt) == (5, 12)
    assert bama.possessionTime == pytest.approx(31.5)
    assert (bama.points, bama.opp_points, bama.result) == (41, 34, 7)
    # the FCS side keeps its own box score
    assert row(games, 1, "Alabama").opp_thirdDownAtt == 14
    # no box score: stats missing, result kept
    clemson = row(games, 6, "Clemson")
    assert clemson.result == -7 and np.isnan(clemson.totalYards)


def test_rest_days(games):
    bama = games[games.team_name == "Alabama"].set_index("game_id")
    assert bama.loc["1", "rest"] == cfb.MAX_REST  # opener
    assert bama.loc["4", "rest"] == 7
    assert bama.loc["4", "opp_rest"] == 7
    assert bama.loc["7", "rest"] == cfb.MAX_REST  # long layoff, capped


def test_finished_games_lines_are_cached_and_upcoming_refetched(monkeypatch, tmp_path):
    calls = []

    def fetch(game_id):
        calls.append(game_id)
        return {"spread_line": float(game_id)}

    monkeypatch.setattr(cfb, "fetch_odds", fetch)
    cache = cfb.Cache(str(tmp_path))
    first = cfb.season_lines(2024, ["1", "2"], ["11"], cache)
    assert first == {g: {"spread_line": float(g)} for g in ("1", "2", "11")}
    assert json.loads((tmp_path / "odds_2024.json").read_text()).keys() == {"1", "2"}
    calls.clear()
    cfb.season_lines(2024, ["1", "2", "4"], ["11"], cache)
    assert sorted(calls) == ["11", "4"]


def test_download_caches_finished_seasons(monkeypatch, tmp_path):
    requested, odds = [], []

    class Response:
        status_code = 200

        def __init__(self, data):
            self.content = data

        def raise_for_status(self):
            pass

    def get(url):
        requested.append(url)
        buffer = io.BytesIO()
        (SCHEDULE if "schedules" in url else BOX).to_parquet(buffer)
        return Response(buffer.getvalue())

    def fetch(game_id):
        odds.append(game_id)
        return parse_odds({"items": ODDS.get(game_id, [])})

    monkeypatch.setattr(cfb, "_get", get)
    monkeypatch.setattr(cfb, "fetch_odds", fetch)
    out = tmp_path / "raw.csv"
    cache = str(tmp_path / "cache")
    games = cfb.download(str(out), 2024, 2024, cache_dir=cache)
    assert out.exists() and len(games) == len(pd.read_csv(out))
    assert len(requested) == 2
    # every finished FBS game's lines (game 3 is FCS only)
    assert sorted(odds, key=int) == ["1", "2", "4", "6", "7", "8", "9", "10"]
    assert row(games, 2, "Georgia").spread_line == 13.5
    # 2024 is finished: the second download makes no requests
    again = cfb.download(None, 2024, 2024, cache_dir=cache)
    assert len(requested) == 2 and len(odds) == 8
    pd.testing.assert_frame_equal(again, games)


def test_season_of_and_completion():
    assert cfb.season_of(datetime.date(2025, 1, 10)) == 2024
    assert cfb.season_of(datetime.date(2025, 9, 1)) == 2025
    assert cfb.season_complete(2024, datetime.datetime(2025, 3, 1))
    assert not cfb.season_complete(2024, datetime.datetime(2025, 1, 20))


def test_playoff_marks(games):
    from runitback.wager.web.sports.cfb import playoffs

    assert playoffs(games[games.result.notna()]) == {
        2024: {
            "Alabama": "semifinal",
            "Notre Dame": "runner_up",
            "Ohio State": "champion",
        }
    }


def test_kickoff_in_eastern_time(games):
    from runitback.wager.web.sports.cfb import kickoff

    assert kickoff(row(games, 4, "Alabama")) == "2024-09-07 19:30"
    assert kickoff(pd.Series({"start_date": None})) is None
    tbd = pd.Series({"start_date": "2024-09-07T16:00:00.000Z", "start_time_tbd": True})
    assert kickoff(tbd) == "2024-09-07"


def test_view_stats_compute_on_games(games):
    from runitback.wager.web.sports.cfb import VIEW

    bama = games[(games.game_id == "4") & (games.team_name == "Alabama")]
    values = {s.key: s.value(bama).iloc[0] for s in VIEW.stats}
    assert values["yards"] == 500 and values["points_allowed"] == 34
    assert values["third_down_pct"] == pytest.approx(100 * 5 / 12)


def test_team_metadata():
    from runitback.wager.cfb.data.teams import teams_frame
    from runitback.wager.web.views import TEAM_COLUMNS

    info = pd.DataFrame(
        [
            {"school": "Air Force", "mascot": "Falcons", "abbreviation": "AF",
             "conference": "Mountain West", "division": None,
             "classification": "fbs", "color": "#003594", "alt_color": "ffffff",
             "logo": "https://cdn.collegefootballdata.com/logos/500/2005.png"},
            {"school": "Army", "mascot": "Black Knights", "abbreviation": None,
             "conference": "American", "division": "West", "classification": "fbs",
             "color": None, "alt_color": None, "logo": None},
            {"school": "Montana", "mascot": "Grizzlies", "abbreviation": "MONT",
             "conference": "Big Sky", "division": None, "classification": "fcs",
             "color": None, "alt_color": None, "logo": None},
        ]
    )  # fmt: skip
    teams = teams_frame(info).set_index("team")
    assert set(TEAM_COLUMNS) <= {"team", *teams.columns}
    assert "Montana" not in teams.index  # FCS schools are the pooled team
    assert teams.loc["Air Force", "color2"] == "#ffffff"
    assert (teams.loc["Air Force", "abbr"], teams.loc["Army", "abbr"]) == ("AF", "ARMY")
    assert teams.loc["Army", "division"] == "American West"
    assert teams.loc[FCS, "conference"] == FCS
