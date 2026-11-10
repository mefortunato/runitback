import datetime

import pandas as pd
import pytest

from runitback.wager.cbb.data import download as college
from runitback.wager.cbb.data.features import SCHEMA
from runitback.wager.core.process import process
from runitback.wager.core.schema import mirror, validate

TODAY = datetime.datetime(2025, 3, 20, 12, tzinfo=datetime.timezone.utc)

# id: (school, abbreviation, display name, conference id; None: not Division I)
TEAMS = {
    150: ("Duke", "DUKE", "Duke Blue Devils", 2),
    153: ("North Carolina", "UNC", "North Carolina Tar Heels", 2),
    2306: ("Kansas St", "KSU", "Kansas State Wildcats", 8),
    9001: ("Chowan", "CHOW", "Chowan Hawks", None),
    9002: ("Lees-McRae", "LMC", "Lees-McRae Bobcats", None),
}


def game(gid, start, home, away, scores=(None, None), season=2025, **extra):
    hs, as_ = scores
    row = {
        "game_id": gid,
        "season": season,
        "season_type": 2,
        "date": start,
        "neutral_site": False,
        "conference_competition": False,
        "status_type_completed": hs is not None,
        "notes_headline": None,
        "tournament_id": None,
        "home_score": hs,
        "away_score": as_,
    }
    for side, team in (("home", home), ("away", away)):
        school, abbreviation, name, conference = TEAMS[team]
        row |= {
            f"{side}_id": team,
            f"{side}_location": school,
            f"{side}_abbreviation": abbreviation,
            f"{side}_display_name": name,
            f"{side}_conference_id": conference,
        }
    return row | extra


SCHEDULE = pd.DataFrame(
    [
        # a season earlier Kansas State played as "Kansas State"
        game(1, "2023-11-10T00:00Z", 2306, 150, (70, 80), season=2024,
             home_location="Kansas State"),
        game(2, "2024-11-05T00:00Z", 150, 9001, (100, 50)),
        game(3, "2024-11-06T00:00Z", 153, 9002, (90, 60)),
        game(4, "2025-03-09T01:00Z", 153, 150, (80, 82),
             conference_competition=True),
        game(5, "2025-03-21T23:00Z", 2306, 153, neutral_site=True,
             season_type=3, tournament_id=22.0),  # upcoming NCAA game
        game(6, "2024-11-01T00:00Z", 150, 153, season_type=1),  # exhibition
        game(7, "2025-01-15T00:00Z", 150, 2306),  # postponed, never played
    ]
)  # fmt: skip

BOX = pd.DataFrame(
    [
        {"game_id": g, "team_id": t, "field_goals_made": 30,
         "field_goals_attempted": 60, "points_in_paint": "30"}
        for g, h, a in [(1, 2306, 150), (2, 150, 9001), (3, 153, 9002),
                        (4, 153, 150)]
        for t in (h, a)
    ]
)  # fmt: skip


@pytest.fixture
def games():
    return college.build_games(SCHEDULE, BOX, {"5": {"spread_line": -2.5}}, TODAY)


def row(games, game_id, team):
    rows = games[(games.game_id == str(game_id)) & (games.team == team)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_schema_columns_and_process(games):
    validate(games, SCHEMA)
    processed = process("cbb", games)
    assert set(processed.game_id) == {"1", "2", "3", "4", "5"}
    assert processed.groupby("game_id").size().eq(2).all()


def test_non_division_one_opponents_are_pooled_with_their_own_names(games):
    chowan = row(games, 2, college.NON_D1)
    assert (chowan.team_name, chowan.opp_team, chowan.conference) == (
        "Chowan Hawks",
        "Duke",
        college.NON_D1,
    )
    lees = row(games, 3, college.NON_D1)
    # the pooled team's rest counts each school's own games
    assert lees.rest == college.MAX_REST and chowan.rest == college.MAX_REST


def test_schools_keep_their_latest_name(games):
    old = row(games, 1, "Kansas St")
    assert old.team_name == "Kansas State Wildcats" and old.result == -10
    names = college.college_keys(
        pd.DataFrame(
            {"id": [1, 2], "opp_id": [2, 1], "school": ["Miami", "Miami"],
             "opp_school": ["Miami", "Miami"], "season": [2010, 2025],
             "day": [0, 0], "conference_id": [1, 1], "opp_conference_id": [1, 1]}
        )
    )[0]  # fmt: skip
    # a defunct id whose name another school took over is told apart
    assert list(names) == ["Miami (1)", "Miami"]


def test_game_types_conferences_and_neutral_sites(games):
    ncaa = row(games, 5, "Kansas St")
    assert (ncaa.game_type, ncaa.location, ncaa.postseason) == ("ncaa", 0, 1)
    assert pd.isna(ncaa.result) and ncaa.spread_line == -2.5
    assert row(games, 5, "North Carolina").spread_line == 2.5
    rivalry = row(games, 4, "Duke")
    assert (rivalry.conference_game, rivalry.conference, rivalry.opp_conference) == (
        1,
        "2",
        "2",
    )
    assert rivalry.game_type == "regular" and rivalry.location == -1
    assert row(games, 4, "Duke").paint_points == 30


def test_mirror_swaps_conferences(games):
    one = games[(games.game_id == "1") & (games.team == "Duke")]
    back = mirror(one, SCHEMA).iloc[0]
    assert (back.team, back.conference, back.opp_conference) == ("Kansas St", "8", "2")


def test_lines_only_for_division_one_games():
    games = college.espn.schedule_frame(SCHEDULE, college.SEASON_TYPES)
    wanted = games[college.division_one(games)]
    assert set(wanted.game_id) == {"1", "4", "5", "7"}


@pytest.mark.parametrize(
    "league", ["mens-college-basketball", "womens-college-basketball"]
)
def test_forbidden_espn_odds_do_not_block_college_training(
    monkeypatch, league, tmp_path
):
    from runitback.wager import espn

    calls = []

    class Forbidden:
        status_code = 403

    def get(url):
        calls.append(url)
        return Forbidden()

    cache = espn.Cache(str(tmp_path))
    monkeypatch.setattr(espn, "_ODDS_FORBIDDEN", set())
    monkeypatch.setattr(espn, "_get", get)

    def fetch(game_id):
        return espn.fetch_odds("basketball", league, game_id)

    assert espn.season_lines(2025, ["1", "2"], [], cache, fetch, workers=1) == {}
    assert len(calls) == 1
    assert not (tmp_path / "odds_2025.json").exists()


def test_season_of_and_completion():
    assert college.season_of(datetime.date(2024, 11, 5)) == 2025
    assert college.season_of(datetime.date(2025, 4, 7)) == 2025
    assert not college.season_complete(2025, TODAY)
    assert college.season_complete(2024, TODAY)


def test_women_share_the_builder():
    from runitback.wager.core.sport import get_sport

    assert get_sport("wcbb").schema is SCHEMA
    assert college.WOMEN.league.league == "womens-college-basketball"
    assert college.WOMEN.league.first_lines_season == 2022


def test_conferences_fall_back_to_the_cached_copy(monkeypatch, tmp_path):
    import httpx

    from runitback.wager import espn
    from runitback.wager.cbb.data import teams

    groups = [{"name": "ACC", "shortName": "ACC", "teams": [{"id": "150"}]}]
    cache = espn.Cache(str(tmp_path))
    monkeypatch.setattr(teams, "_fetch_conferences", lambda league, season: groups)
    assert teams.conferences("mens-college-basketball", 2025, cache) == groups

    def refused(league, season):
        request = httpx.Request("GET", "https://sports.core.api.espn.com")
        response = httpx.Response(403, request=request)
        raise httpx.HTTPStatusError("403", request=request, response=response)

    monkeypatch.setattr(teams, "_fetch_conferences", refused)
    assert teams.conferences("mens-college-basketball", 2025, cache) == groups
    # nothing cached: teams go without conferences instead of failing
    assert teams.conferences("mens-college-basketball", 2024, cache) == []


def test_build_stops_when_odds_were_refused(monkeypatch, tmp_path):
    from runitback.wager import espn
    from runitback.wager.web import build as web_build

    class Sport:
        def download(self):
            espn._ODDS_FORBIDDEN.add(("basketball", "mens-college-basketball"))
            return pd.DataFrame()

    monkeypatch.setattr(espn, "_ODDS_FORBIDDEN", set())
    monkeypatch.setattr(web_build, "get_sport", lambda sport: Sport())
    monkeypatch.setattr(web_build, "get_view", lambda sport: None)
    monkeypatch.setattr(web_build, "process", lambda sport, games: games)
    with pytest.raises(RuntimeError, match="refused odds"):
        web_build.build(sports=["cbb"], output_dir=str(tmp_path))
    assert not (tmp_path / "cbb" / "games.csv").exists()
