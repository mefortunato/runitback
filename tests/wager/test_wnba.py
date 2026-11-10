import datetime

import pandas as pd

from runitback.wager.core.process import process
from runitback.wager.wnba.data import download as wnba
from runitback.wager.wnba.data.names import odds_map

TODAY = datetime.datetime(2025, 9, 1, tzinfo=datetime.timezone.utc)

TEAMS = {3: "TUL", 17: "SAS", 5: "IND", 18: "CONN", 98: "EAST", 99: "WEST"}


def game(gid, start, home, away, scores=(None, None), season=2025, **extra):
    hs, as_ = scores
    row = {"game_id": gid, "season": season, "season_type": 2, "date": start,
           "neutral_site": False, "status_type_completed": hs is not None,
           "notes_headline": None, "home_score": hs, "away_score": as_}  # fmt: skip
    for side, team in (("home", home), ("away", away)):
        row |= {f"{side}_id": team, f"{side}_abbreviation": TEAMS[team],
                f"{side}_display_name": f"{TEAMS[team]} team"}  # fmt: skip
    return row | extra


def schedule():
    rows = [
        # Tulsa (id 3) became Dallas, San Antonio (17) Las Vegas
        *(
            game(i, f"2014-06-0{i}T23:00Z", 3, 17, (80, 70), season=2014)
            for i in (1, 3, 4, 5, 6)
        ),
        game(2, "2014-07-20T19:00Z", 98, 99, (120, 115), season=2014),  # All-Star
    ]
    gid = 10
    for day in range(6):
        for home, away in ((3, 5), (17, 18)):
            rows.append(game(gid, f"2025-06-{day + 1:02d}T23:00Z", home, away,
                             (85, 80)))  # fmt: skip
            gid += 1
    rows.append(game(99, "2025-09-02T23:00Z", 17, 3))  # upcoming
    frame = pd.DataFrame(rows)
    frame.loc[frame.season == 2025, "home_abbreviation"] = frame["home_id"].map(
        {3: "DAL", 17: "LV", 5: "IND", 18: "CON"}
    )
    frame.loc[frame.season == 2025, "away_abbreviation"] = frame["away_id"].map(
        {3: "DAL", 17: "LV", 5: "IND", 18: "CON"}
    )
    return frame


def test_franchises_keep_their_latest_abbreviation():
    games = wnba.build_games(schedule(), pd.DataFrame(), today=TODAY)
    old = games[(games.game_id == "1") & (games.location == 1)].iloc[0]
    assert (old.team, old.opp_team, old.team_name) == ("DAL", "LV", "TUL team")
    assert set(games.team) == {"DAL", "LV", "IND", "CON"}


def test_all_star_teams_are_dropped_and_upcoming_kept():
    games = process("wnba", wnba.build_games(schedule(), pd.DataFrame(), today=TODAY))
    assert "2" not in set(games.game_id)
    upcoming = games[games.result.isna()]
    assert set(upcoming.game_id) == {"99"}


def test_seasons_are_calendar_years():
    assert wnba.season_of(datetime.date(2025, 10, 10)) == 2025
    assert not wnba.season_complete(2025, TODAY)
    assert wnba.season_complete(2024, TODAY)


def test_odds_names_map_onto_current_abbreviations():
    assert len(set(odds_map.values())) == len(odds_map) == 15
    assert odds_map["Las Vegas Aces"] == "LV"


def test_stats_espn_only_has_since_2023_are_not_model_inputs():
    from runitback.wager.cbb.data.features import RECENT_STATS
    from runitback.wager.core.sport import get_sport

    for sport in ("wnba", "cbb", "wcbb"):
        stats = set(get_sport(sport).schema.stats)
        assert not stats & {*RECENT_STATS, *(f"opp_{c}" for c in RECENT_STATS)}
    assert "paint_points" in get_sport("nba").schema.stats
