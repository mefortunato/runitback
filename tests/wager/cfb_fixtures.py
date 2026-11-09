"""Frames shaped like sportsdataverse's CFB release files and ESPN's odds."""

import pandas as pd

TEAM_IDS = {"Alabama": 333, "Georgia": 61, "Clemson": 228, "Auburn": 2,
            "Ohio State": 194, "Notre Dame": 87, "Montana State": 147,
            "Montana": 149}  # fmt: skip


def game(gid, week, home, away, hp, ap, start, season=2024, season_type="regular",
         neutral=False, conf=False, notes=None, completed=True,
         home_class="fbs", away_class="fbs"):  # fmt: skip
    return {
        "game_id": gid,
        "season": season,
        "week": week,
        "season_type": season_type,
        "start_date": start,
        "start_time_tbd": False,
        "completed": completed,
        "neutral_site": neutral,
        "conference_game": conf,
        "venue": "Stadium",
        "home_id": TEAM_IDS[home],
        "home_team": home,
        "home_division": home_class,
        "home_conference": "SEC" if home_class == "fbs" else "Big Sky",
        "home_points": hp if completed else None,
        "away_id": TEAM_IDS[away],
        "away_team": away,
        "away_division": away_class,
        "away_conference": "SEC" if away_class == "fbs" else "Big Sky",
        "away_points": ap if completed else None,
        "notes": notes,
    }


SCHEDULE = pd.DataFrame(
    [
        game(1, 1, "Alabama", "Montana State", 49, 7, "2024-08-31T23:00:00.000Z",
             away_class="fcs"),
        game(2, 1, "Georgia", "Clemson", 34, 3, "2024-08-31T19:30:00.000Z",
             neutral=True),
        game(3, 1, "Montana", "Montana State", 20, 24, "2024-08-31T20:00:00.000Z",
             home_class="fcs", away_class="fcs"),
        game(4, 2, "Alabama", "Georgia", 41, 34, "2024-09-07T23:30:00.000Z",
             conf=True),
        # cancelled: never completed, in the past
        game(5, 2, "Clemson", "Auburn", None, None, "2024-09-07T16:00:00.000Z",
             completed=False),
        game(6, 3, "Clemson", "Georgia", 17, 24, "2024-09-21T16:00:00.000Z"),
        game(7, 1, "Georgia", "Alabama", 10, 28, "2024-12-20T01:00:00.000Z",
             season_type="postseason", neutral=True, notes="CFP First Round Game"),
        game(8, 1, "Clemson", "Auburn", 31, 30, "2024-12-28T20:00:00.000Z",
             season_type="postseason", neutral=True, notes="Gator Bowl"),
        game(9, 1, "Alabama", "Ohio State", 20, 27, "2025-01-10T00:30:00.000Z",
             season_type="postseason", neutral=True,
             notes="CFP Semifinal at the Cotton Bowl Classic"),
        game(10, 1, "Notre Dame", "Ohio State", 23, 34, "2025-01-21T00:30:00.000Z",
             season_type="postseason", neutral=True,
             notes="College Football Playoff National Championship"),
        # scheduled, not played yet
        game(11, 1, "Georgia", "Auburn", None, None, "2099-12-31T20:00:00.000Z",
             season_type="postseason", neutral=True, completed=False,
             notes="Sugar Bowl"),
    ]
)  # fmt: skip


def box_team(gid, name, home_away, yards, third="5-12"):
    return {
        "firstDowns": "22",
        "thirdDownEff": third,
        "fourthDownEff": "1-2",
        "totalYards": str(yards),
        "netPassingYards": "250",
        "completionAttempts": "20/31",
        "yardsPerPass": "8.1",
        "rushingYards": str(yards - 250),
        "rushingAttempts": "35",
        "yardsPerRushAttempt": "5.0",
        "totalPenaltiesYards": "5-45",
        "turnovers": "1",
        "fumblesLost": "0",
        "interceptions": "1",
        "possessionTime": "31:30",
        "team_id": TEAM_IDS[name],
        "team_abbreviation": name[:4].upper(),
        "team_name": f"{name} Mascots",  # ESPN's display name, not the key
        "home_away": home_away,
        "game_id": gid,
        "season": 2024,
    }


BOX = pd.DataFrame(
    [
        box_team(1, "Alabama", "home", 550),
        box_team(1, "Montana State", "away", 210, "3-14"),
        box_team(2, "Georgia", "home", 450),
        box_team(2, "Clemson", "away", 240),
        box_team(4, "Alabama", "home", 500),
        box_team(4, "Georgia", "away", 480),
    ]
)


def item(name, spread=None, total=None, home_ml=None, away_ml=None, **kw):
    """One book's line in ESPN's odds API (``spread``: the home handicap)."""
    return {
        "provider": {"name": name},
        "spread": spread,
        "overUnder": total,
        "overOdds": kw.get("over"),
        "underOdds": kw.get("under"),
        "homeTeamOdds": {"moneyLine": home_ml, "spreadOdds": kw.get("home_price"),
                         "open": kw.get("home_open")},
        "awayTeamOdds": {"moneyLine": away_ml, "spreadOdds": kw.get("away_price")},
        "open": kw.get("open"),
    }  # fmt: skip


ODDS = {
    # home favored by 38.5; the preferred book's moneyline is a placeholder
    "1": [item("BOVADA.lv", -38.5, 55.5, 0, 0), item("teamrankings", -37.0, 54.0)],
    "2": [item("numberfire", -10.0, 47.0),
          item("DraftKings", -13.5, 48.5, -500, 375)],
    "4": [item("ESPN BET", 1.5, 50.5, 105, -125, home_price=-108, away_price=-112,
               over=-110, under=-110)],
    "11": [item("DraftKings", -3.0, 52.5, -150, 130)],
}  # fmt: skip
