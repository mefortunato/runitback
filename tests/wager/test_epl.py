import datetime

import numpy as np
import pandas as pd
import pytest

from runitback.wager.core.process import process
from runitback.wager.core.schema import mirror, validate
from runitback.wager.epl.data import download as epl
from runitback.wager.epl.data.features import SCHEMA
from runitback.wager.epl.data.teams import CLUBS, team_codes

TODAY = datetime.datetime(2025, 4, 2, 12, tzinfo=datetime.timezone.utc)

HEADER = ("Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,"
          "HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,B365H,B365D,B365A,PSCH,PSCD,PSCA,"
          "B365>2.5,B365<2.5,AHh,AHCh,AvgCH,AvgCD,AvgCA,AvgC>2.5,AvgC<2.5")  # fmt: skip


def csv_bytes(*rows, header=HEADER, encoding="utf-8-sig"):
    return "\n".join([header, *rows]).encode(encoding)


# played; closing Pinnacle prices; four-digit year; handicap -0.5 -> closing -0.75
ARS_CHE = (
    "E0,10/08/2024,17:30,Arsenal,Chelsea,2,2,D,1,0,H,M Oliver,"
    "14,9,6,3,10,12,7,4,1,2,0,0,2.0,3.5,4.0,1.9,3.6,4.4,1.8,2.0,-0.5,-0.75,1.95,3.55,4.2,1.85,1.95"
)
# played; no closing prices (bet365), two-digit year, a Man City accent-free name
CHE_MCI = (
    "E0,17/08/24,,Chelsea,Man City,0,1,A,0,0,D,A Taylor,"
    "8,15,2,6,11,9,3,8,3,1,1,0,5.0,4.0,1.67,,,,,,,"
)
# not played yet and in the future
MCI_ARS = "E0,05/04/2025,15:00,Man City,Arsenal,,,,,,,,,,,,,,,,,,,,2.2,3.4,3.1,,,,,,,"
# postponed and past: dropped
ARS_MCI = "E0,01/03/2025,15:00,Arsenal,Man City,,,,,,,,,,,,,,,,,,,,2.2,3.4,3.1,,,,,,,"
# the following season's first game, in the same file the season column decides
CHE_ARS = (
    "E0,10/08/2025,17:30,Chelsea,Arsenal,3,1,H,2,0,H,M Oliver,"
    "10,10,5,3,9,9,5,5,1,2,0,0,2.4,3.4,3.0,,,,,,,"
)


@pytest.fixture
def raw():
    first = epl.read_csv(csv_bytes(ARS_CHE, CHE_MCI, MCI_ARS, ARS_MCI)).assign(
        season=2025
    )
    second = epl.read_csv(csv_bytes(CHE_ARS)).assign(season=2026)
    return pd.concat([first, second], ignore_index=True)


@pytest.fixture
def games(raw):
    return epl.build_games(raw, today=TODAY)


def row(games, season, home, team):
    rows = games[(games.game_id == f"{season}_{home}") & (games.team == team)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_read_csv_handles_encodings_blank_rows_and_wide_rows():
    wide = ARS_CHE + ",9.9,9.9,9.9"  # more cells than the header names
    data = csv_bytes(wide, ",,,,", "", ARS_CHE, encoding="latin-1")
    frame = epl.read_csv(data.replace(b"M Oliver", b"M \xd6liver"))
    assert len(frame) == 2 and list(frame.columns) == HEADER.split(",")
    assert frame.Referee.tolist() == ["M Öliver", "M Öliver"]
    assert frame.B365H.tolist() == ["2.0", "2.0"]  # extras dropped, not shifted


def test_american_odds():
    out = epl.american(pd.Series([2.0, 3.0, 1.5, 1.0, np.nan, "x"]))
    assert out[:3].tolist() == [100, 200, -200]
    assert np.isnan(out[3:]).all()


def test_schema_process_and_two_rows_per_game(games):
    validate(games, SCHEMA)
    processed = process("epl", games)
    assert set(processed.game_id) == {"2025_ARS_CHE", "2025_CHE_MCI",
                                      "2025_MCI_ARS", "2026_CHE_ARS"}  # fmt: skip
    assert games.groupby("game_id").size().eq(2).all()


def test_results_and_stats_from_both_sides(games):
    ars, che = row(games, 2025, "ARS_CHE", "ARS"), row(games, 2025, "ARS_CHE", "CHE")
    assert (ars.location, che.location) == (1, -1)
    assert (ars.result, che.result, ars.total) == (0, 0, 4)
    assert (ars.shots, ars.opp_shots, che.shots, che.opp_shots) == (14, 9, 9, 14)
    assert (ars.shots_on_target, ars.corners, ars.yellow_cards) == (6, 7, 1)
    assert (ars.half_time_goals, che.half_time_goals) == (1, 0)
    assert ars.team_name == "Arsenal" and ars.opp_name == "Chelsea"
    assert ars.referee == "M Oliver"
    away = row(games, 2025, "CHE_MCI", "MCI")
    assert (away.result, away.opp_goals) == (1, 0)


def test_future_games_kept_unplayed_past_ones_dropped(games):
    assert "2025_ARS_MCI" not in set(games.game_id)
    upcoming = row(games, 2025, "MCI_ARS", "MCI")
    assert pd.isna(upcoming.result) and np.isnan(upcoming.shots)
    assert upcoming.moneyline == pytest.approx(120)  # 2.2 -> +120


def test_odds_prefer_closing_prices_and_convert_to_american(games):
    ars, che = row(games, 2025, "ARS_CHE", "ARS"), row(games, 2025, "ARS_CHE", "CHE")
    assert ars.odds_source == "pinnacle_close"
    assert (ars.moneyline, ars.draw_moneyline, ars.opp_moneyline) == (-111, 260, 340)
    assert (che.moneyline, che.opp_moneyline) == (340, -111)  # mirrored
    assert (ars.total_line, ars.over_odds, ars.under_odds) == (2.5, -118, -105)  # the closing average
    assert ars.spread_line == 0.75 and che.spread_line == -0.75  # closing handicap
    # the closing market average has its own columns, whatever the 1X2 source
    assert (ars.avg_moneyline, ars.avg_draw_moneyline, ars.opp_avg_moneyline) == (
        -105,
        255,
        320,
    )
    assert (che.avg_moneyline, che.opp_avg_moneyline) == (320, -105)
    assert (ars.avg_over_odds, ars.avg_under_odds) == (-118, -105)
    opening = row(games, 2025, "CHE_MCI", "CHE")
    assert opening.odds_source == "bet365" and opening.moneyline == 400
    assert np.isnan(opening.total_line) and np.isnan(opening.spread_line)
    assert np.isnan(opening.avg_moneyline) and np.isnan(opening.avg_draw_moneyline)
    back = mirror(games[games.index == ars.name], SCHEMA).iloc[0]
    assert (back.team, back.moneyline, back.spread_line) == ("CHE", 340, -0.75)


def test_dates_rest_and_seasons_come_from_the_file(games):
    che = row(games, 2025, "CHE_MCI", "CHE")
    assert che.game_date == "2024-08-17" and che.rest == 7
    assert che.start_date != che.start_date  # no kickoff time in the file
    ars = row(games, 2025, "ARS_CHE", "ARS")
    assert ars.start_date == "2024-08-10T16:30Z"  # BST -> UTC
    assert ars.day == 0 and che.day == 7 and ars.rest == 14  # opener: the cap
    assert set(games[games.game_id.str.startswith("2026")].season) == {2026}
    assert row(games, 2026, "CHE_ARS", "CHE").rest == 14


def test_unknown_team_raises():
    with pytest.raises(ValueError, match="Hartlepool"):
        team_codes({"Arsenal", "Hartlepool"})


def test_clubs_have_unique_codes_and_names():
    codes = [code for code, _ in CLUBS.values()]
    assert len(set(codes)) == len(codes)
    assert len({full for _, full in CLUBS.values()}) == len(CLUBS)


def test_seasons():
    assert epl.season_of(datetime.date(2024, 8, 16)) == 2025
    assert epl.season_of(datetime.date(2025, 5, 25)) == 2025
    assert epl.season_code(2026) == "2526" and epl.season_code(2001) == "0001"
    assert epl.season_complete(2024, TODAY) and not epl.season_complete(2025, TODAY)


LEI_ARS = (
    "E0,17/08/2025,15:00,Leicester,Arsenal,0,2,A,0,1,A,M Dean,"
    "8,12,2,6,10,9,4,6,2,1,0,0,3.8,3.5,1.9,,,,,,,"
)


def test_promoted_clubs_are_those_not_in_the_previous_seasons_league(raw):
    second = epl.read_csv(csv_bytes(CHE_ARS, LEI_ARS)).assign(season=2026)
    games = epl.build_games(pd.concat([raw[raw.season == 2025], second], ignore_index=True), today=TODAY)
    # Leicester were not in 2025: promoted; Arsenal and Chelsea were
    leicester, arsenal = row(games, 2026, "LEI_ARS", "LEI"), row(games, 2026, "LEI_ARS", "ARS")
    assert (leicester.promoted, leicester.opp_promoted) == (1, 0)
    assert (arsenal.promoted, arsenal.opp_promoted) == (0, 1)
    chelsea = row(games, 2026, "CHE_ARS", "CHE")
    assert (chelsea.promoted, chelsea.opp_promoted) == (0, 0)
    # the first season has no previous one to compare with: nobody is promoted
    assert (games[games.season == 2025][["promoted", "opp_promoted"]] == 0).all().all()


def test_odds_closing_says_which_prices_are_closing_prices(games):
    assert row(games, 2025, "ARS_CHE", "ARS").odds_closing == 1  # closing Pinnacle
    assert row(games, 2025, "CHE_MCI", "CHE").odds_closing == 0  # bet365's opening prices
    assert row(games, 2025, "CHE_MCI", "MCI").odds_closing == 0  # both sides of the game
