"""MLB live-feed parsing: plate appearances, base / out replay, lineups, substitutions, defense.

The fixtures are real Stats API feeds, pruned as the cache stores them:
- 745444: 2024 regular season, Dodgers at Padres in Seoul (pinch hitter, defensive switches);
- 718723: 2023 regular season, 11 innings (automatic runner on second in extras);
- 715769: 2022 wild card, 15 innings (postseason: no automatic runner).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from runitback.mlb.data import feed as F
from runitback.mlb.data import statsapi

FIXTURES = Path(__file__).parent / "fixtures"
GAMES = (745444, 718723, 715769)
MISMATCHES = ("base_mismatch", "post_mismatch", "outs_mismatch", "runs_mismatch", "box_pa_mismatch",
              "fielder_mismatch")


def load(pk: int) -> dict:
    return statsapi.read_cached(FIXTURES / f"feed_{pk}.json.gz")


@pytest.fixture(scope="module", params=GAMES)
def parsed(request):
    return F.parse_feed(load(request.param))


def test_replay_agrees_with_feed(parsed):
    checks = parsed["checks"]
    assert {k: checks.get(k, 0) for k in MISMATCHES} == dict.fromkeys(MISMATCHES, 0)
    assert checks["fielder_checks"] > 50


def test_runs_add_up_to_final_score(parsed):
    plays = pd.DataFrame(parsed["plays"])
    game = parsed["game"]
    for side, team in (("home", game["home_id"]), ("away", game["away_id"])):
        mine = plays[plays["bat_team_id"] == team]
        assert (mine["runs_pre"] + mine["runs"]).sum() == game[f"{side}_runs"]
        # runs_rest at each half inning's first play is that half's runs
        firsts = mine.groupby(["inning", "half"]).head(1)
        assert firsts["runs_rest"].sum() == game[f"{side}_runs"]


def test_pa_counts_match_box_score(parsed):
    plays = pd.DataFrame(parsed["plays"])
    lineups = pd.DataFrame(parsed["lineups"])
    pa = plays[plays["is_pa"]].groupby("batter_id").size()
    box = lineups.dropna(subset=["box_pa"]).set_index("player_id")["box_pa"]
    box = box[box > 0]
    assert pa.sort_index().to_dict() == box.astype(int).sort_index().to_dict()
    bf = plays[plays["is_pa"]].groupby("pitcher_id").size()
    box_bf = lineups.dropna(subset=["box_bf"]).set_index("player_id")["box_bf"]
    assert bf.sort_index().to_dict() == box_bf[box_bf > 0].astype(int).sort_index().to_dict()


def test_states_are_consistent(parsed):
    plays = pd.DataFrame(parsed["plays"])
    assert plays[["start_bases", "pre_bases", "end_bases"]].isin(range(8)).all().all()
    assert plays[["start_outs", "pre_outs"]].isin(range(3)).all().all()
    assert plays["end_outs"].isin(range(4)).all()
    assert (plays.loc[plays["end_outs"] == 3, "end_bases"] == 0).all()
    # the next play in a half inning starts where this one ended
    nxt = plays.shift(-1)
    same = (nxt["inning"] == plays["inning"]) & (nxt["half"] == plays["half"])
    assert (nxt.loc[same, "start_bases"] == plays.loc[same, "end_bases"]).all()
    assert (nxt.loc[same, "start_outs"] == plays.loc[same, "end_outs"]).all()
    # every half inning starts with no outs
    firsts = plays.groupby(["inning", "half"]).head(1)
    assert (firsts["start_outs"] == 0).all()
    assert plays["result"].dropna().isin(set(F.RESULTS.values()) | {"OUT"}).all()


def test_first_plate_appearances():
    plays = pd.DataFrame(F.parse_feed(load(745444))["plays"])
    first = plays.iloc[0]  # Mookie Betts walks against Yu Darvish
    assert (first["batter_id"], first["pitcher_id"], first["spot"]) == (605141, 506433, 1)
    assert (first["bat_side"], first["pitch_hand"], first["result"]) == ("R", "R", "BB")
    assert (first["start_bases"], first["start_outs"], first["end_bases"], first["end_outs"]) == (0, 0, 1, 0)
    assert first["end_on_1b"] == 605141
    # 2nd inning: Teoscar Hernandez strikes out while the runner on first steals second
    steal = plays.iloc[8]
    assert (steal["result"], steal["start_bases"], steal["pre_bases"], steal["end_bases"]) == ("K", 1, 2, 2)
    assert (steal["pre_outs"], steal["end_outs"]) == (0, 1)


def test_automatic_runner_in_regular_season_extras():
    out = F.parse_feed(load(718723))
    plays = pd.DataFrame(out["plays"])
    assert out["checks"]["ghost_runners"] == 4  # 10th and 11th, both halves
    extras = plays[plays["inning"] >= 10].groupby(["inning", "half"]).head(1)
    assert (extras["start_bases"] == 2).all() and extras["start_on_2b"].notna().all()
    assert extras.iloc[0]["start_on_2b"] == 682985
    regulation = plays[plays["inning"] < 10].groupby(["inning", "half"]).head(1)
    assert (regulation["start_bases"] == 0).all()


def test_no_automatic_runner_in_postseason():
    out = F.parse_feed(load(715769))
    plays = pd.DataFrame(out["plays"])
    assert out["game"]["game_type"] == "F" and out["game"]["innings"] == 15
    assert "ghost_runners" not in out["checks"]
    assert (plays.groupby(["inning", "half"]).head(1)["start_bases"] == 0).all()


def test_lineups_and_substitutions():
    out = F.parse_feed(load(745444))
    lineups = pd.DataFrame(out["lineups"])
    starters = lineups[lineups["starter"]]
    for side in ("home", "away"):
        mine = starters[starters["side"] == side]
        assert sorted(mine["spot"]) == list(range(1, 10))
        # nine batters plus a starting pitcher; the DH game's fielders 2-9 each once
        assert sorted(mine["start_position"]) == sorted(["10", *F.FIELD_POSITIONS])
        assert lineups[(lineups["side"] == side) & lineups["is_sp"]].shape[0] == 1
    game = out["game"]
    assert (game["home_sp"], game["away_sp"]) == (506433, 607192)
    assert (game["home_probable_sp"], game["away_probable_sp"]) == (506433, 607192)
    subs = pd.DataFrame(out["substitutions"])
    ph = subs[subs["position"] == "11"].iloc[0]  # Enrique Hernandez pinch-hits for Jason Heyward
    assert (ph["player_id"], ph["replaced_id"], ph["batting_order"], ph["side"]) == (571771, 518792, 801, "away")
    assert set(subs["event_type"]) <= F.SUB_EVENTS
    # after the switch he plays left field
    plays = pd.DataFrame(out["plays"])
    later = plays[(plays["at_bat_index"] > 47) & (plays["fld_team_id"] == game["away_id"])]
    assert (later["fielder_7"].head(3) == 571771).all()


def test_prune_keeps_what_the_parser_reads():
    fd = load(745444)
    assert statsapi.prune(fd) == fd  # the fixture is already pruned: idempotent
    assert statsapi.is_final(fd)


def test_bases_code():
    assert F.bases_code({"1B": 1, "2B": None, "3B": 3}) == 5
    assert F.ghost_runner_rule(2020, "R") and not F.ghost_runner_rule(2019, "R")
    assert not F.ghost_runner_rule(2024, "W")
