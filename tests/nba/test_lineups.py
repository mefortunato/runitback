from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from runitback.nba.data.lineups import (
    TEAM_MINUTES, evaluate, parse_injuries, parse_posted_starters, parse_roster, project_live,
    projected_lineups, scale_minutes,
)

BOS = [1, 2, 3, 4, 5, 6, 7, 8]   # 1-5 start; 6-8 bench (6 plays most)
MINUTES = {1: 36, 2: 34, 3: 32, 4: 30, 5: 28, 6: 24, 7: 20, 8: 12, 9: 10}


def rows(game, date, team="BOS", players=BOS, minutes=None, starters=(1, 2, 3, 4, 5), dnp=None):
    """dnp: player -> reason (ESPN text)."""
    from runitback.nba.data.players import dnp_kind
    out = []
    for p in players:
        reason = (dnp or {}).get(p)
        out.append(dict(game_id=game, season=2025, game_date=date, team=team, player_id=p,
                        group="GFC"[p % 3], starter=p in starters and reason is None, dnp=reason is not None,
                        dnp_reason=reason, dnp_kind=dnp_kind(reason) if reason else None,
                        minutes=0.0 if reason else float((minutes or MINUTES)[p])))
    return out


def toy(target_dnp=None, target_players=BOS, target_minutes=None):
    r = rows("g1", "2025-01-01") + rows("g2", "2025-01-03") + rows("g3", "2025-01-05")
    r += rows("g4", "2025-01-07", players=target_players, dnp=target_dnp, minutes=target_minutes)
    r += rows("g5", "2025-01-09")
    return pd.DataFrame(r)


def team_game(proj, game):
    return proj[proj.game_id == game].set_index("player_id")


def test_scale_minutes():
    m = scale_minutes(np.array([30.0, 30, 30, 30, 30, 20, 20, 10]))
    assert m.sum() == pytest.approx(TEAM_MINUTES)
    assert m[0] / m[5] == pytest.approx(1.5)
    capped = scale_minutes(np.array([40.0, 10, 10, 10, 10, 10]))
    assert capped.max() <= 42 + 1e-9 and capped.sum() == pytest.approx(TEAM_MINUTES)
    assert scale_minutes(np.zeros(3)).tolist() == [0, 0, 0]


def test_injured_starter_is_ruled_out_and_replaced():
    proj = projected_lineups(toy(target_dnp={1: "LEFT ANKLE SPRAIN"}))
    d = team_game(proj, "g4")
    assert d.loc[1, "status"] == "out" and d.loc[1, "rank"] == -1 and d.loc[1, "proj_minutes"] == 0
    assert d.loc[1, "out_reason"] == "LEFT ANKLE SPRAIN"
    av = d[d.status == "available"]
    assert av.proj_minutes.sum() == pytest.approx(TEAM_MINUTES, abs=0.05)
    assert set(av.index[av.proj_starter]) == {2, 3, 4, 5, 6}   # the top bench player moves up
    assert av.sort_values("rank").index.tolist() == [2, 3, 4, 5, 6, 7, 8]
    assert (d.source == "box").all()


def test_coach_decision_and_rest_are_not_ruled_out():
    proj = projected_lineups(toy(target_dnp={8: "COACH'S DECISION", 7: "REST"}))
    d = team_game(proj, "g4")
    assert (d.status == "available").all()


def test_only_earlier_games_are_used():
    """The target game's minutes and starters, and later games, do not change its projection."""
    base = projected_lineups(toy(target_dnp={1: "ILLNESS"}))
    other = toy(target_dnp={1: "ILLNESS"}, target_minutes={p: 48 - 5 * p for p in BOS})
    other.loc[other.game_id == "g4", "starter"] = other.player_id.isin([6, 7, 8, 2, 3])
    other.loc[other.game_id == "g5", "minutes"] = 1.0
    changed = projected_lineups(other)
    pd.testing.assert_frame_equal(team_game(base, "g4"), team_game(changed, "g4"))
    first = team_game(base, "g1")   # no history yet: on the roster, no minutes
    assert (first.recent_games == 0).all() and (first.proj_minutes == 0).all()
    assert team_game(base, "g2").exp_minutes.to_dict() == {p: float(MINUTES[p]) for p in BOS}


def test_roster_from_box_vs_previous_rotation():
    # 5 is not in g4's box (traded / inactive); 9 is new
    pg = toy(target_players=[1, 2, 3, 4, 6, 7, 8, 9])
    box = team_game(projected_lineups(pg), "g4")
    assert 5 not in box.index and 9 in box.index and box.loc[9, "proj_minutes"] == 0
    rot = team_game(projected_lineups(pg, roster=False), "g4")
    assert 5 in rot.index and 9 not in rot.index and rot.loc[5, "proj_starter"]
    assert (rot.source == "rotation").all()


def test_evaluate_beats_previous_game_when_a_starter_is_out():
    pg = toy(target_dnp={1: "RIGHT KNEE SORENESS"})
    pg.loc[(pg.game_id == "g4") & (pg.player_id == 6), "starter"] = True
    ev = evaluate(projected_lineups(pg), pg)
    assert ev.loc[2025, "missed"] == 1 and ev.loc[2025, "missed_flagged"] == 1.0
    assert ev.loc[2025, "starters"] > ev.loc[2025, "prev_starters"]


def test_live_parsers_and_projection():
    injuries = parse_injuries({"injuries": [{"injuries": [
        {"status": "Out", "date": "2026-11-08T22:15Z", "shortComment": "x",
         "athlete": {"team": {"abbreviation": "BOS"},
                     "links": [{"href": "https://www.espn.com/nba/player/_/id/1/p-one"}]}},
        {"status": "Day-To-Day", "athlete": {"id": "2", "team": {"abbreviation": "BOS"}}}]}]})
    assert injuries[["player_id", "status"]].values.tolist() == [[1, "Out"], [2, "Day-To-Day"]]
    roster = parse_roster({"athletes": [{"id": str(p), "position": {"abbreviation": "G"}} for p in BOS]})
    posted = parse_posted_starters({"boxscore": {"players": [{"team": {"abbreviation": "BOS"}, "statistics": [
        {"athletes": [{"starter": p in (2, 3, 4, 5, 7), "athlete": {"id": str(p)}} for p in BOS]}]}]}})
    assert posted == {"BOS": [2, 3, 4, 5, 7]}
    assert parse_posted_starters({"boxscore": {"teams": []}}) == {}
    pg = toy()
    games = pd.DataFrame([dict(game_id="u1", game_date="2025-01-11", season=2025, home="BOS", away="NYK")])
    live = project_live(pg, games, {"BOS": roster}, injuries).set_index("player_id")
    assert live.loc[1, "status"] == "out" and live.loc[2, "status"] == "available"
    assert set(live.index[live.proj_starter]) == {2, 3, 4, 5, 6}
    assert (live.source == "live").all()
    assert live.proj_minutes.sum() == pytest.approx(TEAM_MINUTES, abs=0.05)
    with_posted = project_live(pg, games, {"BOS": roster}, injuries, {("u1", "BOS"): posted["BOS"]})
    w = with_posted.set_index("player_id")
    assert set(w.index[w.proj_starter]) == {2, 3, 4, 5, 7} and (w.source == "posted").all()


PROCESSED = Path("data/nba/processed")


@pytest.mark.skipif(not (PROCESSED / "player_games.parquet").exists(), reason="no processed NBA player games")
def test_real_projection_beats_previous_game():
    pg = pd.read_parquet(PROCESSED / "player_games.parquet")
    pg = pg[pg.season >= pg.season.max() - 2]
    ev = evaluate(projected_lineups(pg), pg)
    ev = ev[ev.team_games > 500]
    assert (ev.starters > ev.prev_starters).all()
    assert (ev.minutes_mae < ev.prev_minutes_mae).all()
    assert (ev.missed_flagged > 0.5).all()
