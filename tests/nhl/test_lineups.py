from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from runitback.nhl.data.events import AWAY_SKATERS, HOME_SKATERS
from runitback.nhl.data.lineups import (
    build_lines,
    build_lineups,
    evaluate,
    lineup,
    played_lineups,
    projected_lineups,
    team_games,
    usual_starter,
)

F = list(range(101, 113))        # home forwards: lines (101-103), (104-106), (107-109), (110-112)
D = list(range(201, 207))        # home defense: pairs (201, 202), (203, 204), (205, 206)
OF = list(range(301, 313))       # away
OD = list(range(401, 407))
SECONDS = [60, 45, 30, 15]        # per shift of line 1..4 (pair k with line k; pair 3 also with line 4)
GROUPS = pd.Series({**{p: "F" for p in F + OF + [113]}, **{p: "D" for p in D + OD}, 1: "G", 2: "G", 9: "G"})


def game_events(gid, forwards=F, defense=D, goalie=1, away_goalie=9, swap=None, pp=False):
    """A game of repeated shifts: home line k with pair min(k, 3), away likewise, at 5v5. `swap`:
    (out, in) replaces a home skater. `pp`: one home power-play segment (4 forwards + 1 D) first."""
    fw = list(forwards)
    if swap:
        fw = [swap[1] if p == swap[0] else p for p in fw]
    rows, t = [], 0
    if pp:
        rows.append(dict(home=fw[:4] + defense[:1], away=OF[:3] + OD[:1], state="5v4", dt=120))
    for _ in range(10):
        for k, sec in enumerate(SECONDS):
            pair = min(k, 2)
            rows.append(dict(home=fw[3 * k:3 * k + 3] + defense[2 * pair:2 * pair + 2],
                             away=OF[3 * k:3 * k + 3] + OD[2 * pair:2 * pair + 2], state="5v5", dt=sec))
    out = []
    for i, r in enumerate(rows):
        d = dict(game_id=gid, event_idx=i, game_seconds=t, period_type="REGULAR", strength_state=r["state"],
                 home_goalie=goalie, away_goalie=away_goalie)
        for j, c in enumerate(HOME_SKATERS):
            d[c] = sorted(r["home"])[j] if j < len(r["home"]) else pd.NA
        for j, c in enumerate(AWAY_SKATERS):
            d[c] = sorted(r["away"])[j] if j < len(r["away"]) else pd.NA
        out.append(d)
        t += r["dt"]
    df = pd.DataFrame(out)
    for c in HOME_SKATERS + AWAY_SKATERS + ["home_goalie", "away_goalie"]:
        df[c] = df[c].astype("Int64")
    return df


DATES = {1: "2025-01-01", 2: "2025-01-03", 3: "2025-01-04", 4: "2025-01-07"}


def games_table(dates=DATES):
    return pd.DataFrame([dict(game_id=g, season=2025, game_type="REG", game_date=d, home="BOS", away="TOR")
                         for g, d in dates.items()])


def toy():
    """Game 3 (the second night of a back-to-back): the backup (2) starts and 113 replaces 112."""
    ev = pd.concat([game_events(1), game_events(2, pp=True), game_events(3, goalie=2, swap=(112, 113)),
                    game_events(4)], ignore_index=True)
    return played_lineups(ev, games_table(), GROUPS)


def test_build_lines_groups_by_shared_time():
    toi = {1: 10, 2: 9, 3: 6, 4: 5, 5: 8, 6: 7, 7: 1}
    shared = {(1, 5): 9, (1, 6): 8, (2, 3): 7, (2, 4): 6, (1, 2): 1}
    lines = build_lines(list(toi), toi, shared, 3)
    assert lines[1] == lines[5] == lines[6] == 1
    assert lines[2] == lines[3] == lines[4] == 2
    assert lines[7] == 3     # an odd player out makes a short last line


def test_played_lineups_rebuild_lines_toi_and_starter():
    p = toy()
    g1 = p[(p.game_id == 1) & (p.side == "home")].set_index("player_id")
    assert len(g1) == 12 + 6 + 1
    assert [g1.loc[F[3 * k:3 * k + 3], "line"].unique().tolist() for k in range(4)] == [[1], [2], [3], [4]]
    assert [g1.loc[D[2 * k:2 * k + 2], "line"].unique().tolist() for k in range(3)] == [[1], [2], [3]]
    assert g1.loc[101, "toi"] == pytest.approx(10 * 60 / 60)
    assert g1.loc[101, "slot"] == 1 and g1.loc[1, "started"] and g1.loc[1, "group"] == "G"
    # the last event's duration is 0, so the last line's ten shifts give 9 * 15 s
    assert g1.loc[205, "toi"] == pytest.approx((10 * 30 + 9 * 15) / 60)
    g2 = p[(p.game_id == 2) & (p.side == "home")].set_index("player_id")
    assert g2.loc[101, "toi"] > g2.loc[101, "ev_toi"]          # the power play is not 5-on-5
    assert g2.loc[101, "line"] == 1                             # lines come from 5-on-5 only
    a1 = p[(p.game_id == 1) & (p.side == "away")]
    assert set(a1.team) == {"TOR"} and a1.player_id.isin(OF + OD + [9]).all()


def test_projection_is_the_previous_game():
    played = toy()
    proj = projected_lineups(played)
    assert 1 not in set(proj.game_id)          # nobody's previous game
    g3 = proj[(proj.game_id == 3) & (proj.team == "BOS")].set_index("player_id")
    assert (g3.prev_game_id == 2).all() and (g3.rest_days == 1).all()
    assert 112 in g3.index and 113 not in g3.index       # the skaters are game 2's
    assert g3.loc[101, "line"] == 1 and g3.loc[110, "line"] == 4
    # goalies: (a) game 2's starter, (b) the actual starter, each on its own row
    goalies = g3[g3.group == "G"]
    assert set(goalies.index) == {1, 2}
    assert goalies.loc[1, "goalie_prev"] and not goalies.loc[1, "goalie_actual"]
    assert goalies.loc[2, "goalie_actual"] and not goalies.loc[2, "goalie_prev"]
    assert goalies.loc[2, "prev_toi"] == pytest.approx(0) or np.isnan(goalies.loc[2, "prev_toi"])
    g4 = proj[(proj.game_id == 4) & (proj.team == "BOS")].set_index("player_id")
    assert 113 in g4.index and 112 not in g4.index
    g = g4[g4.group == "G"]
    assert g.loc[2, "goalie_prev"] and g.loc[1, "goalie_actual"]       # back to the starter
    assert g.loc[1, "prev_toi"] > 0                                    # his TOI in his last game (2)
    g2 = proj[(proj.game_id == 2) & (proj.team == "BOS") & (proj.group == "G")]
    assert g2.player_id.tolist() == [1] and g2.goalie_prev.all() and g2.goalie_actual.all()  # one row when (a) = (b)


def test_projection_uses_nothing_from_the_game():
    """Changing game 3 (its skaters, starter and TOI) changes nothing in game 3's projected skaters
    or (a) goalie; only (b) follows the actual starter."""
    base = projected_lineups(toy())
    ev = pd.concat([game_events(1), game_events(2, pp=True),
                    game_events(3, goalie=1, forwards=F[::-1], swap=(101, 113), pp=True), game_events(4)],
                   ignore_index=True)
    alt = projected_lineups(played_lineups(ev, games_table(), GROUPS))
    pick = lambda d: d[(d.game_id == 3) & ~d.goalie_actual | (d.game_id == 3) & d.goalie_prev] \
        .sort_values(["team", "player_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(pick(base).drop(columns="goalie_actual"), pick(alt).drop(columns="goalie_actual"))


def test_lineup_picks_one_goalie():
    proj = projected_lineups(toy())
    a = lineup(proj, "prev")
    b = lineup(proj, "actual")
    for d in (a, b):
        assert (d[d.group == "G"].groupby(["game_id", "team"]).size() == 1).all()
        assert len(d[(d.game_id == 3) & (d.team == "BOS")]) == 19
    assert a[(a.game_id == 3) & (a.team == "BOS") & (a.group == "G")].player_id.tolist() == [1]
    assert b[(b.game_id == 3) & (b.team == "BOS") & (b.group == "G")].player_id.tolist() == [2]
    assert not a.known_at_drop.any()
    assert b[b.group == "G"].known_at_drop.all() and not b[b.group != "G"].known_at_drop.any()
    with pytest.raises(ValueError):
        lineup(proj, "posted")


def test_team_games_rest_days():
    tg = team_games(toy())
    bos = tg[tg.team == "BOS"].set_index("game_id")
    assert np.isnan(bos.loc[1, "rest_days"]) and bos.loc[3, "rest_days"] == 1 and bos.loc[4, "prev_game_id"] == 3


def test_evaluate():
    played = toy()
    ev = evaluate(projected_lineups(played), played).loc[2025]
    assert ev.team_games == 6                        # games 2-4, both teams
    # BOS: 18/18, 17/18 (113 new), 17/18 (112 back); TOR: all 18
    assert ev.skaters == pytest.approx((18 * 4 + 17 * 2) / (18 * 6))
    assert ev.all18 == pytest.approx(4 / 6)
    assert ev.goalie_prev == pytest.approx(4 / 6)    # BOS games 3 (backup) and 4 (starter back) miss
    assert ev.goalie_prev_b2b == pytest.approx(1 / 2) and ev.goalie_changed == pytest.approx(2 / 6)
    assert ev.f_line == 1.0 and ev.d_pair == 1.0
    # the reference rule: game 3 (back-to-back) -> not the previous starter, but 1 is the only other
    # starter so far, so it stays 1 (wrong); game 4 -> most starts (1) is right
    assert ev.goalie_usual == pytest.approx(5 / 6)


def test_usual_starter_flips_on_back_to_back():
    starts = [(1, "2025-01-01", 1), (2, "2025-01-03", 2), (3, "2025-01-05", 1), (4, "2025-01-06", 1),
              (5, "2025-01-08", 1), (6, "2025-01-09", 1)]
    played = pd.DataFrame([dict(game_id=g, game_date=d, team="BOS", player_id=p, group="G", started=True)
                           for g, d, p in starts])
    u = usual_starter(played).set_index("game_id").usual_g
    assert u[2] == 1 and u[3] == 2          # the most starts so far (ties: the most recent)
    assert u[4] == 2 and u[6] == 2          # back-to-backs: the other goalie
    assert u[5] == 1

def test_build_lineups(tmp_path):
    ev = pd.concat([game_events(g) for g in DATES], ignore_index=True)
    ev.to_parquet(tmp_path / "events_2025.parquet")
    games_table().to_parquet(tmp_path / "games.parquet")
    GROUPS.rename("group").rename_axis("player_id").reset_index().to_parquet(tmp_path / "players.parquet")
    summary = build_lineups(tmp_path)
    assert summary[2025]["skaters"] == 1.0 and summary[2025]["goalie_prev"] == 1.0
    assert (tmp_path / "projected_lineups.parquet").exists() and (tmp_path / "played_lineups.parquet").exists()


PROCESSED = Path("data/nhl/processed")


@pytest.mark.skipif(not (PROCESSED / "events_2025.parquet").exists(), reason="no processed NHL events")
def test_real_season():
    games = pd.read_parquet(PROCESSED / "games.parquet")
    groups = pd.read_parquet(PROCESSED / "players.parquet").set_index("player_id").group
    ev = pd.read_parquet(PROCESSED / "events_2025.parquet")
    played = played_lineups(ev, games, groups)
    per = played.groupby(["game_id", "side"])
    assert per.group.agg(lambda g: g.ne("G").sum()).median() == 18
    assert (per.started.sum() == 1).mean() > 0.99
    r = evaluate(projected_lineups(played), played).loc[2025]
    assert r.skaters > 0.9 and 0.35 < r.goalie_prev < 0.75 and r.f_line > 0.4 and r.goalie_usual > r.goalie_prev
