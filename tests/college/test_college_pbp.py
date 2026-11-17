"""ESPN college play-by-play to engine events (runitback.nba.data.college), on small hand-made games, and
the team keys. The real-data replay check runs when data/<league>/raw exists."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from runitback.nba.data import college
from runitback.nba.league import CBB, WCBB
from runitback.nba.vocab import FREE_THROW, LIVE, TIP

MEN = replace(CBB, min_events=1)
WOMEN = replace(WCBB, min_events=1)
H, A = 1, 2  # ESPN team ids


class Game:
    """Builds ESPN-style rows with running scores, in play order."""

    def __init__(self, period_col="start_period_seconds_remaining"):
        self.rows, self.score, self.col = [], {H: 0, A: 0}, period_col

    def add(self, team, kind, period, clock, text="", pts=0, shot=False, made=False):
        if made:
            self.score[team] += pts
        self.rows.append({
            "game_id": 1, "game_play_number": len(self.rows) + 1, "season": 2025, "season_type": 2,
            "game_date": "2025-01-05", "period_number": period, self.col: clock, "type_text": kind, "text": text,
            "team_id": float(team) if team else float("nan"), "scoring_play": made, "score_value": pts,
            "shooting_play": shot, "home_score": self.score[H], "away_score": self.score[A],
            "home_team_id": H, "away_team_id": A, "home_team_spread": 3.5})
        return self

    def shot(self, team, period, clock, pts=2, made=True):
        return self.add(team, "JumpShot", period, clock, f"x {'made' if made else 'missed'} Jumper"
                        + (" Three Point" if pts == 3 else ""), pts, True, made)

    def ft(self, team, period, clock, made=True):
        return self.add(team, "MadeFreeThrow", period, clock, f"x {'made' if made else 'missed'} Free Throw.",
                        1, True, made)

    def foul(self, team, period, clock):
        return self.add(team, "PersonalFoul", period, clock, "Foul on x.")

    def to(self, team, period, clock):
        return self.add(team, "Lost Ball Turnover", period, clock, "x Turnover.")

    def reb(self, team, period, clock, off=False):
        return self.add(team, f"{'Offensive' if off else 'Defensive'} Rebound", period, clock, "x Rebound.")

    def timeout(self, team, period, clock):
        return self.add(team, "ShortTimeOut", period, clock, "x Timeout")

    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def process(game: Game, lg=MEN, keys=("Duke", "UNC", False)):
    res = college.process_game(game.df(), lg, keys)
    assert res is not None
    meta, rows = res
    ok, total, misses = college.replay_agreement(meta, rows, lg)
    assert ok == total, misses  # the engine reproduces what the rows say, step by step
    return meta, pd.DataFrame(rows)


def kinds(rows: pd.DataFrame) -> list[str]:
    return [f"{t}/{r}" for t, r in zip(rows.play_type, rows.result)]


def test_tip_is_the_first_possession_and_every_period_starts_with_a_toss():
    g = Game().shot(A, 1, 1190, made=True, pts=2).shot(H, 1, 1170, made=False).reb(A, 1, 1165)
    g.shot(A, 1, 3, made=True, pts=2).shot(H, 2, 1180, made=True, pts=3).shot(A, 2, 1.0 * 0, made=False)
    meta, rows = process(g)
    assert meta["tip_winner"] == 1 and meta["home"] == "Duke" and meta["home_final"] == 3 and meta["away_final"] == 4
    k = kinds(rows)
    assert k[0] == "jump_ball/lost" and rows.phase[0] == TIP                   # AWAY won the opening tip
    assert "jump_ball/won" in k[1:] or "jump_ball/lost" in k[1:]                  # the second half starts with a toss
    second = rows[(rows.qtr == 2)].iloc[0]
    assert second.play_type == "jump_ball" and second.phase == TIP and second.qsec == 1200
    assert rows.qtr.max() == 2 and kinds(rows)[-1] == "shot2/missed_def" or rows.play_type.iloc[-1] in ("run_out", "shot2")


def test_a_toss_row_at_the_end_of_the_half_is_the_next_halfs_possession():
    g = Game().shot(H, 1, 1190).shot(A, 1, 4).add(H, "Jumpball", 1, 0, "Jump Ball won by Duke")
    g.add(A, "Jumpball", 1, 0, "Jump Ball lost by UNC").add(None, "End Period", 1, 0, "End of 1st half")
    g.shot(H, 2, 1180).shot(A, 2, 1160)
    _, rows = process(g)
    toss = rows[rows.qtr == 2].iloc[0]
    assert (toss.play_type, toss.qsec, toss.phase) == ("jump_ball", 1200, TIP)
    assert rows.play_type.eq("jump_ball").sum() == 2                             # opening tip and this toss; "lost by" dropped


def test_shooting_fouls_and_one():
    g = Game().shot(H, 1, 1190)
    g.shot(A, 1, 1170, pts=2, made=True).foul(H, 1, 1170).ft(A, 1, 1170, made=True)              # and-one
    g.shot(H, 1, 1150, pts=3, made=False).foul(A, 1, 1150).ft(H, 1, 1150).ft(H, 1, 1150, made=False)
    g.ft(H, 1, 1150, made=True)                                                                  # fouled on a three
    g.shot(A, 1, 1130, made=False).reb(H, 1, 1127)
    _, rows = process(g)
    k = kinds(rows)
    assert "shot2/made_fouled" in k and "shot3/missed_fouled" in k
    assert not any(x.startswith("foul/") for x in k)                                              # absorbed into the shots
    i = k.index("shot3/missed_fouled")
    assert k[i + 1:i + 4] == ["free_throw/made", "free_throw/missed", "free_throw/made"]
    assert list(rows.fts_left[i + 1:i + 4]) == [3, 2, 1]
    assert rows.foul_a.iloc[-1] == 1 and rows.foul_h.iloc[-1] == 1                                # each team's foul counted once


def test_common_foul_without_free_throws_is_an_event_and_counts_toward_the_bonus():
    g = Game().shot(H, 1, 1190)
    for i in range(6):                                                                            # 6 fouls on Away, none sends to the line
        g.shot(A, 1, 1180 - 40 * i, made=False).reb(H, 1, 1178 - 40 * i)
        g.foul(A, 1, 1170 - 40 * i).to(H, 1, 1165 - 40 * i)
    g.shot(A, 1, 800, made=False).reb(H, 1, 798)
    g.foul(A, 1, 700).ft(H, 1, 700, made=True).ft(H, 1, 700, made=True)                          # 7th: the 1-and-1, made first
    g.shot(A, 1, 650, made=False).reb(H, 1, 648)
    g.foul(A, 1, 600).ft(H, 1, 600, made=False).reb(A, 1, 598)                                    # 8th: first shot missed
    _, rows = process(g)
    k = kinds(rows)
    assert k.count("foul/fts_0") == 6 and k.count("foul/fts_2") == 2
    first = rows[rows.play_type == "foul"].iloc[6]
    assert first.foul_a == 6 and first.result == "fts_2"                                          # the 7th foul sends them to the line
    shots = rows[rows.play_type == "free_throw"]
    assert list(shots.oo) == [1, 0, 1] and list(shots.fts_left) == [2, 1, 2]                      # 1-and-1, then the 8th's first shot
    assert list(shots.result) == ["made", "made", "missed_def"]
    assert shots.foul_a.iloc[-1] == 8


def test_double_bonus_is_two_shots_and_fouls_reset_in_the_second_half():
    g = Game().shot(A, 1, 1190, made=False).reb(H, 1, 1188)
    for i in range(10):
        g.foul(A, 1, 1180 - 100 * i)
        if i >= 9:
            g.ft(H, 1, 1180 - 100 * i).ft(H, 1, 1180 - 100 * i)
    g.shot(A, 1, 100, made=False).reb(H, 1, 98)
    g.shot(H, 2, 1190, made=False).reb(H, 2, 1188, off=True).foul(A, 2, 1100).shot(H, 2, 1050)
    _, rows = process(g)
    f = rows[rows.play_type == "foul"]
    assert list(f.result)[-2:] == ["fts_2", "fts_0"] and f.foul_a.tolist()[:10] == list(range(10))
    ft = rows[rows.play_type == "free_throw"]
    assert list(ft.oo) == [0, 0] and list(ft.fts_left) == [2, 1]
    assert f.iloc[-1].foul_a == 0                                                                  # the count restarted


def test_offensive_and_team_control_fouls_are_turnovers_and_technicals_keep_the_ball():
    g = Game().shot(H, 1, 1190)
    g.foul(A, 1, 1150).to(A, 1, 1150)                                                             # offensive foul + turnover row
    g.shot(H, 1, 1100, made=False).reb(A, 1, 1098)
    g.add(H, "Technical Foul", 1, 1050, "Technical Foul on x.").ft(A, 1, 1050).ft(A, 1, 1050)    # Away shoots two
    g.shot(A, 1, 1030, made=True)
    meta, rows = process(g)
    k = kinds(rows)
    assert k.count("turnover/turnover") == 1 and not any(x.startswith("foul/") for x in k)
    assert k.count("technical/tech_offense") == 2
    assert rows.foul_a.iloc[-1] == 0 and rows.foul_h.iloc[-1] == 0                                # neither is a team foul


def test_intentional_foul_that_keeps_the_ball_is_a_foul_then_technical_shots():
    g = Game().shot(H, 1, 1190)
    g.shot(A, 1, 1170, made=False).reb(H, 1, 1168)
    g.foul(H, 1, 1100).ft(A, 1, 1100).ft(A, 1, 1100)                                              # two made, then Away shoots again
    g.shot(A, 1, 1080, made=True)
    _, rows = process(g)
    k = kinds(rows)
    i = k.index("foul/fts_0")
    assert k[i + 1:i + 3] == ["technical/tech_offense"] * 2 and k[i + 3] == "shot2/made"
    assert rows.foul_h.iloc[-1] == 1                                                              # but it was a foul on Home


def test_free_throw_trips_skip_the_dead_ball_rebound_and_timeouts_are_counted():
    g = Game().shot(H, 1, 1190)
    g.shot(A, 1, 1170, made=False).foul(H, 1, 1170).ft(A, 1, 1170, made=False).reb(A, 1, 1170, off=True)
    g.ft(A, 1, 1170, made=True).timeout(H, 1, 1100).shot(H, 1, 1090)
    _, rows = process(g)
    k = kinds(rows)
    assert k.count("free_throw/missed") == 1 and k.count("free_throw/made") == 1
    assert "timeout/timeout_offense" in k or "timeout/timeout_defense" in k
    t = rows[rows.play_type == "timeout"].iloc[0]
    assert (t.to_h, t.to_a) == (4, 4) and rows.to_h.iloc[-1] == 3


def test_women_use_the_quarter_clock_and_reset_fouls_every_quarter():
    g = Game("start_quarter_seconds_remaining").shot(A, 1, 590, made=False).reb(H, 1, 588)
    g.foul(A, 1, 500).foul(A, 1, 400).foul(A, 1, 300)
    g.shot(H, 2, 590, made=False).reb(H, 2, 588, off=True).foul(A, 2, 500)
    meta, rows = process(g, WOMEN)
    f = rows[rows.play_type == "foul"]
    assert list(f.foul_a) == [0, 1, 2, 0] and set(rows.qtr) == {1, 2} and rows.qsec[0] == 600


def test_non_division_one_pairs_and_preseason_games_are_dropped():
    g = Game().shot(H, 1, 1190)
    assert college.process_game(g.df(), MEN, ("Non-D1", "Non-D1", False)) is None
    assert college.process_game(g.df(), MEN, ("Duke", "Non-D1", False)) is not None
    pre = g.df().assign(season_type=1)
    assert college.process_game(pre, MEN, ("Duke", "UNC", False)) is None
    assert college.process_game(g.df(), CBB, ("Duke", "UNC", False)) is None  # fewer events than CBB.min_events


def test_team_keys_pool_non_division_one_opponents():
    sched = pd.DataFrame({
        "game_id": [1, 2, 3], "season": [2024, 2025, 2025], "neutral_site": [False, True, False],
        "game_date": ["2024-01-10T00:00Z", "2025-01-10T00:00Z", "2025-01-12T00:00Z"],
        "home_id": [10, 10, 50], "away_id": [20, 30, 40],
        "home_location": ["Duke", "Duke Blue", "Tiny College"], "away_location": ["UNC", "Elon", "Other"],
        "home_conference_id": [2.0, 2.0, None], "away_conference_id": [2.0, 3.0, None],
    })
    t = college.team_table(sched).set_index("game_id")
    assert t.loc["1", "home"] == "Duke Blue"        # the school's latest name keeps one history
    assert t.loc["2", "away"] == "Elon" and bool(t.loc["2", "neutral"])
    assert t.loc["3", "home"] == t.loc["3", "away"] == college.NON_D1
    # a school ESPN files under no conference that season is the pooled team in that season's games only
    sched.loc[2, ["home_conference_id", "away_conference_id"]] = [2.0, None]
    t = college.team_table(sched).set_index("game_id")
    assert t.loc["3", "away"] == college.NON_D1 and t.loc["2", "away"] == "Elon"


RAW = Path(__file__).resolve().parents[2] / "data"


@pytest.mark.parametrize("name", ["cbb", "wcbb"])
@pytest.mark.skipif(not (RAW / "cbb" / "raw" / "pbp_2025.parquet").exists(), reason="college data not downloaded")
def test_real_season_replays_at_the_target_agreement(name, tmp_path):
    lg = {"cbb": CBB, "wcbb": WCBB}[name]
    stats = college.preprocess(lg, [2025], RAW / name / "raw", tmp_path)
    assert stats["agreement"] >= 0.99 and stats["games"] > 4000
    assert stats["events_per_game"]["max"] <= 512
