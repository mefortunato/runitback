from runitback.nfl.data.players import GROUPS
from runitback.nfl.engine.rules import GameState, Outcome
from runitback.nfl.model.lineups import N_SLOTS, SLOT_IDX
from runitback.nfl.sim.stats import player_stats
from runitback.nfl.vocab import SCRIMMAGE


def state(pos=0, yl=20):
    return GameState(2025, "REG", "KC", "BUF", phase=SCRIMMAGE, pos=pos, yl=yl)


def test_rushing_receiving_and_touchdowns_are_attributed_to_the_possession_team():
    qb, rb, wr = (SLOT_IDX[(g, 0)] for g in ("QB", "RB", "WR"))
    history = [
        (state(), Outcome("run", "run", -4, who=qb)),  # slot zero is a valid player
        (state(), Outcome("run", "run", 20, who=rb)),
        (state(), Outcome("pass", "complete", 20, who=wr)),
        (state(), Outcome("pass", "incomplete", who=wr)),
        (state(), Outcome("pass", "interception", 0, who=wr)),  # return TD is not a receiving TD
        (state(1), Outcome("pass", "complete", -2, who=wr)),
        (state(1), Outcome("run", "run", 7, who=rb)),
    ]
    lineups = {"home": {"depth_chart": {"RB": ["rusher"], "WR": ["receiver"]}},
               "away": {"depth_chart": {"WR": ["opponent"]}}}
    stats = player_stats(history, lineups, {"rusher": "Runner", "receiver": "Receiver", "opponent": "Opponent"})
    home = {p["slot"]: p for p in stats["home"]}
    away = {p["slot"]: p for p in stats["away"]}
    assert home[qb]["carries"] == 1 and home[qb]["rushing_yards"] == -4
    assert home[rb]["name"] == "Runner" and home[rb]["rushing_tds"] == 1
    assert home[wr]["targets"] == 3 and home[wr]["receptions"] == 1
    assert home[wr]["receiving_yards"] == 20 and home[wr]["receiving_tds"] == 1
    assert away[wr]["name"] == "Opponent" and away[wr]["receiving_yards"] == -2
    assert away[wr]["receiving_tds"] == 0 and away[rb]["rushing_yards"] == 7


def test_fumble_spots_and_unattributed_plays_do_not_become_player_yards():
    rb, wr = SLOT_IDX[("RB", 0)], SLOT_IDX[("WR", 0)]
    stats = player_stats([
        (state(), Outcome("run", "fumble_lost", 90, who=rb)),
        (state(), Outcome("pass", "fumble_lost", 80, who=wr)),
        (state(), Outcome("pass", "sack", -8)),
        (state(), Outcome("pass", "incomplete")),  # throwaway
        (state(), Outcome("qb_kneel", "run", -1, who=rb)),
        (state(), Outcome("kickoff", "kick_received", 0, who=wr)),
        (state(), Outcome("run", "run", 10, who=-1)),
    ], None, {})
    lines = {p["slot"]: p for p in stats["home"]}
    assert set(lines) == {rb, wr}
    assert lines[rb]["carries"] == 1 and lines[rb]["rushing_yards"] == 0
    assert lines[wr]["targets"] == 1 and lines[wr]["receptions"] == 0
    assert lines[wr]["receiving_yards"] == 0 and lines[rb]["rushing_tds"] == 0
    assert stats["away"] == []


def test_new_empty_and_off_chart_slots_stay_distinct():
    slots = [SLOT_IDX[("RB", rank)] for rank in range(3)] + [N_SLOTS + GROUPS.index("RB")]
    stats = player_stats([(state(), Outcome("run", "run", 3, who=slot)) for slot in slots],
                         {"home": {"depth_chart": {"RB": ["NEW", "NEW"]}}}, {})
    lines = stats["home"]
    assert [p["slot"] for p in lines] == slots
    assert [p["name"] for p in lines] == ["New player", "New player", "Other RB", "Other RB (off-chart)"]
    assert [p["role"] for p in lines] == ["RB1", "RB2", "RB3", "RB"]
    assert all(p["carries"] == 1 and p["rushing_yards"] == 3 for p in lines)
