"""Rushing and receiving lines attributed by the model's who-gets-the-ball head."""
from __future__ import annotations

from ..data.players import GROUPS
from ..engine.rules import GameState, Outcome
from ..model.lineups import N_SLOTS, SLOTS


def player_stats(history: list[tuple[GameState, Outcome]], lineups: dict | None,
                 names: dict[str, str]) -> dict[str, list[dict]]:
    """Build lines from the current history, so undo removes a play's stats too.

    Unnamed depth-chart slots and off-chart players keep separate, explicit labels.
    Fumble yards encode the turnover spot, not a player's gain; do not count them.
    """
    rows: dict[str, dict[int, dict]] = {"home": {}, "away": {}}
    for state, outcome in history:
        slot = outcome.who
        if slot is None or slot < 0 or slot >= N_SLOTS + len(GROUPS):
            continue
        if outcome.play_type not in ("run", "pass"):
            continue
        side = "home" if state.pos == 0 else "away"
        if slot not in rows[side]:
            if slot < N_SLOTS:
                group, rank = SLOTS[slot]
                role = f"{group}{rank + 1}"
                chart = ((lineups or {}).get(side) or {}).get("depth_chart", {})
                ids = chart.get(group, [])
                pid = ids[rank] if rank < len(ids) else None
                name = "New player" if pid == "NEW" else names.get(pid, pid or f"Other {group}")
            else:
                group = GROUPS[slot - N_SLOTS]
                role, pid, name = group, None, f"Other {group} (off-chart)"
            rows[side][slot] = {
                "slot": slot, "id": pid, "name": name, "role": role,
                "targets": 0, "receptions": 0, "receiving_yards": 0, "receiving_tds": 0,
                "carries": 0, "rushing_yards": 0, "rushing_tds": 0,
            }
        row = rows[side][slot]
        if outcome.play_type == "run":
            row["carries"] += 1
            if outcome.result == "run":
                row["rushing_yards"] += outcome.yards
                row["rushing_tds"] += int(outcome.yards >= state.yl)
        else:
            row["targets"] += 1
            if outcome.result == "complete":
                row["receptions"] += 1
                row["receiving_yards"] += outcome.yards
                row["receiving_tds"] += int(outcome.yards >= state.yl)
    return {side: [team[slot] for slot in sorted(team)] for side, team in rows.items()}
