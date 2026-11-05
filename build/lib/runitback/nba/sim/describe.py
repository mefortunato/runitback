"""Human-readable NBA event descriptions from (state, outcome, next state)."""
from __future__ import annotations

from ..engine.rules import GameState, Outcome
from ..vocab import TIP


def period_name(q: int) -> str:
    if q <= 4:
        return ["", "1st quarter", "2nd quarter", "3rd quarter", "4th quarter"][q]
    return "overtime" if q == 5 else f"{q - 4}OT"


def describe(s: GameState, o: Outcome, n: GameState) -> str:
    off, dfn = s.off_team, s.def_team
    t, r = o.play_type, o.result
    if t == "jump_ball":
        winner = off if r == "won" else dfn
        if s.phase == TIP:
            return f"{winner} wins the tip-off." if s.qtr == 1 else f"{winner} wins the overtime tip."
        return f"Jump ball, {winner} gains possession."
    if t in ("shot2", "shot3"):
        what = "a 2-pointer" if t == "shot2" else "a 3-pointer"
        return {
            "made": f"{off} hits {what}.",
            "made_fouled": f"{off} hits {what} and draws the foul. And-one!",
            "missed_off": f"{off} misses {what}; {off} grabs the offensive rebound.",
            "missed_def": f"{off} misses {what}; {dfn} rebounds.",
            "missed_fouled": f"{off} is fouled on {what}. {2 if t == 'shot2' else 3} free throws.",
        }[r]
    if t == "free_throw":
        k = s.fts_total - s.fts_left + 1 if s.fts_total else 1
        of = f"{k} of {s.fts_total}" if s.fts_total else ""
        return {
            "made": f"{off} makes free throw {of}.",
            "missed": f"{off} misses free throw {of}.",
            "missed_off": f"{off} misses free throw {of}; {off} rebounds.",
            "missed_def": f"{off} misses free throw {of}; {dfn} rebounds.",
        }[r].replace("  ", " ").replace(" .", ".").replace(" ;", ";")
    if t == "turnover":
        return f"{off} turns it over. {dfn} ball."
    if t == "foul":
        return f"Foul on {dfn}. {off} shoots {r[-1]} free throw{'s' if r[-1] != '1' else ''}."
    if t == "technical":
        if r == "tech_missed":
            return "Technical free throw missed."
        return f"{off if r == 'tech_offense' else dfn} makes a technical free throw."
    if t == "timeout":
        return f"Timeout {off if r == 'timeout_offense' else dfn}."
    if t == "run_out":
        if n.over:
            return f"{off} runs out the clock. Final."
        return f"{off} holds for the last shot. End of the {period_name(s.qtr)}."
    return f"{off} {t} {r}."
