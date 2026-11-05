"""Human-readable play descriptions from (state, outcome)."""
from __future__ import annotations

from ..engine.rules import GameState, Outcome
from ..vocab import KICKOFF


def spot_name(team: str, opp: str, yl: int) -> str:
    if yl == 50:
        return "the 50"
    return f"{opp} {yl}" if yl < 50 else f"{team} {100 - yl}"


def _yds(n: int) -> str:
    return f"{n} yard{'s' if abs(n) != 1 else ''}"


def describe(s: GameState, o: Outcome, n: GameState) -> str:
    off, dfn = s.off_team, s.def_team
    t, r, y = o.play_type, o.result, o.yards
    scored = [n.score[i] - s.score[i] for i in (0, 1)]
    td_team = None
    for i, d in enumerate(scored):
        if d == 6:
            td_team = n.home if i == 0 else n.away
    td = f" TOUCHDOWN {td_team}!" if td_team else ""

    def landed(holder: str, other: str, yl: int) -> str:
        return " and it is returned for a touchdown" if yl <= 0 else f", ball at the {spot_name(holder, other, yl)}"

    if t == "timeout":
        return f"Timeout {off if r == 'timeout_offense' else dfn}."
    if s.phase == KICKOFF:
        if r == "kick_recovered":
            return f"{off} kicks off; {off} RECOVERS{landed(off, dfn, y)}.{td}"
        return f"{off} kicks off to {dfn}{landed(dfn, off, y)}.{td}"
    if t in ("extra_point", "two_point"):
        what = "Extra point" if t == "extra_point" else "Two-point attempt"
        return f"{off} {what} is {'GOOD' if r == 'good' else 'NO GOOD'}."
    if t == "field_goal":
        dist = s.yl + 17
        if r == "good":
            return f"{off} {dist}-yard field goal is GOOD."
        return f"{off} {dist}-yard field goal is NO GOOD{landed(dfn, off, y)}.{td}"
    if t == "punt":
        if r == "kick_recovered":
            return f"{off} punts; muffed, {off} recovers{landed(off, dfn, y)}.{td}"
        return f"{off} punts to {dfn}{landed(dfn, off, y)}.{td}"
    if r in ("interception", "fumble_lost"):
        what = "Pass INTERCEPTED" if r == "interception" else "FUMBLE, lost"
        if y <= 0:
            return f"{off} {what}, returned for a touchdown!{td}"
        return f"{off} {what}! {dfn} takes over{landed(dfn, off, y)}.{td}"
    if n.score[1 - s.pos] - s.score[1 - s.pos] == 2:
        return f"{off} {t} for {_yds(y)}. SAFETY!"
    verb = {
        "run": f"run for {_yds(y)}", "complete": f"pass complete for {_yds(y)}",
        "incomplete": "pass incomplete", "sack": f"QB sacked for {_yds(y)}",
        "kneel": "QB kneels", "spike": "QB spikes the ball",
        "penalty": f"penalty, {_yds(abs(y))} {'against the defense' if y > 0 else 'against the offense'}",
        "penalty_first_down": f"penalty on the defense, {_yds(y)}, automatic first down",
    }.get(r, r)
    tail = ""
    if not td and n.pos != s.pos and n.phase != KICKOFF:
        tail = " Turnover on downs."
    elif not td and n.down == 1 and n.pos == s.pos and t != "penalty" or r == "penalty_first_down":
        tail = " First down."
    return f"{off} {verb}.{td}{tail}"
