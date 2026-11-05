"""Shared vocabularies for game state and play outcomes (single source of truth)."""
from __future__ import annotations

TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB",
    "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]
TEAM_IDX = {t: i for i, t in enumerate(TEAMS)}
# Historical abbreviations mapped to the current franchise.
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS", "LAR": "LA", "JAC": "JAX"}

FIRST_SEASON = 1999
MAX_SEASON = 2040  # embedding table size; seasons beyond clamp

# Game phases
KICKOFF, PAT, SCRIMMAGE = 0, 1, 2
PHASES = ["kickoff", "pat", "scrimmage"]

# Play types
PLAY_TYPES = [
    "kickoff", "run", "pass", "punt", "field_goal", "extra_point", "two_point",
    "qb_kneel", "qb_spike", "penalty", "timeout",
]
PT = {p: i for i, p in enumerate(PLAY_TYPES)}

# Results, each tagged with the play types it may follow.
RESULTS = [
    "run",            # run, normal
    "complete", "incomplete", "sack",
    "interception", "fumble_lost",
    "kneel", "spike",
    "penalty", "penalty_first_down",
    "kick_received",  # punt/kickoff fielded by receiving team
    "kick_recovered", # kicking team keeps the ball (onside, muff)
    "good", "no_good",
    "timeout_offense", "timeout_defense",
]
RS = {r: i for i, r in enumerate(RESULTS)}

RESULTS_FOR_TYPE = {
    "kickoff": ["kick_received", "kick_recovered"],
    "run": ["run", "fumble_lost"],
    "pass": ["complete", "incomplete", "sack", "interception", "fumble_lost"],
    "punt": ["kick_received", "kick_recovered"],
    "field_goal": ["good", "no_good"],
    "extra_point": ["good", "no_good"],
    "two_point": ["good", "no_good"],
    "qb_kneel": ["kneel", "fumble_lost"],
    "qb_spike": ["spike"],
    "penalty": ["penalty", "penalty_first_down"],
    "timeout": ["timeout_offense", "timeout_defense"],
}
TYPES_FOR_PHASE = {
    KICKOFF: ["kickoff", "timeout"],
    PAT: ["extra_point", "two_point", "timeout"],
    SCRIMMAGE: ["run", "pass", "punt", "field_goal", "qb_kneel", "qb_spike", "penalty", "timeout"],
}

# Yards: a single integer field whose meaning depends on the result.
#  * "gain" results: net yards for the offense, in [yl-100, yl]. gain == yl is a touchdown,
#    gain == yl-100 is a safety.
#  * "spot" results (kicks, turnovers, missed FG): yardline_100 of the team holding the ball
#    afterwards, 0 meaning that team scored a touchdown on the return.
#  * "none" results (PAT, FG good, timeouts): always 0.
YARDS_MIN, YARDS_MAX = -99, 100
N_YARDS = YARDS_MAX - YARDS_MIN + 1
SPOT_RESULTS = {"kick_received", "kick_recovered", "interception", "fumble_lost", "no_good"}
NONE_RESULTS = {"good", "timeout_offense", "timeout_defense"}


def yards_kind(play_type: str, result: str) -> str:
    if result in NONE_RESULTS or play_type in ("extra_point", "two_point"):
        return "none"
    if result in SPOT_RESULTS:
        return "spot"
    return "gain"


def yards_to_idx(y: int) -> int:
    return int(min(max(y, YARDS_MIN), YARDS_MAX)) - YARDS_MIN


def idx_to_yards(i: int) -> int:
    return int(i) + YARDS_MIN


# Clock seconds elapsed on a play, 0..MAX_ELAPSED (clamped).
MAX_ELAPSED = 60
N_ELAPSED = MAX_ELAPSED + 1

# Input-feature buckets
N_DOWN = 5          # 0 = n/a, 1..4
N_YTG = 31          # 0..29 yards, 30 = 30+
N_YL = 101          # 0..100 (0 used for n/a)
N_QTR = 7           # 1..4, 5..6 OT (index = min(qtr, 6))
N_QSEC = 61         # quarter seconds remaining bucketed by 15s: 0..60
N_DIFF = 61         # score diff (offense - defense) clamped to [-30, 30]
N_TO = 4            # timeouts 0..3
N_GAME_TYPE = 2     # REG, POST


def bucket_qsec(qsec: float) -> int:
    return int(min(max(qsec, 0), 900) // 15)


def bucket_diff(d: int) -> int:
    return int(min(max(d, -30), 30)) + 30
