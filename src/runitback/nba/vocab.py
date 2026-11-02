"""NBA vocabularies for game state and possession events (single source of truth).

The NBA is modeled one *event* at a time. An event is something the team with the ball does
that the rules engine needs to know about: a shot, a free throw, a turnover, a foul that sends
someone to the line, a technical free throw, a timeout, a jump ball, or holding the ball until
the period ends. Rebounds are folded into the missed shot or free throw they follow
(`missed_off` / `missed_def`), so possession always follows from the outcome.
"""
from __future__ import annotations

TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND",
    "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX",
    "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]
TEAM_IDX = {t: i for i, t in enumerate(TEAMS)}
# ESPN and historical abbreviations mapped to the current franchise.
TEAM_ALIASES = {
    "GS": "GSW", "NO": "NOP", "NOH": "NOP", "NOK": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA",
    "WSH": "WAS", "NJ": "BKN", "NJN": "BKN", "BRK": "BKN", "SEA": "OKC", "VAN": "MEM", "CHO": "CHA",
    "CHH": "CHA", "PHO": "PHX", "GOS": "GSW",
}
TEAM_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets", "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons", "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors", "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}


def canon_team(abbrev: str) -> str:
    return TEAM_ALIASES.get(abbrev, abbrev)


# Seasons are named by the year they end in (2025 = the 2024-25 season), as in the ESPN data.
FIRST_SEASON = 2002
MAX_SEASON = 2040  # embedding table size; seasons beyond clamp

QUARTER_SECONDS = 720
OT_SECONDS = 300
TIMEOUTS_PER_GAME = 7
MAX_Q4_TIMEOUTS = 4
OT_TIMEOUTS = 2

# Game phases
TIP, LIVE, FREE_THROW = 0, 1, 2
PHASES = ["tip", "live", "free_throw"]

PLAY_TYPES = [
    "jump_ball", "shot2", "shot3", "free_throw", "turnover", "foul", "technical", "timeout", "run_out",
]
PT = {p: i for i, p in enumerate(PLAY_TYPES)}

RESULTS = [
    "won", "lost",                                   # jump ball: the team with the ball keeps it or not
    "made", "made_fouled",                           # shots (made_fouled = and-one)
    "missed_off", "missed_def", "missed_fouled",     # shots; missed_off / missed_def also end free throws
    "missed",                                        # a free throw that is not the last of the trip
    "turnover",
    "fts_1", "fts_2", "fts_3",                       # non-shooting foul on the defense: free throws awarded
    "tech_offense", "tech_defense", "tech_missed",   # technical free throw: who scored, or a miss
    "timeout_offense", "timeout_defense",
    "end",                                           # run_out: the team holds the ball until the period ends
]
RS = {r: i for i, r in enumerate(RESULTS)}

SHOT_RESULTS = ["made", "made_fouled", "missed_off", "missed_def", "missed_fouled"]
RESULTS_FOR_TYPE = {
    "jump_ball": ["won", "lost"],
    "shot2": SHOT_RESULTS,
    "shot3": SHOT_RESULTS,
    "free_throw": ["made", "missed", "missed_off", "missed_def"],
    "turnover": ["turnover"],
    "foul": ["fts_1", "fts_2", "fts_3"],
    "technical": ["tech_offense", "tech_defense", "tech_missed"],
    "timeout": ["timeout_offense", "timeout_defense"],
    "run_out": ["end"],
}
TYPES_FOR_PHASE = {
    TIP: ["jump_ball"],
    LIVE: ["shot2", "shot3", "turnover", "foul", "technical", "timeout", "jump_ball", "run_out"],
    FREE_THROW: ["free_throw", "technical", "timeout"],
}

# Clock seconds elapsed before an event, 0..MAX_ELAPSED (clamped). A possession is at most a
# 24-second shot clock, but side-out fouls and offensive rebounds reset it without an event.
MAX_ELAPSED = 60
N_ELAPSED = MAX_ELAPSED + 1

# Input-feature buckets
N_QTR = 7           # index min(qtr, 6): 1..4, 5 = OT, 6 = 2OT and later
N_QSEC = 49         # quarter seconds remaining bucketed by 15s: 0..48
N_DIFF = 81         # score diff (offense - defense) clamped to [-40, 40]
N_TO = 8            # timeouts 0..7
N_FTS = 4           # free throws left 0..3
N_GAME_TYPE = 2     # REG, POST


def bucket_qsec(qsec: float) -> int:
    return int(min(max(qsec, 0), QUARTER_SECONDS) // 15)


def bucket_diff(d: int) -> int:
    return int(min(max(d, -40), 40)) + 40
