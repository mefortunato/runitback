"""NHL vocabularies for game state and events (single source of truth).

The NHL is modeled one *event* at a time. Unlike football and basketball there is no possession
to track: each event belongs to the team that did it (`side`), or to nobody (a stoppage, the end of
a period). Events are faceoffs, shot attempts (goal / saved / missed / blocked), hits, giveaways,
takeaways, penalties, delayed-penalty calls, stoppages, goalie pulls and returns, the end of a
period, and shootout attempts.

The rules engine (`engine/rules.py`) owns everything the events imply: the score, which phase the
game is in (a faceoff to restart play, live play, a penalty shot, a shootout), the penalty box and
therefore the skaters on the ice, empty nets, overtime and the shootout.
"""
from __future__ import annotations

TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET", "EDM", "FLA", "LAK",
    "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL",
    "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
]
TEAM_IDX = {t: i for i, t in enumerate(TEAMS)}
# Historical and alternate abbreviations mapped to the current franchise: the Thrashers moved to
# Winnipeg (2011-12); Phoenix became Arizona (2014-15), whose hockey operations moved to Utah
# (2024-25). ESPN-style two-letter codes appear in other sources.
TEAM_ALIASES = {
    "ATL": "WPG", "PHX": "UTA", "ARI": "UTA", "UTAH": "UTA", "LA": "LAK", "NJ": "NJD", "SJ": "SJS",
    "TB": "TBL", "L.A": "LAK", "N.J": "NJD", "S.J": "SJS", "T.B": "TBL", "MON": "MTL", "WAS": "WSH",
    "CLS": "CBJ", "NAS": "NSH", "CAL": "CGY", "VEG": "VGK",
}
TEAM_NAMES = {
    "ANA": "Anaheim Ducks", "BOS": "Boston Bruins", "BUF": "Buffalo Sabres", "CAR": "Carolina Hurricanes",
    "CBJ": "Columbus Blue Jackets", "CGY": "Calgary Flames", "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche", "DAL": "Dallas Stars", "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers", "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild",
    "MTL": "Montreal Canadiens", "NJD": "New Jersey Devils", "NSH": "Nashville Predators",
    "NYI": "New York Islanders", "NYR": "New York Rangers", "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers", "PIT": "Pittsburgh Penguins", "SEA": "Seattle Kraken",
    "SJS": "San Jose Sharks", "STL": "St. Louis Blues", "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs", "UTA": "Utah Mammoth", "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights", "WPG": "Winnipeg Jets", "WSH": "Washington Capitals",
}


def canon_team(abbrev: str) -> str:
    return TEAM_ALIASES.get(abbrev, abbrev)


# Seasons are named by the year they end in (2025 = the 2024-25 season). The sportsdataverse
# `nhl_pbp_full` file names are not consistent about this, so read the `season` column instead.
FIRST_SEASON = 2011
MAX_SEASON = 2040  # embedding table size; seasons beyond clamp

PERIOD_SECONDS = 1200
REG_OT_SECONDS = 300       # regular-season overtime; playoff overtime periods are full 20 minutes
FIRST_3V3_SEASON = 2016    # 3-on-3 regular-season overtime from 2015-16 (4-on-4 before)
FIRST_SHOOTOUT_SEASON = 2006
SHOOTOUT_ROUNDS = 3

SKATERS = 5                # full strength
MIN_SKATERS = 3            # a team never has fewer than 3 skaters on the ice
MAX_SERVING = 2            # penalties beyond two per team wait ("stacked") until one expires

# Penalty lengths in seconds; misconducts and coincidental penalties do not change the strength.
MINOR_SECONDS = 120
DOUBLE_MINOR_SECONDS = 240
MAJOR_SECONDS = 300
# A `matched_minor` is one of two minors (one per team) at the same stoppage at full 5-on-5
# strength, played 4-on-4 without substitutes (NHL rule 19.1); a goal never releases it.
STRENGTH_PENALTIES = {"minor": MINOR_SECONDS, "matched_minor": MINOR_SECONDS,
                      "double_minor": DOUBLE_MINOR_SECONDS, "major": MAJOR_SECONDS}
# Penalties that end early when the other team scores while the penalized team is short-handed
# (a double minor ends only its current two minutes).
RELEASABLE = ("minor", "double_minor")

# Game phases
FACEOFF, LIVE, PENALTY_SHOT, SHOOTOUT = 0, 1, 2, 3
PHASES = ["faceoff", "live", "penalty_shot", "shootout"]

PLAY_TYPES = [
    "faceoff", "shot", "hit", "giveaway", "takeaway", "penalty", "delayed_penalty", "stoppage",
    "goalie_pull", "goalie_return", "period_end", "shootout",
]
PT = {p: i for i, p in enumerate(PLAY_TYPES)}

STOPPAGES = ["goalie_freeze", "icing", "offside", "out_of_play", "timeout", "other"]
PENALTIES = ["minor", "matched_minor", "double_minor", "major", "coincidental", "misconduct",
             "penalty_shot"]
RESULTS = [
    "won",                                           # faceoff: the side won the draw
    "goal", "saved", "missed", "blocked",            # shot attempts (and shootout / penalty shots)
    "hit", "giveaway", "takeaway", "delayed_penalty", "goalie_pull", "goalie_return",
    *PENALTIES,
    *STOPPAGES,
    "end",                                           # period_end
]
RS = {r: i for i, r in enumerate(RESULTS)}

RESULTS_FOR_TYPE = {
    "faceoff": ["won"],
    "shot": ["goal", "saved", "missed", "blocked"],
    "hit": ["hit"],
    "giveaway": ["giveaway"],
    "takeaway": ["takeaway"],
    "penalty": PENALTIES,
    "delayed_penalty": ["delayed_penalty"],
    "stoppage": STOPPAGES,
    "goalie_pull": ["goalie_pull"],
    "goalie_return": ["goalie_return"],
    "period_end": ["end"],
    "shootout": ["goal", "saved", "missed"],
}
# Events with no team: the side is None.
SIDELESS = ("stoppage", "period_end")
TYPES_FOR_PHASE = {
    FACEOFF: ["faceoff", "penalty", "stoppage", "goalie_pull", "goalie_return", "period_end"],
    LIVE: ["shot", "hit", "giveaway", "takeaway", "penalty", "delayed_penalty", "stoppage",
           "goalie_pull", "goalie_return", "period_end"],
    PENALTY_SHOT: ["shot"],
    SHOOTOUT: ["shootout"],
}

# Clock seconds elapsed before an event, 0..MAX_ELAPSED (clamped). Real gaps between events are
# under 100 seconds 99.9% of the time and never above about 210 (2010-11 to 2024-25).
MAX_ELAPSED = 240
N_ELAPSED = MAX_ELAPSED + 1

# Input-feature buckets
N_PERIOD = 6        # index min(period, 5): 1..3, 4 = OT, 5 = 2OT and later
N_PSEC = 41         # period seconds remaining bucketed by 30s: 0..40
N_DIFF = 21         # score diff (home - away) clamped to [-10, 10]
N_SKATERS = 7       # skaters on the ice 0..6 (0/1 in a penalty shot or shootout)
N_PEN_SEC = 11      # seconds left on the soonest-expiring penalty, bucketed by 30s: 0..10 (300s)
N_GAME_TYPE = 2     # REG, POST


def bucket_psec(psec: float) -> int:
    return int(min(max(psec, 0), PERIOD_SECONDS) // 30)


def bucket_diff(d: int) -> int:
    return int(min(max(d, -10), 10)) + 10
