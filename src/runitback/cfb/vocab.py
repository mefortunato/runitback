"""College football vocabularies.

The play vocabulary (phases, play types, results, the yards and clock encodings) is the NFL's
(``runitback.nfl.vocab``): a college play is described by the same outcome, so the NFL play
transformer trains on college plays unchanged. What differs is the rules engine
(``runitback.cfb.engine.rules``) and the constants below.

Teams are keyed by ESPN team id (``"2"``, ``"333"``, ...): there are several hundred schools, the
set changes every season, and ids never collide with the NFL's abbreviations, so the NFL model's
franchise table leaves them all at its unknown index and team identity comes from the team-season
tokens built from the data.
"""
from __future__ import annotations

from ..nfl.vocab import (  # noqa: F401  (re-exported: the shared play vocabulary)
    KICKOFF, MAX_ELAPSED, N_ELAPSED, N_YARDS, NONE_RESULTS, PAT, PHASES, PLAY_TYPES, PT, RESULTS,
    RESULTS_FOR_TYPE, RS, SCRIMMAGE, SPOT_RESULTS, TYPES_FOR_PHASE, YARDS_MAX, YARDS_MIN,
    idx_to_yards, yards_kind, yards_to_idx,
)

FIRST_SEASON = 2004  # espn_cfb_pbp starts with the 2004 season

# Tries (PAT kicks and two-point attempts) are snapped from the 3-yard line in college football,
# and so are the two-point attempts of the overtime shootout.
PAT_SPOT = 3
# Each overtime possession starts 1st and 10 at the opponent's 25 (no kickoff, no game clock).
OT_SPOT = 25
QUARTER_SECONDS = 900
TIMEOUTS_PER_HALF = 3
TIMEOUTS_PER_OT = 1  # per team, per overtime period
