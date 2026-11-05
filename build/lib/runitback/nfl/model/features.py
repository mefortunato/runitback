"""Feature encoding shared by training (vectorized over a dataframe) and sampling (GameState)."""
from __future__ import annotations

import numpy as np

from ..engine.rules import GameState, Outcome, legal_results, legal_types, yards_range
from ..vocab import (
    FIRST_SEASON, MAX_ELAPSED, N_ELAPSED, N_YARDS, PLAY_TYPES, PT, RESULTS, RS, TEAM_IDX,
    YARDS_MIN, bucket_diff, bucket_qsec, yards_to_idx,
)

# Input feature names (each an integer index into its own embedding table)
FEATURES = [
    "qtr", "qsec", "phase", "down", "ytg", "yl", "diff", "to_off", "to_def", "is_home",
    "off_team", "def_team", "off_ts", "def_ts", "season", "game_type",
    "prev_type", "prev_result", "prev_yards", "prev_elapsed",
]
# Continuous team-strength inputs, from the possession team's point of view.
CONT_FEATURES = ["off_off_epa", "off_def_epa", "def_off_epa", "def_def_epa", "elo_diff"]


def cont_vector(off: dict | None, dfn: dict | None) -> list[float]:
    """off/dfn: pre-game ratings dicts {off_epa, def_epa, elo}; None means unknown (zeros)."""
    if off is None or dfn is None:
        return [0.0] * len(CONT_FEATURES)
    return [off["off_epa"] * 10, off["def_epa"] * 10, dfn["off_epa"] * 10, dfn["def_epa"] * 10,
            (off["elo"] - dfn["elo"]) / 100]


def feature_sizes(n_team_seasons: int) -> dict[str, int]:
    from ..vocab import N_DIFF, N_DOWN, N_GAME_TYPE, N_QSEC, N_QTR, N_TO, N_YL, N_YTG, MAX_SEASON
    return dict(
        qtr=N_QTR, qsec=N_QSEC, phase=3, down=N_DOWN, ytg=N_YTG, yl=N_YL, diff=N_DIFF,
        to_off=N_TO, to_def=N_TO, is_home=2, off_team=len(TEAM_IDX) + 1, def_team=len(TEAM_IDX) + 1,
        off_ts=n_team_seasons, def_ts=n_team_seasons, season=MAX_SEASON - FIRST_SEASON + 1,
        game_type=2, prev_type=len(PLAY_TYPES) + 1, prev_result=len(RESULTS) + 1,
        prev_yards=N_YARDS + 1, prev_elapsed=N_ELAPSED + 1,
    )


class TeamSeasonVocab:
    """Maps (team, season) -> embedding index; 0 is UNK (unseen team-season)."""

    def __init__(self, keys: list[str]):
        self.keys = sorted(set(keys))
        self.idx = {k: i + 1 for i, k in enumerate(self.keys)}

    def __len__(self) -> int:
        return len(self.keys) + 1

    def get(self, team: str, season: int) -> int:
        return self.idx.get(f"{team}_{season}", 0)


def encode_state(s: GameState, prev: Outcome | None, ts: TeamSeasonVocab,
                 ratings: tuple[dict, dict] | None = None) -> dict:
    """Integer features plus "cont" (list of floats); ratings = (home, away) pre-game ratings."""
    off, dfn = s.off_team, s.def_team
    r_off, r_def = (ratings[s.pos], ratings[1 - s.pos]) if ratings else (None, None)
    return dict(
        cont=cont_vector(r_off, r_def),
        qtr=min(s.qtr, 6), qsec=bucket_qsec(s.qsec), phase=s.phase, down=min(s.down, 4),
        ytg=min(s.ytg, 30), yl=s.yl, diff=bucket_diff(s.score[s.pos] - s.score[1 - s.pos]),
        to_off=s.timeouts[s.pos], to_def=s.timeouts[1 - s.pos], is_home=int(s.pos == 0),
        off_team=TEAM_IDX.get(off, -1) + 1, def_team=TEAM_IDX.get(dfn, -1) + 1,
        off_ts=ts.get(off, s.season), def_ts=ts.get(dfn, s.season),
        season=min(s.season - FIRST_SEASON, 2040 - FIRST_SEASON), game_type=int(s.game_type == "POST"),
        **encode_prev(prev),
    )


def encode_prev(prev: Outcome | None) -> dict[str, int]:
    if prev is None:
        return dict(prev_type=0, prev_result=0, prev_yards=0, prev_elapsed=0)
    return dict(prev_type=PT[prev.play_type] + 1, prev_result=RS[prev.result] + 1,
                prev_yards=yards_to_idx(prev.yards) + 1, prev_elapsed=min(prev.elapsed, MAX_ELAPSED) + 1)


def type_mask(s: GameState) -> np.ndarray:
    m = np.zeros(len(PLAY_TYPES), dtype=bool)
    m[[PT[t] for t in legal_types(s)]] = True
    return m


def result_mask(s: GameState, play_type: str) -> np.ndarray:
    m = np.zeros(len(RESULTS), dtype=bool)
    m[[RS[r] for r in legal_results(s, play_type)]] = True
    return m


def yards_mask(s: GameState, play_type: str, result: str) -> np.ndarray:
    lo, hi = yards_range(s, play_type, result)
    ar = np.arange(N_YARDS) + YARDS_MIN
    return (ar >= lo) & (ar <= hi)


def elapsed_mask(s: GameState) -> np.ndarray:
    m = np.zeros(N_ELAPSED, dtype=bool)
    m[: min(MAX_ELAPSED, max(s.qsec, 0)) + 1] = True
    return m
