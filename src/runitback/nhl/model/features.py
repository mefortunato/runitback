"""Feature encoding shared by training (vectorized over the plays table) and sampling (GameState).

Both paths go through `encode_arrays`, which takes the engine-state columns of
`data.preprocess.state_row`, so the generator sees the same inputs when it trains and when it
samples. The state is from the home team's side (score difference home - away); the teams enter
as home / away team and team-season embeddings, plus the pre-game ratings as continuous inputs.
"""
from __future__ import annotations

import numpy as np

from ..data.preprocess import SIDE_IDX, state_row
from ..engine.rules import GameState, Outcome, elapsed_range, legal_results, legal_sides, legal_types
from ..vocab import (
    FIRST_SEASON, MAX_ELAPSED, MAX_SEASON, N_DIFF, N_ELAPSED, N_GAME_TYPE, N_PEN_SEC, N_PERIOD, N_PSEC,
    N_SKATERS, PERIOD_SECONDS, PHASES, PLAY_TYPES, PT, RESULTS, RS, TEAM_IDX,
)

N_SIDES = 3        # home, away, none
SO_ROUNDS = 7      # shootout attempts per team so far, 0..6+
SO_DIFF = 3        # shootout goal difference clamped to [-3, 3]

FEATURES = [
    "period", "psec", "phase", "diff", "sk_h", "sk_a", "en", "pen_h", "pen_a", "so", "so_next",
    "home_team", "away_team", "home_ts", "away_ts", "season", "game_type",
    "prev_type", "prev_side", "prev_result", "prev_elapsed",
]
# Continuous team-strength inputs (pre-game ratings, runitback.nhl.data.ratings)
CONT_FEATURES = ["home_off", "home_def", "home_xg_off", "home_xg_def",
                 "away_off", "away_def", "away_xg_off", "away_xg_def", "elo_diff"]


def feature_sizes(n_team_seasons: int) -> dict[str, int]:
    return dict(
        period=N_PERIOD, psec=N_PSEC, phase=len(PHASES), diff=N_DIFF, sk_h=N_SKATERS, sk_a=N_SKATERS, en=4,
        pen_h=N_PEN_SEC, pen_a=N_PEN_SEC, so=SO_ROUNDS * (2 * SO_DIFF + 1), so_next=N_SIDES,
        home_team=len(TEAM_IDX) + 1, away_team=len(TEAM_IDX) + 1, home_ts=n_team_seasons,
        away_ts=n_team_seasons, season=MAX_SEASON - FIRST_SEASON + 1, game_type=N_GAME_TYPE,
        prev_type=len(PLAY_TYPES) + 1, prev_side=N_SIDES + 1, prev_result=len(RESULTS) + 1,
        prev_elapsed=N_ELAPSED + 1,
    )


class TeamSeasonVocab:
    """Maps (team, season) -> embedding index; 0 is UNK (an unseen team-season)."""

    def __init__(self, keys: list[str]):
        self.keys = sorted(set(keys))
        self.idx = {k: i + 1 for i, k in enumerate(self.keys)}

    @classmethod
    def from_keys(cls, keys: list[str]) -> "TeamSeasonVocab":
        ts = cls([])
        ts.keys = list(keys)
        ts.idx = {k: i + 1 for i, k in enumerate(ts.keys)}
        return ts

    def __len__(self) -> int:
        return len(self.keys) + 1

    def get(self, team: str, season: int) -> int:
        return self.idx.get(f"{team}_{season}", 0)


def cont_vector(home: dict | None, away: dict | None) -> list[float]:
    """Pre-game ratings dicts (runitback.nhl.data.ratings.RATING_COLS) -> CONT_FEATURES; None: zeros."""
    if home is None or away is None:
        return [0.0] * len(CONT_FEATURES)
    return [home["off_rtg"], home["def_rtg"], home["xg_off"], home["xg_def"],
            away["off_rtg"], away["def_rtg"], away["xg_off"], away["xg_def"],
            (home["elo"] - away["elo"]) / 100]


def encode_arrays(st: dict, home, away, season, game_type, prev: dict, ts: TeamSeasonVocab) -> dict:
    """Integer features from arrays of engine-state columns (`state_row`), the teams, season,
    game type ("REG" / "POST") and the previous event (`prev`: play_type / side / result indices
    and elapsed, -1 before the first event)."""
    a = {k: np.asarray(v) for k, v in st.items()}
    season = np.asarray(season)
    so_round = np.minimum(np.maximum(a["so_shots_h"], a["so_shots_a"]), SO_ROUNDS - 1)
    so_diff = np.clip(a["so_goals_h"].astype(int) - a["so_goals_a"], -SO_DIFF, SO_DIFF) + SO_DIFF

    def prev_idx(key, cap=None):
        v = np.asarray(prev[key]).astype(np.int64)
        if cap is not None:
            v = np.minimum(v, cap)
        return np.where(v < 0, 0, v + 1)

    return dict(
        period=np.minimum(a["period"], N_PERIOD - 1),
        psec=np.clip(a["psec"], 0, PERIOD_SECONDS) // 30,
        phase=a["phase"].astype(np.int64),
        diff=np.clip(a["score_h"].astype(int) - a["score_a"], -10, 10) + 10,
        sk_h=np.clip(a["sk_h"], 0, N_SKATERS - 1), sk_a=np.clip(a["sk_a"], 0, N_SKATERS - 1),
        en=a["en_h"].astype(int) + 2 * a["en_a"].astype(int),
        pen_h=np.where(a["pen_h"] > 0, np.minimum((a["pen_left_h"] + 29) // 30, N_PEN_SEC - 1), 0),
        pen_a=np.where(a["pen_a"] > 0, np.minimum((a["pen_left_a"] + 29) // 30, N_PEN_SEC - 1), 0),
        so=so_round * (2 * SO_DIFF + 1) + so_diff, so_next=a["so_next"].astype(np.int64),
        home_team=np.array([TEAM_IDX.get(t, -1) + 1 for t in np.atleast_1d(home)]),
        away_team=np.array([TEAM_IDX.get(t, -1) + 1 for t in np.atleast_1d(away)]),
        home_ts=np.array([ts.get(t, int(s)) for t, s in zip(np.atleast_1d(home), np.atleast_1d(season))]),
        away_ts=np.array([ts.get(t, int(s)) for t, s in zip(np.atleast_1d(away), np.atleast_1d(season))]),
        season=np.clip(season - FIRST_SEASON, 0, MAX_SEASON - FIRST_SEASON),
        game_type=(np.asarray(game_type) == "POST").astype(np.int64),
        prev_type=prev_idx("type"), prev_side=prev_idx("side"), prev_result=prev_idx("result"),
        prev_elapsed=prev_idx("elapsed", MAX_ELAPSED),
    )


def prev_of(o: Outcome | None) -> dict:
    if o is None:
        return dict(type=-1, side=-1, result=-1, elapsed=-1)
    return dict(type=PT[o.play_type], side=SIDE_IDX[o.side], result=RS[o.result], elapsed=o.elapsed)


def encode_states(states: list[GameState], prevs: list[Outcome | None], ts: TeamSeasonVocab) -> dict:
    """Integer features for a batch of engine states (one step each)."""
    rows = [state_row(s) for s in states]
    st = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    pv = [prev_of(p) for p in prevs]
    prev = {k: np.array([p[k] for p in pv]) for k in pv[0]}
    return encode_arrays(st, [s.home for s in states], [s.away for s in states], [s.season for s in states],
                         [s.game_type for s in states], prev, ts)


# ---------------------------------------------------------------- masks from the engine

def type_mask(s: GameState) -> np.ndarray:
    m = np.zeros(len(PLAY_TYPES), dtype=bool)
    m[[PT[t] for t in legal_types(s)]] = True
    return m


def side_mask(s: GameState, play_type: str) -> np.ndarray:
    m = np.zeros(N_SIDES, dtype=bool)
    m[[SIDE_IDX[x] for x in legal_sides(s, play_type)]] = True
    return m


def result_mask(s: GameState, play_type: str) -> np.ndarray:
    m = np.zeros(len(RESULTS), dtype=bool)
    m[[RS[r] for r in legal_results(s, play_type)]] = True
    return m


def elapsed_mask(s: GameState, play_type: str) -> np.ndarray:
    lo, hi = elapsed_range(s, play_type)
    ar = np.arange(N_ELAPSED)
    return (ar >= lo) & (ar <= min(hi, MAX_ELAPSED))
