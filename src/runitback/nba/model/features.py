"""Feature encoding shared by training (vectorized over a dataframe) and sampling (GameState)."""
from __future__ import annotations

import numpy as np

from ..engine.rules import GameState, Outcome, elapsed_range, legal_results, legal_types
from ..league import BASE_FEATURES, NBA, League, get_league
from ..vocab import (
    FIRST_SEASON, MAX_ELAPSED, MAX_SEASON, N_DIFF, N_ELAPSED, N_GAME_TYPE, N_QTR, N_TO, PHASES, PLAY_TYPES, PT,
    bucket_diff,
)

# Input feature names (each an integer index into its own embedding table): the NBA's. A league with
# a bonus adds the team-foul counts (``League.features``).
FEATURES = BASE_FEATURES
# Continuous team-strength inputs, from the possession team's point of view.
CONT_FEATURES = ["off_off_rtg", "off_def_rtg", "def_off_rtg", "def_def_rtg", "elo_diff"]


def cont_vector(off: dict | None, dfn: dict | None) -> list[float]:
    """off/dfn: pre-game ratings dicts {off_rtg, def_rtg, elo}; None means unknown (zeros)."""
    if off is None or dfn is None:
        return [0.0] * len(CONT_FEATURES)
    return [off["off_rtg"] / 5, off["def_rtg"] / 5, dfn["off_rtg"] / 5, dfn["def_rtg"] / 5,
            (off["elo"] - dfn["elo"]) / 100]


def feature_sizes(n_team_seasons: int, league: League | str | None = None, n_teams: int | None = None
                  ) -> dict[str, int]:
    """Embedding-table sizes. ``n_teams`` is the number of teams in the vocabulary (the league's own
    list when it has one); index 0 of each team table is UNK."""
    lg = get_league(league)
    n_teams = len(lg.teams) if n_teams is None and lg.teams else (n_teams or 0)
    sizes = dict(
        qtr=N_QTR, qsec=lg.n_qsec, phase=len(PHASES), fts_left=lg.n_fts, diff=N_DIFF, to_off=N_TO, to_def=N_TO,
        is_home=2, off_team=n_teams + 1, def_team=n_teams + 1,
        off_ts=n_team_seasons, def_ts=n_team_seasons, season=MAX_SEASON - FIRST_SEASON + 1,
        game_type=N_GAME_TYPE, prev_type=len(PLAY_TYPES) + 1, prev_result=lg.n_results + 1,
        prev_elapsed=N_ELAPSED + 1,
    )
    if lg.bonus:
        sizes |= dict(fouls_off=lg.n_fouls, fouls_def=lg.n_fouls)
    return sizes


class TeamSeasonVocab:
    """Maps (team, season) -> embedding index; 0 is UNK (unseen team-season). Also carries the
    league and its team list (the NBA's fixed 30, or the teams of a college league's training
    games), whose index 0 is UNK too."""

    def __init__(self, keys: list[str], league: League | str | None = None, teams: list[str] | None = None):
        self.league = get_league(league)
        self.keys = sorted(set(keys))
        self.idx = {k: i + 1 for i, k in enumerate(self.keys)}
        self.teams = list(teams) if teams is not None else list(self.league.teams or [])
        self.team_idx = {t: i for i, t in enumerate(self.teams)}

    def __len__(self) -> int:
        return len(self.keys) + 1

    def get(self, team: str, season: int) -> int:
        return self.idx.get(f"{team}_{season}", 0)

    def team(self, team: str) -> int:
        return self.team_idx.get(team, -1) + 1

    def sizes(self) -> dict[str, int]:
        return feature_sizes(len(self), self.league, len(self.teams))

    def meta(self) -> dict:
        """What a checkpoint stores next to the weights (``from_checkpoint`` reads it back)."""
        return dict(team_seasons=self.keys, league=self.league.name, teams=self.teams)

    @classmethod
    def from_checkpoint(cls, ck: dict) -> "TeamSeasonVocab":
        ts = cls([], ck.get("league"), ck.get("teams"))
        ts.keys = list(ck["team_seasons"])
        ts.idx = {k: i + 1 for i, k in enumerate(ts.keys)}
        return ts


def encode_state(s: GameState, prev: Outcome | None, ts: TeamSeasonVocab,
                 ratings: tuple[dict, dict] | None = None) -> dict:
    """Integer features plus "cont" (list of floats); ratings = (home, away) pre-game ratings."""
    off, dfn = s.off_team, s.def_team
    lg = s.league
    r_off, r_def = (ratings[s.pos], ratings[1 - s.pos]) if ratings else (None, None)
    feats = dict(
        cont=cont_vector(r_off, r_def),
        qtr=min(s.qtr, 6), qsec=lg.bucket_qsec(s.qsec), phase=s.phase, fts_left=min(s.fts_left, 3),
        diff=bucket_diff(s.score[s.pos] - s.score[1 - s.pos]),
        to_off=min(s.timeouts[s.pos], N_TO - 1), to_def=min(s.timeouts[1 - s.pos], N_TO - 1),
        is_home=int(s.pos == 0),
        off_team=ts.team(off), def_team=ts.team(dfn),
        off_ts=ts.get(off, s.season), def_ts=ts.get(dfn, s.season),
        season=min(max(s.season - FIRST_SEASON, 0), MAX_SEASON - FIRST_SEASON),
        game_type=int(s.game_type == "POST"),
        **encode_prev(prev, lg),
    )
    if lg.bonus:
        feats["fts_left"] = 4 if s.one_and_one and lg.n_fts > 4 else feats["fts_left"]
        feats["fouls_off"] = min(s.fouls[s.pos], lg.n_fouls - 1)
        feats["fouls_def"] = min(s.fouls[1 - s.pos], lg.n_fouls - 1)
    return feats


def encode_prev(prev: Outcome | None, league: League = NBA) -> dict[str, int]:
    if prev is None:
        return dict(prev_type=0, prev_result=0, prev_elapsed=0)
    return dict(prev_type=PT[prev.play_type] + 1, prev_result=league.rs[prev.result] + 1,
                prev_elapsed=min(prev.elapsed, MAX_ELAPSED) + 1)


def type_mask(s: GameState) -> np.ndarray:
    m = np.zeros(len(PLAY_TYPES), dtype=bool)
    m[[PT[t] for t in legal_types(s)]] = True
    return m


def result_mask(s: GameState, play_type: str) -> np.ndarray:
    m = np.zeros(s.league.n_results, dtype=bool)
    rs = s.league.rs
    m[[rs[r] for r in legal_results(s, play_type)]] = True
    return m


def elapsed_mask(s: GameState, play_type: str) -> np.ndarray:
    lo, hi = elapsed_range(s, play_type)
    ar = np.arange(N_ELAPSED)
    return (ar >= lo) & (ar <= hi)
