"""Leak-free pre-game team-strength ratings: EWMA EPA/play (offense, defense) and Elo."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..vocab import TEAM_ALIASES

ALPHA = 0.15          # EWMA weight of the newest game
CARRY = 2 / 3         # fraction of a rating carried into a new season
ELO_MEAN, ELO_K, ELO_HFA = 1505.0, 20.0, 48.0
RATING_COLS = ["off_epa", "def_epa", "elo"]
NEUTRAL = {"off_epa": 0.0, "def_epa": 0.0, "elo": ELO_MEAN}


def game_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per game: teams, final score and each team's offensive EPA/play."""
    df = raw.copy()
    for c in ("home_team", "away_team", "posteam"):
        df[c] = df[c].replace(TEAM_ALIASES)
    plays = df[df.play_type.isin(["run", "pass"]) & df.epa.notna()]
    epa = plays.groupby(["game_id", "posteam"]).epa.mean().unstack()
    g = df.groupby("game_id").agg(season=("season", "first"), week=("week", "first"),
                                  game_date=("game_date", "first"), home=("home_team", "first"),
                                  away=("away_team", "first"), home_score=("home_score", "first"),
                                  away_score=("away_score", "first")).reset_index()
    g["home_epa"] = [epa.at[i, h] if i in epa.index and h in epa.columns else np.nan for i, h in zip(g.game_id, g.home)]
    g["away_epa"] = [epa.at[i, a] if i in epa.index and a in epa.columns else np.nan for i, a in zip(g.game_id, g.away)]
    return g


def _regress(r: dict) -> dict:
    return {"off_epa": r["off_epa"] * CARRY, "def_epa": r["def_epa"] * CARRY,
            "elo": ELO_MEAN + (r["elo"] - ELO_MEAN) * CARRY}


def scale_ratings(r: dict, scale: float) -> dict:
    """Stretch a team's rating deviations from the league mean (calibration of team-strength gaps)."""
    if scale == 1.0:
        return dict(r)
    return {"off_epa": r["off_epa"] * scale, "def_epa": r["def_epa"] * scale,
            "elo": ELO_MEAN + (r["elo"] - ELO_MEAN) * scale}


def compute_ratings(games: pd.DataFrame) -> pd.DataFrame:
    """Walk games chronologically; each row holds a team's ratings before (pre_*) and after (post_*) a game."""
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    state: dict[str, dict] = {}
    season_of: dict[str, int] = {}
    rows = []
    for g in games.itertuples(index=False):
        pre = {}
        for t in (g.home, g.away):
            r = state.get(t, dict(NEUTRAL))
            if t in season_of and season_of[t] != g.season:
                r = _regress(r)
            pre[t] = r
        post = {t: dict(r) for t, r in pre.items()}
        # EPA EWMA: offense from own EPA, defense from EPA allowed
        for t, own, opp in ((g.home, g.home_epa, g.away_epa), (g.away, g.away_epa, g.home_epa)):
            if not math.isnan(own):
                post[t]["off_epa"] = (1 - ALPHA) * pre[t]["off_epa"] + ALPHA * own
            if not math.isnan(opp):
                post[t]["def_epa"] = (1 - ALPHA) * pre[t]["def_epa"] + ALPHA * opp
        # Elo (538-style margin-of-victory multiplier)
        margin = g.home_score - g.away_score
        diff = pre[g.home]["elo"] + ELO_HFA - pre[g.away]["elo"]
        exp_home = 1 / (1 + 10 ** (-diff / 400))
        result = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        winner_diff = diff if margin > 0 else -diff
        mult = math.log(abs(margin) + 1) * 2.2 / (winner_diff * 0.001 + 2.2) if margin != 0 else 1.0
        delta = ELO_K * mult * (result - exp_home)
        post[g.home]["elo"] = pre[g.home]["elo"] + delta
        post[g.away]["elo"] = pre[g.away]["elo"] - delta
        for t, side in ((g.home, "home"), (g.away, "away")):
            rows.append({"game_id": g.game_id, "season": g.season, "week": g.week, "team": t, "side": side,
                         **{f"pre_{k}": pre[t][k] for k in RATING_COLS},
                         **{f"post_{k}": post[t][k] for k in RATING_COLS}})
            state[t], season_of[t] = post[t], g.season
    return pd.DataFrame(rows)


def build_ratings(seasons: list[int], raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    cols = ["game_id", "season", "week", "game_date", "home_team", "away_team", "posteam", "play_type",
            "epa", "home_score", "away_score"]
    games = pd.concat([game_stats(pd.read_parquet(raw_dir / f"pbp_{s}.parquet", columns=cols)) for s in seasons])
    r = compute_ratings(games)
    out_dir.mkdir(parents=True, exist_ok=True)
    r.to_parquet(out_dir / "ratings.parquet")
    return r


class RatingsTable:
    """Lookup of team ratings: as-of a specific game, or the latest available for a season."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values(["season", "week"]).reset_index(drop=True)

    def to_records(self) -> list[dict]:
        return self.df.to_dict("records")

    @classmethod
    def from_records(cls, recs: list[dict]) -> "RatingsTable":
        return cls(pd.DataFrame(recs))

    def latest(self, team: str, season: int, week: int | None = None) -> dict:
        """Ratings for `team` going into a game in (season, week); week=None means after all known games."""
        d = self.df[self.df.team == team]
        before = d[(d.season < season) | ((d.season == season) & (d.week < week if week is not None else True))]
        if before.empty:
            return dict(NEUTRAL)
        last = before.iloc[-1]
        r = {k: float(last[f"post_{k}"]) for k in RATING_COLS}
        return _regress(r) if last.season != season else r
