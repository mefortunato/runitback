"""Leak-free pre-game NBA team-strength ratings: EWMA offensive/defensive efficiency and Elo.

Efficiency is points per 100 possessions relative to the league average so far that season, so
the ratings mean the same thing in a 2004 grind and a 2025 shootout.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ALPHA = 0.08          # EWMA weight of the newest game (82-game seasons)
CARRY = 0.75          # fraction of a rating carried into a new season
ELO_MEAN, ELO_K, ELO_HFA = 1505.0, 20.0, 100.0
LEAGUE_EFF = 110.0    # starting league average points per 100 possessions
RATING_COLS = ["off_rtg", "def_rtg", "elo"]
NEUTRAL = {"off_rtg": 0.0, "def_rtg": 0.0, "elo": ELO_MEAN}


def _regress(r: dict) -> dict:
    return {"off_rtg": r["off_rtg"] * CARRY, "def_rtg": r["def_rtg"] * CARRY,
            "elo": ELO_MEAN + (r["elo"] - ELO_MEAN) * CARRY}


def scale_ratings(r: dict, scale: float) -> dict:
    """Stretch a team's rating deviations from the league mean (calibration of team-strength gaps)."""
    if scale == 1.0:
        return dict(r)
    return {"off_rtg": r["off_rtg"] * scale, "def_rtg": r["def_rtg"] * scale,
            "elo": ELO_MEAN + (r["elo"] - ELO_MEAN) * scale}


def elo_win_prob(home_elo: float, away_elo: float, hfa: float = ELO_HFA) -> float:
    return 1 / (1 + 10 ** (-(home_elo + hfa - away_elo) / 400))


def compute_ratings(games: pd.DataFrame) -> pd.DataFrame:
    """Walk games chronologically. `games` needs game_id, season, game_date, home, away,
    home_final, away_final and, optionally, poss_h/poss_a (possessions; else estimated as 100).
    Each output row holds a team's ratings before (pre_*) and after (post_*) a game."""
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    state: dict[str, dict] = {}
    season_of: dict[str, int] = {}
    league = {"season": None, "sum": 0.0, "n": 0, "prev": LEAGUE_EFF}
    rows = []
    for g in games.itertuples(index=False):
        if league["season"] != g.season:
            if league["n"]:
                league["prev"] = league["sum"] / league["n"]
            league.update(season=g.season, sum=0.0, n=0)
        avg = league["sum"] / league["n"] if league["n"] >= 30 else league["prev"]
        pre = {}
        for t in (g.home, g.away):
            r = state.get(t, dict(NEUTRAL))
            if t in season_of and season_of[t] != g.season:
                r = _regress(r)
            pre[t] = r
        post = {t: dict(r) for t, r in pre.items()}
        ph, pa = getattr(g, "poss_h", None), getattr(g, "poss_a", None)
        poss = (ph + pa) / 2 if ph is not None and pa is not None and ph + pa > 0 else 100.0
        eff = {g.home: 100 * g.home_final / poss, g.away: 100 * g.away_final / poss}
        for t, o in ((g.home, g.away), (g.away, g.home)):
            post[t]["off_rtg"] = (1 - ALPHA) * pre[t]["off_rtg"] + ALPHA * (eff[t] - avg)
            post[t]["def_rtg"] = (1 - ALPHA) * pre[t]["def_rtg"] + ALPHA * (eff[o] - avg)
        league["sum"] += eff[g.home] + eff[g.away]
        league["n"] += 2
        # Elo (538-style NBA margin-of-victory multiplier)
        margin = g.home_final - g.away_final
        p_home = elo_win_prob(pre[g.home]["elo"], pre[g.away]["elo"])
        result = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        winner_diff = (pre[g.home]["elo"] + ELO_HFA - pre[g.away]["elo"]) * (1 if margin > 0 else -1)
        mult = (abs(margin) + 3) ** 0.8 / (7.5 + 0.006 * winner_diff) if margin else 1.0
        delta = ELO_K * mult * (result - p_home)
        post[g.home]["elo"] = pre[g.home]["elo"] + delta
        post[g.away]["elo"] = pre[g.away]["elo"] - delta
        for t, side in ((g.home, "home"), (g.away, "away")):
            rows.append({"game_id": g.game_id, "season": g.season, "game_date": g.game_date, "team": t,
                         "side": side, **{f"pre_{k}": pre[t][k] for k in RATING_COLS},
                         **{f"post_{k}": post[t][k] for k in RATING_COLS}})
            state[t], season_of[t] = post[t], g.season
    return pd.DataFrame(rows)


def build_ratings(games: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    r = compute_ratings(games)
    out_dir.mkdir(parents=True, exist_ok=True)
    r.to_parquet(out_dir / "ratings.parquet")
    return r


class RatingsTable:
    """Lookup of team ratings: as of a date, or the latest available for a season."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values(["season", "game_date", "game_id"]).reset_index(drop=True)

    def to_records(self) -> list[dict]:
        return self.df.to_dict("records")

    @classmethod
    def from_records(cls, recs: list[dict]) -> "RatingsTable":
        return cls(pd.DataFrame(recs))

    def latest(self, team: str, season: int, before: str | None = None) -> dict:
        """Ratings for `team` going into a game in `season` (on date `before`, YYYY-MM-DD);
        before=None means after all known games."""
        d = self.df[self.df.team == team]
        cut = (d.season < season) | ((d.season == season) & ((d.game_date < before) if before else True))
        prior = d[cut]
        if prior.empty:
            return dict(NEUTRAL)
        last = prior.iloc[-1]
        r = {k: float(last[f"post_{k}"]) for k in RATING_COLS}
        return _regress(r) if last.season != season else r
