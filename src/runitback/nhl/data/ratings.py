"""Leak-free pre-game NHL team-strength ratings: EWMA goals and expected goals for and against, and Elo.

Walks the games in date order; a team's `pre_*` ratings use only its earlier games. Goals and
expected goals (the release's `xg`, summed per team without the shootout) are per game relative to
the league's average so far that season, so a rating means the same thing in a 2012 grind and a
2024 shootout year. A game without xG leaves the xG ratings where they were. Ratings shrink toward
the league mean between seasons. Elo uses the wager model's NHL settings (K 6, home ice 30 points;
`runitback.wager.nhl.SPORT`).
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ALPHA = 0.06          # EWMA weight of the newest game (82-game seasons; goals are noisy)
CARRY = 0.7           # fraction of a rating carried into a new season
ELO_MEAN, ELO_K, ELO_HFA = 1500.0, 6.0, 30.0
LEAGUE_GOALS = 2.9    # starting league average goals (and xG) per team-game
RATING_COLS = ["off_rtg", "def_rtg", "xg_off", "xg_def", "elo"]
NEUTRAL = {"off_rtg": 0.0, "def_rtg": 0.0, "xg_off": 0.0, "xg_def": 0.0, "elo": ELO_MEAN}


def _regress(r: dict) -> dict:
    out = {k: r[k] * CARRY for k in RATING_COLS if k != "elo"}
    out["elo"] = ELO_MEAN + (r["elo"] - ELO_MEAN) * CARRY
    return out


def scale_ratings(r: dict, scale: float) -> dict:
    """Stretch a team's rating deviations from the league mean (calibration of team-strength gaps)."""
    if scale == 1.0:
        return dict(r)
    out = {k: r[k] * scale for k in RATING_COLS if k != "elo"}
    out["elo"] = ELO_MEAN + (r["elo"] - ELO_MEAN) * scale
    return out


def elo_win_prob(home_elo: float, away_elo: float, hfa: float = ELO_HFA) -> float:
    return 1 / (1 + 10 ** (-(home_elo + hfa - away_elo) / 400))


def compute_ratings(games: pd.DataFrame) -> pd.DataFrame:
    """`games`: game_id, season, game_date, home, away, home_final, away_final and optionally
    xg_h / xg_a. One row per team and game with its ratings before (pre_*) and after (post_*)."""
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    state: dict[str, dict] = {}
    season_of: dict[str, int] = {}
    league = {"season": None, "g": 0.0, "x": 0.0, "n": 0, "nx": 0, "prev_g": LEAGUE_GOALS,
              "prev_x": LEAGUE_GOALS}
    rows = []
    for g in games.itertuples(index=False):
        if league["season"] != g.season:
            if league["n"]:
                league["prev_g"] = league["g"] / league["n"]
            if league["nx"]:
                league["prev_x"] = league["x"] / league["nx"]
            league.update(season=g.season, g=0.0, x=0.0, n=0, nx=0)
        avg_g = league["g"] / league["n"] if league["n"] >= 60 else league["prev_g"]
        avg_x = league["x"] / league["nx"] if league["nx"] >= 60 else league["prev_x"]
        pre = {}
        for t in (g.home, g.away):
            r = state.get(t, dict(NEUTRAL))
            if t in season_of and season_of[t] != g.season:
                r = _regress(r)
            pre[t] = r
        post = {t: dict(r) for t, r in pre.items()}
        goals = {g.home: g.home_final, g.away: g.away_final}
        if getattr(g, "shootout", False):  # the shootout goal is not a goal scored in play
            winner = g.home if g.home_final > g.away_final else g.away
            goals[winner] -= 1
        xh, xa = getattr(g, "xg_h", math.nan), getattr(g, "xg_a", math.nan)
        has_xg = not (pd.isna(xh) or pd.isna(xa))
        xg = {g.home: xh, g.away: xa}
        for t, o in ((g.home, g.away), (g.away, g.home)):
            post[t]["off_rtg"] = (1 - ALPHA) * pre[t]["off_rtg"] + ALPHA * (goals[t] - avg_g)
            post[t]["def_rtg"] = (1 - ALPHA) * pre[t]["def_rtg"] + ALPHA * (goals[o] - avg_g)
            if has_xg:
                post[t]["xg_off"] = (1 - ALPHA) * pre[t]["xg_off"] + ALPHA * (xg[t] - avg_x)
                post[t]["xg_def"] = (1 - ALPHA) * pre[t]["xg_def"] + ALPHA * (xg[o] - avg_x)
        league["g"] += goals[g.home] + goals[g.away]
        league["n"] += 2
        if has_xg:
            league["x"] += xh + xa
            league["nx"] += 2
        margin = g.home_final - g.away_final
        p_home = elo_win_prob(pre[g.home]["elo"], pre[g.away]["elo"])
        result = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        delta = ELO_K * (result - p_home)
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
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    r.to_parquet(Path(out_dir) / "ratings.parquet")
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
