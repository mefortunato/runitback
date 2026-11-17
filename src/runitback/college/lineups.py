"""College basketball projected rotations: who is expected to play each game, and for how long.

There is no history of college injury reports, and ESPN's college box scores give no did-not-play
reason. The plan's projection (work package C3) is **recent minutes minus players who didn't play**:

- **Candidates** (``rule="dnp"``, the default): the players listed in the team's box score for the
  game (the roster at tip-off: transfers in, departures out) **minus everyone who did not play**.
  This is the documented stand-in for an injury report. **The leak:** a player who sat for a
  coach's decision (or was ruled out after the line was set) counts as known out at tip-off. For
  rotation players that is mostly injury, illness and suspension, which a pre-game report would
  have; for end-of-bench players it changes nothing, because they have no projected minutes.
  With ``rule="previous"`` the candidates are the players who played for the team in any of its
  previous ``TEAM_GAMES`` games and nobody is removed: no leak, but it misses absences and arrivals.
- **Expected minutes.** Each candidate's recency-weighted mean (``DECAY`` per game back) of his
  minutes over the games he played before the game's date, **for any team**: a transfer brings his
  history with him (ESPN athlete ids carry across schools). A player with no earlier games (a
  freshman, or a player new to the data) gets ``NEW_MINUTES``. Minutes are then scaled to the
  team's 200 (no player above ``MAX_MINUTES``), so a missing starter's minutes go to the others
  in proportion.
- **Starters:** the five with the highest recency-weighted share of starts (ties by expected
  minutes). ``rank`` orders the candidates by projected minutes (the rotation slots).

Nothing else about the game is used: its minutes, starters and results are not.

Output (``projected_lineups.parquet``), one row per (game, team, player): ``game_id, season,
game_date, team_id, team, player_id, group, exp_minutes, proj_minutes, proj_starter, rank,
recent_games, source`` (``box`` | ``previous``).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DECAY = 0.85          # weight of a player's k-th most recent game: DECAY ** k
TEAM_GAMES = 3        # rule="previous": the team's last TEAM_GAMES games' players
TEAM_MINUTES = 200.0  # regulation (5 x 40)
MAX_MINUTES = 38.0
NEW_MINUTES = 6.0     # expected minutes of a player with no earlier games
STARTERS = 5
RULES = ("dnp", "previous")

COLUMNS = ["game_id", "season", "game_date", "team_id", "team", "player_id", "group", "exp_minutes",
           "proj_minutes", "proj_starter", "rank", "recent_games", "source"]


def player_history(pg: pd.DataFrame, decay: float = DECAY) -> pd.DataFrame:
    """Per played game of each player (any team), the recency-weighted means of his minutes and of
    his starts over his games up to and including it: ``player_id, game_date, ewm_minutes,
    ewm_start, games`` (``games``: played games so far). Several games on one date keep the last."""
    p = pg[~pg.dnp].sort_values(["player_id", "game_date", "game_id"], kind="stable")
    alpha = 1 - decay
    g = p.groupby("player_id", sort=False)
    out = pd.DataFrame({
        "player_id": p.player_id.to_numpy(), "game_date": pd.to_datetime(p.game_date).to_numpy(),
        "ewm_minutes": g.minutes.transform(lambda s: s.ewm(alpha=alpha, adjust=True).mean()).to_numpy(),
        "ewm_start": g.starter.transform(lambda s: s.astype(float).ewm(alpha=alpha, adjust=True).mean()).to_numpy(),
        "games": (g.cumcount() + 1).to_numpy(),
    })
    return out.drop_duplicates(["player_id", "game_date"], keep="last").reset_index(drop=True)


def history_before(hist: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """``keys`` (``player_id, game_date`` plus anything) with each player's history as of his last
    played game strictly before ``game_date`` (NaN / 0 games when there is none)."""
    k = keys.assign(_d=pd.to_datetime(keys.game_date), _i=np.arange(len(keys))).sort_values("_d", kind="stable")
    h = hist.rename(columns={"game_date": "_d"}).sort_values("_d", kind="stable")
    m = pd.merge_asof(k, h, on="_d", by="player_id", allow_exact_matches=False)
    m = m.sort_values("_i").drop(columns=["_d", "_i"]).reset_index(drop=True)
    m["games"] = m.games.fillna(0).astype(int)
    return m


def scale_minutes(exp: np.ndarray, total: float = TEAM_MINUTES, cap: float = MAX_MINUTES) -> np.ndarray:
    """Scale expected minutes to sum to ``total``, no one above ``cap`` (the excess goes to the
    others in proportion)."""
    exp = np.maximum(np.asarray(exp, dtype=float), 0.0)
    if exp.sum() <= 0:
        return exp
    cap = max(cap, total / max(int((exp > 0).sum()), 1))   # too few players: the cap gives way
    capped = np.zeros(len(exp), dtype=bool)
    while True:
        free = total - cap * capped.sum()
        rest = np.where(capped, 0.0, exp)
        out = np.where(capped, cap, rest * free / rest.sum())
        over = ~capped & (out > cap + 1e-12)
        if not over.any():
            return out
        capped |= over


def candidates(pg: pd.DataFrame, rule: str = "dnp", team_games: int = TEAM_GAMES) -> pd.DataFrame:
    """The candidate players of every team-game: ``game_id, season, game_date, team_id, team,
    player_id, group, source`` (see the module docstring for ``rule``)."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    keep = ["game_id", "season", "game_date", "team_id", "team", "player_id", "group"]
    if rule == "dnp":
        return pg.loc[~pg.dnp, keep].assign(source="box").reset_index(drop=True)
    order = pg[["game_id", "season", "game_date", "team_id", "team"]].drop_duplicates(["game_id", "team_id"])
    order = order.sort_values(["team_id", "game_date", "game_id"], kind="stable").reset_index(drop=True)
    order["i"] = order.groupby("team_id").cumcount()
    played = pg.loc[~pg.dnp, ["game_id", "team_id", "player_id", "group"]].merge(
        order[["game_id", "team_id", "i"]], on=["game_id", "team_id"])
    parts = []
    for lag in range(1, team_games + 1):
        parts.append(order.assign(i=order.i - lag).merge(played.drop(columns="game_id"), on=["team_id", "i"]))
    c = pd.concat(parts, ignore_index=True).drop_duplicates(["game_id", "team_id", "player_id"])
    return c[keep].assign(source="previous").reset_index(drop=True)


def projected_lineups(pg: pd.DataFrame, rule: str = "dnp", decay: float = DECAY) -> pd.DataFrame:
    """Every team-game's projected rotation (see the module docstring): :data:`COLUMNS`."""
    hist = player_history(pg, decay)
    c = history_before(hist, candidates(pg, rule))
    c["exp_minutes"] = c.ewm_minutes.fillna(NEW_MINUTES)
    c["_start"] = c.ewm_start.fillna(0.0)
    c = c.sort_values(["game_id", "team_id", "exp_minutes"], ascending=[True, True, False], kind="stable")
    key = c.game_id + "_" + c.team_id.astype(str)
    c["proj_minutes"] = c.groupby(key.to_numpy()).exp_minutes.transform(lambda s: pd.Series(
        scale_minutes(s.to_numpy()), index=s.index))
    c["rank"] = c.groupby(key.to_numpy()).cumcount()
    st = c.sort_values(["game_id", "team_id", "_start", "exp_minutes"], ascending=[True, True, False, False],
                       kind="stable")
    st_rank = st.groupby(["game_id", "team_id"]).cumcount()
    c["proj_starter"] = (st_rank.reindex(c.index) < STARTERS).to_numpy()
    c = c.rename(columns={"games": "recent_games"})
    return c[COLUMNS].reset_index(drop=True)


def evaluate(proj: pd.DataFrame, pg: pd.DataFrame) -> pd.DataFrame:
    """Per season: how the projection compares with who actually played.

    ``minutes_mae``: mean absolute error of projected vs actual minutes over every player who
    played or was projected (0 for the missing side); ``top5``: share of each team-game's five
    most-used players (by actual minutes) among its five highest projected; ``starters``: share of
    actual starters among the projected starters; ``missed``: share of actual minutes played by
    players not in the projection."""
    act = pg.loc[~pg.dnp, ["game_id", "team_id", "season", "player_id", "minutes", "starter"]]
    m = act.merge(proj[["game_id", "team_id", "player_id", "proj_minutes", "rank", "proj_starter"]],
                  on=["game_id", "team_id", "player_id"], how="outer")
    m = m[m.game_id.isin(set(proj.game_id))]
    m["season"] = m.game_id.map(pg.drop_duplicates("game_id").set_index("game_id").season)
    m["minutes"] = m.minutes.fillna(0.0)
    m["proj_minutes"] = m.proj_minutes.fillna(0.0)
    m["err"] = (m.minutes - m.proj_minutes).abs()
    m["act_rank"] = m.sort_values("minutes", ascending=False, kind="stable").groupby(
        ["game_id", "team_id"]).cumcount().reindex(m.index)
    top = m[m.act_rank < 5]
    rows = []
    for s, d in m.groupby("season"):
        t = top[top.season == s]
        st = d[d.starter.fillna(False).astype(bool)]
        rows.append({"season": int(s), "team_games": int(d.groupby(["game_id", "team_id"]).ngroups),
                     "minutes_mae": float(d.err.mean()),
                     "top5": float((t["rank"].fillna(99) < 5).mean()),
                     "starters": float(st.proj_starter.fillna(False).astype(bool).mean()),
                     "missed": float(d.loc[d["rank"].isna(), "minutes"].sum() / d.minutes.sum())})
    return pd.DataFrame(rows)


def build_lineups(processed: Path, rule: str = "dnp") -> pd.DataFrame:
    """``projected_lineups.parquet`` in ``processed`` (from ``player_games.parquet``); returns the
    per-season check against who actually played."""
    pg = pd.read_parquet(processed / "player_games.parquet")
    proj = projected_lineups(pg, rule)
    proj.to_parquet(processed / ("projected_lineups.parquet" if rule == "dnp" else f"projected_lineups_{rule}.parquet"),
                    index=False)
    return evaluate(proj, pg)
