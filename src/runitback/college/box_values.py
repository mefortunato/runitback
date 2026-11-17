"""Minutes-weighted box-score player values for college basketball, refit weekly without leakage.

Women's college basketball has no on-floor lineups, and men's has none before 2023-24
(``experiments/cbb/lineup_probe.md``), so every player also gets a **box-score value**: what his
box-score rates say about how many points per 100 possessions his team scores (offense) and allows
(defense) while he is on the floor. Men's adjusted plus-minus (:mod:`.player_values`) shrinks toward
it.

- **Rates.** Each player's per-40-minute rates of the counts in :data:`FEATURES`, from his games
  before the cut-off, for any team (a transfer keeps his history), with earlier seasons weighted
  ``DECAY ** seasons back`` over ``WINDOW`` seasons plus the current one. They are shrunk toward his
  position group's **replacement rates** (the pooled rates of players with under ``MIN_MINUTES``
  minutes in a season, from the seasons before) with ``PSEUDO_MINUTES`` minutes of them. A player's
  features are his shrunk rates minus the replacement rates, so a player with no history is at
  replacement (0).
- **The fit.** Every team-game's points per 100 possessions is regressed on the minutes-weighted sum
  of its own players' features (offense coefficients) and of the opponent's players' features
  (defense coefficients), plus home and season controls, by ridge (penalty ``LAMBDA``), with the
  same recency weights. The features of every game in the fit are its players' rates **as of that
  game's week**, so a game's own box score never explains its own result.
- **Values.** A player's offense value is the offense coefficients times his features (points per
  100 possessions his team gains when he plays all 40 minutes instead of a replacement), and his
  defense value is minus the defense coefficients times them (points per 100 his team saves), so
  higher is better on both.
- **Leak-free and weekly.** Each cut-off (a Monday) uses only games before it: the rates, the
  replacement level and the coefficients.

:func:`weekly_values` returns one values table per cut-off, in the format of the adjusted plus-minus
(:func:`runitback.core.player_values.fit_apm`): ``side, player_id, group, minutes, value`` with
``value`` in points **per possession**.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DEFAULT_GROUP, GROUPS, REGULATION_MINUTES

log = logging.getLogger(__name__)

FEATURES = ("pts", "fgm", "fga", "fg3m", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf")
WINDOW = 3            # previous seasons in each fit (plus the current one)
DECAY = 0.85          # weight of a game from k seasons back: DECAY ** k
PSEUDO_MINUTES = 100  # minutes of replacement rates every player's rates are shrunk with
MIN_MINUTES = 150     # a season under this many minutes counts toward the replacement rates
LAMBDA = 30.0         # ridge penalty on the (standardized) rate coefficients (light: 26 coefficients, 50k games)
TEAM_MINUTES = (190.0, 330.0)   # team-games outside this range of player-minutes are left out of fits
MIN_POSS = 40.0


def day_number(date) -> int:
    """Days since 1970-01-01 of a ``YYYY-MM-DD`` date."""
    return int((pd.Timestamp(str(date)[:10]) - pd.Timestamp("1970-01-01")).days)


def week_start(t):
    """The Monday on or before day ``t`` (1970-01-01 was a Thursday)."""
    return t - (t + 3) % 7


@dataclass(frozen=True)
class BoxConfig:
    window: int = WINDOW
    decay: float = DECAY
    pseudo_minutes: float = PSEUDO_MINUTES
    min_minutes: float = MIN_MINUTES
    lam: float = LAMBDA


def _played(pg: pd.DataFrame) -> pd.DataFrame:
    p = pg[~pg.dnp].copy()
    p["t"] = pd.to_datetime(p.game_date).sub(pd.Timestamp("1970-01-01")).dt.days.astype(np.int64)
    p["week"] = week_start(p.t)
    p["e"] = p.minutes / REGULATION_MINUTES          # exposure in 40-minute units
    p["g"] = p.group.where(p.group.isin(GROUPS), DEFAULT_GROUP).map({g: i for i, g in enumerate(GROUPS)})
    return p


def replacement_rates(p: pd.DataFrame, cfg: BoxConfig) -> dict[int, np.ndarray]:
    """{season: (groups x features) per-40 rates of the players under ``min_minutes`` in the seasons
    ``season - window .. season - 1``} (the first season: its own)."""
    f = list(FEATURES)
    ps = p.groupby(["player_id", "season"]).agg(minutes=("minutes", "sum"))
    low = ps[ps.minutes < cfg.min_minutes].reset_index()[["player_id", "season"]]
    lp = p.merge(low, on=["player_id", "season"])
    sums = lp.groupby(["season", "g"])[[*f, "e"]].sum()
    seasons = sorted(p.season.unique())
    out = {}
    for s in seasons:
        prev = [x for x in seasons if s - cfg.window <= x < s] or [s]
        tot = sums[sums.index.get_level_values(0).isin(prev)].groupby(level=1).sum()
        rates = np.zeros((len(GROUPS), len(f)))
        for gi in range(len(GROUPS)):
            if gi in tot.index and tot.loc[gi, "e"] > 0:
                rates[gi] = tot.loc[gi, f].to_numpy() / tot.loc[gi, "e"]
        out[int(s)] = rates
    return out


def rates_at(p: pd.DataFrame, cfg: BoxConfig, repl: dict[int, np.ndarray]) -> pd.DataFrame:
    """Every player's shrunk features as of each week he played in (his games before that week,
    recency-weighted for its season): ``player_id, season, week, minutes`` (weighted minutes behind
    the rates) and one column per feature (shrunk rate minus his group's replacement rate)."""
    f = list(FEATURES)
    wk = p.groupby(["player_id", "season", "week"], sort=True).agg(**{c: (c, "sum") for c in [*f, "e"]},
                                                                  g=("g", "last")).reset_index()
    # within-season sums before each week
    cur = wk.groupby(["player_id", "season"])[[*f, "e"]].cumsum() - wk[[*f, "e"]]
    # earlier seasons' totals, weighted by decay ** seasons back
    st = wk.groupby(["player_id", "season"])[[*f, "e"]].sum()
    prior = pd.DataFrame(0.0, index=st.index, columns=[*f, "e"])
    for k in range(1, cfg.window + 1):
        sh = st.copy()
        sh.index = pd.MultiIndex.from_arrays([sh.index.get_level_values(0), sh.index.get_level_values(1) + k])
        prior = prior.add(sh.reindex(prior.index).fillna(0.0) * cfg.decay ** k, fill_value=0.0)
    # seasons a player skipped: his earlier seasons still count (index by the seasons he played in)
    pri = prior.reindex(pd.MultiIndex.from_arrays([wk.player_id, wk.season])).to_numpy()
    tot = cur.to_numpy() + pri
    num, e = tot[:, :-1], tot[:, -1]
    rep = np.stack([repl[int(s)][int(g)] for s, g in zip(wk.season, wk.g)]) if len(wk) else np.zeros((0, len(f)))
    k = cfg.pseudo_minutes / REGULATION_MINUTES
    z = (num + k * rep) / (e + k)[:, None] - rep
    out = wk[["player_id", "season", "week"]].copy()
    out["minutes"] = e * REGULATION_MINUTES
    out[f] = z
    return out


def _skipped_seasons(p: pd.DataFrame, cfg: BoxConfig, repl: dict[int, np.ndarray], rates: pd.DataFrame,
                     players: np.ndarray, season: int, week: int) -> pd.DataFrame:
    """Features at ``week`` of ``season`` for ``players``: their games before the week, as
    :func:`rates_at` but for any cut-off (also players without a game in ``season`` yet)."""
    f = list(FEATURES)
    q = p[p.player_id.isin(players) & (p.season >= season - cfg.window) & (p.season <= season)
          & (p.week < week)]
    w = cfg.decay ** (season - q.season.to_numpy())
    num = pd.DataFrame(q[f].to_numpy() * w[:, None], columns=f).assign(player_id=q.player_id.to_numpy(),
                                                                         e=q.e.to_numpy() * w)
    tot = num.groupby("player_id").sum()
    g = p[p.player_id.isin(tot.index)].groupby("player_id").g.last().reindex(tot.index).to_numpy()
    rep = repl[int(season)][g.astype(int)] if len(g) else np.zeros((0, len(f)))
    k = cfg.pseudo_minutes / REGULATION_MINUTES
    z = (tot[f].to_numpy() + k * rep) / (tot.e.to_numpy() + k)[:, None] - rep
    out = pd.DataFrame(z, columns=f, index=tot.index).reset_index()
    out["minutes"] = tot.e.to_numpy() * REGULATION_MINUTES
    out["g"] = g
    return out


@dataclass
class BoxFit:
    """One cut-off's coefficients: ``coef_off`` / ``coef_def`` per feature (points per 100
    possessions per 40-minute unit of feature), ``scale`` (the features' standard deviations)."""
    before: int
    season: int
    coef_off: np.ndarray
    coef_def: np.ndarray
    n_games: int


def team_features(p: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """Every team-game's minutes-weighted sum of its players' features (as of the game's week):
    ``game_id, team_id`` plus one column per feature."""
    f = list(FEATURES)
    m = p[["game_id", "team_id", "player_id", "season", "week", "e"]].merge(
        rates[["player_id", "season", "week", *f]], on=["player_id", "season", "week"], how="left")
    m[f] = m[f].fillna(0.0).to_numpy() * m.e.to_numpy()[:, None]
    return m.groupby(["game_id", "team_id"])[f].sum().reset_index()


def fit_rows(tg: pd.DataFrame, tf: pd.DataFrame) -> pd.DataFrame:
    """Team-games usable in the fits, with the offense's (``o_*``) and the opponent's (``d_*``)
    features and the target ``y`` (points per 100 possessions)."""
    f = list(FEATURES)
    t = tg[(tg.minutes >= TEAM_MINUTES[0]) & (tg.minutes <= TEAM_MINUTES[1]) & (tg.poss >= MIN_POSS)
           & (tg.players >= 5) & tg.points.notna()].copy()
    t["t"] = pd.to_datetime(t.game_date).sub(pd.Timestamp("1970-01-01")).dt.days.astype(np.int64)
    o = tf.rename(columns={c: f"o_{c}" for c in f})
    d = tf.rename(columns={**{c: f"d_{c}" for c in f}, "team_id": "opponent_id"})
    t = t.merge(o, on=["game_id", "team_id"]).merge(d, on=["game_id", "opponent_id"])
    t["y"] = 100 * t.points / t.poss
    return t.sort_values("t", kind="stable").reset_index(drop=True)


def fit_box(rows: pd.DataFrame, before: int, season: int, cfg: BoxConfig) -> BoxFit:
    """Ridge of ``y`` on the offense and opponent features (standardized) plus home and season, on
    the rows with ``t < before`` in seasons ``season - window .. season``."""
    f = list(FEATURES)
    r = rows[(rows.t < before) & (rows.season >= season - cfg.window) & (rows.season <= season)]
    w = cfg.decay ** (season - r.season.to_numpy())
    X = r[[*(f"o_{c}" for c in f), *(f"d_{c}" for c in f)]].to_numpy(dtype=float)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    Z = X / sd
    seasons = np.sort(r.season.unique())
    ctrl = np.column_stack([r.home.to_numpy(dtype=float), *(r.season.to_numpy() == s for s in seasons)])
    A = np.column_stack([Z, ctrl])
    pen = np.r_[np.full(Z.shape[1], cfg.lam), np.full(ctrl.shape[1], 1e-6)]
    AtW = A.T * w
    b = np.linalg.solve(AtW @ A + np.diag(pen * w.sum() / len(w)), AtW @ r.y.to_numpy())
    coef = b[:Z.shape[1]] / sd
    n = len(f)
    return BoxFit(before, season, coef[:n], coef[n:], len(r))


def values_table(feat: pd.DataFrame, fit: BoxFit) -> pd.DataFrame:
    """``side, player_id, group, minutes, value`` (points per possession over replacement; higher is
    better on both sides) from players' features at the fit's cut-off."""
    Z = feat[list(FEATURES)].to_numpy(dtype=float)
    off = Z @ fit.coef_off / 100
    dfn = -(Z @ fit.coef_def) / 100
    grp = np.array(GROUPS)[feat.g.to_numpy(dtype=int)]
    base = {"player_id": feat.player_id.to_numpy(), "group": grp, "minutes": feat.minutes.to_numpy()}
    return pd.concat([pd.DataFrame(base | {"side": "off", "value": off}),
                      pd.DataFrame(base | {"side": "def", "value": dfn})], ignore_index=True)[
        ["side", "player_id", "group", "minutes", "value"]]


@dataclass
class BoxModel:
    """Everything the weekly fits need, built once from ``player_games`` and ``team_games``."""
    played: pd.DataFrame
    repl: dict
    rates: pd.DataFrame
    rows: pd.DataFrame
    cfg: BoxConfig

    @classmethod
    def build(cls, pg: pd.DataFrame, tg: pd.DataFrame, cfg: BoxConfig | None = None) -> BoxModel:
        cfg = cfg or BoxConfig()
        p = _played(pg)
        repl = replacement_rates(p, cfg)
        rates = rates_at(p, cfg, repl)
        rows = fit_rows(tg, team_features(p, rates))
        return cls(p, repl, rates, rows, cfg)

    def fit(self, before: int, season: int) -> BoxFit:
        return fit_box(self.rows, before, season, self.cfg)

    def features(self, before: int, season: int, players=None) -> pd.DataFrame:
        """Features at cut-off ``before`` (a Monday) of ``season`` for ``players`` (default: everyone
        who played in the window before it)."""
        p = self.played
        if players is None:
            players = p.loc[(p.season >= season - self.cfg.window) & (p.t < before), "player_id"].unique()
        return _skipped_seasons(p, self.cfg, self.repl, self.rates, np.asarray(players), season, before)

    def values(self, before: int, season: int, players=None) -> pd.DataFrame:
        return values_table(self.features(before, season, players), self.fit(before, season))


def weekly_values(model: BoxModel, befores: list[int], season_of) -> dict[int, pd.DataFrame]:
    """{cut-off: values table} for each Monday in ``befores`` (``season_of(before)``: its season)."""
    return {b: model.values(b, season_of(b)) for b in befores}


def holdout_score(model: BoxModel, season: int) -> dict:
    """Fit on everything before ``season`` and predict its team-games' points per 100 possessions
    from the players' values at the season's start, weighted by the minutes they played: the
    out-of-sample R² vs a per-season constant and the correlation. For tuning ``cfg``."""
    rows = model.rows[model.rows.season == season]
    before = int(week_start(rows.t.min()))
    v = model.values(before, season)
    off = dict(zip(v[v.side == "off"].player_id, v[v.side == "off"].value))
    dfn = dict(zip(v[v.side == "def"].player_id, v[v.side == "def"].value))
    p = model.played[model.played.season == season]
    p = p.assign(o=p.player_id.map(off).fillna(0.0) * p.e, d=p.player_id.map(dfn).fillna(0.0) * p.e)
    s = p.groupby(["game_id", "team_id"])[["o", "d"]].sum()
    r = rows.join(s, on=["game_id", "team_id"]).join(s.rename(columns={"o": "oo", "d": "od"}),
                                                      on=["game_id", "opponent_id"])
    pred = 100 * (r.o - r.od).fillna(0.0)
    y = r.y - r.y.mean()
    pred = pred - pred.mean()
    beta = float((pred * y).sum() / (pred ** 2).sum())
    return {"season": season, "r2": float(1 - ((y - pred) ** 2).mean() / (y ** 2).mean()),
            "corr": float(np.corrcoef(pred, y)[0, 1]), "slope": beta, "games": len(r)}
