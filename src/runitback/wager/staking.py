"""Confidence-weighted staking: how many units to bet on each game.

Flat betting stakes one unit on every pick, so a coin-flip pick right at the
line counts as much as one where the model disagrees with the market by a
touchdown. Here a staking policy sizes each bet from the model's edge (its
disagreement with the market) and is scored on profit at the posted odds,
weighted by the stake.

Policies that learn their stakes are fit on the model's out-of-sample
(walk-forward) predictions of earlier seasons only and scored on the next
season, the same no-lookahead loop as the backtest. Stakes are relative
units: each policy's average stake on its training bets is one unit, so its
total staked is comparable with flat betting's and ROI is profit per unit
staked.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing

from .backtest import payout
from .core.io import Table, read_table
from .core.schema import GameSchema
from .core.sport import get_sport

Market = Literal["spread", "moneyline", "total"]
MARKETS: tuple[str, ...] = ("spread", "moneyline", "total")
SPREAD_ODDS = -110.0


def market_bets(results: pd.DataFrame, schema: GameSchema, market: Market):
    """The model's pick on ``market`` for every game of a walk-forward
    results table (:func:`~runitback.wager.backtest.walk_forward`).

    Returns one row per game with a line and a non-zero edge: ``season``,
    ``date``, ``game_id``, ``edge`` (the model's disagreement with the
    market toward its pick: points for the spread and total, win
    probability for the moneyline; always > 0), ``odds`` (American, of the
    picked side), ``outcome`` (+1 the pick won, -1 lost, 0 push) and
    ``prob``: the model's probability that the pick wins, a push left out
    (from ``cover_prob`` / ``win_prob`` / ``over_prob``; NaN where the
    results lack it).
    """
    s = schema
    prob = None
    if market == "spread":
        frame = results.dropna(subset=["pred_margin", "market_margin", s.target])
        signed = frame["pred_margin"] - frame["market_margin"]
        went = frame[s.target] - frame["market_margin"]
        odds = np.full(len(frame), SPREAD_ODDS)
        if "cover_prob" in frame:
            prob = frame["cover_prob"]
    elif market == "moneyline":
        team, opp = s.market_moneyline
        frame = results.dropna(subset=["win_prob", "market_prob", team, opp, s.target])
        signed = frame["win_prob"] - frame["market_prob"]
        went = frame[s.target]
        odds = np.where(signed > 0, frame[team], frame[opp]).astype(float)
        prob = frame["win_prob"]
    elif market == "total":
        if "pred_total" not in results:
            raise ValueError(
                "the results have no pred_total (a model without the total)"
            )
        frame = results.dropna(subset=["pred_total", s.market_total, s.total])
        signed = frame["pred_total"] - frame[s.market_total]
        went = frame[s.total] - frame[s.market_total]
        odds = np.full(len(frame), SPREAD_ODDS)
        if s.market_total_odds and all(c in frame for c in s.market_total_odds):
            over, under = (frame[c].to_numpy(dtype=float) for c in s.market_total_odds)
            odds = np.where(signed > 0, over, under)
            odds = np.where(np.isfinite(odds), odds, SPREAD_ODDS)
        if "over_prob" in frame:
            prob = frame["over_prob"]
    else:
        raise ValueError(f"unknown market {market!r}; choose from {MARKETS}")
    bets = pd.DataFrame(
        {
            "season": frame[s.season].to_numpy(),
            "date": frame[s.date].to_numpy(),
            "game_id": frame[s.game_id].to_numpy(),
            "edge": np.abs(signed.to_numpy(dtype=float)),
            "odds": np.asarray(odds, dtype=float),
            "outcome": (np.sign(signed) * np.sign(went)).to_numpy(dtype=float),
            "prob": np.nan
            if prob is None
            else np.where(signed > 0, prob, 1 - prob).astype(float),
        }
    )
    return bets[bets["edge"] > 0].reset_index(drop=True)


def returns(bets: pd.DataFrame) -> np.ndarray:
    """Profit per unit staked on each bet: the payout, -1 or 0 (push)."""
    win = payout(bets["odds"].to_numpy())
    outcome = bets["outcome"].to_numpy()
    return np.where(outcome > 0, win, np.where(outcome < 0, -1.0, 0.0))


class Policy:
    """Units to stake on each bet; ``fit`` sees earlier seasons' bets only."""

    name = "policy"

    def fit(self, bets: pd.DataFrame) -> Policy:
        return self

    def stakes(self, bets: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {}


class Flat(Policy):
    """One unit on every pick (the usual record)."""

    name = "flat"

    def stakes(self, bets):
        return np.ones(len(bets))


@dataclass
class Threshold(Policy):
    """One unit when the edge exceeds ``min_edge``, nothing otherwise."""

    min_edge: float

    @property
    def name(self) -> str:
        return f"edge>{self.min_edge:g}"

    def stakes(self, bets):
        return (bets["edge"].to_numpy() > self.min_edge).astype(float)


@dataclass
class Proportional(Policy):
    """Stake proportional to the edge, one unit at the training mean edge."""

    name = "proportional"
    scale: float = 1.0

    def fit(self, bets):
        mean = float(bets["edge"].mean()) if len(bets) else 0.0
        self.scale = 1 / mean if mean > 0 else 0.0
        return self

    def stakes(self, bets):
        return bets["edge"].to_numpy() * self.scale


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def has_prob(bets: pd.DataFrame) -> bool:
    return "prob" in bets and len(bets) > 0 and bool(bets["prob"].notna().all())


def features(bets: pd.DataFrame) -> np.ndarray:
    """The learned policy's inputs: the edge, the log-odds of the win
    probability the pick's odds imply (constant at -110, but on the
    moneyline a big edge is usually an underdog, which wins less often) and,
    where the results have it, the log-odds of the model's own
    probability that the pick wins."""
    implied = 1 / (1 + payout(bets["odds"].to_numpy()))
    columns = [bets["edge"].to_numpy(), logit(implied)]
    if has_prob(bets):
        columns.append(logit(bets["prob"]))
    return np.column_stack(columns)


@dataclass
class LearnedKelly(Policy):
    """Learns P(pick wins | edge, odds) from earlier seasons and stakes the
    (fractional) Kelly bet at each pick's odds.

    A logistic regression of win (vs loss; pushes left out) on the edge and
    the odds' implied win probability (:func:`features`) gives each pick's
    win probability ``p``; at payout ``b`` the Kelly
    fraction is ``max(0, p - (1 - p) / b)``: nothing when the model's edge
    has not historically beaten the odds, more the larger an edge that has.
    Stakes are those fractions in relative units (mean 1 on the training
    bets that get a stake; all zero if none does). Kelly sizing maximizes
    the expected log bankroll, so it learns both whether to bet and how
    much.
    """

    name = "learned"
    fraction: float = 0.5
    max_units: float = 5.0
    model: sklearn.pipeline.Pipeline | None = None
    unit: float = 0.0
    coef: dict = field(default_factory=dict)

    def win_probability(self, bets) -> np.ndarray:
        if self.model is None:
            return np.full(len(bets), np.nan)
        return self.model.predict_proba(features(bets))[:, 1]

    def kelly(self, bets) -> np.ndarray:
        p = self.win_probability(bets)
        b = payout(bets["odds"].to_numpy())
        return self.fraction * np.clip(p - (1 - p) / b, 0, None)

    def fit(self, bets):
        decided = bets[bets["outcome"] != 0]
        self.model = None
        if decided["outcome"].nunique() == 2:
            # standardized, so the regularization means the same for points
            # (spread, total) and probabilities (moneyline)
            self.model = sklearn.pipeline.make_pipeline(
                sklearn.preprocessing.StandardScaler(),
                sklearn.linear_model.LogisticRegression(C=1.0),
            )
            self.model.fit(features(decided), decided["outcome"] > 0)
            # win probability at the median and 90th-percentile edge, at
            # the median odds (and model probability)
            edges = np.quantile(decided["edge"], [0.5, 0.9])
            typical = {"edge": edges, "odds": float(np.median(decided["odds"]))}
            if has_prob(decided):
                typical["prob"] = float(np.median(decided["prob"]))
            p50, p90 = self.win_probability(pd.DataFrame(typical))
            self.coef = {
                "edge_slope_per_sd": float(self.model[-1].coef_[0, 0]),
                "median_edge": float(edges[0]),
                "p_win_median_edge": float(p50),
                "p90_edge": float(edges[1]),
                "p_win_p90_edge": float(p90),
            }
        kelly = self.kelly(bets) if self.model is not None else np.zeros(len(bets))
        staked = kelly[kelly > 0]
        self.unit = float(staked.mean()) if len(staked) else 0.0
        return self

    def stakes(self, bets):
        if self.model is None or self.unit == 0:
            return np.zeros(len(bets))
        return np.clip(self.kelly(bets) / self.unit, 0, self.max_units)

    def describe(self) -> dict:
        return {**self.coef, "unit_fraction": self.unit}


@dataclass
class ModelKelly(Policy):
    """Takes the model's probability that the pick wins at face value and
    stakes the (fractional) Kelly bet at its odds: nothing where the model
    does not beat the odds' break-even. Nothing is learned but the unit
    (mean 1 on the training bets that get a stake)."""

    name = "model_kelly"
    fraction: float = 0.25
    max_units: float = 5.0
    unit: float = 0.0

    def kelly(self, bets) -> np.ndarray:
        if not has_prob(bets):
            return np.zeros(len(bets))
        p = bets["prob"].to_numpy()
        b = payout(bets["odds"].to_numpy())
        return self.fraction * np.clip(p - (1 - p) / b, 0, None)

    def fit(self, bets):
        kelly = self.kelly(bets)
        staked = kelly[kelly > 0]
        self.unit = float(staked.mean()) if len(staked) else 0.0
        return self

    def stakes(self, bets):
        if self.unit == 0:
            return np.zeros(len(bets))
        return np.clip(self.kelly(bets) / self.unit, 0, self.max_units)

    def describe(self) -> dict:
        return {"unit_fraction": self.unit}


def default_policies(market: Market, with_prob: bool = False) -> list[Policy]:
    """Flat, a few edge thresholds, proportional, learned (Kelly) and, for
    results with the model's probabilities, the model's own Kelly."""
    cuts = (0.03, 0.06) if market == "moneyline" else (2.0, 4.0)
    policies = [Flat(), *(Threshold(c) for c in cuts), Proportional(), LearnedKelly()]
    return policies + ([ModelKelly()] if with_prob else [])


def walk_forward_stakes(
    bets: pd.DataFrame,
    policies: list[Policy],
    min_train_seasons: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """Each policy's stake on every bet of every season that has at least
    ``min_train_seasons`` earlier seasons of bets to learn from.

    Returns ``(stakes, fits)``: the test bets with one ``stake_<name>``
    column per policy plus ``return`` (profit per unit), and each learned
    policy's fit per season (:meth:`Policy.describe`).
    """
    seasons = sorted(bets["season"].unique())
    out, fits = [], {}
    for i, season in enumerate(seasons):
        if i < min_train_seasons:
            continue
        train = bets[bets["season"] < season]
        test = bets[bets["season"] == season].copy()
        for policy in policies:
            policy.fit(train)
            test[f"stake_{policy.name}"] = policy.stakes(test)
            if policy.describe():
                fits.setdefault(policy.name, {})[int(season)] = policy.describe()
        out.append(test)
    if not out:
        raise ValueError(
            f"need more than {min_train_seasons} seasons of bets "
            f"(have {len(seasons)}): run the walk-forward backtest over more seasons"
        )
    stakes = pd.concat(out, ignore_index=True)
    stakes["return"] = returns(stakes)
    return stakes, fits


def max_drawdown(profit: np.ndarray) -> float:
    """Largest drop of cumulative profit from its running peak (units)."""
    cumulative = np.concatenate([[0.0], np.cumsum(profit)])
    return float(np.max(np.maximum.accumulate(cumulative) - cumulative))


def summarize_stakes(
    stakes: pd.DataFrame,
    names: list[str],
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Weighted profit of each policy on the same bets.

    Per policy: bets with a stake, units staked, profit, ROI (profit per
    unit staked) with a bootstrap 95% CI over games, the stake-weighted win
    rate, the largest drawdown (in date order) and the paired ROI difference
    vs flat betting with its bootstrap CI.
    """
    rng = np.random.default_rng(seed)
    n = len(stakes)
    draws = rng.integers(0, n, size=(n_boot, n))
    r = stakes["return"].to_numpy()
    # positions in season / date order (the index may be any labels)
    ordered = np.lexsort((stakes["date"].to_numpy(), stakes["season"].to_numpy()))
    decided = stakes["outcome"].to_numpy() != 0
    won = stakes["outcome"].to_numpy() > 0
    flat = stakes.get("stake_flat", pd.Series(np.ones(n))).to_numpy()

    def roi(stake, idx=None):
        if idx is None:
            total = stake.sum()
            return (stake * r).sum() / total if total else np.nan
        total = stake[idx].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (stake[idx] * r[idx]).sum(axis=1) / total

    flat_boot = roi(flat, draws)
    rows = []
    for name in names:
        stake = stakes[f"stake_{name}"].to_numpy()
        boot = roi(stake, draws)
        diff = boot - flat_boot
        weight = (stake * decided).sum()
        profit = stake * r
        rows.append(
            {
                "policy": name,
                "bets": int((stake > 0).sum()),
                "staked": float(stake.sum()),
                "profit": float(profit.sum()),
                "roi": roi(stake),
                "roi_ci95_low": float(np.nanpercentile(boot, 2.5)),
                "roi_ci95_high": float(np.nanpercentile(boot, 97.5)),
                "weighted_win_pct": float((stake * won).sum() / weight)
                if weight
                else np.nan,
                "max_drawdown": max_drawdown(profit[ordered]),
                "roi_vs_flat": roi(stake) - roi(flat),
                "roi_vs_flat_ci95_low": float(np.nanpercentile(diff, 2.5)),
                "roi_vs_flat_ci95_high": float(np.nanpercentile(diff, 97.5)),
            }
        )
    return pd.DataFrame(rows).set_index("policy")


def by_season(stakes: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Each policy's profit (units) per test season."""
    return pd.DataFrame(
        {
            name: (stakes[f"stake_{name}"] * stakes["return"])
            .groupby(stakes["season"])
            .sum()
            for name in names
        }
    )


def staking(
    results: Table,
    sport: str,
    markets: list[Market] | None = None,
    min_train_seasons: int = 3,
    output_dir: str | None = None,
    print_summary: bool = True,
) -> dict:
    """Score confidence-weighted staking on walk-forward backtest results.

    Args:
        results: Per-game walk-forward predictions, e.g. ``backtest``'s
            ``<family>_games.csv`` (several files' rows can be concatenated
            to cover more seasons). Learned stakes need at least
            ``min_train_seasons`` earlier seasons, so run the backtest over
            more seasons than you want scored.
        sport: Sport name, e.g. ``"nfl"``.
        markets: Markets to score (default: every market the results have).
        min_train_seasons: Seasons of earlier predictions each test season's
            policies learn from.
        output_dir: Write ``staking_summary.json`` and ``staking_bets.csv``.

    Returns:
        ``{market: {"summary", "by_season", "fits", "stakes"}}``
    """
    results = read_table(results)
    schema = get_sport(sport).schema
    if markets is None:
        markets = (["spread"] if schema.market_spread else []) + ["moneyline"]
        markets += ["total"] if "pred_total" in results else []
    report = {}
    for market in markets:
        bets = market_bets(results, schema, market)
        policies = default_policies(market, has_prob(bets))
        stakes, fits = walk_forward_stakes(bets, policies, min_train_seasons)
        names = [p.name for p in policies]
        report[market] = {
            "summary": summarize_stakes(stakes, names),
            "by_season": by_season(stakes, names),
            "fits": fits,
            "stakes": stakes.assign(market=market),
        }
    if output_dir:
        path = pathlib.Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        pd.concat([r["stakes"] for r in report.values()]).to_csv(
            path / "staking_bets.csv", index=False
        )
        payload = {
            market: {
                "summary": r["summary"].reset_index().to_dict("records"),
                "by_season": {
                    int(k): v for k, v in r["by_season"].to_dict("index").items()
                },
                "fits": r["fits"],
            }
            for market, r in report.items()
        }
        (path / "staking_summary.json").write_text(
            json.dumps(payload, indent=2, default=float)
        )
    if print_summary:
        print(format_staking(report))
    return report


def format_staking(report: dict) -> str:
    fmt = {"float_format": lambda v: f"{v:.3f}"}
    parts = []
    for market, r in report.items():
        seasons = sorted(r["by_season"].index)
        parts.append(
            f"== {market}: {len(r['stakes'])} picks, seasons {seasons[0]}-{seasons[-1]}"
            " (relative units: average stake 1)\n"
            + r["summary"].to_string(**fmt)
            + "\n\nProfit by season (units)\n"
            + r["by_season"].to_string(**fmt)
        )
        if "learned" in r["fits"]:
            fits = pd.DataFrame(r["fits"]["learned"]).T
            parts.append(
                "Learned win-probability fit by season\n" + fits.to_string(**fmt)
            )
    return "\n\n".join(parts)


def edge_grid(edges: np.ndarray, n: int = 12) -> np.ndarray:
    """``n`` edges from 0 to the 95th percentile, rounded to a readable step."""
    top = float(np.quantile(edges, 0.95)) if len(edges) else 0.0
    return np.round(np.linspace(0.0, top, n + 1)[1:], 4)


def performance_staking(
    results: pd.DataFrame,
    schema: GameSchema,
    min_train_seasons: int = 3,
) -> dict:
    """Edge-weighted staking for the web app's performance page.

    Per market the model could bet: the walk-forward scoring of every
    default policy (:func:`summarize_stakes`, each season's stakes learned
    from earlier seasons only), and ``curve``: the learned policy fit on all
    the bets, as units staked at a ladder of edges (at the median odds), so
    the page can show how the stake grows with the edge. Markets with too
    few seasons of bets are left out; ``{}`` if none has enough.
    """
    markets = (["spread"] if schema.market_spread else []) + ["moneyline"]
    markets += ["total"] if "pred_total" in results else []
    out = {}
    for market in markets:
        try:
            bets = market_bets(results, schema, market)
            policies = default_policies(market, has_prob(bets))
            stakes, fits = walk_forward_stakes(bets, policies, min_train_seasons)
        except (ValueError, KeyError):
            continue
        names = [p.name for p in policies]
        summary = summarize_stakes(stakes, names)
        learned = LearnedKelly().fit(bets)
        typical = {"odds": [float(bets["odds"].median())]}
        if has_prob(bets):
            typical["prob"] = [float(bets["prob"].median())]
        grid = edge_grid(bets["edge"].to_numpy())
        curve = learned.stakes(
            pd.DataFrame(
                {"edge": grid, **{k: v * len(grid) for k, v in typical.items()}}
            )
        )
        out[market] = {
            "seasons": sorted(int(x) for x in stakes["season"].unique()),
            "n": len(stakes),
            "policies": summary.reset_index().to_dict("records"),
            "curve": [
                {"edge": float(e), "units": float(u)} for e, u in zip(grid, curve)
            ],
            "max_units": learned.max_units,
            "fraction": learned.fraction,
            "fit": fits.get("learned", {}).get(int(stakes["season"].max()), {}),
        }
    return out
