"""The data behind the web app's pages: one :class:`Snapshot` per sport.

A snapshot directory (written by :mod:`runitback.wager.web.build`) holds:

* ``games.csv``: the processed games table (played and upcoming);
* ``teams.json``: team display metadata;
* ``model/``: the saved predictor;
* ``pool.csv``: out-of-sample walk-forward predictions for historical games
  (the challenge pool, and the model performance page);
* ``rankings.csv``: the walk-forward models' power ratings before each date
  of the backtest seasons (the current season's are computed on load);
* ``meta.json``: sport, build time and backtest summary.
"""

from __future__ import annotations

import datetime
import json
import math
import pathlib
import random
from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..core.io import read_table
from ..core.predictor import Predictor, load_predictor
from ..core.schema import one_row_per_game, played, upcoming
from ..core.sport import get_sport
from ..inference.utils import power_ratings
from .lineups import LineupEditor
from .views import SportView, get_view


def py(value):
    """JSON-safe Python value (numpy scalars -> builtins, NaN -> None)."""
    if isinstance(value, dict):
        return {k: py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [py(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


ATS_RESULT = {"home": "W", "away": "L", "push": "P"}

# The markets a game can be played on in the challenge.
MARKETS = ("spread", "moneyline", "total")
# Odds assumed for spread bets.
SPREAD_ODDS = -110.0


def cover_side(margin: float, spread: float) -> str | None:
    """Which side covered: ``spread`` is the home team's expected margin."""
    if spread is None or pd.isna(spread) or pd.isna(margin):
        return None
    if margin > spread:
        return "home"
    if margin < spread:
        return "away"
    return "push"


def winner_side(margin: float) -> str | None:
    if margin is None or pd.isna(margin):
        return None
    return "home" if margin > 0 else "away" if margin < 0 else "push"


def total_side(total: float, line: float) -> str | None:
    """``over`` / ``under`` / ``push`` of combined points against the line."""
    if line is None or total is None or pd.isna(line) or pd.isna(total):
        return None
    return "over" if total > line else "under" if total < line else "push"


def profit(odds: float, won: bool) -> float:
    """Units won (or -1 lost) on 1 unit at American ``odds``."""
    if not won:
        return -1.0
    return odds / 100 if odds > 0 else 100 / -odds


def _value(row, column):
    """A row's numeric value, None if the column is unset, absent or NaN."""
    if column is None:
        return None
    value = row.get(column)
    return None if value is None or pd.isna(value) else float(value)


class Snapshot:
    """Everything the site shows for one sport (``web export`` writes it out).

    Upcoming-game predictions are computed once on construction; team
    summaries are computed per request from games strictly before the game
    being shown, so historical (challenge) games never reveal their result.
    """

    def __init__(
        self,
        view: SportView,
        games: pd.DataFrame,
        predictor: Predictor,
        pool: pd.DataFrame,
        teams: pd.DataFrame,
        meta: dict | None = None,
        model_dir: str | None = None,
        rankings: pd.DataFrame | None = None,
    ):
        self.view = view
        self.schema = schema = get_sport(view.sport).schema
        self.games = games.reset_index(drop=True)
        # ids arrive as strings in URLs; keep them strings everywhere
        self.games[schema.game_id] = self.games[schema.game_id].astype(str)
        self.played = played(self.games, schema)
        # each team's played rows (positions), for team_summary: a scan of
        # the whole table per summary dominated the challenge export
        self._team_played = self.played.groupby(schema.team).indices
        # ... and only the columns it reads (taking every column of a wide
        # table per summary was most of the rest)
        columns = [
            schema.game_id, schema.season, schema.date, schema.team, schema.opp,
            schema.target, schema.location, schema.market_spread, *view.score,
        ]  # fmt: skip
        self._summary_rows = self.played[
            list(dict.fromkeys(c for c in columns if c and c in self.played))
        ]
        self.teams = {row["team"]: py(row.to_dict()) for _, row in teams.iterrows()}
        self.meta = meta or {}
        self.model_dir = model_dir
        self.sigma = predictor.sigma
        self.total_sigma = getattr(predictor, "total_sigma", None)
        self._rows = self.games.groupby(schema.game_id).indices
        self.stat_values = pd.DataFrame(
            {stat.key: stat.value(self.played) for stat in view.stats},
            index=self.played.index,
        )
        self.playoffs = view.playoffs(self.played) if view.playoffs else {}
        self.date_names = self._date_names()

        if len(pool):
            pool = pool.assign(**{schema.game_id: pool[schema.game_id].astype(str)})
        self.results = pool  # every backtest game, pushes included
        self.performance_summary = self._performance() if len(pool) else None
        self.trust = self._trust() if len(pool) else {}
        if len(pool) and "market_margin" in pool:
            # lined games; a push on one market can still be played on another
            pool = pool.dropna(subset=["market_margin"])
        self.pool = pool.set_index(schema.game_id) if len(pool) else pool
        self._markets: dict = {}

        self.baked_rankings = rankings if rankings is not None else pd.DataFrame()
        self.rankings_table = self._with_current_rankings(predictor)
        self.team_map = self._team_map(predictor)
        self.recent_games = self._predict_recent(predictor)
        self._recent_rows = {
            g: i for i, g in enumerate(self.recent_games.get(schema.game_id, []))
        }
        self.upcoming_games = self._predict_upcoming(predictor)
        self.lineup_editor = LineupEditor.load(predictor)

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(
        cls, directory: str | pathlib.Path, games: pd.DataFrame | None = None
    ) -> Snapshot:
        """The snapshot in ``directory``; with ``games``, built from those
        (freshly downloaded) games instead of the saved ``games.csv``."""
        path = pathlib.Path(directory)
        meta = json.loads((path / "meta.json").read_text())
        if games is not None:
            meta = {**meta, "refreshed_at": _now()}
        pool_path = path / "pool.csv"
        rankings_path = path / "rankings.csv"
        return cls(
            view=get_view(meta["sport"]),
            games=read_table(str(path / "games.csv")) if games is None else games,
            predictor=load_predictor(str(path / "model")),
            pool=read_table(str(pool_path)) if pool_path.exists() else pd.DataFrame(),
            teams=pd.DataFrame(json.loads((path / "teams.json").read_text())),
            meta=meta,
            model_dir=str(path / "model"),
            rankings=read_table(str(rankings_path)) if rankings_path.exists() else None,
        )

    def refreshed(self, games: pd.DataFrame) -> Snapshot:
        """A new snapshot with newer games (same model, pool and teams)."""
        teams = pd.DataFrame(list(self.teams.values()))
        meta = {**self.meta, "refreshed_at": _now()}
        return Snapshot(
            self.view,
            games,
            load_predictor(self.model_dir),
            self.results,
            teams,
            meta,
            self.model_dir,
            self.baked_rankings,
        )

    # ----------------------------------------------------------------- pieces
    def team(self, key) -> dict:
        info = self.teams.get(key)
        return py(info) if info else {"team": key, "name": str(key)}

    def stat_definitions(self) -> list[dict]:
        return [
            {
                "key": s.key,
                "label": s.label,
                "decimals": s.decimals,
                "higher_is_better": s.higher_is_better,
            }
            for s in self.view.stats
        ]

    def _row(self, game_id, team=None) -> pd.Series | None:
        """One perspective of a game: ``team``'s if given, else the home team's."""
        index = self._rows.get(game_id)
        if index is None:
            return None
        rows = self.games.iloc[index]
        if team is not None:
            rows = rows[rows[self.schema.team] == team]
        return one_row_per_game(rows, self.schema).iloc[0] if len(rows) else None

    def _date_names(self) -> dict:
        """``{(season, date): name}`` from the view's ``date_name``."""
        if self.view.date_name is None:
            return {}
        s = self.schema
        firsts = self.games.drop_duplicates([s.season, s.date])
        return {
            (py(row[s.season]), py(row[s.date])): self.view.date_name(row)
            for _, row in firsts.iterrows()
        }

    def date_name(self, season, date) -> str | None:
        return self.date_names.get((py(season), py(date)))

    def _spread(self, row: pd.Series):
        s = self.schema
        return py(row.get(s.market_spread)) if s.market_spread else None

    def _lines(self, row: pd.Series) -> dict:
        """The market's moneylines, total line and over/under odds."""
        s = self.schema
        moneyline = None
        if s.market_moneyline:
            home, away = (_value(row, c) for c in s.market_moneyline)
            if home is not None and away is not None:
                moneyline = {"home": home, "away": away}
        total_odds = None
        if s.market_total_odds:
            over, under = (_value(row, c) for c in s.market_total_odds)
            if over is not None and under is not None:
                total_odds = {"over": over, "under": under}
        return {
            "moneyline": moneyline,
            "total_line": _value(row, s.market_total),
            "total_odds": total_odds,
        }

    def _header(self, row: pd.Series) -> dict:
        s = self.schema
        location = row[s.location] if s.location else 1
        return {
            "game_id": py(row[s.game_id]),
            "season": py(row[s.season]),
            "date": py(row[s.date]),
            "date_name": self.date_name(row[s.season], row[s.date]),
            "kickoff": self.view.kickoff(row),
            "home": self.team(row[s.team]),
            "away": self.team(row[s.opp]),
            "neutral": bool(location == 0),
            "spread": self._spread(row),
            **self._lines(row),
        }

    def _prediction(self, pred, header: dict) -> dict:
        """The model's numbers for a game and its pick in each market.

        ``pred`` has ``pred_margin`` / ``win_prob`` (and ``pred_total`` /
        ``over_prob`` from a model that predicts the total); ``header`` the
        game's lines (:meth:`_header`).
        """
        from ..backtest import no_vig_probability

        margin, win_prob = pred["pred_margin"], pred["win_prob"]
        spread = header["spread"]
        edge = None if spread is None else margin - spread
        pick = None
        if edge is not None and edge != 0:
            pick = "home" if edge > 0 else "away"
        out = {"margin": margin, "win_prob": win_prob, "edge": edge, "ats_pick": pick}

        # moneyline: the side whose model probability beats the market's
        market_prob = ml_edge = ml_pick = None
        if header["moneyline"]:
            ml = header["moneyline"]
            market_prob = float(no_vig_probability(ml["home"], ml["away"]))
            ml_edge = win_prob - market_prob
            ml_pick = "home" if ml_edge > 0 else "away" if ml_edge < 0 else None
        out |= {"market_prob": market_prob, "ml_edge": ml_edge, "ml_pick": ml_pick}

        total = _value(pred, "pred_total")
        line = header["total_line"]
        total_edge = ou_pick = None
        if total is not None and line is not None:
            total_edge = total - line
            ou_pick = "over" if total_edge > 0 else "under" if total_edge < 0 else None
        out |= {
            "total": total,
            "over_prob": _value(pred, "over_prob"),
            "total_edge": total_edge,
            "ou_pick": ou_pick,
        }
        return py(out)

    def _result(self, row: pd.Series) -> dict | None:
        s = self.schema
        if pd.isna(row[s.target]):
            return None
        score, opp_score = self.view.score
        home, away = _value(row, score), _value(row, opp_score)
        total = _value(row, s.total)
        if total is None and home is not None and away is not None:
            total = home + away
        return py(
            {
                "home_score": row.get(score),
                "away_score": row.get(opp_score),
                "margin": row[s.target],
                "covered": cover_side(row[s.target], self._spread(row)),
                "winner": winner_side(row[s.target]),
                "total": total,
                "went": total_side(total, _value(row, s.market_total)),
            }
        )

    def _with_current_rankings(self, predictor: Predictor) -> pd.DataFrame:
        """Baked rankings plus the current season's, from the saved model.

        One rating per team before each played date of the season (or
        those :func:`~runitback.wager.web.build.ranking_dates` keeps) and before its
        next date, from the games played until then (like the backtest).
        """
        from .build import ranking_dates

        s = self.schema
        baked = self.baked_rankings
        season = (
            upcoming(self.games, s)[s.season].max()
            if len(upcoming(self.games, s))
            else self.played[s.season].max()
        )
        if pd.isna(season) or (len(baked) and season in set(baked["season"])):
            return baked
        current = self.games[self.games[s.season] == season]
        dates = ranking_dates(
            current.loc[current[s.target].notna(), s.date].unique(),
            self.view.ranking_every,
        )
        later = current.loc[current[s.target].isna(), s.date]
        # a date with games still to play may be in progress: ranked once
        if len(later) and later.min() not in set(dates):
            dates.append(later.min())
        playing = current[s.team].unique()
        rows = []
        for date in dates:
            known = self.played[
                (self.played[s.season] < season)
                | ((self.played[s.season] == season) & (self.played[s.date] < date))
            ]
            predictor.update_state(known)
            for team, rating in power_ratings(predictor, playing).items():
                rows.append(
                    {"season": season, "date": date, "team": team, "rating": rating}
                )
        return pd.concat([baked, pd.DataFrame(rows)], ignore_index=True)

    def _team_map(self, predictor: Predictor) -> dict | None:
        """The latest ranked season's teams in 2-D, before each of its
        ranking dates: the served model's team embeddings from the games
        played until then, every (team, date) reduced together with UMAP so
        all dates share one space and a team's path is its movement. Each
        team also carries, per date, the metrics the map can be colored by
        (:meth:`_map_metrics`). ``None`` for a model without embeddings or
        too few teams.
        """
        table = self.rankings_table
        if not len(table):
            return None
        s = self.schema
        season = table["season"].max()
        frame = table[table["season"] == season]
        ratings = frame.pivot_table(index="team", columns="date", values="rating")
        dates = list(ratings.columns)
        vectors, cells = [], []
        for i, date in enumerate(dates):
            known = self.played[
                (self.played[s.season] < season)
                | ((self.played[s.season] == season) & (self.played[s.date] < date))
            ]
            predictor.update_state(known)
            embeddings = predictor.model.team_embeddings()
            if embeddings is None:
                return None
            rated = ratings[date].dropna()
            embeddings = embeddings.loc[embeddings.index.isin(rated.index)]
            vectors.append(embeddings.to_numpy(dtype=np.float32))
            cells += [(team, i) for team in embeddings.index]
        if len({team for team, _ in cells}) < MIN_MAP_TEAMS:
            return None
        xy = umap_layout(np.concatenate(vectors), dims=2)

        positions = {team: [None] * len(dates) for team, _ in cells}
        for (team, i), point in zip(cells, xy):
            positions[team][i] = [round(float(v), 4) for v in point]
        metrics, values, records = self._map_metrics(season, dates, ratings)
        return {
            "season": season,
            "dates": dates,
            "date_names": [self.date_name(season, d) for d in dates]
            if self.date_names
            else None,
            "metrics": metrics,
            "teams": [
                {
                    "team": team,
                    "positions": positions[team],
                    "values": {key: v.get(team) for key, v in values.items()},
                    "records": records.get(team),
                }
                for team in sorted(positions, key=str)
            ],
        }

    def _map_metrics(self, season, dates: list, ratings: pd.DataFrame):
        """What the team map can be colored by, each a value per team per
        date from the season's games before that date: the model's average
        predicted margin, the actual average margin, win percentage (a tie
        counts half) and the view's stats per game. Returns the metric
        definitions, ``{key: {team: [value per date]}}`` and each team's
        ``[wins, losses, ties]`` per date.

        ``center`` says where the color scale is neutral: ``"zero"`` for
        margins, ``"half"`` for win percentage, ``"mean"`` (the league
        average at that date) for stats.
        """
        s = self.schema
        metrics = [
            {"key": "rating", "label": "Predicted margin", "decimals": 1,
             "higher_is_better": True, "center": "zero"},
            {"key": "margin", "label": "Actual margin", "decimals": 1,
             "higher_is_better": True, "center": "zero"},
            {"key": "win_pct", "label": "Win %", "decimals": 3,
             "higher_is_better": True, "center": "half"},
        ]  # fmt: skip
        stats = [
            st for st in self.view.stats if st.key not in {m["key"] for m in metrics}
        ]
        metrics += [
            {"key": st.key, "label": st.label, "decimals": st.decimals,
             "higher_is_better": st.higher_is_better, "center": "mean"}
            for st in stats
        ]  # fmt: skip

        def rounded(series, decimals):
            return {
                team: py(round(float(v), decimals + 1)) for team, v in series.items()
            }

        teams = list(ratings.index)
        values = {m["key"]: {t: [None] * len(dates) for t in teams} for m in metrics}
        records = {t: [[0, 0, 0] for _ in dates] for t in teams}
        current = self.played[self.played[s.season] == season]
        stat_rows = self.stat_values.loc[current.index]
        for i, date in enumerate(dates):
            for team, v in rounded(ratings[date].dropna(), 1).items():
                values["rating"][team][i] = v
            before = current[current[s.date] < date]
            if not len(before):
                continue
            by_team = before.groupby(s.team)
            margin = before[s.target]
            win = (margin > 0) + 0.5 * (margin == 0)
            per_team = {
                "margin": (by_team[s.target].mean(), 1),
                "win_pct": (win.groupby(before[s.team]).mean(), 3),
            }
            stat_means = stat_rows.loc[before.index].groupby(before[s.team]).mean()
            for st in stats:
                per_team[st.key] = (stat_means[st.key].dropna(), st.decimals)
            for key, (series, decimals) in per_team.items():
                for team, v in rounded(series, decimals).items():
                    if team in values[key]:
                        values[key][team][i] = v
            counts = (
                pd.DataFrame({"w": margin > 0, "l": margin < 0, "t": margin == 0})
                .groupby(before[s.team])
                .sum()
            )
            for team, row in counts.iterrows():
                if team in records:
                    records[team][i] = [int(row["w"]), int(row["l"]), int(row["t"])]
        return metrics, values, records

    def team_map_view(self) -> dict | None:
        """The team map with team display objects, or ``None``."""
        if self.team_map is None:
            return None
        teams = [{**t, "team": self.team(t["team"])} for t in self.team_map["teams"]]
        return py({**self.team_map, "teams": teams})

    def _performance(self) -> dict:
        """Walk-forward backtest metrics: overall, by season, each market's
        bets, calibration (of win and over probabilities)."""
        from ..backtest import ats_record, moneyline_record, summarize, totals_record

        s = self.schema
        results = self.results
        summary = summarize(results, s)
        if "market_prob" in results:
            lined = results.dropna(subset=["market_margin", "market_prob"])
            by_season = lined.groupby(s.season)
            if s.market_spread:
                summary["ats_by_season"] = {
                    int(season): ats_record(frame, s, thresholds=(0,)).iloc[0].to_dict()
                    for season, frame in by_season
                }
            if "moneyline" in summary:
                summary["moneyline_by_season"] = {
                    int(season): moneyline_record(frame, s, thresholds=(0,))
                    .iloc[0]
                    .to_dict()
                    for season, frame in by_season
                }
            if "totals" in summary:
                summary["totals_by_season"] = {
                    int(season): totals_record(frame, s, thresholds=(0,))
                    .iloc[0]
                    .to_dict()
                    for season, frame in by_season
                }
        else:
            lined = results
        decided = lined[lined[s.target] != 0]
        won = decided[s.target] > 0
        sources = {"model": "win_prob"}
        if "market_prob" in decided:
            sources["market"] = "market_prob"
        summary["calibration"] = {
            name: _calibration(decided[column], won) for name, column in sources.items()
        }
        if "totals" in summary:
            totals = lined.dropna(subset=["over_prob", s.market_total, s.total])
            totals = totals[totals[s.total] != totals[s.market_total]]
            over = totals[s.total] > totals[s.market_total]
            sources = {"model": "over_prob"}
            if totals.get("market_over_prob", pd.Series(dtype=float)).notna().any():
                sources["market"] = "market_over_prob"
            summary["over_calibration"] = {
                name: _calibration(totals[column], over)
                for name, column in sources.items()
            }
        summary["seasons"] = sorted(int(x) for x in results[s.season].unique())
        from ..staking import performance_staking

        if staked := performance_staking(results, s):
            summary["staking"] = staked
        return py(summary)

    def _trust(self) -> dict:
        """How much of the model's disagreement with the market to believe,
        per market, fit on the walk-forward backtest (the browser prices
        any line with it).

        A probability is moved toward the market's by ``1 - k``: ``p = m +
        k (p_model - m)``, with ``m`` 0.5 for the spread and total (the
        market's line) and the no-vig win probability for the moneyline.
        ``k`` minimizes the Brier score on the backtest's decided games
        (clipped to 0-1): 0 when the disagreements have not paid, 1 when
        they were as right as the model claimed.
        """
        from scipy.stats import norm

        s, r = self.schema, self.results

        def fit(prob, market, happened):
            ok = np.isfinite(prob) & np.isfinite(market) & np.isfinite(happened)
            gap = (prob - market)[ok]
            denom = float((gap**2).sum())
            if not denom:
                return None
            k = float((gap * (happened - market)[ok]).sum()) / denom
            return float(np.clip(k, 0, 1))

        out = {}
        if s.market_spread and "market_margin" in r:
            lined = r.dropna(subset=["market_margin"])
            lined = lined[lined[s.target] != lined["market_margin"]]
            prob = norm.cdf(
                (lined["pred_margin"] - lined["market_margin"]) / self.sigma
            )
            covered = (lined[s.target] > lined["market_margin"]).astype(float)
            out["spread"] = fit(prob, 0.5, covered.to_numpy())
        if "market_prob" in r:
            decided = r.dropna(subset=["market_prob"])
            decided = decided[decided[s.target] != 0]
            won = (decided[s.target] > 0).astype(float).to_numpy()
            out["moneyline"] = fit(
                decided["win_prob"].to_numpy(), decided["market_prob"].to_numpy(), won
            )
        if self.total_sigma and "pred_total" in r and s.market_total in r:
            lined = r.dropna(subset=["pred_total", s.market_total, s.total])
            lined = lined[lined[s.total] != lined[s.market_total]]
            prob = norm.cdf(
                (lined["pred_total"] - lined[s.market_total]) / self.total_sigma
            )
            over = (lined[s.total] > lined[s.market_total]).astype(float)
            out["total"] = fit(prob, 0.5, over.to_numpy())
        return {k: v for k, v in out.items() if v is not None}

    def _predict_recent(self, predictor: Predictor) -> pd.DataFrame:
        """Pre-game predictions for the games of the last
        ``view.recent_dates`` played dates, one row per game.

        A game the walk-forward backtest covers keeps its backtest
        prediction (``prediction_source`` "backtest"); the others are
        replayed with the served model from team state built on the games
        before their date only ("replay"), the way the upcoming games were
        predicted. The served model never trains on the latest season (it
        is the test season of :func:`~runitback.wager.core.splits.split_seasons`),
        so a replay of a current-season game is out of sample.
        """
        s = self.schema
        rows = one_row_per_game(self.played, s)
        keys = (
            rows[[s.season, s.date]]
            .drop_duplicates()
            .sort_values([s.season, s.date])
            .tail(self.view.recent_dates)
        )
        tested = (
            self.results.drop_duplicates(s.game_id).set_index(s.game_id)
            if len(self.results)
            else None
        )
        pred_columns = ["pred_margin", "win_prob", "pred_total", "over_prob"]
        frames = []
        for season, date in keys.itertuples(index=False):
            day = rows[(rows[s.season] == season) & (rows[s.date] == date)]
            in_pool = (
                day[s.game_id].isin(tested.index)
                if tested is not None
                else pd.Series(False, index=day.index)
            )
            for game_id in day.loc[in_pool, s.game_id]:
                pool = tested.loc[game_id]
                row = self._row(game_id, pool[s.team])
                preds = {c: pool[c] for c in pred_columns if c in pool}
                frames.append(
                    pd.DataFrame([{**row.to_dict(), **preds}]).assign(
                        prediction_source="backtest"
                    )
                )
            rest = day[~in_pool]
            if rest.empty:
                continue
            known = self.played[
                (self.played[s.season] < season)
                | ((self.played[s.season] == season) & (self.played[s.date] < date))
            ]
            if known.empty:
                continue
            predictor.update_state(known)
            teams = set(predictor.model.teams)
            rest = rest[rest[s.team].isin(teams) & rest[s.opp].isin(teams)]
            if len(rest):
                frames.append(
                    predictor.predict(rest).assign(prediction_source="replay")
                )
        if not frames:
            return pd.DataFrame(columns=[*rows.columns, "pred_margin", "win_prob"])
        out = pd.concat(frames, ignore_index=True)
        out[s.game_id] = out[s.game_id].astype(str)
        return out

    def _predict_upcoming(self, predictor: Predictor) -> pd.DataFrame:
        predictor.update_state(self.games)
        rows = one_row_per_game(upcoming(self.games, self.schema), self.schema)
        # a game whose kickoff day has passed is not upcoming, even while its
        # result is missing (still in progress at download, or a late refresh)
        today = _today_eastern()
        past = [
            (kickoff := self.view.kickoff(row)) is not None and kickoff[:10] < today
            for _, row in rows.iterrows()
        ]
        rows = rows[~np.array(past, dtype=bool)]
        known = set(predictor.model.teams)
        rows = rows[
            rows[self.schema.team].isin(known) & rows[self.schema.opp].isin(known)
        ]
        if rows.empty:
            return rows.assign(pred_margin=[], win_prob=[])
        preds = predictor.predict(rows)
        return preds.set_index(rows.index)

    # ------------------------------------------------------------- summaries
    def team_summary(self, team, season, date) -> dict:
        """``team`` as of just before ``(season, date)``: only earlier games."""
        s, view = self.schema, self.view
        mine = self._summary_rows.iloc[self._team_played.get(team, [])]
        before = (mine[s.season] < season) | (
            (mine[s.season] == season) & (mine[s.date] < date)
        )
        history = mine[before].sort_values([s.season, s.date])
        basis = season
        current = history[history[s.season] == season]
        if current.empty and not history.empty:
            basis = history[s.season].max()
            current = history[history[s.season] == basis]

        margin = current[s.target]
        record = {
            "wins": int((margin > 0).sum()),
            "losses": int((margin < 0).sum()),
            "ties": int((margin == 0).sum()),
        }
        ats = {"wins": 0, "losses": 0, "pushes": 0}
        if s.market_spread:
            lined = current.dropna(subset=[s.market_spread])
            cover = lined[s.target] - lined[s.market_spread]
            ats = {
                "wins": int((cover > 0).sum()),
                "losses": int((cover < 0).sum()),
                "pushes": int((cover == 0).sum()),
            }
        averages = self.stat_values.loc[current.index].mean().round(3)

        score, opp_score = view.score
        recent = []
        for _, row in history.tail(view.recent_games).iloc[::-1].iterrows():
            spread = self._spread(row)
            recent.append(
                {
                    "game_id": row[s.game_id],
                    "season": row[s.season],
                    "date": row[s.date],
                    "date_name": self.date_name(row[s.season], row[s.date]),
                    "opp": self.team(row[s.opp]),
                    "location": row[s.location] if s.location else None,
                    "score": row.get(score),
                    "opp_score": row.get(opp_score),
                    "margin": row[s.target],
                    "spread": spread,
                    # this team against the spread: "W", "L" or "P"
                    "ats": ATS_RESULT.get(cover_side(row[s.target], spread)),
                }
            )
        return py(
            {
                "team": self.team(team),
                "season": basis,
                "current_season": bool(basis == season),
                "games": len(current),
                "record": record,
                "ats": ats,
                "averages": {k: averages.get(k) for k in self.stat_values},
                "recent": recent,
            }
        )

    def _summaries(self, row: pd.Series) -> dict:
        s = self.schema
        season, date = row[s.season], row[s.date]
        return {
            "home": self.team_summary(row[s.team], season, date),
            "away": self.team_summary(row[s.opp], season, date),
        }

    # ------------------------------------------------------------ endpoints
    def upcoming(self, season=None, date=None) -> dict:
        s = self.schema
        games = self.upcoming_games
        if games.empty:
            return {"season": None, "date": None, "dates": [], "games": []}
        season = games[s.season].max() if season is None else season
        games = games[games[s.season] == season]
        dates = sorted(games[s.date].unique())[: self.view.upcoming_dates]
        date = dates[0] if date is None else date
        week = games[games[s.date] == date]
        cards = []
        for _, row in week.iterrows():
            header = self._header(row)
            header["prediction"] = self._prediction(row, header)
            cards.append(header)
        cards.sort(key=lambda g: (g["kickoff"] or "", str(g["game_id"])))
        names = {py(d): self.date_name(season, d) for d in dates}
        return py(
            {
                "season": season,
                "date": date,
                "dates": dates,
                "date_names": names if any(names.values()) else None,
                "games": cards,
            }
        )

    def recent_dates(self) -> list[dict]:
        """The recent-games page's dates, newest first:
        ``[{"season", "date", "name"}]``."""
        s = self.schema
        games = self.recent_games
        if games.empty:
            return []
        keys = games[[s.season, s.date]].drop_duplicates()
        keys = keys.sort_values([s.season, s.date], ascending=False)
        return py(
            [
                {"season": season, "date": date, "name": self.date_name(season, date)}
                for season, date in keys.itertuples(index=False)
            ]
        )

    def recent(self, season=None, date=None) -> dict:
        """One recent date's played games: lines, the model's pre-game
        prediction (against the lines the result is graded on), the result
        and where the prediction came from (``prediction_source``:
        "backtest" or "replay", see :meth:`_predict_recent`)."""
        s = self.schema
        dates = self.recent_dates()
        if not dates:
            return {"season": None, "date": None, "dates": [], "games": []}
        if season is None or date is None:
            season, date = dates[0]["season"], dates[0]["date"]
        games = self.recent_games
        day = games[(games[s.season] == season) & (games[s.date] == date)]
        cards = []
        for _, row in day.iterrows():
            header = self._header(row)
            cards.append(
                {
                    **header,
                    "prediction": self._prediction(row, header),
                    "result": self._result(row),
                    "prediction_source": row["prediction_source"],
                }
            )
        cards.sort(key=lambda g: (g["kickoff"] or "", str(g["game_id"])))
        return py({"season": season, "date": date, "dates": dates, "games": cards})

    def game(self, game_id) -> dict | None:
        """Header, prediction, result (if played) and both teams' summaries."""
        s = self.schema
        in_pool = len(self.pool) and game_id in self.pool.index
        recent = self._recent_rows.get(game_id)
        team = None
        if in_pool:
            team = self.pool.loc[game_id, s.team]
        elif recent is not None:  # the perspective its prediction is from
            team = self.recent_games.iloc[recent][s.team]
        row = self._row(game_id, team)
        if row is None:
            return None
        header = self._header(row)
        prediction = None
        lineups = None
        if game_id in set(self.upcoming_games.get(s.game_id, [])):
            pred = self.upcoming_games[self.upcoming_games[s.game_id] == game_id].iloc[
                0
            ]
            prediction = self._prediction(pred, header)
            if self.lineup_editor is not None:
                lineups = self.lineup_editor.game(
                    game_id, header["home"]["team"], header["away"]["team"]
                )
        elif in_pool:
            prediction = self._prediction(self.pool.loc[game_id], header)
        elif recent is not None:
            prediction = self._prediction(self.recent_games.iloc[recent], header)
        return py(
            {
                **header,
                "prediction": prediction,
                **({"lineups": lineups} if lineups else {}),
                "result": self._result(row),
                "summaries": self._summaries(row),
                "stats": self.stat_definitions(),
            }
        )

    def markets(self, game_id) -> list[str]:
        """Markets a pool game can be played on: it has the line (and, for the
        total, a model total) and did not land on it."""
        if game_id in self._markets:
            return self._markets[game_id]
        s = self.schema
        pool = self.pool.loc[game_id]
        row = self._row(game_id, pool[s.team])
        lines = self._lines(row)
        result = self._result(row)
        out = []
        if result["covered"] in ("home", "away"):
            out.append("spread")
        if lines["moneyline"] and result["winner"] in ("home", "away"):
            out.append("moneyline")
        has_total = "pred_total" in pool and not pd.isna(pool["pred_total"])
        if has_total and result["went"] in ("over", "under"):
            out.append("total")
        self._markets[game_id] = out
        return out

    def challenge_next(
        self, rng: random.Random, exclude: Iterable = (), market: str = "spread"
    ) -> dict:
        """A random historical game with pre-game info only (no result/model)
        that can be played on ``market``."""
        if not len(self.pool):
            raise LookupError("no historical games available")
        playable = [g for g in self.pool.index if market in self.markets(g)]
        if not playable:
            raise LookupError(f"no historical games for the {market} market")
        ids = [g for g in playable if g not in set(exclude)] or playable
        return self.challenge_game(rng.choice(ids))

    def challenge_game(self, game_id) -> dict:
        """A pool game's pre-game info only (no result/model)."""
        row = self._row(game_id, self.pool.loc[game_id, self.schema.team])
        header = self._header(row)
        return py(
            {
                **header,
                "markets": self.markets(game_id),
                "summaries": self._summaries(row),
                "stats": self.stat_definitions(),
                "pool_size": len(self.pool),
            }
        )

    def challenge_answers(self, game_id) -> dict:
        """Each playable market's outcome, odds and the model's pick, without
        the player's pick (the browser scores it: ``outcome`` is the winning
        pick, ``odds`` each pick's American odds)."""
        s = self.schema
        pool = self.pool.loc[game_id]
        row = self._row(game_id, pool[s.team])
        header = self._header(row)
        result = self._result(row)
        prediction = self._prediction(pool, header)
        answers = {}
        for market in self.markets(game_id):
            if market == "spread":
                outcome, model_pick = result["covered"], prediction["ats_pick"]
                odds = {"home": SPREAD_ODDS, "away": SPREAD_ODDS}
                extra = {"spread": header["spread"], "covered": outcome}
            elif market == "moneyline":
                outcome, model_pick = result["winner"], prediction["ml_pick"]
                odds = header["moneyline"]
                extra = {}
            else:
                outcome, model_pick = result["went"], prediction["ou_pick"]
                odds = header["total_odds"] or {"over": -110.0, "under": -110.0}
                extra = {"total_line": header["total_line"]}
            correct = model_pick == outcome
            model_units = (
                profit(odds[model_pick], correct) if model_pick is not None else 0.0
            )
            answers[market] = py(
                {
                    "game_id": game_id,
                    "market": market,
                    "outcome": outcome,
                    "odds": odds,
                    "result": result,
                    **extra,
                    "model": {
                        **prediction,
                        "pick": model_pick,
                        "correct": correct,
                        "units": model_units,
                    },
                }
            )
        return answers

    def challenge_answer(self, game_id, pick: str, market: str = "spread") -> dict:
        """Score the player's pick on ``market`` and reveal the model's."""
        answer = self.challenge_answers(game_id).get(market)
        if answer is None:
            raise LookupError(f"game {game_id} cannot be played on {market}")
        correct = pick == answer["outcome"]
        return {
            **answer,
            "pick": pick,
            "correct": correct,
            "units": profit(answer["odds"][pick], correct),
        }

    def available_markets(self) -> list[str]:
        """Markets the challenge offers (some pool game can be played on)."""
        if not len(self.pool):
            return []
        found = set()
        for game_id in self.pool.index:
            found |= set(self.markets(game_id))
            if len(found) == len(MARKETS):
                break
        return [m for m in MARKETS if m in found]

    def performance(self) -> dict | None:
        return self.performance_summary

    def rankings(self) -> dict:
        """Power ratings per season: one value per team per date, and how
        the season ended for the teams that reached the last playoff rounds."""
        table = self.rankings_table
        baked = (
            set(self.baked_rankings["season"]) if len(self.baked_rankings) else set()
        )
        seasons = []
        for season, frame in sorted(table.groupby("season"), reverse=True):
            dates = sorted(frame["date"].unique())
            wide = frame.pivot_table(
                index="team", columns="date", values="rating"
            ).reindex(columns=dates)
            teams = [
                {"team": self.team(team), "ratings": [py(round(v, 2)) for v in row]}
                for team, row in wide.iterrows()
            ]
            seasons.append(
                {
                    "season": season,
                    "source": "backtest" if season in baked else "current",
                    "dates": dates,
                    "date_names": [self.date_name(season, d) for d in dates]
                    if self.date_names
                    else None,
                    "teams": teams,
                    "playoffs": self.playoffs.get(int(season), {}),
                }
            )
        return py(
            {
                "seasons": seasons,
                "date_label": self.view.date_label,
                "playoff_labels": dict(self.view.playoff_labels),
            }
        )

    def info(self) -> dict:
        return py(
            {
                "sport": self.view.sport,
                "label": self.view.label,
                "date_label": self.view.date_label,
                "split_season": self.view.split_season,
                "has_spread": bool(self.schema.market_spread),
                "has_team_map": self.team_map is not None,
                # what the browser needs to price any line: the model's
                # residual spreads and how much to trust its disagreements
                "pricing": {
                    "sigma": self.sigma,
                    "total_sigma": self.total_sigma,
                    "trust": self.trust,
                },
                "challenge_games": len(self.pool),
                "markets": self.available_markets(),
                "built_at": self.meta.get("built_at"),
                "refreshed_at": self.meta.get("refreshed_at"),
                "model": self.meta.get("model"),
                "backtest": self.meta.get("backtest"),
            }
        )


# UMAP needs a few neighbours per point to find any structure.
MIN_MAP_TEAMS = 5


def umap_layout(vectors: np.ndarray, dims: int = 3, seed: int = 0) -> np.ndarray:
    """``[n, d]`` vectors laid out in ``dims`` dimensions with UMAP (cosine
    distance, a fixed seed so a re-export keeps the layout), centered on
    their bounding box and scaled into [-1, 1]."""
    import umap

    neighbors = max(2, min(15, len(vectors) - 1))
    reducer = umap.UMAP(
        n_components=dims,
        n_neighbors=neighbors,
        min_dist=0.3,
        metric="cosine",
        # spectral initialization needs more points than a small league has
        init="spectral" if len(vectors) > 30 else "random",
        random_state=seed,
    )
    out = reducer.fit_transform(vectors)
    out = out - (out.min(axis=0) + out.max(axis=0)) / 2
    scale = np.abs(out).max()
    return out / scale if scale > 0 else out


def _calibration(prob: pd.Series, happened: pd.Series) -> list[dict]:
    """Mean predicted probability vs how often it happened, in 10% bins."""
    bins = pd.cut(prob, np.linspace(0, 1, 11), include_lowest=True)
    grouped = happened.groupby(bins, observed=True)
    predicted = prob.groupby(bins, observed=True).mean()
    return [
        {"predicted": predicted[b], "actual": grouped.mean()[b], "n": n}
        for b, n in grouped.size().items()
        if n
    ]


def _today_eastern() -> str:
    """Today's date in US Eastern (kickoffs are listed in it), ``YYYY-MM-DD``."""
    return pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
