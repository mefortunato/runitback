"""Naive baselines: the yardsticks a learned model has to beat.

* ``home``: home-field advantage only.
* ``point_diff``: season-to-date average margin difference, scaled by least
  squares (plus home field).
* ``elo``: FiveThirtyEight-style Elo with margin-of-victory updates and
  regression to the mean between seasons; no fitted parameters.

Each is a :class:`~runitback.wager.core.predictor.MatchupModel`, so it is served,
calibrated and backtested exactly like the learned families.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd

from ..core.evaluation import finish_fit
from ..core.io import Table, read_table
from ..core.predictor import MatchupModel, prepare_save_dir
from ..core.schema import GameSchema, one_row_per_game, played
from ..core.splits import SeasonSplit, split_seasons
from ..core.sport import get_sport
from .sklearn.features import latest_team_state, training_frame


def _location(matchups: pd.DataFrame, schema: GameSchema) -> np.ndarray:
    if not schema.location or schema.location not in matchups:
        return np.zeros(len(matchups))
    return matchups[schema.location].fillna(0).to_numpy(dtype=float)


def _prepare(games: Table, sport: str, seasons: dict, save_dir, overwrite):
    prepare_save_dir(save_dir, overwrite)
    schema = get_sport(sport).schema
    games = played(read_table(games), schema)
    split = split_seasons(games[schema.season].unique(), **seasons)
    return schema, games, split


class HomeFieldModel(MatchupModel):
    """Margin = home advantage x location."""

    def __init__(self, schema: GameSchema, home_advantage: float):
        self.schema = schema
        self.home_advantage = home_advantage
        self._teams: list = []

    def update_state(self, games: pd.DataFrame) -> None:
        games = played(games, self.schema)
        current = games[games[self.schema.season] == games[self.schema.season].max()]
        self._teams = list(pd.unique(current[self.schema.team]))

    @property
    def teams(self) -> list:
        return self._teams

    def predict_margin(self, matchups: pd.DataFrame) -> np.ndarray:
        return self.home_advantage * _location(matchups, self.schema)


def fit_home(
    games: Table,
    sport: str,
    save_dir: str | None = None,
    train_seasons: list[int] | None = None,
    val_seasons: list[int] | None = None,
    test_seasons: list[int] | None = None,
    overwrite: bool = False,
    print_metrics: bool = True,
) -> dict:
    """Home-field-only baseline; the advantage is the mean home margin."""
    seasons = dict(train=train_seasons, val=val_seasons, test=test_seasons)
    schema, games, split = _prepare(games, sport, seasons, save_dir, overwrite)
    train = games[games[schema.season].isin(split.train)]
    home = train[_location(train, schema) > 0]
    model = HomeFieldModel(schema, float(home[schema.target].mean()))
    rows = games[games[schema.season].isin(split.all)]
    return finish_fit(
        model, games, rows, model.predict_margin(rows), split, sport, save_dir,
        print_metrics,
    )  # fmt: skip


class PointDifferentialModel(MatchupModel):
    """Margin = a x (team avg margin - opponent avg margin) + b x location."""

    def __init__(self, schema: GameSchema, coef: float, home_advantage: float):
        self.schema = schema
        self.coef = coef
        self.home_advantage = home_advantage
        self.average: pd.Series | None = None

    @property
    def margin_schema(self) -> GameSchema:
        return dataclasses.replace(self.schema, stats=(self.schema.target,))

    def update_state(self, games: pd.DataFrame) -> None:
        state = latest_team_state(games, self.margin_schema)
        self.average = state[f"{self.schema.target}_avg"]

    @property
    def teams(self) -> list:
        return [] if self.average is None else list(self.average.index)

    def predict_margin(self, matchups: pd.DataFrame) -> np.ndarray:
        team = self.average.reindex(matchups[self.schema.team].values).to_numpy()
        opp = self.average.reindex(matchups[self.schema.opp].values).to_numpy()
        location = _location(matchups, self.schema)
        return self.coef * (team - opp) + self.home_advantage * location


def fit_point_diff(
    games: Table,
    sport: str,
    save_dir: str | None = None,
    train_seasons: list[int] | None = None,
    val_seasons: list[int] | None = None,
    test_seasons: list[int] | None = None,
    min_games: int = 1,
    overwrite: bool = False,
    print_metrics: bool = True,
) -> dict:
    """Season-to-date point differential, scaled by least squares."""
    seasons = dict(train=train_seasons, val=val_seasons, test=test_seasons)
    schema, games, split = _prepare(games, sport, seasons, save_dir, overwrite)
    model = PointDifferentialModel(schema, coef=0.0, home_advantage=0.0)
    X = training_frame(games, model.margin_schema)
    avg = f"{schema.target}_avg"
    usable = (
        (X["team_n_games"] >= min_games)
        & (X["opp_n_games"] >= min_games)
        & games[schema.season].isin(split.all)
    )
    X, rows = X[usable], games[usable]
    diff = (X[f"team_{avg}"] - X[f"opp_{avg}"]).to_numpy()
    location = _location(rows, schema)
    train = rows[schema.season].isin(split.train).to_numpy()
    design = np.column_stack([diff, location])[train]
    (model.coef, model.home_advantage), *_ = np.linalg.lstsq(
        design, rows[schema.target].to_numpy(dtype=float)[train], rcond=None
    )
    raw = model.coef * diff + model.home_advantage * location
    return finish_fit(model, games, rows, raw, split, sport, save_dir, print_metrics)


@dataclasses.dataclass(frozen=True)
class EloParams:
    """FiveThirtyEight's NFL Elo settings."""

    k: float = 20.0
    home: float = 65.0  # Elo points of home-field advantage
    regress: float = 1 / 3  # toward the mean between seasons
    mean: float = 1505.0
    points_per_elo: float = 1 / 25


class EloModel(MatchupModel):
    """Margin = (team Elo - opponent Elo + home edge) x points_per_elo."""

    def __init__(self, schema: GameSchema, params: EloParams = EloParams()):
        self.schema = schema
        self.params = params
        self.ratings: dict = {}
        self.season = None
        self._teams: list = []

    def replay(self, games: pd.DataFrame) -> pd.Series:
        """Run Elo through every played game; returns each row's pre-game
        team-perspective Elo difference (including home edge)."""
        s, p = self.schema, self.params
        games = played(games, s)
        ordered = one_row_per_game(games, s).sort_values([s.season, s.date])
        ratings, season, diffs, before = {}, None, {}, {}
        for game, team, opp, loc, margin, game_season in zip(
            ordered[s.game_id],
            ordered[s.team],
            ordered[s.opp],
            _location(ordered, s),
            ordered[s.target],
            ordered[s.season],
        ):
            if game_season != season:
                ratings = {t: self._regress(r) for t, r in ratings.items()}
                season = game_season
            team_elo, opp_elo = ratings.get(team, p.mean), ratings.get(opp, p.mean)
            diff = team_elo - opp_elo + p.home * loc
            diffs[game] = diff
            before[game] = (team_elo, opp_elo)
            expected = 1 / (1 + 10 ** (-diff / 400))
            actual = 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
            winner_diff = diff if margin > 0 else -diff
            multiplier = math.log(abs(margin) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
            shift = p.k * multiplier * (actual - expected)
            ratings[team], ratings[opp] = team_elo + shift, opp_elo - shift
        self.ratings, self.season = ratings, season
        self._before, self._listed = before, ordered.index
        latest = games[games[s.season] == season] if season is not None else games
        self._teams = list(pd.unique(latest[s.team]))
        home_diff = games[s.game_id].map(diffs)
        is_listed = games.index.isin(ordered.index)
        return pd.Series(np.where(is_listed, home_diff, -home_diff), index=games.index)

    def pre_game_ratings(self, games: pd.DataFrame) -> pd.DataFrame:
        """Each played row's ``team_elo`` and ``opp_elo`` before the game.

        Depends only on earlier games, so it is safe as a feature.
        """
        s = self.schema
        games = played(games, s)
        self.replay(games)
        listed = pd.DataFrame(
            self._before.values(),
            index=pd.Index(self._before.keys(), name=s.game_id),
            columns=["first", "second"],
        ).reindex(games[s.game_id])
        is_listed = games.index.isin(self._listed)
        first, second = listed["first"].to_numpy(), listed["second"].to_numpy()
        return pd.DataFrame(
            {
                "team_elo": np.where(is_listed, first, second),
                "opp_elo": np.where(is_listed, second, first),
            },
            index=games.index,
        )

    def _regress(self, rating: float) -> float:
        return self.params.mean + (rating - self.params.mean) * (
            1 - self.params.regress
        )

    def update_state(self, games: pd.DataFrame) -> None:
        self.replay(games)

    @property
    def teams(self) -> list:
        return self._teams

    def predict_margin(self, matchups: pd.DataFrame) -> np.ndarray:
        s, p = self.schema, self.params
        new_season = (
            matchups[s.season].to_numpy() > self.season
            if s.season in matchups
            else np.zeros(len(matchups), dtype=bool)
        )

        def elo(keys):
            raw = np.array([self.ratings.get(k, p.mean) for k in keys])
            return np.where(new_season, self._regress(raw), raw)

        diff = elo(matchups[s.team]) - elo(matchups[s.opp])
        return (diff + p.home * _location(matchups, s)) * p.points_per_elo


def fit_elo(
    games: Table,
    sport: str,
    save_dir: str | None = None,
    train_seasons: list[int] | None = None,
    val_seasons: list[int] | None = None,
    test_seasons: list[int] | None = None,
    overwrite: bool = False,
    print_metrics: bool = True,
) -> dict:
    """Elo with fixed FiveThirtyEight settings, or the sport's own
    (``Sport.elo``); nothing is fit but sigma."""
    seasons = dict(train=train_seasons, val=val_seasons, test=test_seasons)
    schema, games, split = _prepare(games, sport, seasons, save_dir, overwrite)
    model = EloModel(schema, EloParams(**get_sport(sport).elo))
    raw = model.replay(games) * model.params.points_per_elo
    in_split = games[schema.season].isin(split.all)
    return finish_fit(
        model, games, games[in_split], raw[in_split], split, sport, save_dir,
        print_metrics,
    )  # fmt: skip


SPLIT_KWARGS = ("train_seasons", "val_seasons", "test_seasons")


def split_kwargs(split: SeasonSplit) -> dict:
    return dict(zip(SPLIT_KWARGS, (split.train, split.val, split.test)))
