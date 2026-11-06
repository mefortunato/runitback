from __future__ import annotations

import itertools
import pathlib

import pandas as pd

from ..core.io import Table, read_table
from ..core.predictor import Predictor, load_predictor
from ..core.schema import one_row_per_game, upcoming


def all_matchups(predictor: Predictor, location: int = 0) -> pd.DataFrame:
    """Every ordered pair of teams the predictor has state for."""
    schema = predictor.schema
    pairs = pd.DataFrame(
        itertools.permutations(predictor.model.teams, 2),
        columns=[schema.team, schema.opp],
    )
    if schema.location:
        pairs[schema.location] = location
    return pairs


def power_ratings(
    predictor: Predictor, teams=None, location: int = 0, chunk: int = 4096
) -> pd.Series:
    """Each team's average predicted margin against every other team.

    Neutral site by default. ``teams`` limits both sides to those teams (e.g.
    the ones playing this season); the ratings then average to zero.
    Matchups are predicted ``chunk`` at a time: every pair of a college
    league is ~130k, too many for one batch through a sequence model.
    """
    schema = predictor.schema
    pairs = all_matchups(predictor, location)
    if teams is not None:
        teams = set(teams)
        pairs = pairs[pairs[schema.team].isin(teams) & pairs[schema.opp].isin(teams)]
    pairs = pairs.reset_index(drop=True)
    preds = pd.concat(
        [
            predictor.predict(pairs.iloc[start : start + chunk])
            for start in range(0, max(len(pairs), 1), chunk)
        ],
        ignore_index=True,
    )
    ratings = preds.groupby(schema.team)["pred_margin"].mean()
    return ratings.sort_values(ascending=False)


def long_to_wide(
    preds: pd.DataFrame,
    team: str = "team",
    opp: str = "opp",
    value: str = "win_prob",
    team_name_map: dict | None = None,
    sorted: bool = True,
) -> pd.DataFrame:
    """Matchup matrix: rows are teams, columns opponents, cells ``value``."""
    wide = preds.pivot_table(values=value, index=team, columns=opp)
    if team_name_map:
        wide = wide.rename(index=team_name_map, columns=team_name_map)
    if sorted:
        order = wide.mean(axis=1).sort_values(ascending=False).index
        wide = wide.loc[order, order]
    return wide


def upcoming_matchups(
    games: pd.DataFrame, predictor: Predictor, season=None, date=None
) -> pd.DataFrame:
    """Unplayed games on one date, one row per game (home team's view).

    Defaults to the latest season and its earliest date with unplayed games.
    """
    schema = predictor.schema
    games = upcoming(games, schema)
    if games.empty:
        raise ValueError("games table has no unplayed games")
    season = games[schema.season].max() if season is None else season
    games = games[games[schema.season] == season]
    date = games[schema.date].min() if date is None else date
    games = one_row_per_game(games[games[schema.date] == date], schema)
    columns = schema.key_columns + schema.pre_game_columns + list(schema.display)
    columns += [c for c in (schema.team_name, schema.opp_name) if c]
    return games[[c for c in columns if c in games]].reset_index(drop=True)


def predict(
    model_dir: str,
    games: Table | None = None,
    matchups: Table | None = None,
    season: int | None = None,
    date: float | None = None,
    output_file: str | None = None,
) -> pd.DataFrame:
    """Predict games with a saved model.

    Args:
        model_dir: Saved model directory.
        games: Processed games table. Refreshes the model's team state from
            its played games and, unless ``matchups`` is given, predicts its
            unplayed games on ``date`` of ``season``.
        matchups: Explicit matchups to predict instead (team/opp columns and
            optional pre-game columns).
        season / date: Which unplayed games to predict; default is the next
            date with unplayed games.
        output_file: Write predictions as .csv or .json (records).
    """
    predictor = load_predictor(model_dir)
    if games is not None:
        games = read_table(games)
        predictor.update_state(games)
    if matchups is None:
        if games is None:
            raise ValueError("pass games (to find upcoming games) or matchups")
        matchups = upcoming_matchups(games, predictor, season, date)
    else:
        matchups = read_table(matchups)

    preds = predictor.predict(matchups)
    if output_file:
        path = pathlib.Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            preds.to_json(path, orient="records", indent=2)
        else:
            preds.to_csv(path, index=False)
    return preds
