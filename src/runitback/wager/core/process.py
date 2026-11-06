from __future__ import annotations

import logging
import pathlib

import pandas as pd

from .io import Table, read_table
from .schema import validate
from .sport import get_sport

logger = logging.getLogger(__name__)


def process(
    sport: str,
    games: Table,
    output_file: str | None = None,
) -> pd.DataFrame:
    """Validate and clean a downloaded games table.

    Keeps upcoming games (missing target) so they can be predicted. Played
    games must appear exactly twice, once per team; others are dropped.
    Missing box-score stats are kept and handled by the models.
    """
    schema = get_sport(sport).schema
    games = read_table(games)
    validate(games, schema)
    games = games.dropna(subset=schema.key_columns)

    rows_per_game = games.groupby(schema.game_id)[schema.team].transform("size")
    bad = rows_per_game != 2
    if bad.any():
        logger.warning("dropping %d rows of games without two rows", bad.sum())
        games = games[~bad]

    games = games.copy()
    if schema.location:
        games[schema.location] = pd.to_numeric(games[schema.location]).fillna(0)

    games = games.sort_values([schema.season, schema.date, schema.game_id])
    games = games.reset_index(drop=True)

    if output_file:
        pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        games.to_csv(output_file, index=False)
    return games
