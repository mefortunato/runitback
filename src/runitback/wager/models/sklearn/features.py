"""Season-to-date team features shared by training and serving.

Training rows and served matchups go through the same two functions,
:func:`team_state` and :func:`matchup_features`, so the column layout the
estimator sees cannot drift between the two.
"""

from __future__ import annotations

import pandas as pd

from ...core.schema import GameSchema, played


def team_state(
    games: pd.DataFrame,
    schema: GameSchema,
    windows: tuple[int, ...] = (),
    include_current: bool = False,
) -> pd.DataFrame:
    """Each row's team averages over its games so far this season.

    Args:
        games: Played games only.
        windows: Also average each team's last ``w`` games, for each ``w``.
        include_current: Include the row's own game (state after the game)
            instead of only earlier games (state before the game).

    Returns:
        Frame aligned with ``games.index`` with ``n_games`` (games averaged),
        ``<stat>_avg`` and ``<stat>_last<w>`` columns.
    """
    stats = list(schema.stats)
    keys = [schema.season, schema.team]
    ordered = games.sort_values([schema.season, schema.date])
    shift = 0 if include_current else 1

    def per_team(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.groupby(level=[0, 1], sort=False).shift(shift)
        return frame.droplevel([0, 1])

    grouped = ordered.groupby(keys, sort=False)
    parts = [
        (grouped.cumcount() + 1 - shift).rename("n_games"),
        per_team(grouped[stats].expanding().mean()).add_suffix("_avg"),
    ]
    for window in windows:
        rolling = grouped[stats].rolling(window, min_periods=1).mean()
        parts.append(per_team(rolling).add_suffix(f"_last{window}"))
    return pd.concat(parts, axis=1).reindex(games.index)


def latest_team_state(
    games: pd.DataFrame, schema: GameSchema, windows: tuple[int, ...] = ()
) -> pd.DataFrame:
    """State after each team's latest game in the latest season, by team."""
    games = played(games, schema)
    season = games[schema.season].max()
    current = games[games[schema.season] == season]
    state = team_state(current, schema, windows, include_current=True)
    last = (
        current.sort_values(schema.date).drop_duplicates(schema.team, keep="last").index
    )
    return state.loc[last].set_axis(current.loc[last, schema.team].values)


def matchup_features(
    team: pd.DataFrame,
    opp: pd.DataFrame,
    matchups: pd.DataFrame,
    schema: GameSchema,
) -> pd.DataFrame:
    """Model input: team state, opponent state and pre-game columns.

    ``team`` and ``opp`` are team-state rows aligned with ``matchups``.
    Pre-game columns missing from ``matchups`` are filled with NaN, which the
    fitted preprocessing imputes.
    """
    index = matchups.index
    pre_game = matchups.reindex(columns=schema.pre_game_columns)
    for col in schema.pre_game_columns:
        if col in schema.categorical:
            pre_game[col] = pre_game[col].astype(str)
        else:
            pre_game[col] = pd.to_numeric(pre_game[col], errors="coerce")
    return pd.concat(
        [
            team.set_axis(index).add_prefix("team_"),
            opp.set_axis(index).add_prefix("opp_"),
            pre_game,
        ],
        axis=1,
    )


def training_frame(
    games: pd.DataFrame, schema: GameSchema, windows: tuple[int, ...] = ()
) -> pd.DataFrame:
    """Matchup features for every played row, using pre-game states only.

    The opponent's state is taken from the opponent's own row of the same
    game, i.e. computed exactly like a served opponent's state.
    """
    games = played(games, schema)
    state = team_state(games, schema, windows)
    by_team = state.set_axis(
        pd.MultiIndex.from_frame(games[[schema.game_id, schema.team]])
    )
    opp = by_team.reindex(pd.MultiIndex.from_frame(games[[schema.game_id, schema.opp]]))
    return matchup_features(state, opp, games, schema)
