"""Column roles shared by every sport and every model family.

A games table has one row per team per game, so each game appears twice: once
from each team's perspective. Upcoming games are rows whose target is missing.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GameSchema:
    """Maps the columns of a sport's games table onto the roles models need.

    Attributes:
        season: Season identifier (int-like).
        date: Orders games within a season (week, day number, ...).
        team: Key of the team whose perspective the row is written from.
        opp: Key of the opponent.
        game_id: Identifies a game; shared by both of its rows.
        target: Team points minus opponent points. Missing for upcoming games.
        stats: Box-score stats from the row, only known after the game.
        location: +1 home, 0 neutral, -1 away, from the team's perspective.
        context: Numeric features known before the game (rest days, ...).
        categorical: Categorical features known before the game.
        paired: Column pairs swapped when a row is mirrored to the opponent's
            perspective, e.g. ``("rest", "opp_rest")``.
        negated: Columns negated when a row is mirrored, e.g. a point spread.
        team_name: Optional display name of ``team``.
        opp_name: Optional display name of ``opp``.
        display: Columns carried into predictions for display only (kickoff
            time, betting lines, ...); never used as features.
        market_spread: The market's expected margin for ``team`` (positive =
            favored), used as the backtest baseline.
        market_moneyline: ``(team, opponent)`` American moneyline columns,
            for market-implied win probabilities.
        total: Combined points of both teams (missing for upcoming games);
            models that predict it can be scored against ``market_total``.
        market_total: The market's over/under line.
        market_total_odds: ``(over, under)`` American odds on the total.
        draws: The sport has three outcomes (win, draw, loss), as football
            does: models and the backtest also produce and score a draw
            probability (``draw_prob``), and the market's three prices
            (``market_moneyline`` and ``market_draw``) give its win and draw
            probabilities. False (the default) leaves every two-outcome
            computation as it is.
        market_draw: American price of the draw (``draws`` only).
        market_average: ``(team, draw, opponent)`` American price columns of
            the market arm in a sport with draws (the closing market average,
            from the same bookmakers in every season), where the sport's
            ``market_moneyline`` and ``market_draw`` are whichever bookmaker
            the games table preferred. Missing: those are the market arm.
        market_closing: Column that is 1 where the market's prices for the game
            are closing prices and 0 for opening ones (a sport whose source
            changed bookmakers over the years): a feature of its market tokens.
        devig: How the bookmaker margin is removed from the three prices
            (``"proportional"``, ``"power"`` or ``"shin"``; see
            :func:`runitback.wager.backtest.no_vig_probabilities`).
    """

    season: str
    date: str
    team: str
    opp: str
    game_id: str
    target: str
    stats: tuple[str, ...]
    location: str | None = None
    context: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()
    paired: tuple[tuple[str, str], ...] = ()
    negated: tuple[str, ...] = ()
    team_name: str | None = None
    opp_name: str | None = None
    display: tuple[str, ...] = ()
    market_spread: str | None = None
    market_moneyline: tuple[str, str] | None = None
    total: str | None = None
    market_total: str | None = None
    market_total_odds: tuple[str, str] | None = None
    draws: bool = False
    market_draw: str | None = None
    market_average: tuple[str, str, str] | None = None
    market_closing: str | None = None
    devig: str = "proportional"

    @property
    def pre_game_columns(self) -> list[str]:
        """Columns a matchup request may carry: known before kickoff."""
        cols = [self.location] if self.location else []
        return cols + list(self.context) + list(self.categorical)

    @property
    def key_columns(self) -> list[str]:
        return [self.season, self.date, self.game_id, self.team, self.opp]

    def with_features(
        self,
        context: list[str] | None = None,
        categorical: list[str] | None = None,
    ) -> GameSchema:
        """Return a copy with the pre-game feature columns replaced."""
        changes = {}
        if context is not None:
            changes["context"] = tuple(context)
        if categorical is not None:
            changes["categorical"] = tuple(categorical)
        return dataclasses.replace(self, **changes)


def validate(games: pd.DataFrame, schema: GameSchema) -> None:
    """Raise ValueError if ``games`` lacks columns the schema requires."""
    required = schema.key_columns + [schema.target] + list(schema.stats)
    required += schema.pre_game_columns
    missing = [col for col in required if col not in games.columns]
    if missing:
        raise ValueError(f"games table is missing columns: {missing}")


def played(games: pd.DataFrame, schema: GameSchema) -> pd.DataFrame:
    """Rows of games that have a result."""
    return games[games[schema.target].notna()]


def upcoming(games: pd.DataFrame, schema: GameSchema) -> pd.DataFrame:
    """Rows of games that have not been played yet."""
    return games[games[schema.target].isna()]


def one_row_per_game(games: pd.DataFrame, schema: GameSchema) -> pd.DataFrame:
    """Keep one perspective of each game: the home team's where there is one."""
    if schema.location:
        games = games.sort_values(schema.location, ascending=False, kind="stable")
    return games.drop_duplicates(schema.game_id)


def mirror(matchups: pd.DataFrame, schema: GameSchema) -> pd.DataFrame:
    """Rewrite each row from the opponent's perspective.

    Columns the schema does not mention are copied unchanged. Columns that are
    absent from ``matchups`` are skipped.
    """
    out = matchups.copy()
    pairs = [(schema.team, schema.opp), *schema.paired]
    if schema.team_name and schema.opp_name:
        pairs.append((schema.team_name, schema.opp_name))
    for a, b in pairs:
        if a in matchups and b in matchups:
            out[a] = matchups[b].values
            out[b] = matchups[a].values
    negated = [*schema.negated] + ([schema.location] if schema.location else [])
    for col in negated:
        if col in matchups:
            out[col] = -matchups[col]
    if schema.target in matchups:
        out[schema.target] = -matchups[schema.target]
    return out
