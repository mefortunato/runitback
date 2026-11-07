"""Games table -> each row's team / opponent game-history token sequences.

A token is one earlier game of a team: its box-score stats, result and
location, standardized, plus (optionally) how strong both teams were going
into it, how its margin compared with what Elo expected, what the betting
market expected of it (``market_tokens``) and the sport's per-game rate
stats (``efficiency``, :mod:`runitback.wager.core.efficiency`). A row's history is
the team's previous ``history`` played games, most recent first, crossing
season boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ...core.schema import GameSchema, mirror, played

# Pre-game strength of both teams in each token's game (opponent_strength).
STRENGTH_COLUMNS = ["_team_elo", "_opp_elo", "_opp_margin_avg"]
# The game's margin minus the margin pre-game Elo expected (elo_margin).
ELO_MARGIN_COLUMNS = ["_margin_vs_elo"]
# What the market expected of each token's game, and how the game went
# against it (market_tokens): all set before that game, so nothing leaks.
MARKET_TOKEN_COLUMNS = [
    "_mkt_spread",  # the team's closing spread (expected margin)
    "_mkt_cover",  # margin minus the spread
    "_mkt_prob_z",  # probit of the no-vig moneyline win probability
    "_mkt_win_surprise",  # won (1, 0.5 tie, 0) minus that probability
    "_mkt_total",  # the over/under line
    "_mkt_total_resid",  # combined points minus the line
    "_mkt_has_line",  # 1 where the game had a spread or moneylines
]


# The market's running rating of each token game's two teams (market_rating):
# ratings fit to every earlier closing line, so ``rating - opp_rating + home``
# reproduces the spread; each game's own line is in (it was set before the
# game), its result never is.
MARKET_RATING_COLUMNS = ["_mkt_rating", "_mkt_opp_rating", "_mkt_rating_gap"]


# play-by-play game embeddings (runitback.nfl.embed): ``_pbp_0..`` and a missing-game indicator
PBP_PREFIX = "_pbp_"
PBP_MISSING = "_pbp_missing"


def pbp_columns(dim: int) -> list[str]:
    return [f"{PBP_PREFIX}{i}" for i in range(dim)] + [PBP_MISSING] if dim else []


def pbp_fold_path(template: str, season: int) -> str:
    """The embedding table of the foundation model for test season ``season``."""
    return template.format(season=int(season))


def pbp_features(games: pd.DataFrame, schema: GameSchema, path: str, dim: int) -> pd.DataFrame:
    """Each row's team-game embedding from ``path`` (columns ``game_id``, ``team``, ``_pbp_*``),
    with ``_pbp_missing`` 1 where the game has none (no play-by-play)."""
    table = _read_pbp(path)
    cols = pbp_columns(dim)[:-1]
    key = pd.MultiIndex.from_arrays([games[schema.game_id], games[schema.team]])
    out = table[cols].reindex(key)
    out.index = games.index
    out[PBP_MISSING] = out[cols[0]].isna().astype(np.float32)
    return out


# Per-game lineup context from the foundation (runitback.nfl.lineup_features): ``_lp_*`` the
# projected lineup's pooled player embeddings, ``_lu_*`` how they differ from the lineups behind the
# team's history games, ``_cf_*`` the history re-embedded under the projected lineup minus the real
# one. Joined for the team and, as
# ``_opp<col>``, its opponent, into the head's context (not the history tokens).
# ``_pv_*`` (scripts/player_values.py): the projected lineup's player values in points over
# replacement per position group (QB, RB, WR, TE, OL, DL, LB, DB, kicking) and the offense /
# defense change vs the previous game's lineup.
# ``pva``: the same values plus each group's change vs the team's last 8 lineups (``_pv_11`` ..
# ``_pv_19``), used only through sign-constrained additive terms (PairHead), so a better player
# at any position always moves the prediction his team's way.
LINEUP_SETS = {"lp": ("_lp_", 16), "lu": ("_lu_", 16), "cf": ("_cf_", 32), "pv": ("_pv_", 11),
               "pva": ("_pv_", 20)}
# pva columns: groups (QB, RB, WR, TE, OL, DL, LB, DB, kicking) as levels and as changes
PVA_LEVELS, PVA_CHANGES = tuple(range(9)), tuple(range(11, 20))
PVA_TOTAL_SIGN = (1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0)
LINEUP_MISSING = "_lu_missing"


def lineup_columns(which: str | None) -> list[str]:
    """The team-side context columns of a ``lineup_set`` ("lp", "lu", "cf", or several: "lp+lu")."""
    if not which:
        return []
    cols = [f"{LINEUP_SETS[k][0]}{i}" for k in which.split("+") for i in range(LINEUP_SETS[k][1])]
    return cols + [LINEUP_MISSING]


def additive_spec(n_before: int, which: str | None) -> dict:
    """PairHead additive-term arguments for the ``pva`` block of lineup set ``which`` (context
    columns: ``n_before`` others, then the team's lineup columns, then the opponent's)."""
    if not which or "pva" not in which.split("+"):
        return {}
    team = lineup_columns(which)
    start = n_before
    for k in which.split("+"):
        if k == "pva":
            break
        start += LINEUP_SETS[k][1]
    opp = n_before + len(team)
    idx = list(PVA_LEVELS) + list(PVA_CHANGES)
    sign = list(PVA_TOTAL_SIGN) * 2
    pairs = [(start + i, opp + (start - n_before) + i) for i in idx]
    mask = [start + i for i in range(LINEUP_SETS["pva"][1])]
    mask += [opp + (start - n_before) + i for i in range(LINEUP_SETS["pva"][1])]
    return {"additive_pairs": pairs, "additive_total_sign": sign, "mask_context": mask}


def lineup_context_columns(which: str | None) -> list[str]:
    cols = lineup_columns(which)
    return cols + [f"_opp{c}" for c in cols]


def lineup_features(games: pd.DataFrame, schema: GameSchema, path: str, which: str) -> pd.DataFrame:
    """Both sides' lineup context for each row (missing flag 1 where the table has no row)."""
    table = _read_pbp(path)
    cols = lineup_columns(which)[:-1]
    out = []
    for side, prefix in ((schema.team, ""), (schema.opp, "_opp")):
        key = pd.MultiIndex.from_arrays([games[schema.game_id], games[side]])
        part = table.reindex(columns=cols).reindex(key)
        part[LINEUP_MISSING] = part[cols[0]].isna().astype(np.float32)
        part.columns = [f"{prefix}{c}" for c in part.columns]
        part.index = games.index
        out.append(part)
    return pd.concat(out, axis=1)


_PBP_CACHE: dict[str, pd.DataFrame] = {}


def _read_pbp(path: str) -> pd.DataFrame:
    if path not in _PBP_CACHE:
        table = pd.read_parquet(path)
        _PBP_CACHE[path] = table.set_index(["game_id", "team"])
    return _PBP_CACHE[path]


def efficiency_columns(sport: str | None) -> list[str]:
    """The ``_eff_*`` columns ``sport``'s efficiency function derives."""
    from ...core.sport import get_sport

    fn = get_sport(sport).efficiency if sport else None
    return [] if fn is None else list(fn(pd.DataFrame()).columns)


def extra_columns(
    opponent_strength: bool,
    elo_margin: bool,
    market_tokens: bool = False,
    efficiency: bool = False,
    sport: str | None = None,
    market_rating: bool = False,
    pbp_dim: int = 0,
    pbp_only: bool = False,
) -> list[str]:
    return (
        (STRENGTH_COLUMNS if opponent_strength else [])
        + (ELO_MARGIN_COLUMNS if elo_margin else [])
        + (MARKET_TOKEN_COLUMNS if market_tokens else [])
        + (efficiency_columns(sport) if efficiency and not pbp_only else [])
        + (MARKET_RATING_COLUMNS if market_rating else [])
        + pbp_columns(pbp_dim)
    )


def base_token_columns(schema: GameSchema, pbp_only: bool = False) -> list[str]:
    """Box score, result and location; only result and location with ``pbp_only`` (the
    play-by-play embedding stands in for the box score)."""
    location = [schema.location] if schema.location else []
    stats = [] if pbp_only else list(schema.stats)
    return list(dict.fromkeys([*stats, schema.target, *location]))


def market_line(games: pd.DataFrame, schema: GameSchema) -> pd.Series:
    """Each row's market line for its team: the spread where the sport has
    one, else the probit of the no-vig moneyline probability."""
    import scipy.stats

    from ...backtest import no_vig_probability

    if schema.market_spread and schema.market_spread in games:
        return pd.to_numeric(games[schema.market_spread], errors="coerce")
    if schema.market_moneyline and all(c in games for c in schema.market_moneyline):
        prob = no_vig_probability(*(games[c] for c in schema.market_moneyline))
        z = scipy.stats.norm.ppf(np.clip(prob, 0.005, 0.995))
        return pd.Series(np.where(np.isfinite(prob), z, np.nan), index=games.index)
    return pd.Series(np.nan, index=games.index)


def market_ratings(
    games: pd.DataFrame,
    schema: GameSchema,
    k: float = 0.3,
    carry: float = 0.8,
    home_k: float = 0.01,
) -> pd.DataFrame:
    """:data:`MARKET_RATING_COLUMNS` for each row of ``games``.

    Games are replayed in order; at each game with a line ``s`` (the first
    row's team's), the error ``s - (r_team - r_opp + h * location)`` moves
    both ratings by ``k`` of it (in opposite directions) and the home edge
    ``h`` by ``home_k``. Ratings shrink by ``carry`` between seasons. A row
    gets both ratings after its game's line (before its result).
    """
    line = market_line(games, schema)
    loc = (
        pd.to_numeric(games[schema.location], errors="coerce").fillna(0)
        if schema.location
        else pd.Series(0.0, index=games.index)
    )
    order = games.sort_values([schema.season, schema.date], kind="stable")
    first = ~order[schema.game_id].duplicated()
    order = order[first]
    teams = pd.Index(pd.unique(pd.concat([games[schema.team], games[schema.opp]])))
    team_i = teams.get_indexer(order[schema.team])
    opp_i = teams.get_indexer(order[schema.opp])
    seasons = order[schema.season].to_numpy()
    lines = line.reindex(order.index).to_numpy(dtype=float)
    locs = loc.reindex(order.index).to_numpy(dtype=float)

    rating = np.zeros(len(teams))
    home = 0.0
    after = np.zeros((len(order), 2))
    previous = None
    for i in range(len(order)):
        if previous is not None and seasons[i] != previous:
            rating *= carry
        previous = seasons[i]
        a, b = team_i[i], opp_i[i]
        if np.isfinite(lines[i]):
            err = lines[i] - (rating[a] - rating[b] + home * locs[i])
            rating[a] += k * err
            rating[b] -= k * err
            home += home_k * err * locs[i]
        after[i] = rating[a], rating[b]

    by_game = pd.DataFrame(
        {"_first_team": order[schema.team].to_numpy(), "_ra": after[:, 0],
         "_rb": after[:, 1]},
        index=order[schema.game_id].to_numpy(),
    )  # fmt: skip
    joined = by_game.reindex(games[schema.game_id].to_numpy())
    same = joined["_first_team"].to_numpy() == games[schema.team].to_numpy()
    own = np.where(same, joined["_ra"], joined["_rb"])
    opp = np.where(same, joined["_rb"], joined["_ra"])
    return pd.DataFrame(
        {"_mkt_rating": own, "_mkt_opp_rating": opp, "_mkt_rating_gap": own - opp},
        index=games.index,
    )


def market_features(games: pd.DataFrame, schema: GameSchema) -> pd.DataFrame:
    """Each played row's :data:`MARKET_TOKEN_COLUMNS` (NaN without lines)."""
    import scipy.stats

    from ...backtest import no_vig_probability

    def column(name):
        if name and name in games:
            return pd.to_numeric(games[name], errors="coerce").astype(float)
        return pd.Series(np.nan, index=games.index)

    y = column(schema.target)
    spread = column(schema.market_spread)
    if schema.market_moneyline:
        prob = pd.Series(
            no_vig_probability(*(column(c) for c in schema.market_moneyline)),
            index=games.index,
        )
    else:
        prob = pd.Series(np.nan, index=games.index)
    won = (y > 0).astype(float) + 0.5 * (y == 0)
    line = column(schema.market_total)
    return pd.DataFrame(
        {
            "_mkt_spread": spread,
            "_mkt_cover": y - spread,
            "_mkt_prob_z": scipy.stats.norm.ppf(prob.clip(0.005, 0.995)),
            "_mkt_win_surprise": won - prob,
            "_mkt_total": line,
            "_mkt_total_resid": column(schema.total) - line,
            "_mkt_has_line": (spread.notna() | prob.notna()).astype(float),
        },
        index=games.index,
    )


def strength(
    games: pd.DataFrame,
    schema: GameSchema,
    opponent_strength: bool = True,
    elo_margin: bool = False,
) -> pd.DataFrame:
    """Each played row's pre-game strength features.

    ``_team_elo`` / ``_opp_elo``: Elo ratings going into the game;
    ``_opp_margin_avg``: the opponent's season-to-date average margin before
    the game; ``_margin_vs_elo``: the game's margin minus Elo's expected
    margin (rating gap plus home edge, in points). Elo and the averages
    depend on earlier games only; ``_margin_vs_elo`` adds the game's own
    result, like the raw margin already in each token.
    """
    import dataclasses

    from ..baselines import EloModel
    from ..sklearn.features import training_frame

    games = played(games, schema)
    out = pd.DataFrame(index=games.index)
    elo = EloModel(schema)
    if opponent_strength:
        ratings = elo.pre_game_ratings(games)
        margin = dataclasses.replace(schema, stats=(schema.target,))
        out["_team_elo"] = ratings["team_elo"]
        out["_opp_elo"] = ratings["opp_elo"]
        out["_opp_margin_avg"] = training_frame(games, margin)[
            f"opp_{schema.target}_avg"
        ]
    if elo_margin:
        expected = elo.replay(games) * elo.params.points_per_elo
        out["_margin_vs_elo"] = games[schema.target] - expected
    return out


@dataclass
class SequenceFeatures:
    """Token / context columns and their standardization; saved with a model."""

    token_columns: list[str]
    context_columns: list[str]
    history: int
    mean: dict[str, float]
    std: dict[str, float]
    schema: GameSchema | None = None
    opponent_strength: bool = False
    elo_margin: bool = False
    market_tokens: bool = False
    efficiency: bool = False
    sport: str | None = None
    market_rating: bool = False
    pbp: str | None = None  # this fold's embedding table (resolved path)
    pbp_dim: int = 0
    pbp_only: bool = False
    lineup: str | None = None  # this fold's lineup context table (resolved path)
    lineup_set: str | None = None

    @classmethod
    def fit(
        cls,
        train_games: pd.DataFrame,
        schema: GameSchema,
        history: int,
        opponent_strength: bool = False,
        elo_margin: bool = False,
        market_tokens: bool = False,
        efficiency: bool = False,
        sport: str | None = None,
        market_rating: bool = False,
        pbp: str | None = None,
        pbp_dim: int = 0,
        pbp_only: bool = False,
        lineup: str | None = None,
        lineup_set: str | None = None,
    ) -> SequenceFeatures:
        location = [schema.location] if schema.location else []
        tokens = base_token_columns(schema, pbp_only)
        features = cls(
            token_columns=[],
            context_columns=[],
            history=history,
            mean={},
            std={},
            schema=schema,
            opponent_strength=opponent_strength,
            elo_margin=elo_margin,
            market_tokens=market_tokens,
            efficiency=efficiency,
            sport=sport,
            market_rating=market_rating,
            pbp=pbp if pbp_dim else None,
            pbp_dim=pbp_dim if pbp else 0,
            pbp_only=pbp_only,
            lineup=lineup if lineup_set else None,
            lineup_set=lineup_set if lineup else None,
        )
        extra = extra_columns(
            opponent_strength, elo_margin, market_tokens, efficiency, sport,
            market_rating, features.pbp_dim, pbp_only,
        )  # fmt: skip
        if extra:
            tokens += extra
            train_games = features.with_derived(train_games)
        context = location + list(schema.context)
        if features.lineup:
            context += lineup_context_columns(features.lineup_set)
            train_games = features.with_lineup(train_games)
        columns = list(dict.fromkeys(tokens + context))
        values = train_games.reindex(columns=columns)
        values = values.apply(pd.to_numeric, errors="coerce")
        features.token_columns = tokens
        features.context_columns = context
        features.mean = values.mean().fillna(0.0).to_dict()
        features.std = values.std().replace(0, 1).fillna(1.0).to_dict()
        return features

    def with_derived(self, games: pd.DataFrame) -> pd.DataFrame:
        """``games`` with the derived token columns (strength, Elo margin,
        market, efficiency) this featurizer uses joined on."""
        # saved before these options existed
        market = getattr(self, "market_tokens", False)
        efficiency = getattr(self, "efficiency", False)
        parts = []
        if self.opponent_strength or self.elo_margin:
            parts.append(
                strength(games, self.schema, self.opponent_strength, self.elo_margin)
            )
        if market:
            parts.append(market_features(games, self.schema))
        if getattr(self, "market_rating", False):
            parts.append(market_ratings(games, self.schema))
        if getattr(self, "pbp", None) and getattr(self, "pbp_dim", 0):
            parts.append(pbp_features(games, self.schema, self.pbp, self.pbp_dim))
        if efficiency and not getattr(self, "pbp_only", False):
            from ...core.sport import get_sport

            fn = get_sport(self.sport).efficiency
            if fn is not None:
                parts.append(fn(games))
        for part in parts:
            games = games.join(part.reindex(games.index))
        return games

    @property
    def token_dim(self) -> int:
        return len(self.token_columns)

    @property
    def context_dim(self) -> int:
        return len(self.context_columns)

    def _standardize(self, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
        values = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
        mean = pd.Series(self.mean)[columns]
        std = pd.Series(self.std)[columns]
        z = ((values - mean) / std).fillna(0.0)
        return z.to_numpy(dtype=np.float32).reshape(len(frame), len(columns))

    def tokens(self, games: pd.DataFrame) -> np.ndarray:
        """Standardized tokens for played ``games`` (strength computed from them)."""
        return self._standardize(self.with_derived(games), self.token_columns)

    def with_lineup(self, games: pd.DataFrame) -> pd.DataFrame:
        """``games`` with both sides' lineup context joined (when this featurizer uses it)."""
        if not getattr(self, "lineup", None):
            return games
        part = lineup_features(games, self.schema, self.lineup, self.lineup_set)
        return games.drop(columns=part.columns, errors="ignore").join(part)

    def context(self, matchups: pd.DataFrame) -> np.ndarray:
        return self._standardize(self.with_lineup(matchups), self.context_columns)


def histories(games: pd.DataFrame, schema: GameSchema, history: int) -> np.ndarray:
    """Positions (``iloc``) of each row's team's previous ``history`` games.

    Shape ``[len(games), history]``, most recent first, -1 where the team has
    fewer earlier games. ``games`` must be played games only.
    """
    team = pd.factorize(games[schema.team])[0]
    order = np.lexsort(
        (games[schema.date].to_numpy(), games[schema.season].to_numpy(), team)
    )
    ordered_team = team[order]
    out = np.full((len(games), history), -1, dtype=np.int64)
    for k in range(1, history + 1):
        previous = np.full(len(games), -1, dtype=np.int64)
        same_team = np.zeros(len(games), dtype=bool)
        previous[k:] = order[:-k]
        same_team[k:] = ordered_team[k:] == ordered_team[:-k]
        out[order, k - 1] = np.where(same_team, previous, -1)
    return out


def opponent_rows(games: pd.DataFrame, schema: GameSchema) -> np.ndarray:
    """``iloc`` of the opponent's row of the same game (-1 if absent)."""
    position = pd.Series(
        np.arange(len(games)),
        index=pd.MultiIndex.from_frame(games[[schema.game_id, schema.team]]),
    )
    opp = pd.MultiIndex.from_frame(games[[schema.game_id, schema.opp]])
    return position.reindex(opp).fillna(-1).astype(np.int64).to_numpy()


def sequence(tokens, seasons, index, season) -> dict[str, torch.Tensor]:
    """Gather token sequences; ``index`` -1 entries are padding.

    Args:
        tokens: ``[N, F]`` token table; seasons: ``[N]`` their seasons.
        index: ``[B, K]`` rows of ``tokens`` per sequence.
        season: ``[B]`` season being predicted (for ``seasons_ago``).
    """
    pad = index < 0
    safe = index.clamp(min=0)
    seasons_ago = (season[:, None] - seasons[safe]).masked_fill(pad, 0)
    return {
        "tokens": tokens[safe].masked_fill(pad[..., None], 0.0),
        "seasons_ago": seasons_ago.float(),
        "pad": pad,
    }


class SequenceData(torch.utils.data.Dataset):
    """Training samples (rows of played games) as batches of sequences.

    Index it with a list of sample positions (e.g. via a ``BatchSampler``)
    to get a whole batch.
    """

    def __init__(
        self,
        games: pd.DataFrame,
        schema: GameSchema,
        features: SequenceFeatures,
        rows: np.ndarray | None = None,
        extra_targets: tuple[str, ...] = (),
    ):
        self.tokens = torch.from_numpy(features.tokens(games))
        self.seasons = torch.tensor(games[schema.season].to_numpy(dtype=np.float32))
        self.history = torch.from_numpy(histories(games, schema, features.history))
        opp = torch.from_numpy(opponent_rows(games, schema))
        self.opp_history = torch.where(
            (opp >= 0)[:, None], self.history[opp.clamp(min=0)], -1
        )
        self.context = torch.from_numpy(features.context(games))
        # the context from the opponent's side, for antisymmetric heads
        self.opp_context = torch.from_numpy(features.context(mirror(games, schema)))
        targets = [schema.target, *extra_targets]
        y = games[targets].apply(pd.to_numeric, errors="coerce")
        self.y = torch.tensor(y.to_numpy(dtype=np.float32))
        if not extra_targets:
            self.y = self.y[:, 0]
        self.rows = torch.arange(len(games)) if rows is None else torch.as_tensor(rows)

    def with_rows(self, rows) -> SequenceData:
        other = object.__new__(SequenceData)
        other.__dict__ = {**self.__dict__, "rows": torch.as_tensor(rows)}
        return other

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, positions) -> dict:
        rows = self.rows[torch.as_tensor(positions)]
        season = self.seasons[rows]
        return {
            "team": sequence(self.tokens, self.seasons, self.history[rows], season),
            "opp": sequence(self.tokens, self.seasons, self.opp_history[rows], season),
            "context": self.context[rows],
            "opp_context": self.opp_context[rows],
            "y": self.y[rows],
        }
