"""Train an ensemble's members in separate processes, then assemble them.

``fit_transformer`` trains ``ensemble`` members one after another, each after
``seed_and_reset(member, seed + i)``, so member ``i`` depends only on its own seed, never on the
members trained before it. :func:`fit_member` trains one of them (``seed=i, ensemble=1``) and
keeps its best checkpoint; :func:`assemble` calls the same ``fit_transformer`` with the members'
checkpoints in place of training, so the ensemble, the validation predictions and the calibration
(``finish_fit``) are made by the same code as an in-process fit. Train predictions runs each
member in its own job (``.github/workflows/train-predictions.yml``).

Lives under ``web/`` (not hashed by :func:`runitback.wager.web.build.code_hash`), so the cached
backtest seasons stay valid; nothing here changes how a model trains.
"""

from __future__ import annotations

import pathlib
import shutil
import types

from ..core.io import Table
from ..models.lightning import default_trainer
from ..models.transformer import fit_transformer


def fit_member(
    games: Table,
    sport: str,
    seed: int,
    out: str | pathlib.Path,
    max_epochs: int = 100,
    **fit_kwargs,
) -> pathlib.Path:
    """Train the ensemble member with seed ``seed`` (the one an in-process fit with the same
    arguments trains ``seed`` places after its first) and copy its best checkpoint to
    ``out/member<seed>.ckpt``. ``fit_kwargs`` go to ``fit_transformer`` (the season split, and
    ``model=`` for a non-default model), as the in-process fit would get them."""
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    trainers = []

    def trainer():
        t = default_trainer(
            max_epochs=max_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainers.append(t)
        return t

    fit_transformer(
        games,
        sport,
        trainer=trainer,
        ensemble=1,
        seed=seed,
        print_metrics=False,
        **fit_kwargs,
    )
    best = trainers[-1].checkpoint_callback.best_model_path
    if not best:
        raise RuntimeError(f"{sport} member {seed}: training left no checkpoint")
    target = out / f"member{seed}.ckpt"
    shutil.copyfile(best, target)
    return target


def assemble(
    games: Table,
    sport: str,
    checkpoints: list[str | pathlib.Path],
    **fit_kwargs,
) -> dict:
    """The fit an in-process ``fit_transformer(..., ensemble=len(checkpoints))`` returns
    (``{"predictor", "metrics", "split", "model"}``), with member ``i`` loaded from
    ``checkpoints[i]`` instead of trained. ``fit_kwargs`` are the in-process fit's, except
    ``trainer``, ``ensemble`` and ``seed``."""
    for key in ("trainer", "ensemble", "seed"):
        if key in fit_kwargs:
            raise TypeError(f"assemble() takes no {key!r}: the checkpoints are the members")
    stubs = iter(
        types.SimpleNamespace(
            fit=lambda *args, **kwargs: None,
            logger=None,
            checkpoint_callback=types.SimpleNamespace(best_model_path=str(path)),
        )
        for path in checkpoints
    )
    return fit_transformer(
        games,
        sport,
        trainer=lambda: next(stubs),
        ensemble=len(checkpoints),
        seed=0,
        **fit_kwargs,
    )


# ---------------------------------------------------------------------------------------------
# The pipeline: plan -> fit-member (matrix) -> main + season (matrix) -> combine (``build``)
# ---------------------------------------------------------------------------------------------
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402

import pandas as pd  # noqa: E402

from ..backtest import backtest, complete_seasons  # noqa: E402
from ..core.io import read_table  # noqa: E402
from ..core.schema import played  # noqa: E402
from ..core.splits import split_seasons  # noqa: E402
from ..core.sport import get_sport  # noqa: E402
from ..inference.utils import power_ratings  # noqa: E402
from ..models.transformer import default_model  # noqa: E402
from ..models.transformer.fit import fit_settings, token_options  # noqa: E402
from .build import (  # noqa: E402
    backtest_key,
    cached_backtest,
    code_hash,
    model_key,
    ranking_dates,
)
from .cache import open_store  # noqa: E402
from .equivalent_code import EQUIVALENT_CODE  # noqa: E402
from .views import get_view  # noqa: E402

MAIN = "main"


def data_sha(games: pd.DataFrame) -> str:
    """Identifies the games table every job of a run must use (the main model's checkpoints are
    only reused for identical data)."""
    return hashlib.sha256(pd.util.hash_pandas_object(games, index=True).to_numpy().tobytes()).hexdigest()


def member_name(sport: str, fit: str | int, seed: int, key: str, data: str | None = None) -> str:
    """The cache entry of member ``seed`` of a fit (``"main"`` or a backtest season); the main
    model's also names the games it trained on."""
    spec = json.dumps([sport, str(fit), int(seed), key, data])
    return f"{sport}-member-{hashlib.sha256(spec.encode()).hexdigest()[:16]}.ckpt"


def ensemble_size(sport: str) -> int:
    return int(fit_settings(sport).get("ensemble", 1))


def season_fit(games: pd.DataFrame, sport: str, season: int) -> tuple[pd.DataFrame, dict]:
    """``(history, fit_kwargs)`` of the model a walk-forward backtest fits for ``season``
    (:func:`runitback.wager.backtest.walk_forward` with :func:`transformer_fitter`)."""
    schema = get_sport(sport).schema
    games = played(read_table(games), schema)
    history = games[games[schema.season] < season]
    split = split_seasons(history[schema.season].unique(), test=[])
    return history, {
        "model": default_model(sport, None, None, None),
        "train_seasons": split.train,
        "val_seasons": split.val,
        "test_seasons": split.test,
    }


def main_reuses(games: pd.DataFrame, sport: str, seasons: list[int]) -> list[int]:
    """The ``seasons`` that the main model stands in for (see :func:`build`): it trained on every
    earlier season and early-stopped on this one. Not with play-by-play embeddings."""
    if token_options(sport).get("pbp"):
        return []
    schema = get_sport(sport).schema
    seasons_played = sorted(played(read_table(games), schema)[schema.season].unique())
    split = split_seasons(seasons_played)
    return [
        s for s in seasons if split.val == [s] and set(split.train) == {x for x in seasons_played if x < s}
    ]


def plan(
    games: pd.DataFrame,
    sport: str,
    cache: str | None = None,
    previous: str | None = None,
    backtest_seasons: int = 5,
    max_epochs: int = 100,
    refresh_backtest: bool = False,
    max_new_seasons: int | None = None,
) -> dict:
    """What a build needs trained: the seasons to backtest (as :func:`build` decides: the
    ``backtest_seasons`` latest complete ones neither the previous build in ``previous`` nor the
    ``cache`` has, the latest ``max_new_seasons`` of them), which of those the main model stands
    in for, and so which fits (``"main"`` and each other season) × seeds to train."""
    schema = get_sport(sport).schema
    table = read_table(games)
    store = open_store(cache)
    code = code_hash(sport)
    every = get_view(sport).ranking_every
    seasons = complete_seasons(table, schema)[-backtest_seasons:]
    keys = {s: backtest_key(sport, s, max_epochs, code, every) for s in seasons}
    main_keys = {s: backtest_key(sport, s, max_epochs, code, every, main_model=True) for s in seasons}
    accepted = {
        s: [keys[s], main_keys[s]]
        + [
            backtest_key(sport, s, max_epochs, old, every, main)
            for old in EQUIVALENT_CODE.get(code, ())
            for main in (False, True)
        ]
        for s in seasons
    }
    reused = {}
    if not refresh_backtest and previous:
        _, _, reused = cached_backtest(pathlib.Path(previous), accepted, schema.season)
    fetched = []
    if store is not None and not refresh_backtest:
        for s in seasons:
            if s not in reused and any(store.has_season(sport, s, k) for k in accepted[s]):
                fetched.append(s)
    missing = [s for s in seasons if s not in reused and s not in fetched]
    if max_new_seasons is not None and len(missing) > max_new_seasons:
        missing = missing[len(missing) - max_new_seasons :]
    main_for = main_reuses(table, sport, missing)
    own = [s for s in missing if s not in main_for]
    n = ensemble_size(sport)
    return {
        "sport": sport,
        "ensemble": n,
        "data": data_sha(table),
        "reused": list(reused),
        "cached": fetched,
        "seasons": missing,
        "main_for": main_for,
        "fits": [MAIN, *map(str, own)],
        "seeds": list(range(n)),
    }


def fit_one(
    games: pd.DataFrame,
    sport: str,
    fit: str,
    seed: int,
    out: str | pathlib.Path,
    cache: str | None = None,
    max_epochs: int = 100,
) -> pathlib.Path:
    """Train member ``seed`` of ``fit`` (``"main"`` or a season) unless the cache already has it,
    leave it at ``out/member<seed>.ckpt`` and add it to the cache."""
    out = pathlib.Path(out)
    store = open_store(cache)
    code = code_hash(sport)
    every = get_view(sport).ranking_every
    if fit == MAIN:
        name = member_name(sport, fit, seed, model_key(sport, max_epochs, code), data_sha(games))
    else:
        name = member_name(sport, fit, seed, backtest_key(sport, int(fit), max_epochs, code, every))
    target = out / f"member{seed}.ckpt"
    out.mkdir(parents=True, exist_ok=True)
    if store is not None and name in store.names() and (data := store.get(name)) is not None:
        target.write_bytes(data)
        print(f"{sport} {fit} member {seed}: from the cache")
        return target
    if fit == MAIN:
        path = fit_member(games, sport, seed, out, max_epochs)
    else:
        history, kwargs = season_fit(games, sport, int(fit))
        path = fit_member(history, sport, seed, out, max_epochs, **kwargs)
    if store is not None:
        store.put(name, path.read_bytes())
    return path


def _checkpoints(directory: str | pathlib.Path, n: int) -> list[pathlib.Path]:
    found = sorted(pathlib.Path(directory).rglob("member*.ckpt"), key=lambda p: int(p.stem.removeprefix("member")))
    seeds = [int(p.stem.removeprefix("member")) for p in found]
    if seeds != list(range(n)):
        raise FileNotFoundError(f"{directory}: members {seeds}, need {list(range(n))}")
    return found


def build_main(
    games: pd.DataFrame,
    sport: str,
    checkpoints: str | pathlib.Path,
    output_dir: str = "webdata",
    cache: str | None = None,
    max_epochs: int = 100,
) -> dict:
    """The main model from its members: saved to ``<output_dir>/<sport>/model`` and stored in the
    cache as :func:`build` would have (so ``build(reuse_model_days=...)`` finds it)."""
    path = pathlib.Path(output_dir) / sport
    path.mkdir(parents=True, exist_ok=True)
    fitted = assemble(
        games, sport, _checkpoints(checkpoints, ensemble_size(sport)), save_dir=str(path / "model"), overwrite=True
    )
    store = open_store(cache)
    if store is not None:
        store.put_model(
            sport,
            model_key(sport, max_epochs, code_hash(sport)),
            path / "model",
            {
                "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "metrics": fitted["metrics"],
            },
        )
    return fitted


def backtest_season(
    games: pd.DataFrame,
    sport: str,
    season: int,
    checkpoints: str | pathlib.Path,
    from_main: bool = False,
    cache: str | None = None,
    max_epochs: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest ``season`` with the model assembled from ``checkpoints`` (its own fit's members,
    or with ``from_main`` the main model's), as :func:`build` does, and add the season to the
    cache under the key :func:`build` would store it by. Returns ``(pool, rankings)``."""
    table = read_table(games)
    schema = get_sport(sport).schema
    ckpts = _checkpoints(checkpoints, ensemble_size(sport))
    if from_main:
        fitted = assemble(table, sport, ckpts)
    else:
        history, kwargs = season_fit(table, sport, season)
        fitted = assemble(history, sport, ckpts, print_metrics=False, **kwargs)
    every = get_view(sport).ranking_every
    game_days = played(table, schema).groupby(schema.season)[schema.date]
    to_rank = {s: set(ranking_dates(d.unique(), every)) for s, d in game_days}
    ranked = []

    def rank(predictor, s, date):
        if date not in to_rank[s]:
            return
        playing = table.loc[table[schema.season] == s, schema.team]
        for team, rating in power_ratings(predictor, playing.unique()).items():
            ranked.append({"season": s, "date": date, "team": team, "rating": rating})

    tested = backtest(
        table,
        sport,
        family="transformer",
        seasons=[season],
        max_epochs=max_epochs,
        on_date=rank,
        predictors={season: fitted["predictor"]},
    )
    pool = tested["results"].sort_values(schema.season, kind="stable")
    rankings = pd.DataFrame(ranked).sort_values(["season", "date"], kind="stable")
    store = open_store(cache)
    if store is not None:
        key = backtest_key(sport, season, max_epochs, code_hash(sport), every, main_model=from_main)
        store.put_season(sport, season, key, pool[pool[schema.season] == season], rankings)
    return pool, rankings


# ------------------------------------------------------------------------------ CLI entry points
# (``runitback-wager web prep|plan|fit-member|main|season``; ``web build`` combines)


def prep(sport: str, out: str = "games.pkl") -> None:
    """Download and process ``sport``'s games once, for every job of a run (the games table
    changes as games are played, so each job must read this one)."""
    from .. import espn
    from ..core.process import process

    table = process(sport, get_sport(sport).download())
    if refused := espn.odds_refused():
        raise RuntimeError(
            f"ESPN refused odds requests for {', '.join(refused)} (403): not building {sport} without its "
            "lines. The lines fetched so far are in its download cache: run again later."
        )
    table.to_pickle(out)


def web_plan(sport: str, games: str, out: str = "plan.json", cache: str | None = None,
             previous: str | None = None, backtest_seasons: int = 5, max_epochs: int = 100,
             refresh_backtest: bool = False, max_new_seasons: int | None = None) -> None:  # fmt: skip
    """Write the build's plan (:func:`plan`) as JSON and print it."""
    result = plan(pd.read_pickle(games), sport, cache, previous, backtest_seasons, max_epochs,
                  refresh_backtest, max_new_seasons)  # fmt: skip
    pathlib.Path(out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


def web_fit_member(sport: str, games: str, fit: str, seed: int, out: str, cache: str | None = None,
                   max_epochs: int = 100) -> None:  # fmt: skip
    """Train one member (:func:`fit_one`)."""
    fit_one(pd.read_pickle(games), sport, fit, seed, out, cache, max_epochs)


def web_main(sport: str, games: str, checkpoints: str, output_dir: str = "webdata",
             cache: str | None = None, max_epochs: int = 100) -> None:  # fmt: skip
    """Assemble and store the main model (:func:`build_main`)."""
    build_main(pd.read_pickle(games), sport, checkpoints, output_dir, cache, max_epochs)


def web_season(sport: str, games: str, season: int, checkpoints: str, from_main: bool = False,
               cache: str | None = None, max_epochs: int = 100) -> None:  # fmt: skip
    """Backtest one season (:func:`backtest_season`) and store it."""
    backtest_season(pd.read_pickle(games), sport, season, checkpoints, from_main, cache, max_epochs)


def web_combine(sport: str, games: str, output_dir: str = "webdata", cache: str | None = None,
                backtest_seasons: int = 5, max_epochs: int = 100, max_model_days: float = 0.5) -> None:  # fmt: skip
    """Write the sport's web data (``<output_dir>/<sport>/``) from what the other steps stored in
    the cache: :func:`runitback.wager.web.build.build` with the main model and every backtest season
    taken from it (never trained here). The model must be one of the last ``max_model_days``."""
    from .build import build

    build(
        sports=[sport],
        output_dir=output_dir,
        games=pd.read_pickle(games),
        backtest_seasons=backtest_seasons,
        max_epochs=max_epochs,
        cache=cache,
        reuse_model_days=max_model_days,
        max_new_seasons=0,
    )
