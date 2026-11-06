"""``runitback wager web build``: data, model and backtest for the web app."""

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib

import pandas as pd

from .. import espn
from ..core.io import Table, read_table
from ..core.process import process
from ..core.schema import played
from ..core.sport import BUILTIN_SPORTS, get_sport
from .cache import open_store
from .equivalent_code import EQUIVALENT_CODE
from .views import get_view

# Bump to rebuild every cached backtest season once, e.g. after a change the
# code hash cannot see (a dependency upgrade that changes training).
BACKTEST_CACHE_VERSION = 1

# Source the backtest results depend on: everything but the presentation
# layer, the odds client and the CLI (web/build.py's rankings are included).
_UNHASHED = ("web/", "odds/", "cli.py")


def code_hash(sport: str | None = None) -> str:
    """Hash of the package source that shapes backtest results; with
    ``sport``, without the other built-in sports' packages, so changing one
    sport leaves the others' stored backtests valid."""
    root = pathlib.Path(__file__).resolve().parents[1]
    others = tuple(f"{s}/" for s in BUILTIN_SPORTS if sport and s != sport)
    digest = hashlib.sha256()
    for file in sorted(root.rglob("*.py")):
        rel = file.relative_to(root).as_posix()
        if rel.startswith(_UNHASHED) and rel != "web/build.py":
            continue
        if rel.startswith(others):
            continue
        digest.update(rel.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def ranking_dates(dates, every: int | None = None) -> list:
    """The dates of a season to rank teams before: the first, then each
    one at least ``every`` (in the schema's date units) after the last
    kept, and the last. Every date when ``every`` is ``None``."""
    dates = sorted(dates)
    if not every or len(dates) < 3:
        return dates
    kept = [dates[0]]
    for date in dates[1:-1]:
        if date - kept[-1] >= every:
            kept.append(date)
    return [*kept, dates[-1]]


def model_key(sport: str, max_epochs: int, code: str) -> str:
    """Identifies how a sport's model is trained (settings, epochs, code);
    with the season, :func:`backtest_key`."""
    return backtest_key(sport, None, max_epochs, code)


def backtest_key(
    sport: str,
    season: int | None,
    max_epochs: int,
    code: str,
    ranking_every: int | None = None,
    main_model: bool = False,
) -> str:
    """Identifies one season's walk-forward backtest: the same key means the
    same model, training, code and ranking dates (:func:`ranking_dates`),
    so the stored results can be reused. ``main_model``: the season was
    predicted by the model for upcoming games, which validated on it (see
    :func:`build`), not by one fitted for the backtest.

    Game data is deliberately left out: finished seasons are treated as
    final (pass ``refresh_backtest`` after a source revises old data).
    """
    from ..models.transformer.fit import transformer_config

    spec = {
        "version": BACKTEST_CACHE_VERSION,
        "sport": sport,
        "season": None if season is None else int(season),
        "family": "transformer",
        "settings": get_sport(sport).transformer,
        "model": transformer_config(sport),
        "max_epochs": max_epochs,
        "code": code,
    }
    if ranking_every is not None and season is not None:
        # only when set: the keys of sports ranked before every date stay
        spec["ranking_every"] = ranking_every
    if main_model:
        spec["fit"] = "main model, validated on the season"
    text = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def cached_backtest(
    path: pathlib.Path, keys: dict[int, list[str]], season: str = "season"
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """A previous build's pool and rankings rows for the seasons whose
    backtest key is one of the season's ``keys``, and those seasons (with
    the key each was stored under)."""
    empty = pd.DataFrame(), pd.DataFrame(), {}
    try:
        meta = json.loads((path / "meta.json").read_text())
        pool = read_table(str(path / "pool.csv"))
        rankings = read_table(str(path / "rankings.csv"))
    except (OSError, ValueError):
        return empty
    stored = meta.get("backtest_cache", {})
    reuse = {
        s: stored[str(s)]
        for s, accepted in keys.items()
        if stored.get(str(s)) in accepted
        and (pool[season] == s).any()
        and (rankings["season"] == s).any()
    }
    return (
        pool[pool[season].isin(reuse)],
        rankings[rankings["season"].isin(reuse)],
        reuse,
    )


def build(
    sports: list[str] | None = None,
    output_dir: str = "webdata",
    games: Table | None = None,
    backtest_seasons: int = 5,
    max_epochs: int = 100,
    refresh_backtest: bool = False,
    cache: str | None = None,
    reuse_model_days: float = 0,
    max_new_seasons: int | None = None,
) -> None:
    """Prepare everything the web app shows, one directory per sport.

    Downloads and processes the sport's games, trains the transformer, runs
    a walk-forward backtest of it over the last ``backtest_seasons``
    complete seasons (its out-of-sample predictions are the challenge game
    pool, and its team ratings before each date the power rankings) and
    saves team metadata.

    Backtest seasons are reused from the previous build in ``output_dir``,
    or from the shared ``cache``, when nothing that shapes them changed
    (:func:`backtest_key`: the model settings, ``max_epochs`` and the
    model code), so a routine build only trains the model for upcoming
    games and backtests newly completed seasons. Every season the build
    holds is added to ``cache``, so another machine (a GPU box, the Train
    workflow) can reuse it.

    Args:
        sports: Sports to build (default ``["nfl"]``).
        output_dir: Writes ``<output_dir>/<sport>/``.
        games: Use this processed games table instead of downloading (one
            sport only).
        max_epochs: Transformer training epochs (per fit).
        refresh_backtest: Backtest every season again, ignoring the previous
            build and the cache (e.g. after a data source revised old
            seasons); the cache's entries are replaced.
        cache: Shared store of backtest seasons and trained models
            (:func:`~runitback.wager.web.cache.open_store`: a directory, or
            ``release:<tag>`` for a GitHub release every machine can use).
        reuse_model_days: Use the cache's model for upcoming games instead
            of training one, if it was trained with the same settings and
            code at most this many days ago (0: always train). Its team
            state is refreshed with the newest games either way.
        max_new_seasons: Backtest at most this many of the seasons neither
            the previous build nor the cache has (the latest ones); the
            rest are left out until a build elsewhere caches them
            (``None``: backtest them all).
    """
    from ..backtest import backtest, complete_seasons, summarize
    from ..inference.utils import power_ratings
    from ..models.lightning import default_trainer
    from ..models.transformer import fit_transformer

    sports = sports or ["nfl"]
    if games is not None and len(sports) != 1:
        raise ValueError("pass games only when building a single sport")
    for sport in sports:
        view = get_view(sport)
        path = pathlib.Path(output_dir) / sport
        path.mkdir(parents=True, exist_ok=True)

        table = (
            read_table(games)
            if games is not None
            else process(sport, get_sport(sport).download())
        )
        if games is None and (refused := espn.odds_refused()):
            # Backtests are stored by model and code, not by data: a season
            # backtested without its lines would be reused without them.
            # The lines fetched so far are cached; a later run goes on.
            raise RuntimeError(
                f"ESPN refused odds requests for {', '.join(refused)} (403, "
                "usually after a burst of requests from one IP); not building "
                f"{sport} without its lines. The lines fetched so far are in "
                "its download cache: run again later to fetch the rest."
            )
        table.to_csv(path / "games.csv", index=False)
        teams = view.load_teams()
        (path / "teams.json").write_text(teams.to_json(orient="records", indent=1))

        store = open_store(cache)
        code = code_hash(sport)
        mkey = model_key(sport, max_epochs, code)
        stored_model = (
            store.get_model(sport, mkey, path / "model", reuse_model_days)
            if store is not None and reuse_model_days > 0
            else None
        )
        if stored_model is not None:
            print(f"{sport} model: from the cache ({stored_model['trained_at']})")
            fitted = {"metrics": stored_model["metrics"]}
        else:
            trainer = default_trainer(
                max_epochs=max_epochs,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            fitted = fit_transformer(
                table,
                sport,
                trainer=trainer,
                save_dir=str(path / "model"),
                overwrite=True,
            )
            if store is not None:
                trained_at = datetime.datetime.now(datetime.timezone.utc)
                store.put_model(
                    sport,
                    mkey,
                    path / "model",
                    {
                        "trained_at": trained_at.isoformat(timespec="seconds"),
                        "metrics": fitted["metrics"],
                    },
                )
        schema = get_sport(sport).schema
        seasons = complete_seasons(table, schema)[-backtest_seasons:]
        every = view.ranking_every
        # the key recorded for each season: how its results were produced
        keys = {s: backtest_key(sport, s, max_epochs, code, every) for s in seasons}
        main_keys = {
            s: backtest_key(sport, s, max_epochs, code, every, main_model=True)
            for s in seasons
        }
        # the keys a stored season is reused under, each with the key it is
        # recorded as now: from a backtest model or the main model, and from
        # an equivalent older code (stored again under the current key)
        current = {
            s: {keys[s]: keys[s], main_keys[s]: main_keys[s]}
            | {
                backtest_key(sport, s, max_epochs, old, every, main): new[s]
                for old in EQUIVALENT_CODE.get(code, ())
                for main, new in ((False, keys), (True, main_keys))
            }
            for s in seasons
        }
        accepted = {s: list(c) for s, c in current.items()}
        if refresh_backtest:
            pool, rankings, reused = pd.DataFrame(), pd.DataFrame(), {}
        else:
            pool, rankings, reused = cached_backtest(path, accepted, schema.season)
        for s, key in reused.items():
            keys[s] = current[s][key]
        fetched, stored_as = [], {}
        if store is not None and not refresh_backtest:
            parts = [pool], [rankings]
            for s in seasons:
                if s in reused:
                    continue
                for key in accepted[s]:
                    if (entry := store.get_season(sport, s, key)) is not None:
                        break
                else:
                    continue
                parts[0].append(entry[0])
                parts[1].append(entry[1])
                fetched.append(s)
                stored_as[s] = key
                keys[s] = current[s][key]
            pool = pd.concat(parts[0], ignore_index=True)
            rankings = pd.concat(parts[1], ignore_index=True)
        missing = [s for s in seasons if s not in reused and s not in fetched]
        if max_new_seasons is not None and len(missing) > max_new_seasons:
            skipped = missing[: len(missing) - max_new_seasons]
            missing = missing[len(skipped) :]
            seasons = [s for s in seasons if s not in skipped]
            keys = {s: keys[s] for s in seasons}
            print(f"{sport} backtest: leaving out {skipped} (max_new_seasons)")
        # The main model trained on every season before the one it validated
        # on: backtesting that season with it instead of fitting another
        # model on the same seasons saves a training run (the longest step
        # for college basketball). The season chose when its training
        # stopped, so its key says so. Not for a model from the cache,
        # whose seasons can differ from this table's.
        predictors = {}
        if stored_model is None:
            split = fitted["split"]
            seasons_played = set(played(table, schema)[schema.season].unique())
            for s in missing:
                before = {x for x in seasons_played if x < s}
                if split.val == [s] and set(split.train) == before:
                    predictors[s] = fitted["predictor"]
                    keys[s] = main_keys[s]
        print(
            f"{sport} backtest: reusing {list(reused) or 'none'}, from the cache "
            f"{fetched or 'none'}, backtesting {missing or 'none'}"
            + (f" ({list(predictors)} with the main model)" if predictors else ""),
            flush=True,
        )

        ranked = []
        game_days = played(table, schema).groupby(schema.season)[schema.date]
        to_rank = {s: set(ranking_dates(d.unique(), every)) for s, d in game_days}

        def rank(predictor, season, date):
            if date not in to_rank[season]:
                return
            playing = table.loc[table[schema.season] == season, schema.team]
            for team, rating in power_ratings(predictor, playing.unique()).items():
                ranked.append(
                    {"season": season, "date": date, "team": team, "rating": rating}
                )

        if missing:
            tested = backtest(
                table,
                sport,
                family="transformer",
                seasons=missing,
                max_epochs=max_epochs,
                on_date=rank,
                predictors=predictors,
            )
            pool = pd.concat([pool, tested["results"]], ignore_index=True)
            rankings = pd.concat([rankings, pd.DataFrame(ranked)], ignore_index=True)
        pool = pool.sort_values(schema.season, kind="stable")
        rankings = rankings.sort_values(["season", "date"], kind="stable")
        pool.to_csv(path / "pool.csv", index=False)
        rankings.to_csv(path / "rankings.csv", index=False)
        if store is not None:
            for s in seasons:
                # (again under the current key if it came under an older one)
                if stored_as.get(s) == keys[s] or (
                    s not in missing
                    and s not in stored_as
                    and store.has_season(sport, s, keys[s])
                ):
                    continue
                store.put_season(
                    sport,
                    s,
                    keys[s],
                    pool[pool[schema.season] == s],
                    rankings[rankings["season"] == s],
                )

        summary = {"seasons": [int(s) for s in seasons], **summarize(pool, schema)}
        meta = {
            "sport": sport,
            "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "model": {"family": "transformer", "metrics": fitted["metrics"]},
            "backtest": {
                "seasons": summary["seasons"],
                "overall": summary["overall"],
                "ats": summary.get("ats", [{}])[0],
                "moneyline": summary.get("moneyline", [{}])[0],
                "totals": {
                    k: v
                    for k, v in summary.get("totals", {}).items()
                    if k in ("n", "rmse", "brier")
                }
                | {"record": summary.get("totals", {}).get("record", [{}])[0]},
            },
            "backtest_cache": {str(s): key for s, key in keys.items()},
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
        print(f"built {sport} -> {path}")


def push_cache(
    cache: str, data_dir: str = "webdata", sports: list[str] | None = None
) -> None:
    """Add a finished build's backtest seasons (``<data_dir>/<sport>/``,
    with the keys its ``meta.json`` recorded) to the shared ``cache``, e.g.
    to seed it from the current ``webdata`` release."""
    store = open_store(cache)
    root = pathlib.Path(data_dir)
    for path in sorted(p for p in root.iterdir() if (p / "meta.json").exists()):
        sport = path.name
        if sports and sport not in sports:
            continue
        keys = json.loads((path / "meta.json").read_text()).get("backtest_cache", {})
        season = get_sport(sport).schema.season
        pool, rankings, reused = cached_backtest(
            path, {int(s): [k] for s, k in keys.items()}, season
        )
        added = []
        for s in reused:
            if store.has_season(sport, s, keys[str(s)]):
                continue
            store.put_season(
                sport,
                s,
                keys[str(s)],
                pool[pool[season] == s],
                rankings[rankings["season"] == s],
            )
            added.append(s)
        print(f"{sport}: added {added or 'nothing'} to {cache}")


__all__ = ["build", "push_cache"]
