"""Ablations: several transformer configurations through the same
walk-forward backtest, scored on the same games with paired differences.

Each configuration is a dict with any of

- ``model``: :func:`~runitback.wager.models.transformer.default_model` overrides,
  e.g. ``{"market_tokens": True, "antisymmetric": True, "lr": 1e-3}``;
- ``fit``: :func:`~runitback.wager.models.transformer.fit_transformer` arguments,
  e.g. ``{"ensemble": 5, "batch_size": 128}``;
- ``max_epochs`` / ``patience``: training length and early stopping.

Every (configuration, season) pair is one job: a fit on the earlier seasons
and the season replayed date by date, as in :func:`~runitback.wager.backtest.
walk_forward`. Jobs run in parallel processes (``n_jobs``, each with
``threads`` torch threads) and each one's predictions are cached as
``<output_dir>/<name>/<season>.csv``, so a rerun only does what is missing;
change a configuration's name when you change its settings.

:func:`ablation_table` then scores every configuration on the games all of
them predicted (and the market priced):

- Brier, log loss and RMSE;
- ``d_brier``: the paired per-game Brier difference to the ``baseline``
  configuration, with its 95% CI (negative is better);
- ``vs_market``: the same against the market's no-vig probability;
- ``blend_w``: the weight ``w`` that minimizes the Brier score of
  ``(1 - w) * market + w * model`` on these games, with its 95% CI. A ``w``
  above 0 means the model's probability carries information the market's
  does not; ``blend_gain`` is the Brier the blend saves over the market
  (in-sample, so an upper bound);
- a ``vegas`` row: the market's own scores on the same games;
- ``edge_corr`` (the correlation of model minus market margin with the
  result minus the market margin), against-the-spread and over/under pick
  rates.
"""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import pathlib
import time

import numpy as np
import pandas as pd

from .backtest import ats_record, totals_record
from .core.io import Table, read_table
from .core.metrics import evaluate
from .core.schema import GameSchema
from .core.sport import get_sport

# the ablation table's row for the betting market itself
MARKET_ROW = "vegas"


def _run_job(
    games_path: str,
    sport: str,
    name: str,
    config: dict,
    season: int,
    output: str,
    threads: int,
) -> str:
    import logging
    import warnings

    import torch

    from .backtest import transformer_fitter, walk_forward

    warnings.filterwarnings("ignore")
    logging.getLogger("lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    torch.set_num_threads(threads)
    start = time.time()
    games = read_table(games_path)
    fitter = transformer_fitter(
        sport,
        max_epochs=config.get("max_epochs", 100),
        patience=config.get("patience", 10),
        model=config.get("model") or {},
        **(config.get("fit") or {}),
    )
    results = walk_forward(games, sport, fitter, [season])
    path = pathlib.Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
    return f"{name} {season}: {len(results)} games, {time.time() - start:.0f}s"


def run_experiments(
    games: str,
    sport: str,
    configs: dict[str, dict],
    seasons: list[int],
    output_dir: str,
    n_jobs: int = 4,
    threads: int = 1,
) -> dict[str, pd.DataFrame]:
    """Backtest every configuration on every season (missing ones only).

    Args:
        games: Path of the processed games table.
        configs: ``{name: config}`` (see the module docstring).
        seasons: Test seasons.
        output_dir: Cache of per-season predictions.
        n_jobs / threads: Parallel processes and torch threads per process.

    Returns:
        ``{name: per-game walk-forward results over seasons}``
    """
    out = pathlib.Path(output_dir)
    jobs = [
        (name, config, int(season), out / name / f"{season}.csv")
        for name, config in configs.items()
        for season in seasons
        if not (out / name / f"{season}.csv").exists()
    ]
    if jobs:
        print(f"{len(jobs)} backtest jobs on {n_jobs} processes", flush=True)
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(n_jobs, mp_context=context) as pool:
            futures = [
                pool.submit(
                    _run_job, games, sport, name, config, season, str(path), threads
                )
                for name, config, season, path in jobs
            ]
            for future in concurrent.futures.as_completed(futures):
                print(future.result(), flush=True)
    return load_results(output_dir, list(configs), seasons)


def load_results(
    output_dir: str, names: list[str], seasons: list[int]
) -> dict[str, pd.DataFrame]:
    """Cached per-season predictions of each configuration in ``names``."""
    out = pathlib.Path(output_dir)
    results = {}
    for name in names:
        frames = [
            pd.read_csv(path, low_memory=False)
            for season in seasons
            if (path := out / name / f"{season}.csv").exists()
        ]
        if frames:
            results[name] = pd.concat(frames, ignore_index=True)
    return results


def _paired(diff: np.ndarray) -> tuple[float, float]:
    diff = diff[np.isfinite(diff)]
    return float(diff.mean()), float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff)))


def blend_weight(won, market, model) -> dict[str, float]:
    """Brier-optimal ``w`` for ``(1 - w) * market + w * model``, its 95% CI
    (from per-game influence) and the Brier it saves over the market."""
    won, market, model = (np.asarray(a, dtype=float) for a in (won, market, model))
    d = model - market
    r = won - market
    w = float(np.sum(d * r) / np.sum(d * d))
    influence = d * (r - w * d) / np.mean(d * d)
    half = float(1.96 * influence.std(ddof=1) / np.sqrt(len(d)))
    gain = float(np.mean(r**2) - np.mean((r - w * d) ** 2))
    return {"blend_w": w, "blend_w_ci95": half, "blend_gain": gain}


def common_games(
    results: dict[str, pd.DataFrame], schema: GameSchema
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Restrict every configuration's results to the games all of them
    predicted and the market priced; returns (base table, aligned results)."""
    ids = set.intersection(*(set(r[schema.game_id]) for r in results.values()))
    first = next(iter(results.values()))
    base = first[first[schema.game_id].isin(ids)].dropna(
        subset=["market_prob", "market_margin"]
    )
    base = base.drop_duplicates(schema.game_id).set_index(schema.game_id)
    aligned = {
        name: frame.drop_duplicates(schema.game_id)
        .set_index(schema.game_id)
        .reindex(base.index)
        for name, frame in results.items()
    }
    return base, aligned


def ablation_table(
    results: dict[str, pd.DataFrame], sport: str, baseline: str | None = None
) -> pd.DataFrame:
    """Score each configuration on common games (see the module docstring)."""
    schema = get_sport(sport).schema
    baseline = baseline or next(iter(results))
    base, aligned = common_games(results, schema)
    y = base[schema.target].to_numpy(dtype=float)
    decided = y != 0
    won = (y > 0).astype(float)
    market = base["market_prob"].to_numpy(dtype=float)

    def brier(prob):
        return (prob - won) ** 2

    ref = brier(aligned[baseline]["win_prob"].to_numpy(dtype=float))
    rows = []
    for name, frame in {**aligned, MARKET_ROW: None}.items():
        if frame is None:
            margin, prob = base["market_margin"].to_numpy(dtype=float), market
        else:
            margin = frame["pred_margin"].to_numpy(dtype=float)
            prob = frame["win_prob"].to_numpy(dtype=float)
        m = evaluate(y, margin, prob)
        row = {
            "config": name,
            "n": int(m["n"]),
            "brier": m["brier"],
            "log_loss": m["log_loss"],
            "rmse": m["rmse"],
        }
        if frame is not None:
            row["d_brier"], row["d_brier_ci95"] = _paired((brier(prob) - ref)[decided])
            row["vs_market"], row["vs_market_ci95"] = _paired(
                (brier(prob) - brier(market))[decided]
            )
            row |= blend_weight(won[decided], market[decided], prob[decided])
            edge = margin - base["market_margin"].to_numpy(dtype=float)
            cover = y - base["market_margin"].to_numpy(dtype=float)
            row["edge_corr"] = float(np.corrcoef(edge, cover)[0, 1])
            one = base.reset_index().assign(pred_margin=margin)
            if schema.market_spread:
                ats = ats_record(one, schema, thresholds=(0,)).iloc[0]
                row["ats_pct"] = float(ats["win_pct"])
            if "pred_total" in frame and frame["pred_total"].notna().any():
                one["pred_total"] = frame["pred_total"].to_numpy(dtype=float)
                totals = totals_record(one, schema, thresholds=(0,)).iloc[0]
                row["ou_pct"] = float(totals["win_pct"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("config")


def season_table(
    results: dict[str, pd.DataFrame], sport: str, baseline: str | None = None
) -> pd.DataFrame:
    """Each configuration's Brier difference to ``baseline`` by season."""
    schema = get_sport(sport).schema
    baseline = baseline or next(iter(results))
    base, aligned = common_games(results, schema)
    y = base[schema.target].to_numpy(dtype=float)
    won = (y > 0).astype(float)
    decided = y != 0
    season = base[schema.season].to_numpy()
    ref = (aligned[baseline]["win_prob"].to_numpy(dtype=float) - won) ** 2
    out = {}
    for name, frame in aligned.items():
        diff = (frame["win_prob"].to_numpy(dtype=float) - won) ** 2 - ref
        out[name] = pd.Series(diff[decided]).groupby(season[decided]).mean()
    return pd.DataFrame(out)


def format_table(table: pd.DataFrame) -> str:
    """Markdown table of :func:`ablation_table`."""

    def pm(value, half, digits=4):
        if pd.isna(value):
            return ""
        return f"{value:+.{digits}f} ± {half:.{digits}f}"

    lines = [
        "| config | n | Brier | log loss | RMSE | Δ Brier vs baseline "
        "| vs market | blend w | edge r | ATS | O/U |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in table.iterrows():
        cells = [
            name,
            f"{r['n']}",
            f"{r['brier']:.4f}",
            f"{r['log_loss']:.4f}",
            f"{r['rmse']:.2f}",
            pm(r.get("d_brier"), r.get("d_brier_ci95")),
            pm(r.get("vs_market"), r.get("vs_market_ci95")),
            pm(r.get("blend_w"), r.get("blend_w_ci95"), 2),
            "" if pd.isna(r.get("edge_corr")) else f"{r['edge_corr']:+.3f}",
            "" if pd.isna(r.get("ats_pct")) else f"{100 * r['ats_pct']:.1f}%",
            "" if pd.isna(r.get("ou_pct")) else f"{100 * r['ou_pct']:.1f}%",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_seasons(by_season: pd.DataFrame) -> str:
    """Markdown table of :func:`season_table`."""
    lines = [
        "| season | " + " | ".join(by_season.columns) + " |",
        "|---" * (len(by_season.columns) + 1) + "|",
    ]
    for season, row in by_season.iterrows():
        cells = [f"{v:+.4f}" for v in row]
        lines.append(f"| {season} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ablate(
    games: Table,
    sport: str,
    configs: dict[str, dict],
    seasons: list[int],
    output_dir: str,
    baseline: str | None = None,
    n_jobs: int = 4,
    threads: int = 1,
) -> dict:
    """Run :func:`run_experiments` and write ``ablation.md`` /
    ``ablation.json`` (the table and the per-season differences) to
    ``output_dir``; ``baseline`` defaults to the first configuration."""
    if not isinstance(games, str):
        raise ValueError("pass the games table's path (jobs run in subprocesses)")
    results = run_experiments(
        games, sport, configs, seasons, output_dir, n_jobs, threads
    )
    table = ablation_table(results, sport, baseline)
    by_season = season_table(results, sport, baseline)
    text = format_table(table)
    path = pathlib.Path(output_dir)
    (path / "ablation.md").write_text(
        f"{sport}, seasons {list(seasons)}\n\n{text}\n\nΔ Brier by season\n\n"
        + format_seasons(by_season)
        + "\n"
    )
    (path / "ablation.json").write_text(
        json.dumps(
            {
                "sport": sport,
                "seasons": [int(s) for s in seasons],
                "configs": configs,
                "table": table.reset_index().to_dict("records"),
                "by_season": by_season.reset_index().to_dict("records"),
            },
            indent=2,
            default=float,
        )
    )
    print(text)
    return {"table": table, "by_season": by_season, "results": results}
