"""SHA-256 fingerprints of the two-outcome wager outputs on the toy league (tests/wager/conftest.py).

    python scripts/wager_fingerprint.py            # prints {"<piece>": "<sha256>", ...}

Run from the repository root. It uses only deterministic numpy / pandas code (the naive baselines'
walk-forward results, the summaries, comparisons, experiment tables, tuning score and betting
tables built from them), with floats rounded to 6 places, so the hashes do not depend on the
machine. `tests/wager/test_outcome_identity.py` pins them: the three-outcome (draws) mode must not
change what the other sports compute. To check a change, run this on `origin/main` (in a
worktree: `PYTHONPATH=<worktree>/src python <worktree>/scripts/wager_fingerprint.py`) and on the
branch, and compare.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runitback.wager import backtest as bt
from runitback.wager import experiments, staking, tune
from runitback.wager.core.sport import Sport, register_sport
from tests.wager.conftest import TOY_SCHEMA, make_games

DECIMALS = 6


def _canonical(value):
    if isinstance(value, pd.DataFrame):
        value = value.reset_index()
        return {c: _canonical(value[c].tolist()) for c in value.columns}
    if isinstance(value, pd.Series):
        return _canonical(value.tolist())
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_canonical(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else round(float(value), DECIMALS)
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def digest(value) -> str:
    text = json.dumps(_canonical(value), sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def fingerprints() -> dict[str, str]:
    register_sport(
        Sport(name="toy", schema=TOY_SCHEMA, download=lambda output_file=None: None)
    )
    games = make_games(persistence=0.7)
    schema = TOY_SCHEMA
    seasons = [2023]
    results = {
        name: bt.walk_forward(games, "toy", bt.baseline_fitter("toy", name), seasons)
        for name in ("home", "point_diff", "elo")
    }
    out = {f"walk_forward/{name}": digest(frame) for name, frame in results.items()}
    for name, frame in results.items():
        out[f"summarize/{name}"] = digest(bt.summarize(frame, schema))
    table, per_game = bt.compare(results, schema)
    out["compare/table"], out["compare/games"] = digest(table), digest(per_game)
    out["ablation_table"] = digest(experiments.ablation_table(results, "toy", "home"))
    out["season_table"] = digest(experiments.season_table(results, "toy", "home"))
    out["tune/season_brier"] = digest(
        {name: tune.season_brier(frame, "toy") for name, frame in results.items()}
    )
    elo = results["elo"]
    for market in staking.MARKETS:
        try:  # the baselines predict no total
            out[f"market_bets/{market}"] = digest(
                staking.market_bets(elo, schema, market)
            )
        except ValueError as error:
            out[f"market_bets/{market}"] = digest(str(error))
    out["no_vig_probability"] = digest(
        bt.no_vig_probability(
            np.array([-150.0, 120.0, 250.0, -110.0, np.nan]),
            np.array([130.0, -140.0, -300.0, -110.0, 100.0]),
        )
    )
    return out


if __name__ == "__main__":
    print(json.dumps(fingerprints(), indent=2, sort_keys=True))
