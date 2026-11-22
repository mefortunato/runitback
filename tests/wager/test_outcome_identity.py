"""The three-outcome (draws) mode must not change what the two-outcome sports compute.

``scripts/wager_fingerprint.py`` hashes the deterministic two-outcome outputs (walk-forward results
of the naive baselines, summaries, comparisons, experiment tables, the tuning score and the betting
tables) on the toy league. These hashes were recorded on ``origin/main`` (c33f968), before any draws
code. A change that moves one of them changed a two-outcome sport's output: if that is intended,
say so in the PR and re-record them (``python scripts/wager_fingerprint.py``).
"""

import importlib.util
import pathlib

SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "wager_fingerprint.py"
)

MAIN = {
    "ablation_table": "9acbc10dba9b62146c70c6f67d29a5f191a449deb5818663a5f6466d2daab4d2",
    "compare/games": "52e64e30601bceac60b38c605c979a5df670fa215b3f8b186b623cd15c74ccc3",
    "compare/table": "d68cd78c6b4ea5d45e7690b27260e679591d7e0c7a5adfa3630f039434a1cea6",
    "market_bets/moneyline": "449bad29ba248b3510f4af1f4aea61218630a69e4dc546199ea138b3aea442d4",
    "market_bets/spread": "3225681c7afd68eb27c08334b037a301cd0de2f1afc0ff9235bc241f3875846c",
    "market_bets/total": "2307fd371fab34b7fb9204127074df8acc1c730c044b1d5986ba3fe0d552e79f",
    "no_vig_probability": "e7679934f289f76210ff6fced8334e4e9c1fe8ad125516e36f71bce49b8309d2",
    "season_table": "6534adfe8f4d419608c807bbe7480165c5ff734fa88609b383a3ff25c819c981",
    "summarize/elo": "a0c82022349a2d440d2b6ac233bb1b7f7d54db22ce6fc2caa3b6a7c990c4c01d",
    "summarize/home": "4c665c1bee05f6197594b0314476bca683a788e69796be7f9d290d84755fbed9",
    "summarize/point_diff": "75d6130cb88c276e76676d3ec64d48ba613c77e4b0606c6a1d2ff783b09d9061",
    "tune/season_brier": "ec958d1d2a2b0cf0066110067c359471d7f65655ff1429a9c9fffdda9dd28075",
    "walk_forward/elo": "a2ebe13922cce46fd5f94aa22e698dc05535438c2bb28475e7701d6e21b58afa",
    "walk_forward/home": "f9d27dc14bca28593c56da5d21d70997e7d1e1044a2415957576b23a9e5aa5fd",
    "walk_forward/point_diff": "f3d73f46345bc361ce8d9c2ab262bb9f239d4b1cb9db70f7619c7b3b9ee8c89c",
}


def load_script():
    spec = importlib.util.spec_from_file_location("wager_fingerprint", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_outcome_outputs_match_main():
    assert load_script().fingerprints() == MAIN
