"""RPS of a walk-forward backtest on promoted clubs' first games, against the rest.

    python scripts/epl_promoted_rps.py data/epl/games.csv backtests/epl/elo_games.csv [--first 5]

`games` is the processed games table (`runitback-wager epl process`) and `results` a backtest's
`<family>_games.csv`. A game counts when a club that was promoted that season (`promoted`: not in the
previous season's league) is playing one of its first `--first` league games of the season, the games a
model with no history of the club has the least to go on. Reports the model's and the market arm's RPS
(and the paired difference) on those games and on all the others.
"""
from __future__ import annotations

import argparse

import pandas as pd

from runitback.wager.core.metrics import paired_difference, three_way_scores


def early_promoted(games: pd.DataFrame, first: int) -> set[str]:
    """Ids of the games in which a promoted club plays one of its first ``first`` games of a season."""
    rows = games.sort_values(["season", "day", "game_id"])
    rows = rows.assign(number=rows.groupby(["season", "team"]).cumcount() + 1)
    hit = (rows["promoted"] == 1) & (rows["number"] <= first)
    return set(rows.loc[hit, "game_id"])


def report(games: pd.DataFrame, results: pd.DataFrame, first: int = 5) -> pd.DataFrame:
    early = early_promoted(games, first)
    rows = []
    for name, mask in (
        (f"promoted club's first {first}", results["game_id"].isin(early)),
        ("all other games", ~results["game_id"].isin(early)),
    ):
        frame = results[mask].dropna(subset=["market_prob", "market_draw_prob"])
        model = three_way_scores(frame["result"], frame["win_prob"], frame["draw_prob"])["rps"]
        market = three_way_scores(frame["result"], frame["market_prob"], frame["market_draw_prob"])["rps"]
        diff, half = paired_difference(model - market)
        rows.append({"games": name, "n": len(frame), "model_rps": model.mean(),
                     "market_rps": market.mean(), "model_minus_market": diff, "ci95": half})  # fmt: skip
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("games")
    parser.add_argument("results")
    parser.add_argument("--first", type=int, default=5)
    args = parser.parse_args()
    games = pd.read_csv(args.games)
    results = pd.read_csv(args.results)
    print(report(games, results, args.first).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
