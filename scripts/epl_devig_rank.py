"""Rank the de-vig methods by the market's own ranked probability score.

    python scripts/epl_devig_rank.py [--cache data/epl/epl_cache]

For each period it removes the bookmaker margin from the 1X2 prices with every method in
`runitback.wager.backtest.DEVIG_METHODS` and scores the resulting home / draw / away probabilities
against what happened (RPS, lower is better), with the paired difference to `proportional`.
Seasons are by ending year. 2006-2016 use the prices the ingest prefers (opening bet365, then
closing Pinnacle: `odds_source`); 2020-2021 use the closing market average. All of them are before
the first test season (2022), so choosing a method from them leaks nothing.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from runitback.wager.backtest import DEVIG_METHODS, no_vig_probabilities
from runitback.wager.core.metrics import three_way_scores
from runitback.wager.epl.data.download import download

PERIODS = {
    "2006-2016 (opening bet365, then closing Pinnacle)": (range(2006, 2017), ("moneyline", "draw_moneyline", "opp_moneyline")),
    "2020-2021 (closing average)": (range(2020, 2022), ("avg_moneyline", "avg_draw_moneyline", "opp_avg_moneyline")),
}  # fmt: skip


def rank(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    home = games[(games["location"] == 1) & games["result"].notna()]
    for period, (seasons, columns) in PERIODS.items():
        frame = home[home["season"].isin(list(seasons))].dropna(subset=list(columns))
        scores = {}
        # the prices are (home, draw, away): the team is the home side, so its win is column 0
        for method in DEVIG_METHODS:
            p = no_vig_probabilities(*(frame[c] for c in columns), method=method)
            scores[method] = three_way_scores(frame["result"], p[:, 0], p[:, 1])
        base = scores["proportional"]["rps"]
        for method, s in scores.items():
            diff = s["rps"] - base
            half = (
                1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
                if method != "proportional"
                else 0.0
            )
            rows.append({"period": period, "method": method, "games": len(frame), "rps": s["rps"].mean(),
                         "log_loss3": s["log_loss3"].mean(), "d_rps_vs_proportional": diff.mean(), "ci95": half})  # fmt: skip
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="data/epl/epl_cache")
    table = rank(download(cache_dir=parser.parse_args().cache))
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
