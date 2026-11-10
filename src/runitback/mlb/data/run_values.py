"""RE24 run values for plate appearances, with run expectancy fit on earlier seasons only.

Run expectancy ``RE(bases, outs)`` is the mean number of runs scored from a base / out state to
the end of the half inning, over plate appearances that start in that state in half innings that
ended with three outs (walk-offs and stopped games are left out). A plate appearance's run value is

    run_value = RE(end) - RE(pre) + runs

with ``pre`` the state just before the event that ended the plate appearance (after steals, wild
pitches and pickoffs during it), ``end`` the state after it (``RE = 0`` at three outs) and
``runs`` the runs scored on that event. This credits the batter and pitcher with their own
event only. ``run_value_pa`` is the same from the state at the start of the plate appearance,
counting every run scored during it.

**No leakage:** season S's values use the run expectancy fit on seasons S-``lookback`` .. S-1.
A season with no earlier season available gets NaN run values (download one extra season first).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATES = pd.MultiIndex.from_product([range(8), range(3)], names=["bases", "outs"])


def run_expectancy(pa: pd.DataFrame, min_count: int = 1) -> pd.DataFrame:
    """``RE(bases, outs)`` from plate appearances (the ``pa`` table): 24 rows, indexed by
    ``(bases, outs)``, with ``re`` (mean runs to the end of the half inning) and ``n``."""
    pa = pa[pa["half_complete"].astype(bool) & (pa["start_outs"] < 3)]
    grouped = pa.groupby(["start_bases", "start_outs"])["runs_rest"].agg(["mean", "size"])
    grouped.index = grouped.index.set_names(["bases", "outs"])
    table = grouped.reindex(STATES).rename(columns={"mean": "re", "size": "n"})
    table["n"] = table["n"].fillna(0).astype(int)
    table.loc[table["n"] < min_count, "re"] = np.nan
    return table


def lookup(table: pd.DataFrame, bases: pd.Series, outs: pd.Series) -> np.ndarray:
    """``RE`` of each (bases, outs); 0 at three outs."""
    re = table["re"].to_numpy().reshape(8, 3)
    b = bases.to_numpy(dtype=int)
    o = outs.to_numpy(dtype=int)
    out = np.zeros(len(b))
    live = o < 3
    out[live] = re[b[live], o[live]]
    return out


def add_run_values(pa: pd.DataFrame, table: pd.DataFrame | None) -> pd.DataFrame:
    """``pa`` with ``re_start``, ``re_pre``, ``re_end``, ``run_value`` and ``run_value_pa``
    (NaN when ``table`` is None)."""
    pa = pa.copy()
    if table is None:
        for col in ("re_start", "re_pre", "re_end", "run_value", "run_value_pa"):
            pa[col] = np.nan
        return pa
    pa["re_start"] = lookup(table, pa["start_bases"], pa["start_outs"])
    pa["re_pre"] = lookup(table, pa["pre_bases"], pa["pre_outs"])
    pa["re_end"] = lookup(table, pa["end_bases"], pa["end_outs"])
    pa["run_value"] = pa["re_end"] - pa["re_pre"] + pa["runs"]
    pa["run_value_pa"] = pa["re_end"] - pa["re_start"] + pa["runs_pre"] + pa["runs"]
    return pa


def season_run_values(seasons: dict[int, pd.DataFrame], lookback: int = 1
                      ) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    """Run values for every season in ``seasons`` (season -> pa table), each from the run
    expectancy of the ``lookback`` seasons before it. Returns ``(pa by season, RE by season)``;
    a season without an earlier one available has NaN values and no RE table."""
    values, tables = {}, {}
    for season in sorted(seasons):
        earlier = [seasons[s] for s in range(season - lookback, season) if s in seasons]
        table = run_expectancy(pd.concat(earlier, ignore_index=True)) if earlier else None
        values[season] = add_run_values(seasons[season], table)
        if table is not None:
            tables[season] = table
    return values, tables
