"""Score the foundation experiment's wager configurations on the three markets.

Reads <dir>/<config>/<season>.csv (written by `runitback-wager nfl ablate`), keeps the games every
configuration predicted and the market priced, and reports per market (moneyline: home win;
spread: home cover; total: over; pushes left out) each configuration's Brier and log loss, the
paired per-game Brier difference to the baseline with its 95% CI, and whether it passes: a CI that
includes 0 or favors the configuration ("not significantly worse").

    python scripts/foundation_report.py experiments/nfl/foundation --baseline prod
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MARKETS = {
    "moneyline": ("win_prob", "market_prob"),
    "spread": ("cover_prob", None),
    "total": ("over_prob", "market_over_prob"),
}


def load(root: Path, configs: list[str] | None) -> dict[str, pd.DataFrame]:
    out = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(d.glob("[0-9]*.csv"))
        if not files or (configs and d.name not in configs):
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        if "win_prob" not in df:
            continue
        out[d.name] = df[df.location == 1].set_index("game_id") if (df.location == 1).any() else df.set_index("game_id")
    return out


def outcomes(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """market -> (outcome, decided)"""
    return {
        "moneyline": ((df.result > 0).astype(float).to_numpy(), (df.result != 0).to_numpy()),
        "spread": ((df.result > df.market_margin).astype(float).to_numpy(),
                   (df.result != df.market_margin).to_numpy() & df.market_margin.notna().to_numpy()),
        "total": ((df.total > df.total_line).astype(float).to_numpy(),
                  (df.total != df.total_line).to_numpy() & df.total_line.notna().to_numpy()),
    }


def report(results: dict[str, pd.DataFrame], baseline: str) -> pd.DataFrame:
    common = None
    for df in results.values():
        ok = df.index[df.market_margin.notna() & df.total_line.notna() & df.cover_prob.notna() & df.over_prob.notna()]
        common = ok if common is None else common.intersection(ok)
    base = results[baseline].loc[common]
    ys = outcomes(base)
    rows = []
    for name, df in results.items():
        df = df.loc[common]
        for market, (col, mcol) in MARKETS.items():
            y, decided = ys[market]
            p = np.clip(df[col].to_numpy(dtype=float), 1e-4, 1 - 1e-4)[decided]
            pb = np.clip(base[col].to_numpy(dtype=float), 1e-4, 1 - 1e-4)[decided]
            yy = y[decided]
            b, bb = (p - yy) ** 2, (pb - yy) ** 2
            diff = b - bb
            ci = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)) if name != baseline else 0.0
            row = dict(config=name, market=market, n=len(yy), brier=b.mean(),
                       log_loss=float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean()),
                       d_brier=diff.mean(), ci=ci, passes=bool(diff.mean() - ci <= 0))
            if mcol and mcol in df:
                m = np.clip(df[mcol].to_numpy(dtype=float), 1e-4, 1 - 1e-4)[decided]
                vm = b - (m - yy) ** 2
                row |= dict(vs_market=vm.mean(), vs_market_ci=1.96 * vm.std(ddof=1) / np.sqrt(len(vm)))
            rows.append(row)
        rows.append(dict(config=name, market="margin_rmse", n=len(df),
                         brier=float(np.sqrt(((df.pred_margin - base.result) ** 2).mean()))))
    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame) -> str:
    lines = ["| config | market | n | Brier | log loss | Δ Brier vs baseline (95% CI) | passes | vs market |",
             "|---|---|---|---|---|---|---|---|"]
    for r in table[table.market != "margin_rmse"].itertuples():
        vm = f"{r.vs_market:+.4f} ± {r.vs_market_ci:.4f}" if "vs_market" in table and pd.notna(r.vs_market) else ""
        lines.append(f"| {r.config} | {r.market} | {r.n} | {r.brier:.4f} | {r.log_loss:.4f} | "
                     f"{r.d_brier:+.4f} ± {r.ci:.4f} | {'yes' if r.passes else '**no**'} | {vm} |")
    rm = table[table.market == "margin_rmse"]
    lines += ["", "| config | margin RMSE |", "|---|---|"] + [f"| {r.config} | {r.brier:.2f} |" for r in rm.itertuples()]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--baseline", default="prod")
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    table = report(load(a.root, a.configs), a.baseline)
    text = to_markdown(table)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")
        table.to_csv(a.out.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
