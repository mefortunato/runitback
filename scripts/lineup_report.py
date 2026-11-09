"""Score the lineup experiment's arms overall and on the games where personnel moved.

Lineup changes are rare, so a whole-season average mostly measures weeks that look like the week
before. Besides all games, this scores the arms on subsets chosen from the lineup tables'
change flags (experiments/nfl/lineups/changes/<season>.csv, either team):

- qb_changed: the starting QB differed from the team's previous game (as played: an evaluation
  subset only, not a model input)
- qb_projected: the projected QB1 differs from the previous game's (what the model was told)
- turnover: the top quarter of games by previous-game starters missing from the projected chart
- steady: none of the above

    python scripts/lineup_report.py experiments/nfl/lineups --baseline pbp32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from foundation_report import load, report  # noqa: E402


def subsets(root: Path) -> dict[str, set[str]]:
    files = sorted((root / "changes").glob("*.csv"))
    ch = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    g = ch.groupby("game_id").agg(qb=("_qb_change_actual", "max"), proj=("_qb_change", "max"),
                                  dropped=("_dropped", "max"))
    top = g.dropped >= g.dropped.quantile(0.75)
    moved = (g.qb > 0) | (g.proj > 0) | top
    return {"qb_changed": set(g.index[g.qb > 0]), "qb_projected": set(g.index[g.proj > 0]),
            "turnover": set(g.index[top]), "steady": set(g.index[~moved])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--baseline", default="pbp32")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    results = load(a.root, None)
    parts = [report(results, a.baseline).assign(subset="all")]
    for name, ids in subsets(a.root).items():
        sub = {k: v[v.index.isin(ids)] for k, v in results.items()}
        parts.append(report(sub, a.baseline).assign(subset=name))
    table = pd.concat(parts, ignore_index=True)
    table = table[table.market != "margin_rmse"]
    lines = ["Paired Brier difference vs the baseline (negative is better), ± 95% CI half-width; "
             "n = games in the subset.", ""]
    for market in ("moneyline", "spread", "total"):
        t = table[table.market == market]
        subs = list(dict.fromkeys(t.subset))
        lines += [f"### {market}", "", "| config | " + " | ".join(
            f"{s} (n={int(t[t.subset == s].n.iloc[0])})" for s in subs) + " |",
            "|---|" + "---|" * len(subs)]
        for cfg in dict.fromkeys(t.config):
            cells = []
            for s in subs:
                r = t[(t.subset == s) & (t.config == cfg)].iloc[0]
                cells.append(f"{r.brier:.4f}" if cfg == a.baseline else f"{r.d_brier:+.4f} ± {r.ci:.4f}")
            lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")
        table.to_csv(a.out.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
