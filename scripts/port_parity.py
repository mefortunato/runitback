"""Per-season agreement of two wager configurations' walk-forward predictions (a port vs its source).

Reads <dir>/<a>/<season>.csv and <dir>/<b>/<season>.csv (as `scripts/wager_arm.py` writes them),
matches games by id (home side) and reports per season the games both predicted, how many have
identical win probabilities, the largest absolute win-probability and margin differences, and each side's
moneyline Brier. Identical predictions mean the port reproduces its source exactly; otherwise the
paired Brier difference in the ablation table is the test.

    python scripts/port_parity.py experiments/cfb/baseline --a prod --b sportsml_prod
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df[df.location != -1].drop_duplicates("game_id").set_index("game_id")


def parity(root: Path, a: str, b: str) -> pd.DataFrame:
    rows = []
    for path in sorted((root / a).glob("[0-9]*.csv")):
        if not (root / b / path.name).exists():
            continue
        x, y = load(path), load(root / b / path.name)
        ids = x.index.intersection(y.index)
        x, y = x.loc[ids], y.loc[ids]
        won = (x.result > 0).astype(float)
        decided = x.result != 0
        rows.append({
            "season": int(path.stem),
            "games": len(ids),
            "identical": int((x.win_prob == y.win_prob).sum()),
            "max Δ win_prob": float((x.win_prob - y.win_prob).abs().max()),
            "max Δ margin": float((x.pred_margin - y.pred_margin).abs().max()),
            f"Brier {a}": float(((x.win_prob - won)[decided] ** 2).mean()),
            f"Brier {b}": float(((y.win_prob - won)[decided] ** 2).mean()),
        })
    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(table.columns) + " |", "|---" * len(table.columns) + "|"]
    for r in table.itertuples(index=False):
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    text = to_markdown(parity(args.root, args.a, args.b))
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
