"""Average two configurations' per-season predictions (margins, totals and probabilities) into a
third: python scripts/average_predictions.py <out_dir> <config_dir_a> <config_dir_b>"""
import sys
from pathlib import Path

import pandas as pd

COLS = ("pred_margin", "win_prob", "cover_prob", "pred_total", "over_prob")
out, a_dir, b_dir = (Path(x) for x in sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
for f in sorted(a_dir.glob("*.csv")):
    if not (b_dir / f.name).exists():
        continue
    a = pd.read_csv(f, low_memory=False)
    b = pd.read_csv(b_dir / f.name, low_memory=False).drop_duplicates("game_id").set_index("game_id")
    # both files are one row per game from the same rows (team = the same side)
    b = b.reindex(a.game_id)
    assert (b.team.to_numpy() == a.team.to_numpy())[b.team.notna().to_numpy()].all(), f"sides differ in {f}"
    for c in COLS:
        a[c] = 0.5 * a[c].to_numpy() + 0.5 * b[c].to_numpy()
    a.dropna(subset=["win_prob"]).to_csv(out / f.name, index=False)
