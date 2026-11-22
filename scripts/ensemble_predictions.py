"""Average several configurations' per-season predictions (margins, totals and probabilities) into one:

    python scripts/ensemble_predictions.py <out_dir> <config_dir> <config_dir> [<config_dir> ...]

The college wager experiments (experiments/cbb, experiments/wcbb) fit each seed of a 4-seed ensemble in
its own Actions job (a men's college fit takes 1-2 hours; four in one job would pass the 6-hour limit)
and average the seeds here (seeds 0-3, the members ``fit: {ensemble: 4}`` would train). One difference
from the NBA's ``_e4``: that ensemble averages the members' margins and totals and calibrates one sigma
on the validation seasons; here the members' predicted margins, totals and probabilities are each
averaged. The same for every arm, so ``<arm>_e4`` still compares with ``<arm>_e4``. Only seasons present
in every directory are written; every file must be the same games from the same side.
"""
import sys
from pathlib import Path

import pandas as pd

COLS = ("pred_margin", "win_prob", "cover_prob", "pred_total", "over_prob")


def average(out: Path, dirs: list[Path]) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    seasons = sorted(set.intersection(*(set(p.name for p in d.glob("[0-9]*.csv")) for d in dirs)))
    for name in seasons:
        first = pd.read_csv(dirs[0] / name, low_memory=False)
        cols = [c for c in COLS if c in first]
        acc = first[cols].astype(float).copy()
        for d in dirs[1:]:
            b = pd.read_csv(d / name, low_memory=False).drop_duplicates("game_id").set_index("game_id")
            b = b.reindex(first.game_id)
            ok = b.team.notna().to_numpy()
            if not (b.team.to_numpy() == first.team.to_numpy())[ok].all():
                raise SystemExit(f"sides differ in {d / name}")
            acc += b[cols].astype(float).to_numpy()      # NaN where a member lacks the game
        first[cols] = acc / len(dirs)
        first.dropna(subset=["win_prob"]).to_csv(out / name, index=False)
    return seasons


if __name__ == "__main__":
    done = average(Path(sys.argv[1]), [Path(x) for x in sys.argv[2:]])
    print(f"{sys.argv[1]}: {len(sys.argv) - 2} members, seasons {[s[:-4] for s in done]}")
