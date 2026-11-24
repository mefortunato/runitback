"""The (configuration, season) jobs of one stage of the EPL wager experiment, as a GitHub matrix.

    python scripts/epl_wager_matrix.py --stage screen
    python scripts/epl_wager_matrix.py --stage final --arms tf_mkt_e4,tf_eff_e4

Reads the stage's seasons and configurations from experiments/epl/wager.yaml and prints
`{"include": [{"config": ..., "season": ...}, ...]}`. The `final` stage has no fixed configurations: they are
`--arms` (the reference arm and the screen's picks). Every name must be a configuration of the file, or such a
name with `_e4` (the same configuration averaging 4 seeds); a mistake stops here, not in an hour-long job.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def matrix(experiment: dict, stage: str, arms: str = "") -> dict:
    """The matrix of ``stage``; ``arms`` (comma separated) are the ``final`` stage's configurations."""
    stages = experiment["stages"]
    if stage not in stages:
        raise ValueError(f"unknown stage {stage!r}; choose from {sorted(stages)}")
    names = [a.strip() for a in arms.split(",") if a.strip()] or list(stages[stage]["configs"])
    if not names:
        raise ValueError(f"stage {stage!r} runs the configurations you give it: pass --arms")
    known = set(experiment["configs"])
    unknown = [n for n in names if n not in known and not (n.endswith("_e4") and n[:-3] in known)]
    if unknown:
        raise ValueError(f"not configurations of the experiment: {unknown} (known: {sorted(known)})")
    if len(set(names)) != len(names):
        raise ValueError(f"a configuration is listed twice: {names}")
    return {"include": [{"config": n, "season": int(s)} for n in names for s in stages[stage]["seasons"]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True)
    parser.add_argument("--arms", default="", help="comma separated configurations (the final stage's)")
    parser.add_argument("--experiment", default="experiments/epl/wager.yaml")
    args = parser.parse_args()
    experiment = yaml.safe_load(Path(args.experiment).read_text())
    print(json.dumps(matrix(experiment, args.stage, args.arms), separators=(",", ":")))


if __name__ == "__main__":
    main()
