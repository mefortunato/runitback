"""Score the S4c MLB baseline backtest and compare it with sportsml's production numbers.

Reads <dir>/<config>/<season>.csv for every configuration in the experiment's YAML (written by
`.github/workflows/mlb-baseline.yml`), scores them with the ablation table (paired per-game Brier
against the baseline, on the games every configuration predicted and the market priced; this also
writes ablation.md / ablation.json), and writes a report comparing the baseline with sportsml's
recorded production MLB result on the same seasons (sportsml experiments/mlb/ablation: the
``market_eff`` configuration, which is sportsml's production MLB transformer).

The port reproduces sportsml when the baseline's Brier is within seed noise of sportsml's: the
larger of the plan's ±0.002 noise floor and twice the measured difference between the two seeds.

    python scripts/mlb_baseline_report.py experiments/mlb/baseline.yaml --games data/mlb/games.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from runitback.wager.experiments import ablate

# sportsml experiments/mlb/ablation/ablation.json (commit 74cf42f, 2026-11-04), seasons 2024-2026
SPORTSML = {
    "config": "market_eff",
    "seasons": [2024, 2025, 2026],
    "n": 7386,
    "brier": 0.24381714733817542,
    "vegas_brier": 0.24179051695275972,
    "vs_market": 0.002026630385415704,
}
NOISE_FLOOR = 0.002


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", help="experiments/mlb/baseline.yaml")
    parser.add_argument("--games", default="data/mlb/games.csv")
    parser.add_argument("--out", default=None, help="report path (default: <dir>/report.md)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(args.config).with_suffix("")
    seasons = [int(s) for s in cfg["seasons"]]
    if seasons != SPORTSML["seasons"]:
        raise SystemExit(f"sportsml's numbers are for {SPORTSML['seasons']}, not {seasons}")
    complete = {
        n: c
        for n, c in cfg["configs"].items()
        if all((out_dir / n / f"{s}.csv").exists() for s in seasons)
    }
    baseline = cfg["baseline"]
    if baseline not in complete:
        raise SystemExit(f"the baseline {baseline!r} is missing seasons in {out_dir}")
    # every cell is cached: this only scores
    result = ablate(args.games, "mlb", complete, seasons, str(out_dir), baseline, 1, 1)
    table = result["table"]

    base = table.loc[baseline]
    seed_rows = [n for n in complete if n != baseline]
    seed_gap = max((abs(table.loc[n, "brier"] - base["brier"]) for n in seed_rows), default=0.0)
    tolerance = max(NOISE_FLOOR, 2 * seed_gap)
    delta = base["brier"] - SPORTSML["brier"]
    passed = abs(delta) <= tolerance
    vegas = table.loc["vegas", "brier"] if "vegas" in table.index else float("nan")

    lines = [
        f"# MLB baseline (S4c): runitback.wager vs sportsml, seasons {seasons}",
        "",
        "| | games | Brier (moneyline) | vegas Brier | model − vegas |",
        "|---|---|---|---|---|",
        f"| sportsml `{SPORTSML['config']}` (recorded) | {SPORTSML['n']} | {SPORTSML['brier']:.4f} "
        f"| {SPORTSML['vegas_brier']:.4f} | {SPORTSML['vs_market']:+.4f} |",
    ]
    for name in complete:
        row = table.loc[name]
        lines.append(
            f"| runitback `{name}` | {int(row['n'])} | {row['brier']:.4f} | {vegas:.4f} "
            f"| {row['vs_market']:+.4f} ± {row['vs_market_ci95']:.4f} |"
        )
    lines += [
        "",
        f"- Δ Brier, runitback `{baseline}` − sportsml: **{delta:+.4f}**",
        f"- seed noise: |`{baseline}` − other seeds| = {seed_gap:.4f}; tolerance "
        f"max({NOISE_FLOOR}, 2 × that) = {tolerance:.4f}",
        f"- **{'PASS' if passed else 'FAIL'}**: the port "
        f"{'reproduces' if passed else 'does not reproduce'} sportsml's production Brier within seed noise",
        f"- same games as sportsml: {'yes' if int(base['n']) == SPORTSML['n'] else 'no'} "
        f"({int(base['n'])} vs {SPORTSML['n']}); vegas Brier {vegas:.4f} vs {SPORTSML['vegas_brier']:.4f}",
        "",
        "## Ablation table",
        "",
        (out_dir / "ablation.md").read_text(),
    ]
    out = Path(args.out) if args.out else out_dir / "report.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
