"""Score one stage of the EPL wager experiment (experiments/epl/wager.yaml) from its per-season results.

    python scripts/epl_report.py screen    runs/wager --out report.md
    python scripts/epl_report.py baselines runs/wager --out report.md
    python scripts/epl_report.py final     runs/wager --out report.md --reference elo --games data/epl/games.csv

`runs/wager/<config>/<season>.csv` are the walk-forward results `scripts/wager_arm.py` writes. Everything is
scored by the ranked probability score (RPS, lower is better) on the games every configuration predicted,
as paired per-game differences with 95% CIs, against the closing-average market (`vegas`) and the Pinnacle
close (`book`).

- `screen` (2017-2021, one seed): ranks the arms, chooses the **reference arm** (the best of `elo`, `sklearn`,
  `gnn` and the default transformer `tf`) and the two transformer arms worth a 4-seed ensemble, and prints the
  `arms` to give the workflow's `final` stage. Nothing here looks at 2022-2026.
- `baselines` (2022-2026): the baselines on the test seasons, and the noise floor: how far apart four seeds of
  the same transformer land, which any 1-seed difference has to beat.
- `final` (2022-2026, 4-seed ensembles): each arm against the reference, with a promotion verdict (the CI of its
  RPS difference includes 0 or favors the arm), by season, and on a promoted club's first games.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from epl_promoted_rps import early_promoted

from runitback.wager.core.metrics import (
    paired_difference,
    three_way_scores,
)
from runitback.wager.core.sport import get_sport
from runitback.wager.experiments import (
    ablation_table,
    common_games,
    format_seasons,
    format_table,
    load_results,
    season_table,
)

SPORT = "epl"
REFERENCE_CANDIDATES = ("elo", "sklearn", "gnn", "tf")  # the non-market arms the reference is chosen from
SEED_ARM = re.compile(r"tf_s\d+")  # another seed of `tf`: noise, not an arm
STAGE_SEASONS = {"screen": [2017, 2018, 2019, 2020, 2021]}


def names_in(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and any(p.glob("[0-9]*.csv")))


def complete(root: Path, seasons: list[int]) -> list[str]:
    """Configurations with a result file for every season (a failed cell leaves the arm out, loudly)."""
    have = [n for n in names_in(root) if all((root / n / f"{s}.csv").exists() for s in seasons)]
    partial = sorted(set(names_in(root)) - set(have))
    if partial:
        print(f"left out, a season is missing: {partial}", file=sys.stderr)
    return have


def choose(table: pd.DataFrame) -> dict:
    """The reference arm and the transformer arms to run as 4-seed ensembles, from the screen's table."""
    candidates = [c for c in REFERENCE_CANDIDATES if c in table.index]
    reference = min(candidates, key=lambda c: table.loc[c, "rps"])
    transformers = [c for c in table.index if c.startswith("tf") and not SEED_ARM.fullmatch(c) and c != reference]
    top = sorted(transformers, key=lambda c: table.loc[c, "rps"])[:2]
    arms = [reference + ("_e4" if reference == "tf" else "")] + [f"{c}_e4" for c in top]
    return {"reference": reference, "top": top, "arms": ",".join(arms)}


def screen_report(results: dict[str, pd.DataFrame]) -> tuple[str, dict]:
    baseline = "tf" if "tf" in results else next(iter(results))
    table = ablation_table(results, SPORT, baseline).sort_values("rps")
    pick = choose(table)
    seasons = sorted({int(s) for f in results.values() for s in f["season"]})
    text = (
        f"## Screen: 2017-2021, one seed ({len(results)} arms, seasons {seasons})\n\n"
        f"Ranked probability score on the games every arm predicted and the market priced; Δ RPS is the paired "
        f"difference to `{baseline}` (negative is better), `vs market` to the closing average.\n\n"
        + format_table(table)
        + f"\n\nΔ RPS to `{baseline}` by season (2021 is the season without crowds)\n\n"
        + format_seasons(season_table(results, SPORT, baseline))
        + f"\n\n**Reference arm: `{pick['reference']}`** (the best of {', '.join(REFERENCE_CANDIDATES)} on these seasons).\n"
        f"**Transformer arms to run as 4-seed ensembles: `{'`, `'.join(pick['top'])}`.**\n\n"
        f"Final stage `arms` input: `{pick['arms']}`\n"
    )
    return text, pick


def noise_floor(results: dict[str, pd.DataFrame]) -> str:
    """How far apart seeds of the same transformer (``tf``, ``tf_s1``, ...) land on the same games."""
    seeds = [n for n in results if n == "tf" or SEED_ARM.fullmatch(n)]
    if len(seeds) < 2:
        return "Noise floor: needs two or more seeds of `tf`.\n"
    base, aligned = common_games({n: results[n] for n in seeds}, get_sport(SPORT).schema)
    y = base[get_sport(SPORT).schema.target].to_numpy(dtype=float)
    rps = {n: three_way_scores(y, aligned[n]["win_prob"], aligned[n]["draw_prob"])["rps"] for n in seeds}
    means = pd.Series({n: v.mean() for n, v in rps.items()})
    pairs = []
    for i, a in enumerate(seeds):
        for b in seeds[i + 1 :]:
            mean, half = paired_difference(rps[a] - rps[b])
            pairs.append((a, b, mean, half))
    worst = max(abs(m) for _, _, m, _ in pairs)
    lines = [f"| {a} vs {b} | {m:+.5f} ± {h:.5f} |" for a, b, m, h in pairs]
    return (
        f"### Noise floor: {len(seeds)} seeds of `tf` on {len(y)} games\n\n"
        f"Mean RPS by seed: {', '.join(f'{n} {v:.5f}' for n, v in means.items())}; "
        f"standard deviation across seeds **{means.std(ddof=1):.5f}**; largest gap between two seeds "
        f"**{worst:.5f}**. A 1-seed difference smaller than that is not an effect.\n\n"
        "| Seeds | Paired Δ RPS (95% CI) |\n|---|---|\n" + "\n".join(lines) + "\n"
    )


def baselines_report(results: dict[str, pd.DataFrame]) -> str:
    table = ablation_table(results, SPORT, "tf").sort_values("rps")
    seasons = sorted({int(s) for f in results.values() for s in f["season"]})
    return (
        f"## Baselines on the test seasons {seasons}, one seed\n\n"
        "Δ RPS is the paired difference to `tf` (negative is better), `vs market` to the closing average.\n\n"
        + format_table(table)
        + "\n\nΔ RPS to `tf` by season\n\n"
        + format_seasons(season_table(results, SPORT, "tf"))
        + "\n\n"
        + noise_floor(results)
    )


def verdict(table: pd.DataFrame, reference: str) -> str:
    rows = ["| Arm | Δ RPS vs reference | Verdict |", "|---|---|---|"]
    for name, r in table.iterrows():
        if name == reference or pd.isna(r.get("d_rps")):
            continue
        low, high = r["d_rps"] - r["d_rps_ci95"], r["d_rps"] + r["d_rps_ci95"]
        if high < 0:
            word = "better: the whole CI is below 0"
        elif low <= 0:
            word = "passes: the CI includes 0"
        else:
            word = "worse: the whole CI is above 0"
        rows.append(f"| `{name}` | {r['d_rps']:+.5f} ± {r['d_rps_ci95']:.5f} | {word} |")
    return "\n".join(rows)


def final_report(results: dict[str, pd.DataFrame], reference: str, games: pd.DataFrame | None) -> str:
    table = ablation_table(results, SPORT, reference).sort_values("rps")
    seasons = sorted({int(s) for f in results.values() for s in f["season"]})
    text = (
        f"## Final: 4-seed ensembles on {seasons}, against `{reference}`\n\n"
        "Paired per-game RPS differences with 95% CIs. A promotion needs the CI to include 0 or favor the arm.\n\n"
        + format_table(table)
        + "\n\n"
        + verdict(table, reference)
        + f"\n\nΔ RPS to `{reference}` by season\n\n"
        + format_seasons(season_table(results, SPORT, reference))
        + "\n"
    )
    if games is not None:
        ids = early_promoted(games, 5)
        subset = {n: f[f["game_id"].isin(ids)] for n, f in results.items()}
        early = ablation_table(subset, SPORT, reference)
        text += (
            f"\n### A promoted club's first 5 games ({int(early['n'].max())} games every arm predicted)\n\n"
            + format_table(early)
            + "\n"
        )
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["screen", "baselines", "final"])
    parser.add_argument("root", type=Path, help="the directory of <config>/<season>.csv")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--reference", default="elo", help="the reference arm (final)")
    parser.add_argument("--games", type=Path, help="the processed games table (final: promoted clubs' games)")
    parser.add_argument("--seasons", type=int, nargs="+")
    args = parser.parse_args()
    seasons = args.seasons or STAGE_SEASONS.get(args.stage, [2022, 2023, 2024, 2025, 2026])
    names = complete(args.root, seasons)
    if not names:
        raise SystemExit(f"no complete configurations in {args.root} for seasons {seasons}")
    results = load_results(str(args.root), names, seasons)
    if args.stage == "screen":
        text, _ = screen_report(results)
    elif args.stage == "baselines":
        text = baselines_report(results)
    else:
        games = pd.read_csv(args.games) if args.games else None
        if args.reference not in results:
            raise SystemExit(f"the reference arm {args.reference!r} has no results in {args.root}")
        text = final_report(results, args.reference, games)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
