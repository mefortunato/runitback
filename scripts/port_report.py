"""Does a sport's port reproduce sportsml's production Brier? (S4a, docs/multisport/PLAN.md)

Reads <root>/<arm>/<season>.csv for the arms `sportsml` (sportsml's code, production config),
`prod` (runitback.wager, the same config) and `prod_seed1` (runitback.wager, another seed: the
noise floor), all backtested on the same games table, and writes a markdown report:

- per market (moneyline, spread cover, over), each arm's Brier on the games every arm predicted
  and the market priced, and the paired per-game differences prod − sportsml and
  prod_seed1 − prod with 95% CIs;
- the verdict: the port reproduces sportsml when, on every market, the prod − sportsml CI
  includes 0 or the difference is no larger than the reseeding difference;
- moneyline Brier by season, and how many games had lines (a 403 from ESPN's odds API would
  show up here as missing lines);
- for context, sportsml's published production Brier where it has one (its
  experiments/<sport>/ablation, seasons 2024-2026, `market_eff`).

    python scripts/port_report.py experiments/nba/port --sport nba --out experiments/nba/port/report.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from foundation_report import report  # noqa: E402

ARMS = ("sportsml", "prod", "prod_seed1")
# sportsml experiments/<sport>/ablation/ablation.md, config market_eff (its production settings)
PUBLISHED = {"nba": ((2024, 2025, 2026), 0.2080, 3961), "wnba": ((2024, 2025, 2026), 0.2108, 922)}


def load(root: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for arm in ARMS:
        files = sorted((root / arm).glob("[0-9]*.csv"))
        if files:  # walk_forward writes one row per game (the home side's, or the listed one's)
            out[arm] = pd.concat([pd.read_csv(f, low_memory=False) for f in files]).set_index("game_id")
    return out


def paired(a: pd.Series, b: pd.Series, y: pd.Series) -> tuple[float, float]:
    d = ((a - y) ** 2 - (b - y) ** 2).to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--sport", required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    results = load(a.root)
    missing = [arm for arm in ARMS if arm not in results]
    if missing:
        raise SystemExit(f"missing arms under {a.root}: {missing}")
    seasons = sorted(results["prod"].season.unique())
    lines = [f"# {a.sport}: runitback.wager port vs sportsml", "",
             f"Walk-forward, seasons {seasons[0]}–{seasons[-1]}, the same games table for every arm.", ""]

    # per market, on the games every arm predicted and the market priced
    table = report(results, "sportsml").set_index(["config", "market"])
    noise = report({k: results[k] for k in ("prod", "prod_seed1")}, "prod").set_index(["config", "market"])
    lines += ["| market | n | sportsml | prod | prod_seed1 | prod − sportsml (95% CI) | "
              "prod_seed1 − prod (noise) | reproduces |", "|---|---|---|---|---|---|---|---|"]
    ok_all = True
    for market in ("moneyline", "spread", "total"):
        s, p, p1 = (table.loc[(arm, market)] for arm in ARMS)
        nz = noise.loc[("prod_seed1", market)]
        ok = bool(abs(p.d_brier) <= p.ci or abs(p.d_brier) <= abs(nz.d_brier))
        ok_all &= ok
        lines.append(f"| {market} | {int(p.n)} | {s.brier:.4f} | {p.brier:.4f} | {p1.brier:.4f} | "
                     f"{p.d_brier:+.4f} ± {p.ci:.4f} | {nz.d_brier:+.4f} ± {nz.ci:.4f} | "
                     f"{'yes' if ok else '**no**'} |")
    rmse = table.xs("margin_rmse", level="market").brier
    lines += ["", "Margin RMSE: " + ", ".join(f"{arm} {rmse[arm]:.2f}" for arm in ARMS), "",
              f"**Verdict: {'reproduces' if ok_all else 'does NOT reproduce'} sportsml within seed noise.**",
              ""]

    # moneyline by season, with line coverage
    common = results["sportsml"].index.intersection(results["prod"].index).intersection(
        results["prod_seed1"].index)
    sm, pr, p1 = (results[arm].loc[common] for arm in ARMS)
    gaps = {c: float((pr[c] - sm[c]).abs().max()) for c in ("win_prob", "cover_prob", "over_prob")
            if c in pr and c in sm}
    lines.insert(-2, "Largest per-game difference prod − sportsml (same seed, "
                 f"{len(common)} games): " + ", ".join(f"{c} {g:.2e}" for c, g in gaps.items()) + "")
    lines.insert(-2, "")
    decided = pr.result != 0
    y = (pr.result > 0).astype(float)
    lines += ["Moneyline by season (every played game the arms predicted; `lined`: games with a "
              "spread / a total)", "",
              "| season | games | lined | sportsml | prod | prod − sportsml | prod_seed1 − prod |",
              "|---|---|---|---|---|---|---|"]
    for season in seasons:
        m = (pr.season == season) & decided
        d, ci = paired(pr.win_prob[m], sm.win_prob[m], y[m])
        dn, cin = paired(p1.win_prob[m], pr.win_prob[m], y[m])
        lined = f"{int(pr.market_margin[m].notna().sum())} / {int(pr.total_line[m].notna().sum())}"
        lines.append(f"| {season} | {int(m.sum())} | {lined} | "
                     f"{((sm.win_prob[m] - y[m]) ** 2).mean():.4f} | {((pr.win_prob[m] - y[m]) ** 2).mean():.4f} | "
                     f"{d:+.4f} ± {ci:.4f} | {dn:+.4f} ± {cin:.4f} |")

    if a.sport in PUBLISHED:
        pub_seasons, pub_brier, pub_n = PUBLISHED[a.sport]
        m = pr.season.isin(pub_seasons) & pr.market_prob.notna() & decided
        if m.any():
            lines += ["", f"sportsml's published production Brier on {pub_seasons[0]}–{pub_seasons[-1]} "
                      f"(`market_eff`, {pub_n} games): **{pub_brier:.4f}**. This run on those seasons "
                      f"({int(m.sum())} games with a moneyline): sportsml "
                      f"{((sm.win_prob[m] - y[m]) ** 2).mean():.4f}, prod "
                      f"{((pr.win_prob[m] - y[m]) ** 2).mean():.4f}. The published figure is from an "
                      "earlier download (ESPN's files and lines have been refreshed since), so the "
                      "paired comparison above is the test."]
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
        table.reset_index().to_csv(a.out.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
