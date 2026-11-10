# MLB: the box-score baseline ported from sportsml (S4c)

Does `runitback.wager.mlb`, the MLB game model ported from `mefortunato/sportsml`, reproduce
sportsml's production Brier in a walk-forward backtest? It is the baseline every MLB player-model
arm (M2–M5) will be compared against.

**Short answer**
- **The inputs are identical.** The ported downloader rebuilds the same games table from MLB's
  Stats API and ESPN: on 2024–2026, the same 7,386 games as sportsml's ablation, and the market's
  Brier on them is 0.241791 in both, the same to every printed digit.
- **The backtest:** pending the `mlb-baseline.yml` run (table below). A 1-epoch smoke run of the
  production configuration in a cloud container already scored 0.2437 against sportsml's 0.2438.

## Reproducing the results

Only the configuration (`baseline.yaml`) and this write-up are kept in git. The per-game
predictions, `ablation.md` / `ablation.json` and `report.md` are in the `mlb-baseline-results`
artifact of `.github/workflows/mlb-baseline.yml`. That workflow rebuilds the games table from the
public sources in the same run: five jobs each download a few seasons' Stats API and ESPN responses
(ESPN refuses a runner after a burst of odds requests), and one job merges their caches and builds
the table. It then backtests each (configuration, season) in its own job and scores them in a
report job. Start it from the Actions tab, or push a change to `baseline.yaml` or the workflow on a
`claude/**` branch.

Locally:

```bash
runitback-wager mlb download --output_file data/mlb/raw.csv --first_season 2008 --last_season 2026
runitback-wager mlb process --games data/mlb/raw.csv --output_file data/mlb/games.csv
runitback-wager mlb ablate --config experiments/mlb/baseline.yaml --games data/mlb/games.csv \
  --output_dir experiments/mlb/baseline
python scripts/mlb_baseline_report.py experiments/mlb/baseline.yaml --games data/mlb/games.csv
```

A full download takes about 10 minutes from a cloud container (about 35,000 ESPN odds requests).

## Setup

- **Data.** 2008–2026 games, team box scores (hitting, pitching, fielding) and the announced
  starting pitchers' records going into each game (shrunk toward a league-average starter, from
  starts before the game's date only), from MLB's Stats API. Moneylines, totals and run lines from
  ESPN, 2013 on.
- **Model.** sportsml's production MLB transformer, unchanged: 50-game history, past-line tokens
  and efficiency tokens (`runitback.wager.mlb.SPORT`; sportsml's `market_eff` ablation arm).
- **Walk-forward, 2024–2026.** These are the seasons of sportsml's MLB ablation
  (`experiments/mlb/ablation.yaml` there), so the numbers compare directly. Each season's model is
  fit on earlier seasons only.
- **Arms (`baseline.yaml`):** `prod`, and `prod_seed1` (the same with another seed) for the noise
  floor.
- **Bar.** `prod`'s moneyline Brier must be within seed noise of sportsml's 0.2438: the larger of
  the plan's ±0.002 floor and twice the gap between the two seeds.

## Results

| | games | Brier (moneyline) | vegas Brier | model − vegas |
|---|---|---|---|---|
| sportsml `market_eff` (recorded, 2026-11-04) | 7,386 | 0.2438 | 0.2418 | +0.0020 ± 0.0012 |
| runitback `prod`, 1-epoch smoke run (not the acceptance run) | 7,386 | 0.2437 | 0.2418 | +0.0019 ± 0.0010 |
| runitback `prod` | pending | | | |
| runitback `prod_seed1` | pending | | | |

## Next steps

- M1–M3 build the plate-appearance table, player values (M2: `player_values.md`) and projected
  lineups. M5 compares its
  arms against `prod` from this backtest, rerun alongside them in the same workflow run (4-seed
  ensembles for promotion decisions). M5 also moves the evaluation to the plan's last five
  complete seasons.
