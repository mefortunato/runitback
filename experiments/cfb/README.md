# College football: the box-score baseline ported from sportsml

Does `runitback.wager.cfb`, the college football model ported from `mefortunato/sportsml`
(`src/sportsml/cfb/`), reproduce sportsml's production backtest? This box-score transformer is
the baseline every later CFB arm (`pbp32`, `pbp32_qb`; work package F3) is compared against.

**Short answer**
- _Pending the first run of `cfb-baseline.yml`; filled in from its `cfb-baseline-results`
  artifact._
- **Locally the port is exact.** The ported downloader and `process` produce an identical games
  table to sportsml's (7,224 rows, 2021–2024), and a 2-epoch backtest of 2024 with the same seed
  gives identical predictions for all 918 games in both packages.

## Reproducing the results

Only the configuration (`baseline.yaml`) and this write-up are kept in git. The workflow
`.github/workflows/cfb-baseline.yml` rebuilds everything from the public sources in the run
(sportsdataverse schedules and team box scores, ESPN's odds API) and uploads the per-game
predictions and reports as the `cfb-baseline-results` artifact:

| file | what |
|---|---|
| `ablation.md` | every cell on 2021–2025, paired moneyline Brier vs `sportsml_prod` |
| `last3/ablation.md` | the same on 2023–2025, the seasons of sportsml's published table |
| `report.md` | moneyline, spread and total Brier per configuration (`scripts/foundation_report.py`) |
| `parity.md` | per season, how many games `prod` and `sportsml_prod` predict identically (`scripts/port_parity.py`) |
| `<config>/<season>.csv` | per-game walk-forward predictions |

Start it from the Actions tab ("Run workflow"), or push a change to `baseline.yaml` on a
`claude/**` branch. ESPN refuses a runner after a burst of odds requests, so each season's lines
are fetched by its own job; if one is refused, "Re-run failed jobs".

To run one cell locally: `runitback-wager cfb download --last_season 2025`, `runitback-wager cfb
process`, then `python scripts/wager_arm.py prod 2024 --sport cfb --experiment
experiments/cfb/baseline.yaml` (add `--package sportsml` with sportsml installed for its side).

## Setup

- **Walk-forward evaluation, 2021–2025** (the plan's last 5 complete seasons). For each test
  season S the model is fit on earlier seasons only and S is replayed date by date.
- **Games.** FBS games (an FCS opponent is pooled into one team, `FCS`), box scores from 2004 and
  lines from 2013, as in sportsml.
- **One change from sportsml.** Kickoff times are parsed as ISO 8601 explicitly. sportsdataverse's
  2020 schedule mixes `…T04:00Z` and `…T23:30:00.000Z`, and inferring the first row's format
  turned 1,125 of its 1,247 kickoffs into NaT, which crashes the bowl-week numbering (sportsml has
  the same bug). The 2021–2024 games table is unchanged by the fix. Both packages read the same
  raw table, so the comparison below is unaffected.
- **Model.** sportsml's production CFB transformer, pinned in `baseline.yaml`: 28-game history,
  total weight 1, opponent strength, past-line tokens and efficiency rates.
- **Arms.**
  - `prod`, `prod_seed1`: `runitback.wager`, seeds 0 and 1.
  - `sportsml_prod`, `sportsml_prod_seed1`: the same configurations on sportsml's own code (commit
    `b1d1632`), processing the same raw games table itself.
- **Bar.** The port reproduces sportsml when `prod` vs `sportsml_prod` is within the seed noise
  (`sportsml_prod_seed1` vs `sportsml_prod`, about ±0.002 Brier) and `prod`'s 2023–2025 moneyline
  Brier is within that noise of sportsml's published 0.1672 (`market_eff` in
  sportsml's `experiments/cfb/ablation/ablation.md`; Vegas 0.1590 on the same 2,704 games).

## Results

_Pending._

## Next steps

1. F3 (`experiments/cfb/`, arms `prod`, `pbp32`, `pbp32_qb`) uses `prod` from here as its
   baseline.
2. A1 adds college football to the predictions site (`web/src/App.tsx`, the nav and the server).
