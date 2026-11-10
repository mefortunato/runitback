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

## Foundation folds (`pbp32`, F1)

**Short answer**
- **All six folds trained and passed the leak check** (2021–2025 for the backtest, 2026 for upcoming
  games). Each was validated on season S-1 from week 10 on (bowls included), and its team-season
  vocabulary has no season-S entry. Run 42816728580 (`cfb-foundation.yml`), 54–97 minutes per fold
  on a 4-core runner.
- **Training converged within the 10 epochs.** The best epoch was the 9th or 10th in every fold, and
  the last few epochs improved validation NLL by under 0.005.
- **Each fold's `pbp_32.parquet` holds all 33,318 team-games** (16,659 games, 2004 to 2026 so far: the
  games the engine replay keeps). Its 32 PCA dimensions, fit on games before S, explain 73.8–76.3% of
  the variance of the raw 384-dimension offense/defense pooled states.

| fold S | validated on | leak-free | best epoch | val NLL total (type / result / yards / elapsed) | PCA 32 explained | train time |
|---|---|---|---|---|---|---|
| 2021 | 2020 wk 10+ | yes | 9 / 10 | 4.825 (0.732 / 0.407 / 2.379 / 1.307) | 73.8% | 72 min |
| 2022 | 2021 wk 10+ | yes | 10 / 10 | 4.998 (0.752 / 0.401 / 2.377 / 1.468) | 74.8% | 79 min |
| 2023 | 2022 wk 10+ | yes | 9 / 10 | 4.899 (0.753 / 0.407 / 2.354 / 1.384) | 74.7% | 54 min |
| 2024 | 2023 wk 10+ | yes | 10 / 10 | 4.627 (0.774 / 0.411 / 2.369 / 1.073) | 75.3% | 92 min |
| 2025 | 2024 wk 10+ | yes | 10 / 10 | 5.882 (0.766 / 0.404 / 2.333 / 2.378) | 76.0% | 58 min |
| 2026 | 2025 wk 10+ | yes | 9 / 10 | 6.125 (0.737 / 0.409 / 2.294 / 2.685) | 76.3% | 97 min |

**The totals are not comparable across folds; the type, result and yards columns are.** The elapsed
term only counts plays with a per-play clock (`elapsed_known`; coarse-clock plays add zero), and
that share of the validation season varies: 33–47% of 2020–2023 games have a per-play clock,
against 79% in 2024 and 87% in 2025. So folds 2025 and 2026 carry an elapsed loss on most of their
validation plays. Type, result and yards are flat across folds (0.73–0.77, 0.40–0.41, 2.29–2.38
nats).

Sources: the per-fold "Fold summary" step of run 42816728580 (validation history and leak check
from each `model.pt`, explained variance from `pca_32.npz`). The folds are the run's
`cfb-fold<S>` artifacts (`model.pt`, `pca_32.npz`, `pbp_32.parquet`), and the processed data and
replay summary (`replay.json`) are `cfb-data`, all kept for 30 days.

**Setup**
- **Generator.** The NFL play transformer (`runitback.nfl.train`) trained on college plays from the
  college rules engine's preprocess (`runitback.cfb`, README "College football"). It is team-level
  (no college personnel data): team-season tokens plus pre-game EPA/play and Elo ratings. It uses
  the NFL fold's trunk (d_model 192, 4 layers) and 10 epochs. Settings: `runitback.cfb.foundation.TRAIN`.
- **Folds.** `runitback cfb foundation --season S` runs the sport-agnostic fold code
  (`runitback.core.foundation`, S3). The model is trained on 2004+ games before season S-1 week 10
  and validated on the rest of S-1, so it never sees S. Every played game is embedded (offense and
  defense states mean-pooled per team-game), and the PCA is fit on games before S.
- **Keys.** `team` is the ESPN team id. The CFB wager model keys teams by school name, with FCS
  schools pooled, so F3 joins through the schedule's `home_team_id` / `away_team_id`.

### Reproducing the folds

Start `CFB foundation` (`.github/workflows/cfb-foundation.yml`) from the Actions tab, or push a change
to that file on a `claude/**` branch. It downloads `espn_cfb_pbp` for 2004–2026, preprocesses it
(writing the engine replay summary to the job summary) and trains each fold in its own matrix job.
Locally: `runitback cfb download`, `runitback cfb preprocess`, `runitback cfb foundation --season S`.
