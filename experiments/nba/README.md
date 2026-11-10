# NBA: the event generator as a foundation for the wager model

Work toward the NFL's player-based wager model for the NBA (`docs/multisport/PLAN.md`, NBA
packages N1–N6). This write-up follows `experiments/nfl/README.md`.

## Foundation folds (`pbp32`, N4)

**Short answer:** pending the first run of `.github/workflows/nba-foundation.yml`; the fold
table goes here (from the run's `nba-foundation-results` artifact, `report.md`). Whether `pbp32`
helps the wager model is N5's question, not this one.

**Setup**
- **One generator per test season ("fold").** Fold S is the existing `runitback.nba` event
  transformer (`runitback nba train`'s production recipe: 6 epochs, d_model 192, 4 layers, pre-game
  ratings), trained on 2002+ games before the last 40% of season S-1, validated on that 40%.
  It never sees season S. Seasons are named by the year they end in (2026 = 2025-26).
  Settings: `runitback.nba.foundation.TRAIN`.
- **Folds:** 2022–2026 (the backtest's last 5 complete seasons, as `experiments/nba/port.yaml`)
  plus 2027, the current season, for upcoming games.
- **Game embeddings, pooled per possession** (`runitback.nba.embed`). Each game is run teacher-forced
  through the trunk. A possession is a run of consecutive events in one period with the same team
  on offense; its state is the mean over its events, so a free-throw trip or a timeout doesn't
  count for more than a one-shot possession. A team-game's raw embedding is the mean possession
  state over its offensive possessions next to the mean over its defensive ones (2 × 192).
- **PCA to 32 dimensions** fit on the games before S only (`runitback.core.foundation`, shared with
  the NFL). Later games are projected with the stored projection, so the table a wager model was fit
  on never changes. Output per fold: `model.pt`, `pca_32.npz`, `pbp_32.parquet` (`game_id`, `team`,
  `season`, `_pbp_0..31`; `game_id` is ESPN's as a string, `team` the current franchise
  abbreviation, which is how the wager model's NBA games table keys teams too; N5 checks the join).
- **Checks** (`scripts/nba_foundation_report.py`, run by the workflow; it fails if a fold saw its
  test season): validation season and the checkpoint's team-season vocabulary per fold, validation
  loss, explained variance, and a team-strength sanity check: a ridge (fit before S) from each
  team's previous-10-game mean embedding to the home margin in S, against the same from pre-game Elo.
- **Not done (optional in the plan):** lineup inputs from N1's on-court fives. The NFL found that
  the generator's player embeddings didn't rank players (personnel came through the player values,
  layers B → C), so the NBA's personnel signal is left to N2/N3.

## Reproducing the results

`NBA foundation` (`.github/workflows/nba-foundation.yml`; run it from the Actions tab, or push a
change to the file on a `claude/**` branch) calls `nba-experiment-inputs.yml`, which downloads the
ESPN play-by-play (sportsdataverse) for 2002–2027, preprocesses it and trains each fold in its own
job. Artifacts: `nba-data`, `nba-fold<S>`, `nba-foundation-results` (the report) and
`nba-foundation-models` (all folds). Experiments that use the folds (N5) call
`nba-experiment-inputs.yml` themselves rather than reading this run's artifacts.

Locally: `runitback nba download --start 2002 --end 2027 --no-players`, `runitback nba preprocess
--start 2002 --end 2027`, then `runitback nba foundation --root foundation/nba` (or `--season S` for
one fold), and `python scripts/nba_foundation_report.py foundation/nba`.
