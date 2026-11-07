# NFL: the play-by-play generator as a foundation for the wager model

Can the runitback play-by-play transformer serve as a foundation model under the
spread / moneyline / total transformer (ported from sportsml), without making the wager predictions
worse, and do the two models agree with each other?

**Short answer**
- **Frozen embeddings work.** Feeding the frozen generator's game embeddings into the wager model's
  history tokens passes the non-inferiority bar on all three markets, at every embedding size.
  The 32-dimension version (`pbp32`) is slightly better than production on every market, but the
  gain is within the noise floor.
- **Fine-tuning the generator on the wager loss makes things worse.** Both fine-tuned arms fail
  the bar on moneyline and totals, whether or not the next-play loss is kept.
- **Win and margin agree; totals don't.** The generator and the wager head agree on win
  probability and margin (correlation 0.85). On totals they don't, because the generator
  under-scores by about 8 points a game.

Results: `foundation/ablation.md`, `foundation/report.md` (per market, with the pass/fail column),
`foundation/consistency.csv`, the per-game predictions in `foundation/<config>/<season>.csv` and
the generator rollouts in `foundation/generator/<season>.csv`.

## Setup

- **Walk-forward evaluation, 2021–2025** (5 folds, about 1,400 games). For each test season S, the
  wager model and the play-by-play foundation model are fit only on earlier data.
- **Foundation, `runs/foundation/fold<S>`.** The full v0.2.0 configuration with the "who" head
  (lineups, coaches, personnel): 10 epochs, validated on season S-1 weeks 15+. It never sees
  season S.
- **Game embeddings, `runitback nfl embed`.** Teacher-forced trunk states are mean-pooled over each
  team's offensive plays and, separately, its defensive plays. PCA to 16/32/64 dimensions is fit
  only on games before S.
- **Coupling.** The embeddings join the wager model's per-team-game history tokens (columns
  `_pbp_*` plus a missing flag). Nothing else changes.
- **Arms (`foundation.yaml`):**
  - `prod`: sportsml's production NFL transformer, the baseline.
  - `pbp16`, `pbp32`, `pbp64`: box score plus embedding.
  - `pbp32_only`: the embedding replaces the box-score features.
  - `pbp32_ft`: the frozen fit, then the top trunk block, `ln_f` and the wager model train jointly
    on the wager loss, keeping the next-play loss as an auxiliary.
  - `pbp32_ft_noaux`: the same without the auxiliary loss.
  - `prod_seed1`: the baseline with another seed, to measure the noise floor.
- **Non-inferiority bar.** The paired per-game Brier difference against `prod` must have a 95% CI
  that includes 0 or favors the arm, on each of cover, moneyline and over.
- **Where it ran.** Everything except `prod` and `prod_seed1` ran on GitHub Actions
  (`.github/workflows/foundation-experiment.yml`, run 42286987708) as one job per fold, per
  (arm, season) and per generator season. It took about 2¾ hours of walltime. `prod` and
  `prod_seed1` were run earlier in a cloud container on the same seasons. The 2021 fold's
  embeddings came out identical in both environments (14,788 team-games, 79.2% of the variance in
  64 dimensions).

## Wager performance

Paired Brier difference against `prod` (negative is better), ± the 95% CI half-width. The noise
floor from a reseeded baseline is about ±0.002.

| arm | moneyline | spread (cover) | total (over) | passes | margin RMSE |
|---|---|---|---|---|---|
| prod | 0.2191 | 0.2536 | 0.2587 | — | 13.02 |
| **pbp32** | **−0.0012 ± 0.0024** | **−0.0025 ± 0.0028** | **−0.0007 ± 0.0025** | yes | **12.90** |
| pbp16 | −0.0003 ± 0.0024 | −0.0006 ± 0.0027 | −0.0003 ± 0.0024 | yes | 12.93 |
| pbp64 | −0.0000 ± 0.0024 | −0.0010 ± 0.0027 | −0.0024 ± 0.0026 | yes | 12.97 |
| pbp32_only | +0.0003 ± 0.0035 | +0.0015 ± 0.0040 | +0.0016 ± 0.0042 | yes | 12.94 |
| prod_seed1 (noise) | +0.0018 ± 0.0023 | +0.0010 ± 0.0026 | −0.0010 ± 0.0021 | yes | 13.06 |
| pbp32_ft | +0.0057 ± 0.0039 | +0.0042 ± 0.0048 | +0.0361 ± 0.0107 | **no** | 13.26 |
| pbp32_ft_noaux | +0.0048 ± 0.0035 | +0.0031 ± 0.0044 | +0.0258 ± 0.0091 | **no** | 13.21 |

For context, the Vegas moneyline Brier on the same games is 0.2115. Every model is still about
0.007–0.008 behind the market.

- **pbp32 is the best arm, but not significantly.** It has the lowest Brier in every market and
  the lowest margin RMSE. Its improvement is about half the seed-to-seed noise, though, so this is
  "at least as good", not "better". Its gains are concentrated in 2024 and 2025 (−0.0037 each on
  moneyline), the seasons where the generator had the most play-by-play history.
- **pbp32_only is as good as the box score.** Replacing the box-score features with the embedding
  costs nothing measurable. The generator's embedding carries roughly the information the box
  score does.
- **Fine-tuning fails, mainly on totals.** The fine-tuned arms shift predicted totals up by 4–5
  points (mean 51.0 and 49.9 against 45.5 for pbp32 and a mean line of 44.6). Their over
  probability averages 0.67 when the actual over rate is 0.49. Moneyline is also worse. Keeping
  the next-play loss doesn't help (`pbp32_ft` is no better than `pbp32_ft_noaux`).
- **Why fine-tuning fails (likely).** The fine-tune stops on validation loss, so the shift happens
  out of sample. Once the embeddings are trainable, they drift away from what the wager model was
  fit on, and about 1,400 training games per fold are far too few to retrain a trunk block.
  Fine-tuning isn't worth pursuing at this data size.

## Generator vs wager consistency

For each game, `scripts/foundation_consistency.py` runs 100 full rollouts of the foundation model
that never saw that season. The rollouts use previous-game lineups, both opening kickers, and
pre-game ratings scaled by a per-fold strength scale fit on the previous season. It then compares
the share of rollouts won, covered and over against the wager head's probabilities.

| | moneyline | spread (cover) | total (over) |
|---|---|---|---|
| generator Brier (prod wager: 0.2191 / 0.2536 / 0.2587) | 0.2276 | 0.2655 | 0.3151 |
| corr(generator, prod) | 0.83 | 0.47 | 0.44 |
| corr(generator, pbp32) | 0.85 | 0.51 | 0.47 |
| mean abs gap vs pbp32 (probability) | 0.083 | 0.092 | 0.273 |
| calibration slope: generator / pbp32 | 0.70 / 1.03 | 0.01 / 0.42 | 0.09 / 0.05 |

Points: margin correlation 0.85 with a mean gap of 2.9 points. Totals correlation 0.51 with a
mean gap of 8.6 points.

- **Win and margin are consistent.** The generator and the wager head agree on who wins and by
  how much (correlation 0.85). The generator's mean margin (2.0) is close to the actual 2.25.
  Coupling also raises agreement: pbp32's predictions track the generator more closely than
  prod's in every market.
- **The generator is overconfident.** Its win probabilities have a calibration slope of 0.70, so
  it is too sure of itself, and its Brier is about 0.009 worse than the wager head's.
- **The generator under-scores.** It averages 37.0 total points against 45.2 actual (a line of
  44.6), short every season by 4–11 points. So it calls the under about 75% of the time, and on
  totals it is the generator, not the wager head, that is off.
- **Its margins are too narrow.** The spread of margins across games is too small (6.7 points,
  against 14.2 for actual results), consistent with the strength scale hitting its limits (1.0–2.0).
  This isn't a coupling problem; it is a calibration target for the generator itself.

## Recommendation and next steps

1. **Adopt `pbp32` (frozen) for production.** It is at least as good as `prod` on every market,
   slightly better on margin, and more consistent with the generator. Done: `runitback nfl
   foundation` keeps one fold per test season, and the predictions workflows set
   `RUNITBACK_NFL_FOUNDATION` (README, "Predictions data").
2. **Drop generator fine-tuning** at this data size.
3. **Fix the generator's scoring bias** before showing its totals next to the wager model's on the
   game page. A good first step is checking the rollout setup (clock and elapsed sampling,
   end-of-half behavior) for the 8-point shortfall. Then either recalibrate the strength scale
   jointly on margin and total, or post-hoc scale points.
4. **Rerun** with `git push` after editing `foundation.yaml`, or dispatch "Foundation experiment"
   once it is on `main`. The models are kept as the run's `foundation-models` artifact for 90 days.

## Personnel: does knowing who plays this week help?

The wager model's history (box scores plus the frozen `pbp32` game embeddings) describes the
teams as they played; the only signal about *this* week's players was the injury features
(`inj_<unit>`, expected missing snap share per unit). `experiments/nfl/lineups.yaml` adds the game
being predicted's **projected lineup**, built only from what is known before kickoff
(`runitback nfl lineup-features`): the team's published depth chart for the week (else its
previous game's), minus players listed Out/Doubtful or off the active roster, with the next
player in each position group moved up. The projected QB1 is right in 95% of 2025 games (88% in
2024), and in 70% (41%) of games where the starter changed, against 18% for the previous game's
lineup alone. Each fold's foundation model turns the lineup into features for both teams:

| arm | adds to `pbp32` (production) |
|---|---|
| `pbp32_lp` | the projected players' embeddings, pooled by the foundation's lineup encoder (as offense and defense) |
| `pbp32_lp_lu` | + how that pooled lineup differs from the lineups behind the history games |
| `pbp32_lp_cf` | + the history re-embedded as if the projected lineup had played it, minus as played |
| `pbp32_noinj` | production without the injury features |
| `pbp32_lp_noinj` | the projected lineup in place of the injury features |

**Short answer: no arm beats production, and none should be adopted yet.**
- **`pbp32_lp` (the pooled projected lineup) passes on all three markets and is a wash**:
  Δ Brier moneyline −0.0009 ± 0.0023, spread +0.0014 ± 0.0027, total +0.0009 ± 0.0027.
- **The history-relative arms hurt totals.** `lp_lu` (+0.0030 ± 0.0023) and `lp_cf`
  (+0.0036 ± 0.0025) fail the bar on totals, worst on high-turnover games (+0.008). The extra
  16–64 context columns look like noise the head overfits on 1,400 test games a fold.
- **On the games where personnel moved, the lineup leans the right way but within noise.** Where
  the starting QB changed (352 games), `pbp32_lp` is −0.0018 ± 0.0045 (moneyline),
  −0.0028 ± 0.0054 (spread), −0.0031 ± 0.0053 (total); the other arms are worse there than
  production (`subsets.md`). Roughly 4× the test games would be needed to tell.
- **The injury features don't carry much either.** Dropping them (`pbp32_noinj`) costs
  +0.0005 / +0.0025 / +0.0007, all within noise, and the projected lineup roughly replaces them
  (`pbp32_lp_noinj`). Personnel news is largely in the closing line already, which every arm still
  trails (+0.006–0.008 Brier vs the market).

**QB-swap check** (are the player embeddings informative at all?): rolling out the 2025 fold's
generator with named QBs in every home team's QB1 slot. Pending: `.github/workflows/qb-swap.yml`,
results in `lineups/qb_swap.*.csv`.

Results: `lineups/ablation.md`, `lineups/report.md` (per market, pass/fail), `lineups/subsets.md`
(QB changed / QB projected to change / top-quartile turnover / steady), and the per-game change
flags in `lineups/changes/<season>.csv`.

**Next steps for personnel**
1. Keep `pbp32` in production; don't wire projected lineups in yet (it would also need live
   depth charts and injury reports in the refresh workflow).
2. Get more signal per parameter: a single scalar per side (e.g. the foundation's predicted
   point-margin change from swapping the projected lineup in, via rollouts or the trunk) instead
   of 16–64 columns, and train on more seasons (lineup data starts in 2013).
3. Score on the subsets with more seasons (2016–2025) to see whether the QB-change lean is real.
