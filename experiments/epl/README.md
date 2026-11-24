# Premier League experiments

The EPL's games table and odds are `runitback.wager.epl`; the plan is `docs/epl/PLAN.md` (on
`claude/epl-plan-draws-transformer`). Seasons are by ending year (2025-26 is 2026).

## D1: which way to remove the bookmaker's margin (2026-11)

`scripts/epl_devig_rank.py` removes the margin from the 1X2 prices three ways
(`runitback.wager.backtest.no_vig_probabilities`) and scores the home / draw / away probabilities with the ranked
probability score (RPS, lower is better). Differences are paired per game with 95% CIs. Both periods are before
the first test season (2022), so the choice leaks nothing.

| Period (prices) | Games | Method | RPS | Multiclass log loss | RPS vs proportional |
|---|---|---|---|---|---|
| 2006-2016 (opening bet365, then closing Pinnacle) | 4,180 | proportional | 0.19321 | 0.96363 | |
| | | power | 0.19301 | 0.96275 | -0.00019 ± 0.00021 |
| | | shin | 0.19305 | 0.96290 | -0.00016 ± 0.00015 |
| 2020-2021 (closing market average) | 760 | proportional | 0.20475 | 0.98460 | |
| | | power | 0.20503 | 0.98612 | +0.00029 ± 0.00046 |
| | | shin | 0.20494 | 0.98573 | +0.00020 ± 0.00033 |

**Short answer: `proportional`.** The two periods disagree, and every difference is tiny and inside or at the
edge of its CI. The plan follows the closing-average period for the average-price market arm, and proportional
is also what the other sports use. The schema records it (`GameSchema.devig`, set in
`runitback/wager/epl/data/features.py`).

## D3: the three-way backtest on real data (smoke run, 2026-11)

`runitback-wager epl backtest --family elo` (test seasons 2022-2026, 1,853 games; each season fit on the earlier
ones, replayed date by date). The market arm is the closing market average with the margin removed proportionally;
`book` is the games table's preferred bookmaker (closing Pinnacle through 2025).

| Source | RPS | Multiclass Brier | Log loss | 3-way accuracy |
|---|---|---|---|---|
| Elo (the NFL's settings) | 0.2036 | 0.5887 | 0.9875 | 52.9% |
| Market (closing average) | 0.1941 | 0.5684 | 0.9575 | 55.9% |
| Book (closing Pinnacle) | 0.1942 | 0.5685 | 0.9576 | 55.3% |

Elo minus market: RPS +0.00948 ± 0.00276, log loss +0.02994 ± 0.00879 (95% CIs, paired per game). The market is
better in every test season.

**This is a pipeline check, not a result.** Elo runs with the NFL's FiveThirtyEight settings (`points_per_elo`
1/25: a margin RMSE of 5.0 goals against the market's 1.67), draws never move its ratings, and a promoted club's
first games are skipped; the plan's D5 (football Elo) and E2 (cold start) address those. The ordered probit
absorbs the wrong scale into its spread, so its probabilities are still sensible.

## D5: football Elo (2026-11)

`EloParams(mov="football")` is World Football Elo's goal-difference multiplier (1 for a margin of 0 or 1, 1.5 for 2,
`(11 + margin) / 8` above), so a draw moves ratings by `K * (0.5 - expected)`: the NFL rule's `log(margin + 1)` is 0
for a draw. `new_team` is the rating a club starts at the first time it appears. `scripts/epl_elo_tune.py` replays Elo
from 2001 and scores 2006-2016 only (2001-2005 burn-in; all before the first test season), by the RPS of the ordered
probit of the rating gap (`points_per_elo` is the least-squares slope of the goal difference on the gap).

The plan's grid put its best settings on its edges (`regress` 0.1, `home` 85), so `regress` was extended down to 0
and `home` up to 100 (500 settings).

| | k | home | regress | new club | points per Elo | RPS, 2006-2016 |
|---|---|---|---|---|---|---|
| NFL settings (FiveThirtyEight rule) | 20 | 65 | 0.33 | 1505 | 1/155 | 0.19663 |
| Best of the grid | 15 | 85 | 0.05 | 1355 | 1/205 | 0.19572 |
| **Frozen in `epl/__init__.py`** | **15** | **80** | **0.05** | **1405** | **1/200** | **0.19577** |

**The surface is flat.** Mean RPS over the grid by setting: `k` 10 to 30 spans 0.19685-0.19796 (15 best), `home` 40
to 100 spans 0.19669-0.19862 (85 best, 70-100 within 0.0004), `regress` 0 to 0.33 spans 0.19690-0.19815 (0.05 and 0.1
tied), and a new club's start 0 to 150 below the mean spans 0.19722-0.19738, nothing. The tuned setting beats the
NFL's by 0.0009 in the tuning window. A new club at 1405 (100 below the mean) is within 0.00004 of the best (150).

**The home edge is 80 on every window** (finer grid of 10): 2006-2016, 2006-2021 and 2006-2021 without 2020-21. The
no-crowd season does not move it; it does cost RPS (0.19653 with 2021 against 0.19468 without, on 2006-2021).

### What it does on the test seasons

`runitback-wager epl backtest --family elo`, 2022-2026, 1,853 games, walk-forward as before (calibration fit on the
last training season):

| Elo | RPS | Margin RMSE | Minus market (closing average) |
|---|---|---|---|
| NFL settings (D3 smoke run) | 0.2036 | 5.04 | +0.00948 ± 0.00276 |
| Football Elo | 0.2029 | 1.71 | +0.00881 ± 0.00251 |

The scale is fixed (the margin error is in goals now) and the RPS improves by 0.0007, less than the CI of either row.
The market is better in every test season. Where the rest of the gap comes from: with the probit fit on the test
seasons themselves (an oracle, not a forecast) the same ratings score 0.2009 (by season 0.1959, 0.2029, 0.1890,
0.2061, 0.2084), so **about 0.002 of the gap is calibration on one validation season and about 0.007 is what a
rating-only model lacks** against prices that see lineups and injuries. That is the room for the transformer and
the player data. Promoted clubs' first games are still left out of the walk-forward (E2).

## E2 and E3: cold start, promoted clubs and the football market tokens (2026-11)

**Cold start (`Sport.cold_start`, on for the EPL).** A club the model has no state for (newly promoted, or back after
years away) is predicted from an empty history, the same as a team's first game in training, instead of being left out
of the backtest and refused in production. Elo and the transformer support it (`MatchupModel.supports_cold_start`);
the other families ignore it. Three settings go with it: the transformer trains on such games (`min_history` 0, which
`Sport.transformer` now carries); a team's history stops two seasons back (`drop_older_than` 2, applied in
`sequence()` so training sees exactly what serving does, since a club absent for more than that has no state); and the
context columns `promoted` / `opp_promoted` (1 where a club was not in the previous season's league, known before the
season starts; 0 in the first season of the table).

The walk-forward now predicts **every** test game: 380 in each of 2022-2026 (1,900), where 1,853 were predicted before
(47 left out, about 9 a season). Elo's RPS on the 72 games in which a promoted club plays one of its first five
games (`scripts/epl_promoted_rps.py`):

| Elo (football), 2022-2026 | n | model RPS | market RPS | model minus market |
|---|---|---|---|---|
| a promoted club's first 5 games | 72 | 0.1640 | 0.1549 | +0.0091 ± 0.0112 |
| all other games | 1,828 | 0.2038 | 0.1951 | +0.0087 ± 0.0025 |
| every game | 1,900 | 0.2022 | 0.1936 | +0.0087 ± 0.0025 |

The model is no further behind the market on promoted clubs' first games than elsewhere (the interval is wide: 72
games). Those games are easier for both: the promoted side is usually the underdog.

**Market tokens for a sport with draws (`market_features_with_draws`).** `_mkt_win_minus_loss` and `_mkt_draw_prob` (from
the three de-vigged prices) replace the two-way probit; `_mkt_points_surprise` (points won, 3 / 1 / 0, minus the points the
prices expected) replaces the win surprise; `_mkt_over_z` (the probit of the over's no-vig probability) replaces the constant
2.5 line; `_mkt_closing` is 1 where the game's prices are closing prices (`odds_closing`: opening bet365 through 2012,
closing Pinnacle after), so the model can weight sharp prices more. `_mkt_spread` (the handicap line) and its cover stay as
tokens only. Other sports' tokens are unchanged. A constant column, such as the over's probit when every game is priced
-110 / -110, has its zero spread replaced by 1 when the features are standardized, so it becomes 0 and never divides by 0.

**`eff2` (`core/efficiency.epl_extra`, an option of its own).** Second-half goals and a red-card indicator, for both
sides. The plan's third column, shots on target conceded per shot, is already there: it is the opponent's `accuracy` in
`epl_efficiency` (`_eff_opp_accuracy`), so it is not added again.

**A latent bug the smoke run found:** a transformer with `distribution=True` could not be given `margin_range` /
`total_range` (`TransformerModel` did not take them), so it always predicted margins over -70..70, the NFL's. They now
reach the model (defaults unchanged), so an EPL transformer can use -10..10.

**Smoke run, not a result:** the transformer with every option on (history 20, d_model 32, market tokens, efficiency,
`eff2`, Elo strength, `distribution=True` over -10..10) trained for 2 epochs on 2001-2024 (validation 2025) and
predicted all 380 games of 2026 with finite probabilities (win 0.14-0.74, draw 0.14-0.27). That is a check that the
pipeline runs end to end, nothing more; the arms are E4's.

## E1 and E4: the baselines and the transformer arms (running, 2026-11)

`experiments/epl/wager.yaml` lists the arms and the three stages; `.github/workflows/epl-wager-experiment.yml` runs a
stage as one Actions job per (arm, season) and `scripts/epl_report.py` scores it. The games table is rebuilt in the
run, nothing is committed. A stage is hours of compute, so the arms are not chosen on the test seasons:

1. **`screen`**: every arm on 2017-2021, one seed (each season fit on the ones before it). Its report picks the
   **reference arm** (the best of `elo`, `sklearn`, `gnn` and the default transformer `tf` on those seasons) and the
   **two transformer arms** worth a 4-seed ensemble, and prints the `arms` to give stage 3. (15 arms x 5 seasons.)
2. **`baselines`**: the baselines and four seeds of `tf` on 2022-2026, which give the **noise floor**: how far apart
   two seeds of the same model land, so a 1-seed difference smaller than that is not an effect. (8 x 5.)
3. **`final`**: the reference arm and the screen's two picks as 4-seed ensembles (`<arm>_e4`) on 2022-2026, each
   against the reference and the market, with the plan's verdict (the CI of the RPS difference includes 0 or favors the
   arm), by season and on a promoted club's first games. Run it with the `arms` and `reference` inputs the screen printed.

Cost, measured on four cores here: one season's fit takes about 8 minutes for a transformer, 2 for the GNN, under 1
for sklearn and Elo; a 4-seed ensemble about 30. Stages 1 and 2 are about 100 jobs.

The arms: `tf` (the box score and the result), history 19 and 57, the draws market tokens with the closing flag
(`tf_mkt`), efficiency (`tf_eff`), efficiency with `eff2` (`tf_eff2`), Elo strength (`tf_opp`), their combinations
(`tf_mkt_eff`, `tf_mkt_eff_opp`, `tf_all`) and the RPS auxiliary loss (`tf_rps`, weight 0.5).

A dress rehearsal of the same scripts on real data (two seasons, two epochs, `scripts/wager_arm.py` then
`scripts/epl_report.py`) ran before the first stage; it found and fixed only formatting, and `tf_rps` trained.

### `baselines` stage: the baselines on the test seasons (2022-2026, one seed)

Run 45312464607, scored by `scripts/epl_report.py baselines` on the 1,739 games every arm predicted (sklearn and the GNN
have no cold start, so they skip 161 promoted clubs' early games; Elo and the transformer predict all 1,900). RPS, lower
is better; the last column is the paired difference to the closing market average with its 95% CI.

| Arm | RPS | Multiclass log loss | Behind the market |
|---|---|---|---|
| Market (closing average) | 0.19392 | 0.9574 | |
| Pinnacle close (`book`) | 0.19396 | 0.9574 | +0.00004 |
| Elo (football) | 0.20300 | 0.9861 | +0.00908 ± 0.00261 |
| `tf` (transformer, seed 0) | 0.20327 | 0.9872 | +0.00935 ± 0.00297 |
| sklearn | 0.20483 | 0.9919 | +0.01091 ± 0.00337 |
| GNN | 0.20509 | 0.9925 | +0.01117 ± 0.00335 |
| home advantage only | 0.23297 | 1.0711 | +0.03904 ± 0.00630 |

**Every model is behind the market**, by about 0.009 to 0.011 RPS, in every test season and with a CI that excludes 0.
The models are indistinguishable from one another: the transformer with only the box score is level with Elo (-0.00027
± 0.00320) and a little ahead of sklearn and the GNN (about 0.0016-0.0018, inside their CIs). Home advantage alone is far
behind (+0.030).

**The noise floor.** Four seeds of the same transformer (`tf`, `tf_s1`, `tf_s2`, `tf_s3`, on 1,900 games) score 0.20267,
0.20417, 0.20385 and 0.20256: a standard deviation of **0.00082**, and **0.00162** between the best and worst seed. So a
1-seed difference under about 0.0016 is not an effect. That is the same size as the whole spread among Elo, `tf`, sklearn
and the GNN, so one seed cannot rank them, which is why the plan promotes on 4-seed ensembles. It is also smaller than the gap
to the market (0.009), so that gap is not seed luck.

These are the arms' numbers on the test seasons; the **reference arm is not chosen from them**: it comes from the
`screen` on 2017-2021 (below).

### `screen` stage: the arms on the validation seasons (2017-2021, one seed)

Run 45312451331 (75 jobs, all succeeded). **Its own first report was wrong and is superseded.** It said 2017-2021 but scored
only 2020 and 2021 (740 of 1,782 games): the comparison dropped every game the market had no prices for, and the closing
average starts in 2019-20. Nothing was wrong with the arms, which ran on all five seasons; the same results were
re-scored with the fix (`scripts/epl_report.py`: the market is the closing average where it exists and the preferred
bookmaker's close, Pinnacle, before; and every report now says how many of the games every arm predicted it scored and
warns when that is not nearly all). Scored on all 1,782 games every arm predicted:

| Arm | RPS | Δ vs `tf` (95% CI) | Behind the market |
|---|---|---|---|
| Market | 0.19267 | | |
| `tf_opp` (+ Elo strength) | 0.19792 | -0.00228 ± 0.00153 | +0.00525 |
| `tf_mkt` (+ market tokens) | 0.19807 | -0.00213 ± 0.00176 | +0.00539 |
| `tf_mkt_eff_opp` | 0.19823 | -0.00197 ± 0.00202 | +0.00556 |
| `tf_mkt_eff` | 0.19846 | -0.00174 ± 0.00183 | +0.00578 |
| `tf_all` | 0.19917 | -0.00103 ± 0.00181 | +0.00649 |
| GNN | 0.19951 | -0.00069 ± 0.00174 | +0.00684 |
| `tf_eff` | 0.19992 | -0.00029 ± 0.00152 | +0.00724 |
| `tf` (box score and result only) | 0.20020 | | +0.00753 |
| `tf_h57` (history 57) | 0.20043 | +0.00023 ± 0.00114 | +0.00775 |
| `tf_h19` (history 19) | 0.20055 | +0.00035 ± 0.00167 | +0.00788 |
| Elo | 0.20057 | +0.00037 ± 0.00356 | +0.00790 |
| `tf_eff2` | 0.20087 | +0.00066 ± 0.00140 | +0.00819 |
| `tf_rps` (RPS loss, weight 0.5) | 0.20147 | +0.00127 ± 0.00073 | +0.00880 |
| sklearn | 0.20238 | +0.00218 ± 0.00222 | +0.00971 |
| home advantage only | 0.23459 | +0.03439 ± 0.00628 | +0.04192 |

The rule picked **reference `gnn`** (the best of Elo, sklearn, GNN and `tf`; it leads `tf` by 0.0007, inside the noise) and the
transformer arms **`tf_opp` and `tf_mkt`** for 4-seed ensembles. What the screen says:

- **Elo strength and the market tokens help, a little.** Each takes about 0.0021-0.0023 off the default transformer, which is
  larger than the biggest gap between two seeds of `tf` (0.0016) but not by much, and their CIs only just exclude 0. Stacking
  them (`tf_mkt_eff_opp`, `tf_all`) is no better than either alone.
- **Efficiency columns, `eff2` and a history of 19 or 57 games do nothing** (all within +-0.0007 of `tf`).
- **The RPS auxiliary loss hurts** at weight 0.5: +0.0013 ± 0.0007, a CI that excludes 0 on the wrong side. The margin NLL
  already prices the draw; a second loss on the same probabilities does not help.
- **Even the best arm is 0.005 behind the market** on these seasons (the closing line, Pinnacle's, is sharper than anything
  here, and has the lineups and injuries these models do not).

### `final` stage: 4-seed ensembles on the test seasons (2022-2026)

Run 45319135123 (20 jobs, all succeeded): `gnn` (the screen's reference, one seed) and `tf`, `tf_opp` and `tf_mkt` (the
screen's two picks) as 4-seed ensembles. `tf_e4` is not in the plan: without it the picks' ensembles would be compared with
a single-seed reference, and averaging four seeds would flatter them. Scored on the 1,739 games every arm predicted. The
last two columns are the paired RPS difference (95% CI) to the arm in the heading; negative is better.

| Arm | RPS | Behind the market | vs `tf_e4` (like for like) | vs `gnn` (the reference) | vs Elo |
|---|---|---|---|---|---|
| Market (closing average) | 0.19392 | | | | |
| `tf_opp_e4` (+ Elo strength) | 0.20241 | +0.00849 ± 0.00263 | -0.00087 ± 0.00127 | -0.00274 ± 0.00200 | -0.00059 ± 0.00266 |
| `tf_mkt_e4` (+ market tokens) | 0.20291 | +0.00899 ± 0.00262 | -0.00038 ± 0.00156 | -0.00224 ± 0.00228 | -0.00009 ± 0.00291 |
| Elo (football) | 0.20300 | +0.00908 ± 0.00261 | | | |
| `tf_e4` (box score and result) | 0.20329 | +0.00936 ± 0.00296 | | -0.00186 ± 0.00190 | +0.00029 ± 0.00324 |
| `gnn` (reference, 1 seed) | 0.20515 | +0.01123 ± 0.00335 | +0.00186 ± 0.00190 | | +0.00215 ± 0.00347 |

**What it says**

1. **The screen's picks were mostly noise.** On the validation seasons `tf_opp` and `tf_mkt` looked 0.0021-0.0023 better than
   `tf`; on the test seasons, like for like, they are 0.0009 and 0.0004 better with CIs that include 0. The screen chose
   the best two of 15 near-ties, and the best of many noisy results is lucky by construction. Neither option is shown to help;
   neither is shown to hurt.
2. **The plan's verdict ("the CI includes 0 or favors the arm") passes both arms against every reference, but that only
   means "not significantly worse".** Against `gnn`, `tf_opp_e4` is the only arm whose CI is wholly below 0 (-0.0027 ± 0.0020,
   its upper end at -0.0007), and so is `tf_e4`'s nearly (+0.00004): that comparison mostly shows a single-seed GNN
   behind a 4-seed ensemble, not an effect of the arm.
3. **Nothing beats Elo.** The football Elo (no training, no seeds) is level with the best transformer arm (0.0006 apart, CI
   +-0.0027) and with the default one. Four seeds of a transformer are 0.0006 better than their average single seed
   (0.20329 against 0.20387 on these games) and that is all ensembling bought.
4. **Every model is 0.0085 to 0.011 behind the market**, with a CI that excludes 0, in the test seasons taken together. There
   is no edge in models of the games table alone.
5. **A promoted club's first five games are where the market is furthest ahead** (45 games every arm predicted; a small
   sample, and it leaves out the first game, which the GNN cannot predict): the market scores 0.141, the models 0.179-0.188,
   +0.038 to +0.047 behind with CIs of about +-0.02, four to five times the gap on the other games. The market knows the
   squads and the signings; a model that has seen nothing of the club cannot. That is the most direct argument for the player
   data.
6. **By season** the transformer arms beat `gnn` in 2022-2025 and trail it in 2026; `gnn` is far behind in 2025 (+0.009
   against `tf_e4`).

**What is not shown.** The noise floor measured earlier (sd 0.0008 for one seed) is for single seeds; the ensembles' own
spread was not measured, so a 0.0009 gap between two ensembles cannot be called noise or effect. These are also 1,739 games
in five seasons: the CIs are about +-0.0013 even for paired ensembles.

**Conclusion for the plan.** E1 and E4 are done and found no transformer arm worth promoting over the Elo baseline: the options
the plan expected to help (market tokens, efficiency, Elo strength, longer or shorter history, the RPS loss) each move the RPS
by less than the noise, and the RPS auxiliary loss hurts. The remaining gap to the market (about 0.009) is information the
games table does not have: the lineups, injuries and squads that the player data can supply. E5 (promote and serve) therefore
waits for the player-data arm, as already decided; there is nothing in these results to promote.
