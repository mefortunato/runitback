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

Results go here when the stages finish.
