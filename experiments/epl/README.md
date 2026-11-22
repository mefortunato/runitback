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
