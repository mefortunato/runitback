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
