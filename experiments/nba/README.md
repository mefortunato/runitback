# NBA: player values (N2), foundation folds (N4) and the wager model on them (N5)

Work package N2 (`docs/multisport/PLAN.md`): layer B of the NFL player model for the NBA. Each player
gets an offense and a defense value from a possession-level adjusted plus-minus. Each team-game gets
a `pv.parquet` row in the NFL table's format, for the wager model's lineup terms (N5).

**Short answer**
- **The values look right.** Before the 2026-27 season the top players are Wembanyama,
  Gilgeous-Alexander, Antetokounmpo, Leonard and Curry. Caruso and White rank high on defense, Curry
  on offense alone. The bottom is backup centers.
- **The lineup value carries most of what the market knows.** The projected rotations' value
  difference alone, with no team history at all, correlates **0.83 with the market spread** and
  **0.40 with the final margin** (the market: 0.46). One point of value is worth **0.99 points of
  margin**. These are about the NFL's numbers (0.86 and 0.40).
- **The projection is a stand-in until N3:** the team's previous game's rotation. That makes the two
  previous-game change columns (`_pv_6`, `_pv_7`) zero for now.

## Reproducing

Only the code and this write-up are in git. Everything below is rebuilt from the public
SportsDataverse releases (ESPN play-by-play and player box scores):

```bash
runitback nba download --start 2015 --end 2026
runitback nba players --start 2015 --end 2026          # N1: on_court / player_games / players
runitback nba player-values --start 2015 --end 2026    # possessions.parquet, then player_values/nba/
python scripts/nba_player_values.py tune               # the penalty table below
python scripts/nba_player_values.py report             # possession checks, top players, pv vs margin
```

`player-values` fits 374 weeks (2016-2026) in about 4 minutes on 4 cores. The outputs are
`pv.parquet`, `slots.json` (possessions per side at each rotation slot, by season) and
`values_latest.csv`.

## Setup

- **Possessions** (`runitback.nba.player_values.season_possessions`).
  - A possession is a run of one team's actions (shots, non-technical free throws, turnovers,
    offensive rebounds). It ends at a made shot (unless an and-one follows), a made last free
    throw, a turnover, the other team's defensive rebound or action, or the end of the period.
  - The target is the offense's points. Technical free throws are excluded.
  - The ten players are the ones on court at the possession's first action. That uses N1's
    `credit_lineups`, so a free-throw trip counts for the five on court when the foul was called.
    0.1% of possessions lack ten known players and are dropped.
  - All-Star, Rising Stars and exhibition games are dropped (`franchise_games`).
- **The fit** (on `runitback.core.player_values`). Points per possession on an offense
  coefficient for each of the five offensive players and a defense coefficient for each of the
  five defenders, with these controls:
  - home offense;
  - how the possession started: period start, after a make, after a defensive rebound, after a
    turnover, or other;
  - season.

  It is a ridge with penalty 5,000. Seasons are weighted 0.7 per season back, over four seasons
  plus the current one; points are clipped at 4.
- **Groups: G / F / C from the ESPN position** (the choice recorded for N2). Groups set each
  player's replacement level and the lineup table's columns. Rotation slots are used for usage
  instead (below). Players with no ESPN position go to F; none had one missing in 2015-2026.
- **Replacement level by minutes.** Players with under 250 minutes on court in the window share a
  replacement coefficient for their side and group (counted as 521 possessions at 100 per 48
  minutes). A player's value is his coefficient over it, with the sign flipped for defense so
  higher is better.
- **Points per game through rotation slots.** A player's value per possession times the
  possessions on court of his rotation slot: the k-th player of a team by minutes, averaged over
  the previous season's games. A starter (slot 0) plays about 75 possessions a side.
- **Leak-free and weekly.** Every Monday from 2016 on is fit on the possessions before it, and each
  game uses its week's fit. The usage per slot comes from the previous season (the first season,
  2015, has no `pv.parquet` rows). The tests check that changing every possession on or after a
  cut-off leaves its fit identical.

### The lineup table (`pv.parquet`)

One row per team-game (`game_id`, `team`, with the wager model's team keys), in the layout of
`runitback.wager.nba.lineup.LINEUP`:

| columns | meaning | sign on the total |
|---|---|---|
| `_pv_0 .. _pv_2` | G, F, C offense: the projected rotation's points per game over replacement | + |
| `_pv_3 .. _pv_5` | G, F, C defense | − |
| `_pv_6`, `_pv_7` | offense / defense change vs the previous game's rotation | |
| `_pv_8 .. _pv_13` | each group's change vs the mean of the team's last 10 rotations | |

Every NBA player plays both ends, so each position has an offense and a defense column (the NFL's
groups are one or the other). The layout is not registered on the NBA `Sport` yet; N5 does that
when it adds the `pvc` arms. The projected rotation is the team's previous game's players who
played, ranked by minutes. It is known at tip-off, but it misses injuries, returns and trades.
`game_table(lineups=...)` takes N3's projection instead.

## Checks

**Possessions** (2015-2026, 3.04M): 95-100 per team-game and 1.04-1.15 points each, which track
the league's published pace and offensive rating. Summed per team-game, possession points match the
box score's team points exactly in 63.5% of games and within 3 points in 96.0% (MAE 0.78). Most of
the gap is technical free throws (excluded on purpose) and the dropped possessions.

**Tuning** (fit on everything before a season, predict that season's possessions from the ten
players' values; out-of-sample squared-error reduction per possession vs a constant, × 1e-4):

| penalty | 2024 | 2025 | 2026 | mean |
|---|---|---|---|---|
| 1,000 | 17.0 | 13.5 | 6.4 | 12.3 |
| 3,000 | 19.7 | 16.8 | 12.9 | 16.5 |
| **5,000** | **19.7** | **17.2** | **13.9** | **16.9** |
| 10,000 | 17.8 | 16.4 | 13.4 | 15.9 |
| 30,000 | 12.1 | 11.9 | 9.7 | 11.3 |

With penalty 5,000, a window of 4 seasons was slightly better than 3 or 2 (17.1 vs 15.8 and 15.1).
The replacement threshold barely mattered: 100 / 250 / 500 minutes gave 17.3 / 17.1 / 16.5. Decay
0.5 / 0.7 / 0.85 gave 15.2 / 15.8 / 15.8 at window 3. These grids ran before the All-Star filter
(within 0.1 of the table above). A single possession is mostly noise (the prediction correlates
0.03-0.04 with its points), so the comparison is between settings, not an absolute fit.

**Top and bottom of the latest fit** (before 2026-27; points per game over replacement as a
starter):

| player | group | offense | defense | total |
|---|---|---|---|---|
| Victor Wembanyama | F | +3.5 | +4.9 | +8.4 |
| Shai Gilgeous-Alexander | G | +6.0 | +2.2 | +8.2 |
| Giannis Antetokounmpo | F | +5.9 | +2.1 | +8.0 |
| Kawhi Leonard | F | +5.5 | +2.0 | +7.5 |
| Alex Caruso | G | +2.9 | +3.9 | +6.8 |
| Stephen Curry | G | +6.7 | +0.0 | +6.8 |
| Derrick White | G | +4.2 | +2.4 | +6.7 |
| Kyrie Irving | G | +5.7 | +0.9 | +6.6 |
| … Nikola Jokić (16th) | C | +4.5 | +1.6 | +6.1 |
| James Wiseman (last) | C | −0.8 | −0.6 | −1.3 |

Centers average about 1 point over replacement against 3.5 for guards and forwards. Values are over
each group's own replacement level, and replacement-level centers (rim protection, offensive
rebounds) are closer to the regulars than replacement guards are. Within a group the order looks
right; across groups it is not a ranking.

**Lineup value vs the game** (each game's projected-rotation value difference, home − away, summed
over the six level columns):

| season | games | corr with margin | corr with market spread | market vs margin | margin pts per value pt |
|---|---|---|---|---|---|
| 2016 | 1316 | 0.463 | 0.866 | 0.501 | 1.13 |
| 2017 | 1308 | 0.405 | 0.855 | 0.422 | 0.97 |
| 2018 | 1311 | 0.380 | 0.883 | 0.440 | 0.89 |
| 2019 | 1311 | 0.416 | 0.883 | 0.445 | 0.95 |
| 2020 | 1143 | 0.384 | 0.848 | 0.431 | 0.96 |
| 2021 | 1165 | 0.390 | 0.822 | 0.441 | 1.02 |
| 2022 | 1320 | 0.341 | 0.814 | 0.446 | 0.88 |
| 2023 | 1320 | 0.319 | 0.728 | 0.402 | 0.75 |
| 2024 | 1319 | 0.420 | 0.842 | 0.483 | 1.14 |
| 2025 | 1321 | 0.407 | 0.803 | 0.513 | 1.07 |
| 2026 | 1322 | 0.451 | 0.846 | 0.524 | 1.15 |
| **all** | **14156** | **0.398** | **0.832** | **0.465** | **0.99** |

The home edge is not in the difference (the market's and the margin's are). Late-season rest and
tanking, which the previous-game rotation misses, are the likeliest reason some seasons (2023) are
weaker; N3's projection should show whether that is so.

## Next steps

1. **N3:** the projected rotation (previous games minus DNP-injury reasons, live ESPN injuries).
   Pass it as `game_table(lineups=...)`. The previous-game change columns then become informative.
2. **N5:** register `LINEUP` on the NBA `Sport` and run `prod` / `pbp32` / `pbp32_pvc` with 4
   seeds, plus the swap test (one starter per position group to best regular / worst regular /
   replacement / out).
3. **Better values:** a box-score prior for rookies and low-minute players (today they sit at
   replacement until 250 minutes), and in-season weighting.

## Foundation folds (`pbp32`, N4)

**Short answer**
- **All six folds trained and passed the leak checks** (2022–2026 for the backtest, 2027 for upcoming
  games). Each was validated on the end of the season before its test season, and its team-season
  vocabulary stops there. Run 42753563613, about 2¼–3¼ hours per fold on a 4-core runner.
- **The embedding carries team strength, about as much as pre-game Elo and little beyond it.** A
  ridge on each team's previous-10-game mean `pbp_32` (fit before S) correlates 0.27–0.44 with the
  home margin in S, against 0.31–0.46 for the Elo difference; both together match Elo alone. That is
  the NFL's pattern (layer A mainly carries team strength). Whether it helps the wager model is N5's
  question, not this one.
- **Every fold kept its last epoch**, so the generator was still improving at 6 epochs. More epochs
  are a cheap thing to try if N5 finds `pbp32` worth keeping.

| fold S | validated on | latest train season | leak-free | val loss (type / result / clock) | PCA 32 explained | games before S / in S | corr: pbp32 | corr: Elo | corr: both |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | 2021 | 2021 | yes | 4.188 (1.042 / 0.799 / 2.347) | 74.5% | 24,389 / 1,323 | 0.344 | 0.361 | 0.363 |
| 2023 | 2022 | 2022 | yes | 4.173 (1.046 / 0.797 / 2.329) | 75.8% | 25,712 / 1,319 | 0.270 | 0.307 | 0.297 |
| 2024 | 2023 | 2023 | yes | 4.155 (1.034 / 0.801 / 2.320) | 76.2% | 27,031 / 1,319 | 0.394 | 0.416 | 0.418 |
| 2025 | 2024 | 2024 | yes | 4.235 (1.054 / 0.806 / 2.375) | 74.9% | 28,350 / 1,321 | 0.439 | 0.450 | 0.455 |
| 2026 | 2025 | 2025 | yes | 4.217 (1.067 / 0.799 / 2.352) | 76.1% | 29,671 / 1,320 | 0.444 | 0.459 | 0.458 |
| 2027 | 2026 | 2026 | yes | 4.173 (1.059 / 0.796 / 2.318) | 77.4% | 30,991 / 146 | 0.371 | 0.442 | 0.429 |

Each fold's `pbp_32.parquet` has all 62,274 team-games (31,137 games, 2002 to 2026-27 so far). The
2027 row covers only the 146 games played so far this season. "corr" is the margin correlation
described under "Checks" below; it is a sanity check of the embedding, not a wager result. The
table is the run's `report.md` (`nba-foundation-results` artifact); the folds are its
`nba-foundation-models` artifact.

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

### Reproducing the folds

`NBA foundation` (`.github/workflows/nba-foundation.yml`; run it from the Actions tab, or push a
change to the file on a `claude/**` branch) calls `nba-experiment-inputs.yml`, which downloads the
ESPN play-by-play (sportsdataverse) for 2002–2027, preprocesses it and trains each fold in its own
job. Artifacts: `nba-data`, `nba-fold<S>`, `nba-foundation-results` (the report) and
`nba-foundation-models` (all folds). Experiments that use the folds (N5) call
`nba-experiment-inputs.yml` themselves rather than reading this run's artifacts.

Locally: `runitback nba download --start 2002 --end 2027 --no-players`, `runitback nba preprocess
--start 2002 --end 2027`, then `runitback nba foundation --root foundation/nba` (or `--season S` for
one fold), and `python scripts/nba_foundation_report.py foundation/nba`.

## The wager model on the foundation and the player values (N5)

Does the NBA wager model get better with the foundation's game embeddings (`pbp32`, N4) and the
player values of each game's projected rotation (`pvc`, N2 on N3's projection), and does changing a
starter move its prediction the right way at every position?

**Short answer: pending the first run of `NBA wager experiment`** (`.github/workflows/
nba-wager-experiment.yml`). The tables below are filled from its `nba-wager-results` artifact.

### Setup

- **Walk-forward, 2022–2026** (5 folds, about 1,300 games each). For each test season S the wager
  model, the foundation fold and its PCA see only games before S; the player values are refit weekly
  on possessions before each game's Monday.
- **Arms** (`experiments/nba/wager.yaml`):
  - `prod`: the S4a box-score production model, the baseline.
  - `pbp32`: + fold S's frozen 32-dim per-possession game embeddings in the history tokens.
  - `pbp32_pvc`: + each position group's (G / F / C, offense and defense) change in the projected
    rotation's value vs the team's last 10 rotations, through sign-constrained additive terms on the
    margin and the total (the NFL's production `pvc`). The values are hidden from the head's MLP.
  - `prod_pvc`: the player values without the foundation, in case `pbp32` does not pass.
  - `_e4`: the same with 4 seeds averaged. Promotion compares `_e4` with `_e4`.
- **Projected rotation** (N3, `runitback nba lineups`): the game's box-score roster minus players
  with an injury did-not-play reason, ranked by recency-weighted minutes. **The documented leak:**
  there is no history of NBA injury reports, so a player ruled out after the line was set counts as
  known at tip-off (see N3's notes in `runitback.nba.data.lineups`). `runitback nba player-values
  --lineups data/nba/processed/projected_lineups.parquet` sums it into `pv.parquet`.
- **Join fix.** The NBA games table's ESPN ids come back from `games.csv` as integers, while the
  embedding and value tables key games by the id string, so every row would have been "missing".
  `runitback.wager.models.transformer.data.game_keys` puts both on the string; the NFL's string ids
  are untouched (`tests/wager/test_nba_lineup.py`).
- **Swap test** (`scripts/nba_lineup_swap.py`): `pbp32_pvc_e4` fit on the seasons before 2026, then
  every 2026 game re-predicted with one change to the home team's projected rotation: the top-ranked
  G, F or C replaced by the group's best / worst regular (offense + defense value per possession in
  the fit before the season, at least 2,000 minutes on court in the window), by a replacement-level
  player, or ruled out (everyone below him moves up a slot). A position passes when the margin and
  the win probability move with the value in every game where it moved by more than 0.05 points.

### Wager performance

Paired Brier difference (negative is better) ± 95% CI half-width. Pending.

### Swap test

Pending.

### Reproducing

Run `NBA wager experiment` from the Actions tab, or push a change to `experiments/nba/wager.yaml` or
the workflow on a `claude/**` branch. It rebuilds the games with lines and the foundation folds
(`nba-experiment-inputs.yml`), and in its `values` job downloads the 2015–2027 play-by-play and
player box scores and runs `runitback nba players`, `lineups` and `player-values --lineups`. One
cell locally: `python scripts/wager_arm.py pbp32_pvc_e4 2025 --sport nba --experiment
experiments/nba/wager.yaml` (with `runs/foundation/fold2025/` and `player_values/nba/pv.parquet` in
place).
