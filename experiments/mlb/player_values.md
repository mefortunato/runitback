# MLB: player values

Work package M2 (`docs/multisport/PLAN.md`): layer B of the NFL player model for MLB. Each player
gets a batting value and a pitching value from a plate-appearance adjusted plus-minus on RE24 run
value. Each team-game gets a `pv.parquet` row for the wager model's lineup terms (M5).

**Short answer**
- **The values look right.** Before 2027 the top batters are Judge, Alvarez, Soto and Ohtani
  (+63 to +77 runs over replacement per 600 PA). The top pitchers are Wheeler, Skubal, Yamamoto,
  Sale and Skenes (+45 to +48 per 700 batters faced).
- **The lineup value carries most of what the market knows.** The home-minus-away value of the
  projected lineup, starter and bullpen alone, with no team history, correlates **0.80 with the
  market's home win probability** and **0.157 with the home team winning** (the market: 0.211).
  One run of value is worth **0.90 runs of margin**.
- **The projection is M3's** (merged while this package was in progress): the probable starter,
  and the previous lineup against his throwing hand. M3's posted-lineup stand-in and this
  package's own previous-nine stand-in score within 0.01 of it (below).
- **Fielding is left out.** The optional Statcast fielder terms (`fielder_2..9`) made the
  out-of-sample fit worse (below).

## Reproducing

Only the code and this write-up are in git. Everything is rebuilt from MLB's Stats API (and, for
the market comparison only, the wager games table):

```bash
runitback mlb process --start 2014 --end 2026        # M1: PA / games / lineups / players tables
runitback mlb lineups                                 # M3: projected_lineups / posted_lineups
runitback mlb player-values --start 2015 --end 2026  # -> player_values/mlb/ (--lineups projected)
python scripts/mlb_player_values.py tune              # the tuning tables below
runitback-wager mlb download --output_file data/mlb/raw.csv --first_season 2015 --last_season 2026
python scripts/mlb_player_values.py report            # PA checks, top players, pv vs the game and market
```

2014 is there only for 2015's run-expectancy matrix. The feeds took about 45 minutes to download
from a cloud container (6 threads, 0.08 s between requests). `player-values` fits 336 weeks
(2016-2026) in under 5 minutes on 4 cores. The outputs are:
- `pv.parquet`;
- `usage.json`: plate appearances per batting slot, and batters faced per start and in relief, by
  season;
- `values_latest.csv`.

## Setup

- **Events: plate appearances** from M1's `pa_<S>` tables, 2015-2026: 2.1M PAs, regular season and
  postseason.
  - The target is the PA's RE24 run value: run expectancy after the PA's own event, minus before
    it, plus the runs it scored.
  - Season S's run-expectancy matrix is fit on season S-1, so the target is leak-free too.
- **The fit** (`runitback.mlb.player_values`, on `runitback.core.player_values`). Run value on a
  batting coefficient for the batter and a pitching coefficient for the pitcher, with these
  controls:
  - park (venue);
  - the batter/pitcher handedness matchup (LL / LR / RL / RR);
  - the base/out state before the event (24);
  - home batting;
  - the pitcher's role: a starter's first, second, third or fourth-plus time through the order, or
    a reliever. This makes a pitcher's coefficient role-neutral;
  - season.

  It is a ridge with penalty 300. Pitchers' columns are scaled 0.7, so they are shrunk harder than
  batters. Seasons are weighted 0.7 per season back, over three seasons plus the current one.
- **Role-specific players and replacement level.**
  - A player's batting and pitching coefficients are separate columns with separate replacement
    levels. Two-way players have both.
  - Groups:
    - `B`: batters;
    - `PB`: pitchers batting (mostly the pre-2022 National League), so they don't drag down the
      batters' replacement level;
    - `P`: pitchers;
    - `XP`: position players pitching in blowouts. They allowed +0.13 runs per PA against −0.004 for
      pitchers in 2024-26. Without their own group they inflated every pitcher by about 3 runs per
      700 batters faced.
  - Players under 300 PAs in a role (in the window) share that role's replacement coefficient.
  - A value is the coefficient over replacement level, with the sign flipped for pitching, so
    higher is better.
  - Groups use each player's primary position from M1's players table (its latest listing). That
    only decides which replacement pool a player is compared against.
- **Runs per game through usage** (the plan's "SP, bullpen, lineup slots by PA share"):

  | group | value per game |
  |---|---|
  | `BAT` | each of the nine starters' value per PA × his batting slot's PAs per game. A slot's PAs are the league's starters in that slot, previous season: 4.5 for the leadoff spot down to 2.9 for the ninth, or 3.8 since the universal DH |
  | `SP` | the starter's value per batter faced × his expected batters faced: his last 10 starts, shrunk toward the league's mean start by 3 pseudo-starts. The mean start was 24.5 batters in 2015 and 21.9 in 2025 |
  | `BP` | the relievers the team used in its previous 15 games, weighted by their share of its relief batters faced there, × the league's relief batters faced per game (15.8 in 2025). The game's starter is excluded |
- **Leak-free and weekly.**
  - Every Monday from 2016 on is fit on the PAs before it, and each game uses its week's fit.
  - Usages come from earlier games, or from the previous season for league means.
  - The tests check that changing every PA on or after a cut-off leaves its fit identical, and
    that `game_table`'s weekly fits equal a fit cut at that Monday.

### The lineup table (`pv.parquet`)

One row per team-game. The keys are `game_pk` / `team_id`, plus the wager model's `game_id` (the
game_pk as a string) and `team` (MLB's current abbreviation). All 51,274 team-games of 2016-2026
join the wager games table on `(game_id, team)`. The columns follow
`runitback.wager.mlb.lineup.LINEUP`, in runs per game over replacement:

| columns | meaning | sign on the total |
|---|---|---|
| `_pv_0` `BAT` | the projected starting nine (mean 1.35, sd 0.40) | + |
| `_pv_1` `SP` | the starting pitcher (mean 0.72, sd 0.42) | − |
| `_pv_2` `BP` | the bullpen (mean 0.33, sd 0.12) | − |
| `_pv_3` | lineup change vs the previous game's actual nine (sd 0.14) | |
| `_pv_4` | pitching (SP + BP) change vs the previous game (sd 0.54) | |
| `_pv_5 .. _pv_7` | each group's change vs the mean of the team's last 10 games | |

The starting pitcher keeps a column, and so a sign-constrained term, of his own, as M5 asks. The
layout is not registered on the MLB `Sport` yet; M5 does that when it adds the `pvc` arms.

**The projection** (`runitback mlb player-values --lineups`, read through `m3_lineups`):
- **`projected` (the default):** M3's projection. The starter is the probable pitcher. M3
  documents its leak: the Stats API reports the actual starter as the probable for finished games,
  so a late scratch counts as known; they agree in 99.8% of games. The lineup is the team's latest
  lineup against a starter of the same hand, updated for moves.
- **`posted`:** M3's stand-in for the lineup posted hours before first pitch, which is the actual
  nine. It leaks a late scratch after posting.
- **`previous`:** this package's own stand-in. It uses the previous game's nine and the actual
  starter.

The bullpen is always projected from recent usage, so it misses call-ups and injured-list moves.

## Checks

**Plate appearances** (M1's replay checks, 2015-2026): every season's PA count matches the box
scores exactly. Each season has at most 2 base or post-state mismatches and 2 fielder mismatches
in about 205,000 fielding credits. Run value per PA has mean −0.010 to +0.008 by season and
sd 0.47-0.51.

**Tuning.** Each setting is fit on everything before a season and predicts that season's PAs from
the players' values. The cells are the out-of-sample squared-error reduction per PA against a
constant (× 1e-4), then the prediction's correlation with the run value:

| penalty | pitcher scale | min PA | 2023 | 2024 | 2025 | 2026 | mean gain | mean corr |
|---|---|---|---|---|---|---|---|---|
| 100 | 0.7 | 300 | −1.42 / .0336 | −0.65 / .0353 | 1.98 / .0418 | −3.07 / .0267 | −0.79 | .0344 |
| **300** | **0.7** | **300** | **0.05 / .0323** | **0.43 / .0337** | **2.80 / .0404** | **−1.67 / .0251** | **0.40** | **.0329** |
| 300 | 1.0 | 300 | −0.38 / .0323 | 0.06 / .0337 | 2.67 / .0409 | −2.01 / .0256 | 0.09 | .0331 |
| 1000 | 0.7 | 300 | 0.50 / .0281 | 0.68 / .0291 | 2.62 / .0359 | −0.94 / .0211 | 0.71 | .0286 |
| 1000 | 1.0 | 300 | 0.37 / .0286 | 0.54 / .0295 | 2.69 / .0367 | −1.03 / .0219 | 0.64 | .0292 |

- **Why the selection rule.** A single PA is mostly noise, and the run value has heavy tails (a
  bases-loaded home run is +3.3), so the squared-error gain swings by season.
  - The correlation keeps rising as the penalty drops, then flattens (about .0347 at 10-100).
    Meanwhile the gain falls: with less shrinkage the values spread too wide out of sample. At
    penalty 1,000 the next season's run value moves only 0.3-0.75 runs per run of predicted value.
  - The wager model learns each column's weight, so ranking matters more than scale. But the
    values should stay readable as runs (the lineup editor's arithmetic).
  - The rule: **the highest correlation among settings whose mean gain is not negative.** That is
    penalty 300, pitcher scale 0.7 and 300 PAs, with 95% of the best correlation.
- **Other settings** (earlier grids, before the `XP` group; within 0.01 of the table above where
  rerun):
  - window: 2 / 3 / 4 seasons were within noise;
  - replacement threshold: 150 PAs was worse everywhere; 600 gave a better gain but a lower
    correlation;
  - pitcher scale 0.5 gave a better gain but a lower correlation.
- **Fielding.** Adding a fielding coefficient for each of the eight fielders (Statcast's
  `fielder_2..9`) cut the correlation from .0326 to .0291 and the gain from +0.14 to −3.44. It
  stays off (`--fielding` keeps it available). A per-PA fielding term spreads one ball's credit
  over eight players. Outs-above-average-style data would be the way to add defense later.

**Top of the latest fit** (before 2027; runs over replacement per 600 PA / per 700 batters faced):

| batter | runs | pitcher | runs |
|---|---|---|---|
| Aaron Judge | +76.7 | Zack Wheeler | +48.0 |
| Yordan Alvarez | +69.1 | Tarik Skubal | +47.9 |
| Juan Soto | +62.9 | Yoshinobu Yamamoto | +47.2 |
| Shohei Ohtani | +62.6 | Jacob Misiorowski | +47.0 |
| Nick Kurtz | +56.3 | Drew Rasmussen | +46.7 |
| Kyle Schwarber | +47.8 | Chris Sale | +46.6 |
| Yandy Díaz | +47.7 | Cam Schlittler | +45.9 |
| Fernando Tatis Jr. | +47.6 | Cade Smith (RP) | +45.9 |
| Pete Alonso | +47.4 | Mason Miller (RP) | +45.3 |
| Bobby Witt Jr. | +46.9 | Paul Skenes | +45.1 |

Per batter faced, elite relievers rate with elite starters, because the role control takes out
the reliever's short-stint advantage. Their game impact is much smaller: they face far fewer
batters, and the bullpen column weights by usage. The 583 batters and 698 pitchers with their own
value all sit at or above replacement; the lowest are near 0.

**Lineup value vs the game.** Each game's projected value difference (home − away, `BAT + SP + BP`)
against the result and the market, with M3's projection. The market is the de-vigged home
moneyline probability from the wager games table.

| season | games | corr with margin | corr with home win | corr with market p | market vs home win | margin runs per value run |
|---|---|---|---|---|---|---|
| 2016 | 2463 | 0.171 | 0.151 | 0.775 | 0.161 | 0.90 |
| 2017 | 2468 | 0.152 | 0.118 | 0.769 | 0.168 | 0.79 |
| 2018 | 2464 | 0.198 | 0.170 | 0.802 | 0.241 | 0.89 |
| 2019 | 2466 | 0.235 | 0.208 | 0.779 | 0.380 | 1.02 |
| 2020 | 951 | 0.152 | 0.111 | 0.765 | 0.191 | 0.71 |
| 2021 | 2466 | 0.204 | 0.163 | 0.827 | 0.196 | 0.97 |
| 2022 | 2470 | 0.241 | 0.208 | 0.856 | 0.226 | 1.04 |
| 2023 | 2471 | 0.183 | 0.127 | 0.786 | 0.176 | 0.89 |
| 2024 | 2472 | 0.173 | 0.154 | 0.784 | 0.189 | 0.85 |
| 2025 | 2477 | 0.156 | 0.136 | 0.803 | 0.164 | 0.82 |
| 2026 | 2469 | 0.148 | 0.142 | 0.796 | 0.157 | 0.83 |
| **all** | **25637** | **0.186** | **0.157** | **0.795** | **0.211** | **0.90** |

**By projection** (all seasons):

| projection | corr with margin | corr with home win | corr with market p |
|---|---|---|---|
| `previous` (previous nine, actual starter) | 0.186 | 0.157 | 0.797 |
| **`projected`** (M3) | **0.186** | **0.157** | **0.795** |
| `posted` (M3, the actual nine) | 0.189 | 0.160 | 0.805 |

- Knowing the actual nine helps a little.
- M3's hand-matched projection is no better than the previous nine at the game level. It does
  make the lineup change column informative (`_pv_3` sd 0.14, against 0 under the previous-nine
  stand-in), which M5's change terms read.
- **Per group** (all seasons, M3's projection; correlation of the home − away difference with the
  margin, then margin runs per value run): `BAT` 0.158 / 1.37, `SP` 0.128 / 1.01,
  `BP` 0.123 / 3.75.
  - Fit jointly, the slopes are 0.98 / 0.71 / 1.58.
  - The bullpen column is worth more per run than its own arithmetic says. Recent bullpen usage
    partly stands in for team quality: good teams use their best relievers in close games. M5's
    learned weights will absorb that.
- **Totals.** The sum of both lineups' batting minus both teams' pitching correlates 0.18-0.32
  with the market's total line within a season, but only 0.04-0.08 with the actual total. That is
  in line with how little the market's total explains a single game's runs.
- **2019's market row** (0.380 against about 0.16-0.24 in other seasons) looks too good. M5 should
  check that season's ESPN lines for post-game prices before trusting a 2019 comparison.

## Next steps

1. **Daily refresh** (A2): rebuild `pv.parquet` from M3's live projection
   (`projected_lineups_upcoming.parquet`) for upcoming games. `game_table` already takes any
   lineup in `m3_lineups`' form. The weekly fit for an upcoming game is the latest one.
2. **M5:** register `LINEUP` on the MLB `Sport` and run `prod` / `prod_pvc` (and the `pbp32` arms
   once M4 exists) with 4 seeds on moneyline and total, plus the swap test:
   - a lineup slot or the starter changed to the best regular, worst regular or replacement, or
     scratched;
   - a bullpen arm removed.
3. **Better values:**
   - a prior for rookies and call-ups (today they sit at replacement until 300 PAs), from minor
     league stats or draft slot;
   - in-season weighting;
   - park factors by season (a renovated park keeps one venue id).
