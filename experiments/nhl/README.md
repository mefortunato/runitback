# NHL: player values (H2)

Work package H2 (`docs/multisport/PLAN.md`) builds layer B of the NFL player model for the NHL.

- **Skaters:** each skater gets an offense and a defense value. Both come from an expected-goals adjusted plus-minus per 60 minutes, fit on shift segments.
- **Goalies:** each goalie gets goals saved above expected (GSAx).
- **Output:** each team-game gets a `pv.parquet` row in the NFL table's format, for the wager model's lineup terms (H6).

**Short answer**
- **The values look right.**
  - **Forwards:** before 2026-27 the top forwards are McDavid, MacKinnon, Matthews, the Tkachuks and Kaprizov. Bergeron and Barkov rank high on defense.
  - **Defensemen:** the top of the list is Fox, Werenski, Hutson and Hughes, with Slavin on defense alone.
  - **Goalies:** the top of the list is Shesterkin, Thompson, Vasilevskiy and Sorokin.
- **Lineup value tracks the market.** The lineup is H3's projection, with the previous game's starting goalie. The value difference uses no team history at all.
  - It correlates **0.87 with the market's moneyline logit** (2020-2026).
  - It correlates **0.26 with the goal margin**; the market correlates 0.27.
  - One goal of value is worth **0.90 goals of margin**.
  - It adds a little beyond the market: the partial correlation with the margin is **+0.044 ± 0.020**, with the market held fixed.
- **Skaters carry nearly all of it.**
  - **Previous game's starter:** the goalie column correlates 0.05 with the margin and 0.28 with the market.
  - **Actual starter** (`pv_actual.parquet`, known at puck drop): 0.045 with the margin and 0.30 with the market.
  - The goalie barely moves the margin, but the market prices the starter. Goaltending is mostly noise game to game, and GSAx shrinks hard (below).

## Reproducing

Only the code and this write-up are in git. Everything below is rebuilt from the public sportsdataverse `nhl_pbp_full` releases and the NHL API (H1):

```bash
runitback nhl download --start 2011 --end 2027
runitback nhl events --start 2011 --end 2027
runitback nhl lineups                                   # H3: projected_lineups.parquet
runitback nhl player-values --start 2011 --end 2026     # segments / shots, then player_values/nhl/
python scripts/nhl_player_values.py tune --lam 30 100 300 1000 3000   # the penalty table below
python scripts/nhl_player_values.py goalies             # the goalie shrinkage table
runitback-wager nhl download --first_season 2020 --output_file data/nhl/wager_games.csv \
    --cache_dir data/nhl/nhl_cache                      # the market lines for the report
python scripts/nhl_player_values.py report              # segment checks, top players, pv vs margin / market
```

- **Run time:** `player-values` takes about 1.5 hours on 2 cores.
  - It builds 11.8M segment rows (2011-2026).
  - It fits 446 weeks (2013-2026).
- **Outputs:**

| file | contents |
|---|---|
| `pv.parquet` | leak-free: the previous game's starting goalie |
| `pv_actual.parquet` | the game's actual starter |
| `fits.parquet`, `goalie_fits.parquet` | every weekly fit, so another lineup projection doesn't need a refit (`game_table(fitted=...)`) |
| `usage.json` | minutes per slot and shots per starter, by season |
| `values_latest.csv` | the latest fit's values |

## Setup

- **Shift segments** (`runitback.nhl.player_values.season_segments`).
  - **What a segment is:** a run of events within one period with the same ten skaters and two goalies on ice.
  - **Where it ends:** at the next segment's start, or at the period's last event.
  - **What is kept:** segments with positive length, both goalies in, and 3-5 skaters a side. Empty-net time and the shootout are dropped.
  - **Two rows per segment**, one for each team attacking.
    - The target is that team's expected goals per 60, using the release's `xg` on unblocked shots.
    - Each row is weighted by its minutes.
- **The fit** (on `runitback.core.player_values`).
  - **Coefficients:** an offense coefficient for each attacking skater and a defense coefficient for each defending skater.
  - **Controls:** home offense; strength (n-on-m, 9 codes); how the segment started (on the fly, or an offensive / neutral / defensive zone faceoff); the score (clipped at ±3); season.
  - **Ridge:** penalty 300, with each row weighted by minutes times 0.7 per season back. The window is four seasons plus the current one.
- **S2 core change.** `APMConfig` gains two fields. Both default off, and the NFL's values are identical (test below).
  - `weight`: a per-event weight column, here minutes.
  - `count_weight`: the replacement threshold counts that weight instead of events.
  - The threshold is in minutes because segments got about a third shorter in 2023, when the shift data got finer. An event count would have moved every player's replacement status.
- **Groups F / D** come from the NHL API position.
  - Players with under 100 minutes on ice in the window share a replacement coefficient for their side and group.
  - A player's value is his coefficient over that level; defense is sign-flipped so that higher is better.
- **Goalies: GSAx per unblocked shot faced.**
  - Each shot scores `xg − goal`, minus that season's mean, using only shots before the cut-off.
  - Shots are decayed like the skaters' minutes.
  - Values are shrunk toward the league mean with 6,000 pseudo-shots (`core.shrunk_rates`). Unknown goalies sit at the mean.
  - A replacement level below the mean was worse out of sample (table below).
- **Goals per game through slots.**
  - **Skaters:** value per 60 × the time on ice of the player's slot, divided by 60. The slot is the k-th forward or defenseman of the team by TOI, averaged over the previous season.
    - Usage (2026, from 2025): the top forward plays 20.0 minutes and the 12th 8.2; the top defenseman plays 23.7 and the 6th 13.3.
  - **Goalie:** GSAx per shot × the unblocked shots a starter faced per game the previous season (40.8).
- **Leak-free and weekly.**
  - Every Monday from 2013 on is fit on the segments and shots before it, and each game uses its week's fit.
  - The tests check that changing every segment on or after a cut-off leaves its fit identical.
- **Projected lineups from H3** (`runitback.nhl.data.lineups`, `from_projected`). Each team-game uses H3's projected 12+ forwards and 6+ defensemen, ranked into slots.
  - `pv.parquet` takes H3's `goalie_prev`: the previous game's starter, known before the game.
  - `pv_actual.parquet` takes `goalie_actual`. That is the starter as known at puck drop, which H3 documents as a leak for anything earlier. Use it only for questions that assume the confirmed starter.
  - Without `projected_lineups.parquet`, `player-values` falls back to the previous game's lineup.

### The lineup table (`pv.parquet`)

There is one row per team-game, keyed by `game_id` and `team`.
- `game_id` is a string.
- `team` is the wager model's team key: Atlanta→WPG, Arizona/Phoenix→UTA.
- The columns follow `runitback.wager.nhl.lineup.LINEUP`:

| columns | meaning | sign on the total |
|---|---|---|
| `_pv_0`, `_pv_1` | F, D offense: the projected lineup's goals per game over replacement | + |
| `_pv_2`, `_pv_3` | F, D defense | − |
| `_pv_4` | starting goalie, GSAx per game | − |
| `_pv_5`, `_pv_6`, `_pv_7` | offense / defense / goalie change vs the previous game's lineup | |
| `_pv_8 .. _pv_12` | each group's change vs the mean of the team's last 10 lineups | |

- **`_pv_7` in `pv.parquet`:** it is always 0, because the previous starter is by definition the previous game's goalie.
- **`_pv_7` in `pv_actual.parquet`:** it is non-zero in 48% of team-games (sd 0.057 goals).
- **Registration:** the layout isn't registered on the NHL `Sport` yet. H6 does that when it adds the `pvc` arms.

## Checks

**Segments.** Each season has 59-60 fitted minutes per game. The rest is empty-net time and dropped segments.

| season | games | segments | seconds each | xG per team-game | goals / xG |
|---|---|---|---|---|---|
| 2011 | 1318 | 349,076 | 13.6 | 3.61 | 0.71 |
| 2014 | 1322 | 356,208 | 13.4 | 3.51 | 0.71 |
| 2019 | 1358 | 356,754 | 13.6 | 3.81 | 0.71 |
| 2022 | 1401 | 446,152 | 11.2 | 3.69 | 0.76 |
| 2023 | 1400 | 551,180 | 9.1 | 3.83 | 0.74 |
| 2024 | 1400 | 537,854 | 9.1 | 3.53 | 0.76 |
| 2025 | 1398 | 548,609 | 9.0 | 2.84 | 0.93 |
| 2026 | 1394 | 542,635 | 9.1 | 3.03 | 0.89 |

There are three data facts to know about:
- **2015 (2014-15) is missing** from the release (H1).
- **Segments are shorter from 2023.** The shift data is finer, so there are more, shorter segments for the same minutes. Weighting by minutes makes this neutral.
- **The release's xG changed scale in 2025:** goals per xG went from 0.71 to about 0.9.
  - The per-season controls absorb the level.
  - Goalie GSAx removes each season's mean.
  - Skater values are relative within the window, but a window that spans 2025 mixes the two scales.

**Tuning.** Each setting is fit on everything before a season and predicts that season's segments. The score is the minutes-weighted out-of-sample reduction in squared error vs a constant, × 1e-4, averaged over 2023-2026.

| penalty | window | decay | replacement minutes | gain |
|---|---|---|---|---|
| 30 | 4 | 0.85 | 300 | 1.064 |
| 100 | 4 | 0.85 | 300 | 1.104 |
| 300 | 4 | 0.85 | 300 | 1.124 |
| 1,000 | 4 | 0.85 | 300 | 1.067 |
| 3,000 | 4 | 0.85 | 300 | 0.900 |
| 10,000 | 4 | 0.85 | 300 | 0.611 |
| 300 | 2 / 3 | 0.85 | 300 | 1.061 / 1.099 |
| 300 | 4 | 0.5 / 0.7 | 300 | 1.098 / 1.124 |
| 300 | 4 | 0.7 | 30 | 1.327 (2023-26: 1.716 / 1.494 / 0.952 / 1.147) |
| **300** | **4** | **0.7** | **100** | **1.263 (1.426 / 1.377 / 1.036 / 1.214)** |
| 300 | 4 | 0.7 | 600 | 1.036 |
| 1,000 | 4 | 0.7 | 100 | 1.157 |

- **Why 100 minutes and not 30:** 100 is better in the two latest seasons. 30 wins the mean only on 2023-24, when the new shift data came in.
- **Noise:** a segment is mostly noise, so the comparison is between settings, not an absolute fit.

**Goalie shrinkage.** The score is the out-of-sample MSE reduction of each shot's `xg − goal` (season mean removed) vs zero, × 1e-4, averaged over 2022-2026.

| pseudo-shots | 500 | 1,500 | 3,000 | **6,000** | 12,000 |
|---|---|---|---|---|---|
| gain | −0.050 | 0.011 | 0.027 | **0.029** | 0.023 |

- With a replacement level below the mean for low-shot goalies, every setting was negative.
- A starter's GSAx is real, but it is small next to the per-shot noise. 6,000 shots is about two seasons of a starter's workload.

**Top and bottom of the latest fit** (before 2026-27). Goals per game is the value in the top slot: 20 minutes for a forward, 23.7 for a defenseman, 40.8 shots for a goalie.

| player | group | offense / 60 | defense / 60 | goals per game |
|---|---|---|---|---|
| Connor McDavid | F | +1.49 | +0.36 | +0.61 |
| Patrice Bergeron | F | +0.92 | +0.77 | +0.56 |
| Nathan MacKinnon | F | +1.54 | +0.06 | +0.53 |
| Auston Matthews | F | +1.26 | +0.33 | +0.53 |
| Matthew Tkachuk | F | +1.33 | +0.18 | +0.50 |
| Kirill Kaprizov | F | +1.01 | +0.47 | +0.49 |
| Aleksander Barkov | F | +0.87 | +0.53 | +0.47 |
| Adam Fox | D | +0.89 | +0.82 | +0.67 |
| Zach Werenski | D | +0.73 | +0.52 | +0.50 |
| Lane Hutson | D | +0.75 | +0.48 | +0.48 |
| Quinn Hughes | D | +0.65 | +0.55 | +0.47 |
| Jaccob Slavin | D | +0.14 | +0.95 | +0.43 |
| Travis Boyd (last F) | F | −0.07 | −0.27 | −0.11 |
| Victor Mancini (last D) | D | −0.22 | −0.15 | −0.15 |

- **Bergeron:** he retired in 2023 but is still in the window (2023 is weighted 0.7³). Lineups use only players who are projected to dress, so he no longer appears in any `pv.parquet` row.

| goalie | shots in window | GSAx per shot | per game |
|---|---|---|---|
| Igor Shesterkin | 13,520 | +0.0046 | +0.19 |
| Logan Thompson | 8,980 | +0.0042 | +0.17 |
| Andrei Vasilevskiy | 13,643 | +0.0036 | +0.15 |
| Ilya Sorokin | 12,400 | +0.0035 | +0.14 |
| Connor Hellebuyck | 13,797 | +0.0025 | +0.10 |
| Samuel Ersson (last) | 5,138 | −0.0035 | −0.14 |

**Lineup value vs the game.**
- **What is compared:** each game's projected-lineup value difference (home − away), summed over the five level columns, against the margin of regulation and overtime goals. A shootout game counts as 0.
- **"Margin per value":** the slope of the margin on the difference.
- **Win rates:** the home win rate in the top vs the bottom third of the difference.
- **Previous game's starter** (`pv.parquet`):

| season | games | corr with margin | skaters | goalie | margin per value | home win, top / bottom third |
|---|---|---|---|---|---|---|
| 2013 | 806 | 0.134 | 0.124 | 0.060 | 0.61 | 0.651 / 0.480 |
| 2014 | 1322 | 0.236 | 0.231 | 0.050 | 0.90 | 0.660 / 0.449 |
| 2016 | 1320 | 0.160 | 0.161 | 0.020 | 0.71 | 0.598 / 0.452 |
| 2017 | 1317 | 0.200 | 0.202 | 0.022 | 0.89 | 0.654 / 0.481 |
| 2018 | 1354 | 0.142 | 0.142 | 0.023 | 0.68 | 0.645 / 0.467 |
| 2019 | 1358 | 0.169 | 0.172 | −0.013 | 0.73 | 0.620 / 0.437 |
| 2020 | 1211 | 0.150 | 0.144 | 0.045 | 0.67 | 0.599 / 0.465 |
| 2021 | 952 | 0.281 | 0.281 | 0.051 | 1.08 | 0.672 / 0.406 |
| 2022 | 1400 | 0.310 | 0.305 | 0.121 | 1.14 | 0.679 / 0.355 |
| 2023 | 1400 | 0.322 | 0.322 | 0.086 | 1.01 | 0.660 / 0.353 |
| 2024 | 1400 | 0.287 | 0.283 | 0.106 | 0.99 | 0.657 / 0.405 |
| 2025 | 1397 | 0.208 | 0.206 | 0.074 | 0.87 | 0.689 / 0.451 |
| 2026 | 1394 | 0.183 | 0.189 | 0.001 | 0.83 | 0.617 / 0.460 |
| **all** | **16631** | **0.226** | **0.225** | **0.051** | **0.90** | **0.647 / 0.433** |

**Against the market** (2020-2026: moneylines from `runitback-wager nhl download`, vig removed, as a home-win logit; the total line):

| season | games | pv diff vs market | skaters | goalie | market vs margin | pv diff vs margin | pv total vs total line | pv total vs goals |
|---|---|---|---|---|---|---|---|---|
| 2020 | 1210 | 0.773 | 0.759 | 0.142 | 0.191 | 0.150 | 0.205 | 0.104 |
| 2021 | 951 | 0.859 | 0.850 | 0.221 | 0.282 | 0.278 | 0.123 | 0.048 |
| 2022 | 1400 | 0.888 | 0.878 | 0.320 | 0.319 | 0.310 | 0.505 | 0.108 |
| 2023 | 1400 | 0.926 | 0.915 | 0.327 | 0.325 | 0.322 | 0.287 | 0.070 |
| 2024 | 1398 | 0.880 | 0.868 | 0.334 | 0.293 | 0.286 | 0.117 | 0.053 |
| 2025 | 1397 | 0.866 | 0.852 | 0.338 | 0.247 | 0.208 | 0.444 | 0.069 |
| 2026 | 1394 | 0.821 | 0.814 | 0.200 | 0.200 | 0.183 | 0.339 | 0.089 |
| **all** | **9150** | **0.868** | **0.858** | **0.281** | **0.271** | **0.257** | **0.129** | **0.050** |

- **Partial correlation** of the pv difference with the margin, the market logit held fixed: **+0.044 ± 0.020** (95%, 9,150 games).
- **Actual starter** (`pv_actual.parquet`):
  - margin 0.226 overall, the same as the previous starter;
  - market 0.875 (goalie 0.295);
  - total line 0.134;
  - partial correlation +0.043 ± 0.020.
- **Starter choice:** knowing the actual starter moves the goalie column's market correlation up (0.28 → 0.30, and 0.20 → 0.29 in 2026), but it doesn't help predict the margin.

**What this means:**
- **Side:** the lineup's value is close to a market-quality power rating for the side. The margin correlation is about the market's (0.26 vs 0.27).
- **Totals:** the pv total is weak. It correlates 0.13 with the total line and 0.05 with goals.
  - Expected-goals adjusted plus-minus is built for differentials.
  - The total also depends on the xG scale break and on the season level, which the values remove.
  - H6's wager model has its own totals history, so the `_pv_*` columns mainly carry the side.
- **Comparison with other sports:** the NBA's 0.40 margin correlation and the NFL's 0.40 are higher, but hockey margins are much noisier. The market itself reaches only 0.27.

## Next steps

1. **H6:** register `LINEUP` on the NHL `Sport` and run `prod` / `pbp32` / `pbp32_pvc` with 4 seeds. Then run the swap test: one forward, one defenseman and the starting goalie each go to best regular / worst regular / replacement / out.
   - Use `pv.parquet` for the leak-free arms.
   - `pv_actual.parquet` answers what knowing the confirmed starter is worth.
2. **xG scale:** rescale the release's 2025+ xG to the earlier level (or refit an in-house xG). That would let windows spanning 2025 compare like with like and may help the pv total.
3. **Better values:**
   - a box-score or junior/AHL prior for rookies, who sit at replacement until 100 minutes;
   - separate power-play and penalty-kill values (today strength is only a control).
