# NBA: player values

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
