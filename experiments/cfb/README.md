# College football: the box-score baseline ported from sportsml

Does `runitback.wager.cfb`, the college football model ported from `mefortunato/sportsml`
(`src/sportsml/cfb/`), reproduce sportsml's production backtest? This box-score transformer is
the baseline every later CFB arm (`pbp32`, `pbp32_qb`; work package F3) is compared against.

**Short answer**
- _Pending the first run of `cfb-baseline.yml`; filled in from its `cfb-baseline-results`
  artifact._
- **Locally the port is exact.** The ported downloader and `process` produce an identical games
  table to sportsml's (7,224 rows, 2021–2024), and a 2-epoch backtest of 2024 with the same seed
  gives identical predictions for all 918 games in both packages.

## Reproducing the results

Only the configuration (`baseline.yaml`) and this write-up are kept in git. The workflow
`.github/workflows/cfb-baseline.yml` rebuilds everything from the public sources in the run
(sportsdataverse schedules and team box scores, ESPN's odds API) and uploads the per-game
predictions and reports as the `cfb-baseline-results` artifact:

| file | what |
|---|---|
| `ablation.md` | every cell on 2021–2025, paired moneyline Brier vs `sportsml_prod` |
| `last3/ablation.md` | the same on 2023–2025, the seasons of sportsml's published table |
| `report.md` | moneyline, spread and total Brier per configuration (`scripts/foundation_report.py`) |
| `parity.md` | per season, how many games `prod` and `sportsml_prod` predict identically (`scripts/port_parity.py`) |
| `<config>/<season>.csv` | per-game walk-forward predictions |

Start it from the Actions tab ("Run workflow"), or push a change to `baseline.yaml` on a
`claude/**` branch. ESPN refuses a runner after a burst of odds requests, so each season's lines
are fetched by its own job; if one is refused, "Re-run failed jobs".

To run one cell locally: `runitback-wager cfb download --last_season 2025`, `runitback-wager cfb
process`, then `python scripts/wager_arm.py prod 2024 --sport cfb --experiment
experiments/cfb/baseline.yaml` (add `--package sportsml` with sportsml installed for its side).

## Setup

- **Walk-forward evaluation, 2021–2025** (the plan's last 5 complete seasons). For each test
  season S the model is fit on earlier seasons only and S is replayed date by date.
- **Games.** FBS games (an FCS opponent is pooled into one team, `FCS`), box scores from 2004 and
  lines from 2013, as in sportsml.
- **One change from sportsml.** Kickoff times are parsed as ISO 8601 explicitly. sportsdataverse's
  2020 schedule mixes `…T04:00Z` and `…T23:30:00.000Z`, and inferring the first row's format
  turned 1,125 of its 1,247 kickoffs into NaT, which crashes the bowl-week numbering (sportsml has
  the same bug). The 2021–2024 games table is unchanged by the fix. Both packages read the same
  raw table, so the comparison below is unaffected.
- **Model.** sportsml's production CFB transformer, pinned in `baseline.yaml`: 28-game history,
  total weight 1, opponent strength, past-line tokens and efficiency rates.
- **Arms.**
  - `prod`, `prod_seed1`: `runitback.wager`, seeds 0 and 1.
  - `sportsml_prod`, `sportsml_prod_seed1`: the same configurations on sportsml's own code (commit
    `b1d1632`), processing the same raw games table itself.
- **Bar.** The port reproduces sportsml when `prod` vs `sportsml_prod` is within the seed noise
  (`sportsml_prod_seed1` vs `sportsml_prod`, about ±0.002 Brier) and `prod`'s 2023–2025 moneyline
  Brier is within that noise of sportsml's published 0.1672 (`market_eff` in
  sportsml's `experiments/cfb/ablation/ablation.md`; Vegas 0.1590 on the same 2,704 games).

## Results

_Pending._

## Next steps

1. F3 (`experiments/cfb/`, arms `prod`, `pbp32`, `pbp32_qb`) uses `prod` from here as its
   baseline.
2. A1 adds college football to the predictions site (`web/src/App.tsx`, the nav and the server).

# College football: quarterback values

Work package F2 (`docs/multisport/PLAN.md`): layer B of the NFL player model for college football.
College play-by-play has no on-field personnel, so only the quarterback gets a value: EPA per
quarterback play, adjusted for the situation and the defense, shrunk, in points per game over
replacement. Each team-game gets a `pv.parquet` row with its projected starter's value and a
starting-QB-change term, for the wager model's `pbp32_qb` arm (F3).

**Short answer**
- **Passers and rushers have ESPN ids.** The play-by-play already carries ESPN athlete ids (the
  same as `espn_cfb_player_box`) for 98% of 2012's passes and over 99.6% from 2014. The plan's
  "names only, no ids" turned out too pessimistic. Matching names to the box scores fills the rest,
  and agrees with ESPN's own id on 99.6–100% of the plays that have both. 0.0–0.7% of passer rows
  stay unmapped (mostly "TEAM").
- **The values look right.** Before the 2025 season the top quarterbacks were Gabriel, Daniels,
  McCord, Nix, Dart and Ward. In the latest fit they are Sayin, Mendoza, Maiava, Mensah and Pavia.
  407 starters started for more than one team, and their values carried over.
- **The starter's value tracks the game.** The projected starters' value difference alone, with no
  team history, correlates **0.55 with the market line** and **0.39 with the final margin** (the
  market: 0.72).
- **QB changes move the result by about the value change.** When the actual starter differs from
  the projection, the margin moves **1.4 ± 0.2 points per point of value difference**. The line moves
  1.2 ± 0.1, so the market prices most of the change (the rest, +0.16 ± 0.12, is not significant).
- **The projection is the weak part.** With no public depth charts or injury history, the projected
  starter is the team's previous game's starter. It is right in 81% of team-games, but only 41% of
  season openers.

## Reproducing

Only the code and this write-up are in git. Everything is rebuilt from the public SportsDataverse
releases (`espn_cfb_pbp`, `espn_cfb_player_box`, `cfb_schedules`):

```bash
runitback cfb download --start 2012 --end 2025      # data/cfb/raw: trimmed play-by-play, player box, schedule
runitback cfb players --start 2012 --end 2025       # data/cfb/processed: qb_plays, team_games, players; prints id coverage
runitback cfb player-values --start 2012 --end 2025 # player_values/cfb: pv.parquet, starters.parquet, usage.json, values_latest.csv
python scripts/cfb_player_values.py tune            # the holdout tables below
python scripts/cfb_player_values.py report          # starters, projection, top QBs, values vs margin and line
```

`player-values` fits 252 weeks (2014–2025) in about a minute on 4 cores. With `--end 2026` it also
covers the season in progress, including upcoming games' projected starters.

## Setup

- **Ids** (`runitback.cfb.data.players.map_ids`). Keep the play-by-play's ESPN id. Otherwise
  normalize the name (case, accents, punctuation, Jr./III, "Last,First", a route word run into
  2012-13 names such as "Ryan Griffin slant"). Then match it to the box scores in this order: the
  same name on the same team in that game; the last name alone if unique on the team that game; the
  same name on the team that season (some box scores miss a player). ESPN's "TEAM" placeholder has
  an athlete id and is dropped.
- **Quarterback plays** (`qb_plays`). Every dropback (pass attempt or sack: cfbfastR's `pass` flag
  on a scrimmage play) is the passer's. Every run by a quarterback (a rusher with 20+ dropbacks that
  season, kneel-downs excluded) is the rusher's. Scrambles are logged as runs and can't be told
  apart from designed runs, so leaving runs out would credit a scrambling quarterback with only his
  failed dropbacks. All games are kept, FCS-only ones included, so FCS quarterbacks have values when
  they transfer up.
- **The fit** (on `runitback.core.player_values`). Each play's EPA (clipped at ±6) on one coefficient
  for its quarterback, plus these controls:
  - down × distance, field position, home / away / neutral, score difference;
  - dropback vs run;
  - **whether the defense is an FBS team**;
  - season;
  - **the defense's team-season**: ridge columns with a quarterback's prior width.

  It is a ridge with penalty 300. Seasons are weighted 0.5 per season back, over four seasons plus
  the current one.
- **Replacement level.** Quarterbacks with under 50 plays in the window share a replacement
  coefficient. A quarterback's value is his coefficient over it.
- **Points per game.** Value per play times the plays a starter makes per game: the league's mean
  in the previous season, about 35 (29 dropbacks plus 6 runs).
- **Starters** (`team_games`). A team-game's starter is the passer with the most of the team's
  first 5 dropbacks, so a trick-play pass on the first snap doesn't count. 95% of starters also lead
  their game in dropbacks (median 30 of the team's 33).
- **Leak-free and weekly.** Every Monday is fit on the plays before it, and each game uses its
  week's fit. The tests check that changing every play on or after a cut-off leaves its fit
  identical, and that projections only use earlier games.

### The lineup table (`pv.parquet`)

One row per team-game of the wager model (every scheduled game with an FBS team, upcoming ones
included). `game_id` is the schedule's, and `team` is the wager key (the school, or `FCS` for a
pooled FCS side). The layout is `runitback.wager.cfb.lineup.LINEUP`; F3 registers it on the CFB
`Sport`:

| column | meaning | sign on the total |
|---|---|---|
| `_pv_0` | the projected starter's points per game over replacement | + |
| `_pv_1` | change vs the previous game's starter (0 with the default projection) | |
| `_pv_2` | change vs the mean of the team's last 7 starters: **the starting-QB-change term** | |

The previous 7 starters are a quarter of the wager model's 28-game history. They are valued with
the same (current) fit, so `_pv_2` is zero for a team with one starter throughout and moves only
when the quarterback changed. `starters.parquet` has, per team-game, the projected, actual and
previous starter and the actual starter's value. The actual starter is for evaluation only (QB-change
subsets in F3) and is never a feature.

**The projected starter is the team's previous game's starter** (`--rule prev_starter`). There is
no public pre-game depth chart, injury report or announced-starter history for college football.
The alternative, the previous game's dropback leader (`--rule prev_leader`), is right more often,
but its values track the game slightly less:

| rule | right: all | season openers | other games | corr(value diff, margin) | corr(value diff, line) |
|---|---|---|---|---|---|
| **previous game's starter** | **81.2%** | **41.4%** | **86.9%** | **0.386** | **0.548** |
| previous game's dropback leader | 82.6% | 41.1% | 88.5% | 0.377 | 0.537 |

(2014–2025 played team-games. In 3,801 the starter changed from the previous game.) `game_table`
takes `starters=` for a better projection, such as a posted depth chart for upcoming games.

## Checks

**Ids** (passer rows with a name; `name match = pbp id`: of the plays with both, how often the name
match finds the play-by-play's own id):

| season | passer rows | ESPN id in the pbp | matched by name | unmapped | name match = pbp id | rusher rows unmapped | QB plays | dropbacks |
|---|---|---|---|---|---|---|---|---|
| 2012 | 56,898 | 98.26% | 1.04% | 0.70% | 100.00% | 0.64% | 67,132 | 56,501 |
| 2013 | 55,722 | 98.60% | 0.81% | 0.59% | 100.00% | 0.37% | 67,297 | 55,387 |
| 2014 | 58,491 | 99.87% | 0.09% | 0.04% | 99.78% | 0.13% | 70,163 | 58,436 |
| 2015 | 58,142 | 99.81% | 0.15% | 0.04% | 99.66% | 0.17% | 70,314 | 58,099 |
| 2016 | 57,561 | 99.96% | 0.02% | 0.02% | 99.93% | 0.14% | 69,613 | 57,537 |
| 2017 | 57,448 | 99.89% | 0.04% | 0.08% | 100.00% | 0.08% | 69,315 | 57,386 |
| 2018 | 58,606 | 99.82% | 0.12% | 0.07% | 99.99% | 0.06% | 71,167 | 58,554 |
| 2019 | 58,886 | 99.90% | 0.10% | 0.00% | 99.97% | 0.02% | 71,312 | 58,868 |
| 2020 | 37,960 | 99.99% | 0.00% | 0.01% | 99.95% | 0.03% | 45,603 | 37,941 |
| 2021 | 55,178 | 99.64% | 0.36% | 0.00% | 99.97% | 0.02% | 65,423 | 55,157 |
| 2022 | 56,765 | 99.97% | 0.00% | 0.03% | 100.00% | 0.02% | 66,850 | 56,734 |
| 2023 | 57,785 | 100.00% | 0.00% | 0.00% | 100.00% | 0.07% | 69,578 | 57,773 |
| 2024 | 60,579 | 100.00% | 0.00% | 0.00% | 100.00% | 0.26% | 73,122 | 60,553 |
| 2025 | 63,178 | 99.97% | 0.00% | 0.03% | 100.00% | 0.10% | 75,395 | 61,768 |
| 2026 (to week 10) | 46,241 | 99.94% | 0.00% | 0.06% | 100.00% | 0.26% | 52,878 | 43,750 |

**Tuning.** The values are used week by week, so each week of the test season is predicted from
the fit on the plays before its Monday (`holdout_weeks`). The table shows the out-of-sample
squared-error reduction per quarterback play vs each week's constant (× 1e-4), the correlation, and
the calibration slope (EPA per point of predicted EPA; 1 is calibrated). 2021–2025, the final setup
unless noted:

| penalty | min plays | decay | defense team-season | mean MSE gain | mean corr | slope 2021 / 2023 / 2025 |
|---|---|---|---|---|---|---|
| 100 | 50 | 0.5 | yes | 62.5 | 0.0529 | 0.73 / 0.84 / 0.73 |
| **300** | **50** | **0.5** | **yes** | **65.9** | **0.0507** | **0.95 / 1.08 / 0.93** |
| 1,000 | 50 | 0.5 | yes | 50.1 | 0.0451 | 1.27 / 1.38 / 1.20 |
| 3,000 | 50 | 0.5 | yes | 33.2 | 0.0399 | 1.83 / 1.90 / 1.66 |
| 300 | 25 | 0.5 | yes | 66.9 | 0.0515 | 1.14 / 1.32 / 1.15 |
| 300 | 150 | 0.5 | yes | 55.2 | 0.0479 | 0.83 / 0.91 / 0.71 |
| 300 | 50 | 0.7 | yes | 63.9 | 0.0502 | 0.89 / 1.00 / 0.89 |
| 300 | 50 | 0.5 | no | 65.9 | 0.0507 | 0.96 / 1.08 / 0.94 |

- Penalty 300 is the best calibrated and has the largest error reduction. 100 correlates slightly
  better but is overspread.
- A threshold of 25 plays is level with 50 on the error but underspread. 150 is clearly worse.
- **The FBS-defense control mattered most.** Without it, FCS quarterbacks topped the list (North
  Dakota State's Cam Miller and Cole Payton) and the 2024–2025 values were overspread (slope about
  0.6). That is the transfer-portal era, with many FCS-to-FBS moves.
- **The defense team-season term ties play by play** (65.9 either way) but helps at game level:
  corr with the line 0.537 vs 0.518, with the margin 0.377 vs 0.364 (previous game's leader rule). So
  it is on.
- **Quarterback runs are in.** Dropbacks alone give corr 0.354 with the margin and 0.512 with the
  line, against 0.386 and 0.548 with runs. The change term's margin effect is also weaker without
  runs (0.12 ± 0.23 vs 0.33 ± 0.20).

**Top of the fits** (points per game over replacement as a starter, about 35 plays a game):

| before 2025 | | latest (through 2025) | |
|---|---|---|---|
| Dillon Gabriel | +12.2 | Julian Sayin | +13.1 |
| Jayden Daniels | +12.0 | Fernando Mendoza | +12.7 |
| Kyle McCord | +12.0 | Jayden Maiava | +12.6 |
| Bo Nix | +11.7 | Darian Mensah | +12.1 |
| Jaxson Dart | +11.7 | Diego Pavia | +11.8 |
| Cam Ward | +11.4 | Drew Mestemaker | +11.5 |
| Will Howard | +11.2 | Devon Dampier | +11.3 |
| Darian Mensah | +10.9 | Carson Beck | +11.3 |

The lowest regulars are about +1 (Kiael Kelly, AJ Hairston). Values are over replacement (under 50
plays), which in college is a long way down, so even weak regulars are well above it. Small samples
can rank high (Sayin, 439 weighted plays). The window keeps players who have left (Daniels, Nix)
until it ages them out; they never appear in a lineup.

**Value vs the game** (each game's projected starters' value difference, home − away, previous
game's starter rule):

| season | games | corr with margin | corr with market line | market vs margin | margin pts per value pt |
|---|---|---|---|---|---|
| 2014 | 846 | 0.361 | 0.529 | 0.705 | 2.23 |
| 2015 | 855 | 0.379 | 0.515 | 0.718 | 2.45 |
| 2016 | 851 | 0.345 | 0.540 | 0.697 | 2.32 |
| 2017 | 867 | 0.359 | 0.528 | 0.709 | 2.37 |
| 2018 | 877 | 0.385 | 0.533 | 0.725 | 2.40 |
| 2019 | 886 | 0.449 | 0.608 | 0.741 | 2.83 |
| 2020 | 562 | 0.390 | 0.568 | 0.675 | 2.13 |
| 2021 | 837 | 0.371 | 0.551 | 0.716 | 2.41 |
| 2022 | 854 | 0.366 | 0.569 | 0.709 | 2.19 |
| 2023 | 899 | 0.449 | 0.577 | 0.715 | 2.73 |
| 2024 | 900 | 0.384 | 0.560 | 0.714 | 2.40 |
| 2025 | 932 | 0.383 | 0.508 | 0.738 | 2.55 |
| **all** | **10166** | **0.386** | **0.548** | **0.715** | **2.43** |

The quarterback carries much of what the market knows, but less than the NFL's or NBA's whole-lineup
values (0.86 / 0.83 with the line). Here the value also stands in for the rest of the team: a good
quarterback plays for a good team. That is why the raw slope is 2.4 margin points per value point.

**Quarterback changes.** The causal question for the wager model is whether a *change* of starter
moves the result by the value change. So the actual starters' value is split into the projection
plus the surprise (actual minus projected, nonzero in the 32% of games where a projection was wrong;
a starter with no value of his own counts at replacement) and regressed by OLS (home − away,
± 1.96 SE):

| outcome | games | per projected value pt | per surprise value pt |
|---|---|---|---|
| market line | 10,148 | 2.64 ± 0.07 | 1.24 ± 0.10 |
| final margin | 10,160 | 2.64 ± 0.11 | 1.40 ± 0.16 |

On the 3,298 games with a wrong projection alone, the surprise is worth 1.30 ± 0.12 points of line
and 1.47 ± 0.17 of margin. Given the line, the surprise adds +0.16 ± 0.12 to the margin: the market
prices the change almost fully. In the wager setting the team's history plays the part of the
projection, so the leak-free change term `_pv_2` was also checked with the recent starters' value
held fixed: **+0.34 ± 0.13 line points and +0.33 ± 0.20 margin points per point of `_pv_2`**.
It is positive and in the right direction, but smaller than the surprise effect. The projection
misses most changes before they happen, so `_pv_2` mostly records changes already a few games old,
which the line has partly absorbed.

(An earlier version of this check used team-season fixed effects and gave a *negative* margin
slope. That is the dynamic-panel bias: in-season values rise after good games, which also raise
the team-season mean. The OLS split above has no fixed effects.)

## Next steps

1. **F3:** register `LINEUP` on the CFB `Sport` and run `prod` / `pbp32` / `pbp32_qb` with 4 seeds,
   plus the QB swap test. Use `starters.parquet` for the QB-change subsets.
2. **Better projections.** The biggest gap is season openers (41%) and in-season changes, which the
   previous game can't see.
   - CollegeFootballData's rosters, returning production and transfer portal (the plan's optional
     part) would fix most openers. They need a `CFBD_API_KEY`, which this session didn't have, so they
     were not added.
   - ESPN's depth charts and injuries for upcoming games would help the live predictions.
3. **Better values:** a recruiting or prior-level prior for quarterbacks under 50 plays (today they
   sit at replacement), and skill-player values (receivers, running backs) from the mapped
   receiver / rusher ids.
