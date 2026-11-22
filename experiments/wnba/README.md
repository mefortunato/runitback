# WNBA

## Engine and foundation folds (W1)

Work package W1 (`docs/multisport/PLAN.md`) makes the WNBA another league of the basketball engine
(`runitback.nba.league.WNBA`) and trains the `pbp32` generator folds the wager model (W2) will read. Rules and
commands: README, "WNBA". Code: `nba/league.py` (the rules), `nba/engine/rules.py`, `nba/data/wnba.py` (ESPN
rows to events), `nba/foundation.py`.

**Short answer**
- **The engine reproduces real WNBA games.** Applying each real outcome to the real state gives the next state
  (period, phase, possession, free throws left, score, team fouls) on **99.50% of 1.20M transitions** (4,954
  games, 2006–2026). 4,954 of the 4,972 games between franchises pass the 90% bar, and every kept game has a
  winner. The per-season agreement is 99.2–99.8%, except 2012 (97.7%; its September games miss rows).
- **The bonus rules come from the data, and they are the NBA's.** The 5th team foul of a quarter is two shots, the
  4th of an overtime, counts reset every period, and inside a period's last two minutes a team's 2nd foul is two
  shots. Offensive fouls don't count. The NBA engine does not track the bonus at all. The WNBA engine does,
  as the college engine (C2) does: a foul with no free throws is an event (`fts_0`), the model sees both teams'
  counts, and the engine never lets a team in the penalty foul without free throws.
- **What is left is the feed, not the rules.** Of the 6,052 mismatched transitions, the largest groups are a team
  acting twice in a row after a made basket (1,340) or a turnover (1,287), where a row is missing between them.
  Quarter starts follow the NBA rule (the loser of the opening tip starts the 2nd and 3rd quarters) with 100
  misses in about 15,000 quarter starts.
- **The six folds (2022–2027) trained in Actions and are leak-free** (one job each). Their embeddings carry team
  strength (0.32–0.44 correlation with the margin) but add nothing beyond Elo, as for the NFL and college.
- **The WNBA folds train 12 epochs, not the NBA's 6.** 12 epochs lowered validation loss on all six folds (by 0.017
  on average). 20 epochs added nothing, and the embeddings' team-strength signal did not change.

### The bonus, measured

For each non-shooting defensive foul (personal, loose-ball, take; offensive fouls excluded), the share followed by
free throws at the same clock, by the fouling team's count in the period:

| team foul # | regulation, before the last 2:00 (2019–2026) | regulation, 2nd+ foul in the last 2:00 | overtime, before the last 2:00 (2016–2026) |
|---|---|---|---|
| 1 | 1.3% (6,527) | — | 3% (95) |
| 2 | 1.2% (5,775) | 100% (65) | 4% (70) |
| 3 | 1.4% (4,793) | 99.6% (244) | 3% (29) |
| 4 | 2.3% (3,199) | 99.3% (449) | **100% (10)** |
| 5 | **99.3% (1,760)** | 100% (521) | 100% (1) |
| 6 | 100% (797) | 99.8% (438) | — |

The 1–2% before the limit are fouls that give free throws anyway: flagrant, clear-path, take and away-from-play
fouls, which the reader turns into a foul plus technical-style shots that keep the ball. Counting offensive fouls
too would make 2.5% of the "5th" fouls penalty fouls (32 of 1,290), so they are not team fouls. Before 2019 the
feed is noisier: 8% of 4th fouls and 93% of 5th fouls come with free throws (2016–2018). That fits fouls missing
from the rows, which only make the engine's count low. A low count is safe: the engine never forbids a real
outcome because of it.

### Replay agreement by season

| season | games | agreement | | season | games | agreement |
|---|---|---|---|---|---|---|
| 2006 | 254 | 99.38% | | 2017 | 219 | 99.78% |
| 2007 | 239 | 99.68% | | 2018 | 218 | 99.63% |
| 2008 | 256 | 99.77% | | 2019 | 218 | 99.44% |
| 2009 | 239 | 99.81% | | 2020 | 145 | 99.31% |
| 2010 | 220 | 99.83% | | 2021 | 210 | 99.38% |
| 2011 | 222 | 99.24% | | 2022 | 240 | 99.42% |
| 2012 | 206 | 97.70% | | 2023 | 261 | 99.50% |
| 2013 | 219 | 99.25% | | 2024 | 263 | 99.53% |
| 2014 | 221 | 99.71% | | 2025 | 311 | 99.57% |
| 2015 | 224 | 99.80% | | 2026 | 353 | 99.63% |
| 2016 | 216 | 99.73% | | **all** | **4,954** | **99.50%** |

Commonest mismatches (all seasons): made 2 then the same team acts (1,340), turnover then the same team (1,287),
made free throw then the same team (687), made 3 then the same team (449), a foul with no free throws then the
other team (340). They look like rows missing from the feed. Some of the free-throw ones may instead be kept
balls the reader doesn't recognize (a foul type the feed doesn't name). 2012's 14 dropped games (of 220) are mostly September
games with sparse rows: misses with no rebound row, and rebounds credited to the shooting team.

**How the rows become events** (what differs from the NBA's reader)
- The feed has the NBA's format ("Free Throw - 1 of 2"), so trips are explicit. A foul whose free throws follow
  is a single `foul` event with that trip. A shooting foul is part of the shot it came with (an and-one or a
  missed shot that drew a foul).
- A jump ball's row carries the first-named jumper's team, not the team that gained possession, so a jump ball is
  credited to the team that acts next. This one fix moved agreement from 99.2% to 99.6% on the seasons it was
  tried on: held balls, and the opening tip that decides who starts the 2nd–4th quarters.
- A foul by the team with the ball and no free throws (loose-ball, inbound) loses the ball: a turnover.
- 20-second, full and (2026) reset timeouts are timeouts. The engine allows 7 a game, 5 in the 4th quarter and
  2 more per overtime, which drops 10 of 39,264 real ones.
- Pre-game ratings use the college settings (EWMA α = 0.10, carry 0.70 between seasons). Checked on seasons
  before the backtest (fit 2008–2015, scored 2016–2021), the settings tried are all within 0.004 in margin
  correlation (0.404–0.408), and these are the best.

### Foundation folds

Recipe: the NBA's model (`runitback.nba.foundation.TRAIN`: d_model 192, 4 layers) trained for **12 epochs**
(`WNBA_EPOCHS`; the NBA trains 6, see "Epochs" below). Each fold is validated on the last 40% of season S-1 by date
and trained on every game before that since 2006 (about 4,500 games for the 2026 fold). Seasons are calendar
years. 2022–2026 are the wager backtest's folds, and 2027 is the fold for next season's games. The leak checks
(validated on S-1, no team-season of S or later, PCA on games before S) are in `scripts/nba_foundation_report.py`
and the unit tests.

The 12-epoch folds, from run 45022405246 (`wnba-foundation.yml`, results artifact `wnba-foundation-results`, folds
`wnba-foundation-models` under `foundation-e12/`), 26–58 minutes per fold on a 4-core runner:

| fold | validated on | trained on | leak-free | best epoch | val loss (type / result / elapsed) | games before / in S | PCA 32 explains | corr pbp32 | corr Elo | corr both |
|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | 2021 | 2006–2021 | yes | 7 | 4.407 (1.159 / 0.803 / 2.445) | 3,526 / 240 | 70.5% | 0.408 | 0.458 | 0.441 |
| 2023 | 2022 | 2006–2022 | yes | 5 | 4.375 (1.156 / 0.804 / 2.415) | 3,766 / 261 | 73.5% | 0.318 | 0.427 | 0.390 |
| 2024 | 2023 | 2006–2023 | yes | 9 | 4.340 (1.154 / 0.799 / 2.386) | 4,027 / 263 | 69.5% | 0.440 | 0.482 | 0.478 |
| 2025 | 2024 | 2006–2024 | yes | 10 | 4.366 (1.159 / 0.804 / 2.404) | 4,290 / 311 | 68.2% | 0.347 | 0.395 | 0.378 |
| 2026 | 2025 | 2006–2025 | yes | 7 | 4.381 (1.169 / 0.804 / 2.408) | 4,601 / 353 | 70.7% | 0.399 | 0.442 | 0.431 |
| 2027 | 2026 | 2006–2026 | yes | 7 | 4.256 (1.147 / 0.794 / 2.316) | 4,954 / 0 | 70.7% | — | — | — |

`corr_*`: the correlation with the home margin in S of a ridge fit before S on the teams' previous-10-game means
(`scripts/nba_foundation_report.py`; a sanity check, not the wager experiment). Best epoch counts from 0.

- **The folds are leak-free.** Each was validated on S-1, has no team-season of S or later, and its PCA was fit on
  games before S.
- **The embedding carries team strength, but not beyond Elo.** Its correlation with the margin (0.32–0.44) is
  below the pre-game Elo's (0.39–0.48), and adding it to Elo does not help. That is the NFL's and C2's finding too:
  layer A mostly carries team strength. Whether `pbp32` helps the wager model is W2's question.

#### Epochs

The first fold run (45019041335, the NBA's 6 epochs) kept the last epoch on every fold: validation loss was still
falling. Run 45022405246 trained every fold three ways in one run: the 6-epoch recipe again, 12 and 20 epochs.
Validation loss is on the same held-out games, so it compares directly; the cosine learning-rate schedule spans each
run's epochs, so 12 epochs is a different run, not 6 more. The best epoch is the one validation picked (counting
from 0); a checkpoint's history ends there.

| fold | 6 epochs: val loss / best / corr pbp32 | 12 epochs | 20 epochs | 12 − 6, val loss |
|---|---|---|---|---|
| 2022 | 4.4272 / 5 / 0.379 | 4.4071 / 7 / 0.408 | 4.4068 / 7 / 0.393 | −0.020 |
| 2023 | 4.3868 / 5 / 0.351 | 4.3751 / 5 / 0.318 | 4.3766 / 5 / 0.323 | −0.012 |
| 2024 | 4.3599 / 5 / 0.455 | 4.3397 / 9 / 0.440 | 4.3416 / 8 / 0.444 | −0.020 |
| 2025 | 4.3838 / 5 / 0.374 | 4.3663 / 10 / 0.347 | 4.3670 / 6 / 0.373 | −0.018 |
| 2026 | 4.3971 / 5 / 0.380 | 4.3808 / 7 / 0.399 | 4.3776 / 7 / 0.407 | −0.016 |
| 2027 | 4.2731 / 5 / — | 4.2557 / 7 / — | 4.2583 / 7 / — | −0.017 |

- **12 epochs is a better generator on every fold**, by 0.012–0.020 nats per event (mean 0.017). The 6-epoch
  rerun reproduced the first run's numbers exactly, so with the same seed the comparison has no run-to-run noise.
  Seed-to-seed noise was not measured; all six folds moving the same way is the evidence.
- **20 epochs adds nothing**: within ±0.003 of 12, with best epochs 5–8. Past about epoch 8 the model overfits.
- **The embedding does not get stronger.** Against 6 epochs, its team-strength correlation is up on two folds and
  down on three (mean −0.005), and with Elo it is unchanged (mean −0.001).
- **Cost:** a 12-epoch fold takes 26–58 minutes on a 4-core runner (6 epochs: 15–20; 20 epochs: 38–83).

So the WNBA folds train 12 epochs (`runitback.nba.foundation.WNBA_EPOCHS`): a better event model at about twice the
time, and embeddings at least as good. `runitback wnba foundation --epochs N` and the workflows' `epochs` input
(a list, 0 = the recipe) train other counts.

**Next:** W2, below: WNBA player values and projected lineups, and the `prod` / `pbp32` / `pbp32_pvc` arms.

## Player values, projected lineups and the wager model on them (W2)

Work package W2 (`docs/multisport/PLAN.md`): layers B and C of the NFL player model for the WNBA, with the
NBA's pipelines (N1 on-court fives, N3 projected rotations, N2 possession-level values) and the WNBA's
settings. The wager model is then tested on the foundation (`pbp32`, W1) and the player values (`pvc`), as the
NBA's N5.

**Short answer**
- **The values look right, and they carry most of what the market knows.** The projected rotations' value
  difference alone, with no team history, correlates **0.88 with the market spread** and **0.45 with the
  final margin** (the market: 0.48) over 2018–2026. One point of value is worth **1.03 points of margin**. That
  is the NBA's pattern (0.83 / 0.40 / 0.99), a little stronger. The top of the latest fit: Burton, A'ja Wilson,
  Stewart, Young, Fiebich, Reese.
- **Less shrinkage than the NBA's penalty, which is still stronger shrinkage.** On 2017–2021 (the seasons
  before the backtest) the best ridge penalty is **2,000** (NBA: 5,000) with **75 minutes** for a player's own
  coefficient. The penalty works on summed squares, and a WNBA season has about 37,000 possessions (a seventh of
  the NBA's), so per possession it shrinks about **2.8× harder** than the NBA's. The plan's "stronger shrinkage"
  holds in that sense; a larger penalty (5,000 to 80,000) predicted the next season's possessions worse.
- **The box-score prior is not done** (optional in the plan): rookies and players under 75 minutes sit at
  replacement level, as in the NBA.
- **Wager performance and the swap test: pending the first run of `WNBA wager experiment`**
  (`.github/workflows/wnba-wager-experiment.yml`). The tables below are filled from its `wnba-wager-results`
  artifact.

### Reproducing

Only the code and this write-up are in git. Everything is rebuilt from the public SportsDataverse releases
(ESPN WNBA play-by-play and player box scores, 2006 on):

```bash
runitback wnba download --start 2006                 # play-by-play (W1)
runitback wnba players --start 2006                  # player box scores; on_court / player_games / players
runitback wnba lineups                               # projected_lineups.parquet
runitback wnba player-values --lineups data/wnba/processed/projected_lineups.parquet   # player_values/wnba/
python scripts/nba_player_values.py --league wnba tune --seasons 2017 2018 2019 2020 2021
python scripts/nba_player_values.py --league wnba report
```

`players` takes about a minute and `player-values` (373 weekly fits, 2008–2026) about a minute on 4 cores.

### Setup

What is the NBA's, and what differs (`runitback.nba.data.players.WNBA_PLAYERS`,
`runitback.nba.player_values.WNBA_VALUES`):

- **On-court fives** (N1's reconstruction, unchanged): box-score starters, both ids of every substitution,
  period starts inferred from who appears first, look-ahead repairs of dropped substitutions. The WNBA feed also
  has lowercase `ejection` rows and the 20-second / reset timeouts, which are not evidence that a player is on the
  floor.
- **Teams are keyed by ESPN team id**, as the franchise's latest abbreviation (`WNBA_TEAM_IDS`), which is how
  `runitback.wager.wnba` keys them: `pv.parquet` joins the wager model's rows on `(game_id, team)` with no
  re-keying (100% of the games from 2009 on; 97–99% in 2008, 2010, 2012, 2016 and 2018, where single games have no
  player box score). All-Star and exhibition teams have no key, so their games are dropped.
- **Projected rotation** (N3's rule): the game's box-score roster minus players with an injury did-not-play
  reason, ranked by recency-weighted minutes over their last 10 available games, scaled to **200 team minutes**
  (40-minute games; no player over 38). **The documented leak is the NBA's:** there is no history of WNBA injury
  reports, so a player ruled out after the line was set counts as known at tip-off. ESPN's did-not-play reasons
  start in 2012–2013, so earlier seasons have no players ruled out (the backtest's are 2022–2026).
- **Possessions and the fit** (N2's, unchanged): points per possession on the five offensive and five defensive
  players, controls for home offense, how the possession started and season; seasons weighted 0.7 per season back
  over four seasons plus the current one; points clipped at 4. Groups G / F / C from the ESPN position (202
  players with no position, all retired by 2011, go to F). The usage per rotation slot is the previous season's.
- **The WNBA's settings:** penalty 2,000 (NBA 5,000), own coefficient from 75 minutes on court in the window
  (NBA 250), 2.0 possessions per side per minute (NBA 100/48). `pv.parquet` rows start in 2008.
- **Leak-free and weekly**, as the NBA's: every Monday is fit on the possessions before it, and each game uses its
  week's fit.

### Checks

**On-court fives** (2006–2026, 1.93M events): both sides have exactly five on the floor for **99.87%** of events
(every season ≥ 99.28%; the lowest are 2018–2019, whose substitutions are sometimes logged in halves, as in the
NBA). Reconstructed minutes are within 1 of the box score for 92.2% of player-games (MAE 0.52), and the on-court
plus-minus matches the box score's exactly for 82.1%.

**Possessions** (790,000): 76.5–81.9 per team-game and 0.93–1.08 points each, rising over the years as the
league's offense did. Summed per team-game, possession points match the box score exactly in 71.9% of team-games
and within 3 points in 98.7% (MAE 0.45).

**Projected rotations vs what happened** (`runitback.nba.data.lineups.evaluate`), mean of 2022–2026:

| | projection | the previous game as played |
|---|---|---|
| starters right | 91.8% | 91.6% |
| all five starters right | 64.3% | 68.2% |
| minutes MAE (per player-game) | **4.53** | 5.73 |
| regulars who missed the game and were not counted on | **84.7%** | — |

Unlike the NBA's, the projection's starters are no better than the previous game's (the NBA's beat them every
season). Its minutes are, by 1.2 per player, and it rules out 85% of the regulars who then missed the game.

**Tuning** (fit on everything before a season, predict that season's possessions from the ten players' values;
out-of-sample squared-error reduction per possession vs a constant, × 1e-4; window 4, decay 0.7):

| penalty | own coefficient from | 2017 | 2018 | 2019 | 2020 | 2021 | mean |
|---|---|---|---|---|---|---|---|
| 1,000 | 75 min | 32.2 | 11.5 | 17.2 | 41.7 | 18.4 | 24.2 |
| **2,000** | **75 min** | **33.4** | **19.3** | **22.3** | **40.0** | **21.9** | **27.4** |
| 2,000 | 150 min | 33.2 | 10.1 | 20.9 | 36.5 | 20.7 | 24.3 |
| 2,000 | 400 min | 35.6 | 10.1 | 21.2 | 38.4 | 22.7 | 25.6 |
| 5,000 | 75 min | 26.1 | 17.8 | 19.4 | 28.7 | 18.4 | 22.1 |
| 10,000 | 75 min | 19.0 | 13.5 | 14.6 | 19.7 | 13.5 | 16.0 |
| 40,000 | 150 min | 8.0 | 5.5 | 6.3 | 8.1 | 5.3 | 6.7 |

Also tried, at penalty 1,000–2,000: windows of 2, 3, 5 and 6 seasons (within 0.9 of 4), decay 0.5 and 0.85
(both worse), 50 minutes (27.4, as 75). The tuning seasons are before the backtest on purpose, so the backtest
did not pick the settings. As for the NBA, a single possession is mostly noise (the prediction correlates
0.03–0.06 with its points), so the comparison is between settings.

**Top and bottom of the latest fit** (before 2027; points per game over replacement as a starter, who plays
67 possessions a side):

| player | group | offense | defense | total |
|---|---|---|---|---|
| Veronica Burton | G | +4.1 | +3.1 | +7.2 |
| A'ja Wilson | C | +4.6 | +2.5 | +7.1 |
| Breanna Stewart | F | +4.8 | +1.4 | +6.2 |
| Jackie Young | G | +4.7 | +1.3 | +5.9 |
| Leonie Fiebich | F | +2.8 | +3.0 | +5.7 |
| Angel Reese | F | +2.9 | +2.5 | +5.4 |
| Kayla McBride | G | +4.5 | +0.9 | +5.4 |
| Chelsea Gray | G | +4.4 | +0.9 | +5.3 |
| Alyssa Thomas | F | +3.4 | +1.9 | +5.3 |
| … Kia Nurse | G | +1.1 | −2.0 | −0.9 |
| Dorka Juhasz (last) | F | −0.3 | −0.7 | −0.9 |

**Lineup value vs the game** (each game's projected-rotation value difference, home − away, summed over the six
level columns; the market spread is the wager table's closing line, from 2018):

| season | games | corr with margin | corr with market spread | market vs margin | margin pts per value pt |
|---|---|---|---|---|---|
| 2018 | 220 | 0.390 | 0.905 | 0.413 | 0.93 |
| 2019 | 220 | 0.344 | 0.820 | 0.382 | 0.89 |
| 2020 | 147 | 0.591 | 0.902 | 0.556 | 1.22 |
| 2021 | 209 | 0.426 | 0.846 | 0.470 | 1.09 |
| 2022 | 240 | 0.482 | 0.845 | 0.499 | 1.21 |
| 2023 | 261 | 0.485 | 0.925 | 0.526 | 1.25 |
| 2024 | 262 | 0.504 | 0.904 | 0.540 | 1.04 |
| 2025 | 310 | 0.439 | 0.902 | 0.467 | 0.85 |
| 2026 | 352 | 0.469 | 0.911 | 0.493 | 1.05 |
| **2018–2026** | **2,221** | **0.454** | **0.884** | **0.484** | **1.03** |

Over 2008–2026 (4,483 games) the correlation with the margin is 0.41. The home edge is not in the difference
(+0.01 points on average; the margin's is +2.5).

### The wager model on the foundation and the player values

**Setup** (as the NBA's N5):

- **Walk-forward, 2022–2026** (5 folds, 240–352 games each). For each test season S the wager model, the
  foundation fold and its PCA see only games before S; the player values are refit weekly on possessions before
  each game's Monday.
- **Arms** (`experiments/wnba/wager.yaml`): `prod` (the S4a box-score production model, the baseline), `pbp32`
  (+ fold S's frozen 32-dim game embeddings, 12-epoch folds from W1), `pbp32_pvc` (+ each position group's change
  in projected-rotation value vs the team's last 10 rotations, through sign-constrained additive terms on margin
  and total), `prod_pvc` (the values without the foundation: the NBA's promoted arm), and `_e4` versions (4 seeds
  averaged). Promotion compares `_e4` with `_e4`.
- **Ensembles in parallel jobs.** Each `_e4` cell trains its four members in four jobs
  (`scripts/wager_arm.py --member i`, `runitback.wager.web.split.arm_member`) and a fifth assembles them and
  replays the season (`--members`). That gives the same CSV as training the four in one job
  (`tests/wager/test_split.py`), as Train predictions' per-member jobs do (`wager-sport.yml`).
- **Swap test** (`scripts/nba_lineup_swap.py --sport wnba`): `pbp32_pvc_e4` fit on the seasons before 2026 (its
  2026 members), then every 2026 game re-predicted with one change to the home team's projected rotation: the
  top-ranked G, F or C replaced by the group's best / worst regular (at least 800 minutes on court in the window),
  by a replacement-level player, or ruled out.

**Wager performance:** paired Brier difference (negative is better) ± 95% CI half-width. Pending.

**Swap test:** pending.

**Reproducing:** run `WNBA wager experiment` from the Actions tab, or push a change to
`experiments/wnba/wager.yaml` or the workflow on a `claude/**` branch. It rebuilds the games with lines and the
foundation folds (`wnba-experiment-inputs.yml` with `lines: true`), and in its `values` job the players, lineups
and weekly values. One cell locally: `python scripts/wager_arm.py pbp32_pvc_e4 2025 --sport wnba --experiment
experiments/wnba/wager.yaml` (with `runs/foundation/fold2025/` and `player_values/wnba/pv.parquet` in place).
