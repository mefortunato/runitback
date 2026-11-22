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

**Not done / next:** W2 (`docs/multisport/PLAN.md`): WNBA player values and projected lineups, and the
`prod` / `pbp32` / `pbp32_pvc` arms (4 seeds), built with the per-member, per-season jobs of
`wager-sport.yml`. W2 joins `pbp_32.parquet` on `(game_id, team)`: the teams are already keyed as
`runitback.wager.wnba` keys them.
