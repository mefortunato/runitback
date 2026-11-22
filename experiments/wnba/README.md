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
- **The six folds (2022–2027) trained in Actions and are leak-free** (one job each, 12–26 minutes). Their embeddings
  carry team strength (0.35–0.45 correlation with the margin) but add nothing beyond Elo, as for the NFL and
  college. Every fold kept its last epoch, so more epochs is worth a try before W2.

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

Recipe: the NBA's (`runitback.nba.foundation.TRAIN`: d_model 192, 4 layers, 6 epochs), validated on the last
40% of season S-1 by date, trained on every game before that since 2006 (about 4,500 games for the 2026 fold).
Seasons are calendar years. 2022–2026 are the wager backtest's folds, and 2027 is the fold for next season's
games. The leak checks (validated on S-1, no team-season of S or later, PCA on games before S) are in
`scripts/nba_foundation_report.py` and the unit tests.

Run 45019041335 (`wnba-foundation.yml`, results artifact `wnba-foundation-results`, folds `wnba-foundation-models`):
every job passed, 12–26 minutes per fold on a 4-core runner (about 30 minutes of walltime for all six).

| fold | validated on | trained on | leak-free | kept epoch | val loss (type / result / elapsed) | games before / in S | PCA 32 explains | corr pbp32 | corr Elo | corr both |
|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | 2021 | 2006–2021 | yes | 5 of 0–5 | 4.427 (1.166 / 0.804 / 2.458) | 3,526 / 240 | 72.0% | 0.379 | 0.458 | 0.420 |
| 2023 | 2022 | 2006–2022 | yes | 5 | 4.387 (1.161 / 0.804 / 2.423) | 3,766 / 261 | 73.6% | 0.351 | 0.427 | 0.408 |
| 2024 | 2023 | 2006–2023 | yes | 5 | 4.360 (1.163 / 0.800 / 2.397) | 4,027 / 263 | 73.5% | 0.455 | 0.482 | 0.482 |
| 2025 | 2024 | 2006–2024 | yes | 5 | 4.384 (1.168 / 0.804 / 2.412) | 4,290 / 311 | 72.7% | 0.374 | 0.395 | 0.393 |
| 2026 | 2025 | 2006–2025 | yes | 5 | 4.397 (1.177 / 0.806 / 2.415) | 4,601 / 353 | 72.6% | 0.380 | 0.442 | 0.421 |
| 2027 | 2026 | 2006–2026 | yes | 5 | 4.273 (1.155 / 0.795 / 2.324) | 4,954 / 0 | 73.9% | — | — | — |

`corr_*`: the correlation with the home margin in S of a ridge fit before S on the teams' previous-10-game means
(`scripts/nba_foundation_report.py`; a sanity check, not the wager experiment).

- **The folds are leak-free.** Each was validated on S-1, has no team-season of S or later, and its PCA was fit on
  games before S.
- **The embedding carries team strength, but not beyond Elo.** Its correlation with the margin (0.35–0.45) is a
  little below the pre-game Elo's (0.39–0.48), and adding it to Elo does not help. That is the NFL's and C2's
  finding too: layer A mostly carries team strength. Whether `pbp32` helps the wager model is W2's question.
- **Every fold kept its last epoch.** Validation loss was still falling at epoch 6, so the NBA recipe may be short
  for a league with a sixth of the NBA's games. More epochs is a cheap thing to try (`--epochs`; a fold takes about
  20 minutes) before W2 settles on these folds. The recipe is left as it is here.

**Not done / next:** W2 (`docs/multisport/PLAN.md`): WNBA player values and projected lineups, and the
`prod` / `pbp32` / `pbp32_pvc` arms (4 seeds), built with the per-member, per-season jobs of
`wager-sport.yml`. W2 joins `pbp_32.parquet` on `(game_id, team)`: the teams are already keyed as
`runitback.wager.wnba` keys them.
