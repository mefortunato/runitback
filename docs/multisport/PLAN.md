# Multi-sport player model: plan and work packages

The goal is to bring the NFL player-based wager model to the other seven sports the wager package
covers (ported from sportsml): NBA, WNBA, men's college basketball (`cbb`), women's college
basketball (`wcbb`), NHL, MLB and college football (`cfb`).

The NFL model being ported is on `main` (merged in mefortunato/runitback#18). Its write-up is
`experiments/nfl/README.md`. Read it before
starting any work package: it records what worked, what didn't, and why.

## 1. The architecture being ported

The NFL production model (`runitback.wager.nfl`) has three layers. Each work package below builds one
of them for one sport.

| Layer | NFL implementation | What a sport needs |
|---|---|---|
| **A. Foundation embeddings (`pbp32`)** | The play-by-play generator runs teacher-forced over each team-game. Hidden states are mean-pooled over offensive and defensive plays, then PCA reduces them to 32 dimensions, fit on games before the fold's test season. The result joins the wager transformer's history tokens. There is one generator per test season ("fold"). Files: `nfl/foundation.py`, `nfl/embed.py` | A play-by-play generator for the sport (engine, vocab, preprocess, model) |
| **B. Player values** | Ridge adjusted plus-minus on play EPA with the 22 players on the field. Refit weekly on earlier plays only. Players under 150 plays share a replacement level. Values are converted to points per game through depth-slot usage. File: `nfl/player_values.py` | Per-event on-field personnel (or a reconstruction of it) and a per-event value target |
| **C. Projected lineup → `pvc`** | The pre-game lineup is the depth chart minus players ruled out or not on the active roster. Each group's value change vs the team's last 8 lineups enters margin and total as sign-constrained additive terms (`PairHead`). This is also the arithmetic behind the lineup editor. Files: `nfl/lineup_features.py`, `wager/models/transformer/data.py`, `wager/web/lineups.py` | A pre-game availability history, position groups, and total signs per group |

The key lesson from the NFL experiments: the generator's player embeddings did **not** rank players.
Personnel signal came through the plus-minus values (layer B → C). Layer A mainly carries team
strength. **For a sport where layer A is expensive (a new simulator) and B is cheap, build B/C
first.**

## 2. Data availability (checked 2026-11-09)

| Sport | Play-by-play | On-field personnel | Player box | Pre-game lineup / availability | runitback simulator |
|---|---|---|---|---|---|
| NHL | `nhl_pbp_full` (sportsdataverse releases), 2010+, ~1.1M events/season, includes `xg` | **Yes**: `home_on_1..7_id`, `away_on_1..7_id`, goalie ids on every event; shift charts at `api.nhle.com/stats/rest/en/shiftcharts` back to 2010 | NHL stats API | Starting goalie: no historical announced-starter source found. ESPN `/injuries` is current-only | None |
| NBA | `espn_nba_pbp`, 2002+, `athlete_id_1..3` | Rebuildable: substitution events carry both ids (all 2025 games, ~57/game); starters from player box | `espn_nba_player_boxscores`: `starter`, `did_not_play`, `reason`, `minutes` | No injury-report history; DNP `reason` as a stand-in; ESPN `/injuries` live | `runitback.nba` (event-level, no players) |
| WNBA | `espn_wnba_pbp`, 2006+ | Rebuildable, as NBA (~49 subs/game, both ids) | Yes (2010+) | As NBA | None (adapt NBA engine) |
| Men's CBB | `espn_mens_college_basketball_pbp`, 2006+ | **Mostly missing**: only 26% of 2025 games have subs, as "Subbing in for Team-Player" with one id. CollegeBasketballData may have lineups (not checked; `CBBD_API_KEY`) | Yes: `starter`, `minutes` | None | None |
| Women's CBB | `espn_womens_college_basketball_pbp`, 2006+ | **Mostly missing**: 25% of games, same format | Yes | None | None |
| CFB | `espn_cfb_pbp`, 2004+, with EPA and win probability | **No**: passer/rusher/receiver/kicker names only, no ids, no 22-man personnel | `espn_cfb_player_box` by stat category, with ids | None (no depth charts or snaps) | None (adapt NFL engine) |
| MLB | MLB Stats API `api/v1.1/game/{pk}/feed/live`: batter and pitcher ids per PA and pitch | **Yes**: batting order, substitutions; Statcast `fielder_2..9` 2015+ (not checked here) | Stats API (sportsml already pulls pitcher logs) | **Yes**: probable pitchers and batting orders pre-game | None |

sportsdataverse files download from
`https://github.com/sportsdataverse/sportsdataverse-data/releases/download/<tag>/<file>.parquet`.
The GitHub API (release listings) is blocked from cloud sessions, but downloads work.

## 3. Shared definition of done (every sport)

This is the NFL promotion bar (`experiments/nfl/README.md`):

1. **Walk-forward folds.** Test seasons are the last 5 complete seasons. Every model, embedding,
   PCA and value fit sees only data before the game it is for.
2. **Non-inferiority.** Compare paired per-game Brier against the sport's box-score production
   model on moneyline, and on spread and total where the sport has them (NHL/MLB: moneyline and
   total; puck/run line optional). The 95% CI must include 0 or favor the arm. Promotion decisions
   use 4-seed ensembles (`_e4` vs `_e4`).
3. **Swap test.** Change one starter at a time (best regular / worst regular / replacement / out)
   and re-predict. **Every position must move the prediction the right way.**
4. Results are written up in `experiments/<sport>/README.md` in the NFL write-up's format: short
   answer first, tables, then next steps.
5. Unit tests cover new data transforms and any new engine (including a replay check against
   real games where the sport has a simulator). `pytest` passes.

## 4. Work packages

`deps` lists the packages that must be merged first. Packages with no deps between them can run in
parallel threads. Each package is sized for one thread and one pull request.

### Shared foundation (start here)

| ID | Title | Deps | Deliverable / acceptance |
|---|---|---|---|
| **S0** | Merge the NFL player model to `main` | none | **Done** (mefortunato/runitback#18) |
| **S1** | Sport-agnostic lineup plumbing in the wager model | none | Move these from hard-coded constants onto `Sport`: NFL position groups, `LINEUP_SETS` widths, `PVA_LEVELS`/`PVA_CHANGES`/`PVA_TOTAL_SIGN`, `RECENT`, `GROUP_COLUMNS` (`wager/models/transformer/data.py`, `wager/web/lineups.py`, `wager/core/sport.py`). NFL predictions must be **bit-identical** before and after (test with a trained 2-seed fixture, as `tests/wager/test_lineup_editor.py` does) |
| **S2** | Generic player-values core | none | Extract a sport-agnostic weekly ridge-APM fitter into `runitback/core/player_values.py`. Inputs: an events table with on-field id arrays per side, a target, controls, group per player, shrink scale per group, replacement threshold, recency decay. `nfl/player_values.py` becomes a thin adapter. **NFL `pv.parquet` must be identical** |
| **S3** | Generic foundation folds | none | Generalize `nfl/foundation.py` and `nfl/embed.py` (fold layout, PCA on games before S, `update_fold`) behind a per-sport generator interface. NFL fold files must be identical |
| **S4** | Port the remaining sportsml sports into `runitback.wager` | none | Port `cfb`, `nba`, `wnba`, `cbb`, `wcbb`, `nhl`, `mlb` from `mefortunato/sportsml` (`src/sportsml/<sport>/`) plus their tests, and register them in `BUILTIN_SPORTS`. One PR per family is fine: S4a basketball (nba, wnba, cbb, wcbb; shares `espn.py`), S4b nhl, S4c mlb, S4d cfb. **Each sport's walk-forward backtest reproduces sportsml's production Brier within seed noise.** This is the baseline every arm below is compared against |

### NBA (first sport: simulator exists, both ids in substitutions)

| ID | Title | Deps | Deliverable / acceptance |
|---|---|---|---|
| **N1** | On-court lineups and player tables | none | `nba/data/players.py`: per-event home/away five from starters plus substitution events. Fix period-start and missing-sub gaps; validate that each side has exactly 5 on court for ≥99% of events. Also add `player_games.parquet` (minutes, starter, DNP reason) and `players.parquet` |
| **N2** | NBA player values | S2, N1 | Possession-level RAPM on points per possession, with offense and defense coefficients and replacement level by minutes. Groups: G / F / C (from ESPN position), or rotation slots by minutes; record the choice. Weekly leak-free fits; per-game `pv.parquet` |
| **N3** | NBA projected lineups | N1 | Previous games' rotation minus players with DNP-injury `reason`, a documented stand-in for injury reports (known at tip-off). Live: ESPN `/injuries` plus posted starters |
| **N4** | NBA foundation folds (`pbp32`) | S3 | Train the existing `runitback.nba` generator per fold. Add an `embed` (offense/defense pooling per possession); optionally add lineup inputs from N1 |
| **N5** | NBA wager integration and experiments | S1, S4a, N2, N3, N4 | `experiments/nba/` arms: `prod`, `pbp32`, `pbp32_pvc` (4 seeds), plus the swap test. Promote if the bar passes |
| **N6** | NBA lineup editor and refresh | N5 | Game-page lineup editor, Train/Refresh workflows rebuild `pv.parquet` daily |

### NHL (best personnel data; needs a simulator)

| ID | Title | Deps | Deliverable / acceptance |
|---|---|---|---|
| **H1** | NHL event and personnel ingest | none | `nhl/data/`: download `nhl_pbp_full` 2010+ to an events table with skaters and goalie on ice per event, strength state and `xg`. Shift-chart fallback for gaps |
| **H2** | NHL player values | S2, H1 | xG-differential RAPM per shift segment (per 60), with F / D groups, plus a separate goalie term (goals saved above expected). Weekly leak-free fits |
| **H3** | NHL projected lineups | H1 | Previous game's lines and goalie. **Starting goalie is the largest swing and has no announced-starter history:** evaluate (a) previous starter and (b) actual starter flagged as known-at-puck-drop, and document the leak |
| **H4** | NHL rules engine and vocab | none | `nhl/engine`, `nhl/vocab.py`: faceoff, shot / goal / miss / block, penalty and power play, empty net, OT (3-on-3 regular season), shootout. Replay real games through the engine (target ≥99% state agreement, like the NFL's 99.55%) |
| **H5** | NHL generator model, training and folds | S3, H1, H4 | Model, dataset, sampler, train CLI, and per-fold training plus `pbp32` embeddings. Evaluate score distribution vs actual |
| **H6** | NHL wager integration and experiments | S1, S4b, H2, H3 (H5 for `pbp32` arms) | Arms: `prod`, `prod_pvc` (if H5 is not ready), `pbp32`, `pbp32_pvc`. Markets: moneyline and total. Swap test including goalie |

### MLB

| ID | Title | Deps | Deliverable / acceptance |
|---|---|---|---|
| **M1** | MLB plate-appearance ingest | none | `mlb/data/`: Stats API live feed → PA table (batter, pitcher, base/out state, inning, result, run value), lineups and substitutions, probable pitchers. Cache finished games; respect rate limits |
| **M2** | MLB player values | S2, M1 | Batter and pitcher run-value RAPM (RE24-style target), with park and handedness controls. Groups: SP, bullpen, lineup slots by PA share. Optional fielding from Statcast `fielder_2..9` |
| **M3** | MLB projected lineups | M1 | Probable SP (already in the sportsml loader), previous lineup vs handedness, posted lineup when available |
| **M4** | MLB simulator | S3, M1 | Engine (count optional; PA-level outcomes, base running, innings, extras with the ghost runner by season), model, sampler, folds and `pbp32`. Replay check |
| **M5** | MLB wager integration and experiments | S1, S4c, M2, M3 (M4 for `pbp32` arms) | Arms as NHL. The SP keeps its own sign-constrained term; markets are moneyline and total |

### WNBA

| ID | Title | Deps | Deliverable / acceptance |
|---|---|---|---|
| **W1** | WNBA engine and generator | N4 | Parameterize the NBA engine/vocab by league (40-minute games, 24s clock, WNBA bonus/OT rules, teams). Replay check; folds and `pbp32` |
| **W2** | WNBA values, lineups and integration | N2, N3, S4a, W1 | Reuse the NBA pipelines with stronger shrinkage (small league); optional box-score prior. Experiments as N5 |

### College basketball (men's and women's)

| ID | Title | Deps | Deliverable / acceptance |
|---|---|---|---|
| **C1** | Lineup data probe (decision gate) | none | Check whether CollegeBasketballData (`CBBD_API_KEY`; sportsml's `cbb/data/cbbd.py`) has on-floor lineups for most D-I games, and measure ESPN sub coverage by season. **Output: a short report choosing RAPM vs box-score values per gender.** Ask the user for the key if it is missing |
| **C2** | College basketball engine and generator | N4 | Parameterize the basketball engine: men's halves, 30s clock, 1-and-1 / double bonus; women's quarters. 360+ teams. Folds and `pbp32` |
| **C3** | College basketball player values and rotations | S2, C1 | RAPM where lineups exist (per C1), else minutes-weighted box plus-minus. Rotation projection from recent minutes minus DNPs. Transfer-aware (player id carries across teams) |
| **C4** | College basketball integration and experiments | S1, S4a, C2, C3 | As N5, per gender |

### College football (partial: no personnel data)

| ID | Title | Deps | Deliverable / acceptance |
|---|---|---|---|
| **F1** | CFB engine and generator | S3 | Adapt the NFL engine to college rules (OT format by season, clock rules, PAT spot) from `espn_cfb_pbp`. Replay check; folds and `pbp32` |
| **F2** | CFB QB / skill values | S2 | Map passer and rusher names to `espn_cfb_player_box` ids. QB EPA per dropback (shrunk); a starting-QB-change term. Optionally returning production and transfers from CollegeFootballData |
| **F3** | CFB integration and experiments | S1, S4d, F1, F2 | Arms: `prod`, `pbp32`, `pbp32_qb`. Swap test on QB only |

### App and operations

| ID | Title | Deps | Deliverable / acceptance |
|---|---|---|---|
| **A1** | Predictions site per sport | S4 | Add each ported sport to `web/src/App.tsx` `SPORTS` and the server; generator pages only where a simulator exists |
| **A2** | Train / Refresh workflows per sport | each sport's integration package | Per-sport Actions workflows for folds, values and predictions, path-filtered like the NFL's |

## 5. Suggested waves

1. **Wave 1 (parallel):** S1, S2, S3, S4a–d, N1, H1, H4, M1, C1. All have no deps.
2. **Wave 2:** N2, N3, N4, H2, H3, M2, M3, F2, F1, H5.
3. **Wave 3:** N5 (first full sport through the bar), H6, M4, M5, W1, C2, C3.
4. **Wave 4:** N6, W2, C4, F3, A1, A2.

## 6. Watch-outs

- **Path-filtered experiment workflows.** Pushing changes to `experiments/nfl/*.yaml` or their
  workflow files on a `claude/**` branch starts multi-hour Actions runs. Only do that deliberately.
- **No result data in git.** Experiment results are uploaded as run artifacts, never committed
  (`.gitignore`). A new sport's experiments should follow `nfl-experiment-inputs.yml`: rebuild every
  input from the public sources inside the run.
- **Leakage.** Every fit, PCA and projected lineup uses only data before the game it is for. Where a
  sport has no pre-game availability history, document the stand-in and its leak.
- **Noise floor.** A reseeded baseline moves Brier about ±0.002; use 4-seed ensembles for decisions.
- **Long runs.** Fold training is heavy. Use GitHub Actions matrix jobs (as
  `.github/workflows/foundation-experiment.yml` does) rather than a single cloud session.
