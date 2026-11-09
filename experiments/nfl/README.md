# NFL: the play-by-play generator as a foundation for the wager model

Can the runitback play-by-play transformer serve as a foundation model under the
spread / moneyline / total transformer (ported from sportsml), without making the wager predictions
worse, and do the two models agree with each other?

**Short answer**
- **Frozen embeddings work.** Feeding the frozen generator's game embeddings into the wager model's
  history tokens passes the non-inferiority bar on all three markets, at every embedding size.
  The 32-dimension version (`pbp32`) is slightly better than production on every market, but the
  gain is within the noise floor.
- **Fine-tuning the generator on the wager loss makes things worse.** Both fine-tuned arms fail
  the bar on moneyline and totals, whether or not the next-play loss is kept.
- **Win and margin agree; totals don't.** The generator and the wager head agree on win
  probability and margin (correlation 0.85). On totals they don't, because the generator
  under-scores by about 8 points a game.

Results: `foundation/ablation.md` and `foundation/report.md` (per market, with the pass/fail
column). The per-game predictions (`foundation/<config>/<season>.csv`), the generator rollouts
(`foundation/generator/<season>.csv`) and `foundation/consistency.csv` are in the run's
`foundation-results` artifact (see "Reproducing the results").

## Reproducing the results

Only the configurations (`experiments/nfl/*.yaml`) and the write-ups (`*.md`) are kept in git. Every
per-game prediction, rollout, player-value table and swap result is regenerated from the public
sources (nflverse / SportsDataverse play-by-play, ESPN lines) by the experiment's workflow:

| experiment | workflow | results artifact |
|---|---|---|
| foundation | `foundation-experiment.yml` | `foundation-results`, `foundation-models` |
| lineups | `lineup-experiment.yml` | `lineup-results` |
| QB swap (production foundation) | `qb-swap.yml` | `qb-swap-results` |
| no ratings / QB target / players only / player effects | `noratings-` / `qbtarget-` / `playersonly-` / `playereffects-experiment.yml` | `<name>-results` |
| player values | `playervalues-experiment.yml` | `playervalues-results` |
| all positions | `allpositions-experiment.yml` | `allpositions-results` |

Each one calls `nfl-experiment-inputs.yml` first, which downloads and preprocesses the data, trains
the production foundation folds, and builds the projected-lineup tables and the weekly player values
in the same run. Baselines such as `pbp32` and `prod` are rerun alongside the arms rather than read
from earlier results. Start a workflow from the Actions tab ("Run workflow"), or push a change to its
configuration on a `claude/**` branch. The results artifact holds the experiment's directory with the
per-game CSVs and the regenerated `*.md` reports; copy the reports back here when they change the
write-up. Seed noise means a rerun reproduces the tables within about ±0.002 Brier, not exactly.

To run a cell locally instead: `runitback nfl download` / `preprocess`, `runitback-wager nfl download`
/ `process`, `runitback nfl foundation` (or `runitback nfl train` + `embed`), then
`python scripts/foundation_arm.py <config> <season> --experiment experiments/nfl/<name>.yaml`.

## Setup

- **Walk-forward evaluation, 2021–2025** (5 folds, about 1,400 games). For each test season S, the
  wager model and the play-by-play foundation model are fit only on earlier data.
- **Foundation, `runs/foundation/fold<S>`.** The full v0.2.0 configuration with the "who" head
  (lineups, coaches, personnel): 10 epochs, validated on season S-1 weeks 15+. It never sees
  season S.
- **Game embeddings, `runitback nfl embed`.** Teacher-forced trunk states are mean-pooled over each
  team's offensive plays and, separately, its defensive plays. PCA to 16/32/64 dimensions is fit
  only on games before S.
- **Coupling.** The embeddings join the wager model's per-team-game history tokens (columns
  `_pbp_*` plus a missing flag). Nothing else changes.
- **Arms (`foundation.yaml`):**
  - `prod`: sportsml's production NFL transformer, the baseline.
  - `pbp16`, `pbp32`, `pbp64`: box score plus embedding.
  - `pbp32_only`: the embedding replaces the box-score features.
  - `pbp32_ft`: the frozen fit, then the top trunk block, `ln_f` and the wager model train jointly
    on the wager loss, keeping the next-play loss as an auxiliary.
  - `pbp32_ft_noaux`: the same without the auxiliary loss.
  - `prod_seed1`: the baseline with another seed, to measure the noise floor.
- **Non-inferiority bar.** The paired per-game Brier difference against `prod` must have a 95% CI
  that includes 0 or favors the arm, on each of cover, moneyline and over.
- **Where it ran.** Everything except `prod` and `prod_seed1` ran on GitHub Actions
  (`.github/workflows/foundation-experiment.yml`, run 42286987708) as one job per fold, per
  (arm, season) and per generator season. It took about 2¾ hours of walltime. `prod` and
  `prod_seed1` were run earlier in a cloud container on the same seasons. The 2021 fold's
  embeddings came out identical in both environments (14,788 team-games, 79.2% of the variance in
  64 dimensions).

## Wager performance

Paired Brier difference against `prod` (negative is better), ± the 95% CI half-width. The noise
floor from a reseeded baseline is about ±0.002.

| arm | moneyline | spread (cover) | total (over) | passes | margin RMSE |
|---|---|---|---|---|---|
| prod | 0.2191 | 0.2536 | 0.2587 | — | 13.02 |
| **pbp32** | **−0.0012 ± 0.0024** | **−0.0025 ± 0.0028** | **−0.0007 ± 0.0025** | yes | **12.90** |
| pbp16 | −0.0003 ± 0.0024 | −0.0006 ± 0.0027 | −0.0003 ± 0.0024 | yes | 12.93 |
| pbp64 | −0.0000 ± 0.0024 | −0.0010 ± 0.0027 | −0.0024 ± 0.0026 | yes | 12.97 |
| pbp32_only | +0.0003 ± 0.0035 | +0.0015 ± 0.0040 | +0.0016 ± 0.0042 | yes | 12.94 |
| prod_seed1 (noise) | +0.0018 ± 0.0023 | +0.0010 ± 0.0026 | −0.0010 ± 0.0021 | yes | 13.06 |
| pbp32_ft | +0.0057 ± 0.0039 | +0.0042 ± 0.0048 | +0.0361 ± 0.0107 | **no** | 13.26 |
| pbp32_ft_noaux | +0.0048 ± 0.0035 | +0.0031 ± 0.0044 | +0.0258 ± 0.0091 | **no** | 13.21 |

For context, the Vegas moneyline Brier on the same games is 0.2115. Every model is still about
0.007–0.008 behind the market.

- **pbp32 is the best arm, but not significantly.** It has the lowest Brier in every market and
  the lowest margin RMSE. Its improvement is about half the seed-to-seed noise, though, so this is
  "at least as good", not "better". Its gains are concentrated in 2024 and 2025 (−0.0037 each on
  moneyline), the seasons where the generator had the most play-by-play history.
- **pbp32_only is as good as the box score.** Replacing the box-score features with the embedding
  costs nothing measurable. The generator's embedding carries roughly the information the box
  score does.
- **Fine-tuning fails, mainly on totals.** The fine-tuned arms shift predicted totals up by 4–5
  points (mean 51.0 and 49.9 against 45.5 for pbp32 and a mean line of 44.6). Their over
  probability averages 0.67 when the actual over rate is 0.49. Moneyline is also worse. Keeping
  the next-play loss doesn't help (`pbp32_ft` is no better than `pbp32_ft_noaux`).
- **Why fine-tuning fails (likely).** The fine-tune stops on validation loss, so the shift happens
  out of sample. Once the embeddings are trainable, they drift away from what the wager model was
  fit on, and about 1,400 training games per fold are far too few to retrain a trunk block.
  Fine-tuning isn't worth pursuing at this data size.

## Generator vs wager consistency

For each game, `scripts/foundation_consistency.py` runs 100 full rollouts of the foundation model
that never saw that season. The rollouts use previous-game lineups, both opening kickers, and
pre-game ratings scaled by a per-fold strength scale fit on the previous season. It then compares
the share of rollouts won, covered and over against the wager head's probabilities.

| | moneyline | spread (cover) | total (over) |
|---|---|---|---|
| generator Brier (prod wager: 0.2191 / 0.2536 / 0.2587) | 0.2276 | 0.2655 | 0.3151 |
| corr(generator, prod) | 0.83 | 0.47 | 0.44 |
| corr(generator, pbp32) | 0.85 | 0.51 | 0.47 |
| mean abs gap vs pbp32 (probability) | 0.083 | 0.092 | 0.273 |
| calibration slope: generator / pbp32 | 0.70 / 1.03 | 0.01 / 0.42 | 0.09 / 0.05 |

Points: margin correlation 0.85 with a mean gap of 2.9 points. Totals correlation 0.51 with a
mean gap of 8.6 points.

- **Win and margin are consistent.** The generator and the wager head agree on who wins and by
  how much (correlation 0.85). The generator's mean margin (2.0) is close to the actual 2.25.
  Coupling also raises agreement: pbp32's predictions track the generator more closely than
  prod's in every market.
- **The generator is overconfident.** Its win probabilities have a calibration slope of 0.70, so
  it is too sure of itself, and its Brier is about 0.009 worse than the wager head's.
- **The generator under-scores.** It averages 37.0 total points against 45.2 actual (a line of
  44.6), short every season by 4–11 points. So it calls the under about 75% of the time, and on
  totals it is the generator, not the wager head, that is off.
- **Its margins are too narrow.** The spread of margins across games is too small (6.7 points,
  against 14.2 for actual results), consistent with the strength scale hitting its limits (1.0–2.0).
  This isn't a coupling problem; it is a calibration target for the generator itself.

## Recommendation and next steps

1. **Adopt `pbp32` (frozen) for production.** It is at least as good as `prod` on every market,
   slightly better on margin, and more consistent with the generator. Done: `runitback nfl
   foundation` keeps one fold per test season, and the predictions workflows set
   `RUNITBACK_NFL_FOUNDATION` (README, "Predictions data").
2. **Drop generator fine-tuning** at this data size.
3. **Fix the generator's scoring bias** before showing its totals next to the wager model's on the
   game page. A good first step is checking the rollout setup (clock and elapsed sampling,
   end-of-half behavior) for the 8-point shortfall. Then either recalibrate the strength scale
   jointly on margin and total, or post-hoc scale points.
4. **Rerun** with `git push` after editing `foundation.yaml`, or dispatch "Foundation experiment"
   once it is on `main`. The models are kept as the run's `foundation-models` artifact and the
   results as `foundation-results`, both for 90 days.

## Personnel: does knowing who plays this week help?

The wager model's history (box scores plus the frozen `pbp32` game embeddings) describes the
teams as they played; the only signal about *this* week's players was the injury features
(`inj_<unit>`, expected missing snap share per unit). `experiments/nfl/lineups.yaml` adds the game
being predicted's **projected lineup**, built only from what is known before kickoff
(`runitback nfl lineup-features`): the team's published depth chart for the week (else its
previous game's), minus players listed Out/Doubtful or off the active roster, with the next
player in each position group moved up. The projected QB1 is right in 95% of 2025 games (88% in
2024), and in 70% (41%) of games where the starter changed, against 18% for the previous game's
lineup alone. Each fold's foundation model turns the lineup into features for both teams:

| arm | adds to `pbp32` (production) |
|---|---|
| `pbp32_lp` | the projected players' embeddings, pooled by the foundation's lineup encoder (as offense and defense) |
| `pbp32_lp_lu` | + how that pooled lineup differs from the lineups behind the history games |
| `pbp32_lp_cf` | + the history re-embedded as if the projected lineup had played it, minus as played |
| `pbp32_noinj` | production without the injury features |
| `pbp32_lp_noinj` | the projected lineup in place of the injury features |

**Short answer: no arm beats production, and none should be adopted yet.**
- **`pbp32_lp` (the pooled projected lineup) passes on all three markets and is a wash**:
  Δ Brier moneyline −0.0009 ± 0.0023, spread +0.0014 ± 0.0027, total +0.0009 ± 0.0027.
- **The history-relative arms hurt totals.** `lp_lu` (+0.0030 ± 0.0023) and `lp_cf`
  (+0.0036 ± 0.0025) fail the bar on totals, worst on high-turnover games (+0.008). The extra
  16–64 context columns look like noise the head overfits on 1,400 test games a fold.
- **On the games where personnel moved, the lineup leans the right way but within noise.** Where
  the starting QB changed (352 games), `pbp32_lp` is −0.0018 ± 0.0045 (moneyline),
  −0.0028 ± 0.0054 (spread), −0.0031 ± 0.0053 (total); the other arms are worse there than
  production (`subsets.md`). Roughly 4× the test games would be needed to tell.
- **The injury features don't carry much either.** Dropping them (`pbp32_noinj`) costs
  +0.0005 / +0.0025 / +0.0007, all within noise, and the projected lineup roughly replaces them
  (`pbp32_lp_noinj`). Personnel news is largely in the closing line already, which every arm still
  trails (+0.006–0.008 Brier vs the market).

**QB-swap check: the player embeddings barely know who the quarterback is.** Rolling out the
2025 fold's generator on 120 games of 2025 (100 rollouts each, same seeds in every condition)
with the same named QB in every home team's QB1 slot moves the home team's win probability by at
most 1.3 points and its margin by under half a point:

| home QB1 | home win | Δ win vs actual lineup | Δ margin (pts) |
|---|---|---|---|
| previous game's (actual) | 0.532 | | |
| Mahomes | 0.542 | +0.009 ± 0.010 | +0.11 ± 0.22 |
| J. Allen | 0.546 | +0.013 ± 0.009 | +0.47 ± 0.24 |
| C. Rush | 0.540 | +0.008 ± 0.009 | +0.15 ± 0.23 |
| T. Siemian | 0.541 | +0.009 ± 0.008 | +0.17 ± 0.20 |
| B. Rypien | 0.536 | +0.004 ± 0.011 | +0.12 ± 0.24 |
| unknown "new QB" | 0.527 | −0.005 ± 0.009 | −0.34 ± 0.23 |

Mahomes minus Siemian is +0.001 ± 0.010; the market would put a swap like that at several points.
The model separates known from unknown players a little (the "new QB" token is the only negative)
but not good from bad ones. That explains the experiment above: the lineup features can't carry
personnel information the foundation never learned. Its team strength comes from the pre-game
ratings and the history, and lineup dropout (0.2) plus unknown-lineup training (0.2) let it lean
on those. The shards are in the `qb-swap-results` artifact (`qb-swap.yml`); summarize them with
`python scripts/lineup_qb_swap.py --summary qb_swap.*.csv`.

**No team-strength inputs (`experiments/nfl/noratings.yaml`): doesn't fix it.** Retraining every
fold without the pre-game ratings (`nr`), and without ratings or team-season tokens (`nrts`), so
team strength can only come from the lineups, coaches and franchise, leaves the QB swap flat:

| home QB1 | Δ win, production | Δ win, `nr` | Δ win, `nrts` |
|---|---|---|---|
| Mahomes | +0.009 ± 0.010 | −0.013 ± 0.010 | −0.011 ± 0.010 |
| J. Allen | +0.013 ± 0.009 | +0.000 ± 0.009 | +0.012 ± 0.010 |
| C. Rush | +0.008 ± 0.009 | +0.006 ± 0.009 | +0.011 ± 0.011 |
| T. Siemian | +0.009 ± 0.008 | −0.006 ± 0.010 | +0.007 ± 0.011 |
| B. Rypien | +0.004 ± 0.011 | −0.009 ± 0.009 | +0.005 ± 0.010 |
| unknown "new QB" | −0.005 ± 0.009 | −0.017 ± 0.009 | −0.007 ± 0.010 |

Elite and journeyman QBs are still indistinguishable (Mahomes is now slightly *below* Siemian);
only the unknown token is consistently negative. And the wager model is a little worse on these
foundations: `nr_pbp32` +0.0014 / +0.0019 / +0.0013 (moneyline / spread / total; fails on
spread), `nrts_pbp32` similar, `nrts_pbp32_lp` fails on moneyline and spread
(`noratings/report.md`, `subsets.md`, `qb_swap.md`). With the ratings gone, the next-play loss still
doesn't push the player embeddings to rank players: one play's outcome says little about who
the QB is, and the in-game state (score, field position, down) explains most of it. Ratings
were not what was masking the lineup signal.

**Passer target with team identity (`experiments/nfl/qbtarget.yaml`): not enough.** Adding the
passer target to production's settings (`qb`), or to them without the ratings (`qbnr`), leaves
the QB swap flat (Mahomes −0.006 / +0.008, Siemian −0.002 / +0.001, the unknown QB
−0.011 / −0.002), and the wager arms slip a little (`qb_pbp32` +0.0016 ± 0.0013 on moneyline,
failing; `qbnr_pbp32` fails moneyline and spread; `qbtarget/report.md`). With the team-season and
franchise tokens in place the players stay redundant.

**Players-only foundation (`experiments/nfl/playersonly.yaml`): the embeddings now know the players,
the simulator still barely uses them.** No team identity at all (franchise / team-season / season
tokens, ratings and coach records removed); each play is predicted from the game so far plus the
possession team's offense unit (offensive and specialist players pooled per position group from
the depth chart, plus its head coach) and the other team's defense unit (defensive players plus
head coach), with the 22 on the field in the result / yards / clock heads. Trained on 2013+ with
fewer hidden lineups; `po` adds the passer and ball-carrier targets (each play's result and
yards from the QB's / the ball carrier's embedding alone), `ponts` doesn't.

| home QB1 | Δ win, production | Δ win, `po` | Δ win, `ponts` |
|---|---|---|---|
| Mahomes | +0.009 | +0.012 ± 0.011 | +0.018 ± 0.011 |
| J. Allen | +0.013 | +0.022 ± 0.011 | −0.005 ± 0.011 |
| C. Rush | +0.008 | +0.009 ± 0.011 | +0.015 ± 0.009 |
| T. Siemian | +0.009 | +0.008 ± 0.011 | +0.000 ± 0.012 |
| B. Rypien | +0.004 | +0.011 ± 0.011 | +0.005 ± 0.011 |
| unknown "new QB" | −0.005 | **−0.040 ± 0.012** (−1.4 pts) | **−0.041 ± 0.010** (−1.4 pts) |

Two things changed. The lineup now matters: replacing the QB with an unknown one costs 4 points of
win probability (1.4 points of margin), eight times production's effect. And with the passer
target the embeddings rank quarterbacks: the passer head's predicted completion rate from each
QB's embedding alone correlates 0.63 with his actual 2016-24 completion rate (97 QBs with 300+
dropbacks; 0.34 for sack rate), Mahomes highest of all and Siemian / Rush near the bottom
(`scripts/passer_probe.py`). But the rollouts still hardly separate Mahomes from Siemian
(+0.012 vs +0.008): the play-outcome heads see the QB's embedding (in the on-field personnel) and
use little of the quality the passer head reads from it. Wager arms: `po_pbp32` passes on all three
markets (+0.0018 / +0.0013 / −0.0003), `ponts_pbp32` fails moneyline and spread, and the `_lp`
arms fail spread (`playersonly/report.md`, `subsets.md`). Dropping team identity costs a little
accuracy that the players don't yet make back.

**Player effects (`experiments/nfl/playereffects.yaml`): the simulator now ranks the players.**
`--player-effects` adds the passer head's result / yards logits (from the QB's embedding) to the
play heads on pass plays, and the ball-carrier head's on runs, in training and in the rollouts, so
a player's quality goes straight into his plays. `pe` is the players-only setup above plus player
effects; `te` is production's setup (team identity and ratings kept) plus the passer / carrier
targets and player effects. QB swap, 120 games of 2025 (Δ home win, ± 95% CI; margin in points):

| home QB1 | production | `po` | `pe` | `te` |
|---|---|---|---|---|
| Mahomes | +0.009 | +0.012 | **+0.080 ± 0.013** (+2.5) | **+0.112 ± 0.015** (+3.6) |
| J. Allen | +0.013 | +0.022 | +0.028 ± 0.014 (+0.9) | +0.060 ± 0.013 (+1.9) |
| C. Rush | +0.008 | +0.009 | −0.056 ± 0.013 (−1.6) | −0.091 ± 0.014 (−3.0) |
| T. Siemian | +0.009 | +0.008 | −0.084 ± 0.013 (−2.4) | −0.109 ± 0.015 (−3.6) |
| B. Rypien | +0.004 | +0.011 | −0.160 ± 0.014 (−5.1) | −0.163 ± 0.016 (−5.7) |
| unknown "new QB" | −0.005 | −0.040 | −0.133 ± 0.013 (−4.4) | −0.130 ± 0.013 (−4.3) |

(The `nr` / `nrts` / `qb` / `qbnr` / `ponts` runs were all within ±0.02 for every named QB.) Mahomes
vs Siemian is now 0.16 (`pe`) to 0.22 (`te`) of win probability, 4.9 to 7.2 points of margin, in the
right order and about the size the market gives a starter-to-backup swap; the passer probe still
reads 0.62 (`pe`) / 0.57 (`te`) correlation with actual completion rate. The local half-run of `pe`
(60 games) agreed (+0.096 / −0.069). Keeping team identity (`te`) gives the larger, more market-like
spread, so the team tokens don't swamp the players once the players have a direct path.

Wager arms (Δ Brier vs `pbp32`, moneyline / spread / total): `pe_pbp32` +0.0012 / +0.0009 / −0.0008,
`te_pbp32` +0.0012 / +0.0011 / +0.0014, `te_pbp32_lp` +0.0007 / +0.0013 / −0.0000 all pass;
`pe_pbp32_lp` fails spread (+0.0057 ± 0.0033). None beats `pbp32` on any market outright. On the
QB-changed and QB-projected subsets most arms lean better on spread (e.g. `te_pbp32_lp` −0.0027 /
−0.0039, `pe_pbp32` −0.0021 / −0.0015) but every CI includes 0 (`playereffects/report.md`,
`subsets.md`). So the foundation now responds to personnel the way it should, at no measurable cost
to the wager models; the payoff should come at inference, when a lineup changes after the
historical games the wager model learned from (an injured starter, a traded QB), which the
backtest over ordinary weeks barely samples.

Results: `lineups/ablation.md`, `lineups/report.md` (per market, pass/fail), `lineups/subsets.md`
(QB changed / QB projected to change / top-quartile turnover / steady). The per-game change flags
(`changes/<season>.csv`) are in each run's `lineup<season>` artifacts.

### Player values: who plays, straight into the wager model

Rather than make the play-by-play simulator carry personnel, give every player a value and put
it straight into the wager model.

**Player values** (`src/runitback/nfl/player_values.py`, `scripts/player_values.py`):
- **The fit.** A ridge adjusted plus-minus on each scrimmage play's EPA, 2016+, with the 22 players
  on the field. Each of the possession team's players gets an offense coefficient and each of the
  other team's a defense coefficient, plus down / distance / field-position / season controls.
- **Shrinkage and recency.** A wider prior for QBs (a narrower one for OL and DL, tuned on the next
  season's plays). Recent seasons are weighted more (0.7 per season back, four seasons plus the
  current one).
- **Replacement level.** Players with under 150 plays share a replacement-level coefficient for
  their side and group, and a player's value is his coefficient over it. Kickers and punters: shrunk
  EPA per attempt over the league mean.
- **Points per game.** Values become points per game over replacement through each depth-chart
  slot's expected snaps.
- **Leak-free and weekly.** Every week from 2016 on is refit on the plays before it (223 fits, two
  minutes), so values update weekly and never see the game they are for.
- **Per-game features.** Each team's projected pre-game lineup (published depth chart minus players
  ruled out) is summed per position group: `pv.parquet` (`scripts/player_values.py`, in the
  `playervalues` artifact of every experiment run that uses it). Columns: QB, RB, WR, TE, OL,
  DL, LB, DB and kicking, plus each group's change vs the team's last 8 lineups.

Do the values look right? Before 2025, the top QBs are Allen, Jackson, Purdy and Tagovailoa, along
with Bosa at DL, Kelce at TE, Henry and Jones at RB, and A.J. Brown and Evans at WR. Some low ends are
plus-minus noise (Bobby Wagner as the "worst" LB regular); `values_<season>.csv` from
`scripts/player_values.py --season-values` has them all. The projected-lineup value difference alone, with no team history at all, correlates 0.86
with the market spread and 0.40 with the final margin (the market: 0.45).

**Backtest** (2021-2025, Brier vs `pbp32`; seed noise is about ±0.002, so the 4-seed averages
`_e4` are compared with each other):

| model | moneyline | spread | total | note |
|---|---|---|---|---|
| `pvg`: margin / total from the lineup values + a team rating learned from results | +0.0024 | **+0.0060 (fails)** | −0.0006 | six numbers per game |
| `ens`: average of `pbp32` and `pvg` | −0.0012 | +0.0003 | −0.0017 | |
| `pbp32_pv`: the values as context for the wager model's head | −0.0013 | +0.0009 | −0.0019 | |
| `pbp32_pv_e4` vs `pbp32_e4` | −0.0015 ± 0.0016 | −0.0001 | −0.0013 | QB projected to change: −0.0054 ± 0.0045 |
| `pbp32_pva_e4` vs `pbp32_e4` (constrained, levels + changes) | +0.0045 | **+0.0145 (fails)** | +0.0002 | margin RMSE 13.4 |
| **`pbp32_pvc_e4` vs `pbp32_e4`** (constrained, changes only) | **−0.0018 ± 0.0037** | +0.0001 | +0.0003 | QB projected to change: **−0.0108 ± 0.0106** |
| `pvg_mkt`: `pvg` blended with the line | −0.0063 | −0.0007 | −0.0072 | ties the market (+0.0005 ± 0.0012) |

The gains are where lineups change. On steady-lineup games every version is within ±0.0002 of
`pbp32`; on games where the QB is projected to change, or with top-quartile turnover, they are 2-10x
larger. Nothing beats the market (`pvg_mkt` only ties it).

**Does changing a player move the prediction the right way?** For every 2025 game, one home
starter at a time was changed and the game re-predicted (`scripts/wager_lineup_sensitivity.py`,
with the best and worst regulars from `swap_refs_2025.csv`):
- **The change.** Each position group's starter goes to the best / worst regular by value, a
  replacement-level player, or out (the group moves up a slot).
- **`pbp32_pv` fails it.** It moves the right way for offense and specialists, but only by about a
  third of the direct model's amount. For every defensive position it moves the **wrong** way
  (Bosa in: −0.16 points). The head uses the values however fits, and a backtest gain does not make
  that sensible.
- **The fix: sign-constrained additive terms (`pva`, `pvc`).**
  - The values are hidden from the head's MLP and enter the margin only as
    `w · Σ_groups (team value − opponent value)`, with non-negative weights, so the change can't
    move the prediction the wrong way.
  - The total works the same way: offense adds points, defense takes them away.
  - One weight per kind is shared by every position, so a point of value is a point anywhere.
    Per-group weights on standardized columns gave punters 9-point swings.
- **Levels plus changes (`pva`)** count team strength twice (the history already has it), with 1.1
  margin points per point of value and overconfident spreads.
- **Changes only (`pvc`)**, i.e. this lineup vs the team's last 8, learned **0.58 margin points per
  point of value**. The direct game model's own fit is 0.6.

`pbp32_pvc`, one home starter changed, all 285 games of 2025 (100% of games move the right way, at
every position):

| change | Δ home margin | Δ home win |
|---|---|---|
| QB1 → Josh Allen / Davis Mills / replacement / backup promoted | +4.1 / −3.0 / −6.6 / −3.0 | +0.11 / −0.09 / −0.18 / −0.08 |
| RB1 → Aaron Jones / replacement | +0.1 / −0.5 | |
| WR1 → Mike Evans / replacement | +0.6 / −0.8 | +0.015 / −0.022 |
| TE1 → Travis Kelce / replacement | +0.4 / −0.4 | |
| OL1 → best regular / replacement | +0.4 / −0.9 | |
| DL1 → Nick Bosa / replacement | +0.3 / −0.4 | |
| LB1 → best regular / replacement | +0.5 / −0.1 | |
| DB1 → best regular / replacement | +0.5 / −0.6 | |
| K → Brandon Aubrey / Jake Moody | +0.2 / −0.3 | |

**Next steps for personnel**
1. **In production: `pbp32_pvc`** (4 seeds; `runitback.wager.nfl`). It passes every market
   against `pbp32` and is better on moneyline, with the gain concentrated in QB-change games. Every
   lineup change moves the prediction the right way by `0.58 × the change in player value`.
2. **How it runs:** `runitback nfl player-values` (`runitback.nfl.player_values.game_table`, the
   same table as `scripts/player_values.py`: identical on every played game) fits the weekly values
   and sums each game's projected lineup, upcoming games included. Train and Refresh predictions
   rebuild it, so the upcoming games follow the latest depth charts and injury reports. Each
   upcoming game's page has a lineup editor: both projected lineups with every player's points over
   replacement at his slot, backups and players ruled out. Changing a starter or ruling a player
   out re-prices the spread, win probability and total in the browser. The model's lineup terms are
   additive, so this is exact: a trained 2-seed model's own predictions with its lineup values
   changed agree with the editor's arithmetic to 1e-4 points (the unit test is
   `tests/wager/test_lineup_editor.py`).
   One gap: nflverse's on-field participation for 2026 is not out, so this season's values can't
   update from 2026 plays yet; until then they come from the fit through 2025 (kickers and punters
   do update: their values come from the play-by-play).
3. **The all-positions foundation (`ae` / `ae0`, `allpositions/`) is not the route for wagers.**
   The play-by-play model with additive per-player effects for all 22 on the field (`ae`: with a
   lineup-only auxiliary loss; `ae0`: without) was trained on the same folds:
   - *Simulated swaps* (`position_swap.md`, 120 games of 2025, one home starter changed): the
     right direction for QB (Allen +1.6, new starter −4.9 / −4.5 points), OL, TE and WR, and for DL
     and LB best/worst. RB is flat. DB, K and P are flat or point the wrong way (Gardner-Johnson
     −0.1 in `ae`; Aubrey −0.05 in `ae0`). It fails the "true at every position" test that `pvc`
     passes.
   - *Its per-player effects don't line up with the player values*: across 4,514 players, the
     correlation with the adjusted plus-minus values is −0.04 to +0.14 at every position. The
     best/worst lists are mostly backups (John Urschel, T.J. Logan). With the team, the play state
     and the other 21 players all in the model, a single player's own term barely identifies
     anything. The effects are there, but they aren't player quality.
   - *Wager arms* (`ablation.md`, `report.md`, `subsets.md`): its game summaries are slightly worse
     than production (`ae_pbp32` +0.0012, `ae0_pbp32` +0.0009 Brier, within noise; `ae_pbp32`
     fails moneyline). Adding the player values on top (`ae_pbp32_pvc`, −0.0001) does no better
     than `pbp32_pvc` (−0.0013).
   - Single-seed `pbp32_pvc` fails total in this run (+0.0025 ± 0.0019). The 4-seed version
     passes (+0.0003), so this is seed noise on the total, but the next promotion check should
     use 4 seeds.

   Keep the production foundation (pbp32) for the game summary and the simulator. Personnel
   enters through the player values (`pvc`). The `ae` / `ae0` heads stay behind `--all-effects`
   for experiments.
4. **Better values:** priors for rookies from draft slot and position (today they are replacement
   level until 150 plays); aging; in-season weighting; and a lineup-value version of the total.
