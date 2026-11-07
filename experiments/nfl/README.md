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

Results: `foundation/ablation.md`, `foundation/report.md` (per market, with the pass/fail column),
`foundation/consistency.csv`, the per-game predictions in `foundation/<config>/<season>.csv` and
the generator rollouts in `foundation/generator/<season>.csv`.

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
   once it is on `main`. The models are kept as the run's `foundation-models` artifact for 90 days.

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
on those. `lineups/qb_swap.<shard>.csv`; `python scripts/lineup_qb_swap.py --summary
experiments/nfl/lineups/qb_swap.*.csv`.

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
(QB changed / QB projected to change / top-quartile turnover / steady), and the per-game change
flags in `lineups/changes/<season>.csv`.

**Next steps for personnel**
1. Keep `pbp32` in production for now; the candidate replacement is `te` (passes everywhere, ranks
   QBs). Switching needs projected lineups in the refresh workflow (live depth charts and injury
   reports) so a mid-week QB change reaches the prediction.
2. Test the inference-time payoff directly: score `te` vs `pbp32` on games where the starting QB
   changed from the previous week, using the projected (not the actual) lineup, over 2016–2025 to
   get enough of them, and against the market's line move for those games.
3. Extend player effects beyond the ball handler (pass rush / coverage on pass plays, the offensive
   line on runs) so non-QB injuries move the rollouts too, and re-check with a swap of other
   positions (WR1, LT, edge rusher).
4. Get more signal per parameter for the wager model: a single scalar per side (the foundation's
   predicted margin change from swapping the projected lineup in) instead of 16–64 columns.
