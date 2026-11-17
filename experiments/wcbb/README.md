# Women's college basketball

C4 (the wager model on the foundation and the player values) is below; the rest points to the men's write-up.

The women's player values and rotations (work package C3) are written up with the men's in
`experiments/cbb/README.md`. Women's use box-score values throughout: CBBD has no women's lineups,
and ESPN's substitutions only start in February 2025 (`experiments/cbb/lineup_probe.md`). Rebuild
them with `runitback college ... --gender wcbb`.

## Engine and foundation folds (C2)

See `experiments/cbb/README.md`, "Engine and foundation folds (C2)": one engine and workflow for both genders; the women's replay agreement is 99.60%.

## The wager model on the foundation and the player values (C4)

Work package C4 (`docs/multisport/PLAN.md`): does the women's college basketball wager model
(`runitback.wager`, the S4a port) get better with C2's frozen `pbp32` game embeddings and C3's player
values of each game's projected rotation (`pvc`), and does changing a starter move its prediction the
right way at every position? It is the NBA's N5 (`experiments/nba/README.md`) with college data.

**Short answer: pending. The code, configurations, workflow and tests are in; the experiment has not been
run.** Nothing below "Results" has numbers because the arms take about 100 jobs of 1-2 hours each, and CLAUDE.md
reserves heavy training for Actions runs that someone starts deliberately. To get the results, dispatch
`College wager experiment` (`.github/workflows/college-wager-experiment.yml`) with `gender: wcbb`
(see "Running it"), then fill the tables from its `wcbb-wager-results` artifact.

### Setup

- **Walk-forward, 2022-2026** (5 folds, about 5,500-6,000 games each, neutral sites included). For each test season S the wager model, the
  foundation fold and its PCA see only games before S, and the player values are refit weekly on games
  before each game's Monday.
- **Games and lines:** ESPN's odds API, one request per game (books' lines on women's games start in 2021-22, so 2022 is the first test season). The wager games table is rebuilt in the run
  (`runitback-wager wcbb download` / `process`).
- **Values:** Box-score values throughout (C3: CBBD has no women's lineups). Rotations are C3's projections.
- **Arms** (`experiments/wcbb/wager.yaml`; every name ends in `_s<seed>`):

| arm | what it adds to `prod` |
|---|---|
| `prod` | nothing: the S4a box-score production model, the baseline |
| `pbp32` | fold S's frozen 32-dim per-possession game embeddings in the history tokens |
| `pbp32_pvc` | + each position group's (G / F / C, offense and defense) change in the projected rotation's value vs the team's last 8 rotations, through sign-constrained additive terms on margin and total (the NFL's production `pvc`). The values are hidden from the head's MLP |
| `prod_pvc` | the player values without the foundation (the NBA's promoted arm) |
| `prod_pvc_prev` | `prod_pvc` on the leak-free rotation (below) |
| `prod_pvc_w6` | `prod_pvc` with the NFL's lineup-weight start (one seed, below) |

- **Four seeds, as four jobs.** A college fit takes 1-2 hours, so four in one job would pass GitHub's
  6-hour limit. Each seed is its own matrix job (`<arm>_s0 .. _s3`, seeds 0-3 as `ensemble: 4` would
  train) and `scripts/ensemble_predictions.py` averages them into `<arm>_e4`. This differs from the
  NBA's `_e4` in one way: the NBA's ensemble averages the members' margins and totals inside one model and
  calibrates one sigma; here the members' margins, totals and probabilities are each averaged. It is the
  same for every arm, so promotion still compares `_e4` with `_e4`. Single-seed tables use seed 0 of each arm.
- **The `pvc` weight.** The lineup terms add `w` points of margin per point of value change, with one
  learned non-negative `w`. The head starts it at 0.6 (the NFL learned 0.58). C3 measured much less in college
  basketball: a change in projected lineup value moves the margin by 0.15 per point within a season (men;
  0.14 women), and 0.25 (men) / 0.47 (women) at season openers, because the values overstate the gap
  between a starter and his backup. So the college arms start `w` at 0.2 (`additive_init`, a new
  option on the head; the default is unchanged, so NFL and NBA models are identical). `prod_pvc_w6`
  starts it at 0.6 to show whether the start matters, and the swap-test report prints the weight each fit
  ended at.
- **The rotation leak, and its comparison.** ESPN's college box scores give no did-not-play reason, so C3's default
  projection removes everyone who did not play. That includes coach's-decision absences, a larger leak
  than the NBA's injury reasons (C3: top-5 right 81% against 74-75% without it). `prod_pvc_prev` uses
  the leak-free rotation (the players of the team's last 3 games, nobody removed;
  `runitback college lineups --rule previous`). **Read the gain of `pvc` against `prod_pvc_prev`
  before believing it.** There are no historical college injury reports to do better.
- **Joins.** Embeddings and values key games by the ESPN id string and teams by the school name the wager
  games table uses. A `joins` job (`scripts/college_join_check.py`) fails the run before the arms start if
  fewer than 80% of a season's Division I games find their rows. On a local check of 2025 men's games, 99.98% found
  their `pv.parquet` row and 99.97% their leak-free one. Non-Division I opponents are the pooled `Non-D1` team and have no
  player values, so those sides carry the missing flag (8.4% of 2025 men's games have one).
- **Swap test** (`scripts/college_lineup_swap.py`, run on `pbp32_pvc_s0` and `prod_pvc_s0`, 2026): the model is fit on
  the seasons before 2026, then every game of 2026 is re-predicted with one change to the listed (home)
  team's projected rotation. The team's G, F or C with the most projected minutes becomes the group's
  best / worst regular (offense plus defense value, 600+ minutes on the floor in the fit window, a 2026 player), a
  replacement-level player (value 0), or is ruled out (the others' minutes scale up to 200, none above
  38). The change goes through the week's values into the `_pv_*` row. A position passes when the predicted
  margin and win probability move with the value in every game where it moved by more than 0.05
  points. By default every second game day is predicted (`swap-stride`).
- **Scoring:** paired per-game Brier difference to the baseline on moneyline, spread and total (95% CI), all
  games including neutral sites (12% of 2025 men's games; `--all-games`), plus the games whose rotation moved most.
  The market columns are the lines in the games table.

### Results

**Not run yet.** The tables fill from the run's `wcbb-wager-results` artifact (`e4/report.md`,
`e4/subsets.md`, `single/report.md`, `swap/*.md`, and the run's step summary).

Paired Brier difference vs `prod_e4` (negative is better), ± 95% CI half-width, 4 seeds:

| arm | n | Brier (moneyline) | Δ moneyline | Δ spread | Δ total | margin RMSE |
|---|---|---|---|---|---|---|
| prod_e4 (baseline) | pending | | | | | |
| pbp32_e4 | | | | | | |
| pbp32_pvc_e4 | | | | | | |
| prod_pvc_e4 | | | | | | |
| prod_pvc_prev_e4 (leak-free rotation) | | | | | | |
| market | | | | | | |

Promotion bar (`docs/multisport/PLAN.md` section 3): on every market the 95% CI includes 0 or favors the arm.

Single seed (`single/report.md`): `pbp32`, `pbp32_pvc`, `prod_pvc`, `prod_pvc_prev`, `prod_pvc_w6` against
`prod`. Pending.

Swap test (2026, 1 seed): pending.

| group | best regular (Δ margin) | worst regular | replacement | out |
|---|---|---|---|---|
| G | | | | |
| F | | | | |
| C | | | | |

### Running it

Run `College wager experiment` from the Actions tab (it has no push trigger, so editing the
configuration or the workflow does not start it):
- `gender`: `wcbb`. Run it once for each gender; the artifacts are named by gender.
- `cbbd`: not used for women's.
- `configs`: a JSON list of arm names to run a subset, e.g. `["prod_s0", "prod_pvc_s0"]` for a first look.
- `swap-stride`: 2 by default; 1 predicts every game day.

It rebuilds every input in the run: `college-experiment-inputs.yml` (play-by-play, engine events and the
five foundation folds 2022-2026, about 2 hours each), the games table, the projected rotations and the weekly
values. About 105 arm jobs (21 configurations x 5 seasons) at up to 355 minutes each. One cell locally:

```bash
python scripts/wager_arm.py prod_pvc_s0 2025 --sport wcbb --experiment experiments/wcbb/wager.yaml
```

(with `data/wcbb/games.csv` and `player_values/wcbb/pv.parquet` in place; `pbp32` arms also need
`runs/foundation/fold2025/`).

### Next steps

1. Run the workflow for both genders and fill the tables above.
2. If an arm passes: the NBA's switch (`RUNITBACK_NBA_PLAYER_VALUES`) is the pattern for turning on the wcbb
   lineup terms in production; that is a production setting and is not part of this package.
3. C3's next steps (better shrinkage for changes) matter most here: the weight the fit learns, printed by the
   swap test, says how far the values are from what a starter's absence costs.
