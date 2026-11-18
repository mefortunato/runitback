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

**Short answer** (run 44214590965 of `college-wager-experiment.yml`, `gender: wcbb`; 4-seed ensembles against `prod_e4`, 10,973
games over 2022-2026, neutral sites included; `cbbd` has no effect on the women's values)
- **`prod_pvc_e4` improves all three markets.** Moneyline **-0.0035 ± 0.0007**, spread **-0.0036 ± 0.0010**, total
  **-0.0039 ± 0.0006** (paired Brier, 95% CI, negative is better), better than the NBA's gain on the total and about the same on the
  others. It is better in every season (moneyline -0.0004 to -0.0052; 2022 and 2026 only slightly).
- **`pbp32_pvc_e4`, the plan's arm, does not pass:** total +0.0024 ± 0.0012 (worse than production), moneyline +0.0004 ± 0.0010 and
  spread -0.0008 ± 0.0012 neutral. `pbp32_e4` alone is neutral on all three. As in the NBA, the foundation embeddings add nothing and the player
  values carry the gain.
- **The leak accounts for about a quarter of the gain.** On the leak-free rotation (`prod_pvc_prev_e4`) the gains are moneyline
  -0.0026 ± 0.0006, spread -0.0024 ± 0.0009 and total -0.0039 ± 0.0007: still significant on every market.
- **The gain is in games where the rotation moved.** `prod_pvc_e4` moneyline: -0.0070 ± 0.0019 on the 2,800 games whose projected
  rotation moved most, -0.0019 ± 0.0008 on the 5,097 steadiest. Spread: -0.0108 ± 0.0027 and -0.0010 ± 0.0010.
- **The swap test passes** for both swapped arms (100% of 2026 games at G, F and C). A regular-to-replacement swing is about 2 points of
  margin at G and F and 1 point at C.
- **The learned weight is above the start.** From 0.2, the fits ended at 0.32 (`pbp32_pvc_s0`) and 0.40 (`prod_pvc_s0`), between C3's
  0.14 (within a season) and 0.47 (season openers). Starting at 0.6 (`prod_pvc_w6`) is within noise of 0.2.
- **Far behind the market.** `prod_pvc_e4` trails the closing moneyline by +0.0151 ± 0.0023 Brier (`prod_e4`: +0.0186 ± 0.0024), more than
  the men's gap.
- **Single seeds are unusable here.** One seed of `pbp32_pvc` shows -0.0125 on moneyline and one seed of `prod_pvc` +0.0219 on the
  total, against +0.0004 and -0.0039 for the 4-seed ensembles. The single-seed intervals are paired per game and do not include
  seed-to-seed variation, so read only the `_e4` tables.

Results: `e4/report.md`, `e4/subsets.md`, `single/report.md` and `swap/*.md` in the run's `wcbb-wager-results` artifact.

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

Paired Brier difference vs `prod_e4` (negative is better), ± 95% CI half-width, 4 seeds, 2022-2026. "fails" means the interval is above 0.

| arm | n | Brier (moneyline) | Δ moneyline | Δ spread | Δ total | margin RMSE |
|---|---|---|---|---|---|---|
| prod_e4 (baseline) | 10,973 | 0.1707 | | | | 13.89 |
| pbp32_e4 | 10,973 | 0.1705 | -0.0003 ± 0.0007 | -0.0001 ± 0.0010 | -0.0003 ± 0.0013 | 13.88 |
| pbp32_pvc_e4 | 10,973 | 0.1711 | +0.0004 ± 0.0010 | -0.0008 ± 0.0012 | **+0.0024 ± 0.0012 (fails)** | 14.00 |
| **prod_pvc_e4** | 10,973 | **0.1673** | **-0.0035 ± 0.0007** | **-0.0036 ± 0.0010** | **-0.0039 ± 0.0006** | 13.68 |
| prod_pvc_prev_e4 (leak-free rotation) | 10,973 | 0.1682 | -0.0026 ± 0.0006 | -0.0024 ± 0.0009 | -0.0039 ± 0.0007 | 13.74 |
| market | 10,932 | 0.1533 | | | | 12.63 |

Promotion bar (`docs/multisport/PLAN.md` section 3): on every market the 95% CI includes 0 or favors the arm. `prod_pvc_e4`,
`prod_pvc_prev_e4` and `pbp32_e4` meet it; `pbp32_pvc_e4` does not (total). Against `prod_pvc_e4` as the baseline, `prod_pvc_prev_e4` is worse
on moneyline (+0.0009 ± 0.0004) and spread (+0.0012 ± 0.0006) and equal on the total; `pbp32_pvc_e4` is worse on all three
(+0.0039, +0.0028 and +0.0063, each ± 0.0009-0.0014).

Moneyline Δ Brier by season, `prod_pvc_e4`: 2022 -0.0008, 2023 -0.0052, 2024 -0.0032, 2025 -0.0024, 2026 -0.0004. `pbp32_pvc_e4`: +0.0093, -0.0012,
-0.0009, +0.0002, +0.0007. The women's play-by-play is thin before 2020 (C2), so the 2022 fold trained on little.

**Games where the rotation moved** (Δ vs `prod_e4`; `rotation_moved` is the top quarter of games by the summed change in both teams' lineup
values, `steady` the bottom half):

| market | arm | all | rotation_moved (n about 2,800) | steady (n about 5,090) |
|---|---|---|---|---|
| moneyline | prod_pvc_e4 | -0.0035 ± 0.0007 | -0.0070 ± 0.0019 | -0.0019 ± 0.0008 |
| moneyline | prod_pvc_prev_e4 | -0.0026 ± 0.0006 | -0.0038 ± 0.0015 | -0.0018 ± 0.0008 |
| spread | prod_pvc_e4 | -0.0036 ± 0.0010 | -0.0108 ± 0.0027 | -0.0010 ± 0.0010 |
| spread | prod_pvc_prev_e4 | -0.0024 ± 0.0009 | -0.0053 ± 0.0022 | -0.0011 ± 0.0011 |
| total | prod_pvc_e4 | -0.0039 ± 0.0006 | -0.0031 ± 0.0011 | -0.0040 ± 0.0010 |

The total improves equally in moved and steady games, unlike the moneyline and spread. That is not the pattern of a rotation effect; it suggests
the values also improve the scoring-level estimate for every team, which is not a lineup-change effect.

**Single seed** (`single/report.md`) is too noisy to use: `pbp32_pvc` shows -0.0125 ± 0.0018 on moneyline and `prod_pvc` +0.0219 ± 0.0019 on the total, where the
4-seed ensembles show +0.0004 and -0.0039. The intervals are paired per game and leave out seed-to-seed variation, which is several times larger
here than the men's. `prod_pvc_w6` against `prod_pvc`: moneyline +0.0002 ± 0.0003, spread -0.0006 ± 0.0005, total +0.0003 ± 0.0003.

### Swap test (2026, `prod_pvc_s0`, 3,139 games, every second game day)

One change to the listed (home) team's projected rotation; Δ margin in points (`swap/swap_2026_prod_pvc_s0.md`; `pbp32_pvc_s0` is about 20% smaller, as its learned weight is 0.32).

| group | best regular (Δ margin) | worst regular | replacement | out |
|---|---|---|---|---|
| G | +3.28 (Rachael Rose) | -2.85 | -2.27 | -0.67 |
| F | +4.63 (Sarah Strong) | -2.51 | -1.85 | -0.59 |
| C | +2.43 (Lauren Betts) | -1.33 | -0.96 | -0.07 |

Each is ± 0.03-0.08 (95% CI) and moves the right way in 100% of games where the value moved by more than 0.05 points. The predicted total moves by
at most 0.4 points.

### What it means

- **Promote `prod_pvc_e4` if anything, not `pbp32_pvc_e4`.** It passes on every market with a significant gain on each, and the gain survives
  the leak-free rotation at about three quarters of its size. Turning the terms on in production is a production setting and is not part of
  this package.
- **The foundation does not help**, and with the values it hurts the total, as in the NBA.
- **The women's model is far from the market** (+0.0151 Brier on moneyline after the values). The gap may come more from the women's data (lines only from 2021-22, thin
  play-by-play before 2020) than the lineup terms can close; this run does not test that.
- **Next:** a larger ensemble (seed variance is large here), the freshman prior and live rotations from C3's list.

### Running it

Once the workflow is on `main`, run `College wager experiment` from the Actions tab (it has no push trigger, so editing the
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

### How these results were run

Run 44214590965 (18 Nov 2026, the C4 branch at the commit that held a temporary push trigger, since a workflow that exists only on a branch cannot be dispatched through
the API; the trigger is removed in the next commit). All jobs succeeded. The artifact is kept 90 days. A rerun reproduces
these tables within seed noise; the women's `_e4` tables are the ones to compare. The runs are not committed: only this write-up is.
