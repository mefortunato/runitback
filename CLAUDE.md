# Run It Back: notes for Claude sessions

Sports game simulators (`runitback.nfl`, `runitback.nba`; shared pieces in `runitback.core`) and
game-level wager models for spread / moneyline / total (`runitback.wager`, ported from
`mefortunato/sportsml`). The server (`runitback.server`) serves every sport's API and the web app
(`web/`, Vite + React + TypeScript). See `README.md` for the full pipeline.

## Setup and checks

- Install: `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then
  `pip install -e ".[data,dev,wager]"`.
- Tests: `pytest` (or the relevant subset: `tests/nfl`, `tests/nba`, `tests/wager`, `tests/core`).
  Some tests skip when `data/<sport>/raw` or `processed` is absent.
- Web: `cd web && npm ci && npm run build` (type-checks with `tsc -b`).
- Before calling work done, run the tests for what you changed and paste the summary line.

## Multi-sport player model

Work on bringing the NFL player-based wager model to the other sports follows
`docs/multisport/PLAN.md`. Each work package (S1, N2, H4, ...) has a GitHub issue titled with its
ID. Read the plan's sections 1–3 and 6, and `experiments/nfl/README.md`, before starting one.

- **One package per session, one draft PR per package.** Title it `[<ID>] <title>` and put
  `Closes #<issue>` in the body.
- **Branch from `main`.** If a dependency isn't merged yet, branch from its branch and say so in the PR.
- **Follow the package's deliverable and acceptance criteria.** If the data or results contradict
  the plan, stop and report what you found instead of redesigning the package.
- **Refactors of NFL code** (S1, S2, S3) must leave NFL outputs identical; prove it with a test.
- **Experiment packages:** report paired Brier deltas with 95% CIs against the sport's production
  baseline (4-seed ensembles for promotion decisions), plus the swap-a-starter test, in
  `experiments/<sport>/README.md` in the NFL write-up's format.

## Rules that always apply

- **No leakage.** Every fit, PCA projection and projected lineup uses only data from before the game
  it is for.
- **No result data in git.** Experiment outputs (CSV, parquet, JSON) are uploaded as workflow
  artifacts, never committed (`.gitignore` blocks `experiments/**/*.csv` etc.). Workflows rebuild
  their inputs from the public sources in the same run; follow
  `.github/workflows/nfl-experiment-inputs.yml`. Don't download artifacts of earlier runs.
- **Heavy training** (foundation folds, multi-seed backtests) runs in GitHub Actions matrix jobs,
  modeled on `.github/workflows/foundation-experiment.yml`, not in a session's container.
- **Workflow triggers cost hours.** Pushing to a `claude/**` branch a change to an experiment's
  configuration (`experiments/<sport>/*.yaml`) or to `.github/workflows/*` starts that workflow.
  Do it only when you mean to run it; otherwise put `[skip ci]` in the commit message.
- **Missing access.** If an API key (e.g. `CBBD_API_KEY`), a network host or a repository is
  missing, say exactly what in your first message and stop. Don't mock or substitute data.
- **Ask first** before merging, force-pushing shared branches, or changing production settings
  (`RUNITBACK_*` variables, `deploy-cloud-run.yml`, `train-predictions.yml`,
  `refresh-predictions.yml`).
