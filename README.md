# Run It Back

Simulate sports games one play at a time, or replay a matchup a thousand times.

Run It Back covers several sports. Each sport has its own package with its rules, data and
vocabulary (`runitback.nfl`, `runitback.nba`). Shared pieces live in `runitback.core`: the
transformer building blocks, the score-distribution summaries, the model scheduler and training
helpers. One server (`runitback.server`) serves every sport's API and the web app:

- `/`: a landing page that explains the model and links to each sport
- `/nfl`: the NFL game, forecast and season simulators
- `/nba`: the NBA game and forecast simulators

## NFL

Generates NFL games one play at a time. A causal transformer reads the game so far and samples
the **next play's outcome**. A deterministic **rules engine** applies that outcome to the game
state (down and distance, field position, possession, score, clock, timeouts, halves, overtime),
so every simulated game is legal.

- **Web app**: simulate a game play by play, call plays yourself, auto-play, undo, and watch a
  live win-probability chart. You can also forecast a matchup over 1,000 simulated games.
- **External use**: sample full games to get outcome distributions through a Python API, an HTTP
  endpoint, or the CLI.

Data comes from [SportsDataverse](https://www.sportsdataverse.org/) (`sportsdataverse.nfl.load_nfl_pbp`,
which reads the nflverse/nflfastR play-by-play releases), 1999–present.

### How it works

| Piece | File | Notes |
|---|---|---|
| Vocabularies | `src/runitback/nfl/vocab.py` | play types, results, and the meaning of the yards field |
| Rules engine | `src/runitback/nfl/engine/rules.py` | `apply_outcome(state, outcome)`, legality, OT rules by season |
| Preprocessing | `src/runitback/nfl/data/preprocess.py` | nflverse rows → (state, outcome); outcomes are derived from the *next* play's state |
| Shared trunk | `src/runitback/core/transformer.py` | causal block with a preallocated KV cache (sport-agnostic) |
| Model | `src/runitback/nfl/model/transformer.py` | one position per play; heads factorized as type → result → yards → clock |
| Sampler | `src/runitback/nfl/sim/sampler.py` | batched rollouts with a KV cache and legal-outcome masks |
| API / server | `src/runitback/nfl/sim/api.py`, `src/runitback/nfl/server/app.py` | `simulate_games`, FastAPI routes under `/api/nfl` |
| Web | `web/` | Vite + React + TypeScript |

**Outcome per play** (each field is sampled conditioned on the ones before it):
`play_type` (kickoff, run, pass, punt, field_goal, extra_point, two_point, qb_kneel, qb_spike, penalty, timeout) →
`result` (e.g. complete/incomplete/sack/interception, kick_received/kick_recovered, good/no_good) →
`yards` (net gain for the offense; for kicks and turnovers, the spot where the new team takes over, where 0 means a return touchdown) →
`elapsed` (game-clock seconds).

**Engine replay check:** replaying every real 1999–2026 play through the engine reproduces the
next real state (down, distance, field position, possession, score, quarter) on **99.55%** of
1.24M transitions.

**Per-play input features:** quarter, clock, phase, down, distance, field position, score
difference, timeouts, home/away, offense and defense team, offense and defense **team-season**
embeddings (unseen team-seasons fall back to an UNK slot that is trained via dropout), season,
regular season vs. playoffs, the previous play's outcome, and five continuous **team-strength
ratings**.

**Team-strength ratings** (`src/runitback/nfl/data/ratings.py`) are computed in date order, so each game
only sees results from before it:
- Each team's offensive EPA/play and defensive EPA/play allowed, as exponentially weighted
  averages (α=0.15 per game).
- A 538-style Elo rating (K=20, home advantage 48, margin-of-victory multiplier).
- At the start of each season, all three regress one third of the way back to the league mean.

The model sees them from the possession team's side: its offense and defense, the opponent's
offense and defense, and the Elo difference. For future games the simulator uses each team's
latest ratings; `ratings={"KC": {...}}` overrides them for what-ifs. `strength_scale` stretches
the rating gaps; production uses 1.5, which roughly matches the market's dispersion.

### Evaluation (held-out: 2025 week 10+ and playoffs, 150 games, 200 sims each)

| | v1 (team embeddings) | v2 (+ ratings) | Elo alone | Vegas spread |
|---|---|---|---|---|
| Brier (coin flip 0.25) | 0.259 | 0.233 | 0.221 | 0.215 |
| Picked winner | 52% | 58% | – | 63% |
| Margin error (pts) | 11.0 | 10.1 | – | 9.5 |
| Correlation with spread | 0.22 | 0.80 | – | – |

Next-play log-loss is 5.84 nats vs 6.42 for an empirical baseline. The simulated games' plays,
passes, punts, FG attempts, turnovers and points per game are all within a few percent of the
real games. Stretching the ratings (`strength_scale` 2.0) widened the predicted margins but left
Brier unchanged (0.232), so the remaining gap to Elo and the spread is about missing information,
not calibration. 150 games is a small test set (Brier standard error about ±0.01).

## NBA

Generates NBA games one **event** at a time with the same design: a causal transformer samples
what the team with the ball does next, and a deterministic rules engine applies it.

- **Web app** (`/nba`): step an event or a whole possession at a time, simulate to the end of the
  quarter or game, call 2s, 3s and timeouts, auto-play, undo, and follow the scoreboard, shot
  chart, line score, team stats and live win probability. Forecast a matchup over 1,000 games.
- **External use**: Python API, HTTP endpoints under `/api/nba`, and the CLI.

Data comes from [SportsDataverse](https://www.sportsdataverse.org/) (`sportsdataverse.nba.load_nba_pbp`,
ESPN play-by-play published by hoopR), 2002–present. Seasons are named by the year they end in
(2026 = 2025-26), as in the data. Team abbreviations are the NBA's (`GSW`, `NYK`, `SAS`, ...);
ESPN and historical ones (`GS`, `NJ`, `SEA`, ...) are mapped to the current franchise.

| Piece | File | Notes |
|---|---|---|
| Vocabularies | `src/runitback/nba/vocab.py` | event types, results, phases, teams |
| Rules engine | `src/runitback/nba/engine/rules.py` | `apply_outcome(state, outcome)`, legality, quarters, OT |
| Preprocessing | `src/runitback/nba/data/preprocess.py` | ESPN rows → (state, event); results derived from what happens next |
| Ratings | `src/runitback/nba/data/ratings.py` | efficiency vs league average (EWMA) and Elo |
| Model | `src/runitback/nba/model/transformer.py` | core trunk; heads type → result → clock |
| Sampler / API / routes | `src/runitback/nba/sim/`, `src/runitback/nba/server/app.py` | `simulate_games`, FastAPI |
| Web | `web/src/nba/` | page, court/shot chart, box score |

**Events** (`play_type` → `result` → `elapsed` seconds):
`jump_ball` (won/lost) · `shot2`, `shot3` (made, made_fouled = and-one, missed_off, missed_def,
missed_fouled) · `free_throw` (made, missed, or on the last of a trip missed_off/missed_def) ·
`turnover` · `foul` (non-shooting foul that sends the offense to the line for 1–3 free throws) ·
`technical` (a technical free throw made by either team, or missed) · `timeout` (offense/defense) ·
`run_out` (the team holds the ball until the period ends). Rebounds are folded into the miss they
follow, so possession always follows from the outcome. Side-out fouls, substitutions and
violations that don't change possession are not events; their clock time goes to the next event.

**Engine rules**: 12-minute quarters and 5-minute overtimes; the team that loses the opening tip
starts the 2nd and 3rd quarters and the winner the 4th; every overtime starts with a jump ball;
free throws awarded at the buzzer are shot before the period ends; 7 timeouts per game, at most
4 in the 4th quarter, 2 per overtime; no ties.

**Engine replay check:** replaying every real 2021–2027 event through the engine reproduces the
next real state (period, phase, possession, free throws left, score) on **99.48%** of 2.2M
transitions, and every kept game ends at its real final score.

**Per-event input features:** period, clock, phase, free throws left, score difference, timeouts,
home/away, offense and defense team and team-season embeddings, season, regular season vs.
playoffs, the previous event, and five **team-strength ratings** from the offense's point of
view: its offensive and defensive efficiency, the opponent's, and the Elo difference. Efficiency
is points per 100 possessions relative to the league average so far that season, as an
exponentially weighted average (α=0.08); Elo is 538-style (K=20, home advantage 100,
margin-of-victory multiplier). Both regress a quarter of the way to the mean between seasons.

### Evaluation (smoke run)

A small model (d_model 128, 3 layers, 0.8M params, 3 epochs, about 12 minutes on a 4-core CPU)
trained on 2020-21 to the first 60% of 2025-26, evaluated on 150 games from the rest of 2025-26
with 100 simulations each:

| | Model | Elo alone | Spread |
|---|---|---|---|
| Brier (coin flip 0.25) | 0.216 | 0.200 | 0.186 |
| Picked winner | 66% | 73% | 69% |
| Margin error (pts) | 13.2 | – | 12.3 |

Simulated games match real ones closely (per game, both teams, simulated vs. real): 283 vs 286
events, 108 vs 107 two-point attempts, 69 vs 69 three-point attempts (26 vs 26 made), 43 vs 45
free throws, 27 vs 28 turnovers, 30 vs 30 offensive rebounds, 225 vs 228 total points. Predicted
margins correlate 0.77 with the spread. As with the NFL, the win-probability gap to Elo and the
spread is mostly about information the model doesn't have (rosters, injuries, rest); a bigger
model, more seasons and `calibrate` are the obvious next steps.

### NBA pipeline

```bash
pip install -e ".[data,dev]"
runitback nba download --start 2015 --end 2027      # raw parquet → data/nba/raw/
runitback nba preprocess --start 2015 --end 2027    # → data/nba/processed/{games,plays,ratings}.parquet, prints engine agreement
runitback nba train --out runs/nba/eval --val-season 2026        # hold out the last 40% of 2025-26
runitback nba evaluate --ckpt runs/nba/eval/model.pt
runitback nba calibrate --ckpt runs/nba/eval/model.pt --season 2025   # strength_scale by Brier
runitback nba train --out runs/nba/prod                           # production model on all data
runitback serve                                                   # http://127.0.0.1:8000/nba
runitback nba simulate BOS NYK --season 2027 --n 1000
```

```python
from runitback.nba import Simulator, simulate_games
sim = Simulator.load("runs/nba/prod/model.pt")
r = simulate_games(sim, home="BOS", away="NYK", season=2027, n=1000, seed=0)
print(r.home_win, r.home_score_mean, r.away_score_mean, r.margin_quantiles)
```

HTTP: `POST /api/nba/simulate {home, away, season, n, before?, ratings?}`, and interactive
`POST /api/nba/games`, `POST /api/nba/games/{id}/step {play_type?, count?, until?: "possession" | "period"}`,
`POST /api/nba/games/{id}/undo`, `GET /api/nba/games/{id}`, `GET /api/nba/games/{id}/winprob`,
`GET /api/nba/teams` (teams always; seasons once a model is loaded).
Set `RUNITBACK_NBA_DATA`, `RUNITBACK_NBA_RUNS` or `RUNITBACK_NBA_CKPT` to move the defaults.

**NBA limitations:** team-level only (no players, injuries or rest); no season simulator yet
(82-game schedule, play-in and best-of-seven playoffs); shot locations on the court are
illustrative because the model predicts a shot's value and result, not where it was taken; rare
sequences are simplified (flagrant and clear-path free throws followed by possession, fouls on
the offense during a loose ball, and coach's challenges).

## Quick start (pre-trained model)

You need Python 3.10+ and Node 18+. This takes about 5 minutes and needs no data download or training.

```bash
git clone https://github.com/mefortunato/runitback && cd runitback
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e .
(cd web && npm install && npm run build)
runitback nfl fetch-model     # downloads the v0.1.0 release to runs/nfl/prod/model.pt, checks its SHA-256
runitback serve               # open http://127.0.0.1:8000
```

No NBA model is released yet. Until you train one (see [NBA](#nba)), the NBA page shows the
commands to do so and the rest of the app works as usual.

The released model (see [Releases](https://github.com/mefortunato/runitback/releases)) includes team
ratings through 2026 week 7. Retrain (below) to pick up later games.

## Training from scratch

```bash
pip install -e ".[data,dev]"   # adds sportsdataverse and test tools
```

### Pipeline

```bash
runitback nfl download --start 1999 --end 2026     # raw parquet → data/nfl/raw/
runitback nfl preprocess --start 1999 --end 2026   # → data/nfl/processed/{games,plays,ratings}.parquet, prints engine agreement
# held-out evaluation run (validation = 2025 week 10+ and playoffs)
runitback nfl train --out runs/nfl/eval --val-season 2025 --epochs 10
runitback nfl evaluate --ckpt runs/nfl/eval/model.pt --sims 200
runitback nfl calibrate --ckpt runs/nfl/eval/model.pt --season 2024   # grid-search strength_scale by Brier
# production model on all data
runitback nfl train --out runs/nfl/prod --epochs 10
```

The default model (d_model 192, 4 layers, 2.6M params) trains on a 4-core CPU in about 40
minutes. Use `--d-model/--n-layers` for larger models on a GPU.

### Use it

**Web app**

```bash
runitback serve --nfl-ckpt runs/nfl/prod/model.pt --nba-ckpt runs/nba/prod/model.pt   # http://127.0.0.1:8000 (serves web/dist)
# frontend dev: (cd web && npm run dev)    # Vite on :5173, proxies /api to :8000
```
In the **Season** view, pick a season (2002–2040, the 32-team era) and run. The backend builds the
schedule from the NFL formula: home-and-away division games; one same-conference division and one
other-conference division in full, on the league's three- and four-year rotations; the same-place
finishers of the other two same-conference divisions; and, from 2021, a 17th game against the
same-place finisher of the other-conference division the team plays in full two seasons later
(hosted by the AFC in odd years). Finish positions come from the previous season's results, read
from `data/nfl/processed` or the raw play-by-play, but only when that season is complete; otherwise
checkpoint preseason Elo stands in, and the UI says which source was used. Each team gets one bye in
the midseason window and the final week is all division games. Standings use the NFL tiebreakers
(head-to-head, division, common games, conference, strength of victory/schedule, net points, coin
toss). The app then runs the era's 12- or 14-team playoffs and saves trainer-format Parquet files
under `data/nfl/synthetic/<season>/<job-id>` (set `RUNITBACK_NFL_DATA` to move `data/nfl`).

Team strengths are each team's preseason rating, held fixed for the season (*checkpoint*), shuffled
between teams (*randomized*) or league average (*neutral*). The exported `ratings.parquet` holds
these unscaled ratings with the same columns as the processed data.

To fine-tune on simulated seasons, combine them with the real data and start from a checkpoint
(`--data` can be repeated). Synthetic seasons train the model but are kept out of the checkpoint's
pre-game rating lookups. Fine-tuning does not carry over `strength_scale`, so re-run `calibrate`:

```bash
runitback nfl train --init runs/nfl/prod/model.pt --data data/nfl/processed \
  --data data/nfl/synthetic/2027/<job-id> --out runs/nfl/finetune --epochs 2 --lr 2e-4
```

**Python**

```python
from runitback.nfl import Simulator, simulate_games
sim = Simulator.load("runs/nfl/prod/model.pt")
r = simulate_games(sim, home="KC", away="BUF", season=2026, n=1000, seed=0)
print(r.home_win, r.home_score_mean, r.away_score_mean, r.margin_quantiles)
r.scores   # [(home, away), ...] for every simulated game
```

To continue a game in progress, call `sim.rollout(state, n, history=[(state, outcome), ...])`.

**HTTP**

```bash
curl -X POST localhost:8000/api/nfl/simulate -H 'Content-Type: application/json' \
  -d '{"home":"KC","away":"BUF","season":2026,"n":1000}'
```

Interactive endpoints (all under `/api/nfl`): `POST /games`, `POST /games/{id}/step {play_type?, count?, temperature?}`,
`POST /games/{id}/undo`, `GET /games/{id}`, `GET /games/{id}/winprob?n=200`, `GET /teams`,
`POST /seasons`, `GET /seasons/{id}`. The same NFL routes are still served without the `/nfl`
prefix (`/api/simulate`, ...) for clients written before the NBA was added.

### Deploy to Google Cloud Run

The repository includes a Docker image that builds the web app, installs the CPU version of PyTorch,
and downloads the checksum-verified `v0.1.0` NFL model during the image build. It runs the combined
server (`runitback.server:app`); the NBA page loads `RUNITBACK_NBA_CKPT`
(`/app/runs/nba/prod/model.pt`) if the image contains one. Install and authenticate
the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), then create and edit the deployment
environment file:

```bash
gcloud auth login
cp .env.example .env
# Set GOOGLE_CLOUD_PROJECT in .env to your GCP project ID.
./deploy-cloud-run.sh
```

The deploy helper reads `GOOGLE_CLOUD_PROJECT`, `CLOUD_RUN_REGION`, `CLOUD_RUN_SERVICE`, and
`CLOUD_RUN_ALLOW_UNAUTHENTICATED` from `.env`. It deploys from source using Cloud Build. Authentication
is required by default; set `CLOUD_RUN_ALLOW_UNAUTHENTICATED=true` to make the service public. Enable
Cloud Build for the project before using this helper. It limits the service to one instance because
interactive game sessions are held in process memory; active sessions are lost if Cloud Run restarts
that instance.

### GitHub Actions deployment

`.github/workflows/deploy-cloud-run.yml` builds the Docker image on the GitHub-hosted runner, pushes
it to Artifact Registry, and deploys it to Cloud Run when a commit is pushed to `main` (or when
manually started with `workflow_dispatch`). It uses Workload Identity Federation rather than a
long-lived service-account key. The deployed service remains private, and the workflow limits it to
one instance.

One-time GCP setup:

1. Enable the Cloud Run, Artifact Registry, IAM Credentials, and Security Token Service APIs.
2. Create an Artifact Registry Docker repository in the Cloud Run region.
3. Configure Workload Identity Federation for GitHub Actions, restricted to this repository, and
   allow its principal to impersonate a deployment service account. Grant that service account
   `roles/artifactregistry.writer`, `roles/run.admin`, and `roles/iam.serviceAccountUser` for the
   Cloud Run runtime service account.
4. In the GitHub repository's **Settings → Environments**, create an environment named
   `production`. Under **Deployment branches and tags**, restrict it to `main` (and optionally add
   required reviewers). Add these six **environment secrets** to it:
   `GOOGLE_CLOUD_PROJECT` (project ID), `CLOUD_RUN_REGION` (for example `us-central1`),
   `CLOUD_RUN_SERVICE` (for example `runitback`), `GAR_REPOSITORY` (Artifact Registry repository
   name), `GCP_WORKLOAD_IDENTITY_PROVIDER` (full provider resource name,
   `projects/<PROJECT_NUMBER>/locations/global/workloadIdentityPools/<POOL>/providers/<PROVIDER>`),
   and `GCP_DEPLOY_SERVICE_ACCOUNT` (deployment service-account email).
   The deploy job runs in the `production` environment, so it reads these secrets from there and is
   subject to the environment's protection rules. Repository-level secrets with the same names are
   overridden by the environment's values, so delete any old repository-level copies. If settings are
   missing, the validation step names each missing secret before authentication or deployment starts.
5. Allow the GitHub principal to impersonate the deployment service account. Because the job uses
   an environment, GitHub's OIDC token subject is `repo:mefortunato/runitback:environment:production`
   (not `...:ref:refs/heads/main`). If the provider maps `attribute.repository=assertion.repository`,
   grant:

   ```bash
   gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
     --role=roles/iam.workloadIdentityUser \
     --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/attribute.repository/mefortunato/runitback"
   ```

   A missing or mismatched binding makes the authentication step fail with
   `Permission 'iam.serviceAccounts.getAccessToken' denied`. If the provider has an attribute
   condition or binds on `google.subject`, update it to match the environment subject above.
   The deployment service account needs `roles/artifactregistry.writer` on the Artifact Registry
   repository (or its project) to push images. The workflow logs Docker in with the federated
   Google access token; if pushes still fail with an upload permission error, verify that role
   binding and that `GAR_REPOSITORY` and `GOOGLE_CLOUD_PROJECT` identify the intended repository.

The workflow's image tag is the commit SHA, so each deployment uses the exact image built for that
commit. No Cloud Build is used by the GitHub Actions workflow.

Interactive next-play inference takes priority over win-probability and forecast rollouts, which
yield between model calls so background simulations do not hold up the game.

**CLI**

```bash
runitback nfl simulate KC BUF --season 2026 --n 1000 --out results.json   # or .csv (one row per game)
```

## Tests

```bash
pytest
```

The suite covers engine rules (scoring, downs, penalties, halftime, PATs after time expires, and
the three OT eras), a replay of real 2024 games through the engine, a check that training-time
legality masks match sampling-time masks, sampler termination, and the API. The NBA tests cover the
engine (and-ones, free-throw trips, quarter possession, OT, timeouts), ratings, a replay of real
games when `data/nba/raw` exists, mask agreement when `data/nba/processed` exists, the sampler,
the NBA API, and the combined server's routes.

## Limitations / next steps (NFL)

- Team-level only, with no players or injuries. A starting-QB change is the biggest signal this
  misses, and team ratings take several games to catch up to one.
- Win-probability accuracy is still below Elo alone and the Vegas spread. Next steps: a
  walk-forward evaluation over full seasons (2021–2025, about 1,400 games), then a starting-QB
  rating feature (nflverse depth charts, weekly rosters and injuries are all available before
  kickoff).
- Some rare situations are simplified: untimed downs, penalty-forced re-kicks, defensive two-point
  returns, the 2024+ dynamic kickoff spot (the model learns the landing spots, but the field view
  shows the old 35-yard tee), and playoff OT coin tosses.
- **Adding a sport** (e.g. `runitback.cfb` via cfbfastR) means writing that sport's vocabulary,
  rules engine, preprocessing and outcome heads, as `runitback.nba` does. The transformer trunk,
  summaries, scheduler and batching in `runitback.core` are reused; the NFL and NBA samplers and
  servers are near-identical templates and are the next candidates to move into `core`.
