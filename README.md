# Run It Back

Simulate sports games one play at a time, or replay a matchup a thousand times.

Run It Back is built to cover several sports. Each sport has its own package with its rules,
data and vocabulary (`runitback.nfl`). Shared pieces live in `runitback.core`: the transformer
building blocks and the score-distribution summaries. **NFL** is the first sport.

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
| API / server | `src/runitback/nfl/sim/api.py`, `src/runitback/nfl/server/app.py` | `simulate_games`, FastAPI |
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

## Quick start (pre-trained model)

You need Python 3.10+ and Node 18+. This takes about 5 minutes and needs no data download or training.

```bash
git clone https://github.com/mefortunato/runitback && cd runitback
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e .
(cd web && npm install && npm run build)
runitback nfl fetch-model     # downloads the v0.1.0 release to runs/nfl/prod/model.pt, checks its SHA-256
runitback nfl serve           # open http://127.0.0.1:8000
```

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
runitback nfl serve --ckpt runs/nfl/prod/model.pt   # http://127.0.0.1:8000 (serves web/dist)
# frontend dev: (cd web && npm run dev)    # Vite on :5173, proxies /api to :8000
```
In the **Season** view, select a season and run. The backend generates a full schedule: home-and-away
division games, rotating full-division conference and interconference matchups, same-finish-position
games, and the 17th interconference matchup from 2021 onward. Division rotations follow three-year
and four-year cycles. Same-finish positions use the previous season's processed results when
available; otherwise checkpoint preseason Elo provides the ranking proxy, which the UI identifies.
Each team receives a bye during the midseason window. The app runs the era-appropriate 12- or
14-team playoffs and automatically saves trainer-format Parquet files under
`data/nfl/synthetic/<season>/<job-id>`.

Pass the returned dataset path directly to training:

```bash
runitback nfl train --data data/nfl/synthetic/2027/<job-id> --out runs/nfl/finetune
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
curl -X POST localhost:8000/api/simulate -H 'Content-Type: application/json' \
  -d '{"home":"KC","away":"BUF","season":2026,"n":1000}'
```

Interactive endpoints: `POST /api/games`, `POST /api/games/{id}/step {play_type?, count?, temperature?}`,
`POST /api/games/{id}/undo`, `GET /api/games/{id}`, `GET /api/games/{id}/winprob?n=200`, `GET /api/teams`.

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
legality masks match sampling-time masks, sampler termination, and the API.

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
- **Adding a sport** (e.g. `runitback.cfb` via cfbfastR, `runitback.nba` via hoopR) means writing
  that sport's vocabulary, rules engine, preprocessing and outcome heads. The transformer trunk and
  the summaries in `runitback.core` are reused. The NFL sampler and server are the template; the
  next refactor is to move their sport-independent parts into `core` as the second sport lands.
