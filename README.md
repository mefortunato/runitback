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
| Players, lineups, coaches | `src/runitback/nfl/data/players.py`, `src/runitback/nfl/model/lineups.py` | depth charts, per-play personnel, NEW PLAYER rule, coach records, personnel sampler |
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

### Players, lineups and coaches

Team-season embeddings and ratings move slowly, so they miss a missing starter. The model also sees
**who plays**:

- **Depth charts (game level).** Each team's lineup is built from that game's snap counts (PFR via
  nflverse, 2012+): players are ranked by snaps within 11 position groups (QB, RB, WR, TE, OL, DL,
  LB, DB, K, P, LS), up to 9 deep. Each group is pooled separately with learned per-slot weights, so
  QB1 has a channel of its own. The possession team's lineup enters every play as the offense and the
  opponent's as the defense. This shapes play calling as well as outcomes.
- **On-field personnel (play level).** For 2016–2025 the 22 players on the field for each play
  (nflverse participation) are pooled per side and position group, then added to the result, yards
  and clock heads *after* the play type is drawn. On-field players change how a play turns out, but
  don't decide what is called (a punt unit on the field would otherwise make the 4th-down call for the
  model). When simulating, a **personnel sampler** picks which depth-chart slots take the field (WR3
  vs WR4, DL rotation, special-teams units) given the play type, down, distance and field position.
  It learned these frequencies from the participation data, then fills the slots from the depth chart.
- **New players.** A player's first 3 career games, and any id the model has never seen, use a
  position-specific `NEW <group>` token (a new QB carries different uncertainty than a new long
  snapper). Veterans with too little history for their own embedding use `OTHER <group>`. Players
  who debuted before 2012 count as veterans.
- **Head coaches** (schedules, 1999+): an offense-coach and a defense-coach embedding (`NEW COACH`
  for a head coach's first 4 games or an unknown name, `OTHER COACH` for too little history), plus
  his record before the game: seasons as head coach, seasons with this team, career games and a
  win% shrunk toward .500. Careers of coaches already in charge in 1999 are truncated.
  Coordinators are not in the nflverse data.

Seasons without lineup data (before 2012) and synthetic seasons train with unknown depth charts,
and training randomly hides depth charts (10% of team-games), personnel (15% of games) and single
players (3%, swapped for `OTHER`). So the model works with any amount of lineup information.

At inference each team defaults to **its most recent game** (who played, ranked by snaps). Users
can replace any position group, reorder it, add a `"NEW"` player or swap the coach.

### Game-level box-score loss (optional)

`--box-weight w` adds a game-level term to the per-play loss:

1. **Per play, expected stats.** Under teacher forcing (the real game states), the play heads give
   P(type), P(result | type) and P(yards | type, result). Together with the engine's scoring rules,
   they give each play's expected contribution for both teams:
   - points: TD if yards ≥ yardline, return TD at spot 0, safety, FG / PAT / two-point good
   - rushing yards: designed runs (nflverse labels scrambles as pass plays)
   - passing yards: completions
   - sacks taken and turnovers

   All play types are marginalized, not just the real one. That's 8 result-head and 12 yards-head
   evaluations per position, all differentiable.
2. **Per game, sum.** Each batch row is one whole game (padded plays), so a game's predicted box
   score is a masked sum over the play axis, split by which team had the ball. That gives 5 stats
   per team plus the margin. The real box score uses the same definitions on the real outcomes; it
   matches the real final score in about 97% of team-games (2024 sample).
3. **Loss.** For each stat, (predicted − real) / σ, where σ is that stat's standard deviation
   across training games (e.g. 10 points, 52 rushing yards, 76 passing yards, 14.6 margin). The
   box term is ½·mean over games and stats of the squared values. Total loss = per-play NLL
   (mean over plays) + w × box term.

**Result: the box loss is off by default because it teaches a shortcut.** Under teacher forcing,
later plays see the real score so far, and the game-total loss rewards using it. The model learns
to shade late-game predictions so they cancel earlier misses: the sum looks accurate while each
play's prediction is biased. On the 2025 held-out games (weight 1.0, after 3 epochs):

| | margin RMSE, teacher-forced | corr(1st-half, 2nd-half margin residual) | next-play NLL (epoch 2) |
|---|---|---|---|
| no box loss (team-only and lineup models) | 11.9 | −0.02 | 5.97 |
| box loss | 7.0 | **−0.47** | 6.10 |

A calibrated model can't get below about 12 points here: each half's residual is about 8.5 points
of genuine uncertainty. `evaluate` reports this check (`margin_residual_halves`) for every
checkpoint. Any proper loss on teacher-forced per-play expectations reduces to a per-play loss.
A game-level penalty that can't be gamed has to score games the model generates itself (rollouts),
or predict the box score from pre-game information only.

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

### Lineup experiments (same held-out split, 10 epochs each, CPU)

| Model | Next-play NLL, actual lineup + on-field 22 | Next-play NLL, previous-game lineup | Brier, actual lineup | Brier, previous lineup |
|---|---|---|---|---|
| Team-only baseline | 5.840 | – | 0.234 | – |
| Lineups v1 (192-wide players, light dropout) | 5.854 | 5.874 | 0.232 | 0.239 |
| v2 (32-wide players, 0.2 lineup dropout, 20% hidden charts) | 5.840 | 5.853 | 0.238 | 0.230 |
| v3 (v2 + 30% of team-games trained on the previous game's chart) | 5.838 | 5.853 | 0.236 | 0.232 |

Paired bootstrap against the baseline (per-game, 95% intervals):

- With who actually played, v2/v3 tie the baseline on next-play NLL (v3 −0.002, interval
  −0.008..+0.004). That includes the 38 games where a team's most-used QB changed from its
  previous game.
- With the previous game's lineup (the realistic pre-kickoff input), lineup models are **worse**:
  +0.013 NLL/play (+0.02 on QB-change games), significant.
- No variant's Brier differs significantly from the baseline (all intervals include 0; 150 games
  resolve roughly ±0.02).

Ablation on v3 (next-play NLL):

| v3 input | NLL |
|---|---|
| actual lineup + actual on-field 22 | 5.838 |
| actual lineup + sampled 22 | 5.848 |
| actual lineup + no 22 | 5.853 |
| previous lineup + sampled 22 | 5.850 |
| previous lineup + no 22 | 5.853 |

**A stale depth chart costs nothing; the gap is per-play personnel.** The real on-field 22 carry
pre-snap information (goal-line and heavy packages, special units) that the model learns to rely
on. The simulator can only sample packages from league-wide frequencies, so at inference that
information is missing. Lineup what-ifs also respond weakly (v3, 400 sims, ±2.5 pts): replacing the home starter with a
`NEW` QB moves win probability by only 1–4 points in five 2026 matchups (Mahomes → Gabbert: none).
A `NEW` QB for the visiting team moves it 5–11 points, and a new head coach 0–7. Team-season embeddings and EPA ratings were
always fit with the same starter and soak up his value (collinearity), leaving little for the
player embedding to carry.

Next experiments, by expected value:
1. Drop team-season and team embeddings (and possibly the EPA ratings) for most training
   team-games whenever a lineup is known, so team strength has to flow through the players.
2. Use per-play participation only as a *training* signal for player embeddings (no on-field 22
   at inference), or teach the model to predict personnel itself so simulated packages are
   team-specific.
3. Evaluate walk-forward over several seasons (about 1,400 games) to resolve Brier differences
   of 0.005.

### Do individual defenders change what opposing players produce? (2026-11 check)

The question: does an elite DB1 hold the opposing WR1 to noticeably fewer yards than an average
DB1 would? `scripts/matchup_analysis.py OFF_GROUP DEF_POSITIONS RANK STAT` tests this from the
data alone, with no model. Roles come from snap counts (the most-used WR, the defense's most-used
CB). Every predictor uses only earlier games. Nested regressions are fit on 2013–2022 and tested
on 2023–2025 (about 1,700 team-games), with game-level bootstrap intervals:

| Pairing | Opponent team defense adds (MSE) | Specific defender's track record adds | Defender year-to-year r |
|---|---|---|---|
| WR1 rec. yards vs CB1 | +7.3 (CI +1.2..+13.7) | −0.1 (−1.3..+1.0) | 0.09 |
| WR2 rec. yards vs CB2 | +1.9 (+0.4..+3.4) | −0.1 (−0.5..+0.2) | 0.11 |
| WR1 targets vs CB1 | +0.05 (+0.01..+0.09) | +0.01 (−0.01..+0.03) | 0.00 |
| TE1 rec. yards vs top safety | +1.4 (−0.2..+3.1) | +0.3 (−1.1..+1.6) | −0.02 |
| RB1 rush yards vs top DL | +15.8 (+5.0..+26.6) | +0.0 (−0.1..+0.1) | −0.06 |
| RB1 rush yards vs top LB | +15.8 (+5.0..+26.6) | −1.1 (−2.2..−0.1) | 0.14 |

For comparison, a WR's own receiving yards have a year-to-year correlation of 0.48. Learning a
per-CB identity effect (shrunk) doesn't improve test RMSE either (37.91–38.18 vs 37.93). When a
defense's usual CB1 is out, opposing WR1s gain −3.6 ± 7.3 yards: not significant, and the
opposite sign to the hypothesis.

The trained lineup model agrees (`scripts/model_defender_effect.py`). On the 300 held-out
team-games, replacing the real DBs with generic ones shifts expected passing yards by only about
1.5 yards per team-game. Those shifts don't correlate with how actual yards differed from the
generic prediction (r = −0.02, 95% CI −0.13..+0.10). DL/LB on rushing: r = −0.08 (−0.19..+0.05).

**Conclusion:** opposing *team* defense is a real, persistent signal. With public data, the identity
of a specific defender is not: it isn't stable from season to season and doesn't predict
out-of-sample. The caveat is that "CB1 by snaps" is not "the CB covering WR1". NFL defenses rarely
shadow, and who covered whom is not in nflverse (it needs NGS/PFF-style coverage charting).
Per-player *offensive* stat lines remain promising: player form is stable (r ≈ 0.48) and opponent
team defense adds to it.

### Player stat lines: the who-gets-the-ball head (2026-11)

`--who` adds a head that picks which of the possession team's 11 on-field players gets the ball
(the rusher on runs; the target on passes, or nobody for sacks and throwaways). It runs after the
play type and personnel and before the result, which must agree with it (no catch without a target,
no sack with one). Labels come from nflverse rusher/receiver ids, and 99.9% are among the 11 on the
field. Simulated outcomes carry the chosen player's depth-chart slot (`Outcome.who`), so
rollouts produce stat lines. `scripts/player_stats_eval.py` simulates the 150 held-out games
(2025 week 10+, 100 sims each) and compares per-player targets, receptions, receiving yards,
carries and rushing yards with actual stat lines and with leak-free baselines: role average,
the player's shrunk exponentially weighted form, and form adjusted for the opponent.

RMSE, RB/WR/TE player-games who played (about 3,100–3,500). The pre-game depth chart is this
game's active players ranked by their previous-game role:

| | form + opponent | model | blend (cross-fitted) |
|---|---|---|---|
| WR targets | 2.37 | 2.82 | **2.35** (significant) |
| WR receiving yards | 26.5 | 29.6 | 26.3 (n.s.) |
| TE targets | 1.80 | 2.09 | **1.78** (significant) |
| TE receiving yards | 18.0 | 19.8 | 17.9 (n.s.) |
| RB carries | 4.19 | 5.31 | 4.16 (n.s.) |
| RB rushing yards | 25.1 | 29.5 | 25.1 (n.s.) |

- **Alone, the model is worse than form** with any pre-game depth chart, and still worse when form
  is frozen at the training cutoff like the model's embeddings. Blended, it adds ~1% on WR/TE
  targets. With the game's own snap ranks (in-game knowledge) the blend beats form on every stat,
  e.g. RB rushing yards 24.1 vs 25.2.
- **Why:** usage is too flat. RB1 gets 12.0 simulated carries vs 14.7 actual, and TE1 3.8 targets
  vs 4.8. Per-touch efficiency is about right.
- **The who head is not the bottleneck** (`scripts/who_head_teacher_forced.py`). Given the real
  plays and the real 11 on the field, it predicts targets better than an equal split among eligible
  players: RMSE WR 1.95 vs 2.17, TE 1.49 vs 1.61. It predicts carries far better (1.35 vs 4.72),
  and both beat form. The information is in *who is on the field and how often*. The personnel
  sampler draws that from league-wide rotation rates, which erases workhorse backs and every-down
  receivers.
- **Player-specific snap shares (tried, 2026-11-05).** The sampler now picks on-field players by
  usage. The league's on-field rate for each depth-chart role, by situation and by how many of that
  group are in the package (fit from 2016–2025 participation), is multiplied by the player's own
  usage relative to his role (leak-free EWMA of snap counts, shrunk toward 1). The package mix is
  tilted toward the team's own group usage. Within a group, players are drawn with exact inclusion
  probabilities. Callers can set `snap_shares` per player. Pre-game results: RMSE improves
  significantly but slightly on RB carries (5.31 → 5.19), WR targets (2.82 → 2.79) and TE targets
  (2.09 → 2.05). Yardage improves within noise, and the blend with form+opponent is unchanged.
- **That diagnosis was wrong.** On the held-out games' real plays, both samplers already put
  pre-game starters on the field *more* than reality (RB1 0.68 vs 0.59, WR1 0.88–0.91 vs 0.78;
  `scripts/sampler_onfield_check.py`). The remaining flatness is in who gets the ball once on the
  field. In simulated runs QB1 takes 16% of carries (24% when only RB2 is on the field), so RB1 gets
  55% of RB carries vs 66% actual. Given the real 22 and the real plays the head does well, so the
  gap is how it allocates in simulated contexts. It has no recent per-player usage inputs (carry
  share, target share), only career-static embeddings.
- **Recent usage features in the who head (2026-11-05).** Each on-field candidate gets six leak-free
  features from his earlier games (`usage.parquet`, `data.players.build_usage`): EWMA share of team
  carries, share of team targets, snap share, shrunk yards per carry and per target, and history
  length. They're added to his embedding in the who head. Retrained on the same split (`--who`
  turns them on automatically when `usage.parquet` exists); validation who-loss drops from 1.203 to
  1.189. Given the real plays and players, the head improves on RB carries (RMSE 1.35 → 1.17), WR
  targets (1.95 → 1.85) and TE targets (1.49 → 1.44).

  Simulated stat lines from the pre-game depth chart (usage sampler), old → new model, vs form+opponent:

  | | old | new | form + opp | blend vs form + opp |
  |---|---|---|---|---|
  | WR targets | 2.79 | **2.63** | 2.37 | 2.34 (significant) |
  | WR receiving yards | 29.4 | **28.1** | 26.5 | 26.3 (significant) |
  | TE targets | 2.05 | **1.99** | 1.80 | 1.78 (significant) |
  | TE receiving yards | 19.7 | **19.1** | 18.0 | 17.9 (n.s.) |
  | RB carries | 5.19 | **5.07** | 4.19 | 4.16 (n.s.) |
  | RB rushing yards | 29.2 | **28.8** | 25.1 | 25.2 (n.s.) |

  Every model-alone change is significant (paired bootstrap; `scripts/compare_samplers.py`), and
  usage is less flat: WR1 targets 6.2 → 6.6 (actual 7.0), TE1 3.9 → 4.2 (4.7), WR3 3.8 → 3.5 (3.4).
  It still trails the form baseline alone. RB1 carries are still short (12.3 vs 14.8): QBs take 15.5%
  of simulated runs vs 12.5% actual, and RBs 80% vs 84%.
- **Where the remaining gap is:** given the real 22 and the real plays, the head beats form easily
  (WR targets 1.85 vs 2.37). In simulation it doesn't. The difference is the sampled game
  context: play volume, play-calling mix and packages. Each simulated game also draws one
  realization, so stat lines inherit the variance of whole-game simulation. A straight blend of the
  two is the most accurate option today.

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
runitback nfl download --start 1999 --end 2026     # raw parquet → data/nfl/raw/ (+ players/: snaps, participation, coaches)
runitback nfl preprocess --start 1999 --end 2026   # → data/nfl/processed/{games,plays,ratings}.parquet, prints engine agreement
                                                   #   + {player_games,personnel,coaches,coach_games,players}.parquet
# held-out evaluation run (validation = 2025 week 10+ and playoffs)
runitback nfl train --out runs/nfl/eval --val-season 2025 --epochs 10
runitback nfl evaluate --ckpt runs/nfl/eval/model.pt --sims 200
runitback nfl calibrate --ckpt runs/nfl/eval/model.pt --season 2024   # grid-search strength_scale by Brier
# production model on all data
runitback nfl train --out runs/nfl/prod --epochs 10
```

Training uses lineups whenever the processed player tables exist; `--no-lineups` trains the
team-only model. `evaluate` scores a lineup model three ways: with who actually played,
with each team's previous-game lineup (what the app defaults to), and with no depth charts.

The default model (d_model 192, 4 layers, 2.6M params) trains on a 4-core CPU in about 40
minutes. Use `--d-model/--n-layers` for larger models on a GPU.

### Use it

**Web app**

```bash
runitback nfl serve --ckpt runs/nfl/prod/model.pt   # http://127.0.0.1:8000 (serves web/dist)
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

# lineup what-ifs (lineup models): groups you pass replace the team's most recent game
sim.lineup_spec("KC", "BUF")["home"]["depth_chart"]["QB"]      # ['00-0033873'] (Patrick Mahomes)
r = simulate_games(sim, home="KC", away="BUF", season=2026, n=1000, seed=0,
                   lineups={"KC": {"depth_chart": {"QB": ["NEW"]}, "coach": "Andy Reid"}})
```

To continue a game in progress, call `sim.rollout(state, n, history=[(state, outcome), ...])`.

**HTTP**

```bash
curl -X POST localhost:8000/api/simulate -H 'Content-Type: application/json' \
  -d '{"home":"KC","away":"BUF","season":2026,"n":1000}'
```

`/api/simulate` and `POST /api/games` take `"lineups": {"KC": {"coach": "...", "depth_chart": {"QB": [ids or "NEW"], ...}}}`.
`GET /api/teams/{team}/lineup` returns the default depth chart with player details and coach record, plus
the team's players to pick from; `GET /api/players?q=&group=` searches the league; `GET /api/coaches` lists coaches.
In the web app, **Depth charts & coaches** opens the editor for both teams.

Interactive endpoints: `POST /api/games`, `POST /api/games/{id}/step {play_type?, count?, temperature?}`,
`POST /api/games/{id}/undo`, `GET /api/games/{id}`, `GET /api/games/{id}/winprob?n=200`, `GET /api/teams`.

### Deploy to Google Cloud Run

The repository includes a Docker image that builds the web app, installs the CPU version of PyTorch,
and downloads the checksum-verified `v0.1.0` model during the image build. Install and authenticate
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
legality masks match sampling-time masks, sampler termination, and the API.

## Limitations / next steps (NFL)

- Player stat lines (`--who`) don't beat a form baseline before kickoff yet: the personnel sampler
  uses league-average rotation, so usage is too flat (see *Player stat lines*).
- Lineup models don't yet beat the team-only model on held-out games, and they react too little
  to quarterback changes (see *Lineup experiments*).
- Lineups are what the user (or the previous game) says. The model doesn't know about injuries
  announced during the week. Per-play participation ends with 2025, so 2026 games train with depth
  charts but without on-field personnel. Player embeddings are static over a career (no aging curve).
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
