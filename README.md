# Run It Back

Simulate sports games one play at a time, or replay a matchup a thousand times.

Run It Back covers several sports. Each sport has its own package with its rules, data and
vocabulary (`runitback.nfl`, `runitback.nba`). Shared pieces live in `runitback.core`: the
transformer building blocks, the score-distribution summaries, the model scheduler and training
helpers. One server (`runitback.server`) serves every sport's API and the web app:

- `/`: a landing page that explains the model and links to each sport
- `/nfl`: NFL predictions (spread, moneyline, over/under for upcoming games, recent results,
  power rankings, team map, the walk-forward backtest and "Beat the model"), and at
  `/nfl/generator` the NFL game, forecast and season simulators
- `/nba`: the NBA game, forecast and season simulators (NBA predictions come later)

The predictions pages come from `runitback.wager`, the game-level models ported from sportsml
(`runitback-wager <sport> ...`, the `wager` extra). They are precomputed static JSON; see
[Predictions data](#predictions-data).

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

For these models, the NFL game UI shows live rushing and receiving player stats for both teams,
including targets, receptions, carries, yards and touchdowns. Stats use the game's depth charts
and follow steps and undo. Models without the who head do not show player stats.

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
## NBA

Generates NBA games one **event** at a time with the same design: a causal transformer samples
what the team with the ball does next, and a deterministic rules engine applies it.

- **Web app** (`/nba`): step an event or a whole possession at a time, simulate to the end of the
  quarter or game, call 2s, 3s and timeouts, auto-play, undo, and follow the scoreboard, shot
  chart, line score, team stats and live win probability. Forecast a matchup over 1,000 games, or
  simulate a full season: 82-game schedule, play-in and best-of-seven playoffs.
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
| Season simulator | `src/runitback/nba/sim/season.py` | schedule formula, standings and tiebreakers, play-in, playoffs |
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
runitback nba season --season 2027 --seed 0 --out season.json   # full season + playoffs
```

**Season simulator** (the **Season** tab on `/nba`, `runitback nba season`, or
`simulate_season(sim, 2027)` from `runitback.nba.sim.season`). Supports seasons 2005–2040 (the
30-team, six-division era). The 82-game schedule follows the NBA formula: four games against each
division rival, four against six of the other ten conference teams and three against the other
four (rotating by season and seed), and two against every team in the other conference, with 41
home games each. Games are spread over about 170 days from opening night (the fourth Tuesday of
October), with no team playing twice in a day. Standings use the NBA tiebreakers (head-to-head,
division winner, division and conference record, point differential, then a seeded coin flip; the
record-against-playoff-teams steps are left out). From 2020-21 on, seeds 7–10 play the play-in
(7 v 8, 9 v 10, then the loser of the first game against the winner of the second). Earlier
seasons seed the top eight directly. Each conference then plays a fixed 1/8–4/5, 3/6–2/7 bracket of
best-of-seven series, with the higher seed hosting games 1, 2, 5 and 7. In the Finals, home court
goes to the better record.

Team strength is held fixed for the whole season: each team's preseason rating (*checkpoint*),
those ratings shuffled between teams (*randomized*), or league average (*neutral*). That makes
games independent, so the simulator batches games with different matchups (128 at a time) and
samples all seven games of a series at once, keeping only the games up to the fourth win. With
fixed ratings this gives the same distribution as playing the games in order. A season is 1,341
sampled games (1,230 regular season, 6 play-in and 15 × 7 playoff games).
HTTP: `POST /api/nba/seasons {season, seed, temperature, strength_mode}` starts a background job;
poll `GET /api/nba/seasons/{id}` for progress and the result.

**Releasing an NBA model**: run `runitback nba sha256 runs/nba/prod/model.pt`, attach the
checkpoint to a GitHub release tagged `nba-v0.1.0` as `runitback-nba-model-v0.1.0.pt`, then add
`"v0.1.0": ("nba-v0.1.0", "runitback-nba-model-v0.1.0.pt", "<sha256>")` to `MODELS` in
`src/runitback/nba/release.py` and set `LATEST = "v0.1.0"`. `runitback nba fetch-model` then
downloads it, and the Docker image (and so the deployed web app) includes it.

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

**NBA limitations:** team-level only (no players, injuries or rest); season simulations hold
team strength fixed (no in-season rating updates) and don't export synthetic training data the
way NFL seasons do; shot locations on the court are
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

For the NBA, `runitback nba fetch-model` downloads the latest released model (`nba-v0.1.0`) to
`runs/nba/prod/model.pt` and checks its SHA-256. Without it (and without one you trained, see
[NBA](#nba)), the NBA page shows the commands to train one and the rest of the app works as usual.

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
minutes.

**GPU training.** `train` (NFL and NBA) picks a device with `--device auto` (the default): a CUDA
GPU if PyTorch sees one, then Apple Silicon (MPS), then the CPU. Pass `--device cpu`, `cuda`,
`cuda:1` or `mps` to choose. On CUDA GPUs with bfloat16 support (Ampere and newer) training uses
bf16 mixed precision and TF32 matmuls; turn mixed precision off with `--no-amp`. Install a
CUDA build of PyTorch first (see [pytorch.org](https://pytorch.org/get-started/locally/)). A GPU
makes larger models practical, e.g.:

```bash
runitback nfl train --out runs/nfl/prod --epochs 10 --d-model 384 --n-layers 8 --n-heads 8 --batch-size 128
```

Checkpoints are always saved with CPU weights, so a model trained on a GPU loads and serves the
same way on a CPU-only machine.

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

# lineup what-ifs (lineup models): groups you pass replace the team's most recent game
sim.lineup_spec("KC", "BUF")["home"]["depth_chart"]["QB"]      # ['00-0033873'] (Patrick Mahomes)
r = simulate_games(sim, home="KC", away="BUF", season=2026, n=1000, seed=0,
                   lineups={"KC": {"depth_chart": {"QB": ["NEW"]}, "coach": "Andy Reid"}})
```

To continue a game in progress, call `sim.rollout(state, n, history=[(state, outcome), ...])`.

**HTTP**

```bash
curl -X POST localhost:8000/api/nfl/simulate -H 'Content-Type: application/json' \
  -d '{"home":"KC","away":"BUF","season":2026,"n":1000}'
```

`/simulate` and `POST /games` take `"lineups": {"KC": {"coach": "...", "depth_chart": {"QB": [ids or "NEW"], ...},
"snap_shares": {"<player id>": 0.8}}}` (lineup models; `snap_shares` optional). `GET /teams/{team}/lineup` returns
the default depth chart with player details and coach record, plus the team's players to pick from;
`GET /players?q=&group=` searches the league; `GET /coaches` lists coaches. In the web app, **Depth charts &
coaches** opens the editor for both teams.

Interactive endpoints (all under `/api/nfl`): `POST /games`, `POST /games/{id}/step {play_type?, count?, temperature?}`,
`POST /games/{id}/undo`, `GET /games/{id}`, `GET /games/{id}/winprob?n=200`, `GET /teams`,
`POST /seasons`, `GET /seasons/{id}`. The same NFL routes are still served without the `/nfl`
prefix (`/api/simulate`, ...) for clients written before the NBA was added.

### Publishing a model to the deployed app

The Docker image bakes in the release marked `latest` in `src/runitback/nfl/releases.json`.
Merging code doesn't change the deployed model; publishing a release does. Two GitHub Actions
workflows do it end to end. Each publishes a GitHub release with the checkpoint and opens a PR that
updates `releases.json`; **merging that PR** runs the Cloud Run deploy, so a person always approves
what goes live:

- **Train NFL model** (`.github/workflows/train-nfl-model.yml`, run by hand with a version such as
  `v0.2.0`). Downloads and preprocesses every season (raw data is cached between runs), optionally
  trains a held-out model first and stops if its validation loss is above a threshold, then trains on
  all data with lineups and the who head, calibrates and smoke tests it. About 45 minutes on a
  public repository's 4-CPU runner, roughly 1h15 on a private repository's 2-CPU runner.
- **Refresh NFL model data** (`.github/workflows/refresh-nfl-lineups.yml`, Tuesdays 14:41 UTC or by
  hand). Fetches the deployed model, runs `refresh-lineups` and publishes the next patch version
  (`v0.2.0` → `v0.2.1`). Weeks without new games publish nothing. A few minutes.

Repository settings both need: **Settings → Actions → General → Workflow permissions →
"Allow GitHub Actions to create and approve pull requests"**. The workflows request `contents: write`
and `pull-requests: write` themselves. Scheduled and manually started workflows only run from the
default branch, so they start working once this file is on `main`. If an older release PR is still
open when a newer one arrives, close the older one; each PR sets `latest` to its own version.

The same steps by hand:

```bash
runitback nfl train --out runs/nfl/prod --epochs 10 --who --prev-lineup-prob 0.3   # all data, no hold-out
runitback nfl calibrate --ckpt runs/nfl/prod/model.pt --write
sha256sum runs/nfl/prod/model.pt
```

Attach the file to a GitHub release (tag `v0.2.0`, asset `runitback-nfl-model-v0.2.0.pt`), then
`runitback nfl register-release v0.2.0 --ckpt runs/nfl/prod/model.pt --file src/runitback/nfl/releases.json`
adds it with its SHA-256 and makes it the latest. Merging that change to `main` deploys it.

**Weekly refresh, no retraining.** A checkpoint carries pre-game team ratings and, for lineup models,
each team's default depth chart (its most recent game), the player directory (career games, usage
multipliers, recent-usage features) and head-coach records. All of these are frozen when it's built.
`refresh-lineups` downloads the latest nflverse data and rebuilds them. The model's weights,
vocabularies and fitted tables stay as trained, and new players and coaches use its NEW / OTHER tokens.
It takes under a minute:

```bash
runitback nfl refresh-lineups --ckpt runs/nfl/prod/model.pt --out runs/nfl/prod/model-refreshed.pt
```

It prints how current the data is, any teams whose latest game is older (byes), and the new file's
SHA-256. Publish the refreshed file as a new release (e.g. `v0.2.1`) as above; the weekly workflow
does this for you. Retrain when the
season rolls over or player vocabularies go stale: players without an embedding fall back to
NEW / OTHER.

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

## Predictions data

The predictions pages read static JSON from `/data/...`: `runitback-wager web export` writes it from
a trained wager model and its walk-forward backtest. The server serves `/data` from
`$RUNITBACK_PREDICTIONS` (else from the web build, e.g. `web/public/data` in development), and the
pages show "not available" for anything missing. Locally:

```bash
pip install -e ".[wager]"
runitback-wager web build --sports '[nfl]' --output_dir webdata     # train + backtest (~1 h on 4 CPUs)
runitback-wager web export --data_dir webdata --output_dir out --refresh true
RUNITBACK_PREDICTIONS=out/data runitback serve
```

In production the data lives in a **Cloud Storage bucket mounted read-only into the Cloud Run
service** at `/mnt/predictions` (`RUNITBACK_PREDICTIONS`), so it refreshes without a redeploy and
deployments keep needing a merge to `main`:

- **Train predictions** (`.github/workflows/train-predictions.yml`, monthly and by hand) trains the
  wager model, backtests it and publishes the `predictions-webdata` release asset (backtest seasons
  are reused through the `predictions-backtest-cache` release while the model is unchanged).
- **Refresh predictions** (`.github/workflows/refresh-predictions.yml`, daily and before each NFL
  kickoff window) downloads that release, re-downloads results and lines (The Odds API with the
  `ODDS_API_KEY` secret; without it the previous lines carry over), refreshes team state (no
  retraining), exports, and syncs the files to the bucket. The pages pick them up within a minute:
  the mount caches metadata for 60 s and `/data` responses carry `Cache-Control: no-cache`.
- **Deploy to Cloud Run** mounts the bucket when the `PREDICTIONS_BUCKET` secret is set (it also
  switches the service to the second-generation execution environment, which Cloud Storage volumes
  need). Without the secret the app deploys as before and the predictions pages are empty.

The NFL wager model reads the play-by-play generator's frozen game embeddings in its history tokens
(the `pbp32` arm of [the foundation experiment](experiments/nfl/README.md): at least as good as the
box-score model on every market). It needs one foundation model per test season, so the backtest
of each season and the model for upcoming games only use a play-by-play model that never trained on
the games they predict. `runitback nfl foundation` keeps them in `foundation/nfl/fold<season>/`. It
trains a missing fold (about an hour; once per season) and adds newly played games to the
embedding tables. The wager side uses them when `RUNITBACK_NFL_FOUNDATION` names that directory;
without it, the model is the box-score one. Both workflows set it, and the folds live in the
`foundation-nfl` release. Locally:

```bash
runitback nfl foundation                    # trains folds missing from foundation/nfl
RUNITBACK_NFL_FOUNDATION=foundation/nfl runitback-wager web build --sports '[nfl]' --output_dir webdata
```

Run the build and export from the directory that holds `foundation/nfl`: the trained model stores
its embedding table's path relative to it.

One-time setup (the deploy service account is the one the deploy workflow already uses; the Cloud
Run runtime service account is the service's, by default the Compute Engine default account):

```bash
gcloud storage buckets create "gs://${PREDICTIONS_BUCKET}" --project "$GOOGLE_CLOUD_PROJECT" \
  --location "$CLOUD_RUN_REGION" --uniform-bucket-level-access
# Refresh predictions writes the files
gcloud storage buckets add-iam-policy-binding "gs://${PREDICTIONS_BUCKET}" \
  --member "serviceAccount:${GCP_DEPLOY_SERVICE_ACCOUNT}" --role roles/storage.objectAdmin
# the running service reads them
gcloud storage buckets add-iam-policy-binding "gs://${PREDICTIONS_BUCKET}" \
  --member "serviceAccount:${CLOUD_RUN_RUNTIME_SERVICE_ACCOUNT}" --role roles/storage.objectViewer
```

Then add `PREDICTIONS_BUCKET` (the bucket name, without `gs://`) and optionally `ODDS_API_KEY`
to the `production` environment's secrets, run **Train predictions** once (it starts the first
refresh), and redeploy (or push to `main`) so the service mounts the bucket. If the `production`
environment requires a reviewer, scheduled refreshes wait for one; give the refresh workflow an
environment without that rule holding the same Google Cloud secrets. `deploy-cloud-run.sh` reads
`PREDICTIONS_BUCKET` from `.env` the same way.

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
- **Adding a sport** (e.g. `runitback.cfb` via cfbfastR) means writing that sport's vocabulary,
  rules engine, preprocessing and outcome heads, as `runitback.nba` does. The transformer trunk,
  summaries, scheduler and batching in `runitback.core` are reused; the NFL and NBA samplers and
  servers are near-identical templates and are the next candidates to move into `core`.
