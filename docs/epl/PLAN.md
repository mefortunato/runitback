# Premier League: draws (three-outcome mode) and the EPL transformer

This plan covers two things: (1) adding a home/draw/away mode to the wager package, and (2) the EPL
transformer built on it. The EPL player data from the Fantasy Premier League API is being built in
another thread and is out of scope here. Work packages follow the style of `docs/multisport/PLAN.md`:
one package per session and one draft PR per package, titled `[<ID>] <title>`.

Starting point: the EPL ingest (`src/runitback/wager/epl/`, `tests/wager/test_epl.py`, commit
`88c7634` on `claude/laughing-einstein-teugev`, not yet on `main`). It gives one row per team per
game from 2000-01 on. `target` (`result`) is goal difference. The 1X2 prices are `moneyline`,
`draw_moneyline` and `opp_moneyline`. The total is always the 2.5 line (`over_odds`/`under_odds`).
`spread_line` is the Asian-handicap (AH) line, turned into the team's expected margin.

## 0. Where the code contradicts the notes (read first)

1. **Draw rate.** `epl/__init__.py` and `epl/data/features.py` both say a draw is "a third of the
   games". It is about 25% in the Premier League; fix the comments in D0.
2. **Draws never move Elo.** `EloModel.replay` (`models/baselines.py`) multiplies every update by
   `log(|margin| + 1)`, which is 0 for a draw. So a draw does not move either rating, even an
   underdog's away draw. This does not matter for NFL/NBA/NHL/MLB, where results are never 0. For
   football it is wrong.
3. **The transformer's Elo tokens ignore `Sport.elo`.** `strength()` in
   `models/transformer/data.py` builds `EloModel(schema)` with the NFL defaults, not
   `EloParams(**get_sport(sport).elo)`. So the NHL's tuned Elo is also not what its `_team_elo`
   tokens use. Fixing this changes other sports' outputs, so D5 only fixes it behind an opt-in.
4. **There are no AH prices.** The ingest keeps the handicap line but not its prices
   (football-data has `PCAHH/PCAHA`, `AvgCAHH/AvgCAHA`, `B365AHH/B365AHA`, ...). The backtest
   (`ats_record`) and staking (`SPREAD_ODDS`) assume -110, and quarter lines (-0.75, -1.25) need
   split-stake settlement, which nothing implements.
5. **Promoted clubs can't be predicted in their first game.**
   `TransformerMatchupModel.update_state` (`models/transformer/model.py`) keeps state only for teams
   that played this season or last. A club promoted after two or more seasons away, or one new to
   the league, therefore has no state before its first game. `Predictor.predict` raises
   `no state for teams` on those games. `walk_forward` drops them silently (`has_state`), so
   backtests leave out about 3 games per season, which biases them, and production fails on the
   opening weekend.
6. **The total line never varies.** Because the total is always 2.5, `_mkt_total` is a constant
   column. Check that `SequenceFeatures._standardize` does not divide by a zero standard deviation.

## 1. Every two-outcome assumption in `src/runitback/wager/`

| # | Where | Assumption | Effect on EPL |
|---|---|---|---|
| 1 | `core/schema.py` `GameSchema.market_moneyline` | A `(team, opp)` pair of prices | The draw price has no role; it is carried only through `display` |
| 2 | `core/metrics.py` `win_probability` | P(win) = Φ(margin/σ); one number per game | No P(draw) |
| 3 | `core/metrics.py` `evaluate` | `decided = y != 0`; Brier and log loss on decided games only; `accuracy` is win/loss | Silently drops about 25% of games; the Brier is conditional on no draw |
| 4 | `core/evaluation.py` `calibrate_and_evaluate` | Fits one σ (residual RMSE of the margin) | Can't calibrate a draw mass |
| 5 | `core/predictor.py` `Predictor.predict` | Emits `win_prob` only | — |
| 6 | `core/predictor.py` `_from_distributions` | `win_prob = beyond(pmf, values, 0)` = P(>0)/(P(>0)+P(<0)) | Conditional on no draw; the pmf already holds P(draw) = pmf[0] but drops it |
| 7 | `backtest.py` `no_vig_probability` | Two prices, normalized | With the 1X2 home/away prices only, it returns P(H)/(P(H)+P(A)), which overstates the win |
| 8 | `backtest.py` `implied_sigma`, `market_sigma` | Probit margin from a win probability | Unneeded when the market gives three probabilities |
| 9 | `backtest.py` `market_predictions` | `market_prob` from the two-way pair | Must become three-way |
| 10 | `backtest.py` `walk_forward` `keep` list | `win_prob`, `market_prob` | Needs `draw_prob`, `market_draw_prob` |
| 11 | `backtest.py` `moneyline_record` | Bets the side with `win_prob > market_prob`; outcome `sign(edge)*sign(margin)` → a draw is a **push** | In 1X2 a draw **loses** both side bets; no draw bets |
| 12 | `backtest.py` `summarize`, `compare` | Binary Brier on decided games; `brier_vs_market` paired CI | Wrong metric for 3 outcomes |
| 13 | `backtest.py` `ats_record`; `staking.py` `SPREAD_ODDS` | -110 and whole/half-point lines | AH needs its own prices and quarter-line settlement (§0.4) |
| 14 | `experiments.py` `ablation_table`, `season_table`, `blend_weight` | `won = y > 0`, binary Brier on decided games | Same as #12 |
| 15 | `tune.py` `season_brier` | Binary Brier on decided games | Same |
| 16 | `staking.py` `market_bets("moneyline")` | Two sides; `prob` vs `1 - prob`; outcome sign | Needs three picks with no push |
| 17 | `staking.py` `features` / `LearnedKelly` | `logit(implied)` of the picked price | Works per pick once the picks are three-way |
| 18 | `models/lightning.py` `MarginModule` metrics | Binary `accuracy` on margin > 0 | Logging only; harmless |
| 19 | `models/baselines.py` `EloModel.replay` | Draw multiplier = 0 (§0.2) | Draws carry no information |
| 20 | `models/transformer/data.py` `market_line`, `market_features` | `_mkt_prob_z` is the probit of the two-way no-vig probability; `_mkt_win_surprise` scores a draw as 0.5 against the two-way probability | Biased tokens (moderate harm) |
| 21 | `odds/providers.py` (The Odds API `h2h`), `odds/utils.py` `format_h2h` | Keeps the home and away outcomes only | Drops the `"Draw"` outcome in live odds |
| 22 | `web/snapshot.py` (`header`, `prediction`, `calibration`, played results `winner in ("home","away")`) | Two-way moneyline, `no_vig_probability(home, away)` | Web data has no draw |
| 23 | `web/src/predictions/lib/{types,pricing,odds}.ts`, `GameCard`, `MarketTable`, `PlayedGameCard` | `moneyline: {home, away}`, P(home) from two prices | UI has no draw |

Things that are fine as they are: `schema.mirror`, because the draw price is symmetric and needs no
pairing; `antisymmetrize`, because the margin is still antisymmetric; and `ScoreDistribution` with
`symmetric=True`, because its learned bias at 0 can lift the draw mass.

## 2. Design: an opt-in three-outcome mode

**One switch.** Add `GameSchema.market_draw: str | None = None` and `GameSchema.draws: bool = False`.
The EPL schema sets `draws=True, market_draw="draw_moneyline"`. Every new branch checks
`schema.draws`. No other schema changes, so on the other sports' code paths every changed function
runs the same arithmetic as before.

**Target.** Keep `result` (goal difference) as the target, so the margin, AH and total models keep
working. Derive the class where it is needed with `outcome_class(y) = sign(y) + 1`
(0 = loss, 1 = draw, 2 = win, from the row's team's perspective) in `core/metrics.py`. Add no new
column.

**Output columns (three-way mode only).**
- `win_prob` becomes the **unconditional** P(team wins). It is no longer conditional on a decided game.
- `draw_prob` is added. P(loss) = 1 − `win_prob` − `draw_prob`.
- The market's equivalents are `market_prob` and `market_draw_prob`.

Keeping the name `win_prob` lets the pick/edge plumbing and the web code read the same column. Its
meaning changes only for EPL.

**De-vigging 1X2.** Add `no_vig_probabilities(*prices, method="proportional") -> ndarray[n, k]` in
`backtest.py`. With k=2 and `proportional` it must equal today's `no_vig_probability` bit for bit;
`no_vig_probability` becomes a thin wrapper around it. Also add `method="power"`, which solves for an
exponent κ with Σ qᵢ^κ = 1, and `"shin"`. Both correct the favourite-longshot bias, which in football
mostly overprices draws and longshots. Pick the method once, by the market's own RPS on the
pre-test seasons 2006–2016, and record it in `epl/__init__.py`. Choosing it from closing prices of
earlier seasons is leak-free.

**Heads and losses.**
- **Transformer (preferred path).** Use the existing `distribution=True` mode (`ScoreDistribution`
  over integer margins). Set `margin_range=(-10, 10)` and `total_range=(0, 12)` (the defaults,
  (-70, 70) and (0, 130), are point-scale ranges). Then P(win) = Σ pmf[v>0], P(draw) = pmf[0], and
  the over 2.5 probability comes from `total_pmf`. The loss is the existing margin NLL, a proper
  scoring rule that already prices the draw.
  - Optional: an RPS auxiliary loss on the three collapsed classes, weight `rps_weight`
    (default 0 = identical).
  - Change `Predictor._from_distributions`: in three-way mode it emits P(>0) and P(=0) instead of
    `beyond(pmf, values, 0)`.
- **Point-margin models (Elo, home, point_diff, sklearn, GNN).** Use an ordered-probit calibration
  in `core/evaluation.py`:
  - P(win) = Φ((m − c)/σ), P(loss) = Φ((−m − c)/σ), P(draw) = the remainder.
  - `c ≥ 0` (the draw half-width) and σ are fit by maximum likelihood on the validation seasons,
    the same rule that picks σ today.
  - Store `draw_cut` on `Predictor`. Its default is `None`, and it is read with `getattr`, as
    `total_sigma` is, so pickled predictors still load.
  - With `c = 0`, P(win) = Φ(m/σ), the current formula. Two-way sports never fit `c`.
- **Market baseline.** `market_predictions` takes the three de-vigged probabilities directly; no σ
  is needed for them. `market_margin` stays as `spread_line` (the AH line).

**Metrics.** Add `three_way_scores(y, p_win, p_draw)` in `core/metrics.py`. It returns per-game:
- **RPS** (primary): ½[(F_L − O_L)² + (F_L+F_D − O_L−O_D)²], ordered loss < draw < win. RPS is the
  football standard and rewards near-misses: a draw predicted when the home team won scores better
  than an away win predicted.
- **multiclass Brier**: Σ over the 3 classes.
- **multiclass log loss**.
- **3-way accuracy**.

`evaluate(..., draw_prob=None)` keeps its current dict when `draw_prob is None`. With it, `evaluate`
scores **all** games and adds `rps`, `brier3`, `log_loss3`, `accuracy3`, and leaves out the binary
`brier`. Promotion uses the paired RPS delta against the closing-line market and against the EPL
production arm, with 95% CIs. Multiclass log loss is a secondary check.

**Backtest and paired deltas.**
- `walk_forward`: add `draw_prob`, `market_draw_prob`, and `schema.market_draw` to `keep`.
- `summarize`, `compare`, `experiments.ablation_table` / `season_table`, `tune.season_brier`: route
  per-game losses through one helper `game_loss(frame, schema, prob_cols)`. That is binary Brier on
  decided games for two-way schemas (today's arithmetic, moved) and RPS on all games for `draws`.
  The output columns are `d_rps`, `d_rps_ci95`, `rps_vs_market` (keep the `brier_*` names for
  two-way sports).
- `blend_weight` gets a three-way version: a logit-space blend of the class probabilities.

**Staking and EV on three-way prices.**
- Add `Market = "1x2"` in `staking.py` (`MARKETS` grows; two-way sports never produce it).
- For each game and each outcome i ∈ {H, D, A}: EV_i = pᵢ·(1 + payout(oddsᵢ)) − 1. The pick is the
  argmax EV.
- `edge` = EV, or pᵢ − qᵢ with q the no-vig market probability. Choose one in D4, and report both.
- `outcome` = +1 if the class matches, else −1. There is no push.
- `prob` = pᵢ.
- `moneyline_record` in draws mode delegates to a new `one_x_two_record` with the same row format
  (`_record`).
- AH, once prices are ingested (§0.4): `settle_asian(margin, line)` gives returns in
  {−1, −½, 0, +½·payout, +payout} for quarter lines. `ats_record` and the `"spread"` market use it
  when `schema.draws`. Until then, D4 skips AH staking for EPL.
- Totals at 2.5 never push; use the posted odds already in `market_total_odds`.

**Web and prediction output** (D7):
- `snapshot.py` header: `moneyline: {home, draw, away}`.
- Prediction: `win_prob`, `draw_prob`, `market_prob`, `market_draw_prob`, `ml_pick` ∈ {home, draw, away}.
- Calibration bins: per class.
- Played results: allow `winner == "draw"`.
- Add `web/sports/epl.py`.
- TS `types.ts`: `moneyline?: {home; away; draw?}`, optional `draw_prob`. `pricing.ts`: a
  `noVig3` that leaves the two-way path untouched when `draw` is absent.
- `GameCard`/`MarketTable`: show a third column when it is present.
- `odds/providers.py` `h2h`: `row["ml_draw"] = prices.get("Draw", {}).get("price")`. `format_h2h`
  is changed the same way. The Odds API sport key is `soccer_epl`.
- Add EPL to `web/src/App.tsx` `SPORTS` and to the server.

**Proving NFL/NBA/NHL/MLB outputs are identical.**
1. **Fingerprint script.** `scripts/wager_fingerprint.py` runs these deterministic pieces on the
   synthetic fixtures in `tests/wager/fixtures` (and the sports' test games):
   - `walk_forward` for `home`/`elo`/`sklearn` (seeded),
   - `summarize`, `compare`, `ablation_table`, `market_bets` for all `MARKETS`, and the staking
     report,
   - a seeded 2-member transformer `Predictor.predict` (as `tests/wager/test_lineup_editor.py` does).

   It prints a SHA-256 of each output, using `pd.util.hash_pandas_object` and canonical JSON.
2. **Pinned hashes.** `tests/wager/test_outcome_identity.py` pins those hashes, recorded on `main`
   before D1 and unchanged by D1–D7. Pinning hashes keeps result data out of git.
3. **Exact equality tests.** `no_vig_probabilities` with k=2 equals the old formula
   (`np.array_equal`). The ordered probit with `c=None` returns `win_probability` exactly.
4. **PR evidence.** Each PR states the fingerprint output on `main` and on the branch.

## 3. Work packages: draw mode

`deps` must be merged first. Packages with no deps between them can run in parallel.

| ID | Title | Deps | Deliverable / acceptance |
|---|---|---|---|
| **D0** | Merge the EPL ingest, plus AH prices | none | Merge `claude/laughing-einstein-teugev`'s ingest. Fix the "a third" comments. Add `spread_odds` / `opp_spread_odds` (closing AH prices: Pinnacle → average → bet365, the same preference order as the 1X2) to `epl/data/download.py`; `tests/wager/test_epl.py` covers them. Report the share of each `odds_source` per season, 2001–2026, in the PR |
| **D1** | Outcome mode, k-way de-vig, three-way metrics | D0 | `GameSchema.draws` / `market_draw`, `no_vig_probabilities` (proportional / power / Shin), `outcome_class`, `three_way_scores`, and `evaluate(..., draw_prob=)`. Unit tests check RPS on hand-computed cases, that the k=2 de-vig is bit-identical, and that the de-vigged probabilities sum to 1. The fingerprint script and pinned hashes land here (**hashes recorded on `main` first**) |
| **D2** | Draw-aware calibration and predictor | D1 | Ordered-probit `draw_cut` in `calibrate_and_evaluate` / `finish_fit` / `Predictor`. `_from_distributions` emits unconditional `win_prob` / `draw_prob`. Old pickles load. Tests: on simulated data with a 25% draw rate, the fitted `c` recovers it within ±0.02; two-way fingerprints unchanged |
| **D3** | Three-way backtest and experiment tables | D2 | `market_predictions` / `walk_forward` / `summarize` / `compare` / `ablation_table` / `season_table` / `tune` use `game_loss`, which is RPS for draws. `runitback-wager backtest epl --family elo` prints RPS for the model and the market with a paired CI. Fingerprints unchanged |
| **D4** | 1X2 and Asian-handicap staking | D3 (D0 for AH) | `market="1x2"` in `staking.py` (EV picks, no push), `one_x_two_record`, `settle_asian`. Tests: a draw loses home and away bets and wins draw bets; quarter-line settlement on −0.25 / −0.75 / −1.25 against margins −1..3. Fingerprints unchanged |
| **D5** | Football Elo | D1 | `EloParams` gains `mov: "nfl" \| "football" = "nfl"`. The football rule is the World Football Elo multiplier: 1 for a margin of 0–1, 1.5 for 2, (11+N)/8 for N≥3, so a draw moves ratings by K·(0.5 − E). It also gains `new_team: float \| None = None` (the start rating for a club not seen before; None = `mean`, as today), and `strength(..., params=None)` (None = today's NFL defaults). `scripts/epl_elo_tune.py` grid-searches on 2001–2016 only (§5). Settings go into `epl/__init__.py` `elo={...}`. Fingerprints unchanged |
| **D6** | Live three-way odds | D1 | The Odds API `h2h` keeps `Draw`; `format_h2h` and `consensus_odds` carry `draw_moneyline`; `teams.odds_map` covers the current 20 clubs. Tested with a recorded `soccer_epl` response fixture |
| **D7** | Web output for draws | D3, D6 | Snapshot JSON fields from §2, `web/sports/epl.py`, and TS types/pricing/cards. `npm run build` passes. Snapshot tests for EPL plus unchanged snapshots for the NFL/NBA fixtures |

## 4. The EPL transformer

**What it needs from §3.** D1–D3 are required: without them the model can't be scored or
calibrated with draws. D5 is needed for the Elo baseline and the `opponent_strength` tokens.
D4/D6/D7 are only needed for betting reports and serving.

**Data size.** About 9,500 games (19,000 rows) since 2000-01, of which about 7,600 are before the
first test season. That is much smaller than the NBA or NHL, so start small: `d_model=32`,
`num_layers=2`, `nhead=4`, `dropout=0.2`, a single fold per test season. Single folds are cheap,
about minutes on CPU, but the arm × season × seed grid still belongs in Actions (§6).

**History length.** A season is 38 games, about one a week. The NBA and NHL use 40, about half a
season of 82. For football, 38 is a full season and spans the previous season's run-in at any point
in the current one.
- Arms: `history` ∈ {19, 38, 57}, with `season_embedding=True`, `season_decay=True` (the encoder
  already supports both) and `max_seasons_ago=2`.
- Start at 38.
- Also cut tokens older than `max_seasons_ago` for a club returning after years away (new encoder
  option `drop_older_than`, default None = off). A club's 2017 games say little about its 2024
  squad.

**Token features available now.**
- **Base tokens** (`base_token_columns`): goals, half-time goals, shots, shots on target, corners,
  fouls, yellow and red cards (team and `opp_`), `result`, `location`.
- **Context**: `rest`, `opp_rest`.
- **Efficiency** (`epl_efficiency`, `efficiency=True`): shots, accuracy (SOT/shots), conversion
  (goals/shots), finishing (goals/SOT), cards (Y + 2R), and shot / SOT / corner shares.
  - Suggested additions (arm `eff2`, in `core/efficiency.py` `_epl_side`):
    - `second_half_goals` = goals − half-time goals, which captures game state,
    - `red_card` = a 0/1 indicator for any red: a game played down to 10 men is a distorted box score,
    - `sot_against_per_shot`, a defensive quality proxy.
- **Market tokens** (`market_tokens=True`) in draws mode:
  - Replace `_mkt_prob_z` with `_mkt_win_minus_loss` (the market's p_W − p_L) and
    `_mkt_draw_prob`.
  - Replace `_mkt_win_surprise` with expected-points surprise: (3·W + 1·D) − (3·p_W + p_D).
  - `_mkt_spread` = the AH line; keep it.
  - Replace the constant `_mkt_total` with `_mkt_over_z`, the probit of the no-vig over 2.5
    probability. Keep `_mkt_total_resid`.
  - All of this is gated by `schema.draws` in `market_features`, so other sports' tokens don't
    change.
- **Opponent strength** (`opponent_strength`): needs D5's `strength(params=...)` so the tokens use
  football Elo.
- **Not available.** There is no xG, lineups or injuries; FPL player data is the other thread. There
  are no cup or European fixtures either, so `rest` misses midweek games (see open questions).

**Outputs.** `distribution=True`, `margin_range=(-10, 10)`, `total_range=(0, 12)`,
`total_weight=1.0` (the over 2.5 comes from `total_pmf`). Check that `margin_range`/`total_range`
reach `MarginModule` through `transformer_config`'s pass-through settings, and add a test. The AH
`cover_prob` comes from `beyond(pmf, values, line)`. A quarter line never lands on an integer, so
it works.

**A team's first games of a season, and promoted clubs (fixes §0.5).** This is package E2.
- `update_state(..., cold_start=True)` gives teams with no state (or none within `max_seasons_ago`)
  an all-padding sequence, instead of raising or being skipped.
- Training for EPL uses `min_history=0` (an option of `fit_transformer` today), so empty-history
  rows teach the model a prior for a club it has not seen.
- New context columns `promoted` / `opp_promoted`: 1 if the club was not in the previous season's
  Premier League. That is known before the season, so there is no leakage. It goes into
  `GameSchema.context` and `paired`.
- The Elo baseline uses D5's `new_team` (and regresses a returning club toward it) instead of the
  league mean.
- Acceptance:
  - Walk-forward coverage is 100% of test games: `walk_forward` predicts every game, and `compare`
    reports n = every game.
  - RPS on promoted clubs' first 5 games is reported separately.
  - For other sports, `cold_start` defaults to False, and the fingerprints are unchanged.

**Elo settings for football (D5).** Starting values:
- `k=20`, `home=65` (about 0.35–0.4 goals before 2020), `regress=0.2`, `mean=1505`,
  `points_per_elo=1/200`, `new_team=1505 − 100`.
- Tune without leakage:
  - Grid `k ∈ {10, 15, 20, 25, 30}`, `home ∈ {40, 55, 70, 85}`, `regress ∈ {0.1, 0.2, 0.33}`,
    `new_team offset ∈ {0, 50, 100, 150}`.
  - Objective: the RPS of Elo's ordered-probit probabilities (D2) on seasons 2006–2016. The
    2001–2005 seasons are burn-in. Every season tuned is before the first test season (2022).
  - `points_per_elo` = least squares of the margin on the rating gap, on the same seasons, as the
    NHL did.
- Freeze the result in `epl/__init__.py` with a comment giving the seasons and score, in the style
  of the NHL's comment.
- The 2020-21 season (`season == 2021`) was played without crowds, and its home edge was about 0.
  Report the tuned home edge with and without it. Do not add a COVID flag unless the user wants one
  (open question).

## 5. Validation protocol

- **Walk-forward by season.** Test seasons are the last 5 complete ones, 2022–2026 (2025-26 finished
  in May 2026; `complete_seasons` already excludes the current one). For test season S, the model
  trains on seasons < S, with val = S−1 for early stopping and calibration (`split_seasons`
  default). State is refreshed each matchday (`walk_forward`'s per-date loop).
- **No leakage.**
  - Elo tuning, the de-vig method choice and every hyperparameter search use seasons ≤ 2016 or the
    fold's own training seasons.
  - Efficiency/standardization statistics come from the training rows (as `SequenceFeatures.fit`
    already does).
  - `promoted` is computed from the previous season's table.
  - Market tokens use each token game's own closing line, which was set before that game.
- **Baselines.**
  - (a) The closing-line market: three-way no-vig 1X2 on the same games.
  - (b) The EPL **production arm**. There is none yet, so E1 sets it: the best of
    `elo`/`sklearn`/`gnn` and the default transformer under three-way scoring, chosen on 2017–2021
    (not the test seasons).
- **Reporting.** For each arm, paired per-game RPS deltas against prod and against the market, with
  95% CIs (as `experiments._paired`), plus multiclass log loss, the RPS by season, and the 1X2/AH/O-U
  records. Report the `odds_source` mix of the test seasons alongside, since the market baseline is
  only as sharp as its prices.
- **Promotion.** Decide on 4-seed ensembles (`ensemble=4`, `_e4` vs `_e4`), per CLAUDE.md. The CI
  must include 0 or favour the arm. The noise floor for a 1-seed RPS is unknown for EPL; E1
  measures it with 4 reseeds of prod.
- **Swap test.** It needs player data, so it waits for the FPL thread. State this in the write-up.
- **Write-up.** `experiments/epl/README.md` in the NFL format: short answer first, then tables, then
  next steps.

## 6. Work packages: EPL transformer

| ID | Title | Deps | Deliverable / acceptance |
|---|---|---|---|
| **E1** | EPL baselines and the production arm | D3, D5 | Walk-forward 2022–2026 for `home`, `elo`, `sklearn`, `gnn` and a default transformer (`distribution=True`, history 38). Choose prod on 2017–2021 and measure the 1-seed noise floor. Write `experiments/epl/README.md`. **Needs a workflow** (`epl-wager-experiment.yml` + `experiments/epl/wager.yaml`, modeled on `nba-wager-experiment.yml`; the inputs job runs `runitback-wager epl download` in the run, no artifacts reused). The workflow and YAML are written in this package, not before |
| **E2** | Cold start and promoted clubs | D2 | `cold_start`, `promoted`/`opp_promoted`, EPL `min_history=0`, `drop_older_than`. 100% test coverage; fingerprints unchanged |
| **E3** | Football market tokens and efficiency | D1 | Three-way `market_features` (gated), `eff2` columns, and a check of the zero-variance column. Unit tests on a 3-game fixture. Fingerprints unchanged |
| **E4** | EPL transformer arms | E1, E2, E3 | Arms: `history` {19, 38, 57} × {`market_tokens`, `efficiency`, `eff2`, `opponent_strength`}, plus `rps_weight` {0, 0.5}. 1-seed screen, then 4-seed ensembles of the top 2 against prod's `_e4`. Results go in the README. Runs in an Actions matrix (arm × season × seed) |
| **E5** | Promote and serve | E4, D4, D6, D7 | `epl/__init__.py` `transformer={...}` / `elo={...}` from E4. Adding EPL to the Train/Refresh workflows **needs the user's approval** (`train-predictions.yml`, `refresh-predictions.yml`, `RUNITBACK_*` are production settings) |

**Order and parallelism.**
- First: D0, then D1.
- After D1, in parallel: D2, D5, D6, E3.
- After D2: D3 and E2.
- After D3: D4 and E1.
- After D3 and D6: D7.
- Last: E4, then E5.

The FPL player thread can merge into E4/E5 later as a `pvc`-style lineup arm with a swap test; it
needs `Sport.lineup` and is not planned here.

**Workflows.** E1 and E4 need new workflows; E5 changes production ones. No workflow or
experiment YAML is added by this plan. Commits that touch `experiments/epl/*.yaml` or
`.github/workflows/*` start runs; use `[skip ci]` unless a run is intended.

## 7. Open questions for the user

1. **What does "production baseline" mean for EPL?** There is no EPL prod model. Is E1's
   "best non-market arm chosen on 2017–2021" the right reference, or should every arm be judged
   only against the closing line?
2. **Primary metric.** Is RPS the promotion metric, with multiclass log loss secondary, or do you
   want multiclass Brier for continuity with the other sports' Brier tables?
3. **De-vig method.** Is it fine to choose proportional / power / Shin by the market's own RPS on
   2006–2016, or do you want proportional only, for consistency with the other sports?
4. **`win_prob` semantics.** Is it acceptable that in EPL it becomes the unconditional P(win), with
   `draw_prob` beside it? The alternative is new column names, such as `home_prob`/`draw_prob`/
   `away_prob`, which means more web and pipeline churn.
5. **Asian handicap.** Do you want AH in scope? It needs D0's price columns and quarter-line
   settlement. Or should EPL bet only 1X2 and over/under 2.5?
6. **Bookmaker.** Pinnacle's closing prices may be missing in recent football-data seasons; D0
   reports this. If they are, is the closing market average an acceptable baseline?
7. **Midweek fixtures.** Should rest days count cup and European games? That needs another source
   (football-data's E0 file only has league games). And should the no-crowd 2020-21 season get a
   flag, or be left to the walk-forward?
8. **Live odds.** The Odds API `soccer_epl` needs the same odds key as the other sports. Should EPL
   predictions be served at all before the FPL player data lands?
9. **Fixing `strength()`'s Elo for every sport.** It changes NHL, NBA and other outputs. Should
   that be done (as a separately measured package), or should it stay opt-in?
