# Premier League: draws (three-outcome mode) and the EPL transformer

This plan covers two things: (1) a home/draw/away mode for the wager package, and (2) the EPL
transformer built on it. The work packages follow `docs/multisport/PLAN.md`: one package per
session, one draft PR per package, titled `[<ID>] <title>`.

**Starting point.** The ingest lives on `claude/laughing-einstein-teugev` and is not merged to `main`.
- **Games:** `src/runitback/wager/epl/`, `runitback-wager epl download`. This is football-data.co.uk:
  one row per team per game, 2000-01 on. `target` (`result`) is goal difference. The 1X2 prices are
  `moneyline`, `draw_moneyline` and `opp_moneyline`. The over/under is always the 2.5-goal line, priced
  by `over_odds`/`under_odds`. `spread_line` is the Asian-handicap (AH) line.
- **Players:** `src/runitback/epl/` (`data/players.py`, `paths.py`), `runitback epl download` →
  `data/epl/players.parquet`. This is FPL appearances from 2019-20 on, laid out like
  `src/runitback/nhl`.

The player-value and lineup work is planned by the player-data thread. It is not planned here; §6
only says where it plugs in.

## Decisions (settled 2026-11-20)

| Topic | Decision |
|---|---|
| Strength Elo | `strength()` will use `Sport.elo` for every sport. The parent session is fixing this on the ingest branch, and D5 builds on that fix |
| Serving | EPL predictions wait until player data is in the model. D6 (live odds), D7 (web) and E5 (promote and serve) are later packages, off the critical path |
| Rest days | League games only, with no cup or European games. This is a known limitation. There is no COVID flag, but the tuned home edge is reported with and without 2020-21 |
| Market baseline | The closing market average is acceptable. §0.3 says which bookmaker each season uses |
| Asian handicap | Out of scope. `spread_line` stays only as a display column and a market token. EPL bets 1X2 and over/under 2.5 |
| Reference arms | Report both: the paired delta against the closing-line market, and against the best non-market arm chosen on 2017–2021 (E1) |
| Metric | RPS is the promotion metric; multiclass log loss is secondary. Other sports keep Brier |
| De-vig | Choose proportional, power or Shin by the market's own RPS on 2006–2016 (see the caveat in §2) |
| `win_prob` | For EPL it is the unconditional P(win), with `draw_prob` beside it |
| Comments | The "a third of the games" comments are being fixed on the ingest branch |

## 0. Where the code contradicts the notes

1. **Draws never move Elo.** `EloModel.replay` (`models/baselines.py`) scales every update by
   `log(|margin| + 1)`, which is 0 for a draw. So a draw never moves a rating, even an underdog's
   away draw. This is harmless for NFL/NBA/NHL/MLB, where results are never 0, but wrong for
   football. D5 fixes it.
2. **Promoted clubs can't be predicted in their first game.**
   `TransformerMatchupModel.update_state` (`models/transformer/model.py`) keeps state only for teams
   that played this season or last. Two kinds of club therefore have no state for their first game:
   one promoted after two or more seasons away, and one new to the league.
   - In production, `Predictor.predict` raises `no state for teams`.
   - In backtests, `walk_forward` silently drops those games (`has_state`), about 3 per season, so
     backtests are biased.
   - E2 fixes it.
3. **Which bookmaker each season's prices come from.** I checked the football-data.co.uk headers and
   fill rates on 2026-11-20. With the ingest's preference order (`BOOKS_1X2`, `BOOKS_TOTAL`):

   | Seasons (ending year) | 1X2 `odds_source` | Over/under 2.5 | Strength of the market arm |
   |---|---|---|---|
   | 2001–2012 | opening bet365 (older books where bet365 is missing). `BbAvH` (the opening average) exists but is not in `BOOKS_1X2` | from 2003: opening bet365 / `BbAv` | weak: opening prices from one book |
   | 2013–2019 | closing Pinnacle (`PSCH`) | opening | sharp |
   | 2020–2025 | closing Pinnacle | closing Pinnacle | sharp |
   | 2026 (2025-26) | closing Pinnacle for 210 of 380 games, then **closing average** (Pinnacle columns go blank mid-season) | the same split | mixed within the season |
   | 2027 (current) | closing average only (no Pinnacle columns) | closing average | slightly softer than Pinnacle |

   Closing averages (`AvgCH`, `AvgC>2.5`) exist from **2019-20 (2020)** on, in every game. So a
   *mixed-source* market arm would test 2022–2025 against Pinnacle and 2026 half against each.
   **Recommendation (D0):** keep the closing-average 1X2 and over/under prices as their own columns
   (`avg_moneyline`, `avg_draw_moneyline`, `avg_opp_moneyline`, `avg_over_odds`,
   `avg_under_odds`), and score the market arm on them for every test season (2022–2026). That makes
   the baseline the same in every test season and matches what serving will see from 2026-27 on.
   Report Pinnacle-close RPS as a second row wherever it exists.

   The market *tokens* in training cover all of 2001–2027, a mix of opening and closing books. E3
   adds an `_mkt_closing` indicator (1 for a closing price), so the model can weight sharp lines
   more.
4. **No variation in the total line.** It is always 2.5, so `_mkt_total` is a constant column. Check
   that `SequenceFeatures._standardize` does not divide by a zero standard deviation (E3).
5. **xG starts in 2026-27.** The 2026-27 file adds `HxG`/`AxG`, and FPL has player xG from 2022-23.
   Neither covers the training seasons, so the team model leaves them out and D0 does not load them
   (settled: a display-only column would be NaN for 25 seasons). If football-data backfills team xG,
   an xG-token arm is a possible later package. The player thread may use FPL xG.

## 1. Every two-outcome assumption in `src/runitback/wager/`

| # | Where | Assumption | Effect on EPL |
|---|---|---|---|
| 1 | `core/schema.py` `GameSchema.market_moneyline` | A `(team, opp)` pair of prices | The draw price has no role; it is carried only through `display` |
| 2 | `core/metrics.py` `win_probability` | P(win) = Φ(margin/σ) | No P(draw) |
| 3 | `core/metrics.py` `evaluate` | `decided = y != 0`; Brier and log loss on decided games only; win/loss `accuracy` | Silently drops about 25% of games |
| 4 | `core/evaluation.py` `calibrate_and_evaluate` | Fits one σ | No draw mass |
| 5 | `core/predictor.py` `Predictor.predict` | `win_prob` only | — |
| 6 | `core/predictor.py` `_from_distributions` | `win_prob = beyond(pmf, values, 0)` = P(>0)/(P(>0)+P(<0)) | Conditional on no draw; P(draw) = pmf[0] is computed and thrown away |
| 7 | `backtest.py` `no_vig_probability` | Two prices | With the 1X2 home/away prices it gives P(H)/(P(H)+P(A)), which overstates the win |
| 8 | `backtest.py` `implied_sigma`, `market_sigma` | Probit margin from a win probability | Unneeded when the market gives three probabilities |
| 9 | `backtest.py` `market_predictions` | Two-way `market_prob` | Must become three-way |
| 10 | `backtest.py` `walk_forward` `keep` | `win_prob`, `market_prob` | Needs `draw_prob`, `market_draw_prob` |
| 11 | `backtest.py` `moneyline_record` | Outcome `sign(edge)*sign(margin)`, so a draw is a **push** | In 1X2 a draw **loses** both side bets; there are no draw bets |
| 12 | `backtest.py` `summarize`, `compare` | Binary Brier on decided games; `brier_vs_market` | Wrong metric; `ats_record`/`edge_correlation` run whenever `market_spread` is set (AH at -110) |
| 13 | `backtest.py` `ats_record`; `staking.py` `"spread"` / `SPREAD_ODDS` | -110 and whole/half-point lines | AH is out of scope: skip both in draws mode |
| 14 | `experiments.py` `ablation_table`, `season_table`, `blend_weight` | `won = y > 0`, binary Brier on decided games | Same as #12 |
| 15 | `tune.py` `season_brier` | Binary Brier on decided games | Same |
| 16 | `staking.py` `market_bets("moneyline")` | Two sides; `prob` vs `1 - prob`; sign outcome | Needs three picks with no push |
| 17 | `staking.py` `features` / `LearnedKelly` | `logit(implied)` of the picked price | Works once the picks are three-way |
| 18 | `models/lightning.py` `MarginModule` metrics | Binary `accuracy` | Logging only |
| 19 | `models/baselines.py` `EloModel.replay` | A draw's multiplier is 0 (§0.1) | Draws carry no information |
| 20 | `models/transformer/data.py` `market_line`, `market_features` | `_mkt_prob_z` is the probit of the two-way no-vig probability; `_mkt_win_surprise` scores a draw as 0.5 against it | Biased tokens |
| 21 | `odds/providers.py` (`h2h`), `odds/utils.py` `format_h2h` | Home and away outcomes only | Drops `"Draw"` from live odds (later, D6) |
| 22 | `web/snapshot.py` | Two-way moneyline, `winner in ("home","away")` | No draw (later, D7) |
| 23 | `web/src/predictions/lib/{types,pricing,odds}.ts` and the cards | `moneyline: {home, away}` | No draw (later, D7) |

These are fine as they are:
- `schema.mirror`: the draw price is symmetric.
- `antisymmetrize`.
- `ScoreDistribution(symmetric=True)`: its learned bias at 0 can lift the draw mass.

## 2. Design: an opt-in three-outcome mode

**Switch.** Add `GameSchema.draws: bool = False` and `GameSchema.market_draw: str | None = None`.
The EPL schema sets `draws=True, market_draw="draw_moneyline"`. Every new branch checks
`schema.draws`, and nothing else in the schemas changes.

**Target.** Keep `result` (goal difference). Derive the class only where it is needed:
`outcome_class(y) = sign(y) + 1`, giving 0 loss, 1 draw, 2 win (`core/metrics.py`). No new column.

**Output columns (draws mode).**
- `win_prob` = unconditional P(team wins).
- `draw_prob` is new; P(loss) = 1 − `win_prob` − `draw_prob`.
- The market's equivalents are `market_prob` and `market_draw_prob`.
- No `cover_prob` for EPL (AH is out of scope).

**De-vigging 1X2.** Add `no_vig_probabilities(*prices, method="proportional") -> [n, k]`
(`backtest.py`).
- `no_vig_probability` becomes its k=2 proportional wrapper and must match today's output bit for bit.
- `"power"` solves Σ qᵢ^κ = 1; `"shin"` is Shin's method. Both correct the favourite-longshot bias.
- The method is chosen by the market's own RPS on 2006–2016 (settled). **Caveat:** those seasons use
  opening bet365 (2006–2012) and closing Pinnacle (2013–2016). Neither has the margin of the closing
  average the market arm uses from 2022 on. D1 also reports the method ranking on 2020–2021 closing
  averages (pre-test, so leak-free). If the two rankings disagree, the plan follows 2020–2021 for
  the average-price arm, and the PR says so.

**Heads and losses.**
- **Transformer.** Use the existing `distribution=True` (`ScoreDistribution`) with
  `margin_range=(-10, 10)` and `total_range=(0, 12)`.
  - Then P(win) = Σ pmf[v>0], P(draw) = pmf[0], and the over-2.5 probability comes from `total_pmf`.
  - The loss is the existing margin NLL, a proper scoring rule that prices the draw.
  - Optional: an RPS auxiliary loss with `rps_weight` (default 0, which changes nothing).
  - `Predictor._from_distributions` emits P(>0) and P(=0) in draws mode.
- **Point-margin models (Elo, home, point_diff, sklearn, GNN).** Use an ordered probit in
  `core/evaluation.py`:
  - P(win) = Φ((m − c)/σ), P(loss) = Φ((−m − c)/σ), and P(draw) is the remainder.
  - σ and `c ≥ 0` are fit by maximum likelihood on the validation seasons.
  - Store `Predictor.draw_cut`, default `None`, read with `getattr` like `total_sigma`, so old
    pickles load.
  - With `c=None`, the output equals `win_probability` exactly.
- **Market baseline.** `market_predictions` takes the three de-vigged probabilities from the market
  arm's prices (the closing averages, §0.3). No σ is needed for them.

**Metrics.** `three_way_scores(y, p_win, p_draw)` in `core/metrics.py` returns, per game:
- **RPS**: ½[(F_L − O_L)² + (F_L+F_D − O_L−O_D)²], with the outcomes ordered loss < draw < win.
- multiclass Brier.
- multiclass log loss.
- 3-way accuracy.

`evaluate(..., draw_prob=None)` returns today's dict when `draw_prob` is None. With it, `evaluate`
scores **all** games, returns `rps`, `brier3`, `log_loss3` and `accuracy3`, and drops the binary
`brier`.

**Backtest and paired deltas.**
- `walk_forward`: add `draw_prob`, `market_draw_prob` and the draw and average-price columns to
  `keep`.
- `summarize`, `compare`, `experiments.ablation_table` / `season_table` and `tune.season_brier`
  route per-game losses through `game_loss(frame, schema, prob_cols)`. That is binary Brier on
  decided games for two-way schemas (today's arithmetic, moved) and RPS on all games in draws mode.
- The columns are `d_rps`/`d_rps_ci95` against the reference arm, `rps_vs_market`/`_ci95` against
  the market arm, and multiclass log loss beside them.
- In draws mode, `ats_record` and `edge_correlation` are skipped.

**Staking and EV.**
- A `"1x2"` market in `staking.py`. For each outcome i, EV_i = pᵢ·(1 + payout(oddsᵢ)) − 1, and the
  pick is the argmax.
- `edge` is EV; pᵢ − qᵢ is reported too. `outcome` = ±1 with no push, and `prob` = pᵢ.
- In draws mode, `moneyline_record` delegates to `one_x_two_record` (same `_record` rows).
- Over/under 2.5 uses the existing `"total"` market at the posted odds (2.5 never pushes).
- In draws mode `MARKETS` is `("1x2", "total")`. The `"spread"` market raises for EPL.

**Proving the other sports' outputs are identical.**
1. `scripts/wager_fingerprint.py` runs deterministic pieces on `tests/wager/fixtures` and prints a
   SHA-256 of each output (via `pd.util.hash_pandas_object` and canonical JSON):
   - `walk_forward` for `home`/`elo`/`sklearn` (seeded),
   - `summarize`, `compare`, `ablation_table`, `market_bets` for every two-way market, and the
     staking report,
   - a seeded 2-member transformer `Predictor.predict` (as `tests/wager/test_lineup_editor.py` does).
2. `tests/wager/test_outcome_identity.py` pins those hashes. They are recorded on `main` **after**
   the ingest branch (with the `strength()` fix, which deliberately changes other sports' transformer
   tokens) merges, and **before** D1.
3. Exact tests: `no_vig_probabilities` with k=2 equals the old formula (`np.array_equal`), and the
   ordered probit with `c=None` equals `win_probability`.
4. Every PR shows the fingerprint output on `main` and on its branch.

## 3. Work packages: draw mode

`deps` must merge first. Packages with no deps between them can run in parallel.

| ID | Title | Deps | Deliverable / acceptance |
|---|---|---|---|
| **D0** | Merge the EPL ingest, plus closing-average columns | ingest branch (incl. `648deed`) | Prerequisite: `claude/laughing-einstein-teugev` merged to `main` through a user-opened PR, or `648deed` (the `strength()` fix) cherry-picked. Merge the ingest (games, players and the `strength()` fix). Add `avg_moneyline`, `avg_draw_moneyline`, `avg_opp_moneyline`, `avg_over_odds` and `avg_under_odds` (`AvgCH/D/A`, `AvgC>2.5`/`<2.5`; NaN before 2019-20) to `epl/data/download.py`, with tests in `tests/wager/test_epl.py`. The PR reports the `odds_source` share per season, 2001–2027 (it should match §0.3) |
| **D1** | Outcome mode, k-way de-vig, three-way metrics | D0 | `GameSchema.draws` / `market_draw`, `no_vig_probabilities`, `outcome_class`, `three_way_scores`, `evaluate(..., draw_prob=)`. The de-vig method ranking on 2006–2016, with 2020–2021 breaking any disagreement,, recorded in `epl/__init__.py`. Tests: hand-computed RPS cases, a bit-identical k=2 de-vig, probabilities that sum to 1. The fingerprint script and pinned hashes, recorded on `main` after `648deed` is there |
| **D2** | Draw-aware calibration and predictor | D1 | The ordered-probit `draw_cut` in `calibrate_and_evaluate` / `finish_fit` / `Predictor`. `_from_distributions` emits unconditional `win_prob` / `draw_prob`. Old pickles load. On simulated data with 25% draws, the fitted `c` recovers the draw rate within ±0.02. Fingerprints unchanged |
| **D3** | Three-way backtest and experiment tables | D2 | `market_predictions` (average-price arm, plus a Pinnacle row where it exists) and `walk_forward` / `summarize` / `compare` / `ablation_table` / `season_table` / `tune` on `game_loss`. ATS is skipped in draws mode. `runitback-wager backtest epl --family elo` prints the model's and the market's RPS with paired CIs. Fingerprints unchanged |
| **D4** | 1X2 staking | D3 | `"1x2"` market in `staking.py` (EV picks, no push), and `one_x_two_record`. Tests: a draw loses home and away bets and wins draw bets, and `LearnedKelly` fits on three-way picks. Fingerprints unchanged |
| **D5** | Football Elo | D1, the `strength()` fix | `EloParams` gains `mov: "nfl" \| "football" = "nfl"`. The football rule is the World Football Elo multiplier: 1 for a margin of 0–1, 1.5 for 2, (11+N)/8 for N≥3, so a draw moves ratings by K·(0.5 − E). It also gains `new_team: float \| None = None` (None = `mean`, as today). `scripts/epl_elo_tune.py` grid-searches on 2001–2016 only (§4). Settings go in `epl/__init__.py` `elo={...}`, and the transformer's `_team_elo` tokens pick them up through the fixed `strength()`. Fingerprints unchanged |
| D6 *(later)* | Live three-way odds | D1; after the lineup arm | The Odds API `h2h` keeps `Draw` (`soccer_epl`); `format_h2h` / `consensus_odds` carry `draw_moneyline`. Tested on a recorded response fixture |
| D7 *(later)* | Web output for draws | D3, D6 | `snapshot.py` fields (`moneyline: {home, draw, away}`, `draw_prob`, `market_draw_prob`, `ml_pick` ∈ {home, draw, away}, per-class calibration, `winner == "draw"`), `web/sports/epl.py`, TS `types.ts`/`pricing.ts` (`noVig3`; the two-way path untouched), the cards, and `App.tsx` `SPORTS`. `npm run build` passes, and the NFL/NBA snapshots are unchanged |

## 4. The EPL transformer

**What it needs from §3.**
- Required: D1–D3 (scoring and calibration with draws) and D5 (football Elo for the baseline and the
  `opponent_strength` tokens).
- Not required: D4, which is needed only for betting reports.
- Deferred: D6 and D7.

**Data size.** About 9,500 games (19,000 rows) from 2000-01, about 7,600 of them before the first
test season. That is small, so start with `d_model=32`, `num_layers=2`, `nhead=4`, `dropout=0.2`.

**History length.**
- A season is 38 games, about one a week. The NBA and NHL use 40, about half their 82-game season.
  For EPL, 38 is a full season.
- Arms: `history` ∈ {19, 38, 57}, with `season_embedding=True`, `season_decay=True` and
  `max_seasons_ago=2`. Start at 38.
- A new encoder option `drop_older_than` (default None) drops tokens more than `max_seasons_ago`
  seasons old. It is for clubs returning after years away.

**Token features.**
- **Base** (`base_token_columns`): goals, half-time goals, shots, shots on target, corners, fouls,
  yellow and red cards (team and `opp_`), `result` and `location`.
- **Context**: `rest`, `opp_rest`. These count league games only (settled), so a midweek European or
  cup game is invisible. Report this as a known limitation in `experiments/epl/README.md`.
- **Efficiency** (`epl_efficiency`): shots, accuracy, conversion, finishing, cards, and shot / SOT /
  corner shares. The arm `eff2` (`core/efficiency.py` `_epl_side`) adds:
  - `second_half_goals`,
  - a `red_card` indicator (a game played down to 10 men),
  - `sot_against_per_shot`.
- **Market tokens** (gated by `schema.draws` in `market_features`, so other sports' tokens don't
  change):
  - `_mkt_prob_z` → `_mkt_win_minus_loss` and `_mkt_draw_prob`.
  - `_mkt_win_surprise` → points surprise, (3W + D) − (3p_W + p_D).
  - The constant `_mkt_total` → `_mkt_over_z`, the probit of the no-vig over-2.5 probability.
    `_mkt_total_resid` stays.
  - `_mkt_spread` = `spread_line`, the AH line used as a token only.
  - New: `_mkt_closing` (§0.3).
- **Opponent strength**: football Elo through the fixed `strength()` (D5).

**Outputs.**
- `distribution=True`, `margin_range=(-10, 10)`, `total_range=(0, 12)`, `total_weight=1.0`. The over
  2.5 comes from `total_pmf`.
- Check, and test, that `margin_range`/`total_range` reach `MarginModule` through
  `transformer_config`'s pass-through settings.
- No cover probability.

**First games and promoted clubs (E2; fixes §0.2).**
- `update_state(..., cold_start=True)` gives teams with no state an all-padding sequence.
- EPL trains with `min_history=0` (an existing `fit_transformer` option), so empty-history rows teach
  the prior.
- New context `promoted` / `opp_promoted`: 1 if the club was not in the previous season's Premier
  League. It is known before the season, and goes in `context` and `paired`.
- The Elo baseline uses D5's `new_team` for clubs it has not seen, and regresses returning clubs
  toward it.
- Acceptance:
  - 100% of test games are predicted.
  - RPS on promoted clubs' first 5 games is reported separately.
  - `cold_start` defaults to False, and the fingerprints are unchanged.

**Elo for football (D5).**
- Start: `k=20`, `home=65` (about 0.35–0.4 goals before 2020), `regress=0.2`, `mean=1505`,
  `points_per_elo=1/200`, `new_team=1405`.
- Grid: `k` {10, 15, 20, 25, 30}, `home` {40, 55, 70, 85}, `regress` {0.1, 0.2, 0.33}, `new_team`
  offset {0, 50, 100, 150}.
- Objective: the RPS of Elo's ordered-probit probabilities on 2006–2016. 2001–2005 are burn-in, and
  every tuned season is before the 2022 test seasons.
- `points_per_elo` = least squares of the margin on the rating gap, on the same seasons.
- No COVID flag (settled). Report the tuned home edge with and without 2020-21. That season is
  outside the tuning window (2006–2016), so it matters only for the walk-forward refits. Report the
  model's RPS on 2021 separately.

## 5. Validation protocol

- **Walk-forward by season.** Test seasons are 2022–2026, the last 5 complete; 2025-26 ended in May
  2026. For test season S, train on < S, with val = S−1 for early stopping and calibration
  (`split_seasons`). State is refreshed every matchday (`walk_forward`).
- **No leakage.**
  - Elo tuning, the de-vig choice and hyperparameter searches use seasons ≤ 2016 (≤ 2021 for the
    2020–2021 de-vig check) or the fold's own training seasons.
  - Standardization comes from the training rows (`SequenceFeatures.fit`).
  - `promoted` comes from the previous season.
  - Market tokens use each token game's own pre-game price.
- **Reference arms** (settled: report both):
  - (a) The market: the closing average, three-way no-vig, on the same games, in every test season.
    A Pinnacle-close row is added where it exists (2022–2025 and 210 games of 2026).
  - (b) The best non-market arm: the best of `elo`, `sklearn`, `gnn` and the default transformer,
    chosen on 2017–2021 (E1). It is the EPL "prod" until something beats it.
- **Reporting.** For each arm:
  - paired per-game RPS deltas with 95% CIs (`experiments._paired`) against (a) and (b),
  - multiclass log loss,
  - RPS by season,
  - 1X2 and over/under 2.5 records.
- **Promotion.** RPS on 4-seed ensembles (`_e4` vs `_e4`); the CI must include 0 or favour the arm.
  E1 measures the 1-seed noise floor with 4 reseeds of the reference arm.
- **Swap test.** It needs player data, so it waits for the lineup arm (§6).
- **Write-up.** `experiments/epl/README.md` in the NFL format: short answer first, then tables, then
  next steps.

## 6. Work packages: EPL transformer

| ID | Title | Deps | Deliverable / acceptance |
|---|---|---|---|
| **E1** | EPL baselines and the reference arm | D3, D5 | Walk-forward 2022–2026 of `home`, `elo`, `sklearn`, `gnn` and a default transformer (`distribution=True`, history 38). Choose the reference arm on 2017–2021 and measure the noise floor. Write `experiments/epl/README.md`. **Needs a workflow** (`epl-wager-experiment.yml` + `experiments/epl/wager.yaml`, modeled on `nba-wager-experiment.yml`). Its inputs job runs `runitback-wager epl download` in the run, with no artifacts reused. The workflow and YAML are written in this package |
| **E2** | Cold start and promoted clubs | D2 | `cold_start`, `promoted`/`opp_promoted`, EPL `min_history=0`, `drop_older_than`. 100% test coverage; fingerprints unchanged |
| **E3** | Football market tokens and efficiency | D1 (D0 for `_mkt_closing`) | Draws-mode `market_features`, `_mkt_closing`, the `eff2` columns and the zero-variance check (§0.4). Unit tests on a 3-game fixture. Fingerprints unchanged |
| **E4** | EPL transformer arms | E1, E2, E3 | `history` {19, 38, 57} × {`market_tokens`, `efficiency`, `eff2`, `opponent_strength`}, plus `rps_weight` {0, 0.5}. A 1-seed screen, then 4-seed ensembles of the top 2 against the reference arm's `_e4`. Results go in the README. Runs in an Actions matrix (arm × season × seed) |
| E5 *(later)* | Promote and serve | E4, the lineup arm, D4, D6, D7 | `epl/__init__.py` `transformer={...}` / `elo={...}`. Train/Refresh workflows; these **need the user's approval** (`train-predictions.yml`, `refresh-predictions.yml`, `RUNITBACK_*`) |

**Order.**
- First: D0, then D1.
- After D1, in parallel: D2, D5, E3.
- After D2: D3 and E2.
- After D3: E1 and D4.
- Then E4.
- Off the critical path, after the lineup arm: D6, D7 and E5.

**Where the player data plugs in (planned by the other thread).** This follows the NHL/NBA layout:
- player values in `src/runitback/epl/player_values.py` (as `src/runitback/nhl/player_values.py`),
  built from `data/epl/players.parquet` (`runitback epl download`), giving each team-game a
  `pv.parquet` row;
- a `LineupSpec` in `src/runitback/wager/epl/lineup.py`, set as `Sport.lineup` (as
  `wager/nba/lineup.py`);
- a `pvc` arm on top of E4's best config, with the swap test.

FPL data starts in 2019-20, so weekly values have 2020–2021 as burn-in before the 2022 test season.
The lineup arm should expect thin early-season values in 2022. The arm's projected lineup and swap
test must not depend on draws-mode code beyond D1–D3.

**Workflows.** E1 and E4 need new workflows, and E5 changes production ones. This plan adds no
workflow or experiment YAML. Commits touching `experiments/epl/*.yaml` or `.github/workflows/*`
start runs, so use `[skip ci]` unless a run is intended.

## 7. Questions

All questions are settled; see the decision table at the top and these follow-ups (2026-11-20):

1. **Market arm:** the closing average in every test season, with Pinnacle as a second row where it
   exists (§0.3, §5).
2. **De-vig:** rank the methods on 2006–2016 with the prices available. Any disagreement is broken
   by the 2020–2021 closing-average ranking, which is leak-free because it is before 2022. Record
   the disagreement, if any, in `experiments/epl/README.md`.
3. **Fingerprints:** the `strength()` fix is commit `648deed` on `claude/laughing-einstein-teugev`,
   not on `main`. Merging that branch (or cherry-picking `648deed`) through a user-opened PR is the
   prerequisite for D0/D1. The hashes are taken on `main` after that.
4. **2026-27 xG:** left for later, with no display-only load (§0.5).
