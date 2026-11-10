# College basketball: lineup data probe (C1)

Is there enough on-floor lineup data for adjusted plus-minus (RAPM) in men's and women's college
basketball, or should C3 build player values from box scores? This is the decision gate in
`docs/multisport/PLAN.md` (work package C1). It covers both genders; `experiments/wcbb/` has no
separate copy.

**Short answer**
- **Men's: RAPM from the 2024 season on, with box-score values before that.**
  CollegeBasketballData (CBBD) lists the ten players on the floor (`onFloor`) on every play. That
  lineup data starts with the 2023-24 season (`season` 2024) and is good enough for RAPM in about
  68% of Division I games in 2024 and about 95% from 2025 on. The lineups are real, not imputed:
  players' minutes on the floor match the box score within 2 minutes for 99% of players, even on
  dates where CBBD lists no substitution plays. **There are no lineups at all before 2024**, so of
  the five walk-forward test seasons (2022–2026), 2022 and 2023 can only have box-score values and
  2024 starts cold.
- **Women's: box-score values.** CBBD has no women's data. ESPN's play-by-play has substitutions
  only from mid-February 2025 (100% of games since), so only one test season (2026) would have any
  lineup history, about 1,400 games. That is too little for RAPM across five folds. ESPN now records
  every game, so RAPM is worth revisiting once 2027 and later seasons are in.
- **ESPN's substitutions don't add anything for men's.** They start at the same point as women's
  (mid-February 2025). CBBD already has those games, plus a season and a half before them.

## Reproducing the results

`scripts/cbb_lineup_probe.py` (needs `CBBD_API_KEY` for the CBBD parts). Outputs go to `--out` and
are not kept in git.

```bash
python scripts/cbb_lineup_probe.py espn --out OUT --first 2006 --last 2026
python scripts/cbb_lineup_probe.py cbbd --out OUT --first 2006 --last 2027
python scripts/cbb_lineup_probe.py minutes --out OUT --dates 2023-12-02 2025-02-08 2025-02-15 2026-02-07
python scripts/cbb_lineup_probe.py report --out OUT
```

The `espn` step downloads every season of sportsdataverse's ESPN play-by-play for both genders
(about 1.5 GB in all, one file at a time). The `cbbd` step takes two API calls per sampled date
(`/games` and `/plays/date`), 86 calls in all. The `minutes` step takes two calls per date. The
key has a monthly call limit (CBBD's `x-calllimit-remaining` header). This probe used about 100
calls, test calls included: 333 were left before the first and 234 after.

## Setup

- **ESPN coverage: every game, every season.** Per game, the number of `Substitution` events for
  each team. A game is *rebuildable* when both teams have at least 10. Events carry one athlete id
  (`Subbing in for <Team>-<Player>` / `Subbing out ...`), never two.
- **CBBD coverage: two sampled Saturdays per season.** The Saturdays nearest 7 December and
  10 February; for the season in progress (2027), 7 November 2026. Every final game CBBD lists that
  UTC day is matched to its plays. A game **has lineups** when all three of these hold:
  - at least 90% of its plays list exactly five players per team;
  - each team uses at least three distinct fives (a real rotation, not starters carried all game);
  - the player credited with a play (substitutions aside) is one of the ten on at least 95% of
    plays.

  Tables count games where both teams are Division I (CBBD gives both a conference).
- **Validation against box scores** (`minutes`). For each player in CBBD's box score
  (`/games/players`), minutes on the floor are summed from `onFloor`: each play's ten get the clock
  time until the next play. These are compared with the box-score minutes. The first play's ten are
  compared with the listed starters. Imputed lineups, filled in without substitution records, would
  fail both checks.

## ESPN substitution coverage (both genders)

Share of games in ESPN's play-by-play with any substitution event and with rebuildable lineups
(both teams ≥ 10 events).

| season | men's games | men's with subs | men's rebuildable | women's games | women's with subs | women's rebuildable |
|---|---|---|---|---|---|---|
| 2006–2024 | 1,104–6,151 | **0%** | 0% | 250–5,779 | **0%** | 0% |
| 2025 | 6,136 | 25.8% | 25.8% | 5,473 | 24.9% | 24.9% |
| 2026 | 6,275 | **100%** | 100% | 6,011 | **100%** | 99.9% |

- **Seasons before 2025 have no substitution events at all**, under any play type or text.
  2024 men's and 2018 women's were also searched for "sub", "enters" and "replac" in the play text.
- **2025's 26% is a mid-season start, not partial coverage.** Substitutions begin in February 2025
  for both genders (59% of February games for men, 62% for women), then cover every game from
  March 2025 on. There are about 124–132 substitution events per game, all with one athlete id.
- Women's ESPN play-by-play before 2020 holds only about 1,500 games per season (about 5,000–6,000
  from 2020 on).

## CBBD on-floor lineups (men's)

Sampled dates, final games between two Division I teams.

| season | games | with plays | **has lineups** | CBBD substitution plays per game (median) |
|---|---|---|---|---|
| 2006–2013 | 192–219 per season (all games) | 24–72% | **0%** | 0 |
| 2014–2023 | 176–252 per season (all games) | 88–99% | **0%** | 0 |
| 2024 | 236 | 99.2% | **68.2%** | 0 |
| 2025 | 233 | 99.1% | **94.8%** | 0 (before mid-February) |
| 2026 | 236 | 100% | **94.9%** | 124–132 |
| 2027 (7 Nov 2026) | 87 | 100% | **93.1%** | 146 |

- **`onFloor` is empty on every play before 2024.** Not a single sampled play from 2006–2023
  lists anyone on the floor, although CBBD has play-by-play for most games from 2014.
- **Failures in 2024 are partial games.** The games that fail mostly have five per team on only
  50–80% of plays; the rest have no lineup on some stretches. Few are failures of the rotation or
  credited-player checks. About 14% of 2024–2027 failures involve a non-Division I opponent.
- **Where lineups exist, they are consistent.** The player credited with a play is on the floor
  for a median of 99.4–100% of plays.

### Validation against box scores

| date | players | games | CBBD sub plays / game | median minutes error | within 2 min | within 5 min | 95th pct error | starters on first play |
|---|---|---|---|---|---|---|---|---|
| 2023-12-02 (2024) | 2,192 | 109 | 0 | 0.40 | 94.5% | 95.0% | 4.6 min | 98.0% |
| 2025-02-08 (2025) | 2,623 | 138 | 0 | 0.33 | **99.0%** | 99.1% | 1.0 min | 100% |
| 2025-02-15 (2025) | 2,439 | 131 | 118 | 0.33 | 98.6% | 99.1% | 1.0 min | 97.3% |
| 2026-02-07 (2026) | 2,601 | 136 | 124 | 0.32 | 99.6% | 99.8% | 0.9 min | 98.5% |

- **The lineups are recorded, not imputed.** On 2025-02-08 CBBD lists no substitution plays, yet
  minutes on the floor match the box score as closely as on 2026-02-07, when substitutions are
  listed. CBBD evidently builds `onFloor` from substitution records that it only began exposing as
  plays in February 2025. Imputed lineups could not track minutes this closely.
- 2024 is noisier (95th percentile error 4.6 minutes), matching its 68% game pass rate. C3 should
  drop the plays without a full ten rather than drop whole games.
- **Player ids.** Every box-score player has an ESPN athlete id (`athleteSourceId`, 100%), so CBBD
  lineups join to ESPN box scores. ESPN's athlete id stays the same when a player transfers.

## Decision

| | **Men's (`cbb`)** | **Women's (`wcbb`)** |
|---|---|---|
| Player values | **RAPM** on possessions from CBBD `onFloor`, 2024 season on | **Minutes-weighted box-score values** |
| Before lineups exist | Box-score values for test seasons 2022–2023, and as the RAPM prior | n/a (box-score values throughout) |
| Lineup source | CBBD `/plays/date` (`onFloor`, CBBD ids → ESPN ids via `/games/players`) | none usable yet (ESPN subs from Feb 2025) |
| Revisit | — | RAPM once 2027+ seasons of ESPN subs exist |

## Next steps (for C3)

- **Men's RAPM.** Fit on CBBD possessions with full tens from 2024 on, refit weekly on earlier
  games only. Shrink toward a box-score prior so the 2024 fold (no earlier lineups) and players
  with few possessions fall back smoothly. Test seasons 2022–2023 use box-score values alone.
  Report the RAPM effect separately on 2024–2026.
- **CBBD call budget.** One `/plays/date` call per game day (about 150 per season, roughly 80 MB
  each). A full 2024–2027 backfill is about 600 calls, more than this key's remaining monthly
  allowance (234), so check the key's limit with the user before planning on it. A workflow needs
  `CBBD_API_KEY` as a secret.
- **Women's.** Box-score values. Use ESPN substitutions only for the current season's rotation
  projection (who is playing), not for RAPM.
