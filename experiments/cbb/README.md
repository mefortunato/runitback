# College basketball: player values and rotations (C3)

Work package C3 (`docs/multisport/PLAN.md`) covers layers B and C of the NFL player model for men's
(`cbb`) and women's (`wcbb`) college basketball:
- every player gets an offense and a defense value, refit weekly without leakage;
- every team-game gets a projected rotation;
- every team-game gets a `pv.parquet` row for the wager model's lineup terms (C4).

The method follows the C1 decision (`lineup_probe.md`). Both genders get box-score values. Men's
games from 2023-24 on add adjusted plus-minus from CollegeBasketballData (CBBD) lineups, shrunk
toward those box values. `experiments/wcbb/` points here.

**Short answer**
- **The values look right and carry most of the team strength.** The projected rotations' value
  difference, with no team history at all, correlates **0.59 (men) and 0.67 (women) with the final
  margin**, with about **1.1 points of margin per point of value**. Before the 2026-27 season the
  top players are Cameron Boozer, Yaxel Lendeborg and Jeremy Fears Jr. (men), and Sarah Strong,
  Hannah Hidalgo and Lauren Betts (women).
- **Changes in the lineup carry less than their value.** Given each team's recent results, a change
  in projected lineup value moves the margin by:

  | change | men | women |
  |---|---|---|
  | season-opening roster turnover | 0.25 per point | 0.47 per point |
  | within the season (absences) | 0.15 ± 0.06 per point | 0.14 ± 0.07 per point |

  A missing top-two player costs 1.5 points (men) and 2.5 points (women) beyond the team's form. The
  signal is there, but the values overstate the gap between a starter and his backup. C4's `pvc`
  weight will learn the scale; better shrinkage is the first next step.
- **The rotation projection follows the plan's stand-in.** It projects from recent minutes minus
  players who didn't play. It gets 81% of each team's five most-used players and 88% of starters
  right, with a minutes error of 5.1–5.3 per player. Removing the players who didn't play is a leak:
  ESPN gives no reason for college DNPs (see Setup). Without the removal (the last three games'
  players) these numbers are 74–75%, 81–83% and 6.8–7.0.
- **Blocked: men's adjusted plus-minus has not been run on real seasons.** The code, tests and the
  CBBD fetch exist and run end to end on one cached day (2026-11-08). Backfilling 2024–2026 needs
  about 450 CBBD calls (one `/plays/date` per game day). The key has **233 calls left this month**.
  Until a larger limit or a persistent cache is agreed (see Next steps), the men's values are the
  box-score values in every season. That matches C1's plan for 2022–2023, and 2024–2026 are pending.

## Reproducing

Only the code and this write-up are in git. Everything is rebuilt from sportsdataverse's ESPN player
box scores, and men's lineups come from CBBD:

```bash
G=cbb   # or wcbb
runitback college download --gender $G          # player_box_<season>.parquet, cbb 2008+, wcbb 2014+
runitback college players --gender $G           # player_games / team_games / players
runitback college lineups --gender $G           # projected_lineups.parquet + the check vs who played
runitback college cbbd --season 2024 --season 2025 --season 2026   # men's only: CBBD_API_KEY, ~150 calls a season
runitback college player-values --gender $G     # player_values/$G/pv.parquet, ppg.json, values_latest.csv
python scripts/college_player_values.py tune --gender $G     # the box-value settings below
python scripts/college_player_values.py report --gender $G   # every table below
```

`player-values` fits 334 weeks (men's, 2012–2027) or 213 (women's, 2017–2027) in 2–4 minutes on one
core. `.github/workflows/college-player-values.yml` runs the same steps in Actions and uploads
`<gender>-player-values`. It is dispatch/call only and fetches CBBD only with `cbbd: true`. The
cached CBBD days (`data/cbb/cbbd/plays_<date>.parquet`, about 1 MB each) are never refetched, and the
fetch stops when 25 calls are left (`--reserve`) so the wager pipeline's CBBD lines keep working.

## Setup

- **Data** (`runitback.college.data`). ESPN player box scores carry starter, did-not-play, minutes
  and the counts. Women's box scores have no minutes before 2014 and only about 1,700 games a season
  before 2017, so the women's tables start in 2014 and `pv.parquet` starts in 2017. Men's start in
  2008, with `pv.parquet` from 2012.
- **Transfers.** Every table is keyed by ESPN athlete id, which stays the same when a player changes
  school. A transfer's minutes history, box rates and values follow him to his new team.
- **Projected rotation** (`runitback.college.lineups`).
  - The candidates are the players listed in the team's box score for the game (the roster at
    tip-off), **minus everyone who didn't play**.
  - Each candidate's expected minutes are a recency-weighted mean (0.85 per game back) of his
    minutes before the game's date, for any team. Players with no earlier games get 6 minutes.
  - Minutes are scaled to the team's 200, with no player above 38.
  - The projected starters are the five with the most recency-weighted starts.
  - **The leak:** ESPN's college box scores give no DNP reason and there is no injury-report
    history. A player who sat for a coach's decision, or was ruled out after the line was set,
    counts as known at tip-off. For regulars that is mostly injury, illness and suspension, which a
    report would have had. For the end of the bench it changes nothing, because those players have
    almost no projected minutes.
  - `rule="previous"` (the players of the team's last three games, nobody removed) is the
    leak-free comparison.
- **Box-score values, both genders** (`runitback.college.box_values`).
  - **Rates.** Each player has per-40 rates of 13 counts: points, FGM, FGA, 3PM, FTM, FTA, offensive
    and defensive rebounds, assists, steals, blocks, turnovers and fouls. They come from his games
    before the week, for any team, over three seasons plus the current one, weighted 0.85 per
    season back.
  - **Shrinkage.** Rates are shrunk with 100 minutes of his position group's **replacement rates**:
    the pooled rates of players under 150 minutes in a season, from the seasons before.
  - **The fit.** Each team-game's points per 100 possessions (`FGA − OREB + TOV + 0.475 FTA`) is
    regressed on the minutes-weighted sums of its players' rates over replacement (offense
    coefficients) and of the opponent's players' rates (defense coefficients), plus home and season.
    Each game's rates are its players' rates **as of that game's week**, so a game's own box score
    never explains its own result.
  - **Values.** A player's offense value is the offense coefficients times his rates. His defense
    value is minus the defense coefficients times them. Both are points per possession over
    replacement, and higher is better on both.
- **Men's adjusted plus-minus from 2023-24** (`runitback.college.cbbd_lineups`,
  `runitback.college.player_values`).
  - **Possessions.** CBBD `/plays/date` gives the ten on the floor on every play. Possessions are cut
    with the NBA's rules (`runitback.nba.player_values.game_possessions`): points scored, technical
    free throws excluded, and the ten players at the first action.
  - **Player ids.** CBBD ids are matched to ESPN ids by name within each game's ESPN roster, with no
    extra calls. A CBBD side is paired with the ESPN team whose roster matches more of its names,
    because the two sources disagree on home and away at some neutral sites.
  - **The fit.** It is the NBA's possession-level ridge (`runitback.core.player_values`) with home,
    possession start and season controls. The target is each possession's points **minus what its
    ten players' box values predict**, so every player is shrunk toward his box value, not toward 0.
  - **Values.** A player's value is his box value plus his own coefficient over his group's
    replacement level. Players under 250 minutes on the floor keep their box value.
- **Groups: G / F / C** from the ESPN position (ATH and missing go to F), as for the NBA.
  `runitback.wager.cbb.lineup.LINEUP` is the NBA's layout with `recent=8` (a quarter of the college
  wager model's 32-game history). It is not registered on the `Sport` yet; C4 does that.
- **Points per game.** A player's points per game are his value per possession times the league's
  possessions per team-game (the previous season's mean: about 70 for men, 72 for women) times his
  projected minutes / 40.
- **Leak-free and weekly.** Each Monday is fit on everything before it. That covers the rates, the
  replacement level, the coefficients and the adjusted plus-minus. Each game uses its week's fit.
  The tests check that rewriting every game on or after a cut-off leaves its values and its
  `pv.parquet` rows unchanged.

### The lineup table (`pv.parquet`)

One row per team-game: `game_id`, `season`, `team_id` (ESPN), `team` (the school's latest name, as
the wager model keys it).

| columns | meaning | sign on the total |
|---|---|---|
| `_pv_0 .. _pv_2` | G, F, C offense: the projected rotation's points per game over replacement | + |
| `_pv_3 .. _pv_5` | G, F, C defense | − |
| `_pv_6`, `_pv_7` | offense / defense change vs the team's previous game | |
| `_pv_8 .. _pv_13` | each group's change vs the mean of the team's last 8 games | |

The earlier games behind the change columns are valued with the same week's values and their own
projected minutes. So a change measures who is in or out, not the projection's smoothing. At a
season's start, the earlier games are last season's, so the change is the roster turnover:
departures (still valued) and arrivals (transfers with their history, freshmen at replacement). The
change columns average −0.1 to −0.3 because more value leaves each year than arrives.

## Checks

**Box-value settings** (`tune`): fit on everything before a season, predict that season's team-games
(points per 100 possessions) from the players' values at the season's start, weighted by the minutes
they played. Mean over 2022–2026:

| window | decay | pseudo-minutes | men R² / corr | women R² / corr |
|---|---|---|---|---|
| 2 | 0.7 | 150 | 0.169 / 0.411 | 0.257 / 0.508 |
| 3 | 0.7 | 300 | 0.168 / 0.410 | 0.253 / 0.504 |
| 3 | 0.7 | 150 | 0.172 / 0.415 | 0.259 / 0.510 |
| **3** | **0.85** | **100** | **0.176 / 0.419** | **0.264 / 0.514** |
| 3 | 0.7 | 600 | 0.163 / 0.404 | 0.244 / 0.497 |

- The prediction's slope is 0.98 (men) and 1.05 (women), so the values are on the right scale.
- The replacement threshold (50 / 150 / 300 minutes) moved R² by under 0.015.
- The ridge penalty barely matters: 26 coefficients on about 50,000 team-games.
- These are values from before the season, with no in-season updates.

**Rotation projection vs who played** (`rule="dnp"`, the default; `rule="previous"` in brackets):

| season | men: top-5 right | men: starters right | men: minutes error | women: top-5 | women: starters | women: minutes error |
|---|---|---|---|---|---|---|
| 2022 | 81.5% (75.1%) | 88.0% (81.2%) | 5.1 (6.8) | 80.9% (75.4%) | 88.0% (82.0%) | 5.3 (6.9) |
| 2023 | 81.4% (74.9%) | 88.5% (81.5%) | 5.1 (6.8) | 80.8% (75.3%) | 88.9% (82.8%) | 5.3 (6.8) |
| 2024 | 81.1% (74.6%) | 88.4% (81.3%) | 5.1 (6.8) | 80.6% (74.9%) | 88.8% (82.7%) | 5.3 (6.8) |
| 2025 | 80.3% (73.6%) | 88.2% (81.0%) | 5.1 (6.8) | 80.6% (74.9%) | 88.9% (82.6%) | 5.3 (6.8) |
| 2026 | 79.8% (72.6%) | 87.8% (80.1%) | 5.2 (7.0) | 80.3% (73.9%) | 88.7% (81.5%) | 5.3 (7.0) |

- "Top-5 right" is the share of each team-game's five most-used players among its five highest
  projected.
- The minutes error is the mean absolute error per player, counting players who played or were
  projected.
- Without the DNP removal, 4–5% (men) and 4% (women) of the minutes went to players not projected
  at all: arrivals and returns.

**Lineup value vs the game** (the home − away difference of the six level columns):

| season | men: games | corr with margin | margin pts per value pt | women: games | corr with margin | margin pts per value pt |
|---|---|---|---|---|---|---|
| 2017 | 5,914 | 0.556 | 1.10 | 5,240 | 0.635 | 1.11 |
| 2018 | 5,999 | 0.565 | 1.06 | 5,272 | 0.635 | 1.08 |
| 2019 | 6,035 | 0.567 | 1.07 | 5,358 | 0.636 | 1.02 |
| 2020 | 5,766 | 0.549 | 1.04 | 5,429 | 0.629 | 1.10 |
| 2021 | 4,281 | 0.579 | 1.10 | 3,820 | 0.631 | 1.03 |
| 2022 | 5,964 | 0.614 | 1.12 | 5,510 | 0.622 | 1.07 |
| 2023 | 6,221 | 0.595 | 1.12 | 5,813 | 0.662 | 1.08 |
| 2024 | 6,241 | 0.603 | 1.12 | 5,898 | 0.677 | 1.10 |
| 2025 | 6,290 | 0.635 | 1.12 | 5,631 | 0.706 | 1.09 |
| 2026 | 6,300 | 0.642 | 1.13 | 6,029 | 0.719 | 1.14 |
| all (men 2012+) | 88,603 | 0.587 | 1.08 | 54,527 | 0.666 | 1.10 |

- These correlations are higher than the NBA's (0.40) because college teams differ far more in
  strength.
- The home edge is not in the difference.
- There are no historical college lines in the free sources: the ESPN play-by-play spread is a
  placeholder, and CBBD lines cost calls. So the market comparison is left to C4, whose games table
  has the lines.

**Does a lineup change predict the margin beyond the team's recent results?** These are the `pvc`
inputs. The final margin is fitted by least squares on home, both teams' recent margins and the
lineup change (home − away). The table gives margin points per point of value change, ± 2 standard
errors:

| games | men | women |
|---|---|---|
| all, vs the last 8 games' margins and rotations (`_pv_8 .. _pv_13`) | 0.00 ± 0.03 | 0.22 ± 0.04 |
| top-decile changes | 0.17 ± 0.04 | 0.35 ± 0.05 |
| season openers (vs last season's games) | 0.25 ± 0.05 | 0.47 ± 0.07 |
| 10+ games into the season, vs the season's margins so far | 0.15 ± 0.06 | 0.14 ± 0.07 |
| one of the team's top two players out (2018+, 4% of games) | −1.5 ± 0.4 pts | −2.5 ± 0.5 pts |

- The level itself is strong: margin points per point of level difference are 1.05 alone and 0.58
  (men) / 0.73 (women) next to recent form.
- Within a season the change column shows the right sign but a small slope. A missing regular costs
  less than the values say, because his backups are valued close to him relative to replacement.
- In the NFL, `pvc` learned 0.58 per point. Expect a smaller weight here, with most of the gain at
  season openers.

**Top and bottom of the latest fit** (before the 2026-27 season; points per 40 minutes over
replacement; players of the last two seasons with 600+ minutes in the window):

| men | team | group | offense | defense | total | | women | team | group | offense | defense | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cameron Boozer | Duke | F | +10.4 | +6.7 | +17.1 | | Sarah Strong | UConn | F | +13.9 | +12.8 | +26.7 |
| Allen Graves | Santa Clara | F | +8.9 | +7.8 | +16.7 | | Brooklyn Meyer | South Dakota State | F | +12.9 | +8.7 | +21.5 |
| Yaxel Lendeborg | Michigan | F | +8.7 | +7.9 | +16.6 | | Hannah Hidalgo | Notre Dame | G | +11.5 | +9.8 | +21.2 |
| Jeremy Fears Jr. | Michigan State | G | +8.7 | +7.5 | +16.2 | | Lauren Betts | UCLA | C | +11.8 | +9.1 | +20.9 |
| Zuby Ejiofor | St. John's | F | +8.0 | +7.3 | +15.2 | | Mikayla Blakes | Vanderbilt | G | +13.2 | +6.6 | +19.7 |
| Dylan Faulkner | Clemson | F | +7.2 | +7.7 | +14.9 | | MiLaysia Fulwiley | LSU | G | +10.4 | +9.0 | +19.4 |
| Darius Acuff Jr. | Arkansas | G | +9.7 | +4.9 | +14.7 | | Ny'Ceara Pryor | St. John's | G | +9.6 | +9.0 | +18.6 |
| Caleb Wilson | North Carolina | F | +7.6 | +6.9 | +14.5 | | Ashlyn Watkins | South Carolina | F | +8.2 | +10.3 | +18.5 |
| … Ryan Archey (last) | North Carolina Central | G | −0.8 | −1.1 | −1.9 | | … Aminata Tal (last) | Presbyterian | G | −1.8 | −2.2 | −3.9 |

- Within a level the order looks right.
- The rates are not opponent-adjusted, so stars at low-major schools rank too high. Allen Graves
  (Santa Clara) and Brooklyn Meyer (South Dakota State) are examples.
- That matters little for `pvc`, which compares a team's lineup with its own recent lineups. It
  matters more for a transfer moving up a level.
- `values_latest.csv` lists every player.

**Men's adjusted plus-minus, on the one cached CBBD day** (2026-11-08, 53 games):
- All 1,169 CBBD players matched ESPN ids. Four sides were paired with the other ESPN team (neutral
  sites).
- 98.4% of possessions had ten matched players. 92% of team-games' possession points are within 3 of
  the box score.
- That day's possessions enter the 2027 weekly fits, which stay at the box values as they should
  with so few possessions.
- The tests check the shrinkage on a synthetic league. A player whose prior is wrong moves toward the
  truth. A player whose prior is right stays there.

## Next steps

1. **Unblock men's adjusted plus-minus (needs a decision).** Either:
   - a CBBD key with a larger monthly limit as the `CBBD_API_KEY` secret, about 450 calls per
     2024–2026 backfill;
   - an Actions cache of `data/cbb/cbbd` (raw plays, refetched only for new days), as an exception to
     "rebuild from source";
   - or ESPN substitutions for men's lineups. They are free, but only exist from February 2025, so
     adjusted plus-minus would cover the 2026 test season only.

   Then run `runitback college cbbd` and `player-values`. Tune the ridge penalty and replacement
   minutes on the next season's possessions, as N2 did. Report the 2024–2026 effect separately, as
   C1 asked.
2. **C4:** register `LINEUP` on both college `Sport`s, join `pv.parquet` on `(game_id, team_id)`,
   and run `prod` / `pbp32` / `pbp32_pvc` with 4 seeds plus the swap test.
3. **Better values for changes.** Tune the box-value shrinkage on how well within-season changes
   predict margins, not only on season-level fit, so that a starter's absence is valued at what it
   costs. Opponent-adjust the rates by the schedule's strength. Add a freshman prior from recruiting
   rank (CBBD `/recruiting/players`; one call a year).
4. **Live rotations** for upcoming games: ESPN team rosters plus recent minutes. There is no college
   injury feed, so posted starters, where ESPN has them, are the only availability signal.
