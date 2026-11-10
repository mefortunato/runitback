"""NHL events table: every play-by-play event with the skaters and goalie on ice for both teams.

Source: sportsdataverse's `nhl_pbp_full` release (see `download.py`), which merges the NHL's shift
charts into the NHL API's events. Normalization, per season:
1. **Game ids.** The release leaves `game_id` empty on every `CHANGE` row. Rows are in game order and
   `event_idx` restarts at 0 for each game, so a change belongs to the game of the rows around it.
2. **Personnel.** `home_on_1..7_id` / `away_on_1..7_id` are the skaters (the goalie is separate in
   `home_goalie_id` / `away_goalie_id`; any goalie found among the skaters is moved there). Skaters
   are written sorted into `home_1..6` / `away_1..6`, with the count in `n_home` / `n_away`
   (6 with the goalie pulled; 7, a mid-change glitch, keeps the first 6 ids). The goalie is
   missing while pulled. On a `CHANGE` row the personnel is the one after the change.
3. **Strength.** `home_skaters` / `away_skaters` as in the release (the goalie not counted), and
   `strength_state` from the home side ("5v4" = home power play). The release's own
   `strength_state` is from the event team's side, so it is not kept.
4. **Teams.** `event_team` is 'home' / 'away'. For a `CHANGE` (which the release does not
   attribute) it is the side whose players the change moves.

**Shift-chart fallback.** A game is a gap when under 99% of its play events (not changes, period /
game ends or the shootout) have skaters on ice for both teams, or when it has no `CHANGE` rows
(174 games of 2019-20 have every event but no personnel). Such a game's personnel is rebuilt from
the NHL's shift charts (`api.nhle.com/stats/rest/en/shiftcharts`): its `CHANGE` rows are replaced by
ones made from the shift starts and ends, and every event gets the players on ice at its time. The
convention matches the release's: at second t a shift [start, end) covers a faceoff or period start
(players coming on at t take the draw), while any other event at t still sees the players whose
shift ends at t (a goal, a stoppage, an on-the-fly change after a shot). `personnel` records the
source per row: 'pbp', 'shifts', or 'none'. `validate_shifts` measures the rebuild against the
release's personnel on games that have both.

Outputs in the processed directory:
- events_<season>.parquet  one row per event, in game order (`event_idx` 0..n-1; the release's own
                           index is `source_idx`, -1 on rebuilt changes)
- games.parquet            one row per game with events: date, teams, final score, periods,
                           shootout, number of events, personnel source and coverage
- players.parquet          player directory: id, name, position (C / L / R / D / G), group (F / D / G),
                           shoots, first / last season
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .download import GAME_TYPES, load_shifts

log = logging.getLogger(__name__)
MAX_SKATERS = 6
HOME_SKATERS = [f"home_{i}" for i in range(1, MAX_SKATERS + 1)]
AWAY_SKATERS = [f"away_{i}" for i in range(1, MAX_SKATERS + 1)]
LINEUP_COLS = HOME_SKATERS + AWAY_SKATERS + ["home_goalie", "away_goalie"]
SRC_HOME = [f"home_on_{i}_id" for i in range(1, 8)]
SRC_AWAY = [f"away_on_{i}_id" for i in range(1, 8)]
CHANGE = "CHANGE"
# rows that need nobody on ice (and the shootout)
NO_PERSONNEL = {CHANGE, "PERIOD_END", "GAME_END", "SHOOTOUT_COMPLETE", "GAME_SCHEDULED", "PERIOD_READY",
                "PERIOD_OFFICIAL", "EARLY_INT_START", "EARLY_INT_END"}
AFTER_CHANGE = {"FACEOFF", "PERIOD_START"}  # see the players coming on at their second
SEASON_TYPES = {"R": "REG", "P": "POST"}
GROUP = {"C": "F", "L": "F", "R": "F", "F": "F", "D": "D", "G": "G"}
MIN_COVERAGE = 0.99
PLAYER_COLS = [f"player_{k}_id" for k in (1, 2, 3)]

SOURCE_COLUMNS = (
    ["game_id", "season", "season_type", "game_date", "event_idx", "event_type", "secondary_type",
     "event_team_type", "period", "period_type", "period_seconds", "game_seconds", "home_score", "away_score",
     "home_abbr", "away_abbr", "event_goalie_id", "penalty_severity", "penalty_minutes", "empty_net",
     "x_fixed", "y_fixed", "shot_distance", "shot_angle", "xg", "home_skaters", "away_skaters",
     "home_goalie_id", "away_goalie_id", "ids_on", "ids_off"]
    + [f"event_player_{k}_{s}" for k in (1, 2, 3) for s in ("id", "type")] + SRC_HOME + SRC_AWAY)

EVENT_COLUMNS = (
    ["game_id", "season", "game_type", "game_date", "event_idx", "source_idx", "event_type", "secondary_type",
     "event_team", "period", "period_type", "period_seconds", "game_seconds", "home_score", "away_score"]
    + PLAYER_COLS + ["player_1_type", "player_2_type", "player_3_type", "event_goalie_id",
                     "penalty_severity", "penalty_minutes", "x_fixed", "y_fixed", "shot_distance", "shot_angle",
                     "xg", "empty_net", "home_skaters", "away_skaters", "strength_state", "n_home", "n_away"]
    + LINEUP_COLS + ["ids_on", "ids_off", "personnel"])


# ---------------------------------------------------------------- release rows -> events

def _ids(s: pd.Series) -> np.ndarray:
    """A player-id column as float (NaN = nobody). All-null columns come back as bool / object."""
    if s.dtype == bool or s.dtype == object and s.map(lambda v: isinstance(v, bool)).any():
        return np.full(len(s), np.nan)
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, na_value=np.nan, copy=True)
    v[v <= 0] = np.nan
    return v


def _id_list(s) -> str | None:
    """'8477979, 8478043' / '0' / None -> '8477979,8478043' / None."""
    if not isinstance(s, str):
        return None
    ids = [p.strip() for p in s.split(",")]
    ids = [p for p in ids if p and p != "0" and p.lower() not in ("nan", "none")]
    return ",".join(str(int(float(p))) for p in ids) or None


def parse_ids(s) -> list[int]:
    return [int(p) for p in s.split(",")] if isinstance(s, str) and s else []


def assign_game_ids(df: pd.DataFrame) -> pd.Series:
    """Fill the game id of the release's CHANGE rows (they have none) from the rows around them: a new
    game starts where `event_idx` stops increasing or the teams change."""
    idx = df["event_idx"].to_numpy()
    new = np.r_[True, idx[1:] <= idx[:-1]]
    for c in ("home_abbr", "away_abbr", "season_type"):
        if c in df:
            v = df[c].astype(str).to_numpy()
            new |= np.r_[True, v[1:] != v[:-1]]
    seg = np.cumsum(new)
    gid = pd.to_numeric(df["game_id"], errors="coerce")
    return gid.groupby(seg).transform(lambda g: g.bfill().ffill())


def _side_of_change(ev: pd.DataFrame) -> pd.Series:
    """'home' / 'away' for each CHANGE row: the side whose skaters or goalie include the players it moves."""
    side = pd.Series(pd.NA, index=ev.index, dtype="object")
    ch = ev.index[ev["event_type"] == CHANGE]
    if not len(ch):
        return side
    home_cols, away_cols = HOME_SKATERS + ["home_goalie"], AWAY_SKATERS + ["away_goalie"]
    roster = {}
    for g, grp in ev.groupby("game_id", sort=False):
        h = set(pd.unique(grp[home_cols].to_numpy().ravel()))
        a = set(pd.unique(grp[away_cols].to_numpy().ravel()))
        roster[g] = (h, a)
    games, on, off = ev.loc[ch, "game_id"].to_numpy(), ev.loc[ch, "ids_on"].to_numpy(), ev.loc[ch, "ids_off"].to_numpy()
    out = []
    for g, a, b in zip(games, on, off):
        ids = parse_ids(a) + parse_ids(b)
        h, aw = roster[g]
        nh, na = sum(p in h for p in ids), sum(p in aw for p in ids)
        out.append("home" if nh > na else "away" if na > nh else pd.NA)
    side.loc[ch] = out
    return side


def _personnel(home: np.ndarray, away: np.ndarray, home_goalie: np.ndarray, away_goalie: np.ndarray,
               goalies: set[int]) -> dict[str, np.ndarray]:
    """Skater id arrays (rows x 7, NaN = empty) and goalie ids -> sorted lineup columns + counts.
    Known goalies among the skaters are moved to the goalie column."""
    out = {}
    gl = np.array(sorted(goalies), dtype=float)
    for side, sk, gk in (("home", home.copy(), home_goalie.copy()), ("away", away.copy(), away_goalie.copy())):
        is_g = np.isin(sk, gl) | (sk == gk[:, None])
        found = np.where(is_g, sk, np.nan)
        fill = np.isnan(gk) & ~np.all(np.isnan(found), axis=1)
        gk[fill] = np.nanmax(found[fill], axis=1)
        sk[is_g] = np.nan
        sk = np.sort(sk, axis=1)  # NaN last
        out[f"n_{side}"] = (~np.isnan(sk)).sum(axis=1)
        for i in range(MAX_SKATERS):
            out[f"{side}_{i + 1}"] = sk[:, i]
        out[f"{side}_goalie"] = gk
    return out


def normalize(raw: pd.DataFrame, season: int, goalies: set[int] = frozenset()) -> pd.DataFrame:
    """The release's rows for one season -> the events table (personnel from the release)."""
    df = raw.reset_index(drop=True)
    df = df.assign(game_id=assign_game_ids(df))
    df = df[df["game_id"].notna()].copy()
    df["game_id"] = df["game_id"].astype("int64")
    df = df.sort_values(["game_id", "event_idx"], kind="stable").reset_index(drop=True)
    col = lambda c: df[c] if c in df else pd.Series(np.nan, index=df.index)  # noqa: E731
    known = goalies | set(pd.unique(np.r_[_ids(col("home_goalie_id")), _ids(col("away_goalie_id")),
                                          _ids(col("event_goalie_id"))]))
    known = {int(g) for g in known if g == g}
    pers = _personnel(np.column_stack([_ids(col(c)) for c in SRC_HOME]),
                      np.column_stack([_ids(col(c)) for c in SRC_AWAY]),
                      _ids(col("home_goalie_id")), _ids(col("away_goalie_id")), known)
    out = pd.DataFrame({
        "game_id": df["game_id"], "season": season,
        "game_type": col("season_type").map(SEASON_TYPES),
        "game_date": col("game_date").astype("string"),
        "source_idx": df["event_idx"].astype("int64"),
        "event_type": df["event_type"].astype("string"),
        "secondary_type": col("secondary_type").astype("string"),
        "event_team": col("event_team_type").where(col("event_team_type").isin(["home", "away"])).astype("object"),
        "period": df["period"].astype("int64"), "period_type": col("period_type").astype("string"),
        "period_seconds": pd.to_numeric(df["period_seconds"]).astype("int64"),
        "game_seconds": pd.to_numeric(df["game_seconds"]).astype("int64"),
        "home_score": pd.to_numeric(df["home_score"]).astype("int64"),
        "away_score": pd.to_numeric(df["away_score"]).astype("int64"),
        **{f"player_{k}_id": _ids(col(f"event_player_{k}_id")) for k in (1, 2, 3)},
        **{f"player_{k}_type": col(f"event_player_{k}_type").astype("string") for k in (1, 2, 3)},
        "event_goalie_id": _ids(col("event_goalie_id")),
        "penalty_severity": col("penalty_severity").astype("string"),
        "penalty_minutes": pd.to_numeric(col("penalty_minutes"), errors="coerce"),
        "x_fixed": pd.to_numeric(col("x_fixed"), errors="coerce"),
        "y_fixed": pd.to_numeric(col("y_fixed"), errors="coerce"),
        "shot_distance": pd.to_numeric(col("shot_distance"), errors="coerce"),
        "shot_angle": pd.to_numeric(col("shot_angle"), errors="coerce"),
        "xg": pd.to_numeric(col("xg"), errors="coerce"),
        "empty_net": col("empty_net").map(lambda v: bool(v) if isinstance(v, (bool, np.bool_)) else pd.NA)
                                     .astype("boolean"),
        "home_skaters": pd.to_numeric(col("home_skaters"), errors="coerce"),
        "away_skaters": pd.to_numeric(col("away_skaters"), errors="coerce"),
        **pers,
        "ids_on": col("ids_on").map(_id_list), "ids_off": col("ids_off").map(_id_list),
        "home_abbr": col("home_abbr").astype("string"), "away_abbr": col("away_abbr").astype("string"),
    })
    out["event_team"] = out["event_team"].where(out["event_type"] != CHANGE, _side_of_change(out))
    out["personnel"] = "pbp"
    return finish(out)


def finish(ev: pd.DataFrame) -> pd.DataFrame:
    """Shared last steps: event order, integer id columns, strength from the home side."""
    ev = ev.reset_index(drop=True)
    ev["event_idx"] = ev.groupby("game_id").cumcount().astype("int64")
    for c in PLAYER_COLS + ["event_goalie_id"] + LINEUP_COLS:
        ev[c] = pd.array(ev[c], dtype="Float64").astype("Int64")
    for c in ("n_home", "n_away"):
        ev[c] = ev[c].astype("int64")
    for c in ("home_skaters", "away_skaters"):
        ev[c] = pd.array(ev[c], dtype="Float64").astype("Int64")
    ok = ev["home_skaters"].notna() & ev["away_skaters"].notna()
    ev["strength_state"] = (ev["home_skaters"].astype("string") + "v" + ev["away_skaters"].astype("string")).where(ok)
    ev["personnel"] = ev["personnel"].where(ev["n_home"].gt(0) | ev["n_away"].gt(0)
                                            | ev["event_type"].isin(NO_PERSONNEL), "none")
    extra = [c for c in ("home_abbr", "away_abbr") if c in ev]
    return ev[EVENT_COLUMNS + extra]


# ---------------------------------------------------------------- coverage and gaps

def play_rows(ev: pd.DataFrame) -> pd.Series:
    """Events that need players on ice: not changes, period / game ends or the shootout."""
    return ~ev["event_type"].isin(NO_PERSONNEL) & ev["period_type"].ne("SHOOTOUT")


def coverage(ev: pd.DataFrame) -> pd.DataFrame:
    """Per game: share of play events with skaters on ice for both teams, and whether it has changes."""
    play = play_rows(ev)
    both = ev["n_home"].gt(0) & ev["n_away"].gt(0)
    g = pd.DataFrame({"game_id": ev["game_id"], "covered": both.where(play), "change": ev["event_type"].eq(CHANGE)})
    return g.groupby("game_id").agg(coverage=("covered", "mean"), changes=("change", "sum")).fillna({"coverage": 0.0})


def gap_games(ev: pd.DataFrame, min_coverage: float = MIN_COVERAGE) -> list[int]:
    c = coverage(ev)
    return c.index[(c["coverage"] < min_coverage) | (c["changes"] == 0)].astype(int).tolist()


def validation_games(ev: pd.DataFrame, n: int) -> list[int]:
    """A fixed sample of `n` complete games (not gaps), for checking the shift-chart rebuild."""
    c = coverage(ev)
    full = pd.Series(c.index[(c["coverage"] >= MIN_COVERAGE) & (c["changes"] > 0)].astype(int))
    return sorted(full.sample(min(n, len(full)), random_state=0).tolist()) if n and len(full) else []


# ---------------------------------------------------------------- shift-chart fallback

def team_sides(shifts: pd.DataFrame, game: pd.DataFrame) -> dict[int, str] | None:
    """Shift-chart team id -> 'home' / 'away', by abbreviation, else by which side's events the
    team's players are credited with."""
    teams = shifts.groupby("team_id")["team"].first()
    if len(teams) != 2:
        return None
    home, away = (game[c].dropna().iloc[0] if c in game and game[c].notna().any() else None
                  for c in ("home_abbr", "away_abbr"))
    if home in teams.values and away in teams.values and home != away:
        return {int(t): ("home" if a == home else "away") for t, a in teams.items()}
    credited = game[game["event_team"].isin(["home", "away"]) & game["player_1_id"].notna()]
    votes = {}
    for t in teams.index:
        players = set(shifts.loc[shifts["team_id"] == t, "player_id"])
        on = credited[credited["player_1_id"].astype("int64").isin(players)]["event_team"]
        votes[int(t)] = (on == "home").sum() - (on == "away").sum()
    (t1, v1), (t2, v2) = votes.items()
    if v1 == v2:
        return None
    return {t1: "home", t2: "away"} if v1 > v2 else {t1: "away", t2: "home"}


def shift_changes(shifts: pd.DataFrame, sides: dict[int, str]) -> list[tuple]:
    """(period, second, side, ids on, ids off) for every moment a side's players on ice change, in order."""
    s = shifts.dropna(subset=["start", "end"])
    s = s[(s["end"] > s["start"]) & s["team_id"].isin(sides)]
    out = []
    for (p, t), grp in pd.concat([s.assign(t=s["start"], on=True), s.assign(t=s["end"], on=False)]).groupby(
            ["period", "t"], sort=True):
        for side in ("home", "away"):
            g = grp[grp["team_id"].map(sides) == side]
            on, off = set(g.loc[g["on"], "player_id"]), set(g.loc[~g["on"], "player_id"])
            on, off = on - off, off - on  # a player whose shift ends and restarts at t never left
            if on or off:
                out.append((int(p), float(t), side, sorted(on), sorted(off)))
    return out


def personnel_from_shifts(game: pd.DataFrame, shifts: pd.DataFrame, goalies: set[int]) -> pd.DataFrame | None:
    """One game's events with personnel rebuilt from its shift charts (CHANGE rows replaced). None
    when the charts are unusable (not two teams, or a side missing from a period with play events)."""
    sides = team_sides(shifts, game)
    if sides is None:
        return None
    ev = game[game["event_type"] != CHANGE].reset_index(drop=True)
    side_of = shifts["team_id"].map(sides)
    for p in ev.loc[play_rows(ev), "period"].unique():
        in_p = side_of[shifts["period"] == p]
        if not {"home", "away"} <= set(in_p):
            return None
    changes = shift_changes(shifts, sides)
    offset = (ev["game_seconds"] - ev["period_seconds"]).groupby(ev["period"]).first()
    state = {"home": set(), "away": set()}
    rows, k = [], 0
    blank = {c: pd.NA for c in ev.columns}

    def emit_change(base: dict, ch: tuple):
        p, t, side, on, off = ch
        state[side] -= set(off)
        state[side] |= set(on)
        r = {**blank, **{c: base[c] for c in ("game_id", "season", "game_type", "game_date", "home_score",
                                               "away_score", "home_abbr", "away_abbr") if c in base}}
        r.update(source_idx=-1, event_type=CHANGE, event_team=side, period=p, period_seconds=int(t),
                 period_type=base["period_type"] if base.get("period") == p else ("REGULAR" if p <= 3 else "OVERTIME"),
                 game_seconds=int(t + offset.get(p, (p - 1) * 1200)),
                 ids_on=",".join(map(str, on)) or None, ids_off=",".join(map(str, off)) or None)
        rows.append(snapshot(r))

    def snapshot(r: dict) -> dict:
        for side in ("home", "away"):
            g = sorted(p for p in state[side] if p in goalies)
            sk = sorted(p for p in state[side] if p not in goalies)
            r[f"n_{side}"] = len(sk)
            for i in range(MAX_SKATERS):
                r[f"{side}_{i + 1}"] = sk[i] if i < len(sk) else np.nan
            r[f"{side}_goalie"] = g[0] if g else np.nan
            r[f"{side}_skaters"] = len(sk)
        r["personnel"] = "shifts"
        return r

    for r in ev.to_dict("records"):
        p, t = r["period"], r["period_seconds"]
        after = r["event_type"] in AFTER_CHANGE
        while k < len(changes) and (changes[k][0] < p or changes[k][0] == p and
                                    (changes[k][1] < t or changes[k][1] == t and after)):
            emit_change(r, changes[k])
            k += 1
        rows.append(snapshot(dict(r)))
    last = ev.iloc[-1].to_dict() if len(ev) else {}
    for ch in changes[k:]:
        emit_change(last, ch)
    out = pd.DataFrame(rows, columns=ev.columns)
    # a shootout has nobody on ice (and the release has no strength there)
    so = out["period_type"].eq("SHOOTOUT")
    out.loc[so, ["home_skaters", "away_skaters"]] = np.nan
    return out


def fill_from_shifts(ev: pd.DataFrame, raw_dir: Path, goalies: set[int], games: list[int] | None = None
                     ) -> tuple[pd.DataFrame, dict]:
    """Rebuild the personnel of the gap games (default: `gap_games(ev)`) from cached shift charts."""
    games = gap_games(ev) if games is None else games
    stats = dict(gap_games=len(games), filled=0, no_shifts=0, unusable=0)
    if not games:
        return ev, stats
    replaced = {}
    for g in games:
        shifts = load_shifts(g, raw_dir)
        if shifts is None or not len(shifts):
            stats["no_shifts"] += 1
            continue
        new = personnel_from_shifts(ev[ev["game_id"] == g], shifts, goalies)
        if new is None:
            stats["unusable"] += 1
            continue
        replaced[g] = new
        stats["filled"] += 1
    if replaced:
        ev = pd.concat([ev[~ev["game_id"].isin(list(replaced))], *replaced.values()], ignore_index=True)
        ev = ev.sort_values(["game_id"], kind="stable")
        ev = finish(ev)
    return ev, stats


def personnel_agreement(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Share of play events (matched on game_id, source_idx) where two personnel assignments agree:
    the same skaters for both sides, and the same skaters and goalies."""
    key = ["game_id", "source_idx"]
    pa, pb = a[play_rows(a) & a["source_idx"].ge(0)], b[play_rows(b) & b["source_idx"].ge(0)]
    m = pa[key + LINEUP_COLS].merge(pb[key + LINEUP_COLS], on=key, suffixes=("_a", "_b"))
    same = lambda cols: np.all([m[f"{c}_a"].fillna(-1).to_numpy() == m[f"{c}_b"].fillna(-1).to_numpy()  # noqa: E731
                                for c in cols], axis=0)
    sk = same(HOME_SKATERS + AWAY_SKATERS)
    allc = sk & same(["home_goalie", "away_goalie"])
    return dict(events=int(len(m)), skaters=float(sk.mean()) if len(m) else float("nan"),
                skaters_and_goalies=float(allc.mean()) if len(m) else float("nan"))


def validate_shifts(ev: pd.DataFrame, raw_dir: Path, goalies: set[int], games: list[int]) -> dict:
    """Rebuild `games` (which have release personnel) from their shift charts and compare."""
    rebuilt = []
    for g in games:
        shifts = load_shifts(g, raw_dir)
        if shifts is not None and len(shifts):
            new = personnel_from_shifts(ev[ev["game_id"] == g], shifts, goalies)
            if new is not None:
                rebuilt.append(new)
    if not rebuilt:
        return dict(games=0)
    new = finish(pd.concat(rebuilt, ignore_index=True))
    return dict(games=len(rebuilt), **personnel_agreement(ev[ev["game_id"].isin(new["game_id"].unique())], new))


# ---------------------------------------------------------------- tables

def load_players(seasons, raw_dir: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for s in seasons if (p := raw_dir / f"players_{s}.parquet").exists()]
    if not frames:
        return pd.DataFrame(columns=["player_id", "name", "position", "group", "shoots", "first_season",
                                     "last_season"])
    df = pd.concat(frames, ignore_index=True).sort_values(["season", "games"])
    span = df.groupby("player_id")["season"].agg(first_season="min", last_season="max")
    last = df.groupby("player_id").last()[["name", "position", "shoots"]]
    out = last.join(span).reset_index()
    out["group"] = out["position"].map(GROUP)
    return out[["player_id", "name", "position", "group", "shoots", "first_season", "last_season"]]


def goalie_ids(players: pd.DataFrame) -> set[int]:
    return set(players.loc[players["position"] == "G", "player_id"].astype(int))


def game_table(ev: pd.DataFrame, schedule: pd.DataFrame | None) -> pd.DataFrame:
    g = ev.groupby("game_id", sort=True)
    out = pd.DataFrame({
        "season": g["season"].first(), "game_type": g["game_type"].first(), "game_date": g["game_date"].first(),
        "home": g["home_abbr"].first(), "away": g["away_abbr"].first(),
        "home_score": g["home_score"].last(), "away_score": g["away_score"].last(),
        "periods": g["period"].max(), "shootout": g["period_type"].agg(lambda s: bool(s.eq("SHOOTOUT").any())),
        "n_events": g.size(),
        "personnel": g["personnel"].agg(lambda s: "shifts" if s.eq("shifts").any() else
                                        "pbp" if s.eq("pbp").any() else "none"),
    }).join(coverage(ev)[["coverage"]])
    if schedule is not None and len(schedule):
        sch = schedule.set_index("game_id")
        out["game_date"] = sch["game_date"].reindex(out.index).fillna(out["game_date"])
        out["game_type"] = sch["game_type"].reindex(out.index).fillna(out["game_type"])
        out["home_final"] = sch["home_final"].reindex(out.index)
        out["away_final"] = sch["away_final"].reindex(out.index)
    return out.reset_index()


def build_events(seasons, raw_dir: Path, out_dir: Path, validate_games: int = 0) -> dict:
    """events_<season>.parquet for every season with release data, games.parquet and players.parquet.
    `validate_games`: also rebuild this many complete games per season from cached shift charts and
    report the agreement with the release's personnel."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seasons = list(seasons)
    players = load_players(seasons, raw_dir)
    goalies = goalie_ids(players)
    summary, games = {}, []
    for s in seasons:
        path = raw_dir / f"pbp_{s}.parquet"
        if not path.exists():
            summary[s] = dict(events=0, note="no release file holds this season")
            log.warning("season %s: no play-by-play", s)
            continue
        import pyarrow.parquet as pq
        cols = [c for c in SOURCE_COLUMNS if c in set(pq.read_schema(path).names)]
        ev = normalize(pd.read_parquet(path, columns=cols), s, goalies)
        before = coverage(ev)
        check = validation_games(ev, validate_games)
        ev, fill = fill_from_shifts(ev, raw_dir, goalies)
        sch_path = raw_dir / f"games_{s}.parquet"
        schedule = pd.read_parquet(sch_path) if sch_path.exists() else None
        if schedule is not None:
            ev["game_date"] = ev["game_id"].map(schedule.set_index("game_id")["game_date"]).fillna(ev["game_date"])
        gt = game_table(ev, schedule)
        games.append(gt)
        play = play_rows(ev)
        both = ev["n_home"].gt(0) & ev["n_away"].gt(0)
        info = dict(
            games=int(ev["game_id"].nunique()), events=int(len(ev)),
            schedule_games=int(len(schedule)) if schedule is not None else None,
            schedule_games_missing=(int((~schedule["game_id"].isin(ev["game_id"]) & schedule["state"].eq(7)).sum())
                                    if schedule is not None else None),
            release_coverage=round(float(before["coverage"].mean()), 4),
            coverage=round(float(both[play].mean()), 4),
            games_full=int((gt["coverage"] >= MIN_COVERAGE).sum()),
            goalie_on_ice=round(float(ev.loc[play, "home_goalie"].notna().mean()), 4),
            xg_on_shots=round(float(ev.loc[ev["event_type"].isin(["SHOT", "GOAL", "MISSED_SHOT"])
                                           & ev["period_type"].ne("SHOOTOUT"), "xg"].notna().mean()), 4),
            strength_known=round(float(ev.loc[play, "strength_state"].notna().mean()), 4),
            **fill)
        if check:
            info["shift_check"] = validate_shifts(ev, raw_dir, goalies, check)
        ev.drop(columns=["home_abbr", "away_abbr"]).to_parquet(out_dir / f"events_{s}.parquet", index=False)
        summary[s] = info
        log.info("season %s: %s", s, info)
    if games:
        pd.concat(games, ignore_index=True).to_parquet(out_dir / "games.parquet", index=False)
    seen = set()
    for s in seasons:
        p = out_dir / f"events_{s}.parquet"
        if p.exists() and s in summary and summary[s].get("events"):
            e = pd.read_parquet(p, columns=LINEUP_COLS)
            seen |= {int(v) for v in pd.unique(e.to_numpy().ravel()) if pd.notna(v)}
    players.to_parquet(out_dir / "players.parquet", index=False)
    summary["players"] = dict(players=int(len(players)), on_ice=len(seen),
                              on_ice_without_position=len(seen - set(players["player_id"].astype(int))))
    return summary


def shift_games(seasons, raw_dir: Path, validate_games: int = 0) -> dict[int, list[int]]:
    """Per season, the games whose shift charts `download` should fetch: the gap games, plus a sample
    of complete games for `validate_games`."""
    import pyarrow.parquet as pq
    players = load_players(seasons, raw_dir)
    out = {}
    for s in seasons:
        path = raw_dir / f"pbp_{s}.parquet"
        if not path.exists():
            continue
        cols = [c for c in SOURCE_COLUMNS if c in set(pq.read_schema(path).names)]
        ev = normalize(pd.read_parquet(path, columns=cols), s, goalie_ids(players))
        out[s] = gap_games(ev) + validation_games(ev, validate_games)
    return out


__all__ = ["normalize", "assign_game_ids", "coverage", "gap_games", "shift_changes", "personnel_from_shifts",
           "fill_from_shifts", "validation_games", "personnel_agreement", "validate_shifts", "build_events", "shift_games",
           "GAME_TYPES"]
