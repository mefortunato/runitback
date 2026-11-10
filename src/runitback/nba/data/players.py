"""NBA players: who is on the court for every play-by-play event, plus per-game player tables.

Sources (sportsdataverse releases, ESPN data):
- play-by-play (`espn_nba_pbp`, the file `download.py` already caches): every substitution names
  both players ("X enters the game for Y": `athlete_id_1` in, `athlete_id_2` out).
- player box scores (`espn_nba_player_boxscores`): team, starter flag, minutes, plus-minus,
  did-not-play flag and reason, ESPN position.

On-court reconstruction, per game, period and team:
1. **Who starts the period.** Q1: the box-score starters. Later periods have no logged substitutions
   at the break, so the five are inferred: every player whose first appearance in the period is not
   as the incoming player of a substitution (he was subbed out, or he shot, rebounded, fouled, ...)
   was on court at the start. A side with fewer than five is filled from the five on court at the end
   of the previous period (for Q1, the box starters), then by minutes played; one with more than
   five keeps those with the earliest appearances.
2. **Walk the period.** Substitutions swap players in and out. ESPN occasionally drops a
   substitution, so the walk repairs gaps with look-ahead: a player who appears in an event (or is
   subbed out) while not on court is put on, and the player taken off for him is the one whose next
   appearance is coming back in (he must be off by then), else one who never appears again in the
   period. Whenever a side has fewer than five, the next player who must be on court is added.
   ESPN logs dead-ball events and substitutions at the same clock in either order, so a player seen
   at the clock he was subbed out (or in) at is not a missing substitution. 2018-2020 also log some
   substitutions in halves (only the incoming or only the outgoing id); the repairs supply the rest.
Technical fouls, ejections and delay-of-game rows are not evidence (bench players and coaches get
them). The lineup recorded for a row is the one after applying it. For crediting points use
`credit_lineups`: the five on court at the start of the row's clock instant, before any
substitutions at that clock, which is the box score's plus-minus convention.

Outputs in the processed directory:
- on_court.parquet      one row per play-by-play event: `game_id`, `game_play_number`, `period`,
                        `clock` (seconds left in the period), `home_score`, `away_score`,
                        `home_1..5`, `away_1..5` (ESPN athlete ids, sorted; missing as <NA>),
                        `n_home`, `n_away`
- player_games.parquet  one row per (game, team, player) in the box score: starter, minutes,
                        on-court minutes and plus-minus from the reconstruction (`oncourt_min`,
                        `oncourt_pm`; NaN for games with no play-by-play), plus-minus, DNP flag, reason and
                        kind (injury / rest / personal / suspension / coach / other), ESPN position
                        and group (G / F / C), and the player's earlier career games
- players.parquet       player directory: id, name, latest position and group, first / last season,
                        latest team, games played
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..vocab import canon_team

log = logging.getLogger(__name__)
BOX_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
           "espn_nba_player_boxscores/player_box_{season}.parquet")
GAME_TYPES = {2: "REG", 3: "POST", 5: "POST"}  # as preprocess.py: regular season, playoffs, play-in

PBP_COLUMNS = ["game_id", "id", "game_play_number", "sequence_number", "period_number",
               "start_quarter_seconds_remaining", "type_text", "text", "team_id", "athlete_id_1", "athlete_id_2",
               "athlete_id_3", "home_team_id", "away_team_id", "home_score", "away_score"]
BOX_COLUMNS = ["game_id", "season", "season_type", "game_date", "athlete_id", "athlete_display_name",
               "athlete_short_name", "team_id", "team_abbreviation", "opponent_team_abbreviation", "home_away",
               "minutes", "plus_minus", "points", "starter", "did_not_play", "reason", "ejected",
               "athlete_position_abbreviation"]
# rows whose players need not be on the court
NOT_EVIDENCE = {"Technical Foul", "Double Technical Foul", "Hanging Technical Foul", "Ejection",
                "Delay Technical", "Excess Timeout Technical", "Too Many Players Technical", "Delay of Game",
                "Full Timeout", "Timeout", "End Period", "End Game", "Substitution"}
SUB = "Substitution"
GROUP = {"G": "G", "PG": "G", "SG": "G", "F": "F", "SF": "F", "PF": "F", "C": "C"}
ON_COURT = 5
HOME, AWAY = 0, 1
LINEUP_COLS = [f"home_{i}" for i in range(1, 6)] + [f"away_{i}" for i in range(1, 6)]


def position_group(pos) -> str | None:
    return GROUP.get(pos.upper()) if isinstance(pos, str) else None


def dnp_kind(reason) -> str | None:
    """Coarse kind of a did-not-play reason (ESPN free text, e.g. 'RIGHT ANKLE SPRAIN')."""
    if not isinstance(reason, str) or not reason.strip():
        return None
    r = reason.upper()
    if "COACH" in r:
        return "coach"
    if "SUSPEN" in r:
        return "suspension"
    if r.strip() == "REST" or "LOAD MANAGEMENT" in r:
        return "rest"
    if any(k in r for k in ("PERSONAL", "NOT WITH TEAM", "BEREAVEMENT", "FAMILY", "BIRTH")):
        return "personal"
    if any(k in r for k in ("G LEAGUE", "G-LEAGUE", "TWO-WAY", "ASSIGNED", "TRADE", "INACTIVE")):
        return "other"
    return "injury"  # body parts, illness, surgery, protocols, reconditioning


# --------------------------------------------------------------------------------------- download

def download_box(season: int, raw_dir: Path, force: bool = False) -> Path | None:
    """Download one season's ESPN player box scores to raw_dir/player_box_<season>.parquet."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"player_box_{season}.parquet"
    if path.exists() and not force:
        return path
    try:
        df = pd.read_parquet(BOX_URL.format(season=season))
    except Exception as e:  # not published (e.g. a season that hasn't started)
        log.warning("could not download the %d player box scores (%s)", season, e)
        return None
    df.to_parquet(path)
    return path


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq
    available = set(pq.read_schema(path).names)
    return pd.read_parquet(path, columns=[c for c in columns if c in available])


# ----------------------------------------------------------------------------- on-court lineups

def _id(x) -> int | None:
    if x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NA:
        return None
    return int(x)


def _sub_ids(r) -> tuple[int | None, int | None]:
    """(in, out) of a substitution row. Usually "X enters the game for Y" with both ids; 2018-2020
    also log halves: "X enters the game for " (only X's id) and " enters the game for Y" (only Y's
    id, in `athlete_id_1`). The walk's repairs supply the missing half."""
    a1, a2 = _id(r.athlete_id_1), _id(r.athlete_id_2)
    if a2 is not None:
        return a1, a2
    text = str(getattr(r, "text", "") or "").strip().lower()
    if text.startswith("enters the game"):
        return None, a1
    return a1, None


class _Side:
    """One team's touches within a period: ('in' | 'out' | 'seen', row index) per player, in order."""

    def __init__(self):
        self.touches: dict[int, list[tuple[int, str]]] = {}

    def add(self, pid: int, row: int, kind: str) -> None:
        self.touches.setdefault(pid, []).append((row, kind))

    def first(self, pid: int) -> tuple[int, str] | None:
        t = self.touches.get(pid)
        return t[0] if t else None

    def next_touch(self, pid: int, row: int) -> tuple[int, str] | None:
        for r, k in self.touches.get(pid, ()):
            if r > row:
                return r, k
        return None

    def prev_touch(self, pid: int, row: int) -> tuple[int, str] | None:
        last = None
        for r, k in self.touches.get(pid, ()):
            if r >= row:
                break
            last = (r, k)
        return last

    def logged_out_of_order(self, pid: int, row: int, clocks: list) -> bool:
        """ESPN logs dead-ball events and substitutions at the same clock in either order: a player
        seen just after being subbed out (or just before being subbed in) at the same clock was on
        court for that event, so it is no evidence of a missing substitution."""
        pt, nt = self.prev_touch(pid, row), self.next_touch(pid, row)
        return (pt is not None and pt[1] == "out" and clocks[pt[0]] == clocks[row]) or \
            (nt is not None and nt[1] == "in" and clocks[nt[0]] == clocks[row])


def _victim(lineup: set[int], side: _Side, row: int, keep: int | None) -> int | None:
    """Who to take off when a side has a player too many: the one coming back in soonest (he must be
    off by then), else one with no further touches in the period."""
    best, best_key = None, None
    for p in lineup:
        if p == keep:
            continue
        nt = side.next_touch(p, row)
        if nt is not None and nt[1] != "in":
            continue  # he must still be on court
        key = (0, nt[0]) if nt is not None else (1, 0)
        if best_key is None or key < best_key:
            best, best_key = p, key
    return best


def _must_be_on(lineup: set[int], side: _Side, row: int) -> int | None:
    """The player not on court whose next touch shows he is on court (seen, or subbed out), soonest."""
    best, best_row = None, None
    for p in side.touches:
        if p in lineup:
            continue
        nt = side.next_touch(p, row)
        if nt is not None and nt[1] != "in" and (best_row is None or nt[0] < best_row):
            best, best_row = p, nt[0]
    return best


def _period_starters(side: _Side, fallback: list[int], minutes: dict[int, float], clocks: list) -> set[int]:
    firsts = {}
    for p in side.touches:
        f = side.first(p)
        if f[1] == "seen" and side.logged_out_of_order(p, f[0], clocks):
            f = (f[0], "in")  # seen at the clock he was subbed in at: he came in
        firsts[p] = f
    on = sorted((f[0], p) for p, f in firsts.items() if f is not None and f[1] != "in")
    if len(on) > ON_COURT:
        on = on[:ON_COURT]
    start = {p for _, p in on}
    if len(start) < ON_COURT:
        for p in fallback:
            f = firsts.get(p)
            if p not in start and (f is None or f[1] != "in"):
                start.add(p)
                if len(start) == ON_COURT:
                    break
    if len(start) < ON_COURT:  # someone played the whole period without touching the ball
        for p in sorted(minutes, key=lambda q: -minutes[q]):
            f = firsts.get(p)
            if p not in start and f is None:
                start.add(p)
                if len(start) == ON_COURT:
                    break
    return start


def game_on_court(g: pd.DataFrame, box: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """On-court fives for one game's play-by-play rows `g`, given that game's box score rows.

    Returns the per-row lineup table and repair counts for logging."""
    order = "game_play_number" if "game_play_number" in g and g.game_play_number.notna().all() \
        else "sequence_number"
    g = g.assign(ordinal=pd.to_numeric(g[order], errors="coerce")).sort_values(
        ["period_number", "ordinal"], kind="stable")
    home_id = _id(g.home_team_id.iloc[0])
    side_of: dict[int, int] = {}
    minutes: list[dict[int, float]] = [{}, {}]
    starters: list[list[int]] = [[], []]
    for r in box.itertuples(index=False):
        pid, tid = _id(r.athlete_id), _id(r.team_id)
        if pid is None or tid is None:
            continue
        s = HOME if tid == home_id else AWAY
        side_of[pid] = s
        mins = 0.0 if pd.isna(r.minutes) else float(r.minutes)
        minutes[s][pid] = mins
        if bool(r.starter):
            starters[s].append(pid)
    starters = [sorted(st, key=lambda p, s=s: -minutes[s][p]) for s, st in enumerate(starters)]
    stats = dict(added=0, removed=0, filled=0)

    out = []
    prev_end: list[list[int]] = [list(starters[0]), list(starters[1])]
    rows = list(g.itertuples(index=False))
    periods = [int(p) for p in g.period_number.dropna().unique()]
    pos = 0
    for q in periods:
        prow = []
        while pos < len(rows) and rows[pos].period_number == q:
            prow.append(rows[pos])
            pos += 1
        sides = [_Side(), _Side()]
        subs: list[tuple[int, int | None, int | None] | None] = []
        seen: list[list[tuple[int, int]]] = []
        for i, r in enumerate(prow):
            seen.append([])
            tt = str(r.type_text)
            if tt == SUB:
                pin, pout = _sub_ids(r)
                s = side_of.get(pin, side_of.get(pout))
                if s is None:
                    subs.append(None)
                    continue
                pin = pin if side_of.get(pin) == s else None
                pout = pout if side_of.get(pout) == s else None
                if pout is not None:
                    sides[s].add(pout, i, "out")
                if pin is not None:
                    sides[s].add(pin, i, "in")
                subs.append((s, pin, pout))
                continue
            subs.append(None)
            if tt in NOT_EVIDENCE:
                continue
            for a in (r.athlete_id_1, r.athlete_id_2, r.athlete_id_3):
                p = _id(a)
                if p is not None and p in side_of:
                    sides[side_of[p]].add(p, i, "seen")
                    seen[i].append((side_of[p], p))
        clocks = [r.start_quarter_seconds_remaining for r in prow]
        fallback = [starters[s] if q == 1 else prev_end[s] for s in (HOME, AWAY)]
        if q == 1:
            lineups = [set(starters[s]) if len(starters[s]) == ON_COURT
                       else _period_starters(sides[s], fallback[s], minutes[s], clocks) for s in (HOME, AWAY)]
        else:
            lineups = [_period_starters(sides[s], fallback[s], minutes[s], clocks) for s in (HOME, AWAY)]
        for i, r in enumerate(prow):
            sub = subs[i]
            for s in (HOME, AWAY):
                lu, side = lineups[s], sides[s]
                if sub is not None and sub[0] == s:
                    _, pin, pout = sub
                    if pout is not None:
                        lu.discard(pout)
                    if pin is not None and pin not in lu:
                        lu.add(pin)
                        if len(lu) > ON_COURT:
                            v = _victim(lu, side, i, keep=pin)
                            if v is not None:
                                lu.discard(v)
                                stats["removed"] += 1
                elif sub is None:
                    for ps, p in seen[i]:
                        if ps == s and p not in lu and not side.logged_out_of_order(p, i, clocks):
                            lu.add(p)
                            stats["added"] += 1
                            if len(lu) > ON_COURT:
                                v = _victim(lu, side, i, keep=p)
                                if v is not None:
                                    lu.discard(v)
                                    stats["removed"] += 1
                while len(lu) < ON_COURT:
                    p = _must_be_on(lu, side, i)
                    if p is None:
                        break
                    lu.add(p)
                    stats["filled"] += 1
            h, a = sorted(lineups[HOME]), sorted(lineups[AWAY])
            out.append((r.game_id, r.ordinal, q, r.start_quarter_seconds_remaining, h, a, r.home_score,
                        r.away_score))
        prev_end = [sorted(lineups[s], key=lambda p, s=s: -minutes[s].get(p, 0)) for s in (HOME, AWAY)]

    def col(lu, k):
        return [x[k] if k < len(x) else pd.NA for x in lu]

    hs, as_ = [o[4] for o in out], [o[5] for o in out]
    df = pd.DataFrame({
        "game_id": [str(o[0]) for o in out], "game_play_number": [o[1] for o in out],
        "period": [o[2] for o in out], "clock": pd.to_numeric([o[3] for o in out], errors="coerce"),
        "home_score": pd.to_numeric([o[6] for o in out], errors="coerce"),
        "away_score": pd.to_numeric([o[7] for o in out], errors="coerce"),
    })
    for k in range(ON_COURT):
        df[f"home_{k + 1}"] = pd.array(col(hs, k), dtype="Int64")
    for k in range(ON_COURT):
        df[f"away_{k + 1}"] = pd.array(col(as_, k), dtype="Int64")
    df["n_home"] = np.array([len(x) for x in hs], dtype=np.int8)
    df["n_away"] = np.array([len(x) for x in as_], dtype=np.int8)
    return df, stats


def credit_lineups(oc: pd.DataFrame) -> pd.DataFrame:
    """The fives to credit each row's points to: those on court at the first row of its clock instant
    (game, period, clock), i.e. before any substitutions made at that clock. This is the box score's
    convention (free throws after a substitution at the same clock count for the five that were on
    when the clock stopped); `oc` must be in game order. Returns `LINEUP_COLS` aligned to `oc`."""
    key = [oc.game_id, oc.period, oc.clock.fillna(-1)]
    return oc.groupby(key, sort=False)[LINEUP_COLS].transform("first")


def on_court_totals(oc: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, player_id): seconds on court (`sec`, from consecutive rows' clocks) and the
    plus-minus while on court (`pm`, with points credited as `credit_lineups` does)."""
    oc = oc.sort_values(["game_id", "period", "game_play_number"], kind="stable").reset_index(drop=True)
    nxt = oc.groupby(["game_id", "period"]).clock.shift(-1).fillna(0)
    dur = (oc.clock - nxt).clip(lower=0).fillna(0).to_numpy()
    margin = (oc.home_score - oc.away_score).groupby(oc.game_id).ffill().fillna(0)
    d_margin = (margin - margin.groupby(oc.game_id).shift(1).fillna(0)).to_numpy()
    credit = credit_lineups(oc)
    parts = []
    for c in LINEUP_COLS:
        sign = 1 if c.startswith("home") else -1
        parts.append(pd.DataFrame({"game_id": oc.game_id.to_numpy(), "player_id": oc[c].to_numpy(), "sec": dur,
                                   "pm": 0.0}))
        parts.append(pd.DataFrame({"game_id": oc.game_id.to_numpy(), "player_id": credit[c].to_numpy(), "sec": 0.0,
                                   "pm": sign * d_margin}))
    t = pd.concat(parts, ignore_index=True).dropna(subset=["player_id"])
    return t.groupby(["game_id", "player_id"], as_index=False)[["sec", "pm"]].sum().astype({"player_id": "int64"})


def build_on_court(pbp: pd.DataFrame, box: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """On-court lineups for every game in `pbp` that has box-score rows."""
    box_by_game = {k: v for k, v in box.groupby(box.game_id.astype(str), sort=False)}
    frames, stats = [], dict(games=0, skipped=0, added=0, removed=0, filled=0)
    for gid, g in pbp.groupby(pbp.game_id.astype(str), sort=False):
        b = box_by_game.get(gid)
        if b is None or len(b) == 0:
            stats["skipped"] += 1
            continue
        df, st = game_on_court(g, b)
        frames.append(df)
        stats["games"] += 1
        for k, v in st.items():
            stats[k] += v
    oc = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["game_id", "game_play_number", "period", "clock", *LINEUP_COLS, "n_home", "n_away"])
    return oc, stats


def five_share(oc: pd.DataFrame) -> float:
    """Share of events with exactly five on court for both sides (the acceptance check: >= 0.99)."""
    return float(((oc.n_home == ON_COURT) & (oc.n_away == ON_COURT)).mean()) if len(oc) else float("nan")


# --------------------------------------------------------------------------------- player tables

def build_player_games(box: pd.DataFrame, totals: pd.DataFrame | None = None) -> pd.DataFrame:
    b = box.copy()
    b = b[b.athlete_id.notna()]
    pg = pd.DataFrame({
        "game_id": b.game_id.astype(str), "season": b.season.astype(int),
        "game_type": b.season_type.map(GAME_TYPES), "game_date": b.game_date.astype(str).str[:10],
        "team": b.team_abbreviation.map(lambda x: canon_team(str(x))),
        "opponent": b.opponent_team_abbreviation.map(lambda x: canon_team(str(x))),
        "home": b.home_away.eq("home"),
        "player_id": b.athlete_id.astype("int64"), "name": b.athlete_display_name,
        "position": b.athlete_position_abbreviation,
        "group": b.athlete_position_abbreviation.map(position_group),
        "starter": b.starter.fillna(False).astype(bool),
        "dnp": b.did_not_play.fillna(False).astype(bool),
        "minutes": pd.to_numeric(b.minutes, errors="coerce").fillna(0.0),
        "plus_minus": pd.to_numeric(b.plus_minus, errors="coerce"),
        "points": pd.to_numeric(b.points, errors="coerce"),
        "ejected": b.ejected.fillna(False).astype(bool) if "ejected" in b else False,
    })
    # ESPN fills `reason` ("COACH'S DECISION") for players who played too; keep it for DNPs only
    reason = b.reason if "reason" in b else pd.Series(None, index=b.index, dtype=object)  # not before 2010s
    pg["dnp_reason"] = reason.where(pg.dnp).where(lambda s: s.notna(), None)
    pg["dnp_kind"] = pg.dnp_reason.map(dnp_kind)
    pg.loc[pg.dnp & pg.dnp_kind.isna(), "dnp_kind"] = "other"
    if totals is not None and len(totals):
        pg = pg.merge(totals.rename(columns={"sec": "oncourt_sec", "pm": "oncourt_pm"}),
                      on=["game_id", "player_id"], how="left")
        has_pbp = pg.game_id.isin(set(totals.game_id))  # NaN where the game has no play-by-play
        pg["oncourt_min"] = (pg.pop("oncourt_sec").fillna(0) / 60).round(2).where(has_pbp)
        pg["oncourt_pm"] = pg.oncourt_pm.fillna(0).round().where(has_pbp).astype("Int64")
    else:
        pg["oncourt_min"], pg["oncourt_pm"] = np.nan, np.nan
    pg = pg.drop_duplicates(["game_id", "player_id"]).sort_values(
        ["game_date", "game_id", "home", "starter", "minutes"], ascending=[True, True, False, False, False],
        kind="stable").reset_index(drop=True)
    played = (~pg.dnp & (pg.minutes > 0)).astype(int)
    pg["career_games"] = played.groupby(pg.player_id).cumsum() - played  # games played before this one
    return pg


def build_player_directory(pg: pd.DataFrame) -> pd.DataFrame:
    pg = pg.sort_values(["game_date", "game_id"], kind="stable")
    played = pg[~pg.dnp & (pg.minutes > 0)]
    last = pg.groupby("player_id").tail(1).set_index("player_id")
    pos = pg[pg.position.notna()].groupby("player_id").position.last()
    d = pd.DataFrame({
        "player_id": last.index, "name": last.name.to_numpy(),
        "position": pos.reindex(last.index).to_numpy(),
        "first_season": pg.groupby("player_id").season.min().reindex(last.index).to_numpy(),
        "last_season": last.season.to_numpy(), "team": last.team.to_numpy(),
        "games": played.groupby("player_id").size().reindex(last.index).fillna(0).astype(int).to_numpy(),
    })
    d["group"] = d.position.map(position_group)
    return d.reset_index(drop=True)


# ------------------------------------------------------------------------------------------ build

def build_players(seasons: list[int], raw_dir: Path, out_dir: Path) -> dict:
    """Write on_court.parquet, player_games.parquet and players.parquet for the seasons whose
    play-by-play and box scores are in raw_dir. Returns the validation summary per season."""
    out_dir.mkdir(parents=True, exist_ok=True)
    oc_parts, boxes, totals, summary = [], [], [], {}
    for season in seasons:
        pbp_path, box_path = raw_dir / f"pbp_{season}.parquet", raw_dir / f"player_box_{season}.parquet"
        if not pbp_path.exists() or not box_path.exists():
            log.warning("season %d: %s or %s missing (download first); skipped", season, pbp_path.name,
                        box_path.name)
            continue
        box = _read(box_path, BOX_COLUMNS)
        box = box[box.season_type.isin(GAME_TYPES)]
        pbp = _read(pbp_path, PBP_COLUMNS)
        pbp = pbp[pbp.game_id.astype(str).isin(set(box.game_id.astype(str)))]
        oc, st = build_on_court(pbp, box)
        share = five_share(oc)
        summary[season] = dict(st, events=len(oc), five_share=round(share, 5))
        log.info("season %d: %d games, %d events, both sides five on court %.4f (repairs: %d added, "
                 "%d removed, %d filled; %d games without box)", season, st["games"], len(oc), share,
                 st["added"], st["removed"], st["filled"], st["skipped"])
        totals.append(on_court_totals(oc))  # per season: the all-seasons table is ~8M rows
        oc_parts.append(oc)
        boxes.append(box)
    if not oc_parts:
        raise FileNotFoundError(f"no seasons with both play-by-play and box scores in {raw_dir}")
    oc = pd.concat(oc_parts, ignore_index=True)
    del oc_parts
    oc.to_parquet(out_dir / "on_court.parquet")
    pg = build_player_games(pd.concat(boxes, ignore_index=True), pd.concat(totals, ignore_index=True))
    pg.to_parquet(out_dir / "player_games.parquet")
    build_player_directory(pg).to_parquet(out_dir / "players.parquet")
    played = pg[~pg.dnp & (pg.minutes > 0) & pg.oncourt_min.notna()]
    err = (played.oncourt_min - played.minutes).abs()
    pm = played[played.plus_minus.notna()]
    pm_exact = float((pm.oncourt_pm == pm.plus_minus).mean()) if len(pm) else float("nan")
    summary["all"] = dict(events=len(oc), five_share=round(five_share(oc), 5),
                          minutes_within_1=round(float((err <= 1).mean()), 4),
                          minutes_mae=round(float(err.mean()), 3),
                          plus_minus_exact=round(pm_exact, 4))
    log.info("all seasons: five on court %.4f over %d events; reconstructed minutes within 1 of the box "
             "score for %.1f%% of player-games (MAE %.2f); plus-minus exact for %.1f%%", summary["all"]["five_share"],
             len(oc), 100 * summary["all"]["minutes_within_1"], summary["all"]["minutes_mae"],
             100 * summary["all"]["plus_minus_exact"])
    return summary
