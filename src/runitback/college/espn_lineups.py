"""Men's college basketball on-floor lineups and possessions from ESPN's substitution events.

sportsdataverse's ESPN men's play-by-play (``espn_mens_college_basketball_pbp``) has substitution
events (``type_text == "Substitution"``, "<Player> subbing in / out for <Team>", one athlete id each)
**from mid-February 2025 on**, and in every game from March 2025 (``experiments/cbb/lineup_probe.md``).
Earlier games have none. The ids are ESPN athlete ids, as in every other college table.

:func:`game_on_floor` rebuilds each side's five on every row:

- the game starts with the box score's five starters;
- each substitution takes its player off or puts him on (a sub-out of a player not on the floor,
  or a sub-in of one already on, is ignored);
- a period starts with the five that ended the previous one, and the substitutions logged at its
  start then apply;
- a player credited with an action (shot, rebound, turnover, foul, ...) who is not on the floor is
  put on (late-logged substitution) when his side has fewer than five.

Rows where a side does not have exactly five are marked invalid: possessions that start there are
dropped, not the whole game (:func:`season_possessions`). Only games where each team has at least
``MIN_SUBS`` substitution events are rebuilt.

Possessions follow the NBA's rules (:func:`runitback.nba.player_values.game_possessions`): a run of
one team's actions, ended by a made shot (unless an and-one follows), a made last free throw, a
turnover, the other team's defensive rebound or the end of the period; the target is the offense's
points (technical free throws excluded) and the ten players are those on the floor at its first
action.

:func:`floor_minutes` checks the rebuild against the box score: each player's seconds on the floor
(each row's five get the clock time until the next row) vs his box-score minutes.
"""
from __future__ import annotations

import itertools
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RELEASES = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
PBP_URL = RELEASES + "/espn_mens_college_basketball_pbp/play_by_play_{season}.parquet"
FIRST_SEASON = 2025            # substitutions start in mid-February 2025
MIN_SUBS = 10                  # substitution events per team for a game to be rebuilt
ON_FLOOR = 5
SUB = "Substitution"
PBP_COLUMNS = ["game_id", "game_play_number", "period_number", "clock_minutes", "clock_seconds", "type_text",
               "text", "team_id", "athlete_id_1", "home_team_id", "away_team_id", "scoring_play",
               "shooting_play", "score_value", "season"]
# rows whose credited player need not be on the floor
NOT_EVIDENCE = {SUB, "Technical Foul", "ShortTimeOut", "RegularTimeOut", "OfficialTVTimeOut", "End Period",
                "End Game", "Jumpball", "Dead Ball Rebound"}
POSS_COLUMNS = ["game_id", "season", "game_date", "t", "period", "off_home", "start", "points",
                *[f"off_{i}" for i in range(1, 6)], *[f"def_{i}" for i in range(1, 6)]]
FT_OF = re.compile(r"(\d)\s*of\s*(\d)")


def download_pbp(season: int, raw_dir: Path, force: bool = False) -> Path | None:
    """One season's ESPN men's play-by-play (the columns used here) to ``raw_dir/pbp_<season>.parquet``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"pbp_{season}.parquet"
    if path.exists() and not force:
        return path
    try:
        df = pd.read_parquet(PBP_URL.format(season=season), columns=PBP_COLUMNS)
    except Exception as e:  # noqa: BLE001 - not published (a season that hasn't started)
        log.warning("could not download the %d play-by-play (%s)", season, e)
        return None
    df.to_parquet(path, index=False)
    return path


def read_pbp(path: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    cols = [c for c in PBP_COLUMNS if c in set(pq.read_schema(path).names)]
    d = pd.read_parquet(path, columns=cols)
    d["game_id"] = d.game_id.astype("int64").astype(str)
    return d


def _id(x) -> int | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else int(x)


def sub_counts(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per game: substitution events of the home and the away team (``home_subs``, ``away_subs``)."""
    s = pbp[pbp.type_text.eq(SUB)]
    side = np.where(s.team_id == s.home_team_id, "home_subs", "away_subs")
    c = s.groupby([s.game_id, side]).size().unstack(fill_value=0)
    for k in ("home_subs", "away_subs"):
        if k not in c:
            c[k] = 0
    return c[["home_subs", "away_subs"]]


def rebuildable(pbp: pd.DataFrame, min_subs: int = MIN_SUBS) -> set[str]:
    c = sub_counts(pbp)
    return set(c.index[(c.home_subs >= min_subs) & (c.away_subs >= min_subs)])


def game_on_floor(g: pd.DataFrame, starters: dict[int, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """One game's rows (in order) -> ``(home, away)``: ``(n, 5)`` arrays of the players on the floor
    after each row (so from it until the next row; an action row's five are the five who made it),
    ``-1`` where the side does not have five (e.g. inside a block of substitutions).
    ``starters``: ``{team_id: five player ids}``."""
    home_id, away_id = int(g.home_team_id.iloc[0]), int(g.away_team_id.iloc[0])
    floor = {home_id: list(starters.get(home_id, []))[:ON_FLOOR], away_id: list(starters.get(away_id, []))[:ON_FLOOR]}
    n = len(g)
    out = {home_id: np.full((n, ON_FLOOR), -1, dtype=np.int64), away_id: np.full((n, ON_FLOOR), -1, dtype=np.int64)}
    types = g.type_text.astype(str).to_numpy()
    text = g.text.astype(str).str.lower().to_numpy()
    teams = g.team_id.to_numpy()
    players = g.athlete_id_1.to_numpy()
    for i in range(n):
        team, pid = _id(teams[i]), _id(players[i])
        if team in floor and pid is not None:
            f = floor[team]
            if types[i] == SUB:
                if "subbing out" in text[i]:
                    if pid in f:
                        f.remove(pid)
                elif "subbing in" in text[i] and pid not in f:
                    f.append(pid)
            elif types[i] not in NOT_EVIDENCE and pid not in f and len(f) < ON_FLOOR:
                f.append(pid)                   # a late-logged substitution
        for t, f in floor.items():
            if len(f) == ON_FLOOR:
                out[t][i] = sorted(f)
    return out[home_id], out[away_id]


def on_floor(pbp: pd.DataFrame, pg: pd.DataFrame, min_subs: int = MIN_SUBS) -> pd.DataFrame:
    """The rows of every rebuildable game in both ``pbp`` and ``pg`` (box-score starters), in game
    order, with ``home_1..5`` / ``away_1..5`` (``-1``: not five), ``valid`` (both sides five),
    ``period`` and ``sec`` (seconds left in the period)."""
    keep = rebuildable(pbp, min_subs) & set(pg.game_id)
    p = pbp[pbp.game_id.isin(keep)].sort_values(["game_id", "game_play_number"], kind="stable").reset_index(drop=True)
    st = pg[pg.starter & pg.game_id.isin(keep)].groupby(["game_id", "team_id"]).player_id.apply(list)
    starters: dict[str, dict[int, list[int]]] = {}
    for (gid, team), ids in st.items():
        starters.setdefault(gid, {})[int(team)] = ids
    homes, aways = [], []
    for gid, g in p.groupby("game_id", sort=False):
        h, a = game_on_floor(g, starters.get(gid, {}))
        homes.append(h)
        aways.append(a)
    h = np.concatenate(homes) if homes else np.zeros((0, 5), dtype=np.int64)
    a = np.concatenate(aways) if aways else np.zeros((0, 5), dtype=np.int64)
    for i in range(ON_FLOOR):
        p[f"home_{i + 1}"] = h[:, i]
        p[f"away_{i + 1}"] = a[:, i]
    p["valid"] = (h >= 0).all(1) & (a >= 0).all(1)
    p["period"] = p.period_number.astype(int)
    p["sec"] = (pd.to_numeric(p.clock_minutes, errors="coerce").fillna(0) * 60
                + pd.to_numeric(p.clock_seconds, errors="coerce").fillna(0)).to_numpy()
    return p


def floor_minutes(of: pd.DataFrame) -> pd.DataFrame:
    """Each player's minutes on the floor per game from :func:`on_floor` (each row's five get the clock
    time until the next row in the period; halves of 20 minutes, overtimes of 5): ``game_id,
    player_id, floor_minutes``, plus ``valid_share`` (the game's share of time with both fives)."""
    gid, per, sec = of.game_id.to_numpy(), of.period.to_numpy(), of.sec.to_numpy()
    nxt_same = np.r_[(gid[1:] == gid[:-1]) & (per[1:] == per[:-1]), False]
    first = np.r_[True, (gid[1:] != gid[:-1]) | (per[1:] != per[:-1])]
    start = np.where(first, np.where(per <= 2, 1200.0, 300.0), sec)
    stop = np.where(nxt_same, np.r_[sec[1:], 0.0], 0.0)
    dur = np.maximum(start - stop, 0.0)
    parts = []
    for side in ("home", "away"):
        ids = of[[f"{side}_{i}" for i in range(1, 6)]].to_numpy()
        ok = ids >= 0
        r, c = np.nonzero(ok)
        parts.append(pd.DataFrame({"game_id": gid[r], "player_id": ids[r, c], "s": dur[r]}))
    m = pd.concat(parts).groupby(["game_id", "player_id"]).s.sum().div(60).rename("floor_minutes").reset_index()
    vs = pd.DataFrame({"game_id": gid, "d": dur, "v": dur * of.valid.to_numpy()}).groupby("game_id")[["d", "v"]].sum()
    m["valid_share"] = m.game_id.map(vs.v / vs.d.where(vs.d > 0))
    return m


def check_minutes(of: pd.DataFrame, pg: pd.DataFrame) -> pd.DataFrame:
    """Per season: rebuilt minutes on the floor vs box-score minutes for the players who played in the
    rebuilt games (the C1 check, here for ESPN's substitutions)."""
    fm = floor_minutes(of)
    b = pg[pg.game_id.isin(set(of.game_id)) & ~pg.dnp][["game_id", "season", "player_id", "minutes", "starter"]]
    m = b.merge(fm.drop(columns="valid_share"), on=["game_id", "player_id"], how="left").fillna({"floor_minutes": 0.0})
    m["err"] = (m.floor_minutes - m.minutes).abs()
    vs = fm.drop_duplicates("game_id").set_index("game_id").valid_share
    rows = []
    for s, d in m.groupby("season"):
        g = d.game_id.unique()
        rows.append({"season": int(s), "games": len(g), "players": len(d),
                     "median_error_min": round(float(d.err.median()), 2),
                     "within_2_min": round(float((d.err <= 2).mean()), 3),
                     "within_5_min": round(float((d.err <= 5).mean()), 3),
                     "p95_error_min": round(float(d.err.quantile(0.95)), 1),
                     "time_with_both_fives": round(float(vs.reindex(g).mean()), 3)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------- possessions

def classify(p: pd.DataFrame) -> pd.DataFrame:
    """Per row: ``side`` (0 home / 1 away / -1), ``kind`` (the NBA adapter's action kinds), ``made``,
    ``last_ft``, ``pts``."""
    from ..nba.player_values import DREB, FT, OREB, OTHER, SHOT, TO

    tt = p.type_text.astype(str)
    text = p.text.astype(str).str.lower()
    side = np.where(p.team_id == p.home_team_id, 0, np.where(p.team_id == p.away_team_id, 1, -1))
    is_ft = tt.str.contains("FreeThrow") | text.str.contains("free throw")
    ft = text.str.extract(FT_OF)
    k, n = pd.to_numeric(ft[0], errors="coerce"), pd.to_numeric(ft[1], errors="coerce")
    tech = is_ft & text.str.contains("technical")
    # ESPN's older format ("X made Free Throw.", most games before 2025-26) has no "k of n": a free
    # throw is the last of its trip unless the next free throw is the same team's at the same clock
    fr = pd.DataFrame({"g": p.game_id.to_numpy(), "per": p.period_number.to_numpy(),
                       "clk": (pd.to_numeric(p.clock_minutes, errors="coerce") * 60
                               + pd.to_numeric(p.clock_seconds, errors="coerce")).to_numpy(),
                       "team": p.team_id.to_numpy()}, index=p.index)[is_ft & ~tech]
    nxt = fr.shift(-1)
    trip_goes_on = (nxt.g == fr.g) & (nxt.per == fr.per) & (nxt.clk == fr.clk) & (nxt.team == fr.team)
    last_unnumbered = pd.Series(False, index=p.index)
    last_unnumbered[fr.index] = ~trip_goes_on.to_numpy()
    numbered_last = k == n
    shot = p.shooting_play.fillna(False).astype(bool) & ~is_ft
    to = tt.str.contains("Turnover")
    kind = np.select([is_ft & ~tech, shot, to, tt == "Offensive Rebound", tt == "Defensive Rebound"],
                     [FT, SHOT, TO, OREB, DREB], OTHER)
    made = p.scoring_play.fillna(False).astype(bool).to_numpy()
    value = pd.to_numeric(p.score_value, errors="coerce").fillna(0).clip(0, 3).to_numpy()
    pts = np.where((kind == SHOT) & made, value, np.where((kind == FT) & made, 1, 0))
    return pd.DataFrame({"side": side.astype(np.int8), "kind": kind.astype(np.int8), "made": made,
                         "last_ft": ((kind == FT) & np.where(k.notna(), numbered_last, last_unnumbered)), "pts": pts.astype(np.int8)}, index=p.index)


def season_possessions(of: pd.DataFrame, pg: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Possessions (:data:`POSS_COLUMNS`) from :func:`on_floor` rows; possessions whose first action has
    no full ten are dropped. Returns them and counts."""
    from ..nba.player_values import game_possessions

    games = pg.drop_duplicates("game_id").set_index("game_id")
    cl = classify(of)
    period = of.period.to_numpy()
    side, kind, made, last_ft, pts = (cl[c].to_numpy() for c in ("side", "kind", "made", "last_ft", "pts"))
    gid = of.game_id.to_numpy()
    valid = of.valid.to_numpy()
    home = of[[f"home_{i}" for i in range(1, 6)]].to_numpy()
    away = of[[f"away_{i}" for i in range(1, 6)]].to_numpy()
    bounds = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1], True])
    rows, total = [], 0
    for a, b in itertools.pairwise(bounds):
        for r, s, st, pt in game_possessions(period[a:b], side[a:b], kind[a:b], made[a:b], last_ft[a:b], pts[a:b]):
            total += 1
            i = a + r
            if not valid[i]:
                continue
            off, dfn = (home[i], away[i]) if s == 0 else (away[i], home[i])
            rows.append((gid[i], int(period[i]), s == 0, st, pt, *off, *dfn))
    poss = pd.DataFrame(rows, columns=["game_id", "period", "off_home", "start", "points",
                                       *[f"off_{i}" for i in range(1, 6)], *[f"def_{i}" for i in range(1, 6)]])
    poss["season"] = poss.game_id.map(games.season).astype(int)
    poss["game_date"] = poss.game_id.map(games.game_date).astype(str)
    poss["t"] = pd.to_datetime(poss.game_date).sub(pd.Timestamp("1970-01-01")).dt.days.astype(np.int32)
    poss["start"] = poss.start.astype(np.int8)
    poss["points"] = poss.points.astype(np.int8)
    stats = {"games": int(len(bounds) - 1), "possessions": int(total), "kept": len(poss)}
    return poss[POSS_COLUMNS].reset_index(drop=True), stats


def build_possessions(seasons: list[int], raw_dir: Path, processed: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``possessions.parquet`` (processed) from the seasons' play-by-play in ``raw_dir`` and the box
    scores' starters (``player_games.parquet``). Returns the possessions and the minutes check."""
    pg = pd.read_parquet(processed / "player_games.parquet")
    polls, checks, stats = [], [], {}
    for s in seasons:
        path = raw_dir / f"pbp_{s}.parquet"
        if not path.exists():
            log.warning("season %d: %s missing; skipped", s, path.name)
            continue
        of = on_floor(read_pbp(path), pg[pg.season == s])
        if not len(of):
            continue
        p, st = season_possessions(of, pg)
        stats[s] = st
        polls.append(p)
        checks.append(check_minutes(of, pg))
        log.info("season %d: %s", s, st)
    poss = pd.concat(polls, ignore_index=True) if polls else pd.DataFrame(columns=POSS_COLUMNS)
    poss.to_parquet(processed / "possessions.parquet", index=False)
    check = pd.concat(checks, ignore_index=True) if checks else pd.DataFrame()
    if len(check):
        check["possessions_kept"] = check.season.map(lambda s: round(stats[s]["kept"] / max(stats[s]["possessions"], 1), 3))
    return poss, check
