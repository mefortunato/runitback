"""Players, lineups and coaches from nflverse: per-game depth charts, per-play on-field personnel
and leak-free head-coach records.

Sources (nflverse-data releases):
- snap counts (PFR, 2012+): who played in each game, with offense / defense / special-teams snaps.
  These become each team's per-game depth chart (players ranked by snaps within a position group).
- pbp participation (2016-2025): the 22 players on the field for each play.
- players: id crosswalk (GSIS <-> PFR), positions and rookie season.
- schedules: the head coach of each team in each game (1999+).

Outputs in the processed directory:
- player_games.parquet  one row per (game, team, player) with snaps, position group, depth rank and
                        the number of earlier career games (`career_games`, for the NEW PLAYER rule)
- personnel.parquet     one row per play with the home and away players on the field
- coaches.parquet       one row per (game, side) with the head coach and his record before the game
- players.parquet       player directory (id, name, position, group, rookie season)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..vocab import TEAM_ALIASES

log = logging.getLogger(__name__)
BASE = "https://github.com/nflverse/nflverse-data/releases/download"
SNAPS_FIRST, PARTICIPATION_FIRST = 2012, 2016

# Position groups. Each player token is tied to one group, and NEW / OTHER fallbacks exist per group.
GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB", "K", "P", "LS"]
GROUP_IDX = {g: i for i, g in enumerate(GROUPS)}
OFFENSE_GROUPS = ("QB", "RB", "WR", "TE", "OL")
DEFENSE_GROUPS = ("DL", "LB", "DB")
SPECIALIST_GROUPS = ("K", "P", "LS")
# Depth-chart slots per group (players beyond this depth are left out of the game lineup).
DEPTH = {"QB": 3, "RB": 4, "WR": 6, "TE": 4, "OL": 9, "DL": 9, "LB": 7, "DB": 9, "K": 1, "P": 1, "LS": 1}
POSITION_GROUP = {
    "QB": "QB", "RB": "RB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL", "OL": "OL", "LT": "OL", "LG": "OL",
    "RG": "OL", "RT": "OL", "OC": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "EDGE": "DL", "DI": "DL", "LDE": "DL", "RDE": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB", "WLB": "LB", "SLB": "LB",
    "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB", "SAF": "DB", "NB": "DB",
    "K": "K", "PK": "K", "P": "P", "LS": "LS",
}


def position_group(pos, side_hint: str | None = None) -> str | None:
    if isinstance(pos, str) and pos.upper() in POSITION_GROUP:
        return POSITION_GROUP[pos.upper()]
    return None


def _team(x):
    return TEAM_ALIASES.get(x, x)


# --------------------------------------------------------------------------------------- download

def _fetch(url: str, path: Path, force: bool) -> Path | None:
    if path.exists() and not force:
        return path
    try:
        df = pd.read_csv(url, low_memory=False) if url.endswith(".csv") else pd.read_parquet(url)
    except Exception as e:  # not published (e.g. participation for the current season)
        log.warning("could not download %s (%s)", url, e)
        return None
    df.to_parquet(path)
    return path


def download_players(seasons: list[int], raw_dir: Path, force: bool = False) -> None:
    """Download snap counts, participation, the player directory and schedules into raw_dir/players."""
    d = raw_dir / "players"
    d.mkdir(parents=True, exist_ok=True)
    # the player directory and schedules change every week and are small: always refresh them
    _fetch(f"{BASE}/players/players.parquet", d / "players.parquet", True)
    _fetch(f"{BASE}/schedules/games.csv", d / "schedules.parquet", True)
    last = max(seasons)
    for s in seasons:
        recent = s >= last - 1  # the current and previous season are still being updated
        if s >= SNAPS_FIRST:
            _fetch(f"{BASE}/snap_counts/snap_counts_{s}.parquet", d / f"snap_counts_{s}.parquet", force or recent)
        if s >= PARTICIPATION_FIRST:
            _fetch(f"{BASE}/pbp_participation/pbp_participation_{s}.parquet",
                   d / f"participation_{s}.parquet", force or recent)


# ------------------------------------------------------------------------------------------ build

def _player_directory(raw: Path) -> pd.DataFrame:
    p = pd.read_parquet(raw / "players.parquet")
    p = p[p.gsis_id.notna()].drop_duplicates("gsis_id")
    grp = p.position.map(position_group)
    grp = grp.fillna(p.position_group.map(position_group))
    return pd.DataFrame({
        "player_id": p.gsis_id.astype(str), "name": p.display_name.fillna(p.gsis_id),
        "position": p.position.fillna(""), "group": grp, "pfr_id": p.pfr_id,
        "rookie_season": pd.to_numeric(p.rookie_season, errors="coerce"),
    })


def build_player_games(raw: Path, seasons: list[int], directory: pd.DataFrame,
                       game_dates: pd.DataFrame) -> pd.DataFrame:
    """Snap counts -> one row per (game, team, player), ranked within group, with career game counts."""
    frames = [pd.read_parquet(raw / f"snap_counts_{s}.parquet") for s in seasons
              if (raw / f"snap_counts_{s}.parquet").exists()]
    if not frames:
        return pd.DataFrame()
    sc = pd.concat(frames, ignore_index=True)
    sc["team"] = sc.team.map(_team)
    pfr = directory.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")
    sc["player_id"] = sc.pfr_player_id.map(pfr.player_id)
    unmapped = sc.player_id.isna()
    sc.loc[unmapped, "player_id"] = "pfr:" + sc.loc[unmapped, "pfr_player_id"].astype(str)
    sc["group"] = sc.position.map(position_group)
    sc["group"] = sc.group.fillna(sc.player_id.map(directory.set_index("player_id").group))
    sc = sc[sc.group.notna()].copy()
    for c in ("offense_snaps", "defense_snaps", "st_snaps"):
        sc[c] = sc[c].fillna(0).astype(int)
    off = sc.group.isin(OFFENSE_GROUPS)
    dfn = sc.group.isin(DEFENSE_GROUPS)
    sc["main_snaps"] = np.where(off, sc.offense_snaps, np.where(dfn, sc.defense_snaps, sc.st_snaps))
    sc = sc[(sc.offense_snaps + sc.defense_snaps + sc.st_snaps) > 0]
    sc = sc.sort_values(["game_id", "team", "group", "main_snaps", "st_snaps"],
                        ascending=[True, True, True, False, False])
    sc["depth"] = sc.groupby(["game_id", "team", "group"]).cumcount()
    sc["name"] = sc.player
    out = sc[["game_id", "season", "week", "team", "player_id", "name", "position", "group", "depth",
              "offense_snaps", "defense_snaps", "st_snaps", "main_snaps"]].reset_index(drop=True)
    # career game count before each game (players who debuted before snap counts exist are veterans)
    out = out.merge(game_dates, on="game_id", how="left")
    out = out.sort_values(["player_id", "game_date", "game_id"])
    out["career_games"] = out.groupby("player_id").cumcount()
    rookie = out.player_id.map(directory.set_index("player_id").rookie_season)
    out.loc[rookie < SNAPS_FIRST, "career_games"] += 100
    return out.drop(columns=["game_date"]).reset_index(drop=True)


def build_personnel(raw: Path, seasons: list[int], schedule: pd.DataFrame) -> pd.DataFrame:
    """Participation -> one row per play: home and away player ids on the field."""
    frames = []
    for s in seasons:
        f = raw / f"participation_{s}.parquet"
        if not f.exists():
            continue
        p = pd.read_parquet(f, columns=["nflverse_game_id", "play_id", "possession_team",
                                        "offense_players", "defense_players"])
        frames.append(p)
    if not frames:
        return pd.DataFrame(columns=["game_id", "play_id", "home_ids", "away_ids"])
    p = pd.concat(frames, ignore_index=True).rename(columns={"nflverse_game_id": "game_id"})
    p = p.merge(schedule[["game_id", "home", "away"]], on="game_id", how="inner")
    p["possession_team"] = p.possession_team.map(_team)
    p = p[p.possession_team.isin(["", None]) == False]  # noqa: E712
    split = lambda x: [i for i in x.split(";") if i] if isinstance(x, str) else []
    offense, defense = p.offense_players.map(split), p.defense_players.map(split)
    home_off = (p.possession_team == p.home).to_numpy()
    known = home_off | (p.possession_team == p.away).to_numpy()
    p["home_ids"] = [o if h else d for o, d, h in zip(offense, defense, home_off)]
    p["away_ids"] = [d if h else o for o, d, h in zip(offense, defense, home_off)]
    p = p[known & (offense.map(len) > 0) & (defense.map(len) > 0)]
    p["play_id"] = p.play_id.astype(int)
    p = p.merge(_ball_ids(raw.parent, seasons), on=["game_id", "play_id"], how="left")
    return p[["game_id", "play_id", "home_ids", "away_ids", "ball_id"]].reset_index(drop=True)


def _ball_ids(pbp_dir: Path, seasons: list[int]) -> pd.DataFrame:
    """Who got the ball on each play: the rusher on runs, the targeted receiver on passes (None for
    sacks, throwaways and other plays). Scrambles are pass plays in nflverse, so they have no target."""
    frames = []
    for s in seasons:
        f = pbp_dir / f"pbp_{s}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f, columns=["game_id", "play_id", "play_type", "rusher_player_id", "receiver_player_id"])
        ball = np.where(d.play_type == "run", d.rusher_player_id,
                        np.where(d.play_type == "pass", d.receiver_player_id, None))
        frames.append(pd.DataFrame({"game_id": d.game_id, "play_id": d.play_id.astype(int), "ball_id": ball}))
    if not frames:
        return pd.DataFrame(columns=["game_id", "play_id", "ball_id"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["game_id", "play_id"])


# ---------------------------------------------------------------------------------------- coaches

COACH_STATS = ["hc_seasons", "team_seasons", "career_games", "career_win_pct"]


def coach_feature_vector(st: dict | None) -> list[float]:
    """Scaled coach record before a game; None (unknown coach) = zeros."""
    if st is None:
        return [0.0] * len(COACH_STATS)
    return [min(st["hc_seasons"], 20) / 10, min(st["team_seasons"], 20) / 10,
            np.log1p(st["career_games"]) / 5, (st["career_win_pct"] - 0.5) * 4]


def coach_record(history: list[tuple], coach: str, team: str, season: int, before_game: int | None = None) -> dict:
    """Coach record from (season, team, wins, losses, ties, games) rows strictly before `season`
    (and, within `season`, the first `before_game` games)."""
    seasons, team_seasons = set(), set()
    w = l = t = 0
    for s, tm, ww, ll, tt in history:
        if s < season:
            seasons.add(s)
            if tm == team:
                team_seasons.add(s)
            w, l, t = w + ww, l + ll, t + tt
    g = w + l + t
    # shrink toward .500 with 8 pseudo-games so a 1-0 start is not a perfect record
    return {"hc_seasons": len(seasons), "team_seasons": len(team_seasons), "career_games": g,
            "career_win_pct": (w + 0.5 * t + 4) / (g + 8)}


def build_coaches(schedule: pd.DataFrame) -> pd.DataFrame:
    """Leak-free head-coach records: each row only counts games before this one."""
    g = schedule[schedule.home_coach.notna() & schedule.away_coach.notna()].sort_values(["gameday", "game_id"])
    first_season = int(g.season.min())
    state: dict[str, dict] = {}
    rows = []
    for r in g.itertuples(index=False):
        for side, coach, team, pf, pa in (("home", r.home_coach, r.home, r.home_score, r.away_score),
                                          ("away", r.away_coach, r.away, r.away_score, r.home_score)):
            st = state.setdefault(coach, {"seasons": set(), "team_seasons": {}, "w": 0.0, "g": 0,
                                          "first": r.season})
            prior_seasons = {s for s in st["seasons"] if s < r.season}
            prior_team = {s for s in st["team_seasons"].get(team, set()) if s < r.season}
            rows.append({
                "game_id": r.game_id, "season": r.season, "side": side, "team": team, "coach": coach,
                "hc_seasons": len(prior_seasons), "team_seasons": len(prior_team),
                "career_games": st["g"], "career_win_pct": (st["w"] + 4) / (st["g"] + 8),
                # coaches already in charge in the first season of data have unknown (longer) careers
                "veteran_unknown": st["first"] == first_season,
            })
            if not (pd.isna(pf) or pd.isna(pa)):
                st["g"] += 1
                st["w"] += 1.0 if pf > pa else 0.5 if pf == pa else 0.0
            st["seasons"].add(r.season)
            st["team_seasons"].setdefault(team, set()).add(r.season)
    return pd.DataFrame(rows)


def schedule_coach_rows(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team) with the coach and the team's result (points margin, NaN if unplayed)."""
    margin = schedule.home_score - schedule.away_score
    h = pd.DataFrame({"game_id": schedule.game_id, "season": schedule.season, "team": schedule.home,
                      "coach": schedule.home_coach, "result": margin})
    a = pd.DataFrame({"game_id": schedule.game_id, "season": schedule.season, "team": schedule.away,
                      "coach": schedule.away_coach, "result": -margin})
    return pd.concat([h, a], ignore_index=True).dropna(subset=["coach"])


# ------------------------------------------------------------------------------- recent usage
USAGE_FEATURES = ["carry_share", "target_share", "snap_share", "ypc", "ypt", "history"]
USAGE_ALPHA = 0.25


def usage_features_vector(st: dict | None) -> list[float]:
    """Scaled recent-usage features from an EWMA state (None = no history: all zeros)."""
    if st is None:
        return [0.0] * len(USAGE_FEATURES)
    ypc = (st["rush_yds"] + 4.3) / (st["carries"] + 1.0)   # shrunk toward league yards per carry
    ypt = (st["rec_yds"] + 7.5) / (st["targets"] + 1.0)    # ... and per target
    return [st["carry_share"], st["target_share"], st["snap_share"], ypc / 10, ypt / 10,
            float(np.log1p(st["n"]) / 3)]


def build_usage(pbp_dir: Path, pg: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """Leak-free recent usage per player-game (features use only the player's earlier games) and each
    player's features after all games. Shares are of the team's designed runs / targets / snaps."""
    seasons = sorted(pg.season.unique())
    frames = []
    for s in seasons:
        f = pbp_dir / f"pbp_{s}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f, columns=["game_id", "posteam", "play_type", "receiver_player_id", "receiving_yards",
                                        "rusher_player_id", "rushing_yards"])
        d["posteam"] = d.posteam.map(_team)
        run = d[(d.play_type == "run") & d.rusher_player_id.notna()]
        pas = d[(d.play_type == "pass") & d.receiver_player_id.notna()]
        team = pd.concat([run.groupby(["game_id", "posteam"]).size().rename("team_carries"),
                          pas.groupby(["game_id", "posteam"]).size().rename("team_targets")], axis=1).fillna(0)
        rp = run.groupby(["game_id", "rusher_player_id"]).agg(carries=("rushing_yards", "size"),
                                                             rush_yds=("rushing_yards", lambda x: x.fillna(0).sum()))
        tp = pas.groupby(["game_id", "receiver_player_id"]).agg(targets=("receiving_yards", "size"),
                                                               rec_yds=("receiving_yards", lambda x: x.fillna(0).sum()))
        rp.index.names = tp.index.names = ["game_id", "player_id"]
        frames.append((team.reset_index().rename(columns={"posteam": "team"}),
                       rp.join(tp, how="outer").fillna(0).reset_index()))
    team = pd.concat([f[0] for f in frames]); prod = pd.concat([f[1] for f in frames])
    d = pg[["game_id", "season", "week", "team", "player_id", "group", "offense_snaps"]].merge(
        prod, on=["game_id", "player_id"], how="left").merge(team, on=["game_id", "team"], how="left").fillna(0)
    tmax = d.groupby(["game_id", "team"]).offense_snaps.transform("max")
    d["snap"] = np.where(tmax > 0, d.offense_snaps / tmax.clip(lower=1), 0.0)
    d["cs"] = np.where(d.team_carries > 0, d.carries / d.team_carries.clip(lower=1), 0.0)
    d["ts"] = np.where(d.team_targets > 0, d.targets / d.team_targets.clip(lower=1), 0.0)
    d = d.sort_values(["season", "week", "game_id"])
    keys = ["carry_share", "target_share", "snap_share", "carries", "rush_yds", "targets", "rec_yds"]
    state: dict[str, dict] = {}
    rows = []
    for gid, pid, cs, ts, sn, c, ry, t, ty in zip(d.game_id, d.player_id, d.cs, d.ts, d.snap, d.carries,
                                                  d.rush_yds, d.targets, d.rec_yds):
        st = state.get(pid)
        rows.append([gid, pid, *usage_features_vector(st)])
        new = dict(zip(keys, (cs, ts, sn, c, ry, t, ty)))
        if st is None:
            state[pid] = new | {"n": 1}
        else:
            for k in keys:
                st[k] = (1 - USAGE_ALPHA) * st[k] + USAGE_ALPHA * new[k]
            st["n"] += 1
    out = pd.DataFrame(rows, columns=["game_id", "player_id", *USAGE_FEATURES])
    out[USAGE_FEATURES] = out[USAGE_FEATURES].astype(np.float32)
    latest = {pid: [round(x, 4) for x in usage_features_vector(st)] for pid, st in state.items()}
    return out, latest


def load_schedule(raw: Path) -> pd.DataFrame:
    s = pd.read_parquet(raw / "schedules.parquet")
    s = s.assign(home=s.home_team.map(_team), away=s.away_team.map(_team))
    return s


def build_players(seasons: list[int], raw_dir: Path, out_dir: Path) -> None:
    raw = raw_dir / "players"
    if not (raw / "players.parquet").exists():
        log.warning("no player data in %s (run: runitback nfl download); skipping lineups", raw)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    directory = _player_directory(raw)
    schedule = load_schedule(raw)
    schedule = schedule[schedule.season.isin(seasons)]
    dates = schedule[["game_id", "gameday"]].rename(columns={"gameday": "game_date"})
    pg = build_player_games(raw, [s for s in seasons if s >= SNAPS_FIRST], directory, dates)
    pg.to_parquet(out_dir / "player_games.parquet")
    usage, latest = build_usage(raw_dir, pg)
    usage.to_parquet(out_dir / "usage.parquet")
    pd.DataFrame({"player_id": list(latest), "features": list(latest.values())}).to_parquet(
        out_dir / "usage_latest.parquet")
    pers = build_personnel(raw, [s for s in seasons if s >= PARTICIPATION_FIRST], schedule)
    pers.to_parquet(out_dir / "personnel.parquet")
    coaches = build_coaches(schedule)
    coaches.to_parquet(out_dir / "coaches.parquet")
    schedule_coach_rows(schedule).to_parquet(out_dir / "coach_games.parquet")
    seen = set(pg.player_id) if len(pg) else set()
    for col in ("home_ids", "away_ids"):
        for ids in pers[col]:
            seen.update(ids)
    d = directory[directory.player_id.isin(seen)]
    extra = pg[~pg.player_id.isin(d.player_id)].drop_duplicates("player_id") if len(pg) else pg
    if len(extra):
        d = pd.concat([d, pd.DataFrame({"player_id": extra.player_id, "name": extra.name,
                                        "position": extra.position, "group": extra.group,
                                        "pfr_id": None, "rookie_season": np.nan})], ignore_index=True)
    d.to_parquet(out_dir / "players.parquet")
    log.info("lineups: %d player-games, %d plays with personnel, %d coach-games, %d players",
             len(pg), len(pers), len(coaches), len(d))
