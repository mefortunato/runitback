"""NHL projected lineups: each game's lines and goalie, projected from the team's previous game.

**Played lineups** (`played_lineups`, from the events table of H1). For every team-game: the
skaters who dressed (everyone who was on the ice), each one's time on ice (TOI: the seconds from
each event to the next, credited to the players on ice for it; the shootout excluded) and 5-on-5
TOI, the starting goalie (in net at the first event of the game that has one) and every goalie who
played. Lines are rebuilt from 5-on-5 shared ice time:
- **Forwards** (`group == "F"`): the remaining forward with the most 5-on-5 TOI anchors a line,
  with the two remaining forwards who shared the most 5-on-5 time with him. Repeat. Lines are then
  numbered by their members' mean 5-on-5 TOI (line 1 the most). With 12 forwards that is four
  lines of three; an odd count leaves a short last line (11 forwards: 3, 3, 3, 2).
- **Defense** (`"D"`): the same with pairs (three with six defensemen).
- `slot` ranks a group's players by all-situation TOI (1 = most).

**Projection** (`projected_lineups`). A team-game's projected skaters are its **previous game's**
(any game type; a season opener's previous game is the last one of the season before), with that
game's lines, slots and TOI. Nothing from the game itself is used for the skaters. There is no
history of injury reports or announced lines, so a player hurt in the previous game or traded since
stays in, and his replacement is missing.

**The starting goalie is the largest swing, and no history of announced starters exists**, so it
is projected two ways, both on the goalie rows:
- **(a) `goalie_prev`**: the previous game's starter. Leak-free: known before the game.
- **(b) `goalie_actual`**: the game's actual starter, treated as **known at puck drop**. This is a
  leak, documented here and measured by `evaluate`:
  - In practice teams confirm the starter on game day (morning skate, or when he leads the team
    out for warm-ups), and late changes are known by puck drop. So a closing line has usually
    priced him in, and so can a prediction made at puck drop.
  - It leaks when he was not known at the time the prediction or the line it is compared with was
    made (an opening line, or a model refit on the morning of the game), and in the few games
    where the announced starter is pulled in warm-ups or the first minutes. Then (b) knows a goalie
    the market did not, and backtests with it will look better than live use can be.
  - Only the starter's identity comes from the game (who was in net at its first event). Not his
    TOI, whether he was pulled, or any result.
  `evaluate` reports how often (a) and (b) differ, overall and on the second night of a
  back-to-back, where teams usually start the backup, next to a leak-free reference rule
  (`usual_starter`: most starts in the last 10 games, the other goalie on a back-to-back) that is
  not a projection mode.

Outputs in the processed directory:
- `projected_lineups.parquet`, one row per (game, team, player): `game_id, season, game_type,
  game_date, team, side, player_id, group (F / D / G), line, slot, prev_toi, prev_ev_toi (minutes in
  the previous game), prev_game_id, rest_days, goalie_prev, goalie_actual` (both False on skater
  rows). Goalie rows are the union of (a) and (b): one row when they agree, two when they don't.
  `lineup(proj, goalie="prev" | "actual")` picks one.
- `played_lineups.parquet`: the same per team-game as played (`toi`, `ev_toi`, `line`, `slot`,
  `started` on goalie rows); the projection's source and the evaluation's truth.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .events import AWAY_SKATERS, HOME_SKATERS

log = logging.getLogger(__name__)

LINE_SIZE = {"F": 3, "D": 2}
EVEN = "5v5"
GOALIE_MODES = ("prev", "actual")
BACK_TO_BACK = 1   # rest_days of the second night of a back-to-back
USUAL_GAMES = 10   # evaluate's reference starter rule: most starts in the team's last USUAL_GAMES

PLAYED_COLUMNS = ["game_id", "season", "game_type", "game_date", "team", "side", "player_id", "group",
                  "toi", "ev_toi", "line", "slot", "started"]
COLUMNS = ["game_id", "season", "game_type", "game_date", "team", "side", "player_id", "group", "line", "slot",
           "prev_toi", "prev_ev_toi", "prev_game_id", "rest_days", "goalie_prev", "goalie_actual"]


# --------------------------------------------------------------------------- played lineups

def _durations(ev: pd.DataFrame) -> np.ndarray:
    """Seconds from each event to the next one of its game (0 for the last one and the shootout)."""
    gs = ev["game_seconds"].to_numpy(float)
    nxt = np.r_[gs[1:], np.nan]
    same = np.r_[ev["game_id"].to_numpy()[1:] == ev["game_id"].to_numpy()[:-1], False]
    dt = np.where(same, nxt - gs, 0.0)
    dt[ev["period_type"].eq("SHOOTOUT").to_numpy()] = 0.0
    return np.clip(np.nan_to_num(dt), 0, None)


def _long(ev: pd.DataFrame, dt: np.ndarray, cols: list[str], side: str, even: np.ndarray) -> pd.DataFrame:
    ids = ev[cols].to_numpy(dtype=float, na_value=np.nan)
    k = ids.shape[1]
    out = pd.DataFrame({"game_id": np.repeat(ev["game_id"].to_numpy(), k), "player_id": ids.ravel(),
                        "dt": np.repeat(dt, k), "ev": np.repeat(even, k)})
    out = out[out.player_id.notna()]
    out["side"] = side
    return out


def _pairs(ev: pd.DataFrame, dt: np.ndarray, cols: list[str], side: str) -> pd.DataFrame:
    """5-on-5 seconds shared by each pair of skaters of one side: game_id, side, a, b, shared (a < b)."""
    ids = ev[cols].to_numpy(dtype=float, na_value=np.nan)
    frames = []
    for i in range(ids.shape[1]):
        for j in range(i + 1, ids.shape[1]):
            ok = ~np.isnan(ids[:, i]) & ~np.isnan(ids[:, j])
            frames.append(pd.DataFrame({"game_id": ev["game_id"].to_numpy()[ok], "a": ids[ok, i], "b": ids[ok, j],
                                        "shared": dt[ok]}))
    p = pd.concat(frames, ignore_index=True)
    p = p.groupby(["game_id", "a", "b"], sort=False, as_index=False).shared.sum()
    p["side"] = side
    return p


def build_lines(players: list[int], ev_toi: dict[int, float], shared: dict[tuple[int, int], float],
                size: int) -> dict[int, int]:
    """Greedy lines (see the module docstring): player -> line number, 1 = most 5-on-5 TOI."""
    left = sorted(players, key=lambda p: (-ev_toi.get(p, 0.0), p))
    lines = []
    while left:
        anchor, rest = left[0], left[1:]
        mates = sorted(rest, key=lambda p: (-shared.get((min(anchor, p), max(anchor, p)), 0.0),
                                            -ev_toi.get(p, 0.0), p))[:size - 1]
        lines.append([anchor] + mates)
        left = [p for p in rest if p not in mates]
    lines.sort(key=lambda m: -np.mean([ev_toi.get(p, 0.0) for p in m]))
    return {p: i + 1 for i, m in enumerate(lines) for p in m}


def played_lineups(ev: pd.DataFrame, games: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Each team-game's lineup as played (see the module docstring). `ev`: events of one or more
    seasons; `games`: games.parquet (game_id, home, away, game_date, game_type, season); `groups`:
    player_id -> F / D / G (players.parquet)."""
    ev = ev.sort_values(["game_id", "event_idx"], kind="stable").reset_index(drop=True)
    dt = _durations(ev)
    even = ev["strength_state"].eq(EVEN).fillna(False).to_numpy() & ev["home_goalie"].notna().to_numpy() \
        & ev["away_goalie"].notna().to_numpy()
    sk = pd.concat([_long(ev, dt, HOME_SKATERS, "home", even), _long(ev, dt, AWAY_SKATERS, "away", even)])
    sk["ev_dt"] = np.where(sk.ev, sk.dt, 0.0)
    sk = sk.groupby(["game_id", "side", "player_id"], sort=False, as_index=False)[["dt", "ev_dt"]].sum()
    gl = pd.concat([_long(ev, dt, [f"{s}_goalie"], s, even) for s in ("home", "away")])
    gl = gl.groupby(["game_id", "side", "player_id"], sort=False, as_index=False).dt.sum().assign(ev_dt=0.0)
    first = {}
    for s in ("home", "away"):
        g = ev[ev[f"{s}_goalie"].notna()].groupby("game_id", sort=False)[f"{s}_goalie"].first()
        first.update({(gid, s): int(p) for gid, p in g.items()})
    sk["player_id"] = sk.player_id.astype("int64")
    gl["player_id"] = gl.player_id.astype("int64")
    sk = sk[~sk.set_index(["game_id", "side", "player_id"]).index.isin(
        gl.set_index(["game_id", "side", "player_id"]).index)]   # a goalie listed as a skater
    sk["group"] = sk.player_id.map(groups).fillna("F").replace({"G": "F"})
    gl["group"] = "G"
    gl["started"] = [first.get((g, s)) == p for g, s, p in zip(gl.game_id, gl.side, gl.player_id)]
    sk["started"] = False

    ev5 = ev[even]
    dt5 = dt[even]
    pairs = pd.concat([_pairs(ev5, dt5, HOME_SKATERS, "home"), _pairs(ev5, dt5, AWAY_SKATERS, "away")])
    shared_of = {k: dict(zip(zip(d.a.astype(int), d.b.astype(int)), d.shared))
                 for k, d in pairs.groupby(["game_id", "side"], sort=False)}
    line = np.zeros(len(sk), dtype=np.int16)
    sk = sk.reset_index(drop=True)
    for (gid, side, grp), d in sk.groupby(["game_id", "side", "group"], sort=False):
        lines = build_lines(d.player_id.tolist(), dict(zip(d.player_id, d.ev_dt)),
                            shared_of.get((gid, side), {}), LINE_SIZE[grp])
        line[d.index] = [lines[p] for p in d.player_id]
    sk["line"] = line
    gl["line"] = np.where(gl.started, 1, 2).astype(np.int16)
    out = pd.concat([sk, gl], ignore_index=True)
    out["slot"] = out.sort_values("dt", ascending=False, kind="stable") \
        .groupby(["game_id", "side", "group"]).cumcount().reindex(out.index).astype(np.int16) + 1
    out["toi"] = (out.dt / 60).round(2)
    out["ev_toi"] = (out.ev_dt / 60).round(2)
    g = games.set_index("game_id")
    out["team"] = np.where(out.side.eq("home"), out.game_id.map(g["home"]), out.game_id.map(g["away"]))
    for c in ("season", "game_type", "game_date"):
        out[c] = out.game_id.map(g[c])
    out = out[out.team.notna()]
    return out.sort_values(["game_id", "side", "group", "line", "slot"], kind="stable")[PLAYED_COLUMNS] \
        .reset_index(drop=True)


# ------------------------------------------------------------------------------- projection

def team_games(played: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game in date order, with the team's previous game and the rest days."""
    tg = played[["game_id", "season", "game_type", "game_date", "team", "side"]].drop_duplicates(["game_id", "side"])
    tg = tg.assign(date=pd.to_datetime(tg.game_date)).sort_values(["team", "date", "game_id"], kind="stable")
    tg["prev_game_id"] = tg.groupby("team").game_id.shift(1)
    tg["rest_days"] = (tg.date - tg.groupby("team").date.shift(1)).dt.days
    return tg.drop(columns="date").reset_index(drop=True)


def projected_lineups(played: pd.DataFrame) -> pd.DataFrame:
    """Every team-game's projected lineup from its previous game (see the module docstring). Team-games
    without a previous game in `played` (each team's first) are left out."""
    tg = team_games(played).dropna(subset=["prev_game_id"])
    tg["prev_game_id"] = tg.prev_game_id.astype("int64")
    key = ["game_id", "team"]
    prev = played.drop(columns=["season", "game_type", "game_date", "side"]).rename(
        columns={"game_id": "prev_game_id", "toi": "prev_toi", "ev_toi": "prev_ev_toi"})
    prev = prev[prev.group.ne("G") | prev.started]
    p = tg.merge(prev, on=["prev_game_id", "team"])
    p["goalie_prev"] = p.group.eq("G")
    actual = tg[key].merge(played[played.group.eq("G") & played.started][key + ["player_id"]], on=key)
    p = p.merge(actual.assign(_act=True), on=key + ["player_id"], how="left")
    p["goalie_actual"] = p.group.eq("G") & p._act.fillna(False).astype(bool)
    # the actual starter where he is not the previous starter, with his TOI in his last game before this one
    new = actual.merge(p.loc[p.goalie_actual, key + ["player_id"]], on=key + ["player_id"], how="left",
                       indicator=True)
    new = new[new._merge.eq("left_only")].drop(columns="_merge")
    new = new.merge(tg[key + ["season", "game_type", "game_date", "side", "prev_game_id", "rest_days"]], on=key)
    new["group"], new["line"], new["slot"] = "G", 1, 1
    new["goalie_prev"], new["goalie_actual"] = False, True
    prior = played[played.group.eq("G")][["player_id", "game_date", "toi", "ev_toi"]]
    prior = prior.assign(date=pd.to_datetime(prior.game_date).astype("datetime64[ns]")).drop(columns="game_date").sort_values("date") \
        .rename(columns={"toi": "prev_toi", "ev_toi": "prev_ev_toi"})
    n = new.assign(_i=np.arange(len(new)), date=pd.to_datetime(new.game_date).astype("datetime64[ns]")).sort_values("date")
    n = pd.merge_asof(n, prior, on="date", by="player_id", allow_exact_matches=False).sort_values("_i")
    new = n.drop(columns=["_i", "date"])
    p = pd.concat([p.drop(columns=["_act", "started"]), new], ignore_index=True)
    order = {"F": 0, "D": 1, "G": 2}
    p = p.assign(_g=p.group.map(order)).sort_values(["game_date", "game_id", "side", "_g", "line", "slot"],
                                                    kind="stable")
    for c in ("line", "slot"):
        p[c] = p[c].astype(np.int16)
    return p[COLUMNS].reset_index(drop=True)


def lineup(proj: pd.DataFrame, goalie: str = "prev") -> pd.DataFrame:
    """The projected skaters plus one goalie per team-game: (a) `goalie="prev"`, the previous game's
    starter, or (b) `"actual"`, the actual starter (known at puck drop: see the module docstring).
    Adds `known_at_drop`: True on rows that use information from the game itself."""
    if goalie not in GOALIE_MODES:
        raise ValueError(f"goalie must be one of {GOALIE_MODES}")
    col = f"goalie_{goalie}"
    out = proj[proj.group.ne("G") | proj[col]].copy()
    out["known_at_drop"] = goalie == "actual"
    out.loc[out.group.ne("G"), "known_at_drop"] = False
    return out.reset_index(drop=True)


# ------------------------------------------------------------------------------- evaluation

def usual_starter(played: pd.DataFrame, games: int = USUAL_GAMES) -> pd.DataFrame:
    """A leak-free reference rule for the starter, for `evaluate` only (not a projection mode): the
    goalie with the most starts in the team's last `games` games (ties: the most recent), except on
    the second night of a back-to-back, where it is the most frequent of them who did not start
    the night before (the previous starter if there is none). game_id, team -> `usual_g`."""
    st = played[played.group.eq("G") & played.started][["game_id", "game_date", "team", "player_id"]]
    st = st.assign(date=pd.to_datetime(st.game_date)).sort_values(["team", "date", "game_id"], kind="stable")
    rows = []
    for team, d in st.groupby("team", sort=False):
        ids, dates, gids = d.player_id.tolist(), d.date.tolist(), d.game_id.tolist()
        for i in range(1, len(ids)):
            recent = ids[max(0, i - games):i]
            count = {}
            for k, g in enumerate(recent):
                count[g] = (count.get(g, (0, 0))[0] + 1, k)
            ranked = sorted(count, key=lambda g: count[g], reverse=True)
            if (dates[i] - dates[i - 1]).days == BACK_TO_BACK:
                ranked = [g for g in ranked if g != ids[i - 1]] or [ids[i - 1]]
            rows.append((gids[i], team, ranked[0]))
    return pd.DataFrame(rows, columns=["game_id", "team", "usual_g"])


def evaluate(proj: pd.DataFrame, played: pd.DataFrame) -> pd.DataFrame:
    """Per season, the projection against the game as played:

    - `skaters`: share of the dressed skaters who were projected; `all18`: share of team-games with
      every dressed skater projected and no one else; `changes`: mean dressed skaters not projected
    - `f_line`, `d_pair`: of the forwards / defensemen projected and dressed, the share on the same
      line / pair as projected; `toi_mae`: mean absolute error of the previous game's TOI against
      this one's, over every skater projected or dressed (minutes, 0 if he did not play)
    - `goalie_prev`: share of team-games where (a) the previous starter started; `goalie_prev_b2b` /
      `goalie_prev_rested`: on the second night of a back-to-back / otherwise; `b2b_share`: share
      of team-games on a back-to-back. (b) is right by construction: `goalie_changed` (1 -
      `goalie_prev`) is the share of team-games where it knows something (a) does not.
    - `goalie_usual`: the same for the leak-free reference rule `usual_starter` (not a projection
      mode), to show how much of (b)'s edge a schedule-aware rule recovers.
    - `opener_skaters`: `skaters` on each team's first game of the season (the previous game is
      the last of the season before)."""
    key = ["game_id", "team", "player_id"]
    sk_p = proj[proj.group.ne("G")][key + ["season", "line", "prev_toi", "rest_days"]]
    sk_a = played[played.group.ne("G")][key + ["line", "toi"]].rename(columns={"line": "act_line"})
    tgs = proj[["game_id", "team", "season", "rest_days", "prev_game_id"]].drop_duplicates(["game_id", "team"])
    sk_a = sk_a.merge(tgs[["game_id", "team"]], on=["game_id", "team"])
    m = sk_p.drop(columns=["season", "rest_days"]).merge(sk_a, on=key, how="outer", indicator=True)
    m = m.merge(tgs, on=["game_id", "team"])
    m["proj"], m["act"] = m._merge.ne("right_only"), m._merge.ne("left_only")
    m["prev_toi"], m["toi"] = m.prev_toi.fillna(0.0), m.toi.fillna(0.0)
    m["same_line"] = m.proj & m.act & m.line.eq(m.act_line)
    grp = played[["player_id", "group"]].drop_duplicates("player_id").set_index("player_id").group
    m["group"] = m.player_id.map(grp)
    prev_season = played[["game_id", "season"]].drop_duplicates("game_id").set_index("game_id").season
    m["opener"] = m.prev_game_id.map(prev_season).ne(m.season)
    a = m[m.act]
    per_tg = a.groupby(["season", "game_id", "team"]).agg(n_act=("act", "size"), hit=("proj", "sum"),
                                                         opener=("opener", "first"))
    extra = m[m.proj & ~m.act].groupby(["game_id", "team"]).size()
    per_tg["extra"] = [extra.get((g, t), 0) for _, g, t in per_tg.index]
    per_tg["exact"] = per_tg.hit.eq(per_tg.n_act) & per_tg.extra.eq(0)

    gp = proj[proj.goalie_prev][["game_id", "team", "player_id"]].rename(columns={"player_id": "prev_g"})
    ga = proj[proj.goalie_actual][["game_id", "team", "player_id"]].rename(columns={"player_id": "act_g"})
    g = tgs.merge(gp, on=["game_id", "team"]).merge(ga, on=["game_id", "team"])
    g["right"] = g.prev_g.eq(g.act_g)
    g["b2b"] = g.rest_days.eq(BACK_TO_BACK)
    g = g.merge(usual_starter(played), on=["game_id", "team"], how="left")
    g["usual_right"] = g.usual_g.eq(g.act_g)
    rows = {}
    for s, t in per_tg.groupby(level="season"):
        ms, gs = m[m.season == s], g[g.season == s]
        both = ms[ms.proj & ms.act]
        f, d = both[both.group.eq("F")], both[both.group.eq("D")]
        op = t[t.opener]
        rows[int(s)] = dict(
            team_games=len(t), skaters=t.hit.sum() / t.n_act.sum(), all18=float(t.exact.mean()),
            changes=float((t.n_act - t.hit).mean()),
            f_line=float(f.same_line.mean()), d_pair=float(d.same_line.mean()),
            toi_mae=float((ms.prev_toi - ms.toi).abs().mean()),
            goalie_prev=float(gs.right.mean()), goalie_prev_b2b=float(gs[gs.b2b].right.mean()),
            goalie_prev_rested=float(gs[~gs.b2b].right.mean()), b2b_share=float(gs.b2b.mean()),
            goalie_changed=float(1 - gs.right.mean()), goalie_usual=float(gs.usual_right.mean()),
            opener_skaters=float(op.hit.sum() / op.n_act.sum()) if len(op) else np.nan)
    return pd.DataFrame(rows).T.rename_axis("season")


# ------------------------------------------------------------------------------------ build

def build_lineups(processed: Path, seasons=None) -> dict:
    """Write `played_lineups.parquet` and `projected_lineups.parquet` from the events tables in
    `processed` (every events_<season>.parquet, or `seasons`); returns `evaluate`'s summary."""
    games = pd.read_parquet(processed / "games.parquet")
    groups = pd.read_parquet(processed / "players.parquet").set_index("player_id").group
    paths = sorted(processed.glob("events_*.parquet"))
    if seasons is not None:
        want = {int(s) for s in seasons}
        paths = [p for p in paths if int(p.stem.split("_")[1]) in want]
    cols = ["game_id", "event_idx", "game_seconds", "period_type", "strength_state"] + HOME_SKATERS \
        + AWAY_SKATERS + ["home_goalie", "away_goalie"]
    frames = []
    for p in paths:
        frames.append(played_lineups(pd.read_parquet(p, columns=cols), games, groups))
        log.info("%s: %d team-games", p.name, frames[-1][["game_id", "side"]].drop_duplicates().shape[0])
    played = pd.concat(frames, ignore_index=True)
    played.to_parquet(processed / "played_lineups.parquet", index=False)
    proj = projected_lineups(played)
    proj.to_parquet(processed / "projected_lineups.parquet", index=False)
    ev = evaluate(proj, played)
    log.info("projected lineups: %d rows\n%s", len(proj), ev.round(4).to_string())
    return {int(s): {k: (round(float(v), 4) if pd.notna(v) else None) for k, v in r.items()}
            for s, r in ev.iterrows()}


__all__ = [
    "GOALIE_MODES",
    "build_lines",
    "build_lineups",
    "evaluate",
    "lineup",
    "played_lineups",
    "projected_lineups",
    "team_games",
    "usual_starter",
]
