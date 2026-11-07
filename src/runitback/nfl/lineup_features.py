"""Who plays this week: projected lineups and the wager model's lineup context.

The wager model's history tokens describe a team's recent games as played, so they cannot see
that this week's lineup differs (an injured quarterback, a traded receiver, a new left tackle).
This module gives it the upcoming game's personnel, through the foundation model:

**Projected lineup** (``projected_lineups``): built only from what is known before kickoff, the
same way for training and for upcoming games. Start from the team's previous-game depth chart;
drop players listed Out or Doubtful on that week's injury report, and players not on the team's
active roster that week (injured reserve, cut, traded, suspended); move the next player in each
position group up. Game-day inactives are not used (unknown when lines are set). Depth charts
come from snap counts, so 2013 on.

**Lineup context** (``lineup_table``), per (game, team), from one fold's foundation model:

- ``_lp_*`` (16): the projected lineup itself: its players' embeddings pooled by the foundation's
  lineup encoder (what it adds to every play for a team's personnel, as offense and as defense),
  PCA fit on games before the fold's season.
- ``_lu_*`` (16): how the projected lineup differs from the lineups behind the team's last
  ``HISTORY`` games, in the foundation's lineup encoder space (what it adds to every play for a
  team's personnel, as offense and as defense), PCA fit on games before the fold's season.
- ``_cf_*`` (32): the counterfactual history. The team's last ``HISTORY`` games re-embedded with
  the projected lineup in place of the one that played (plays, situations and results stay real),
  minus the same games embedded as played, in the fold's ``pbp_32`` PCA space: "these results,
  had this week's personnel played them".

Rows whose history lacks lineups (before late 2013) are left out (the wager model's missing flag).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model.lineups import DEPTH, N_SLOTS, SLOT_IDX, LineupData, PlayerVocab, CoachVocab

log = logging.getLogger(__name__)

HISTORY = 14          # the wager model's history window (runitback.wager.nfl: "history")
LP_DIM, LU_DIM, CF_DIM = 16, 16, 32
OUT_STATUSES = {"Out", "Doubtful"}
ROSTER_OK = {"ACT", "INA"}   # INA: game-day inactive, decided at kickoff, so not used
MOVES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


# -- availability ---------------------------------------------------------------------------------

def load_availability(raw_dir: Path, seasons: list[int], download: bool = True
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(out, active), both ``season, week, team, player_id``: players Out or Doubtful on the final
    injury report or listed off the active roster (injured reserve, cut, ...), and each team's
    active roster that week (a player missing from it was traded or released)."""
    import nflreadpy as nfl

    out, active = [], []
    for s in seasons:
        inj_path, ros_path = raw_dir / f"injuries_{s}.parquet", raw_dir / f"rosters_weekly_{s}.parquet"
        if download and (not ros_path.exists() or s == max(seasons)):
            try:
                nfl.load_injuries([s]).to_pandas().to_parquet(inj_path)
            except Exception as e:  # no report published yet
                log.warning("injuries %d: %s", s, e)
            nfl.load_rosters_weekly([s]).to_pandas().to_parquet(ros_path)
        if inj_path.exists():
            inj = pd.read_parquet(inj_path)
            if "date_modified" in inj:   # (2025 on: one row per player and week)
                inj = inj.sort_values("date_modified")
            inj = inj.drop_duplicates(["season", "week", "team", "gsis_id"], keep="last")
            out.append(inj[inj.report_status.isin(OUT_STATUSES)][["season", "week", "team", "gsis_id"]])
        if ros_path.exists():
            ros = pd.read_parquet(ros_path)
            ros = ros[ros.gsis_id.notna()][["season", "week", "team", "gsis_id", "status"]]
            out.append(ros[~ros.status.isin(ROSTER_OK)].drop(columns="status"))
            active.append(ros[ros.status.isin(ROSTER_OK)].drop(columns="status"))
    cols = ["season", "week", "team", "gsis_id"]
    res = []
    for parts in (out, active):
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
        df["team"] = df["team"].replace(MOVES)
        res.append(df.rename(columns={"gsis_id": "player_id"}).drop_duplicates())
    return res[0], res[1]


# -- published depth charts -------------------------------------------------------------------------

EXTRA_POSITIONS = {
    "LDT": "DL", "RDT": "DL", "RUSH": "DL", "LILB": "LB", "RILB": "LB", "LOLB": "LB", "ROLB": "LB",
    "MIKE": "LB", "SAM": "LB", "WILL": "LB", "LCB": "DB", "RCB": "DB", "NCB": "DB", "NDB": "DB",
    "NICKE": "DB", "NKL": "DB",
}


def _group(pos) -> str | None:
    from .data.players import position_group
    if not isinstance(pos, str):
        return None
    return position_group(pos.strip()) or EXTRA_POSITIONS.get(pos.strip().upper())


def load_depth_charts(raw_dir: Path, seasons: list[int], download: bool = True) -> pd.DataFrame:
    """Teams' published depth charts before each game: ``season, week, team, player_id, group,
    level`` (1 = first team). Weekly charts through 2024; from 2025 nflverse keeps dated snapshots,
    and a game uses the team's latest one from before its game day."""
    import nflreadpy as nfl

    parts = []
    for s in seasons:
        path = raw_dir / f"depth_charts_{s}.parquet"
        if download and (not path.exists() or s == max(seasons)):
            try:
                nfl.load_depth_charts([s]).to_pandas().to_parquet(path)
            except Exception as e:
                log.warning("depth charts %d: %s", s, e)
        if not path.exists():
            continue
        d = pd.read_parquet(path)
        if "club_code" in d:     # weekly format
            d = d.assign(team=d.club_code, week=pd.to_numeric(d.week, errors="coerce"),
                         level=pd.to_numeric(d.depth_team, errors="coerce"),
                         group=[_group(p) or _group(q) for p, q in zip(d.depth_position, d.position)])
            d = d[["season", "week", "team", "gsis_id", "group", "level"]]
        else:                    # dated snapshots: the latest before each game day
            sched = _schedule(raw_dir, s, download)
            d = d.assign(day=d.dt.str[:10], level=pd.to_numeric(d.pos_rank, errors="coerce"),
                         group=[_group(p) for p in d.pos_abb])
            snaps = d[["team", "day"]].drop_duplicates()
            rows = []
            for g in sched.itertuples():
                for team in (g.home_team, g.away_team):
                    days = snaps[(snaps.team == team) & (snaps.day < g.gameday)].day
                    if len(days):
                        rows.append((s, g.week, team, days.max()))
            pick = pd.DataFrame(rows, columns=["season", "week", "team", "day"])
            d = pick.merge(d, on=["team", "day"])[["season", "week", "team", "gsis_id", "group", "level"]]
        parts.append(d.dropna(subset=["gsis_id", "group", "week", "level"]))
    if not parts:
        return pd.DataFrame(columns=["season", "week", "team", "player_id", "group", "level"])
    d = pd.concat(parts, ignore_index=True).rename(columns={"gsis_id": "player_id"})
    d["team"] = d["team"].replace(MOVES)
    d["week"] = d["week"].astype(int)
    # a player listed at several spots keeps his best level in each group
    return d.groupby(["season", "week", "team", "player_id", "group"], as_index=False)["level"].min()


def _schedule(raw_dir: Path, season: int, download: bool) -> pd.DataFrame:
    import nflreadpy as nfl

    path = raw_dir / f"schedule_{season}.parquet"
    if download and not path.exists() or (download and season >= pd.Timestamp.now().year - 1):
        nfl.load_schedules([season]).to_pandas().to_parquet(path)
    sc = pd.read_parquet(path)
    sc = sc.assign(home_team=sc.home_team.replace(MOVES), away_team=sc.away_team.replace(MOVES))
    return sc[["season", "week", "gameday", "home_team", "away_team"]]


# -- projected lineups ----------------------------------------------------------------------------

def projected_lineups(pg: pd.DataFrame, games: pd.DataFrame, out: pd.DataFrame, active: pd.DataFrame,
                      charts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Projected depth chart of each team in ``games`` (game_id, season, week, home, away):
    ``game_id, team, group, rank, player_id, career_games, source, dropped``.

    The base is the team's published depth chart for the week (``charts``; first-team players
    first, then by snaps in the team's previous game) or, without one, its previous game's chart
    by snaps. Players Out/Doubtful or off the active roster (``out``, ``active``) are removed and
    the next in the group moves up. ``dropped``: how many of the previous game's slots are not in
    this chart (personnel turnover)."""
    order = pg[["game_id", "team", "season", "week"]].drop_duplicates().sort_values(["team", "season", "week"])
    by_team = {t: d for t, d in order.groupby("team")}
    rows_by_game = {k: d.sort_values(["group", "depth"]) for k, d in pg.groupby(["game_id", "team"])}
    out_set = set(zip(out.season, out.week, out.team, out.player_id))
    roster = {k: set(d.player_id) for k, d in active.groupby(["season", "week", "team"])}
    chart_of = {} if charts is None else {k: d for k, d in charts.groupby(["season", "week", "team"])}
    career = pg.assign(t=pg.season * 100 + pg.week).sort_values("t")
    career_by = {p: (d.t.to_numpy(), d.career_games.to_numpy(float)) for p, d in career.groupby("player_id")}

    def career_before(pid, season, week):
        hit = career_by.get(pid)
        if hit is None:
            return 0.0
        i = np.searchsorted(hit[0], season * 100 + week) - 1      # last game before this one
        return float(hit[1][i]) + 1.0 if i >= 0 else 0.0

    rows = []
    for g in games.itertuples():
        for team in (g.home, g.away):
            o = by_team.get(team)
            before = None if o is None else o[(o.season < g.season) | ((o.season == g.season) & (o.week < g.week))]
            prev = None if before is None or before.empty else before.game_id.iloc[-1]
            prev_rows = rows_by_game[(prev, team)] if prev is not None else pg.iloc[:0]
            snaps = dict(zip(prev_rows.player_id, prev_rows.main_snaps))
            chart = chart_of.get((g.season, g.week, team))
            if chart is not None and len(chart):
                base = chart.assign(snaps=chart.player_id.map(snaps).fillna(0.0))
                base = base.sort_values(["group", "level", "snaps"], ascending=[True, True, False])
                cands, source = list(zip(base.player_id, base.group)), "chart"
            elif prev is not None:
                cands, source = list(zip(prev_rows.player_id, prev_rows.group)), "previous"
            else:
                continue
            on = roster.get((g.season, g.week, team))
            rank: dict[str, int] = {}
            chart_rows = []
            for pid, grp in cands:
                if (g.season, g.week, team, pid) in out_set or (on is not None and pid not in on):
                    continue
                k = rank.get(grp, 0)
                if k < DEPTH.get(grp, 0):
                    chart_rows.append((grp, k, pid, career_before(pid, g.season, g.week)))
                rank[grp] = k + 1
            starters = {r.player_id for r in prev_rows.itertuples() if r.depth < DEPTH.get(r.group, 0)}
            dropped = len(starters - {c[2] for c in chart_rows}) if prev is not None else -1
            rows += [(g.game_id, team, *c, source, dropped) for c in chart_rows]
    cols = ["game_id", "team", "group", "rank", "player_id", "career_games", "source", "dropped"]
    return pd.DataFrame(rows, columns=cols)


def lineup_tokens(proj: pd.DataFrame, pvocab: PlayerVocab) -> dict[tuple[str, str], np.ndarray]:
    """(game_id, team) -> slot tokens of a projected depth chart under one fold's vocabulary."""
    out = {}
    for key, d in proj.groupby(["game_id", "team"]):
        arr = np.zeros(N_SLOTS, dtype=np.int64)
        for r in d.itertuples():
            arr[SLOT_IDX[(r.group, int(r.rank))]] = pvocab.token(r.player_id, r.group, r.career_games)
        out[key] = arr
    return out


# -- lineup context ---------------------------------------------------------------------------------

def _team_history(order: pd.DataFrame, n: int) -> dict[tuple[str, str], list[str]]:
    """(game_id, team) -> the team's previous ``n`` played games (oldest first)."""
    hist = {}
    for team, d in order.groupby("team"):
        ids = d.game_id.tolist()
        for i, gid in enumerate(ids):
            hist[(gid, team)] = ids[max(0, i - n): i]
    return hist


@torch.no_grad()
def _lineup_vectors(model, tokens: np.ndarray) -> np.ndarray:
    """(N, S) depth charts -> (N, 2d): the foundation's lineup vector as offense and as defense."""
    return model.lineup_vectors(torch.from_numpy(tokens)).numpy()


def _pca(x: np.ndarray, fit: np.ndarray, dim: int) -> tuple[np.ndarray, dict]:
    mean, std = x[fit].mean(0), x[fit].std(0) + 1e-6
    _, _, vt = np.linalg.svd((x[fit] - mean) / std, full_matrices=False)
    comp = vt[:dim]
    return ((x - mean) / std) @ comp.T, {"mean": mean, "std": std, "components": comp}


@torch.no_grad()
def lineup_table(ckpt: Path, data_dir: Path, raw_dir: Path, fold: int, out: Path,
                 sets: tuple[str, ...] = ("lu", "cf"), batch_size: int = 32,
                 max_targets: int | None = None) -> Path:
    """Write <out>/lineup.parquet (game_id, team, _lu_*, _cf_*) for one fold's foundation model:
    every played game from 2013 through ``fold`` whose team has ``HISTORY`` earlier games with
    lineups. ``sets``: "lu" (writes _lp_ and _lu_) and/or "cf". Projections are fit on games before ``fold``; ``_cf_*`` uses <out>/pca_32.npz."""
    from .embed import _run_blocks, load_foundation, pool_sides
    from .model.dataset import collate, encode_games

    model, ts, ck = load_foundation(ckpt)
    lm = ck["lineup_meta"]
    ld = LineupData.load(data_dir, [], PlayerVocab(lm["player_vocab"]), CoachVocab(lm["coach_vocab"]))
    games = pd.read_parquet(data_dir / "games.parquet")
    pg = ld.pg
    order = pg[["game_id", "team", "season", "week"]].drop_duplicates().sort_values(["team", "season", "week"])
    targets = order[order.season <= fold][["game_id", "team", "season"]]
    hist = _team_history(order, HISTORY)
    targets = targets[[len(hist[(g, t)]) == HISTORY for g, t in zip(targets.game_id, targets.team)]]
    if max_targets:
        targets = targets.tail(max_targets)
    seasons = sorted(set(games[games.game_id.isin(targets.game_id)].season))
    out_players, active = load_availability(raw_dir, seasons)
    charts = load_depth_charts(raw_dir, seasons)
    proj = projected_lineups(pg, games[games.game_id.isin(set(targets.game_id))], out_players, active, charts)
    proj_tok = lineup_tokens(proj, ld.pvocab)
    targets = targets[[(g, t) in proj_tok for g, t in zip(targets.game_id, targets.team)]].reset_index(drop=True)
    log.info("fold %d: %d team-games with a projected lineup and %d earlier games", fold, len(targets), HISTORY)
    result = targets.copy()
    before = (targets.season < fold).to_numpy()

    if "lu" in sets:
        p = _lineup_vectors(model, np.stack([proj_tok[(g, t)] for g, t in zip(targets.game_id, targets.team)]))
        h_keys = sorted({(h, t) for g, t in zip(targets.game_id, targets.team) for h in hist[(g, t)]})
        hv = dict(zip(h_keys, _lineup_vectors(model, np.stack([ld.lineup(h, t) for h, t in h_keys]))))
        m = np.stack([np.mean([hv[(h, t)] for h in hist[(g, t)]], 0) for g, t in zip(targets.game_id, targets.team)])
        z, _ = _pca(p - m, before, LU_DIM)
        for j in range(LU_DIM):
            result[f"_lu_{j}"] = z[:, j].astype(np.float32)
        z, _ = _pca(p, before, LP_DIM)
        for j in range(LP_DIM):
            result[f"_lp_{j}"] = z[:, j].astype(np.float32)

    if "cf" in sets:
        with np.load(out / "pca_32.npz") as f:
            std, comp = f["std"], f["components"]
        need = sorted({h for g, t in zip(targets.game_id, targets.team) for h in hist[(g, t)]})
        plays = pd.read_parquet(data_dir / "plays.parquet")
        rpath = data_dir / "ratings.parquet"
        ratings = pd.read_parquet(rpath) if (rpath.exists() and model.cfg.n_cont) else None
        gsub = games[games.game_id.isin(need)]
        items = {it["game_id"]: it for it in encode_games(plays[plays.game_id.isin(need)], gsub, ts,
                                                           model.cfg.max_len, ratings, ld)}
        home_of = dict(zip(gsub.game_id, gsub.home))

        def embed(batch):   # [(item, side, tokens or None)] -> (n, 2d) the side's pooled states
            its = []
            for it, side, tok in batch:
                it = dict(it)
                if tok is not None:
                    lu = it["lineup"].copy()
                    lu[side] = tok
                    it["lineup"] = lu
                its.append(it)
            b = collate(its)
            v = pool_sides(_run_blocks(model, b["feats"]), b["feats"]["is_home"], b["valid"])
            return torch.stack([v[k, side] for k, (_, side, _) in enumerate(batch)]).numpy()

        actual = {}
        keys = sorted({(h, t) for g, t in zip(targets.game_id, targets.team) for h in hist[(g, t)]})
        for i in range(0, len(keys), batch_size):
            chunk = keys[i: i + batch_size]
            vecs = embed([(items[h], 0 if home_of[h] == t else 1, None) for h, t in chunk])
            actual.update(zip(chunk, vecs))
        cf = np.zeros((len(targets), comp.shape[1]))
        jobs = [(k, h, t) for k, (g, t) in enumerate(zip(targets.game_id, targets.team)) for h in hist[(g, t)]]
        for i in range(0, len(jobs), batch_size):
            chunk = jobs[i: i + batch_size]
            vecs = embed([(items[h], 0 if home_of[h] == t else 1, proj_tok[(targets.game_id[k], t)])
                          for k, h, t in chunk])
            for (k, h, t), v in zip(chunk, vecs):
                cf[k] += (v - actual[(h, t)]) / HISTORY
            if (i // batch_size) % 200 == 0:
                log.info("fold %d: counterfactual history %d/%d", fold, i + len(chunk), len(jobs))
        z = (cf / std) @ comp.T
        for j in range(CF_DIM):
            result[f"_cf_{j}"] = z[:, j].astype(np.float32)

    # personnel-change flags, for evaluating on the games where the lineup moved (not model inputs):
    # projected QB1 differs from the previous game's (known before kickoff), actual QB1 differs
    qb1 = pg[(pg.group == "QB") & (pg.depth == 0)].set_index(["game_id", "team"]).player_id
    pqb = proj[(proj.group == "QB") & (proj["rank"] == 0)].set_index(["game_id", "team"]).player_id
    prev = [hist[(g, t)][-1] for g, t in zip(result.game_id, result.team)]
    prev_qb = qb1.reindex(list(zip(prev, result.team))).to_numpy()
    result["_qb_change"] = (pqb.reindex(list(zip(result.game_id, result.team))).to_numpy() != prev_qb).astype(np.int8)
    result["_qb_change_actual"] = (qb1.reindex(list(zip(result.game_id, result.team))).to_numpy() != prev_qb
                                   ).astype(np.int8)
    result = result.join(proj.groupby(["game_id", "team"]).dropped.first().rename("_dropped"), on=["game_id", "team"])
    path = out / "lineup.parquet"
    result.to_parquet(path)
    log.info("fold %d: %d rows -> %s", fold, len(result), path)
    return path
