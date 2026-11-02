"""Player and coach tokens, team depth charts, per-play personnel, and the personnel sampler.

Token layout for players (shared by game lineups and per-play personnel):
    0                 padding / empty slot
    1 .. G            NEW_<group>    a player in his first NEW_GAMES career games, or never seen
    G+1 .. 2G         OTHER_<group>  a veteran without his own embedding (too few training games)
    2G+1 ..           players with their own embedding
Coaches: 0 unknown, 1 NEW_COACH, 2 OTHER_COACH, 3.. coaches.

A team's lineup is its depth chart flattened into SLOTS: (group, rank) pairs, e.g. QB1, QB2, WR1..WR6.
Per-play personnel are the 11 players of the possession team then the 11 of the other team.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.players import (
    COACH_STATS, DEPTH, GROUP_IDX, GROUPS, coach_feature_vector, coach_record, schedule_coach_rows,  # noqa: F401
)

NEW_GAMES = 3          # a player's first NEW_GAMES career games use NEW_<group>
MIN_PLAYER_GAMES = 4   # games (after the NEW window) needed for a player's own embedding
NEW_COACH_GAMES = 4    # a head coach's first games use NEW_COACH
MIN_COACH_GAMES = 8
N_GROUPS = len(GROUPS)
SLOTS = [(g, r) for g in GROUPS for r in range(DEPTH[g])]
SLOT_IDX = {s: i for i, s in enumerate(SLOTS)}
SLOT_GROUP = np.array([GROUP_IDX[g] for g, _ in SLOTS])
N_SLOTS = len(SLOTS)
SIDE = 11              # players per side in per-play personnel
N_PERS = 2 * SIDE
N_COACH_CONT = 2 * len(COACH_STATS)
NEW_LABEL = "NEW"      # depth-chart entry for a new / unknown player


def new_token(group: str | int) -> int:
    return 1 + (GROUP_IDX[group] if isinstance(group, str) else int(group))


def other_token(group: str | int) -> int:
    return 1 + N_GROUPS + (GROUP_IDX[group] if isinstance(group, str) else int(group))


@dataclass
class PlayerVocab:
    ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.idx = {p: i + 1 + 2 * N_GROUPS for i, p in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids) + 1 + 2 * N_GROUPS

    def token(self, pid: str | None, group: str | int, career_games: float | None) -> int:
        """career_games = games before this one (None/NaN = unknown, treated as never seen)."""
        if pid is None or pid == NEW_LABEL:
            return new_token(group)
        tok = self.idx.get(pid)
        if career_games is None or (isinstance(career_games, float) and math.isnan(career_games)):
            return tok if tok else new_token(group)
        if career_games < NEW_GAMES:
            return new_token(group)
        return tok if tok else other_token(group)


@dataclass
class CoachVocab:
    names: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.idx = {c: i + 3 for i, c in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names) + 3

    def token(self, coach: str | None, career_games: int, veteran: bool = False) -> int:
        if not coach:
            return 0
        if career_games < NEW_COACH_GAMES and not veteran:
            return 1
        return self.idx.get(coach, 2)


def build_player_vocab(player_games: pd.DataFrame, game_ids) -> PlayerVocab:
    pg = player_games[player_games.game_id.isin(set(game_ids)) & (player_games.career_games >= NEW_GAMES)]
    n = pg.groupby("player_id").size()
    return PlayerVocab(sorted(n[n >= MIN_PLAYER_GAMES].index))


def build_coach_vocab(coaches: pd.DataFrame, game_ids) -> CoachVocab:
    c = coaches[coaches.game_id.isin(set(game_ids))]
    n = c.groupby("coach").size()
    return CoachVocab(sorted(n[n >= MIN_COACH_GAMES].index))


# ------------------------------------------------------------------------------- training tables

class LineupData:
    """Joins the processed player tables to games/plays for training and evaluation."""

    def __init__(self, player_games: pd.DataFrame, personnel: pd.DataFrame, coaches: pd.DataFrame,
                 players: pd.DataFrame, pvocab: PlayerVocab, cvocab: CoachVocab):
        self.pg, self.personnel, self.coaches, self.players = player_games, personnel, coaches, players
        self.pvocab, self.cvocab = pvocab, cvocab
        pg = player_games[player_games.depth < player_games.group.map(DEPTH)]
        self.pg_tokens = pg.assign(
            slot=[SLOT_IDX[(g, int(d))] for g, d in zip(pg.group, pg.depth)],
            tok=[pvocab.token(p, g, c) for p, g, c in zip(pg.player_id, pg.group, pg.career_games)])
        # per (game, team, player): token, group, slot (for personnel tokens and signatures)
        allpg = player_games.assign(
            tok=[pvocab.token(p, g, c) for p, g, c in zip(player_games.player_id, player_games.group,
                                                           player_games.career_games)],
            slot=[SLOT_IDX.get((g, int(d)), -1) for g, d in zip(player_games.group, player_games.depth)])
        self.player_game = {(g, p): (t, GROUP_IDX[gr], s) for g, p, t, gr, s in
                            zip(allpg.game_id, allpg.player_id, allpg.tok, allpg.group, allpg.slot)}
        self.dir_group = dict(zip(players.player_id, players.group))
        self._lineups: dict[tuple[str, str], np.ndarray] = {}
        self._carried: dict[tuple[str, str], np.ndarray] = {}  # same players, one career game later
        tok_next = [pvocab.token(p, g, c + 1) for p, g, c in zip(pg.player_id, pg.group, pg.career_games)]
        self.pg_tokens["tok_next"] = tok_next
        for (gid, team), d in self.pg_tokens.groupby(["game_id", "team"]):
            arr = np.zeros(N_SLOTS, dtype=np.int32)
            arr[d.slot.to_numpy()] = d.tok.to_numpy()
            self._lineups[(gid, team)] = arr
            nxt = np.zeros(N_SLOTS, dtype=np.int32)
            nxt[d.slot.to_numpy()] = d.tok_next.to_numpy()
            self._carried[(gid, team)] = nxt
        order = self.pg_tokens[["game_id", "team", "season", "week"]].drop_duplicates(["game_id", "team"])
        order = order.sort_values(["team", "season", "week"])
        prev = order.groupby("team").game_id.shift()
        self._prev_game = {(g, t): p for g, t, p in zip(order.game_id, order.team, prev) if isinstance(p, str)}
        self.coach_rows = {(r.game_id, r.side): r for r in coaches.itertuples(index=False)}

    @classmethod
    def load(cls, data_dir, train_game_ids, pvocab: PlayerVocab | None = None,
             cvocab: CoachVocab | None = None) -> "LineupData | None":
        from pathlib import Path
        d = Path(data_dir)
        if not (d / "player_games.parquet").exists():
            return None
        pg = pd.read_parquet(d / "player_games.parquet")
        pers = pd.read_parquet(d / "personnel.parquet")
        coaches = pd.read_parquet(d / "coaches.parquet")
        players = pd.read_parquet(d / "players.parquet")
        pvocab = pvocab or build_player_vocab(pg, train_game_ids)
        cvocab = cvocab or build_coach_vocab(coaches, train_game_ids)
        return cls(pg, pers, coaches, players, pvocab, cvocab)

    def lineup(self, game_id: str, team: str) -> np.ndarray:
        """Slot tokens of a team's depth chart in a game (all zeros = unknown lineup)."""
        return self._lineups.get((game_id, team), np.zeros(N_SLOTS, dtype=np.int32))

    def previous_lineup(self, game_id: str, team: str, season: int | None = None, week: int | None = None
                        ) -> np.ndarray:
        """The team's lineup in its most recent earlier game (what the UI defaults to)."""
        prev = self._prev_game.get((game_id, team))
        if prev is None and season is not None and (game_id, team) not in self._lineups:
            d = self.pg_tokens  # a game without its own lineup (e.g. not played yet): search by date
            d = d[(d.team == team) & ((d.season < season) | ((d.season == season) & (d.week < week)))]
            prev = d.sort_values(["season", "week"]).game_id.iloc[-1] if len(d) else None
        if prev is None:
            return np.zeros(N_SLOTS, dtype=np.int32)
        return self._carried[(prev, team)]

    def coach(self, game_id: str, side: str) -> tuple[int, list[float]]:
        r = self.coach_rows.get((game_id, side))
        if r is None:
            return 0, coach_feature_vector(None)
        st = {k: getattr(r, k) for k in COACH_STATS}
        return self.cvocab.token(r.coach, r.career_games, r.veteran_unknown), coach_feature_vector(st)

    def personnel_tokens(self, game_id: str, ids: list[str]) -> tuple[list[int], list[int], list[int]]:
        """Tokens, groups and depth-chart slots (-1 = off the depth chart) for one side's players."""
        toks, grps, slots = [], [], []
        for pid in ids[:SIDE]:
            hit = self.player_game.get((game_id, pid))
            if hit is not None:
                t, g, s = hit
            else:
                gname = self.dir_group.get(pid) or "OL"
                g, s = GROUP_IDX[gname], -1
                t = self.pvocab.idx.get(pid) or other_token(g)
            toks.append(t), grps.append(g + 1), slots.append(s)
        return toks, grps, slots


# ------------------------------------------------------------------------------ personnel sampler

def situation_key(play_type: str, down: int, ytg: int, yl: int) -> str:
    if play_type in ("kickoff", "punt", "field_goal", "extra_point", "two_point", "timeout",
                     "qb_kneel", "qb_spike"):
        return play_type
    dist = "s" if ytg <= 2 else "m" if ytg <= 6 else "l"
    goal = "g" if yl <= 5 else "r" if yl <= 20 else "f"
    return f"{play_type}:{min(down, 4)}{dist}{goal}"


class PersonnelSampler:
    """Which depth-chart slots take the field, given the play type and situation.

    Learned from participation joined to per-game depth charts. A signature is the sorted tuple of
    the side's 11 slot codes, where code s < N_SLOTS is a depth-chart slot and N_SLOTS + g is a
    player of group g who is not on the depth chart (filled with OTHER_<g>).
    """
    TOP = 400

    def __init__(self, table: dict[str, list] | None = None):
        # key -> [signatures (list of tuples), cumulative probabilities]
        self.table = table or {}
        self._cum = {k: (v[0], np.cumsum(v[1])) for k, v in self.table.items()}

    @classmethod
    def fit(cls, samples: dict[str, Counter]) -> "PersonnelSampler":
        table = {}
        for key, cnt in samples.items():
            top = cnt.most_common(cls.TOP)
            tot = sum(c for _, c in top)
            table[key] = [[list(sig) for sig, _ in top], [c / tot for _, c in top]]
        return cls(table)

    def to_dict(self) -> dict:
        return self.table

    def _lookup(self, role: str, play_type: str, down: int, ytg: int, yl: int):
        for key in (f"{role}|{situation_key(play_type, down, ytg, yl)}", f"{role}|{play_type}",
                    f"{role}|run" if role == "off" else f"{role}|pass"):
            if key in self._cum:
                return self._cum[key]
        return None

    def sample(self, role: str, play_type: str, down: int, ytg: int, yl: int, lineup: np.ndarray,
               rng: np.random.Generator) -> tuple[list[int], list[int]]:
        """Tokens and groups (1-based, 0 = pad) of one side's 11 players for this play."""
        hit = self._lookup(role, play_type, down, ytg, yl)
        if hit is None or not lineup.any():
            return [0] * SIDE, [0] * SIDE
        sigs, cum = hit
        sig = sigs[min(int(np.searchsorted(cum, rng.random() * cum[-1])), len(sigs) - 1)]
        toks, grps = [], []
        for code in sig:
            if code < N_SLOTS:
                g = int(SLOT_GROUP[code])
                t = int(lineup[code]) or other_token(g)
            else:
                g = code - N_SLOTS
                t = other_token(g)
            toks.append(t), grps.append(g + 1)
        pad = SIDE - len(toks)
        return toks + [0] * pad, grps + [0] * pad


def signature(slots: list[int], grps: list[int]) -> tuple:
    return tuple(sorted(s if s >= 0 else N_SLOTS + g - 1 for s, g in zip(slots, grps)))


# --------------------------------------------------------------------------- inference metadata

def depth_chart_from_rows(rows: pd.DataFrame) -> dict[str, list[str]]:
    """{group: [player ids by depth]} from one team-game of player_games."""
    out = {g: [] for g in GROUPS}
    for r in rows.sort_values(["group", "depth"]).itertuples(index=False):
        if r.depth < DEPTH[r.group]:
            out[r.group].append(r.player_id)
    return out


def inference_meta(ld: LineupData, schedule_coaches: pd.DataFrame, recent_seasons: int = 3) -> dict:
    """What the checkpoint needs to build lineups at inference: default depth charts (each team's
    most recent game), a player directory, coach histories and vocabularies."""
    pg = ld.pg
    latest_season = int(pg.season.max())
    last = pg.sort_values(["season", "week"]).groupby("team").game_id.last()
    lineups = {}
    coach_of = {(r.game_id, r.team): r.coach for r in schedule_coaches.itertuples(index=False)}
    for team, gid in last.items():
        rows = pg[(pg.game_id == gid) & (pg.team == team)]
        lineups[team] = {"game_id": gid, "season": int(rows.season.iloc[0]), "week": int(rows.week.iloc[0]),
                         "depth_chart": depth_chart_from_rows(rows), "coach": coach_of.get((gid, team))}
    # player directory: anyone with an embedding or who played in recent seasons
    recent = pg[pg.season > latest_season - recent_seasons]
    keep = set(ld.pvocab.ids) | set(recent.player_id)
    agg = pg[pg.player_id.isin(keep)].sort_values(["season", "week"]).groupby("player_id").agg(
        name=("name", "last"), group=("group", "last"), position=("position", "last"),
        team=("team", "last"), last_season=("season", "last"), games=("career_games", "last"))
    players = {pid: [name, group, pos, team, int(last), int(games) + 1] for pid, name, group, pos, team, last, games
               in zip(agg.index, agg["name"], agg["group"], agg["position"], agg["team"], agg["last_season"],
                      agg["games"])}
    # coach histories: (season, team, wins, losses, ties) per coach-season
    sc = schedule_coaches.dropna(subset=["result"]) if "result" in schedule_coaches else schedule_coaches
    hist: dict[str, list] = defaultdict(list)
    for (coach, season, team), d in sc.groupby(["coach", "season", "team"]):
        hist[coach].append([int(season), team, int((d.result > 0).sum()), int((d.result < 0).sum()),
                            int((d.result == 0).sum())])
    veteran = sorted(set(ld.coaches[ld.coaches.veteran_unknown].coach))
    return {"lineups": lineups, "players": players, "coach_history": dict(hist), "veteran_coaches": veteran,
            "player_vocab": ld.pvocab.ids, "coach_vocab": ld.cvocab.names}


class LineupResolver:
    """Turns user-facing depth charts and coach names into model inputs at inference time."""

    def __init__(self, meta: dict):
        self.meta = meta
        self.pvocab = PlayerVocab(meta["player_vocab"])
        self.cvocab = CoachVocab(meta["coach_vocab"])
        self.players = meta["players"]
        self.history = meta["coach_history"]
        self.veterans = set(meta.get("veteran_coaches", []))
        self.sampler = PersonnelSampler(meta.get("personnel_sampler"))

    def default(self, team: str) -> dict:
        d = self.meta["lineups"].get(team)
        if d is None:
            return {"coach": None, "depth_chart": {g: [] for g in GROUPS}}
        return {"coach": d["coach"], "depth_chart": {g: list(v) for g, v in d["depth_chart"].items()},
                "source": {"game_id": d["game_id"], "season": d["season"], "week": d["week"]}}

    def tokens(self, depth_chart: dict[str, list[str]]) -> np.ndarray:
        arr = np.zeros(N_SLOTS, dtype=np.int32)
        for g, ids in (depth_chart or {}).items():
            if g not in DEPTH:
                raise ValueError(f"unknown position group {g!r}; expected one of {GROUPS}")
            for r, pid in enumerate(ids[: DEPTH[g]]):
                info = self.players.get(pid)
                # a player who has never appeared is NEW; known players carry their career game count
                arr[SLOT_IDX[(g, r)]] = self.pvocab.token(pid, g, info[5] if info else 0)
        return arr

    def coach_inputs(self, coach: str | None, team: str, season: int) -> tuple[int, list[float]]:
        if not coach:
            return 0, coach_feature_vector(None)
        hist = [tuple(h) for h in self.history.get(coach, [])]
        st = coach_record(hist, coach, team, season)
        return self.cvocab.token(coach, st["career_games"], coach in self.veterans), coach_feature_vector(st)

    def coach_summary(self, coach: str, team: str, season: int) -> dict:
        hist = [tuple(h) for h in self.history.get(coach, [])]
        st = coach_record(hist, coach, team, season)
        tok = self.cvocab.token(coach, st["career_games"], coach in self.veterans)
        return st | {"status": {0: "unknown", 1: "new", 2: "limited history"}.get(tok, "known"),
                     "teams": sorted({h[1] for h in hist})}

    def describe(self, pid: str) -> dict:
        info = self.players.get(pid)
        if info is None:
            return {"id": pid, "name": "NEW PLAYER" if pid == NEW_LABEL else pid, "status": "new"}
        name, group, pos, team, last_season, games = info
        tok = self.pvocab.token(pid, group, games)
        status = "new" if tok <= N_GROUPS else "limited history" if tok <= 2 * N_GROUPS else "known"
        return {"id": pid, "name": name, "group": group, "position": pos, "team": team,
                "last_season": last_season, "career_games": games, "status": status}
