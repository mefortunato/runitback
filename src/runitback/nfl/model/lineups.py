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
WHO_NONE = SIDE        # "who got the ball" class for nobody (sacks, throwaways, non-scrimmage plays)
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


# ---------------------------------------------------------------------------------- player usage
# Scrimmage plays pick which players fill each position group from player-specific usage rates:
# rate = league snap share of the depth-chart role (RB1, WR3, ...) x the player's usage multiplier
# (how much more or less he plays than an average player in his role, from recent snap counts).
SHARE_PLAYS = {"run", "pass", "penalty", "qb_kneel", "qb_spike", "timeout"}
USAGE_ALPHA, USAGE_K, MULT_CLIP = 0.2, 3.0, (0.2, 3.0)


def snap_shares(pg: pd.DataFrame) -> np.ndarray:
    """Each player-game's share of his unit's snaps (offense or defense; specialists 1.0)."""
    off = pg.group.isin(["QB", "RB", "WR", "TE", "OL"]).to_numpy()
    dfn = pg.group.isin(["DL", "LB", "DB"]).to_numpy()
    tmax_o = pg.groupby(["game_id", "team"]).offense_snaps.transform("max").to_numpy()
    tmax_d = pg.groupby(["game_id", "team"]).defense_snaps.transform("max").to_numpy()
    sh = np.ones(len(pg))
    sh[off] = pg.offense_snaps.to_numpy()[off] / np.maximum(tmax_o[off], 1)
    sh[dfn] = pg.defense_snaps.to_numpy()[dfn] / np.maximum(tmax_d[dfn], 1)
    return np.clip(sh, 0, 1)


def role_share_table(pg: pd.DataFrame) -> dict[str, list[float]]:
    """League-average snap share by depth-chart role (group, rank)."""
    d = pg.assign(share=snap_shares(pg))
    out = {}
    for g in GROUPS:
        m = d[d.group == g].groupby("depth").share.mean()
        out[g] = [float(m.get(r, m.iloc[-1] if len(m) else 0.0)) for r in range(DEPTH[g])]
    return out


def usage_multipliers(pg: pd.DataFrame, role: dict[str, list[float]]) -> tuple[dict, dict]:
    """Leak-free usage multipliers: ({(game_id, player_id): multiplier before that game},
    {player_id: multiplier after all games}). EWMA of share / role share, shrunk toward 1."""
    d = pg.assign(share=snap_shares(pg)).sort_values(["season", "week", "game_id"])
    before, state = {}, {}
    for gid, pid, g, dp, sh in zip(d.game_id, d.player_id, d.group, d.depth, d.share):
        m, n = state.get(pid, (1.0, 0))
        before[(gid, pid)] = (n * m + USAGE_K) / (n + USAGE_K)
        prior = role[g][min(int(dp), DEPTH[g] - 1)]
        ratio = float(np.clip(sh / max(prior, 0.02), *MULT_CLIP))
        state[pid] = ((1 - USAGE_ALPHA) * m + USAGE_ALPHA * ratio if n else ratio, n + 1)
    final = {pid: (n * m + USAGE_K) / (n + USAGE_K) for pid, (m, n) in state.items()}
    return before, final


def slot_usage(lineup: np.ndarray, role: dict[str, list[float]], mult) -> np.ndarray:
    """Per-slot usage rates for a depth chart: role share x multiplier (mult: slot -> multiplier)."""
    r = np.zeros(N_SLOTS)
    for sl, (g, rank) in enumerate(SLOTS):
        if lineup[sl] > 0:
            r[sl] = min(1.0, role[g][rank] * float(np.clip(mult(sl), *MULT_CLIP)))
    return r


def inclusion_sample(rates: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Pick k of n candidates so candidate i is chosen with probability proportional to its rate
    (capped at 1): systematic sampling with exact inclusion probabilities, in random order."""
    n = len(rates)
    if k >= n:
        return np.arange(n)
    r = np.maximum(rates, 1e-3).astype(float)
    pi = np.zeros(n)
    fixed = np.zeros(n, bool)
    for _ in range(n):
        free = ~fixed
        pi[free] = (k - fixed.sum()) * r[free] / r[free].sum()
        over = free & (pi >= 1)
        if not over.any():
            break
        fixed |= over
        pi[fixed] = 1.0
    order = rng.permutation(n)
    cum = np.cumsum(pi[order])
    picks = np.searchsorted(cum, rng.random() + np.arange(k), side="right")
    return order[np.minimum(picks, n - 1)]


def fit_role_inclusion(ld: "LineupData", plays: pd.DataFrame, min_plays: int = 30) -> dict[str, list[float]]:
    """How often each depth-chart rank is on the field, by side, situation, group and number of that
    group in the package: key "off|<situation>|<group>|<k>" -> on-field rate per rank. Fit from
    participation joined to each game's depth chart (scrimmage plays only)."""
    pe = ld.personnel.merge(plays[["game_id", "play_id", "play_type", "down", "ytg", "yl", "pos"]],
                            on=["game_id", "play_id"])
    pe = pe[pe.play_type.isin(SHARE_PLAYS)]
    on: dict[str, np.ndarray] = {}
    n: dict[str, int] = defaultdict(int)
    for gid, pt, dn, ytg, yl, pos, h_ids, a_ids in zip(pe.game_id, pe.play_type, pe.down, pe.ytg, pe.yl, pe.pos,
                                                      pe.home_ids, pe.away_ids):
        sit = situation_key(pt, int(dn), int(ytg), int(yl))
        for role, ids in (("off", h_ids if pos == 0 else a_ids), ("def", a_ids if pos == 0 else h_ids)):
            by_group: dict[int, list[int]] = defaultdict(list)
            for pid in list(ids)[:SIDE]:
                hit = ld.player_game.get((gid, pid))
                if hit is None:
                    continue
                _, g, sl = hit
                by_group[g].append(SLOTS[sl][1] if sl >= 0 else -1)
            for g, ranks in by_group.items():
                gname = GROUPS[g]
                for key in (f"{role}|{sit}|{gname}|{len(ranks)}", f"{role}|{pt}|{gname}|{len(ranks)}"):
                    arr = on.setdefault(key, np.zeros(DEPTH[gname]))
                    for rk in ranks:
                        if rk >= 0:
                            arr[rk] += 1
                    n[key] += 1
    return {k: (v / n[k]).round(4).tolist() for k, v in on.items() if n[k] >= min_plays}


class PersonnelSampler:
    """Which depth-chart slots take the field, given the play type and situation.

    Learned from participation joined to per-game depth charts. A signature is the sorted tuple of
    the side's 11 slot codes, where code s < N_SLOTS is a depth-chart slot and N_SLOTS + g is a
    player of group g who is not on the depth chart (filled with OTHER_<g>).
    """
    TOP = 400

    def __init__(self, table: dict[str, list] | None = None, role_share: dict[str, list[float]] | None = None,
                 role_inclusion: dict[str, list[float]] | None = None):
        # key -> [signatures (list of tuples), cumulative probabilities]
        self.table = table or {}
        self._cum = {k: (v[0], np.cumsum(v[1])) for k, v in self.table.items()}
        self.role_share = role_share
        self.role_inclusion = role_inclusion or {}
        # per key: probabilities and per-signature counts of each position group (for usage mode)
        self._freq = {k: np.asarray(v[1], float) for k, v in self.table.items()}
        self._counts = {}
        for k, v in self.table.items():
            c = np.zeros((len(v[0]), N_GROUPS))
            for i, sig in enumerate(v[0]):
                for code in sig:
                    c[i, SLOT_GROUP[code] if code < N_SLOTS else code - N_SLOTS] += 1
            self._counts[k] = c
        if role_share:
            self._league_mu = np.array([sum(role_share[g]) for g in GROUPS])

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

    def _key(self, role: str, play_type: str, down: int, ytg: int, yl: int) -> str | None:
        for key in (f"{role}|{situation_key(play_type, down, ytg, yl)}", f"{role}|{play_type}",
                    f"{role}|run" if role == "off" else f"{role}|pass"):
            if key in self._cum:
                return key
        return None

    def _lookup(self, role: str, play_type: str, down: int, ytg: int, yl: int):
        key = self._key(role, play_type, down, ytg, yl)
        return None if key is None else self._cum[key]

    def _role_weights(self, role: str, sit: str, play_type: str, g: int, k: int) -> np.ndarray:
        """On-field rate of each rank of group g in this situation when the package holds k of them
        (falls back to the play type, then to the overall role snap shares)."""
        gname = GROUPS[g]
        for key in (f"{role}|{sit}|{gname}|{k}", f"{role}|{play_type}|{gname}|{k}"):
            if key in self.role_inclusion:
                return np.asarray(self.role_inclusion[key])
        return np.asarray(self.role_share[gname])

    def _sample_usage(self, key: str, lineup: np.ndarray, usage: np.ndarray, rng: np.random.Generator):
        """Package from the situation's league frequencies, tilted toward the team's own group usage;
        then players within each group by their usage rates."""
        mu = np.array([usage[SLOT_GROUP == g].sum() for g in range(N_GROUPS)])
        tilt = np.log(np.clip(mu / np.maximum(self._league_mu, 1e-6), 0.5, 2.0))
        w = self._freq[key] * np.exp(self._counts[key] @ tilt)
        i = min(int(np.searchsorted(np.cumsum(w), rng.random() * w.sum())), len(w) - 1)
        counts = self._counts[key][i].astype(int)
        role_, sit_, pt_ = self._ctx
        toks, grps, codes = [], [], []
        for g in range(N_GROUPS):
            k = counts[g]
            if k == 0:
                continue
            slots = np.flatnonzero((SLOT_GROUP == g) & (lineup > 0))
            if len(slots):
                # situational role rate x the player's own usage relative to his role
                gname = GROUPS[g]
                ranks = np.array([SLOTS[sl][1] for sl in slots])
                share = np.asarray(self.role_share[gname])[ranks]
                mult = np.clip(usage[slots] / np.maximum(share, 1e-3), *MULT_CLIP)
                w = self._role_weights(role_, sit_, pt_, g, k)[ranks] * mult
                chosen = slots[inclusion_sample(w, k, rng)]
            else:
                chosen = np.array([], int)
            for sl in chosen:
                toks.append(int(lineup[sl])), grps.append(g + 1), codes.append(int(sl))
            for _ in range(k - len(chosen)):  # depth chart too short: generic players
                toks.append(other_token(g)), grps.append(g + 1), codes.append(N_SLOTS + g)
        return toks[:SIDE], grps[:SIDE], codes[:SIDE]

    def sample(self, role: str, play_type: str, down: int, ytg: int, yl: int, lineup: np.ndarray,
               rng: np.random.Generator, codes: bool = False, usage: np.ndarray | None = None):
        """Tokens and groups (1-based, 0 = pad) of one side's 11 players for this play; with
        codes=True also each player's slot code (depth-chart slot, or N_SLOTS + group if off-chart).
        With per-slot `usage` rates (and a role table), scrimmage plays pick players by usage."""
        key = self._key(role, play_type, down, ytg, yl)
        if key is None or not lineup.any():
            return ([0] * SIDE, [0] * SIDE, [-1] * SIDE) if codes else ([0] * SIDE, [0] * SIDE)
        if usage is not None and self.role_share and play_type in SHARE_PLAYS:
            self._ctx = (role, situation_key(play_type, down, ytg, yl), play_type)
            toks, grps, cds = self._sample_usage(key, lineup, usage, rng)
            pad = SIDE - len(toks)
            if codes:
                return toks + [0] * pad, grps + [0] * pad, cds + [-1] * pad
            return toks + [0] * pad, grps + [0] * pad
        sigs, cum = self._cum[key]
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
        if codes:
            return toks + [0] * pad, grps + [0] * pad, list(sig) + [-1] * pad
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
    role = role_share_table(pg)
    _, mult = usage_multipliers(pg, role)
    players = {pid: [name, group, pos, team, int(last), int(games) + 1, round(mult.get(pid, 1.0), 3)]
               for pid, name, group, pos, team, last, games
               in zip(agg.index, agg["name"], agg["group"], agg["position"], agg["team"], agg["last_season"],
                      agg["games"])}
    # coach histories: (season, team, wins, losses, ties) per coach-season
    sc = schedule_coaches.dropna(subset=["result"]) if "result" in schedule_coaches else schedule_coaches
    hist: dict[str, list] = defaultdict(list)
    for (coach, season, team), d in sc.groupby(["coach", "season", "team"]):
        hist[coach].append([int(season), team, int((d.result > 0).sum()), int((d.result < 0).sum()),
                            int((d.result == 0).sum())])
    veteran = sorted(set(ld.coaches[ld.coaches.veteran_unknown].coach))
    return {"lineups": lineups, "players": players, "role_share": role, "coach_history": dict(hist), "veteran_coaches": veteran,
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
        self.role_share = meta.get("role_share")
        self.sampler = PersonnelSampler(meta.get("personnel_sampler"), self.role_share, meta.get("role_inclusion"))

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

    def usage(self, depth_chart: dict[str, list[str]], tokens: np.ndarray,
              snap_shares: dict[str, float] | None = None) -> np.ndarray | None:
        """Per-slot usage rates (None for checkpoints without a role table). `snap_shares` sets a
        player's rate directly (fraction of his unit's snaps, 0-1)."""
        if not self.role_share:
            return None
        pid_at = {SLOT_IDX[(g, r)]: pid for g, ids in (depth_chart or {}).items() for r, pid in enumerate(ids[: DEPTH[g]])}
        mult = lambda sl: (self.players.get(pid_at.get(sl), [None] * 7) + [1.0])[6] or 1.0
        r = slot_usage(tokens, self.role_share, mult)
        for sl, pid in pid_at.items():
            if snap_shares and pid in snap_shares and tokens[sl] > 0:
                r[sl] = float(np.clip(snap_shares[pid], 0.0, 1.0))
        return r

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
        name, group, pos, team, last_season, games = info[:6]
        tok = self.pvocab.token(pid, group, games)
        status = "new" if tok <= N_GROUPS else "limited history" if tok <= 2 * N_GROUPS else "known"
        return {"id": pid, "name": name, "group": group, "position": pos, "team": team,
                "last_season": last_season, "career_games": games, "status": status,
                "usage_multiplier": info[6] if len(info) > 6 else 1.0}
