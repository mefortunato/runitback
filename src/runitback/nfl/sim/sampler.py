"""Batched autoregressive game rollouts: model samples an outcome, engine applies it."""
from __future__ import annotations

import random
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, ContextManager

import numpy as np
import torch

from ..data.ratings import RatingsTable, scale_ratings
from ..engine.rules import GameState, Outcome, apply_outcome
from ..model.features import (
    FEATURES, TeamSeasonVocab, elapsed_mask, encode_state, result_mask, type_mask, yards_mask,
)
from ..model.lineups import N_COACH_CONT, N_SLOTS, SIDE, LineupResolver
from ..model.transformer import ModelConfig, PlayTransformer
from ..vocab import PLAY_TYPES, PT, RESULTS, RS, idx_to_yards, yards_to_idx

MAX_PLAYS = 330  # safety cap on plays per game (real games: 130-240)


class GameLineups:
    """Resolved model inputs for both teams: depth-chart slot tokens and coach token/features."""

    def __init__(self, home_tokens, away_tokens, home_coach=(0, None), away_coach=(0, None), spec=None,
                 usage=None):
        self.tokens = np.stack([home_tokens, away_tokens]).astype(np.int64)      # (2, S)
        # per-slot usage rates (2, S) for picking who is on the field; None = league rotation by slot
        self.usage = None if usage is None or usage[0] is None else np.stack(usage).astype(float)
        zero = [0.0] * (N_COACH_CONT // 2)
        self.coach = [(home_coach[0], home_coach[1] or zero), (away_coach[0], away_coach[1] or zero)]
        self.spec = spec  # the user-facing {"home": {...}, "away": {...}} it was built from

    def row(self, pos: int) -> tuple[int, int, list[float]]:
        """(offense coach, defense coach, coach features) from the possession side's view."""
        o, d = self.coach[pos], self.coach[1 - pos]
        return o[0], d[0], list(o[1]) + list(d[1])


class Simulator:
    def __init__(self, model: PlayTransformer, ts: TeamSeasonVocab, meta: dict | None = None,
                 ratings: RatingsTable | None = None):
        self.model = model.eval()
        self.ts = ts
        self.meta = meta or {}
        self.ratings = ratings
        self.strength_scale = float(self.meta.get("strength_scale", 1.0))
        lm = self.meta.get("lineup_meta")
        self.lineups = LineupResolver(lm) if (lm and model.lineup is not None) else None

    def lineup_spec(self, home: str, away: str, override: dict | None = None) -> dict:
        """User-facing lineups for both teams: each team's most recent game, with `override`
        ({team or "home"/"away": {"coach": str, "depth_chart": {group: [player ids]}}}) applied.
        Depth-chart groups given in the override replace that group entirely."""
        out = {}
        for side, team in (("home", home), ("away", away)):
            d = self.lineups.default(team) if self.lineups else {"coach": None, "depth_chart": {}}
            o = (override or {}).get(team) or (override or {}).get(side) or {}
            if o.get("coach") is not None:
                d["coach"] = o["coach"] or None
            if o.get("depth_chart"):
                d["depth_chart"] = d["depth_chart"] | {g: list(v) for g, v in o["depth_chart"].items()}
            if o.get("snap_shares"):
                d["snap_shares"] = dict(o["snap_shares"])
            out[side] = {"team": team, **d}
        return out

    def lineups_for(self, home: str, away: str, season: int, override: dict | None = None) -> GameLineups | None:
        """Model inputs for a game's lineups (None for team-level models)."""
        if self.lineups is None:
            return None
        spec = self.lineup_spec(home, away, override)
        r = self.lineups
        toks = [r.tokens(spec[side]["depth_chart"]) for side in ("home", "away")]
        usage = [r.usage(spec[side]["depth_chart"], t, spec[side].get("snap_shares"))
                 for side, t in zip(("home", "away"), toks)]
        return GameLineups(toks[0], toks[1], r.coach_inputs(spec["home"]["coach"], home, season),
                           r.coach_inputs(spec["away"]["coach"], away, season), spec, usage)

    def ratings_for(self, home: str, away: str, season: int, week: int | None = None,
                    override: dict | None = None, strength_scale: float | None = None) -> tuple[dict, dict] | None:
        """(home, away) pre-game ratings as of (season, week); week=None = latest known.
        `override` maps team -> partial ratings dict to replace (what-ifs). Ratings are then
        stretched by `strength_scale` (default: the calibrated value stored in the checkpoint)."""
        scale = self.strength_scale if strength_scale is None else strength_scale
        if self.ratings is None:
            return None
        out = []
        for t in (home, away):
            r = self.ratings.latest(t, season, week)
            out.append(scale_ratings(r | ((override or {}).get(t) or {}), scale))
        return out[0], out[1]

    @classmethod
    def load(cls, path: str | Path, threads: int | None = None) -> "Simulator":
        if threads:
            torch.set_num_threads(threads)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = PlayTransformer(ModelConfig(**ck["config"]))
        model.load_state_dict(ck["state_dict"])
        ts = TeamSeasonVocab([])
        ts.keys = ck["team_seasons"]
        ts.idx = {k: i + 1 for i, k in enumerate(ts.keys)}
        ratings = RatingsTable.from_records(ck["ratings"]) if ck.get("ratings") else None
        meta = {k: v for k, v in ck.items() if k not in ("state_dict", "ratings")}
        return cls(model, ts, meta, ratings)

    @property
    def seasons(self) -> list[int]:
        return sorted({int(k.split("_")[1]) for k in self.ts.keys})

    def _feats(self, states: list[GameState], prevs: list[Outcome | None], ratings=None,
               lineups: GameLineups | None = None) -> dict[str, torch.Tensor]:
        rows = [encode_state(s, p, self.ts, ratings) for s, p in zip(states, prevs)]
        feats = {f: torch.tensor([[r[f]] for r in rows], dtype=torch.long) for f in FEATURES}
        feats["cont"] = torch.tensor([[r["cont"]] for r in rows], dtype=torch.float32)
        return feats | self._lineup_feats([[s] for s in states], lineups)

    def _lineup_feats(self, states: list[list[GameState]], lineups: GameLineups | None) -> dict:
        """Lineup/coach inputs for a (B, T) grid of states; unknown lineups when None."""
        if self.model.lineup is None:
            return {}
        B = len(states)
        if lineups is None:
            lineups = GameLineups(np.zeros(N_SLOTS), np.zeros(N_SLOTS))
        rows = [[lineups.row(s.pos) for s in seq] for seq in states]
        return {"lineup": torch.as_tensor(lineups.tokens).expand(B, -1, -1),
                "off_coach": torch.tensor([[r[0] for r in seq] for seq in rows], dtype=torch.long),
                "def_coach": torch.tensor([[r[1] for r in seq] for seq in rows], dtype=torch.long),
                "coach_cont": torch.tensor([[r[2] for r in seq] for seq in rows], dtype=torch.float32)}

    def _personnel(self, states: list[GameState], names: list[str], lineups: GameLineups | None,
                   gen: torch.Generator):
        """Sample the 22 players on the field for each row's play (after its play type is drawn).
        Returns (pooled vector or 0, personnel tokens (B, 22), groups (B, 22), offense slot codes)."""
        if self.model.lineup is None:
            return 0, None, None, None
        B = len(states)
        pers = np.zeros((B, 2 * SIDE), dtype=np.int64)
        grp = np.zeros((B, 2 * SIDE), dtype=np.int64)
        codes = [[-1] * SIDE for _ in range(B)]
        if lineups is not None and self.lineups is not None:
            rng = np.random.default_rng(int(torch.randint(2**31, (1,), generator=gen)))
            for i, (s, n) in enumerate(zip(states, names)):
                for k, (role, side) in enumerate((("off", s.pos), ("def", 1 - s.pos))):
                    t, g, c = self.lineups.sampler.sample(
                        role, n, s.down, s.ytg, s.yl, lineups.tokens[side], rng, codes=True,
                        usage=None if lineups.usage is None else lineups.usage[side])
                    pers[i, k * SIDE:(k + 1) * SIDE] = t
                    grp[i, k * SIDE:(k + 1) * SIDE] = g
                    if k == 0:
                        codes[i] = c
        pt, gt = torch.as_tensor(pers), torch.as_tensor(grp)
        return self.model.lineup.personnel(pt[:, None], gt[:, None])[:, 0], pt, gt, codes

    def _prefill(self, history: list[tuple[GameState, Outcome]], ratings=None, lineups: GameLineups | None = None):
        """Run the model over an existing play history (batch 1) and return the cache."""
        cache = self.model.new_cache(1)
        if history:
            states = [s for s, _ in history]
            prevs = [None] + [o for _, o in history[:-1]]
            rows = [encode_state(s, p, self.ts, ratings) for s, p in zip(states, prevs)]
            feats = {f: torch.tensor([[r[f] for r in rows]], dtype=torch.long) for f in FEATURES}
            feats["cont"] = torch.tensor([[r["cont"] for r in rows]], dtype=torch.float32)
            feats |= self._lineup_feats([states], lineups)
            self.model.encode(feats, cache)
        return cache

    @staticmethod
    def _sample(logits: torch.Tensor, masks: np.ndarray, temperature: float, gen: torch.Generator) -> np.ndarray:
        logits = logits.masked_fill(~torch.as_tensor(masks), float("-inf")) / max(temperature, 1e-4)
        return torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(-1).numpy()

    @torch.no_grad()
    def sample_step(self, states: list[GameState], prevs: list[Outcome | None], cache,
                    forced: list[dict | None] | None = None, temperature: float = 1.0,
                    gen: torch.Generator | None = None, ratings=None,
                    lineups: GameLineups | None = None) -> list[Outcome]:
        gen = gen or torch.Generator().manual_seed(random.randrange(2**31))
        forced = forced or [None] * len(states)
        h = self.model.encode(self._feats(states, prevs, ratings, lineups), cache)[:, 0]
        m = self.model

        tm = np.stack([type_mask(s) for s in states])
        for i, f in enumerate(forced):
            if f and f.get("play_type") and tm[i, PT[f["play_type"]]]:
                tm[i] = False
                tm[i, PT[f["play_type"]]] = True
        t = self._sample(m.type_logits(h), tm, temperature, gen)
        tt = torch.as_tensor(t)
        names = [PLAY_TYPES[k] for k in t]
        p, pers, grp, codes = self._personnel(states, names, lineups, gen)
        who_codes = [None] * len(states)
        if m.has_who and pers is not None:
            cand = m.who_candidates(pers, grp)
            wm = m.who_mask(tt, pers).numpy()
            w = self._sample(m.who_logits(h, tt, p, cand), wm, temperature, gen)
            wt = torch.as_tensor(w)
            p = p + m.who_vec(wt, tt, cand)
            who_codes = [codes[i][k] if k < SIDE else None for i, k in enumerate(w)]

        rm = np.stack([result_mask(s, n) for s, n in zip(states, names)])
        if m.has_who and pers is not None:
            rm = rm & m.who_result_mask(tt, wt, rm.shape[-1]).numpy()
        for i, f in enumerate(forced):
            if f and f.get("result") in RS and rm[i, RS[f["result"]]]:
                rm[i] = False
                rm[i, RS[f["result"]]] = True
        r = self._sample(m.result_logits(h, tt, p), rm, temperature, gen)
        rt = torch.as_tensor(r)
        rnames = [RESULTS[k] for k in r]

        ym = np.stack([yards_mask(s, n, rn) for s, n, rn in zip(states, names, rnames)])
        y = self._sample(m.yards_logits(h, tt, rt, p), ym, temperature, gen)
        em = np.stack([elapsed_mask(s) for s in states])
        e = self._sample(m.elapsed_logits(h, tt, rt, torch.as_tensor(y), p), em, temperature, gen)
        # who: depth-chart slot code of the possession team's player who got the ball (None = nobody);
        # codes >= N_SLOTS are off-chart players of group code - N_SLOTS
        return [Outcome(n, rn, idx_to_yards(yy), int(ee), None if c is None or c < 0 else int(c))
                for n, rn, yy, ee, c in zip(names, rnames, y, e, who_codes)]

    @torch.no_grad()
    def rollout(self, start: GameState, n: int, history: list[tuple[GameState, Outcome]] | None = None,
                seed: int | None = None, temperature: float = 1.0, batch_size: int = 200,
                keep_plays: int = 0, ratings: tuple[dict, dict] | None | str = "auto",
                inference_context: Callable[[], ContextManager[None]] | None = None,
                lineups: GameLineups | None | str = "auto"
                ) -> tuple[list[GameState], list[list[tuple[GameState, Outcome]]]]:
        """Simulate `n` continuations of a game from `start` (after `history`). Returns final states
        and, for the first `keep_plays` games, the simulated play logs. `ratings` = (home, away)
        pre-game team ratings; "auto" uses the latest known ratings for the game's season.
        `lineups` (see lineups_for); "auto" uses each team's most recent depth chart and coach.
        `inference_context` can schedule each model call independently."""
        history = history or []
        if ratings == "auto":
            ratings = self.ratings_for(start.home, start.away, start.season)
        if isinstance(lineups, str):
            lineups = self.lineups_for(start.home, start.away, start.season)
        rng = random.Random(seed)
        gen = torch.Generator().manual_seed(rng.randrange(2**31))
        context = inference_context or nullcontext
        with context():
            base = self._prefill(history, ratings, lineups)
        finals, logs = [], []
        for b0 in range(0, n, batch_size):
            B = min(batch_size, n - b0)
            cache = base.expand(B)
            states = [start.copy() for _ in range(B)]
            prevs = [history[-1][1] if history else None] * B
            game_logs = [[] for _ in range(B)]
            for _ in range(MAX_PLAYS - len(history)):
                if all(s.over for s in states) or cache.len >= self.model.cfg.max_len:
                    break
                with context():
                    outs = self.sample_step(states, prevs, cache, temperature=temperature, gen=gen,
                                            ratings=ratings, lineups=lineups)
                for i, (s, o) in enumerate(zip(states, outs)):
                    if s.over:
                        continue
                    if b0 + i < keep_plays:
                        game_logs[i].append((s, o))
                    states[i] = apply_outcome(s, o, rng)
                    prevs[i] = o
            finals += states
            logs += game_logs[: max(0, keep_plays - b0)]
        return finals, logs
