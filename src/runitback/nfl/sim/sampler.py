"""Batched autoregressive game rollouts: model samples an outcome, engine applies it."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from ..data.ratings import RatingsTable, scale_ratings
from ..engine.rules import GameState, Outcome, apply_outcome
from ..model.features import (
    FEATURES, TeamSeasonVocab, elapsed_mask, encode_state, result_mask, type_mask, yards_mask,
)
from ..model.transformer import ModelConfig, PlayTransformer
from ..vocab import PLAY_TYPES, PT, RESULTS, RS, idx_to_yards, yards_to_idx

MAX_PLAYS = 330  # safety cap on plays per game (real games: 130-240)


class Simulator:
    def __init__(self, model: PlayTransformer, ts: TeamSeasonVocab, meta: dict | None = None,
                 ratings: RatingsTable | None = None):
        self.model = model.eval()
        self.ts = ts
        self.meta = meta or {}
        self.ratings = ratings
        self.strength_scale = float(self.meta.get("strength_scale", 1.0))

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

    def _feats(self, states: list[GameState], prevs: list[Outcome | None], ratings=None) -> dict[str, torch.Tensor]:
        rows = [encode_state(s, p, self.ts, ratings) for s, p in zip(states, prevs)]
        feats = {f: torch.tensor([[r[f]] for r in rows], dtype=torch.long) for f in FEATURES}
        feats["cont"] = torch.tensor([[r["cont"]] for r in rows], dtype=torch.float32)
        return feats

    def _prefill(self, history: list[tuple[GameState, Outcome]], ratings=None):
        """Run the model over an existing play history (batch 1) and return the cache."""
        cache = self.model.new_cache(1)
        if history:
            states = [s for s, _ in history]
            prevs = [None] + [o for _, o in history[:-1]]
            rows = [encode_state(s, p, self.ts, ratings) for s, p in zip(states, prevs)]
            feats = {f: torch.tensor([[r[f] for r in rows]], dtype=torch.long) for f in FEATURES}
            feats["cont"] = torch.tensor([[r["cont"] for r in rows]], dtype=torch.float32)
            self.model.encode(feats, cache)
        return cache

    @staticmethod
    def _sample(logits: torch.Tensor, masks: np.ndarray, temperature: float, gen: torch.Generator) -> np.ndarray:
        logits = logits.masked_fill(~torch.as_tensor(masks), float("-inf")) / max(temperature, 1e-4)
        return torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(-1).numpy()

    @torch.no_grad()
    def sample_step(self, states: list[GameState], prevs: list[Outcome | None], cache,
                    forced: list[dict | None] | None = None, temperature: float = 1.0,
                    gen: torch.Generator | None = None, ratings=None) -> list[Outcome]:
        gen = gen or torch.Generator().manual_seed(random.randrange(2**31))
        forced = forced or [None] * len(states)
        h = self.model.encode(self._feats(states, prevs, ratings), cache)[:, 0]
        m = self.model

        tm = np.stack([type_mask(s) for s in states])
        for i, f in enumerate(forced):
            if f and f.get("play_type") and tm[i, PT[f["play_type"]]]:
                tm[i] = False
                tm[i, PT[f["play_type"]]] = True
        t = self._sample(m.type_logits(h), tm, temperature, gen)
        tt = torch.as_tensor(t)
        names = [PLAY_TYPES[k] for k in t]

        rm = np.stack([result_mask(s, n) for s, n in zip(states, names)])
        for i, f in enumerate(forced):
            if f and f.get("result") in RS and rm[i, RS[f["result"]]]:
                rm[i] = False
                rm[i, RS[f["result"]]] = True
        r = self._sample(m.result_logits(h, tt), rm, temperature, gen)
        rt = torch.as_tensor(r)
        rnames = [RESULTS[k] for k in r]

        ym = np.stack([yards_mask(s, n, rn) for s, n, rn in zip(states, names, rnames)])
        y = self._sample(m.yards_logits(h, tt, rt), ym, temperature, gen)
        em = np.stack([elapsed_mask(s) for s in states])
        e = self._sample(m.elapsed_logits(h, tt, rt, torch.as_tensor(y)), em, temperature, gen)
        return [Outcome(n, rn, idx_to_yards(yy), int(ee)) for n, rn, yy, ee in zip(names, rnames, y, e)]

    @torch.no_grad()
    def rollout(self, start: GameState, n: int, history: list[tuple[GameState, Outcome]] | None = None,
                seed: int | None = None, temperature: float = 1.0, batch_size: int = 200,
                keep_plays: int = 0, ratings: tuple[dict, dict] | None | str = "auto") -> tuple[list[GameState], list[list[tuple[GameState, Outcome]]]]:
        """Simulate `n` continuations of a game from `start` (after `history`). Returns final states
        and, for the first `keep_plays` games, the simulated play logs. `ratings` = (home, away)
        pre-game team ratings; "auto" uses the latest known ratings for the game's season."""
        history = history or []
        if ratings == "auto":
            ratings = self.ratings_for(start.home, start.away, start.season)
        rng = random.Random(seed)
        gen = torch.Generator().manual_seed(rng.randrange(2**31))
        base = self._prefill(history, ratings)
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
                outs = self.sample_step(states, prevs, cache, temperature=temperature, gen=gen, ratings=ratings)
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
