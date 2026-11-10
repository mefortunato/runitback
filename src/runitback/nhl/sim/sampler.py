"""Batched autoregressive NHL game rollouts: the model samples an event (type, side, result, clock),
the rules engine applies it. Every sampled event is legal: each head is masked by the engine."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from ..data.ratings import RatingsTable, scale_ratings
from ..engine.rules import GameState, Outcome, apply_outcome, new_game
from ..model.features import (
    FEATURES, TeamSeasonVocab, cont_vector, elapsed_mask, encode_states, result_mask, side_mask, type_mask,
)
from ..model.transformer import EventTransformer, ModelConfig
from ..data.preprocess import SIDES
from ..vocab import PLAY_TYPES, RESULTS

MAX_EVENTS = 640  # safety cap on events per game (real games: about 320, the longest 548)


class Simulator:
    def __init__(self, model: EventTransformer, ts: TeamSeasonVocab, meta: dict | None = None,
                 ratings: RatingsTable | None = None):
        self.model = model.eval()
        self.ts = ts
        self.meta = meta or {}
        self.ratings = ratings

    @classmethod
    def load(cls, path: str | Path, threads: int | None = None) -> "Simulator":
        if threads:
            torch.set_num_threads(threads)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = EventTransformer(ModelConfig(**ck["config"]))
        model.load_state_dict(ck["state_dict"])
        ratings = RatingsTable.from_records(ck["ratings"]) if ck.get("ratings") else None
        meta = {k: v for k, v in ck.items() if k not in ("state_dict", "ratings")}
        return cls(model, TeamSeasonVocab.from_keys(ck["team_seasons"]), meta, ratings)

    def ratings_for(self, home: str, away: str, season: int, before: str | None = None,
                    scale: float = 1.0) -> tuple[dict, dict] | None:
        """(home, away) pre-game ratings from the checkpoint's table (None without ratings)."""
        if self.ratings is None or not self.model.cfg.n_cont:
            return None
        return tuple(scale_ratings(self.ratings.latest(t, season, before), scale) for t in (home, away))

    def _feats(self, states: list[GameState], prevs: list[Outcome | None],
               ratings: list[tuple[dict, dict] | None]) -> dict[str, torch.Tensor]:
        f = encode_states(states, prevs, self.ts)
        feats = {k: torch.as_tensor(np.asarray(f[k], dtype=np.int64))[:, None] for k in FEATURES}
        cont = [cont_vector(*r) if r is not None else cont_vector(None, None) for r in ratings]
        feats["cont"] = torch.tensor(cont, dtype=torch.float32)[:, None]
        return feats

    @staticmethod
    def _sample(logits: torch.Tensor, masks: np.ndarray, temperature: float, gen: torch.Generator) -> np.ndarray:
        logits = logits.float().masked_fill(~torch.as_tensor(masks), float("-inf")) / max(temperature, 1e-4)
        return torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(-1).numpy()

    @torch.no_grad()
    def sample_step(self, states: list[GameState], prevs: list[Outcome | None], cache,
                    ratings: list[tuple[dict, dict] | None], temperature: float = 1.0,
                    gen: torch.Generator | None = None) -> list[Outcome]:
        """One event for each state (finished games get a dummy that the caller ignores)."""
        gen = gen or torch.Generator().manual_seed(random.randrange(2**31))
        live = [s if not s.over else _dummy(s) for s in states]
        h = self.model.encode(self._feats(live, prevs, ratings), cache)[:, 0]
        m = self.model
        t = self._sample(m.type_logits(h), np.stack([type_mask(s) for s in live]), temperature, gen)
        names = [PLAY_TYPES[k] for k in t]
        tt = torch.as_tensor(t)
        sd = self._sample(m.side_logits(h, tt), np.stack([side_mask(s, n) for s, n in zip(live, names)]),
                          temperature, gen)
        st = torch.as_tensor(sd)
        r = self._sample(m.result_logits(h, tt, st), np.stack([result_mask(s, n) for s, n in zip(live, names)]),
                         temperature, gen)
        rt = torch.as_tensor(r)
        e = self._sample(m.elapsed_logits(h, tt, st, rt),
                         np.stack([elapsed_mask(s, n) for s, n in zip(live, names)]), temperature, gen)
        return [Outcome(n, RESULTS[rr], SIDES[ss], int(ee)) for n, ss, rr, ee in zip(names, sd, r, e)]

    @torch.no_grad()
    def rollout_games(self, starts: list[GameState], ratings: list[tuple[dict, dict] | None] | None = None,
                      seed: int | None = None, temperature: float = 1.0, batch_size: int = 128,
                      keep_logs: int = 0) -> tuple[list[GameState], list[list[tuple[GameState, Outcome]]]]:
        """Simulate one game from each start state (each with its own pre-game `ratings`), in
        batches. Returns the final states (a game that hits the event cap comes back with
        over=False) and the event logs of the first `keep_logs` games."""
        ratings = ratings if ratings is not None else [None] * len(starts)
        if len(ratings) != len(starts):
            raise ValueError("starts and ratings must have the same length")
        rng = random.Random(seed)
        gen = torch.Generator().manual_seed(rng.randrange(2**31))
        finals, logs = [], []
        for b0 in range(0, len(starts), batch_size):
            states = [s.copy() for s in starts[b0:b0 + batch_size]]
            rat = ratings[b0:b0 + batch_size]
            cache = self.model.new_cache(len(states))
            prevs: list[Outcome | None] = [None] * len(states)
            game_logs = [[] for _ in states]
            for _ in range(min(MAX_EVENTS, self.model.cfg.max_len)):
                if all(s.over for s in states):
                    break
                outs = self.sample_step(states, prevs, cache, rat, temperature, gen)
                for i, (s, o) in enumerate(zip(states, outs)):
                    if s.over:
                        continue
                    if b0 + i < keep_logs:
                        game_logs[i].append((s, o))
                    states[i] = apply_outcome(s, o)
                    prevs[i] = o
            finals += states
            logs += game_logs[: max(0, keep_logs - b0)]
        return finals, logs

    def simulate(self, home: str, away: str, season: int, n: int = 1000, game_type: str = "REG",
                 before: str | None = None, seed: int | None = None, temperature: float = 1.0,
                 batch_size: int = 128) -> list[GameState]:
        """`n` simulations of one game from the opening faceoff."""
        r = self.ratings_for(home, away, season, before)
        start = new_game(home, away, season, game_type)
        finals, _ = self.rollout_games([start] * n, [r] * n, seed, temperature, batch_size)
        return finals


def _dummy(s: GameState) -> GameState:
    """A playable state standing in for a finished game, so the batch can still be encoded and
    masked (its sampled event is thrown away)."""
    return new_game(s.home, s.away, s.season, s.game_type)
