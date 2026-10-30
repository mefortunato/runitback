"""Public Python API for full-game sampling."""
from __future__ import annotations

from typing import Callable, ContextManager

from ...core.summary import SimResult, summarize_scores
from ..engine.rules import GameState, Outcome, new_game
from .sampler import Simulator


def summarize(finals: list[GameState]) -> SimResult:
    return summarize_scores(finals[0].home, finals[0].away, [tuple(f.score) for f in finals])


def simulate_games(sim: Simulator, home: str, away: str, season: int, n: int = 1000,
                   game_type: str = "REG", seed: int | None = None, temperature: float = 1.0,
                   from_state: GameState | None = None,
                   history: list[tuple[GameState, Outcome]] | None = None, week: int | None = None,
                   ratings: dict | None = None,
                   inference_context: Callable[[], ContextManager[None]] | None = None) -> SimResult:
    """Sample `n` full games (or continuations of `from_state`) and summarize outcomes.

    Team strength defaults to each team's latest known ratings for `season` (or as of `week`).
    `ratings` overrides parts of them, e.g. {"KC": {"off_epa": 0.05}} for a what-if."""
    r = sim.ratings_for(home, away, season, week, ratings)
    start = from_state or new_game(home, away, season, game_type)
    if from_state is None:
        # randomize the opening kickoff per game: half the batch each way
        n1 = n // 2
        k0 = new_game(home, away, season, game_type, opening_kicker=0)
        k1 = new_game(home, away, season, game_type, opening_kicker=1)
        f0, _ = sim.rollout(k0, n1, seed=seed, temperature=temperature, ratings=r,
                            inference_context=inference_context)
        f1, _ = sim.rollout(k1, n - n1, seed=None if seed is None else seed + 1,
                            temperature=temperature, ratings=r, inference_context=inference_context)
        return summarize(f0 + f1)
    finals, _ = sim.rollout(start, n, history=history, seed=seed, temperature=temperature,
                            ratings=r, inference_context=inference_context)
    return summarize(finals)
