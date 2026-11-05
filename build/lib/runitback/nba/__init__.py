"""Run It Back — NBA: event-by-event game generator and simulator."""
from .engine.rules import GameState, Outcome, apply_outcome, new_game
from .sim.api import SimResult, simulate_games
from .sim.sampler import Simulator

__all__ = ["GameState", "Outcome", "apply_outcome", "new_game", "SimResult", "simulate_games", "Simulator"]
