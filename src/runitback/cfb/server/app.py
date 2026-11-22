"""College football API routes: interactive play-by-play sessions and batch simulation.

Teams are ESPN team ids (``"333"`` = Alabama); ``GET /teams`` lists them with the school names the
checkpoint carries. There is no personnel data (the model is team-level), so no lineups or player
stats, and no season simulator (the college schedule is not generated).
"""
from __future__ import annotations

import logging
import random
import threading
import uuid

import torch
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

from ...core.scheduler import MODEL_SCHEDULER
from ...nfl.sim.describe import describe
from ..engine.rules import GameState, Outcome, apply_outcome, legal_types, new_game
from ..paths import checkpoint_path
from ..sim.api import simulate_games, summarize
from ..sim.sampler import Simulator
from ..vocab import PHASES

router = APIRouter()
_sim: Simulator | None = None
_sim_lock = threading.Lock()
_model_scheduler = MODEL_SCHEDULER
SESSIONS: dict[str, "Session"] = {}
MAX_SESSIONS = 500
MISSING_MODEL = ("College football model checkpoint not found: {path}. Download the released one with "
                 "`runitback cfb fetch-model`, train one with `runitback cfb download`, "
                 "`runitback cfb preprocess` and `runitback cfb train`, or set RUNITBACK_CFB_CKPT.")


def get_sim() -> Simulator:
    global _sim
    if _sim is None:
        with _sim_lock:  # the startup warm-up and a request must not load the checkpoint twice
            if _sim is None:
                ckpt = checkpoint_path()
                if not ckpt.exists():
                    raise HTTPException(503, MISSING_MODEL.format(path=ckpt))
                _sim = Simulator.load(ckpt)
    return _sim


def set_sim(sim: Simulator | None) -> None:
    global _sim
    _sim = sim


class Session:
    def __init__(self, state: GameState, seed: int, names: dict[str, str], ratings=None):
        self.ratings = ratings
        self.names = names  # {team id: school}
        self.history: list[tuple[GameState, Outcome]] = []
        self.log: list[dict] = []
        self.state = state
        self.rng = random.Random(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.cache = None
        self.wp: list[dict] = []
        self.lock = threading.RLock()

    def named(self, s: GameState) -> GameState:
        """``s`` with school names for its team ids (what ``describe`` prints)."""
        n = s.copy()
        n.home, n.away = self.names.get(s.home, s.home), self.names.get(s.away, s.away)
        return n

    def state_dict(self, s: GameState) -> dict:
        """``s`` as the API shows it: the home, away and possessing teams are school names."""
        return self.named(s).to_dict()

    def view(self) -> dict:
        with self.lock:
            return {"state": self.state_dict(self.state), "plays": self.log.copy(), "winprob": self.wp.copy(),
                    "legal_play_types": [] if self.state.over else legal_types(self.state)}


class NewGame(BaseModel):
    home: str
    away: str
    season: int = 2026
    game_type: str = Field("REG", pattern="^(REG|POST)$")
    seed: int | None = None


class Step(BaseModel):
    play_type: str | None = None
    count: int = Field(1, ge=1, le=400)
    temperature: float = Field(1.0, gt=0, le=3)


class SimRequest(BaseModel):
    home: str
    away: str
    season: int = 2026
    game_type: str = Field("REG", pattern="^(REG|POST)$")
    n: int = Field(1000, ge=1, le=20000)
    seed: int | None = None
    temperature: float = Field(1.0, gt=0, le=3)
    include_scores: bool = False
    week: int | None = None  # use ratings as of this week (default: latest)
    ratings: dict[str, dict[str, float]] | None = None  # e.g. {"333": {"off_epa": 0.05, "elo": 1600}}


def _check_teams(sim: Simulator, *teams: str) -> None:
    known = set(sim.teams())
    for t in teams:
        if t not in known:
            raise HTTPException(422, f"unknown team {t}")


def _session(gid: str) -> Session:
    if gid not in SESSIONS:
        raise HTTPException(404, "game not found")
    return SESSIONS[gid]


@router.get("/health")
def health():
    return {"ok": True, "model_loaded": _sim is not None}


@router.get("/teams")
def teams(season: int | None = None):
    sim = get_sim()
    names = sim.team_names()
    ids = sim.teams(season)
    return {"teams": [{"id": t, "name": names.get(t, t)} for t in sorted(ids, key=lambda t: names.get(t, t))],
            "seasons": sim.seasons}


@router.post("/games")
def create_game(req: NewGame):
    sim = get_sim()
    _check_teams(sim, req.home, req.away)
    if req.home == req.away:
        raise HTTPException(422, "home and away must differ")
    seed = req.seed if req.seed is not None else random.randrange(2**31)
    s = Session(new_game(req.home, req.away, req.season, req.game_type, rng=random.Random(seed)), seed,
                sim.team_names(), sim.ratings_for(req.home, req.away, req.season))
    if len(SESSIONS) >= MAX_SESSIONS:
        SESSIONS.pop(next(iter(SESSIONS)))
    gid = uuid.uuid4().hex[:12]
    SESSIONS[gid] = s
    return {"id": gid, **s.view()}


@router.get("/games/{gid}")
def get_game(gid: str):
    return {"id": gid, **_session(gid).view()}


@router.post("/games/{gid}/step")
def step(gid: str, req: Step):
    s = _session(gid)
    sim = get_sim()
    new = []
    with s.lock, _model_scheduler.interactive():
        if req.play_type and req.play_type not in legal_types(s.state):
            raise HTTPException(422, f"{req.play_type} not legal now; legal: {legal_types(s.state)}")
        if s.cache is None:
            s.cache = sim._prefill(s.history, s.ratings, None)
        for i in range(req.count):
            if s.state.over or s.cache.len >= sim.model.cfg.max_len:
                break
            forced = [{"play_type": req.play_type}] if (req.play_type and i == 0) else None
            prev = s.history[-1][1] if s.history else None
            o = sim.sample_step([s.state], [prev], s.cache, forced=forced, temperature=req.temperature,
                                gen=s.gen, ratings=s.ratings, lineups=None)[0]
            nxt = apply_outcome(s.state, o, s.rng)
            entry = {"n": len(s.history), "before": s.state_dict(s.state), "outcome": o.__dict__,
                     "text": describe(s.named(s.state), o, s.named(nxt)), "after": s.state_dict(nxt)}
            s.history.append((s.state, o))
            s.log.append(entry)
            new.append(entry)
            s.state = nxt
            if s.cache.len >= sim.model.cfg.max_len:
                s.state.over = True
    return {"id": gid, "new_plays": new, **s.view()}


@router.post("/games/{gid}/undo")
def undo(gid: str, count: int = 1):
    s = _session(gid)
    with s.lock:
        for _ in range(min(count, len(s.history))):
            s.state = s.history.pop()[0]
            s.log.pop()
        s.wp = [w for w in s.wp if w["n"] <= len(s.history)]
        s.cache = None  # rebuilt from history on next step
    return {"id": gid, **s.view()}


@router.get("/games/{gid}/winprob")
def winprob(gid: str, n: int = 200):
    s = _session(gid)
    sim = get_sim()
    n = max(10, min(n, 2000))
    with s.lock:
        state = s.state.copy()
        history = s.history.copy()
        seed = s.rng.randrange(2**31)
        ratings = s.ratings
    if state.over:
        d = state.score[0] - state.score[1]
        res = {"home_win": float(d > 0), "away_win": float(d < 0), "tie": float(d == 0),
               "home_score_mean": state.score[0], "away_score_mean": state.score[1]}
    else:
        finals, _ = sim.rollout(state, n, history=history, seed=seed, ratings=ratings, lineups=None,
                                inference_context=_model_scheduler.background)
        r = summarize(finals)
        res = {k: getattr(r, k) for k in ("home_win", "away_win", "tie", "home_score_mean", "away_score_mean")}
    point = {"n": len(history), "qtr": state.qtr, "qsec": state.qsec, **res}
    with s.lock:
        s.wp = [w for w in s.wp if w["n"] != point["n"]] + [point]
    return point


@router.post("/simulate")
def simulate(req: SimRequest):
    sim = get_sim()
    _check_teams(sim, req.home, req.away)
    r = simulate_games(sim, req.home, req.away, req.season, n=req.n, game_type=req.game_type,
                       seed=req.seed, temperature=req.temperature, week=req.week, ratings=req.ratings,
                       inference_context=_model_scheduler.background)
    names = sim.team_names()
    return r.to_dict(include_scores=req.include_scores) | {"home": names.get(req.home, req.home),
                                                            "away": names.get(req.away, req.away)}


@router.get("/meta")
def meta():
    return {"phases": PHASES}


# API-only app (routes under /api). The combined app with the web UI is runitback.server:app.
app = FastAPI(title="Run It Back — College football", version="0.1.0")
app.include_router(router, prefix="/api")
