"""FastAPI server: interactive play-by-play sessions + batch simulation endpoint."""
from __future__ import annotations

import os
import logging
import random
import threading
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..engine.rules import GameState, Outcome, apply_outcome, legal_types, new_game
from ..sim.api import simulate_games, summarize
from ..sim.describe import describe
from ..sim.season import generate_schedule, simulate_season
from ..sim.sampler import Simulator
from ..vocab import PHASES, TEAMS

app = FastAPI(title="Run It Back — NFL", version="0.1.0")
_sim: Simulator | None = None
_lock = threading.Lock()  # torch model calls are serialized
SESSIONS: dict[str, "Session"] = {}
MAX_SESSIONS = 500
SEASON_JOBS: dict[str, dict] = {}
_season_jobs_lock = threading.Lock()
MAX_ACTIVE_SEASON_JOBS = 1
MAX_SEASON_JOBS = 8


def get_sim() -> Simulator:
    global _sim
    if _sim is None:
        ckpt = os.environ.get("RUNITBACK_NFL_CKPT", "runs/nfl/prod/model.pt")
        if not Path(ckpt).exists():
            raise HTTPException(503, f"model checkpoint not found: {ckpt} (download one with: runitback nfl fetch-model)")
        _sim = Simulator.load(ckpt)
    return _sim


def set_sim(sim: Simulator) -> None:
    global _sim
    _sim = sim


class Session:
    def __init__(self, state: GameState, seed: int, ratings=None):
        self.ratings = ratings
        self.history: list[tuple[GameState, Outcome]] = []
        self.log: list[dict] = []
        self.state = state
        self.rng = random.Random(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.cache = None
        self.wp: list[dict] = []

    def view(self) -> dict:
        return {"state": self.state.to_dict(), "plays": self.log, "winprob": self.wp,
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
    ratings: dict[str, dict[str, float]] | None = None  # e.g. {"KC": {"off_epa": 0.05, "elo": 1600}}


class SeasonRequest(BaseModel):
    season: int = Field(2027, ge=1999, le=2040)
    seed: int = 0
    temperature: float = Field(1.0, gt=0, le=3)
    strength_mode: str = Field("randomized", pattern="^(checkpoint|neutral|randomized)$")


def _check_teams(*teams: str) -> None:
    for t in teams:
        if t not in TEAMS:
            raise HTTPException(422, f"unknown team {t}")


def _session(gid: str) -> Session:
    if gid not in SESSIONS:
        raise HTTPException(404, "game not found")
    return SESSIONS[gid]


@app.get("/api/health")
def health():
    return {"ok": True, "model_loaded": _sim is not None}


@app.get("/api/teams")
def teams():
    sim = get_sim()
    return {"teams": TEAMS, "seasons": sim.seasons}


@app.post("/api/games")
def create_game(req: NewGame):
    home, away = req.home.upper(), req.away.upper()
    _check_teams(home, away)
    if home == away:
        raise HTTPException(422, "home and away must differ")
    seed = req.seed if req.seed is not None else random.randrange(2**31)
    s = Session(new_game(home, away, req.season, req.game_type, rng=random.Random(seed)), seed,
                get_sim().ratings_for(home, away, req.season))
    if len(SESSIONS) >= MAX_SESSIONS:
        SESSIONS.pop(next(iter(SESSIONS)))
    gid = uuid.uuid4().hex[:12]
    SESSIONS[gid] = s
    return {"id": gid, **s.view()}


@app.get("/api/games/{gid}")
def get_game(gid: str):
    return {"id": gid, **_session(gid).view()}


@app.post("/api/games/{gid}/step")
def step(gid: str, req: Step):
    s = _session(gid)
    sim = get_sim()
    if req.play_type and req.play_type not in legal_types(s.state):
        raise HTTPException(422, f"{req.play_type} not legal now; legal: {legal_types(s.state)}")
    new = []
    with _lock:
        if s.cache is None:
            s.cache = sim._prefill(s.history, s.ratings)
        for i in range(req.count):
            if s.state.over or s.cache.len >= sim.model.cfg.max_len:
                break
            forced = [{"play_type": req.play_type}] if (req.play_type and i == 0) else None
            prev = s.history[-1][1] if s.history else None
            o = sim.sample_step([s.state], [prev], s.cache, forced=forced, temperature=req.temperature, gen=s.gen,
                                 ratings=s.ratings)[0]
            nxt = apply_outcome(s.state, o, s.rng)
            entry = {"n": len(s.history), "before": s.state.to_dict(), "outcome": o.__dict__,
                     "text": describe(s.state, o, nxt), "after": nxt.to_dict()}
            s.history.append((s.state, o))
            s.log.append(entry)
            new.append(entry)
            s.state = nxt
            if s.cache.len >= sim.model.cfg.max_len:
                s.state.over = True
    return {"id": gid, "new_plays": new, **s.view()}


@app.post("/api/games/{gid}/undo")
def undo(gid: str, count: int = 1):
    s = _session(gid)
    for _ in range(min(count, len(s.history))):
        s.state = s.history.pop()[0]
        s.log.pop()
    s.wp = [w for w in s.wp if w["n"] <= len(s.history)]
    s.cache = None  # rebuilt from history on next step
    return {"id": gid, **s.view()}


@app.get("/api/games/{gid}/winprob")
def winprob(gid: str, n: int = 200):
    s = _session(gid)
    sim = get_sim()
    n = max(10, min(n, 2000))
    if s.state.over:
        d = s.state.score[0] - s.state.score[1]
        res = {"home_win": float(d > 0), "away_win": float(d < 0), "tie": float(d == 0),
               "home_score_mean": s.state.score[0], "away_score_mean": s.state.score[1]}
    else:
        with _lock:
            finals, _ = sim.rollout(s.state, n, history=s.history, seed=s.rng.randrange(2**31),
                                    ratings=s.ratings)
        r = summarize(finals)
        res = {k: getattr(r, k) for k in ("home_win", "away_win", "tie", "home_score_mean", "away_score_mean")}
    point = {"n": len(s.history), "qtr": s.state.qtr, "qsec": s.state.qsec, **res}
    s.wp = [w for w in s.wp if w["n"] != point["n"]] + [point]
    return point


@app.post("/api/simulate")
def simulate(req: SimRequest):
    home, away = req.home.upper(), req.away.upper()
    _check_teams(home, away)
    with _lock:
        r = simulate_games(get_sim(), home, away, req.season, n=req.n, game_type=req.game_type,
                           seed=req.seed, temperature=req.temperature, week=req.week,
                           ratings=req.ratings)
    return r.to_dict(include_scores=req.include_scores)


def _run_season_job(job_id: str, req: SeasonRequest, schedule: list[dict], bye_weeks: dict[str, int],
                    position_source: str) -> None:
    job = SEASON_JOBS[job_id]

    def update_progress(stage: str, completed: int, total: int) -> None:
        with _season_jobs_lock:
            job["progress"] = {"stage": stage, "completed": completed, "total": total}

    try:
        checkpoint = os.environ.get("RUNITBACK_NFL_CKPT", "runs/nfl/prod/model.pt")
        dataset_dir = Path("data/nfl/synthetic") / str(req.season) / job_id
        result = simulate_season(
            get_sim(), schedule, req.season, seed=req.seed, temperature=req.temperature,
            strength_mode=req.strength_mode, progress=update_progress, inference_lock=_lock,
            dataset_dir=dataset_dir, run_id=job_id, checkpoint=checkpoint,
            position_source=position_source, bye_weeks=bye_weeks,
        )
        with _season_jobs_lock:
            job["status"] = "complete"
            job["result"] = result
    except Exception as exc:
        logging.getLogger(__name__).exception("season simulation %s failed", job_id)
        with _season_jobs_lock:
            job["status"] = "failed"
            job["error"] = str(exc)


@app.post("/api/seasons", status_code=202)
def start_season(req: SeasonRequest):
    sim = get_sim()
    if req.strength_mode == "randomized" and sim.ratings is None:
        raise HTTPException(422, "randomized strength mode requires checkpoint ratings")
    try:
        generated = generate_schedule(sim, req.season, seed=req.seed)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    schedule = generated["games"]
    with _season_jobs_lock:
        if sum(job["status"] == "running" for job in SEASON_JOBS.values()) >= MAX_ACTIVE_SEASON_JOBS:
            raise HTTPException(429, "a season simulation is already running")
        while len(SEASON_JOBS) >= MAX_SEASON_JOBS:
            old_job = next((key for key, job in SEASON_JOBS.items() if job["status"] != "running"), None)
            if old_job is None:
                raise HTTPException(429, "season job history is full")
            SEASON_JOBS.pop(old_job)
        job_id = uuid.uuid4().hex[:12]
        SEASON_JOBS[job_id] = {
            "id": job_id, "status": "running",
            "progress": {"stage": "Starting", "completed": 0,
                         "total": len(schedule) + (13 if req.season >= 2020 else 11)},
            "result": None, "error": None, "position_source": generated["position_source"],
            "bye_weeks": generated["bye_weeks"],
        }
    threading.Thread(target=_run_season_job,
                     args=(job_id, req, schedule, generated["bye_weeks"],
                           generated["position_source"]), daemon=True).start()
    with _season_jobs_lock:
        return dict(SEASON_JOBS[job_id])


@app.get("/api/seasons/{job_id}")
def season_status(job_id: str):
    with _season_jobs_lock:
        job = SEASON_JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "season simulation not found")
        return dict(job)


@app.get("/api/meta")
def meta():
    return {"phases": PHASES}


WEB = Path(os.environ.get("RUNITBACK_WEB", Path(__file__).resolve().parents[4] / "web" / "dist"))
if WEB.exists():
    app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = WEB / path
        return FileResponse(f if path and f.is_file() else WEB / "index.html")
