"""The combined Run It Back server: the generator APIs, the predictions data and the web app.

Routes: /api/nfl/... and /api/nba/... for the sports with a play-by-play generator (the NFL routes
are also served at /api/... for clients written before the NBA was added), /api/sports listing
every sport the site has pages for (:data:`SPORTS`), /data/... for the precomputed predictions pages'
JSON (`runitback-wager web export`; from $RUNITBACK_PREDICTIONS when set, so the data can be
refreshed without rebuilding the web app, else from the web build) and the single-page web app for
every other path.
Run with `runitback serve` or `uvicorn runitback.server:app`.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .nba.server.app import router as nba_router
from .nfl.server.app import router as nfl_router

app = FastAPI(title="Run It Back", version="0.2.0")
app.include_router(nfl_router, prefix="/api/nfl", tags=["nfl"])
app.include_router(nba_router, prefix="/api/nba", tags=["nba"])
app.include_router(nfl_router, prefix="/api", include_in_schema=False)  # legacy NFL paths


# Every sport the web app has pages for, in nav order (as runitback.wager.web.views.BUILTIN_VIEWS
# and web/src/sports.ts): its predictions pages render once `runitback-wager web export` has
# written them; `generator` marks the sports with a simulator served under /api/<sport>.
SPORTS = (
    ("nfl", "NFL", True),
    ("cfb", "CFB", False),
    ("nba", "NBA", True),
    ("cbb", "CBB", False),
    ("wnba", "WNBA", False),
    ("wcbb", "WCBB", False),
    ("nhl", "NHL", False),
    ("mlb", "MLB", False),
)


def _generator_loaded(sport: str) -> bool:
    if sport == "nfl":
        from .nfl.paths import checkpoint_path
    elif sport == "nba":
        from .nba.paths import checkpoint_path
    else:
        return False
    return checkpoint_path().exists()


@app.get("/api/sports", tags=["meta"])
def sports():
    """Each sport: whether it has a generator and whether its generator model is loaded."""
    return {"sports": [
        {"id": sport, "name": name, "generator": generator,
         "model_available": generator and _generator_loaded(sport)}
        for sport, name, generator in SPORTS
    ]}


PREDICTIONS = os.environ.get("RUNITBACK_PREDICTIONS")
if PREDICTIONS and Path(PREDICTIONS).is_dir():
    # In production a Cloud Storage bucket mounted read-only into the Cloud Run service, which the
    # Refresh predictions workflow rewrites several times a week (see README, "Predictions data").
    app.mount("/data", StaticFiles(directory=PREDICTIONS), name="predictions")


@app.middleware("http")
async def revalidate_predictions(request, call_next):
    """Browsers revalidate /data on every load (cheap: an ETag check), so a refresh shows at once."""
    response = await call_next(request)
    if request.url.path.startswith("/data/"):
        response.headers["Cache-Control"] = "no-cache"
    return response

WEB = Path(os.environ.get("RUNITBACK_WEB", Path(__file__).resolve().parents[2] / "web" / "dist"))
if WEB.exists():
    if (WEB / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            from fastapi import HTTPException
            raise HTTPException(404, "not found")
        f = (WEB / path).resolve()
        inside = f.is_relative_to(WEB.resolve())
        if path.startswith("data/") and not (inside and f.is_file()):
            from fastapi import HTTPException
            raise HTTPException(404, "no such predictions file")  # the pages show "not available"
        return FileResponse(f if path and inside and f.is_file() else WEB / "index.html")
