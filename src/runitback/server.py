"""The combined Run It Back server: NFL and NBA APIs plus the web app.

Routes: /api/nfl/... and /api/nba/... for each sport (the NFL routes are also served at /api/...
for clients written before the NBA was added), and the single-page web app for every other path.
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


@app.get("/api/sports", tags=["meta"])
def sports():
    from .nba.paths import checkpoint_path as nba_ckpt
    from .nfl.paths import checkpoint_path as nfl_ckpt
    return {"sports": [
        {"id": "nfl", "name": "NFL", "model_available": nfl_ckpt().exists()},
        {"id": "nba", "name": "NBA", "model_available": nba_ckpt().exists()},
    ]}


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
        return FileResponse(f if path and inside and f.is_file() else WEB / "index.html")
