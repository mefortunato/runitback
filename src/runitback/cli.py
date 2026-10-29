"""Command-line interface: runitback <sport> download | preprocess | train | evaluate | calibrate | simulate | serve."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Run It Back: simulate sports games one play at a time")
nfl = typer.Typer(add_completion=False, help="NFL: data, training, evaluation, simulation and the web app")
app.add_typer(nfl, name="nfl")
DATA = Path("data/nfl")
RUNS = Path("runs/nfl")
DEFAULT_CKPT = RUNS / "prod" / "model.pt"


def _log():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


@nfl.command()
def download(start: int = 1999, end: int = 2026, force: bool = False):
    """Download play-by-play via SportsDataverse."""
    from .nfl.data.download import download_season
    _log()
    for s in range(start, end + 1):
        typer.echo(download_season(s, DATA / "raw", force))


@nfl.command()
def preprocess(start: int = 1999, end: int = 2026):
    """Convert raw play-by-play to engine/model format and report engine replay agreement."""
    from .nfl.data.preprocess import preprocess as run
    _log()
    run(list(range(start, end + 1)), DATA / "raw", DATA / "processed")


@nfl.command()
def train(out: Path = RUNS / "prod", val_season: int = typer.Option(None, help="hold out late-season games of this season"),
          val_week: int = 10, epochs: int = 10, d_model: int = 192, n_layers: int = 4, n_heads: int = 6,
          batch_size: int = 32, lr: float = 1e-3, threads: int = 0, max_games: int = 0):
    """Train the transformer."""
    from .nfl.train import train as run
    _log()
    run(DATA / "processed", out, val_season=val_season, val_week=val_week, epochs=epochs, d_model=d_model,
        n_layers=n_layers, n_heads=n_heads, batch_size=batch_size, lr=lr, threads=threads or None,
        max_games=max_games or None)


@nfl.command()
def evaluate(ckpt: Path = RUNS / "eval" / "model.pt", sims: int = 200, out: Path = RUNS / "eval" / "report.json",
             strength_scale: float = typer.Option(None, help="override the checkpoint's calibrated scale")):
    """Next-play metrics vs baseline and full-game simulation metrics on the held-out games."""
    from .nfl.evaluate import run
    _log()
    report = run(ckpt, DATA / "processed", sims_per_game=sims, raw_dir=DATA / "raw", strength_scale=strength_scale)
    out.write_text(json.dumps(report, indent=1))
    typer.echo(json.dumps(report, indent=1))


@nfl.command()
def calibrate(ckpt: Path = RUNS / "eval_v2" / "model.pt", season: int = 2024, games: int = 100, sims: int = 100,
              write: bool = typer.Option(False, help="store the best scale in the checkpoint")):
    """Fit the team-strength scale (stretches rating gaps) by Brier score on one season."""
    from .nfl.evaluate import calibrate as run
    _log()
    typer.echo(json.dumps(run(ckpt, DATA / "processed", season, games, sims, write=write), indent=1))


@nfl.command()
def simulate(home: str, away: str, season: int = 2026, n: int = 1000, game_type: str = "REG",
             seed: int = None, ckpt: Path = DEFAULT_CKPT, out: Path = None, scores: bool = False):
    """Sample N full games and print / save the outcome distribution (.json or .csv)."""
    from .nfl.sim.api import simulate_games
    from .nfl.sim.sampler import Simulator
    sim = Simulator.load(ckpt)
    res = simulate_games(sim, home.upper(), away.upper(), season, n=n, game_type=game_type, seed=seed)
    if out and out.suffix == ".csv":
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["game", "home", "away", "home_score", "away_score"])
            for i, (h, a) in enumerate(res.scores):
                w.writerow([i, res.home, res.away, h, a])
        typer.echo(f"wrote {len(res.scores)} games to {out}")
    d = res.to_dict(include_scores=scores)
    if out and out.suffix != ".csv":
        out.write_text(json.dumps(d, indent=1))
    typer.echo(json.dumps({k: v for k, v in d.items() if k != "margin_hist"}, indent=1))


@nfl.command("fetch-model")
def fetch_model(version: str = "v0.1.0", dest: Path = DEFAULT_CKPT, force: bool = False):
    """Download a released, pre-trained model (no data download or training needed)."""
    from .nfl.release import fetch_model as run
    typer.echo(f"model ready: {run(dest, version, force)}")


@nfl.command()
def serve(ckpt: Path = DEFAULT_CKPT, host: str = "127.0.0.1", port: int = 8000):
    """Run the API + web app."""
    import os

    import uvicorn
    os.environ["RUNITBACK_NFL_CKPT"] = str(ckpt)
    uvicorn.run("runitback.nfl.server.app:app", host=host, port=port)


if __name__ == "__main__":
    app()
