"""Command-line interface: runitback serve, and runitback <nfl|nba> download | preprocess | train |
evaluate | calibrate | simulate | serve."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import typer

from .nba import paths as nba_paths
from .nfl.paths import data_dir, default_checkpoint, runs_dir

app = typer.Typer(add_completion=False, help="Run It Back: simulate sports games one play at a time")
nfl = typer.Typer(add_completion=False, help="NFL: data, training, evaluation, simulation and the web app")
nba = typer.Typer(add_completion=False, help="NBA: data, training, evaluation, simulation and the web app")
app.add_typer(nfl, name="nfl")
app.add_typer(nba, name="nba")
DATA = data_dir()
RUNS = runs_dir()
DEFAULT_CKPT = default_checkpoint()
NBA_DATA = nba_paths.data_dir()
NBA_RUNS = nba_paths.runs_dir()
NBA_CKPT = nba_paths.default_checkpoint()


def _log():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


@nfl.command()
def download(start: int = 1999, end: int = 2026, force: bool = False,
             players: bool = typer.Option(True, help="also download snap counts, participation, players and coaches"),
             refresh_latest: bool = typer.Option(True, help="re-download the last season even if cached "
                                                              "(it may still be in progress)")):
    """Download play-by-play via SportsDataverse, plus player and coach data from nflverse. Cached seasons
    are kept unless --force."""
    from .nfl.data.download import download_season
    _log()
    for s in range(start, end + 1):
        typer.echo(download_season(s, DATA / "raw", force or (refresh_latest and s == end)))
    if players:
        from .nfl.data.players import download_players
        download_players(list(range(start, end + 1)), DATA / "raw", force)


@nfl.command()
def preprocess(start: int = 1999, end: int = 2026):
    """Convert raw play-by-play to engine/model format and report engine replay agreement."""
    from .nfl.data.preprocess import preprocess as run
    _log()
    run(list(range(start, end + 1)), DATA / "raw", DATA / "processed")


@nfl.command()
def train(out: Path = RUNS / "prod",
          data: list[Path] = typer.Option(None, help="games/plays directory; repeat to combine several "
                                                      "(e.g. processed + synthetic). Default: data/nfl/processed"),
          init: Path = typer.Option(None, help="fine-tune: start from this checkpoint's weights and architecture"),
          val_season: int = typer.Option(None, help="hold out late-season games of this season"),
          val_week: int = 10, epochs: int = 10, d_model: int = 192, n_layers: int = 4, n_heads: int = 6,
          batch_size: int = 32, lr: float = 1e-3, threads: int = 0, max_games: int = 0,
          lineups: bool = typer.Option(True, help="player-level depth charts, per-play personnel and coaches"),
          player_dim: int = typer.Option(32, help="player/coach embedding width (0 = d_model)"),
          lineup_dropout: float = 0.2, unk_lineup_prob: float = 0.2, other_player_prob: float = 0.05,
          prev_lineup_prob: float = typer.Option(0.0, help="share of team-games trained on the previous game's "
                                                            "depth chart (what users start from)"),
          box_weight: float = typer.Option(0.0, help="weight of the game-level box-score loss (0 = off)"),
          who: bool = typer.Option(False, help="predict which player gets the ball (player stat lines)"),
          qb_weight: float = typer.Option(0.0, help="weight of the passer target (each pass play's result "
                                                     "and yards from the QB's embedding alone; 0 = off)"),
          carrier_weight: float = typer.Option(0.0, help="weight of the ball-carrier target (the play's result "
                                                          "and yards from the carrier's embedding; needs --who)"),
          players_only: bool = typer.Option(False, help="no team identity (franchise / team-season / season "
                                                         "tokens, ratings, coach records): offense and defense "
                                                         "are pooled from the players and head coaches"),
          player_effects: bool = typer.Option(False, help="add the passer / carrier heads to the play's result "
                                                          "and yards (with --qb-weight / --carrier-weight)"),
          all_effects: bool = typer.Option(False, help="additive per-player effects for all 22 on the field "
                                                       "(every position) in the play's result and yards"),
          lineup_weight: float = typer.Option(0.0, help="weight of the lineup-only target (result and yards "
                                                         "from the 22 players' effects alone; with --all-effects)"),
          min_season: int = typer.Option(1999, help="first season trained on (players-only: 2013, "
                                                     "where depth charts start)"),
          ratings: bool = typer.Option(True, help="pre-game team-strength ratings as inputs (off: team "
                                                   "strength must come from the lineups and history)"),
          unk_ts_prob: float = typer.Option(0.1, help="share of games trained with the team-season "
                                                       "tokens unknown (1 = never use them)"),
          device: str = typer.Option("auto", help="auto (CUDA GPU, then Apple MPS, then CPU), cpu, cuda, cuda:N or mps"),
          amp: bool = typer.Option(True, help="bf16 mixed precision on CUDA GPUs that support it"),
          ):
    """Train the transformer (or fine-tune one with --init)."""
    from .nfl.train import train as run
    _log()
    run(data or [DATA / "processed"], out, val_season=val_season, val_week=val_week, epochs=epochs,
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, batch_size=batch_size, lr=lr,
        threads=threads or None, max_games=max_games or None, init=init, device=device, amp=amp,
        use_lineups=lineups, player_dim=player_dim, lineup_dropout=lineup_dropout, unk_lineup_prob=unk_lineup_prob,
        other_player_prob=other_player_prob, prev_lineup_prob=prev_lineup_prob, box_weight=box_weight, who_head=who,
        use_ratings=ratings, unk_ts_prob=unk_ts_prob, qb_weight=qb_weight,
        carrier_weight=carrier_weight, players_only=players_only, min_season=min_season,
        player_effects=player_effects, all_effects=all_effects, lineup_weight=lineup_weight)


@nfl.command()
def foundation(root: Path = typer.Option(Path("foundation/nfl"), help="fold directory (RUNITBACK_NFL_FOUNDATION)"),
               season: list[int] = typer.Option(None, help="fold (test season); repeat for several. "
                                                           "Default: the backtest's last 5 and the current"),
               train: bool = typer.Option(True, help="train a fold that has no model (about an hour each)"),
               existing: bool = typer.Option(False, help="only add new games to the folds already in --root "
                                                         "(no training; the daily refresh)"),
               data: Path = DATA / "processed", threads: int = 0):
    """Foundation folds for the production wager model: train missing ones, embed new games."""
    _log()
    from .core.foundation import default_seasons, existing_folds
    from .nfl.foundation import NFL, prepare
    if existing:
        season = existing_folds(root)
        train = False
    elif not season:
        season = default_seasons(NFL, data)
    for s, what in prepare(root, data, season, train=train, threads=threads or None).items():
        typer.echo(f"fold{s}: {what}")


@nfl.command("lineup-features")
def lineup_features(ckpt: Path = typer.Option(..., help="the fold's foundation checkpoint"),
                   out: Path = typer.Option(..., help="the fold's embedding directory (has pca_32.npz)"),
                   fold: int = typer.Option(..., help="the fold's test season (projections fit before it)"),
                   sets: list[str] = typer.Option(["lu", "cf"], help="lu (lineup change) and/or cf "
                                                                      "(counterfactual history)"),
                   data: Path = DATA / "processed", raw: Path = DATA / "raw", threads: int = 0):
    """Projected lineups and the wager model's lineup context (runitback.nfl.lineup_features)."""
    import torch
    _log()
    if threads:
        torch.set_num_threads(threads)
    from .nfl.lineup_features import lineup_table
    typer.echo(lineup_table(ckpt, data, raw, fold, out, tuple(sets)))


@nfl.command("player-values")
def player_values(out: Path = typer.Option(Path("player_values/nfl"), help="output directory "
                                                                           "(RUNITBACK_NFL_PLAYER_VALUES)"),
                  first_season: int = 2016, jobs: int = 0,
                  download: bool = typer.Option(True, help="refresh the current season's depth charts, "
                                                           "injury reports and rosters first"),
                  data: Path = DATA / "processed", raw: Path = DATA / "raw"):
    """Weekly player values and each game's lineup values for the production wager model (pbp32_pvc):
    pv.parquet, plus the lineup editor's lineups.parquet / slots.json (runitback.nfl.player_values)."""
    import os

    import pandas as pd

    from .nfl import player_values as pv
    _log()
    gv = pv.game_table(data, raw, first_season, jobs or os.cpu_count() or 1, download=download)
    pv.write_game_table(gv, out, pd.read_parquet(data / "players.parquet"))
    typer.echo(f"{len(gv.pv)} team-games, {gv.lineups.game_id.nunique()} upcoming games -> {out} "
               f"(latest fit: before {gv.latest.before})")


@nfl.command()
def embed(ckpt: Path = typer.Option(..., help="foundation (play-by-play) checkpoint"),
          out: Path = typer.Option(..., help="output directory"),
          before_season: int = typer.Option(..., help="fit the projection on games before this season "
                                                      "(the walk-forward fold's test season)"),
          dim: list[int] = typer.Option([32], help="projection widths; repeat for several"),
          data: Path = DATA / "processed"):
    """Per-team-game embeddings from the play-by-play model, for the wager models (runitback.wager)."""
    _log()
    from .nfl.embed import embed_fold
    for d in dim:
        typer.echo(embed_fold(ckpt, data, out, before_season, d))


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
    _write_result(res, out, scores)


@nfl.command("refresh-lineups")
def refresh_lineups(ckpt: Path = DEFAULT_CKPT,
                    out: Path = typer.Option(None, help="where to write the refreshed checkpoint (default: overwrite --ckpt)"),
                    start: int = typer.Option(1999, help="first season (ratings, career and coach records need history)"),
                    end: int = typer.Option(None, help="last season (default: the current NFL season)"),
                    download: bool = typer.Option(True, help="download the latest nflverse data first")):
    """Update a checkpoint's ratings, default depth charts, player usage and coach records from the latest
    data, without retraining. Prints the new file's SHA-256 for publishing a release."""
    import datetime as dt

    from .nfl.refresh import refresh_checkpoint
    _log()
    today = dt.date.today()
    end = end or (today.year if today.month >= 9 else today.year - 1)  # seasons are named by their start year
    summary = refresh_checkpoint(ckpt, out or ckpt, DATA / "raw", DATA / "processed", start, end, download)
    typer.echo(json.dumps(summary, indent=1))


@nfl.command("register-release")
def register_release(version: str = typer.Argument(..., help="new version, e.g. v0.2.0 (also the release tag)"),
                     ckpt: Path = typer.Option(..., help="the checkpoint attached to that release"),
                     file: Path = typer.Option(None, help="releases.json to edit (default: the installed package's; "
                                                          "pass src/runitback/nfl/releases.json in a checkout)")):
    """Add a published checkpoint to nfl/releases.json and make it the latest (what the Docker image bakes in)."""
    from .nfl.release import RELEASES_FILE
    from .nfl.release import register_release as run
    path = file or RELEASES_FILE
    rel = run(version, ckpt, path)
    typer.echo(json.dumps({"latest": rel["latest"], "file": str(path), **rel["models"][version]}, indent=1))


@nfl.command("next-version")
def next_version(after: str = typer.Option(None, help="version to bump (default: the latest in --file)"),
                 file: Path = typer.Option(None, help="releases.json to read (default: the installed package's)")):
    """Print the next patch version after the latest release (for weekly refreshes)."""
    from .nfl.release import RELEASES_FILE, load_releases, next_patch_version
    typer.echo(next_patch_version(after or load_releases(file or RELEASES_FILE)["latest"]))


@nfl.command("fetch-model")
def fetch_model(version: str = typer.Option(None, help="release version (default: the latest in nfl/releases.json)"),
                dest: Path = DEFAULT_CKPT, force: bool = False):
    """Download a released, pre-trained model (no data download or training needed)."""
    from .nfl.release import fetch_model as run
    typer.echo(f"model ready: {run(dest, version, force)}")


def _serve(host: str, port: int, nfl_ckpt: Path | None = None, nba_ckpt: Path | None = None) -> None:
    import os

    import uvicorn
    if nfl_ckpt is not None:
        os.environ["RUNITBACK_NFL_CKPT"] = str(nfl_ckpt)
    if nba_ckpt is not None:
        os.environ["RUNITBACK_NBA_CKPT"] = str(nba_ckpt)
    uvicorn.run("runitback.server:app", host=host, port=port)


@app.command("serve")
def serve_all(nfl_ckpt: Path = DEFAULT_CKPT, nba_ckpt: Path = NBA_CKPT, host: str = "127.0.0.1", port: int = 8000):
    """Run the API + web app for every sport (a sport without a checkpoint shows how to get one)."""
    _serve(host, port, nfl_ckpt, nba_ckpt)


@nfl.command()
def serve(ckpt: Path = DEFAULT_CKPT, host: str = "127.0.0.1", port: int = 8000):
    """Run the API + web app (NFL checkpoint from --ckpt; NBA from RUNITBACK_NBA_CKPT)."""
    _serve(host, port, nfl_ckpt=ckpt)


# ---------------------------------------------------------------- NBA

@nba.command("download")
def nba_download(start: int = typer.Option(2015, help="first season (the year it ends: 2015 = 2014-15)"),
                 end: int = 2027, force: bool = False,
                 players: bool = typer.Option(True, help="also download the ESPN player box scores")):
    """Download ESPN play-by-play (and player box scores) via SportsDataverse."""
    from .nba.data.download import download_season
    _log()
    for s in range(start, end + 1):
        typer.echo(download_season(s, NBA_DATA / "raw", force))
        if players:
            from .nba.data.players import download_box
            typer.echo(download_box(s, NBA_DATA / "raw", force))


@nba.command("players")
def nba_players(start: int = 2015, end: int = 2027):
    """On-court fives for every event, plus player_games and players tables (data/nba/processed)."""
    from .nba.data.players import build_players
    _log()
    summary = build_players(list(range(start, end + 1)), NBA_DATA / "raw", NBA_DATA / "processed")
    typer.echo(json.dumps(summary, indent=1))


@nba.command("preprocess")
def nba_preprocess(start: int = 2015, end: int = 2027):
    """Convert raw play-by-play to engine/model events and report engine replay agreement."""
    from .nba.data.preprocess import preprocess as run
    _log()
    run(list(range(start, end + 1)), NBA_DATA / "raw", NBA_DATA / "processed")


@nba.command("train")
def nba_train(out: Path = NBA_RUNS / "prod",
              data: list[Path] = typer.Option(None, help="games/plays directory; repeat to combine several. "
                                                          "Default: data/nba/processed"),
              init: Path = typer.Option(None, help="fine-tune: start from this checkpoint"),
              val_season: int = typer.Option(None, help="hold out the end of this season"),
              val_fraction: float = 0.4, epochs: int = 6, d_model: int = 192, n_layers: int = 4,
              n_heads: int = 6, batch_size: int = 32, lr: float = 1e-3, threads: int = 0, max_games: int = 0,
              min_season: int = 2002,
              device: str = typer.Option("auto", help="auto (CUDA GPU, then Apple MPS, then CPU), cpu, cuda, cuda:N or mps"),
              amp: bool = typer.Option(True, help="bf16 mixed precision on CUDA GPUs that support it"),
              ):
    """Train the NBA event transformer (or fine-tune one with --init)."""
    from .nba.train import train as run
    _log()
    run(data or [NBA_DATA / "processed"], out, val_season=val_season, val_fraction=val_fraction, epochs=epochs,
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, batch_size=batch_size, lr=lr,
        threads=threads or None, max_games=max_games or None, init=init, min_season=min_season, device=device, amp=amp)


@nba.command("evaluate")
def nba_evaluate(ckpt: Path = NBA_RUNS / "eval" / "model.pt", sims: int = 100, games: int = 150,
                 out: Path = NBA_RUNS / "eval" / "report.json",
                 strength_scale: float = typer.Option(None, help="override the checkpoint's calibrated scale")):
    """Next-event NLL and full-game simulation metrics (vs Elo and the spread) on held-out games."""
    from .nba.evaluate import run
    _log()
    report = run(ckpt, NBA_DATA / "processed", sims_per_game=sims, max_games=games, strength_scale=strength_scale)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    typer.echo(json.dumps(report, indent=1))


@nba.command("calibrate")
def nba_calibrate(ckpt: Path = NBA_RUNS / "eval" / "model.pt", season: int = 2025, games: int = 100,
                  sims: int = 100, write: bool = typer.Option(False, help="store the best scale in the checkpoint")):
    """Fit the team-strength scale (stretches rating gaps) by Brier score on one season."""
    from .nba.evaluate import calibrate as run
    _log()
    typer.echo(json.dumps(run(ckpt, NBA_DATA / "processed", season, games, sims, write=write), indent=1))


@nba.command("simulate")
def nba_simulate(home: str, away: str, season: int = 2027, n: int = 1000, game_type: str = "REG",
                 seed: int = None, ckpt: Path = NBA_CKPT, out: Path = None, scores: bool = False):
    """Sample N full games and print / save the outcome distribution (.json or .csv)."""
    from .nba.sim.api import simulate_games
    from .nba.sim.sampler import Simulator
    from .nba.vocab import canon_team
    sim = Simulator.load(ckpt)
    res = simulate_games(sim, canon_team(home.upper()), canon_team(away.upper()), season, n=n,
                         game_type=game_type, seed=seed)
    _write_result(res, out, scores)


@nba.command("season")
def nba_season(season: int = 2027, seed: int = 0, temperature: float = 1.0,
               strength: str = typer.Option("checkpoint", help="checkpoint | neutral | randomized"),
               ckpt: Path = NBA_CKPT, out: Path = typer.Option(None, help="save the full result as .json")):
    """Simulate a full season: 82-game schedule, play-in and best-of-seven playoffs."""
    from .nba.sim.sampler import Simulator
    from .nba.sim.season import simulate_season
    sim = Simulator.load(ckpt)

    def progress(stage: str, completed: int, total: int) -> None:
        typer.echo(f"\r{stage}: {completed}/{total} games", nl=False, err=True)

    res = simulate_season(sim, season, seed=seed, temperature=temperature, strength_mode=strength,
                          progress=progress)
    typer.echo("", err=True)
    for conference, table in res["standings"].items():
        typer.echo(f"\n{conference}")
        for row in table["seeds"]:
            typer.echo(f"{row['seed']:>3} {row['team']:<4} {row['wins']:>2}-{row['losses']:<2} {row['status']}")
    for rnd in res["play_in_rounds"]:
        typer.echo(f"\n{rnd['round']}")
        for g in rnd["games"]:
            typer.echo(f"  {g['round']}: {g['away']} {g['away_score']} @ {g['home']} {g['home_score']}")
    for rnd in res["playoffs"]:
        typer.echo(f"\n{rnd['round']}")
        for s in rnd["series"]:
            typer.echo(f"  ({s['high']['seed']}) {s['high']['team']} vs ({s['low']['seed']}) {s['low']['team']}: "
                       f"{s['summary']}")
    typer.echo(f"\nChampion: {res['champion']}")
    if out:
        out.write_text(json.dumps(res, indent=1))
        typer.echo(f"saved {out}")


@nba.command("fetch-model")
def nba_fetch_model(version: str = typer.Option(None, help="release version (default: the latest)"),
                    dest: Path = NBA_CKPT, force: bool = False,
                    optional: bool = typer.Option(False, help="succeed without downloading if no NBA "
                                                              "model has been released yet")):
    """Download a released, pre-trained NBA model (no data download or training needed)."""
    from .nba import release
    if optional and version is None and release.LATEST is None:
        typer.echo("no NBA model released yet; skipping")
        return
    typer.echo(f"model ready: {release.fetch_model(dest, version, force)}")


@nba.command("sha256")
def nba_sha256(ckpt: Path = NBA_RUNS / "prod" / "model.pt"):
    """Print a checkpoint's SHA-256, for adding a release to runitback/nba/release.py."""
    from .core.release import sha256
    typer.echo(f"{sha256(ckpt)}  {ckpt}")


@nba.command("serve")
def nba_serve(ckpt: Path = NBA_CKPT, host: str = "127.0.0.1", port: int = 8000):
    """Run the API + web app (NBA checkpoint from --ckpt; NFL from RUNITBACK_NFL_CKPT)."""
    _serve(host, port, nba_ckpt=ckpt)


def _write_result(res, out: Path | None, scores: bool) -> None:
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


if __name__ == "__main__":
    app()
