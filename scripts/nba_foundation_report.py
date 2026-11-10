"""Summarize NBA foundation folds (runitback nba foundation): one row per fold.

    python scripts/nba_foundation_report.py runs/foundation --data data/nba/processed --out report.md

Per fold S it checks the walk-forward rules and reports what the fold learned:
- **leak checks**: the checkpoint was validated on S-1, its team-season vocabulary has no season
  >= S, and the PCA was fit on the games before S (the stored mean equals the mean of those rows'
  raw embeddings is checked in the unit tests; here: the projection's explained variance).
- **validation loss** (the event model's next-event loss on the last 40% of S-1) at the kept epoch.
- **signal**: does the embedding carry team strength? For each game of S, each team's mean
  ``_pbp_*`` over its previous 10 games (only games before it) is differenced against the
  opponent's; a ridge fit on the seasons before S maps that to the home margin. Reported as the
  correlation with the actual margin in S, next to the same from the pre-game Elo difference
  (``ratings.parquet``) and from both. This is a sanity check, not the wager experiment (N5).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

WINDOW = 10


def history_means(table: pd.DataFrame, games: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per (game_id, team): the mean of `cols` over the team's previous WINDOW games (NaN for its first)."""
    t = table.merge(games[["game_id", "game_date"]], on="game_id").sort_values(["team", "game_date", "game_id"])
    prev = t.groupby("team")[cols].transform(lambda c: c.shift(1).rolling(WINDOW, min_periods=3).mean())
    return pd.concat([t[["game_id", "team"]], prev], axis=1).set_index(["game_id", "team"])


def ridge(x: np.ndarray, y: np.ndarray, lam: float = 10.0) -> np.ndarray:
    x1 = np.c_[x, np.ones(len(x))]
    reg = lam * np.eye(x1.shape[1])
    reg[-1, -1] = 0
    return np.linalg.solve(x1.T @ x1 + reg, x1.T @ y)


def signal(table: pd.DataFrame, games: pd.DataFrame, ratings: pd.DataFrame | None, season: int) -> dict:
    cols = [c for c in table.columns if c.startswith("_pbp_")]
    hist = history_means(table, games, cols)
    g = games.copy()
    h = hist.reindex(pd.MultiIndex.from_arrays([g.game_id, g.home])).to_numpy()
    a = hist.reindex(pd.MultiIndex.from_arrays([g.game_id, g.away])).to_numpy()
    feats = {"pbp32": h - a}
    if ratings is not None:
        r = ratings.pivot(index="game_id", columns="side", values="pre_elo").reindex(g.game_id)
        feats["elo"] = (r["home"] - r["away"]).to_numpy()[:, None] / 100
        feats["pbp32+elo"] = np.c_[feats["pbp32"], feats["elo"]]
    y = (g.home_final - g.away_final).to_numpy(float)
    out = {}
    for name, x in feats.items():
        ok = ~np.isnan(x).any(1)
        fit, test = ok & (g.season < season).to_numpy(), ok & (g.season == season).to_numpy()
        if fit.sum() < 100 or test.sum() < 30:
            continue
        w = ridge(x[fit], y[fit])
        pred = np.c_[x[test], np.ones(test.sum())] @ w
        out[name] = float(np.corrcoef(pred, y[test])[0, 1])
        out["n"] = int(test.sum())  # games of S with both histories
    return out


def fold_row(d: Path, games: pd.DataFrame, ratings: pd.DataFrame | None) -> dict:
    s = int(d.name[4:])
    ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
    ts_seasons = [int(k.rsplit("_", 1)[1]) for k in ck["team_seasons"]]
    hist = ck.get("history") or []
    best = min(hist, key=lambda h: h.get("val_total", np.inf)) if hist else {}
    with np.load(d / "pca_32.npz") as f:
        explained = float(f["explained"].sum())
    table = pd.read_parquet(d / "pbp_32.parquet")
    played = games[games.game_id.isin(table.game_id)]
    return {
        "fold": s,
        "val_season": ck["val_season"],
        "latest_train_season": max(ts_seasons),
        "leak_free": ck["val_season"] == s - 1 and max(ts_seasons) < s,
        "epochs": len(hist),
        "kept_epoch": best.get("epoch"),
        "val_loss": best.get("val_total"),
        "val_type": best.get("val_type"),
        "val_result": best.get("val_result"),
        "val_elapsed": best.get("val_elapsed"),
        "team_games": len(table),
        "games_before": int((played.season < s).sum()),
        "games_in": int((played.season == s).sum()),
        "pca_explained": explained,
        **{(k if k == "n" else f"corr_{k}"): v for k, v in signal(table, played, ratings, s).items()},
    }


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path, help="directory with fold<S>/ subdirectories")
    ap.add_argument("--data", type=Path, default=Path("data/nba/processed"))
    ap.add_argument("--out", type=Path, default=None, help="markdown report (default: print)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    games = pd.read_parquet(args.data / "games.parquet")
    rpath = args.data / "ratings.parquet"
    ratings = pd.read_parquet(rpath) if rpath.exists() else None
    folds = sorted(d for d in args.root.glob("fold*") if (d / "model.pt").exists())
    df = pd.DataFrame([fold_row(d, games, ratings) for d in folds])
    md = ("# NBA foundation folds\n\n"
          "One event generator per test season S (validated on the last 40% of S-1); `pbp_32` pooled "
          "per possession, PCA fit on games before S. `corr_*`: correlation with the home margin in S "
          f"of a ridge (fit before S) on the teams' previous-{WINDOW}-game means.\n\n" + to_markdown(df))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
    if args.json:
        args.json.write_text(json.dumps(df.to_dict("records"), indent=1, default=str))
    print(md)
    if not df.leak_free.all():
        raise SystemExit("a fold saw its own test season")


if __name__ == "__main__":
    main()
