"""The NFL foundation folds are unchanged by the generic fold code (runitback.core.foundation).

A tiny synthetic NFL dataset (seeded rollouts of an untrained model) goes through the NFL fold
pipeline end to end: ``prepare`` trains a fold, embeds every played game and fits the PCA; new
games then arrive and ``prepare`` adds them with the stored projection (``update_fold``);
``embed_fold`` writes the raw table that ``runitback nfl embed`` keeps. Every file's contents are
hashed and compared with ``fixtures/foundation_fold_golden.json``, which was written by the code on
``main`` before the fold logic moved to ``runitback.core.foundation`` (work package S3).

The checkpoint is hashed through its tensors and metadata, the tables through their columns,
dtypes and values, so the comparison is exact but does not depend on file-format metadata.
Regenerate the golden file (only when a change to the fold outputs is intended) with
``python tests/nfl/test_foundation_folds.py``.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

GOLDEN = Path(__file__).parent / "fixtures" / "foundation_fold_golden.json"
TEAMS = ["KC", "BUF", "MIA", "NYJ"]
SEASONS = (2022, 2023, 2024)
FOLD = 2024
# a fold small enough for a unit test; everything else is the production TRAIN settings
TINY = dict(epochs=1, d_model=16, n_layers=1, n_heads=2, batch_size=8, player_dim=8, device="cpu", amp=False)


def _dataset(out: Path) -> Path:
    """games.parquet / plays.parquet in the processed schema: 3 seasons x 8 weeks x 2 games,
    the plays rolled out by a seeded, untrained model (so every play is legal)."""
    from runitback.nfl.engine.rules import new_game
    from runitback.nfl.model.features import CONT_FEATURES, TeamSeasonVocab, feature_sizes
    from runitback.nfl.model.transformer import ModelConfig, PlayTransformer
    from runitback.nfl.sim.sampler import Simulator

    torch.manual_seed(0)
    ts = TeamSeasonVocab([f"{t}_{s}" for s in SEASONS for t in TEAMS])
    model = PlayTransformer(ModelConfig(feature_sizes(len(ts)), d_model=16, n_layers=1, n_heads=2,
                                        n_cont=len(CONT_FEATURES)))
    sim = Simulator(model, ts)
    rng = random.Random(0)
    games, plays = [], []
    for season in SEASONS:
        for week in range(1, 9):
            pairs = [(TEAMS[0], TEAMS[1]), (TEAMS[2], TEAMS[3])] if week % 2 else \
                    [(TEAMS[1], TEAMS[2]), (TEAMS[3], TEAMS[0])]
            for k, (home, away) in enumerate(pairs):
                gid = f"{season}_{week:02d}_{k}"
                for _ in range(20):
                    finals, logs = sim.rollout(new_game(home, away, season, opening_kicker=rng.choice((0, 1))),
                                               1, seed=rng.randrange(2**32), keep_plays=1, ratings=None)
                    if finals[0].over and logs[0]:
                        break
                f = finals[0]
                games.append(dict(game_id=gid, season=season, week=week, game_type="REG",
                                  game_date=f"{season}-09-{week + 1:02d}", home=home, away=away,
                                  home_score=f.score[0], away_score=f.score[1]))
                for i, (s, o) in enumerate(logs[0]):
                    plays.append(dict(game_id=gid, idx=i, qtr=s.qtr, qsec=s.qsec, phase=s.phase, pos=s.pos,
                                      down=s.down, ytg=s.ytg, yl=s.yl, score_h=s.score[0], score_a=s.score[1],
                                      to_h=s.timeouts[0], to_a=s.timeouts[1], play_type=o.play_type,
                                      result=o.result, yards=o.yards, elapsed=o.elapsed))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(games).to_parquet(out / "games.parquet", index=False)
    pd.DataFrame(plays).to_parquet(out / "plays.parquet", index=False)
    return out


def _h(*parts) -> str:
    d = hashlib.sha256()
    for p in parts:
        d.update(p if isinstance(p, bytes) else str(p).encode())
    return d.hexdigest()


def _frame(path: Path) -> str:
    df = pd.read_parquet(path)
    return _h(list(df.columns), [str(t) for t in df.dtypes],
              *[np.ascontiguousarray(df[c].to_numpy()).tobytes() if pd.api.types.is_numeric_dtype(df[c])
                else json.dumps(df[c].tolist()) for c in df.columns])


def _npz(path: Path) -> str:
    with np.load(path) as f:
        return _h(*[x for k in sorted(f.files) for x in (k, str(f[k].dtype), f[k].shape, f[k].tobytes())])


def _ckpt(path: Path) -> str:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.pop("state_dict")
    meta = json.dumps(ck, sort_keys=True, default=str)
    return _h(meta, *[x for k in sorted(sd) for x in (k, sd[k].numpy().tobytes())])


def _fold_hashes(d: Path) -> dict:
    return {"model.pt": _ckpt(d / "model.pt"), "pca_32.npz": _npz(d / "pca_32.npz"),
            "pbp_32.parquet": _frame(d / "pbp_32.parquet")}


def fold_outputs(tmp: Path, monkeypatch=None) -> dict:
    """Run the NFL fold pipeline on the synthetic data and hash every file it writes."""
    from runitback.nfl import foundation
    from runitback.nfl.embed import embed_fold

    tiny = foundation.TRAIN | TINY
    if monkeypatch is not None:
        monkeypatch.setattr(foundation, "TRAIN", tiny)
    else:
        foundation.TRAIN = tiny
    torch.set_num_threads(1)
    full = _dataset(tmp / "full")
    early = tmp / "early"  # the same data before the last two weeks of the test season
    early.mkdir()
    g = pd.read_parquet(full / "games.parquet")
    g = g[~((g.season == FOLD) & (g.week >= 7))]
    g.to_parquet(early / "games.parquet", index=False)
    p = pd.read_parquet(full / "plays.parquet")
    p[p.game_id.isin(g.game_id)].to_parquet(early / "plays.parquet", index=False)

    root = tmp / "folds"
    out = {"current_season": foundation.current_season(full)}
    out["first"] = foundation.prepare(root, early, [FOLD], threads=1)
    out["first_files"] = _fold_hashes(root / f"fold{FOLD}")
    out["update"] = foundation.prepare(root, full, [FOLD], train=False, threads=1)
    out["update_files"] = _fold_hashes(root / f"fold{FOLD}")
    out["noop"] = foundation.prepare(root, full, [FOLD], train=False, threads=1)
    out["noop_files"] = _fold_hashes(root / f"fold{FOLD}")
    emb = tmp / "emb"  # `runitback nfl embed`: raw table kept, a second width from the same raw
    embed_fold(root / f"fold{FOLD}" / "model.pt", full, emb, FOLD, 32)
    embed_fold(root / f"fold{FOLD}" / "model.pt", full, emb, FOLD, 4)
    out["embed_files"] = {"raw.parquet": _frame(emb / "raw.parquet"),
                          "pca_32.npz": _npz(emb / "pca_32.npz"), "pbp_32.parquet": _frame(emb / "pbp_32.parquet"),
                          "pca_4.npz": _npz(emb / "pca_4.npz"), "pbp_4.parquet": _frame(emb / "pbp_4.parquet")}
    out["fold_listing"] = sorted(x.name for x in (root / f"fold{FOLD}").iterdir())
    out["torch"] = torch.__version__.split("+")[0]
    return {k: ({str(a): b for a, b in v.items()} if isinstance(v, dict) else v) for k, v in out.items()}


def test_nfl_fold_files_are_unchanged(tmp_path, monkeypatch):
    golden = json.loads(GOLDEN.read_text())
    if torch.__version__.split("+")[0] != golden["torch"]:
        pytest.skip(f"golden fold written with torch {golden['torch']}; training differs across versions")
    threads = torch.get_num_threads()
    try:
        assert fold_outputs(tmp_path, monkeypatch) == golden
    finally:
        torch.set_num_threads(threads)  # training sets it; later tests' numerics depend on it


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as t:
        res = fold_outputs(Path(t))
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
