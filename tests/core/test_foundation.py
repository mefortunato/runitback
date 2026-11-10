"""The generic foundation folds (runitback.core.foundation) with a toy, non-NFL generator."""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from runitback.core import foundation as fd


class ToyGenerator:
    """Embeds each team-game as deterministic noise seeded by its id; "trains" by writing a file."""

    sport = "toy"

    def __init__(self):
        self.trained, self.embedded = [], []

    def train(self, data_dir, work_dir, val_season, threads=None):
        games = pd.read_parquet(Path(data_dir) / "games.parquet")
        assert games.season.max() >= val_season  # it is handed all the data; the cutoff is its job
        self.trained.append(val_season)
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        (Path(work_dir) / "out.ckpt").write_text(str(val_season))
        (Path(work_dir) / "log.txt").write_text("scratch")
        return Path(work_dir) / "out.ckpt"

    def embed(self, ckpt, data_dir, game_ids=None):
        games = pd.read_parquet(Path(data_dir) / "games.parquet")
        games = games[games.game_id.isin(self.played_games(data_dir))]
        if game_ids is not None:
            games = games[games.game_id.isin(set(game_ids))]
        self.embedded.append(len(games))
        rows = []
        for g in games.itertuples():
            for team in (g.home, g.away):
                rng = np.random.default_rng(zlib.crc32(f"{g.game_id}|{team}".encode()))
                rows.append({"game_id": g.game_id, "team": team, "season": g.season}
                            | {f"off_{j}": rng.normal() for j in range(4)}
                            | {f"def_{j}": rng.normal() for j in range(4)})
        return pd.DataFrame(rows)

    def played_games(self, data_dir):
        g = pd.read_parquet(Path(data_dir) / "games.parquet")
        return set(g.game_id[g.played])

    def current_season(self, data_dir):
        g = pd.read_parquet(Path(data_dir) / "games.parquet")
        return int(g.season[g.played].max())


def _data(tmp: Path, played_through: int) -> Path:
    rows = [(f"{s}_{i}", s, "A" if i % 2 else "B", "C" if i % 3 else "D", i < played_through or s < 2025)
            for s in (2023, 2024, 2025) for i in range(10)]
    tmp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["game_id", "season", "home", "away", "played"]).to_parquet(tmp / "games.parquet")
    return tmp


def test_toy_generator_satisfies_the_interface():
    assert isinstance(ToyGenerator(), fd.Generator)


def test_prepare_trains_once_then_only_adds_new_games(tmp_path):
    gen, root = ToyGenerator(), tmp_path / "folds"
    first = fd.prepare(gen, root, _data(tmp_path / "d1", 4), [2025, 2024], dim=3)
    assert first == {2024: "trained, embedded", 2025: "trained, embedded"}
    assert gen.trained == [2023, 2024]                     # each fold validated on S-1, never sees S
    d = fd.fold_dir(root, 2025)
    assert sorted(p.name for p in d.iterdir()) == sorted(fd.fold_files(3))  # scratch and raw removed
    before = pd.read_parquet(d / "pbp_3.parquet")
    proj = fd.load_projection(d, 3)

    again = fd.prepare(gen, root, _data(tmp_path / "d2", 7), [2025], train=False, dim=3)
    assert again == {2025: "3 new games"} and gen.embedded[-1] == 3
    after = pd.read_parquet(d / "pbp_3.parquet")
    pd.testing.assert_frame_equal(after.iloc[: len(before)], before)  # earlier rows unchanged
    for k, v in fd.load_projection(d, 3).items():
        np.testing.assert_array_equal(v, proj[k])                      # projection unchanged
    # the added rows are exactly what a full re-embedding projects with the stored PCA
    full = fd.project(gen.embed(None, tmp_path / "d2"), proj)
    key = ["game_id", "team"]
    pd.testing.assert_frame_equal(after.sort_values(key).reset_index(drop=True),
                                  full.sort_values(key).reset_index(drop=True))
    assert fd.prepare(gen, root, tmp_path / "d2", [2025], train=False, dim=3) == {2025: "0 new games"}


def test_projection_ignores_the_test_season(tmp_path):
    gen = ToyGenerator()
    raw = gen.embed(None, _data(tmp_path, 10))
    shifted = raw.copy()
    shifted.loc[shifted.season == 2025, "off_0"] += 100.0
    a, pa = fd.reduce(raw, 2025, dim=2)
    b, pb = fd.reduce(shifted, 2025, dim=2)
    np.testing.assert_array_equal(pa["components"], pb["components"])
    pd.testing.assert_frame_equal(a[a.season < 2025], b[b.season < 2025])


def test_missing_model_without_training_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        fd.prepare(ToyGenerator(), tmp_path / "folds", _data(tmp_path / "d", 4), [2025], train=False)


def test_fold_listing_and_default_seasons(tmp_path):
    gen, root = ToyGenerator(), tmp_path / "folds"
    data = _data(tmp_path / "d", 4)
    assert fd.default_seasons(gen, data) == [2020, 2021, 2022, 2023, 2024, 2025]
    fd.prepare(gen, root, data, [2024, 2025], dim=2)
    (root / "fold2019").mkdir()                            # no model: not a fold
    assert fd.existing_folds(root) == [2024, 2025]


def test_embed_fold_keeps_raw_and_reuses_it(tmp_path):
    gen = ToyGenerator()
    data, out = _data(tmp_path / "d", 4), tmp_path / "emb"
    fd.embed_fold(gen, Path("model.pt"), data, out, 2025, dim=2)
    fd.embed_fold(gen, Path("model.pt"), data, out, 2025, dim=3)
    assert gen.embedded == [len(gen.played_games(data))]   # the second width reuses raw.parquet
    assert {p.name for p in out.iterdir()} == {"raw.parquet", "pca_2.npz", "pbp_2.parquet",
                                               "pca_3.npz", "pbp_3.parquet"}
