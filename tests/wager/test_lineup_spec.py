"""Sport-agnostic lineup plumbing (docs/multisport/PLAN.md, S1): the NFL's position groups, lineup
column widths, additive-term layout and total signs now come from ``Sport.lineup``
(:class:`~runitback.wager.core.lineup.LineupSpec`) instead of module constants. NFL predictions
must be bit-identical to before the move.

``fixtures/lineup_pvc_golden.json`` was written by ``_golden`` on ``main`` before the refactor
(the NFL ``pvc`` constants hard-coded in ``wager/models/transformer/data.py`` and
``wager/web/lineups.py``); the trained 2-seed ensemble below must reproduce it exactly.
"""
import json
import pathlib
import tempfile

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest

from runitback.wager.core.schema import played
from runitback.wager.core.sport import Sport, register_sport

from .conftest import TOY_SCHEMA, make_games

GOLDEN = pathlib.Path(__file__).parent / "fixtures" / "lineup_pvc_golden.json"
EDITOR_GROUPS = ("QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB", "K", "P")


def _trainer():
    return pl.Trainer(
        max_epochs=6,
        default_root_dir=tempfile.mkdtemp(),
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min")],
    )


def _golden(root: pathlib.Path, **sport_args) -> dict:
    """Train a 2-seed ``pvc`` ensemble on the toy league with a synthetic ``pv.parquet`` (and the
    lineup editor's files next to it), and return its predictions, editor weights and editor view.
    ``sport_args`` are extra :class:`Sport` fields (``lineup=`` after the refactor)."""
    from runitback.wager.models.transformer import fit_transformer
    from runitback.wager.web.lineups import LineupEditor, additive_weights

    games = make_games(persistence=0.7)
    rng = np.random.default_rng(1)
    keys = games[["game_id", "team"]].drop_duplicates().reset_index(drop=True)
    pv = keys.assign(**{f"_pv_{i}": rng.normal(size=len(keys)).astype(np.float32) for i in range(20)})
    pv.to_parquet(root / "pv.parquet")
    upcoming = games[games.margin.isna()]
    lineup_rows = []
    for g, t in zip(upcoming.game_id, upcoming.team):
        for i, grp in enumerate(EDITOR_GROUPS):
            for rank in (-1, 0, 1):
                lineup_rows.append((g, t, grp, rank, f"{t}_{grp}_{rank}", f"P {t} {grp} {rank}", grp,
                                    round(float(rng.normal()), 4)))
    pd.DataFrame(lineup_rows, columns=["game_id", "team", "group", "rank", "player_id", "name", "position",
                                       "value_play"]).assign(source="chart").to_parquet(root / "lineups.parquet")
    (root / "slots.json").write_text(json.dumps({g: [10.0 + i, 2.0] for i, g in enumerate(EDITOR_GROUPS)}))

    register_sport(Sport(
        name="toy_pvc",
        schema=TOY_SCHEMA,
        download=lambda output_file=None: make_games(),
        transformer={"history": 8, "d_model": 16, "nhead": 2, "num_layers": 1, "total_weight": 1.0,
                     "lineup": str(root / "pv.parquet"), "lineup_set": "pvc", "ensemble": 2},
        **sport_args,
    ))
    res = fit_transformer(games, "toy_pvc", trainer=_trainer, batch_size=32, seed=0, print_metrics=False)
    predictor = res["predictor"]
    predictor.update_state(played(games, TOY_SCHEMA))
    target = upcoming[upcoming.location >= 0].drop_duplicates("game_id").reset_index(drop=True)
    out = predictor.predict(target)
    editor = LineupEditor.load(predictor)
    g = target.iloc[0]
    return {
        "context_columns": list(predictor.model.features.context_columns),
        "pred_margin": [float(x) for x in out.pred_margin],
        "pred_total": [float(x) for x in out.pred_total],
        "weights": additive_weights(predictor),
        "editor": editor.game(g.game_id, g.team, g.opp),
    }


def test_nfl_lineup_spec_matches_the_old_constants():
    from runitback.wager.core.sport import get_sport
    from runitback.wager.models.transformer.data import (
        additive_spec,
        lineup_columns,
        lineup_sets,
    )

    spec = get_sport("nfl").lineup
    assert spec.groups == ("QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB", "ST")
    assert spec.total_sign == (1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0)
    assert spec.levels == tuple(range(9)) and spec.changes == tuple(range(11, 20))
    assert spec.recent == 8
    assert lineup_sets(spec) == {"lp": ("_lp_", 16), "lu": ("_lu_", 16), "cf": ("_cf_", 32),
                                 "pv": ("_pv_", 11), "pva": ("_pv_", 20), "pvc": ("_pv_", 20)}
    assert spec.editor_groups == EDITOR_GROUPS
    assert [spec.column_of(g) for g in EDITOR_GROUPS] == [*spec.groups[:8], "ST", "ST"]
    # the old hard-coded additive_spec, for every set it served
    for which, levels in (("pva", True), ("pvc", False), ("lu+pvc", False), ("lp+lu+pva", True)):
        n_before = 3
        start = n_before + sum({"lp": 16, "lu": 16}[k] for k in which.split("+")[:-1])
        opp = n_before + len(lineup_columns(which, "nfl"))
        idx = (list(range(9)) if levels else []) + list(range(11, 20))
        old = {
            "additive_pairs": [(start + i, opp + (start - n_before) + i) for i in idx],
            "additive_total_sign": [1.0] * 5 + [-1.0] * 3 + [0.0] + ([1.0] * 5 + [-1.0] * 3 + [0.0] if levels else []),
            "additive_kind": ([0] * 9 if levels else []) + [1] * 9,
            "mask_context": [start + i for i in range(20)] + [opp + (start - n_before) + i for i in range(20)],
        }
        assert additive_spec(n_before, which, "nfl") == old
    from runitback.nfl import player_values as pv

    assert pv.GROUP_COLUMNS == spec.groups and pv.RECENT == 8
    assert pv.PV_COLUMNS == [f"_pv_{i}" for i in range(20)]


def test_lineup_sets_without_player_values_need_no_sport():
    from runitback.wager.models.transformer.data import additive_spec, lineup_columns

    assert len(lineup_columns("lp+lu")) == 33 and additive_spec(3, "lu") == {}
    with pytest.raises(ValueError, match="player-value"):
        lineup_columns("pvc", "toy")


def test_trained_nfl_lineup_model_is_bit_identical(tmp_path):
    from runitback.wager.nfl.lineup import LINEUP

    got = _golden(tmp_path, lineup=LINEUP)
    want = json.loads(GOLDEN.read_text())
    assert got["context_columns"] == want["context_columns"]
    assert np.array_equal(got["pred_margin"], want["pred_margin"])
    assert np.array_equal(got["pred_total"], want["pred_total"])
    assert got["weights"] == want["weights"]
    assert got["editor"] == want["editor"]


def test_another_sports_layout_drives_the_columns_and_terms():
    from runitback.wager.core.lineup import LineupSpec
    from runitback.wager.models.transformer.data import additive_spec, lineup_columns

    # a hockey-like layout: forwards, defense, goalie, no previous-game summaries
    spec = LineupSpec(groups=("F", "D", "G"), total_sign=(1.0, -1.0, -1.0), recent=5,
                      editor_groups=("F", "D", "G"))
    assert spec.changes == (3, 4, 5) and spec.columns == [f"_pv_{i}" for i in range(6)]
    assert lineup_columns("pvc", spec) == [*spec.columns, "_lu_missing"]
    out = additive_spec(2, "pvc", spec)
    assert out["additive_pairs"] == [(5, 12), (6, 13), (7, 14)]
    assert out["additive_total_sign"] == [1.0, -1.0, -1.0] and out["additive_kind"] == [1, 1, 1]
    assert len(lineup_columns("pv", spec)) == 4
