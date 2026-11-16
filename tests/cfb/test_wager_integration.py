"""F3: the college football wager integration: the QB lineup layout on the Sport, the foundation
embeddings keyed as the wager model keys teams, the experiment configuration and the swap test's
helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from runitback.cfb import foundation as fnd
from runitback.cfb import player_values as pv
from runitback.wager.cfb.lineup import LINEUP
from runitback.wager.core.sport import get_sport
from runitback.wager.models.transformer.data import (
    additive_spec,
    lineup_columns,
    lineup_context_columns,
    lineup_features,
    pbp_features,
)

ROOT = Path(__file__).resolve().parents[2]


def _schedule():
    return pd.DataFrame({
        "game_id": [101, 102, 103],
        "home_id": [10, 20, 30], "away_id": [20, 40, 50],
        "home_team": ["Alabama", "Georgia", "Mercer"], "away_team": ["Georgia", "Mercer", "Wofford"],
        "home_division": ["fbs", "fbs", "fcs"], "away_division": ["fbs", "fcs", "fcs"]})


def _emb():
    # a fold's embeddings: ESPN team ids as strings, ESPN game ids as strings
    return pd.DataFrame({"game_id": ["101", "101", "102", "102", "103", "103", "999"],
                         "team": ["10", "20", "20", "40", "30", "50", "10"], "season": 2024,
                         **{f"_pbp_{i}": np.arange(7, dtype=float) + i for i in range(2)}})


def test_wager_keys_use_school_names_and_pool_fcs():
    out = fnd.wager_keys(_emb(), _schedule())
    got = {(g, t) for g, t in zip(out.game_id, out.team)}
    # 101: two FBS schools; 102: Georgia and the pooled FCS side; 103 (FCS vs FCS) and the unknown game: none
    assert got == {("101", "Alabama"), ("101", "Georgia"), ("102", "Georgia"), ("102", "FCS")}
    assert "team_id" not in out and pd.api.types.is_string_dtype(out.team)
    assert out.set_index(["game_id", "team"]).loc[("102", "FCS"), "_pbp_0"] == 3.0   # team 40's row


def test_write_wager_keys(tmp_path):
    d = fnd.fold_dir(tmp_path, 2024)
    d.mkdir(parents=True)
    _emb().to_parquet(d / fnd.FILES[2], index=False)
    assert fnd.write_wager_keys(tmp_path, [2024], _schedule()) == {2024: 4}
    assert len(pd.read_parquet(d / "pbp_32_wager.parquet")) == 4


def test_cfb_sport_has_the_qb_layout():
    sport = get_sport("cfb")
    assert sport.lineup is LINEUP
    # pvc reads the level, the change vs the previous starter and the change vs the last 7, plus a flag
    assert lineup_columns("pvc", "cfb") == ["_pv_0", "_pv_1", "_pv_2", "_lu_missing"]
    spec = additive_spec(3, "pvc", "cfb")
    # one sign-constrained pair (the QB-change column) enters, and a better quarterback scores more
    assert len(spec["additive_pairs"]) == 1 and spec["additive_total_sign"] == [1.0]


def test_embeddings_and_qb_terms_join_integer_game_ids(tmp_path):
    schema = get_sport("cfb").schema
    games = pd.DataFrame({"game_id": [101, 101, 102], "team": ["Alabama", "Georgia", "Georgia"],
                          "opp_team": ["Georgia", "Alabama", "FCS"]})   # read back from CSV as integers
    emb = fnd.wager_keys(_emb(), _schedule())
    emb.to_parquet(tmp_path / "pbp.parquet", index=False)
    out = pbp_features(games, schema, str(tmp_path / "pbp.parquet"), 2)
    assert out["_pbp_missing"].tolist() == [0.0, 0.0, 0.0]
    pvt = pd.DataFrame({"game_id": ["101", "101"], "team": ["Alabama", "Georgia"],
                        **{f"_pv_{i}": np.float32([1.0, -1.0]) for i in range(3)}})
    pvt.to_parquet(tmp_path / "pv.parquet", index=False)
    lu = lineup_features(games, schema, str(tmp_path / "pv.parquet"), "pvc", "cfb")
    assert lu["_pv_2"].tolist() == [1.0, -1.0, 0.0] or lu["_pv_2"].iloc[:2].tolist() == [1.0, -1.0]
    assert lu["_lu_missing"].tolist() == [0.0, 0.0, 1.0]


def test_experiment_arms():
    cfg = yaml.safe_load((ROOT / "experiments/cfb/wager.yaml").read_text())
    assert cfg["seasons"] == [2021, 2022, 2023, 2024, 2025] and cfg["baseline"] == "prod"
    c = cfg["configs"]
    assert {"prod", "pbp32", "pbp32_qb", "prod_e4", "pbp32_e4", "pbp32_qb_e4"} <= set(c)
    assert "pbp" not in c["prod"]["model"] and "lineup" not in c["pbp32"]["model"]
    qb = c["pbp32_qb"]["model"]
    assert qb["lineup_set"] == "pvc" and qb["pbp"].format(season=2024).endswith("fold2024/pbp_32_wager.parquet")
    assert c["pbp32_qb_e4"]["fit"] == {"ensemble": 4}
    # the box-score settings are pinned identically in every arm
    for name in ("pbp32", "pbp32_qb", "prod_qb"):
        assert {k: c[name]["model"][k] for k in c["prod"]["model"]} == c["prod"]["model"]


def test_swap_helpers():
    fit = pd.DataFrame({"player_id": [1, 2, 3, 4], "plays": [900, 500, 100, 800],
                        "value_play": [0.10, 0.30, 0.90, -0.05]})
    refs = pv.regulars(fit, 400)
    assert refs == {"best": 2, "worst": 4}                    # player 3 has too few plays
    assert pv.regulars(fit, 5000) == {}
    assert pv.backup_of([1, 1, 2, 1, 3, 2], 1) == 2 and pv.backup_of([1, 1], 1) is None
    vp = {1: 0.10, 2: 0.30, 4: -0.05}
    assert np.isclose(pv.swap_delta("best", 1, vp, refs, None, 35), 0.20 * 35)
    assert np.isclose(pv.swap_delta("worst", 1, vp, refs, None, 35), -0.15 * 35)
    assert np.isclose(pv.swap_delta("replacement", 1, vp, refs, None, 35), -0.10 * 35)
    assert np.isclose(pv.swap_delta("out", 1, vp, refs, 2, 35), 0.20 * 35)
    assert np.isclose(pv.swap_delta("out", 1, vp, refs, None, 35), -0.10 * 35)   # no backup: replacement
    assert pv.swap_delta("best", 1, vp, {}, None, 35) is None


def _swap_script():
    spec = importlib.util.spec_from_file_location("cfb_lineup_swap", ROOT / "scripts/cfb_lineup_swap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_swap_summary_flags_a_wrong_direction():
    mod = _swap_script()
    rows = [dict(game_id=g, kind=k, ref="X", d_value=v, d_margin=m, d_win=w, d_total=t)
            for g, (v, m, w, t) in enumerate([(2.0, 1.1, 0.04, 0.5), (-3.0, -1.5, -0.05, -0.7)])
            for k in ("best", "out")]
    assert "moves the prediction the right way in every game: yes" in mod.summarize(pd.DataFrame(rows))
    rows.append(dict(game_id=9, kind="best", ref="X", d_value=2.0, d_margin=-0.1, d_win=0.01, d_total=0.1))
    assert "every game: no" in mod.summarize(pd.DataFrame(rows))


def test_qb_change_moves_margin_and_total_the_right_way():
    """The sign-constrained additive term: a better home starter never lowers the home margin or the
    total, and a better away starter never raises the home margin, for any head weights."""
    import torch

    from runitback.wager.models.transformer.model import PairHead

    n_before = 3
    spec = additive_spec(n_before, "pvc", "cfb")
    cols = ["a", "b", "c", *lineup_context_columns("pvc", "cfb")]
    (home_col, away_col), = spec["additive_pairs"]
    for seed in range(5):
        torch.manual_seed(seed)
        head = PairHead(8, len(cols), out_dim=2, **spec).eval()
        with torch.no_grad():
            head.add_w.normal_()
            head.add_u.normal_()
        head.set_additive_scale(torch.rand(1) + 0.5, torch.randn(1), torch.rand(1) + 0.5, torch.randn(1),
                                [14.0, 13.0])
        ctx = torch.randn(6, len(cols))
        up_home, up_away = ctx.clone(), ctx.clone()
        up_home[:, home_col] += 1.0
        up_away[:, away_col] += 1.0
        with torch.no_grad():
            m0, t0 = head.additive(ctx)
            mh, th = head.additive(up_home)
            ma, ta = head.additive(up_away)
        assert (mh >= m0 - 1e-6).all() and (th >= t0 - 1e-6).all()      # better home QB: margin and total up
        assert (ma <= m0 + 1e-6).all() and (ta >= t0 - 1e-6).all()      # better away QB: margin down, total up
        assert (mh > m0).any() or (m0 == mh).all()


def test_upcoming_games_counts_games_without_an_actual_starter():
    st = pd.DataFrame({"game_id": ["1", "1", "2", "2", "3"],
                       "starter": pd.array([5, 6, None, None, 7], dtype="Int64")})
    gv = pv.GameValues(pv=pd.DataFrame(), starters=st, usage={}, fits={}, latest=pd.DataFrame())
    assert pv.upcoming_games(gv) == 1


def test_pbp32_qb_is_the_production_model_only_when_both_variables_are_set(tmp_path, monkeypatch):
    import importlib

    import runitback.wager.cfb as cfb
    try:
        for name in ("RUNITBACK_CFB_FOUNDATION", "RUNITBACK_CFB_PLAYER_VALUES"):
            monkeypatch.delenv(name, raising=False)
        t = importlib.reload(cfb).SPORT.transformer
        assert "pbp" not in t and "lineup" not in t and "ensemble" not in t
        monkeypatch.setenv("RUNITBACK_CFB_FOUNDATION", f"{tmp_path}/folds/")        # one alone: still off
        assert "pbp" not in importlib.reload(cfb).SPORT.transformer
        monkeypatch.setenv("RUNITBACK_CFB_PLAYER_VALUES", f"{tmp_path}/pv/")
        t = importlib.reload(cfb).SPORT.transformer
        assert t["pbp"] == f"{tmp_path}/folds/fold{{season}}/pbp_32_wager.parquet" and t["pbp_dim"] == 32
        assert t["lineup"] == f"{tmp_path}/pv/pv.parquet" and t["lineup_set"] == "pvc" and t["ensemble"] == 4
        assert t["history"] == 28 and t["market_tokens"] and t["efficiency"]        # prod's settings stay
    finally:
        monkeypatch.undo()
        importlib.reload(cfb)
