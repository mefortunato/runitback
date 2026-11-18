"""C4: the college player-value lineup layout on both Sports, the weight start of the lineup terms, the
tables joining a games table whose ids are integers and whose Non-D1 opponents have no rows, and the
experiment configurations."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from runitback.wager.cbb.lineup import LINEUP
from runitback.wager.core.sport import get_sport
from runitback.wager.models.transformer import default_model
from runitback.wager.models.transformer.data import additive_spec, lineup_columns, lineup_features, pbp_features

ROOT = Path(__file__).resolve().parents[2]
softplus = torch.nn.functional.softplus


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("sport", ["cbb", "wcbb"])
def test_sport_has_the_lineup_layout(sport):
    assert get_sport(sport).lineup is LINEUP
    assert len(lineup_columns("pvc", sport)) == 14 + 1
    spec = additive_spec(3, "pvc", sport)
    # the six changes vs the last 8 rotations: offense up and defense down on the total
    assert [i for i, _ in spec["additive_pairs"]] == [3 + 8 + g for g in range(6)]
    assert spec["additive_total_sign"] == [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]
    assert LINEUP.recent == 8


def test_lineup_weight_start():
    kw = dict(lineup="x/pv.parquet", lineup_set="pvc")
    default = default_model("cbb", **kw).head
    # unchanged default: softplus(-0.2), the NFL's ~0.6 points per point (existing models are untouched)
    assert float(default.add_w[0].detach()) == pytest.approx(-0.2) and default.hparams["additive_init"] is None
    college = default_model("cbb", additive_init=0.2, **kw).head
    assert softplus(college.add_w).tolist() == pytest.approx([0.2, 0.2], abs=1e-6)
    assert college.hparams["additive_init"] == 0.2
    # the start moves nothing else: same pairs and signs, and the u weight keeps its start
    assert college.additive_pairs == default.additive_pairs
    assert torch.equal(college.add_u, default.add_u)
    # without the lineup terms the option is ignored
    assert default_model("cbb", additive_init=0.2).head.additive_pairs == ()


def test_tables_join_integer_game_ids_and_leave_non_d1_missing(tmp_path):
    schema = get_sport("cbb").schema
    games = pd.DataFrame({"game_id": [401, 401, 402], "team": ["Duke", "Non-D1", "Duke"],
                          "opp_team": ["Non-D1", "Duke", "UNC"], "season": 2026})
    emb = pd.DataFrame({"game_id": ["401", "401", "402"], "team": ["Duke", "Non-D1", "Duke"], "season": 2026,
                        **{f"_pbp_{i}": [1.0, 2.0, 3.0] for i in range(4)}})
    emb.to_parquet(tmp_path / "pbp.parquet")
    assert pbp_features(games, schema, str(tmp_path / "pbp.parquet"), 4)["_pbp_missing"].tolist() == [0, 0, 0]
    pv = pd.DataFrame({"game_id": ["401", "402", "402"], "team": ["Duke", "Duke", "UNC"], "team_id": 1,
                       "season": 2026, **{f"_pv_{i}": np.float32([0.5, 1.5, -1.0]) for i in range(14)}})
    pv.to_parquet(tmp_path / "pv.parquet")
    lu = lineup_features(games, schema, str(tmp_path / "pv.parquet"), "pvc", "cbb")
    # the pooled Non-D1 side has no player values: flagged missing, the Division I side is found
    assert lu["_lu_missing"].tolist() == [0.0, 1.0, 0.0]
    assert lu["_opp_lu_missing"].tolist() == [1.0, 0.0, 0.0]
    assert lu["_pv_8"].tolist()[0] == 0.5 and lu["_opp_pv_8"].tolist()[2] == -1.0


def test_join_check_counts_division_one_games(tmp_path):
    check = _script("college_join_check")
    schema = get_sport("cbb").schema
    games = pd.DataFrame({"game_id": [1, 2, 3, 4], "team": ["A", "A", "B", "Non-D1"],
                          "opp_team": ["B", "B", "C", "A"], "season": [2025, 2026, 2026, 2026]})
    pd.DataFrame({"game_id": ["1", "2"], "team": ["A", "A"], "season": 2025,
                  **{f"_pbp_{i}": 1.0 for i in range(32)}}).to_parquet(tmp_path / "pbp.parquet")
    cov = check.coverage(games, schema, [str(tmp_path / "pbp.parquet")], [], "cbb", [2025, 2026])
    # game 1 (2025) found; of 2026's two Division I games only game 2 is
    assert cov[str(tmp_path / "pbp.parquet")] == {2025: 1.0, 2026: 0.5}


@pytest.mark.parametrize("gender", ["cbb", "wcbb"])
def test_experiment_configs_build(gender):
    exp = yaml.safe_load((ROOT / "experiments" / gender / "wager.yaml").read_text())
    cfgs = exp["configs"]
    assert exp["seasons"] == [2022, 2023, 2024, 2025, 2026] and exp["baseline"] in cfgs
    arms = {"prod", "pbp32", "pbp32_pvc", "prod_pvc", "prod_pvc_prev"}
    for arm in arms:
        # four seeds each, as their own jobs: seeds 0-3 are what ensemble=4 would train
        assert [cfgs[f"{arm}_s{k}"]["fit"]["seed"] for k in range(4)] == [0, 1, 2, 3]
        assert len({yaml.dump(cfgs[f"{arm}_s{k}"].get("model")) for k in range(4)}) == 1
    assert cfgs["prod_pvc_w6_s0"]["model"]["additive_init"] == 0.6
    for name, cfg in cfgs.items():
        model = dict(cfg.get("model") or {})
        if "pbp" in model:
            assert model["pbp"].format(season=2024) == "runs/foundation/fold2024/pbp_32.parquet"
        if "lineup" in model:
            assert model["lineup"].startswith(f"player_values/{gender}") and model["lineup_set"] == "pvc"
            assert model["additive_init"] in (0.2, 0.6)
        default_model(gender, **model)       # every arm's options are ones the model accepts
        assert set(cfg) <= {"model", "fit", "max_epochs", "patience"}, name


def test_ensemble_predictions_average_members(tmp_path):
    ens = _script("ensemble_predictions")
    base = pd.DataFrame({"game_id": [1, 2, 3], "team": ["A", "B", "C"], "pred_margin": [1.0, 2.0, 3.0],
                         "win_prob": [0.5, 0.6, 0.7], "cover_prob": [0.4, 0.5, 0.6], "pred_total": [140.0, 150, 160],
                         "over_prob": [0.5, 0.5, 0.5], "result": [1, -1, 1]})
    dirs = []
    for k in range(4):
        d = tmp_path / f"m{k}"
        d.mkdir()
        m = base.copy()
        for c in ("pred_margin", "win_prob", "cover_prob", "pred_total", "over_prob"):
            m[c] = m[c] + k * 0.1
        # member 3 lacks game 3: its row drops from the average (no prediction)
        (m if k < 3 else m.iloc[:2]).to_csv(d / "2025.csv", index=False)
        dirs.append(d)
    assert ens.average(tmp_path / "e4", dirs) == ["2025.csv"]
    out = pd.read_csv(tmp_path / "e4" / "2025.csv")
    assert out.game_id.tolist() == [1, 2]
    assert out.pred_margin.tolist() == pytest.approx([1.15, 2.15])
    assert out.win_prob.tolist() == pytest.approx([0.65, 0.75])
    assert out.result.tolist() == [1, -1]


def test_wcbb_player_value_terms_follow_the_environment(tmp_path, monkeypatch):
    import importlib

    import runitback.wager.wcbb as wcbb

    monkeypatch.delenv("RUNITBACK_WCBB_PLAYER_VALUES", raising=False)
    try:
        importlib.reload(wcbb)
        assert "lineup" not in wcbb.SPORT.transformer          # off unless a pipeline sets it
        monkeypatch.setenv("RUNITBACK_WCBB_PLAYER_VALUES", str(tmp_path) + "/")
        t = importlib.reload(wcbb).SPORT.transformer
        assert t["lineup"] == f"{tmp_path}/pv.parquet" and t["lineup_set"] == "pvc"
        assert t["ensemble"] == 4 and t["additive_init"] == 0.2
        # the sport's settings build the model the experiment's prod_pvc arm did
        head = default_model("wcbb", **{k: v for k, v in t.items() if k not in ("history",)}).head
        assert softplus(head.add_w).tolist() == pytest.approx([0.2, 0.2], abs=1e-6)
        assert len(head.additive_pairs) == 6
    finally:
        monkeypatch.delenv("RUNITBACK_WCBB_PLAYER_VALUES", raising=False)
        importlib.reload(wcbb)
