"""The game pages' lineup editor (runitback.wager.web.lineups): its per-point weights must move the
prediction exactly as the model's additive lineup terms do."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from runitback.wager.models.transformer.data import (
    additive_spec,
    lineup_columns,
    lineup_context_columns,
)
from runitback.wager.models.transformer.model import PairHead
from runitback.wager.web.lineups import LineupEditor, additive_weights


def _predictor(n_members=2):
    n_before = 3
    lineup_cols = lineup_context_columns("pvc")
    cols = ["a", "b", "c", *lineup_cols]
    spec = additive_spec(n_before, "pvc")
    members = []
    for seed in range(n_members):
        torch.manual_seed(seed)
        head = PairHead(8, len(cols), out_dim=2, **spec).eval()
        with torch.no_grad():
            head.add_w.normal_()
            head.add_u.normal_()
        k = len(spec["additive_pairs"])
        head.set_additive_scale(torch.rand(k) + 0.5, torch.randn(k), torch.rand(k) + 0.5, torch.randn(k), [14.0, 13.0])
        members.append(SimpleNamespace(head=head))
    module = SimpleNamespace(members=members) if n_members > 1 else members[0]
    features = SimpleNamespace(context_columns=cols, lineup=None)
    return SimpleNamespace(model=SimpleNamespace(module=module, features=features)), cols, n_before


def test_weights_reproduce_the_additive_terms():
    pred, cols, n_before = _predictor()
    w = additive_weights(pred)
    team = len(lineup_columns("pvc"))
    rng = np.random.default_rng(0)
    ctx = torch.randn(4, len(cols))
    dh, da = rng.normal(size=9), rng.normal(size=9)
    margin = total = 0.0
    for m in pred.model.module.members:
        h = m.head
        x = ctx.clone()
        for g in range(9):   # _pv_11 + g: the change columns, in raw points (z = raw / std)
            x[:, n_before + 11 + g] += dh[g] / h.add_std[g, 0]
            x[:, n_before + team + 11 + g] += da[g] / h.add_std[g, 1]
        with torch.no_grad():
            m0, t0 = h.additive(ctx)
            m1, t1 = h.additive(x)
        margin += ((m1 - m0) * h.add_out_std[0]).numpy() / 2
        total += ((t1 - t0) * h.add_out_std[1]).numpy() / 2
    np.testing.assert_allclose(margin, np.dot(w["margin"], dh - da), atol=1e-3)
    np.testing.assert_allclose(total, np.dot(w["total"], dh + da), atol=1e-3)
    # a point of value is worth the same at every position; the total's sign follows the side
    assert np.allclose(w["margin"], w["margin"][0]) and w["margin"][0] > 0
    assert all(t > 0 for t in w["total"][:5]) and all(t < 0 for t in w["total"][5:8]) and w["total"][8] == 0


def test_editor_lists_starters_backups_and_players_ruled_out(tmp_path):
    pred, *_ = _predictor(1)
    rows = [("G1", "KC", "QB", 0, "q1", "Starter", "QB", 0.3), ("G1", "KC", "QB", 1, "q2", "Backup", "QB", 0.1),
            ("G1", "KC", "QB", -1, "q3", "Hurt", "QB", 0.4), ("G1", "KC", "K", 0, "k1", "Kicker", "K", 0.1),
            ("G1", "BUF", "QB", 0, "q4", "Other", "QB", 0.2)]
    pd.DataFrame(rows, columns=["game_id", "team", "group", "rank", "player_id", "name", "position", "value_play"]
                 ).assign(source="chart").to_parquet(tmp_path / "lineups.parquet")
    (tmp_path / "slots.json").write_text('{"QB": [61.0, 8.0, 3.0], "K": [1.7]}')
    pred.model.features.lineup = str(tmp_path / "pv.parquet")
    view = LineupEditor.load(pred).game("G1", "KC", "BUF")
    assert [p["id"] for p in view["home"]["QB"]] == ["q1", "q2", "q3"]
    assert [p["out"] for p in view["home"]["QB"]] == [False, False, True]
    assert view["slots"]["QB"] == [61.0, 8.0, 3.0] and view["slots"]["RB"] == []
    assert view["groups"][8] == {"key": "K", "label": "Kicker", "column": "ST"}
    assert LineupEditor.load(pred).game("G2", "KC", "BUF") is None


def test_matchups_without_a_game_id_have_no_lineup(tmp_path):
    # power ratings predict hypothetical matchups (no game id): their lineup context is missing
    from runitback.wager.core.sport import get_sport
    from runitback.wager.models.transformer.data import lineup_features

    schema = get_sport("nfl").schema
    path = tmp_path / "pv.parquet"
    pd.DataFrame({"game_id": ["G1"], "team": ["KC"], **{f"_pv_{i}": [1.0] for i in range(20)}}).to_parquet(path)
    rows = pd.DataFrame({schema.team: ["KC", "KC"], schema.opp: ["BUF", "BUF"]})
    out = lineup_features(rows, schema, str(path), "pvc")
    assert out["_lu_missing"].tolist() == [1.0, 1.0] and out["_opp_lu_missing"].tolist() == [1.0, 1.0]
    rows[schema.game_id] = ["G1", "G2"]
    out = lineup_features(rows, schema, str(path), "pvc")
    assert out["_lu_missing"].tolist() == [0.0, 1.0] and out["_pv_11"].iloc[0] == 1.0
