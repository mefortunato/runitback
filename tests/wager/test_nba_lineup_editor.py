"""N6: the NBA game pages' lineup editor. An edited rotation re-prices the game exactly as the model's
own additive lineup terms (``prod_pvc_e4``): the editor's arithmetic (slot usage x each player's offense
and defense value in his position's columns, weighted per point; web/src/predictions/lib/lineups.ts)
against ``PairHead.additive`` on the ``pv.parquet`` columns ``runitback.nba.player_values`` builds."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from runitback.nba.player_values import SLOTS, editor_lineups, side_values
from runitback.wager.models.transformer.data import additive_spec, lineup_columns, lineup_context_columns
from runitback.wager.models.transformer.model import PairHead
from runitback.wager.nba.lineup import LINEUP
from runitback.wager.web.lineups import LineupEditor, additive_weights

N = len(LINEUP.groups)    # 6
GROUP_OF = {1: "G", 2: "G", 3: "F", 4: "F", 5: "C", 6: "C", 7: "G", 8: "F", 9: "C", 10: "F", 11: "G", 12: "C",
            13: "F", 14: "G", 15: "C", 16: "F"}
OFF = {p: 0.01 * ((p * 7) % 11 - 5) for p in GROUP_OF}           # per-possession values
DEF = {p: 0.01 * ((p * 5) % 13 - 6) for p in GROUP_OF}
MULT = [60.0 - 4.5 * k for k in range(SLOTS)]
VP = {("off", p): OFF[p] for p in GROUP_OF} | {("def", p): DEF[p] for p in GROUP_OF}


def _predictor(n_members=2):
    n_before = 3
    cols = ["a", "b", "c", *lineup_context_columns("pvc", "nba")]
    spec = additive_spec(n_before, "pvc", "nba")
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
    module = SimpleNamespace(members=members)
    features = SimpleNamespace(context_columns=cols, lineup=None, sport="nba")
    return SimpleNamespace(model=SimpleNamespace(module=module, features=features)), cols, n_before


def _upcoming():
    rows = []
    for team, ids in (("BOS", range(1, 9)), ("NYK", range(9, 17))):
        for rank, p in enumerate(ids):
            rows.append(("G1", 2027, "2026-11-20", team, p, GROUP_OF[p], "available", None, 30.0, 30.0, rank < 5,
                         rank, 10, "live"))
    # a star ruled out by the injury report
    rows.append(("G1", 2027, "2026-11-20", "BOS", 99, "G", "out", "Knee", 30.0, 0.0, False, -1, 10, "live"))
    return pd.DataFrame(rows, columns=["game_id", "season", "game_date", "team", "player_id", "group", "status",
                                       "out_reason", "exp_minutes", "proj_minutes", "proj_starter", "rank",
                                       "recent_games", "source"])


def _editor(tmp_path):
    pred, cols, n_before = _predictor()
    up = _upcoming()
    rot = up[up.status == "available"]
    players = pd.DataFrame({"player_id": [*GROUP_OF, 99], "name": [f"P{p}" for p in [*GROUP_OF, 99]]})
    vals = {}
    from runitback.nba.player_values import day_number, week_start
    vals[week_start(day_number("2026-11-20"))] = VP | {("off", 99): 0.2, ("def", 99): 0.1}
    lineups = editor_lineups(up, rot.assign(rank=rot["rank"]), vals, GROUP_OF | {99: "G"}, players)
    lineups.to_parquet(tmp_path / "lineups.parquet")
    (tmp_path / "slots.json").write_text('{"2027": %s, "ROT": %s}' % (MULT, MULT))
    pred.model.features.lineup = str(tmp_path / "pv.parquet")
    return pred, cols, n_before, LineupEditor.load(pred)


def _columns(rotation):
    """The model's ``_pv_0 .. _pv_5`` for a rotation of ``(rank, player_id)`` (runitback.nba.player_values)."""
    side = side_values(rotation, VP | {("off", 99): 0.2, ("def", 99): 0.1}, GROUP_OF | {99: "G"}, MULT)
    return np.array([side[c] for c in LINEUP.groups])


def _model_effect(pred, cols, n_before, d_home, d_away):
    """Margin and total change under the model's additive terms when the home / away lineups' change
    columns move by ``d_home`` / ``d_away`` (the levels' differences; the recent mean is fixed)."""
    team = len(lineup_columns("pvc", "nba"))
    ctx = torch.randn(4, len(cols))
    margin = total = 0.0
    members = pred.model.module.members
    for m in members:
        h = m.head
        x = ctx.clone()
        for g in range(N):   # _pv_8 + g: the change columns, in raw points (z = raw / std)
            x[:, n_before + 8 + g] += d_home[g] / h.add_std[g, 0]
            x[:, n_before + team + 8 + g] += d_away[g] / h.add_std[g, 1]
        with torch.no_grad():
            m0, t0 = h.additive(ctx)
            m1, t1 = h.additive(x)
        margin += ((m1 - m0) * h.add_out_std[0]).numpy() / len(members)
        total += ((t1 - t0) * h.add_out_std[1]).numpy() / len(members)
    return margin, total


def _editor_effect(view, base, edited):
    """The web editor's arithmetic (lib/lineups.ts ``columnPoints`` / ``lineupEffect``) on the view."""
    cols = view["columns"]
    pts = {}
    for side, lineup in (("home", base), ("away", base)):
        pass

    def column_points(side, ids):
        players = {p["id"]: p for p in view[side]["ROT"]}
        out = np.zeros(len(cols))
        for rank, pid in enumerate(ids):
            mult = view["slots"]["ROT"][rank] if rank < len(view["slots"]["ROT"]) else 0.0
            for col, v in players[pid]["values"].items():
                out[cols.index(col)] += v * mult
        return out

    w = view["weights"]
    dh = column_points("home", edited["home"]) - column_points("home", base["home"])
    da = column_points("away", edited["away"]) - column_points("away", base["away"])
    return float(np.dot(w["margin"], dh - da)), float(np.dot(w["total"], dh + da))


def test_edited_rotation_matches_the_models_own_terms(tmp_path):
    pred, cols, n_before, editor = _editor(tmp_path)
    view = editor.game("G1", "BOS", "NYK")
    assert [g["key"] for g in view["groups"]] == ["ROT"] and view["groups"][0]["column"] == ""
    assert view["columns"] == list(LINEUP.groups) and len(view["slots"]["ROT"]) == SLOTS
    home, away = view["home"]["ROT"], view["away"]["ROT"]
    # the ranked rotation first, then the player ruled out
    assert [p["id"] for p in home[:8]] == list(range(1, 9)) and home[-1]["id"] == 99 and home[-1]["out"]
    base = {"home": [p["id"] for p in home if not p["out"]], "away": [p["id"] for p in away if not p["out"]]}
    assert base["home"] == list(range(1, 9)) and len(base["away"]) == 8

    cases = {
        "star back in at the top": ({"home": [99, *base["home"]], "away": base["away"]}),
        "two starters swapped": ({"home": [base["home"][2], base["home"][1], base["home"][0], *base["home"][3:]],
                                  "away": base["away"]}),
        "visitor's starter out": ({"home": base["home"], "away": base["away"][1:]}),
        "both change": ({"home": [99, *base["home"][1:]], "away": base["away"][2:] + base["away"][:2]}),
    }
    for name, edited in cases.items():
        mine = _editor_effect(view, base, edited)
        d_home = _columns(list(enumerate(edited["home"]))) - _columns(list(enumerate(base["home"])))
        d_away = _columns(list(enumerate(edited["away"]))) - _columns(list(enumerate(base["away"])))
        margin, total = _model_effect(pred, cols, n_before, d_home, d_away)
        np.testing.assert_allclose(mine[0], margin, atol=1e-4, err_msg=name)
        np.testing.assert_allclose(mine[1], total, atol=1e-4, err_msg=name)
        assert abs(mine[0]) > 1e-3 or name == "visitor's starter out" or abs(mine[1]) > 1e-3


def test_value_is_offense_and_defense_in_the_players_position_columns(tmp_path):
    *_, editor = _editor(tmp_path)
    view = editor.game("G1", "BOS", "NYK")
    star = view["home"]["ROT"][-1]
    assert star["values"] == {"G_OFF": 0.2, "G_DEF": 0.1}       # G: columns G_OFF / G_DEF, per possession
    p3 = next(p for p in view["home"]["ROT"] if p["id"] == 3)    # an F
    assert set(p3["values"]) == {"F_OFF", "F_DEF"} and p3["values"]["F_OFF"] == round(OFF[3], 5)
    # better offense raises the total, better defense lowers it, and a point is worth the same margin
    w = view["weights"]
    assert all(t > 0 for t in w["total"][:3]) and all(t < 0 for t in w["total"][3:])
    assert np.allclose(w["margin"], w["margin"][0])
    assert editor.game("G2", "BOS", "NYK") is None
    assert additive_weights(_predictor()[0]) is not None
