"""C4: the swap-a-starter test's rotation changes and their effect on a team-game's ``_pv_*`` row."""
import numpy as np
import pandas as pd
import pytest

from runitback.college import player_values as P
from runitback.wager.cbb.lineup import LINEUP

GROUP = {1: "G", 2: "G", 3: "F", 4: "F", 5: "C", 6: "C", 7: "G", 8: "F", 100: "G", 101: "C"}
ROTATION = [(1, 34.0), (2, 30.0), (3, 32.0), (4, 28.0), (5, 31.0), (6, 25.0), (7, 20.0)]    # sums to 200


def test_swap_rotation_changes_the_groups_starter_only():
    # G starter is player 1 (34 min); the swap keeps his minutes
    best = P.swap_rotation(ROTATION, GROUP, "G", "best", ref=100)
    assert (100, 34.0) in best and all(p != 1 for p, _ in best)
    assert sorted(m for _, m in best) == sorted(m for _, m in ROTATION)
    rep = P.swap_rotation(ROTATION, GROUP, "G", "replacement")
    assert (P.REPLACEMENT_ID, 34.0) in rep
    # the C starter is player 5 (31 min), not the bench center 6
    assert (101, 31.0) in P.swap_rotation(ROTATION, GROUP, "C", "worst", ref=101)


def test_swap_rotation_out_rescales_to_the_teams_200():
    out = P.swap_rotation(ROTATION, GROUP, "F", "out")
    assert all(p != 3 for p, _ in out) and len(out) == len(ROTATION) - 1
    assert sum(m for _, m in out) == pytest.approx(200.0)
    assert max(m for _, m in out) <= 38.0 + 1e-9
    # the 34-minute guard reaches the 38-minute cap; the rest gain in proportion to what they had
    old, new = dict(ROTATION), dict(out)
    assert new[1] == pytest.approx(38.0)
    ratio = {p: new[p] / old[p] for p in new if p != 1}
    assert max(ratio.values()) - min(ratio.values()) < 1e-6 and min(ratio.values()) > 1


def test_swap_rotation_no_swap():
    assert P.swap_rotation(ROTATION[:2], GROUP, "C", "out") is None            # no center
    assert P.swap_rotation(ROTATION, GROUP, "G", "best", ref=2) is None        # already on the team
    assert P.swap_rotation(ROTATION, GROUP, "G", "best", ref=None) is None
    with pytest.raises(ValueError):
        P.swap_rotation(ROTATION, GROUP, "G", "traded")


def _values():
    # offense and defense value per possession over replacement; 100 is a star guard, 2 a weak one
    off = {1: 0.02, 2: -0.01, 3: 0.01, 4: 0.0, 5: 0.015, 6: -0.005, 7: 0.0, 8: 0.0, 100: 0.05, 101: -0.03}
    dfn = {1: 0.01, 2: -0.01, 3: 0.012, 4: 0.0, 5: 0.02, 6: 0.0, 7: 0.0, 8: 0.0, 100: 0.03, 101: -0.02}
    return {("off", p): v for p, v in off.items()} | {("def", p): v for p, v in dfn.items()}


def test_swap_delta_layout_and_direction():
    vals, rate = _values(), 70.0
    best = P.swap_delta(ROTATION, vals, GROUP, rate, "G", "best", ref=100)
    assert best.shape == (len(LINEUP.columns),) == (14,)
    # level columns G_OFF, G_DEF move by (new - old) * rate * minutes / 40; nothing else moves
    assert best[0] == pytest.approx((0.05 - 0.02) * rate * 34 / 40)
    assert best[3] == pytest.approx((0.03 - 0.01) * rate * 34 / 40)
    assert np.allclose(best[[1, 2, 4, 5]], 0)
    # the summaries are the offense / defense sums, and the changes vs recent equal the levels'
    assert best[6] == pytest.approx(best[:3].sum()) and best[7] == pytest.approx(best[3:6].sum())
    assert np.allclose(best[8:], best[:6])
    # better player up, worse down, the starter out (his value lost, the rest play more)
    worst = P.swap_delta(ROTATION, vals, GROUP, rate, "C", "worst", ref=101)
    assert worst[:6].sum() < 0 < best[:6].sum()
    rep = P.swap_delta(ROTATION, vals, GROUP, rate, "G", "replacement")
    assert rep[0] == pytest.approx(-0.02 * rate * 34 / 40)       # replacement level is value 0
    out = P.swap_delta(ROTATION, vals, GROUP, rate, "C", "out")
    assert out[2] < 0 and out[5] < 0       # a positive-value starter leaves: the C columns fall
    assert P.swap_delta(ROTATION, vals, GROUP, rate, "G", "best", ref=None) is None


def test_swap_refs_pick_regulars_by_total_value():
    rows = []
    for pid, g, off, dfn, mins in [(1, "G", 0.02, 0.01, 900), (2, "G", -0.01, -0.01, 900), (7, "G", 0.09, 0.09, 100),
                                   (100, "G", 0.05, 0.03, 700), (3, "F", 0.01, 0.012, 800), (4, "F", 0.0, -0.02, 800),
                                   (5, "C", 0.015, 0.02, 650)]:
        rows += [dict(side="off", player_id=pid, group=g, minutes=mins, value=off),
                 dict(side="def", player_id=pid, group=g, minutes=mins, value=dfn)]
    fit = pd.DataFrame(rows)
    refs = P.swap_refs(fit, 600)
    # player 7 is the best G on paper but has 100 minutes: not a regular
    assert refs[("G", "best")] == 100 and refs[("G", "worst")] == 2
    assert refs[("F", "best")] == 3 and refs[("F", "worst")] == 4
    assert refs[("C", "best")] == refs[("C", "worst")] == 5
    # only players still playing
    assert P.swap_refs(fit, 600, active={1, 2, 3})[("G", "best")] == 1


def test_write_fits_keeps_one_seasons_mondays(tmp_path):
    day = lambda y, m, d: (pd.Timestamp(y, m, d) - pd.Timestamp("1970-01-01")).days
    tab = pd.DataFrame({"side": ["off"], "player_id": [1], "group": ["G"], "minutes": [100.0], "value": [0.01]})
    fits = {day(2025, 11, 3): tab, day(2026, 1, 5): tab, day(2026, 3, 2): tab, day(2026, 11, 2): tab}
    gv = P.GameValues(pd.DataFrame(), {}, fits)
    path = P.write_fits(gv, tmp_path, 2026)
    got = pd.read_parquet(path)
    assert path.name == "fits_2026.parquet"
    assert sorted(got.before.unique()) == [day(2025, 11, 3), day(2026, 1, 5), day(2026, 3, 2)]
