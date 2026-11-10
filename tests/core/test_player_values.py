"""The sport-agnostic ridge-APM fitter (runitback.core.player_values) and the NFL adapter on it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from runitback.core.player_values import (
    APMConfig,
    Events,
    codes_from_lists,
    fit_apm,
    fit_weekly,
    shrunk_rates,
)
from runitback.nfl import player_values as pv

from .pv_synthetic import BEFORES, synthetic_plays

GOLDEN = Path(__file__).parent / "fixtures" / "player_values_golden.json"


@pytest.fixture(scope="module")
def nfl_synth():
    return synthetic_plays()


def test_nfl_values_identical_to_pre_refactor(nfl_synth):
    """The golden tables are the pre-S2 ``nfl.player_values.fit_values`` on the synthetic plays
    (offense / defense / replacement / clipped outliers / kickers and punters): the adapter on the
    generic core reproduces them bit for bit, dtypes and row order included."""
    data, special, _ = nfl_synth
    want = json.loads(GOLDEN.read_text())
    for b in BEFORES:
        got = pv.fit_values(data, special, b).table
        w = want[str(b)]
        assert list(got.columns) == w["columns"]
        assert [str(d) for d in got.dtypes] == w["dtypes"]
        for c in got.columns:
            if got[c].dtype.kind == "f":
                assert np.array_equal(got[c].to_numpy(), np.array(w[c], dtype=float)), (b, c)
            else:
                assert got[c].tolist() == w[c], (b, c)


def test_weekly_fits_match_single_fits(nfl_synth):
    data, special, _ = nfl_synth
    weeks = [202003, 202101]
    for f, t in zip(pv.fit_weeks(data, special, weeks, jobs=2), weeks):
        assert f.before == t
        pd.testing.assert_frame_equal(f.table, pv.fit_values(data, special, t).table, check_exact=True)


def test_nfl_fit_recovers_effects(nfl_synth):
    data, special, effect = nfl_synth
    t = pv.fit_values(data, special, 202301).table
    qb = t[(t.side == "off") & (t.group == "QB")]
    true = qb.player_id.map(effect)
    assert np.corrcoef(qb.value_play, true)[0, 1] > 0.5   # heavy ridge, few synthetic plays
    # the rare players and the player with no group get no value of their own
    assert not t.player_id.isin(["T0QB9", "T1DB9", "T0WR4"]).any()
    assert set(t.side) == {"off", "def", "K", "P"}


# -- the generic core on a non-NFL shape -----------------------------------------------------------

def _toy_events(seed=1, n=6000, before_noise=False):
    """Basketball-like: 5 on 5, two groups, target = home effects - away effects + noise."""
    rng = np.random.default_rng(seed)
    players = [f"p{i}" for i in range(40)]
    eff = dict(zip(players, rng.normal(0, 0.3, len(players))))
    eff["rare"] = 0.0
    a_l, b_l, rows = [], [], []
    for i in range(n):
        pick = rng.choice(len(players), 10, replace=False)
        a = [players[j] for j in pick[:5]]
        b = [players[j] for j in pick[5:]]
        if i % 200 == 0:
            a[0] = "rare"
        season = 2020 + i * 3 // n
        t = season * 1000 + (i % 1000)
        y = sum(eff[p] for p in a) - sum(eff[p] for p in b) + 0.5 * (i % 2) + rng.normal(0, 1.0)
        a_l.append(a)
        b_l.append(b)
        rows.append((t, season, i % 2, y))
    frame = pd.DataFrame(rows, columns=["t", "season", "home", "pts"])
    ids = np.array(sorted(eff))
    code = {p: i for i, p in enumerate(ids)}
    group = np.array([0 if p == "rare" or int(p[1:]) % 2 else 1 for p in ids], dtype=np.int32)
    ev = Events(frame, {"o": codes_from_lists(a_l, code, 5), "d": codes_from_lists(b_l, code, 5)}, ids, group)
    return ev, eff


def _toy_cfg(**kw):
    def controls(df):
        n = len(df)
        return sp.csr_matrix(np.stack([np.ones(n), df.home.to_numpy(float)], 1) * 30.0)
    return APMConfig(groups=("G", "F"), target="pts", controls=controls, sides=(("o", 1.0), ("d", -1.0)),
                     **({"min_events": 100, "decay": 0.8, "window": 2, "lam": 50.0} | kw))


def test_core_recovers_effects_and_signs():
    ev, eff = _toy_events()
    t = fit_apm(ev, _toy_cfg(), before=2023 * 1000, season=2022)
    assert list(t.columns) == ["side", "player_id", "group", "events", "value"]
    for side in ("o", "d"):
        s = t[t.side == side]
        # a defender's eff is the points he takes away: his value is +eff (higher = better)
        assert np.corrcoef(s.value, s.player_id.map(eff))[0, 1] > 0.9
    assert "rare" not in set(t.player_id)        # under min_events: replacement level
    assert (t.events >= 100).all()


def test_core_is_leak_free():
    ev, _ = _toy_events()
    before = 2021 * 1000 + 500
    base = fit_apm(ev, _toy_cfg(), before, 2021)
    fr = ev.frame.copy()
    fr.loc[fr.t >= before, "pts"] = 1e3          # change every event at or after the cut-off
    later = Events(fr, ev.sides, ev.ids, ev.group)
    pd.testing.assert_frame_equal(base, fit_apm(later, _toy_cfg(), before, 2021), check_exact=True)
    assert fit_weekly(ev, _toy_cfg(), [before], lambda t: t // 1000)[0].equals(base)


def test_core_group_scale_and_clip():
    ev, _ = _toy_events()
    plain = fit_apm(ev, _toy_cfg(), 2023 * 1000, 2022)
    wide = fit_apm(ev, _toy_cfg(group_scale={"G": 4.0}), 2023 * 1000, 2022)
    spread = lambda t, g: t[t.group == g].value.std()
    # a wider prior for G (less shrinkage) spreads its values more, F's barely move
    assert spread(wide, "G") > spread(plain, "G")
    clipped = fit_apm(ev, _toy_cfg(target_clip=0.1), 2023 * 1000, 2022)
    assert spread(clipped, "G") < spread(plain, "G")


def test_core_event_weights():
    ev, _ = _toy_events()
    plain = fit_apm(ev, _toy_cfg(), 2023 * 1000, 2022)
    # unit weights: the same fit
    ones = Events(ev.frame.assign(w=1.0), ev.sides, ev.ids, ev.group)
    pd.testing.assert_frame_equal(plain, fit_apm(ones, _toy_cfg(weight="w"), 2023 * 1000, 2022), check_exact=False,
                                  rtol=1e-6, atol=1e-9)
    # weight 2 on every event is the same fit as each event twice, i.e. half the penalty
    twos = Events(ev.frame.assign(w=2.0), ev.sides, ev.ids, ev.group)
    a = fit_apm(twos, _toy_cfg(weight="w"), 2023 * 1000, 2022)
    b = fit_apm(ev, _toy_cfg(lam=25.0), 2023 * 1000, 2022)
    np.testing.assert_allclose(a.value, b.value, rtol=1e-4, atol=1e-6)
    # count_weight: the replacement threshold and the events column count the weight
    c = fit_apm(twos, _toy_cfg(weight="w", count_weight=True, min_events=200), 2023 * 1000, 2022)
    assert c.player_id.tolist() == a.player_id.tolist() and np.allclose(c.events, 2 * a.events)
    half = Events(ev.frame.assign(w=0.5), ev.sides, ev.ids, ev.group)
    assert fit_apm(half, _toy_cfg(weight="w", count_weight=True, min_events=10 ** 6), 2023 * 1000, 2022).empty


def test_shrunk_rates():
    df = pd.DataFrame({"season": [2020, 2021, 2021, 2021], "k": ["a", "a", "b", "b"], "y": [1.0, 3.0, 0.0, -1.0]})
    r = shrunk_rates(df, "k", "y", season=2021, shrink=2.0, decay=0.5)
    w = np.array([0.5, 1.0, 1.0, 1.0])
    mu = (w * df.y).sum() / w.sum()
    assert r.k.tolist() == ["a", "b"]
    assert r.weight.tolist() == [1.5, 2.0]
    assert np.allclose(r.value, [(0.5 * (1 - mu) + (3 - mu)) / 3.5, ((0 - mu) + (-1 - mu)) / 4.0])
