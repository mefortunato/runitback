from conftest import toy_games

from runitback.nfl.data.ratings import ALPHA, CARRY, ELO_MEAN, RatingsTable, compute_ratings


def test_pre_game_ratings_ignore_own_result():
    g = toy_games()
    a = compute_ratings(g)
    g.loc[g.game_id == "g2", ["home_score", "home_epa"]] = [0, -0.5]
    b = compute_ratings(g)
    pre = [c for c in a.columns if c.startswith("pre_")]
    pd_a, pd_b = a[a.game_id == "g2"][pre].to_numpy(), b[b.game_id == "g2"][pre].to_numpy()
    assert (pd_a == pd_b).all()
    assert (a[a.game_id == "g1"][pre].to_numpy() == b[b.game_id == "g1"][pre].to_numpy()).all()


def test_ewma_season_regression_and_zero_sum_elo():
    r = compute_ratings(toy_games()).set_index(["game_id", "team"])
    assert abs(r.at[("g1", "KC"), "post_off_epa"] - ALPHA * 0.25) < 1e-12
    assert abs(r.at[("g3", "KC"), "pre_off_epa"] - r.at[("g2", "KC"), "post_off_epa"] * CARRY) < 1e-12
    d_kc = r.at[("g1", "KC"), "post_elo"] - r.at[("g1", "KC"), "pre_elo"]
    d_buf = r.at[("g1", "BUF"), "post_elo"] - r.at[("g1", "BUF"), "pre_elo"]
    assert d_kc > 0 and abs(d_kc + d_buf) < 1e-9


def test_latest_lookup():
    t = RatingsTable(compute_ratings(toy_games()))
    assert t.latest("KC", 2024, week=1)["elo"] == ELO_MEAN           # before any game
    after = compute_ratings(toy_games()).set_index(["game_id", "team"]).at[("g3", "KC"), "post_elo"]
    assert t.latest("KC", 2025)["elo"] == after                      # latest within season
    assert abs(t.latest("KC", 2026)["elo"] - (ELO_MEAN + (after - ELO_MEAN) * CARRY)) < 1e-9
    assert t.latest("NE", 2025)["elo"] == ELO_MEAN                   # unknown team


def test_strength_scale(tiny_sim):
    base = tiny_sim.ratings_for("KC", "BUF", 2025, strength_scale=1.0)
    same = tiny_sim.ratings_for("KC", "BUF", 2025)
    assert base == same  # default scale is 1.0 for an uncalibrated checkpoint
    wide = tiny_sim.ratings_for("KC", "BUF", 2025, strength_scale=1.5)
    gap = lambda r: abs(r[0]["elo"] - r[1]["elo"])
    assert abs(gap(wide) - 1.5 * gap(base)) < 1e-9
    assert abs(wide[0]["off_epa"] - 1.5 * base[0]["off_epa"]) < 1e-12
