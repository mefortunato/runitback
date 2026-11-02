from runitback.nba.data.ratings import ELO_MEAN, NEUTRAL, RatingsTable, compute_ratings

from .conftest import toy_games


def test_ratings_are_pre_game_and_move_with_results():
    r = compute_ratings(toy_games())
    g1 = r[r.game_id == "g1"].set_index("team")
    assert g1.at["BOS", "pre_elo"] == ELO_MEAN and g1.at["BOS", "pre_off_rtg"] == 0
    assert g1.at["BOS", "post_elo"] > ELO_MEAN > g1.at["NYK", "post_elo"]
    # BOS scored more per possession than NYK: better offense, and allowed fewer: better defense
    assert g1.at["BOS", "post_off_rtg"] > 0 and g1.at["BOS", "post_off_rtg"] > g1.at["NYK", "post_off_rtg"]
    assert g1.at["BOS", "post_def_rtg"] < g1.at["NYK", "post_def_rtg"]
    g2 = r[r.game_id == "g2"].set_index("team")
    assert g2.at["BOS", "pre_elo"] == g1.at["BOS", "post_elo"]


def test_latest_lookup_regresses_into_a_new_season():
    t = RatingsTable(compute_ratings(toy_games()))
    end_2025 = t.latest("BOS", 2025)
    start_2026 = t.latest("BOS", 2026, before="2025-10-01")
    assert abs(start_2026["elo"] - ELO_MEAN) < abs(end_2025["elo"] - ELO_MEAN)
    assert t.latest("BOS", 2025, before="2024-10-01") == NEUTRAL
    assert t.latest("LAL", 2026) == NEUTRAL
