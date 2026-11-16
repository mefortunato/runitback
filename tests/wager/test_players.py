import pandas as pd

from runitback.wager.web.players import player_ranks


def test_nfl_ranks_current_season_only(tmp_path):
    pv, proc = tmp_path / "pv", tmp_path / "proc"
    pv.mkdir(); proc.mkdir()
    pd.DataFrame({"side": "off", "player_id": ["a", "b", "c", "d"], "group": "QB", "plays": 1,
                  "value_play": [0.1, 0.3, 0.2, 0.4], "name": list("ABCD"),
                  "pts_as_starter": [6.0, 18.0, 12.0, 24.0]}).to_csv(pv / "values_latest.csv", index=False)
    rows = [(2026, w, "X", p, "QB", 60) for w in (1, 2) for p in "abc"] + [(2025, 1, "Y", "d", "QB", 500),
                                                                         (2026, 1, "Z", "a", "QB", 5)]
    pd.DataFrame(rows, columns=["season", "week", "team", "player_id", "group", "main_snaps"]).to_parquet(
        proc / "player_games.parquet")
    out = player_ranks("nfl", pv, proc)
    qb = out["positions"][0]["players"]
    assert [p["name"] for p in qb] == ["B", "C", "A"]      # retired / last-year's D is left out
    assert [p["rank"] for p in qb] == [1, 2, 3] and out["through"] == 2
    assert player_ranks("nba", pv, proc) is None
