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
    assert player_ranks("mlb", pv, proc) is None


def test_nba_ranks_offense_plus_defense(tmp_path):
    pv, proc = tmp_path / "pv", tmp_path / "proc"
    pv.mkdir(); proc.mkdir()
    # 1: guard, 2: better guard on defense, 3: center, 4: last season only, 5: too few minutes
    vals = [("off", 1, "G", 0.02, 2.0), ("def", 1, "G", 0.00, 0.0),
            ("off", 2, "G", 0.01, 1.0), ("def", 2, "G", 0.02, 2.0),
            ("off", 3, "C", -0.01, -1.0), ("def", 3, "C", 0.03, 3.0),
            ("off", 4, "F", 0.05, 5.0), ("off", 5, "F", 0.05, 5.0)]
    pd.DataFrame(vals, columns=["side", "player_id", "group", "value_poss", "pts_as_starter"]).assign(
        possessions=1000, name=lambda d: "P" + d.player_id.astype(str)).to_csv(pv / "values_latest.csv", index=False)
    rows = [(f"g{d}", 2027, f"2026-11-{d:02d}", "BOS", "NYK", p, "P%d" % p, "G", False, 60.0)
            for d in (20, 23) for p in (1, 2, 3, 5)]
    rows = [r[:9] + ((3.0,) if r[5] == 5 else (r[9],)) for r in rows]
    rows += [("g0", 2026, "2026-04-01", "LAL", "BOS", 4, "P4", "F", False, 40.0),
             ("x1", 2027, "2026-11-24", "BOS", "WST", 1, "P1", "G", False, 30.0)]   # an exhibition
    pd.DataFrame(rows, columns=["game_id", "season", "game_date", "team", "opponent", "player_id", "name",
                                "group", "dnp", "minutes"]).to_parquet(proc / "player_games.parquet")
    out = player_ranks("nba", pv, proc)
    assert [g["key"] for g in out["positions"]] == ["G", "C"]
    guards = out["positions"][0]["players"]
    assert [(p["name"], p["value"], p["rate"], p["games"]) for p in guards] == [("P2", 3.0, 3.0, 2), ("P1", 2.0, 2.0, 2)]
    assert out["positions"][1]["players"][0]["value"] == 2.0
    assert out["season"] == 2027 and out["through"] == 20261123 and out["through_label"] == "Nov 23"
