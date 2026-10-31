import json
from collections import Counter

import pandas as pd
import pytest
import torch

from runitback.nfl.data.ratings import RatingsTable, compute_ratings
from runitback.nfl.engine.rules import Outcome
from runitback.nfl.sim import season as season_module
from runitback.nfl.sim.sampler import Simulator
from runitback.nfl.sim.season import (
    DIVISIONS, TEAM_DIVISION, _standings, _Tiebreaker, generate_schedule, simulate_season,
)
from runitback.nfl.train import train
from runitback.nfl.vocab import TEAMS

from conftest import toy_games


class FixedWinnerSimulator:
    """Home team always wins 21-10 after a single play."""
    ratings = None
    strength_scale = 1.0

    def __init__(self):
        self.calls = []

    def rollout(self, start, n, **kwargs):
        self.calls.append(dict(kwargs, start=start))
        final = start.copy()
        final.score = [21, 10]
        final.over = True
        return [final], [[(start, Outcome("kickoff", "kick_received", 75, 0))]]


class RatedSimulator(FixedWinnerSimulator):
    def __init__(self, strength_scale=1.0):
        super().__init__()
        self.ratings = RatingsTable(compute_ratings(toy_games()))
        self.strength_scale = strength_scale


def full_schedule(season=2026):
    weeks = 16 if season <= 2020 else 17
    rotating = list(TEAMS)
    schedule = []
    for week in range(weeks):
        schedule.extend({"week": week + 1, "home": rotating[index], "away": rotating[-index - 1]}
                        for index in range(16))
        rotating = [rotating[0], rotating[-1], *rotating[1:-1]]
    return schedule


def opponents(schedule, team, matchup=None):
    return {game["away"] if game["home"] == team else game["home"]
            for game in schedule["games"] if team in (game["home"], game["away"])
            and (matchup is None or game["matchup"] == matchup)}


def division(conference, name):
    return set(DIVISIONS[conference][name])


@pytest.fixture
def no_data(tmp_path):
    return tmp_path / "no-data"


# ---------------------------------------------------------------- season simulation and export


def test_simulate_full_season_and_playoffs_exports_training_data(tmp_path, no_data):
    progress = []
    dataset_dir = tmp_path / "dataset"
    generated = generate_schedule(FixedWinnerSimulator(), 2026, data_root=no_data, seed=7)

    result = simulate_season(FixedWinnerSimulator(), generated["games"], 2026, seed=7,
                             progress=lambda stage, completed, total: progress.append((stage, completed, total)),
                             dataset_dir=dataset_dir, bye_weeks=generated["bye_weeks"],
                             position_source=generated["position_source"])

    assert result["regular_game_count"] == 272
    assert result["playoff_game_count"] == 13
    assert len(result["games"]) == 285
    assert [len(round_result["games"]) for round_result in result["playoffs"]] == [6, 4, 2, 1]
    assert [round_result["games"][0]["week"] for round_result in result["playoffs"]] == [19, 20, 21, 22]
    assert result["champion"] == result["super_bowl"]["winner"]
    assert len(result["standings"]["AFC"]["seeds"]) == 7
    assert len(result["standings"]["NFC"]["seeds"]) == 7
    assert result["bye_weeks"] == generated["bye_weeks"]
    assert all(team["bye_week"] == result["bye_weeks"][team["team"]]
               for conference in result["standings"].values()
               for division_rows in conference["divisions"].values() for team in division_rows)
    assert progress[-1] == ("Super Bowl", 285, 285)

    games = pd.read_parquet(dataset_dir / "games.parquet")
    plays = pd.read_parquet(dataset_dir / "plays.parquet")
    assert len(games) == len(plays) == 285
    assert set(games.game_type) == {"REG", "POST"}
    assert {"home_final", "away_final", "opening_kicker"} <= set(games.columns)
    assert games.game_id.is_unique
    metadata = json.loads((dataset_dir / "generation.json").read_text())
    assert metadata["synthetic"] and metadata["bye_weeks"] == generated["bye_weeks"]


def test_simulate_season_rejects_incomplete_schedule():
    with pytest.raises(ValueError, match="requires 17 games per team"):
        simulate_season(FixedWinnerSimulator(), full_schedule()[:16], 2026)


def test_seasons_before_the_32_team_alignment_are_rejected(no_data):
    with pytest.raises(ValueError, match="between 2002"):
        generate_schedule(FixedWinnerSimulator(), 2001, data_root=no_data)


def test_pre_2020_season_uses_12_team_playoffs_after_week_17():
    result = simulate_season(FixedWinnerSimulator(), full_schedule(2019), 2019, seed=7)

    assert result["regular_game_count"] == 256
    assert result["playoff_team_count"] == 12
    assert result["playoff_game_count"] == 11
    assert len(result["games"]) == 267
    assert [len(round_result["games"]) for round_result in result["playoffs"]] == [4, 4, 2, 1]
    assert [round_result["games"][0]["week"] for round_result in result["playoffs"]] == [18, 19, 20, 21]


def test_unfinished_games_are_resimulated():
    class FlakySimulator(FixedWinnerSimulator):
        def rollout(self, start, n, **kwargs):
            finals, logs = super().rollout(start, n, **kwargs)
            finals[0].over = len(self.calls) % 2 == 0  # every other rollout hits the play cap
            return finals, logs

    result = simulate_season(FlakySimulator(), full_schedule(), 2026)
    assert len(result["games"]) == 285

    class NeverFinishes(FixedWinnerSimulator):
        def rollout(self, start, n, **kwargs):
            finals, logs = super().rollout(start, n, **kwargs)
            finals[0].over = False
            return finals, logs

    with pytest.raises(RuntimeError, match="did not finish"):
        simulate_season(NeverFinishes(), full_schedule(), 2026)


def test_inference_context_is_passed_to_every_rollout():
    sim = FixedWinnerSimulator()
    context = object()
    simulate_season(sim, full_schedule(), 2026, inference_context=context)
    assert len(sim.calls) == 285
    assert all(call["inference_context"] is context for call in sim.calls)


def test_checkpoint_mode_holds_preseason_ratings_and_exports_them_unscaled(tmp_path):
    sim = RatedSimulator(strength_scale=2.0)
    simulate_season(sim, full_schedule(2025), 2025, strength_mode="checkpoint", dataset_dir=tmp_path)

    # The toy ratings include real 2025 week-1 results; the simulation must not see them.
    preseason = sim.ratings.latest("KC", 2025, week=1)
    assert preseason != sim.ratings.latest("KC", 2025, week=2)
    ratings = pd.read_parquet(tmp_path / "ratings.parquet")
    kc = ratings[ratings.team == "KC"]
    assert len(kc) == 17
    for key in ("off_epa", "def_epa", "elo"):
        assert kc[f"pre_{key}"].tolist() == pytest.approx([preseason[key]] * 17)
        assert kc[f"post_{key}"].tolist() == kc[f"pre_{key}"].tolist()
    # ...while the generator itself received the calibrated (scaled) ratings.
    kc_home = [call["ratings"][0] for call in sim.calls if call["start"].home == "KC"]
    assert kc_home and all(r["elo"] == pytest.approx(1505.0 + (preseason["elo"] - 1505.0) * 2.0)
                           for r in kc_home)

    exported = RatingsTable(ratings)
    assert exported.latest("KC", 2026, week=1)["elo"] == pytest.approx(
        1505.0 + (preseason["elo"] - 1505.0) * 2 / 3)


def _save_tiny_checkpoint(sim: Simulator, path):
    torch.save({"state_dict": sim.model.state_dict(), "config": sim.model.cfg.to_dict(),
                "team_seasons": sim.ts.keys, "ratings": sim.ratings.to_records()}, path)


def test_exported_season_fine_tunes_and_reloads(tmp_path, tiny_sim):
    first, second = tmp_path / "synthetic-a", tmp_path / "synthetic-b"
    simulate_season(RatedSimulator(), full_schedule(), 2026, strength_mode="randomized",
                    dataset_dir=first, run_id="a")
    simulate_season(RatedSimulator(), full_schedule(), 2026, strength_mode="randomized",
                    dataset_dir=second, run_id="b")
    base = tmp_path / "base.pt"
    _save_tiny_checkpoint(tiny_sim, base)

    ckpt = train([first, second], tmp_path / "run", val_season=None, epochs=1, init=base)
    tuned = Simulator.load(ckpt)

    assert tuned.ts.keys == tiny_sim.ts.keys
    assert tuned.ratings_for("KC", "BUF", 2027) is not None
    # Synthetic seasons train the model but never become rating lookups.
    assert set(tuned.ratings.df.game_id) == set(tiny_sim.ratings.df.game_id)


def test_training_rejects_duplicate_games_across_directories(tmp_path):
    dataset = tmp_path / "synthetic"
    simulate_season(RatedSimulator(), full_schedule(), 2026, strength_mode="neutral",
                    dataset_dir=dataset, run_id="same")
    with pytest.raises(ValueError, match="more than one training directory"):
        train([dataset, dataset], tmp_path / "run", val_season=None, epochs=1)


# ---------------------------------------------------------------- schedule formula


def test_generate_schedule_uses_nfl_opponent_formula(no_data):
    schedule = generate_schedule(FixedWinnerSimulator(), 2026, data_root=no_data)
    games = schedule["games"]
    counts = Counter(team for game in games for team in (game["home"], game["away"]))
    home_counts = Counter(game["home"] for game in games)
    division_home = Counter(game["home"] for game in games if game["matchup"] == "division")
    week_teams = {}
    for game in games:
        week_teams.setdefault(game["week"], set())
        assert game["home"] not in week_teams[game["week"]]
        assert game["away"] not in week_teams[game["week"]]
        week_teams[game["week"]].update((game["home"], game["away"]))

    assert len(games) == 272
    assert set(counts.values()) == {17}
    assert set(division_home.values()) == {3}
    assert Counter(game["matchup"] for game in games) == {
        "division": 96, "conference-division": 64, "interconference-division": 64,
        "same-finish-conference": 32, "17th-same-finish": 16}
    # Even season: the NFC hosts the 17th game, so every NFC team has nine home games.
    assert {team for team, count in home_counts.items() if count == 9} == set(
        team for team in TEAMS if TEAM_DIVISION[team][0] == "NFC")
    assert set(home_counts.values()) == {8, 9}
    assert schedule["position_source"].startswith("checkpoint preseason Elo proxy")


@pytest.mark.parametrize("season", [2019, 2024, 2025])
def test_every_team_hosts_half_of_each_opponent_group(season, no_data):
    games = generate_schedule(FixedWinnerSimulator(), season, data_root=no_data)["games"]
    for team in TEAMS:
        for matchup, hosted in (("conference-division", 2), ("interconference-division", 2),
                                ("same-finish-conference", 1)):
            own = [game for game in games if game["matchup"] == matchup and team in (game["home"], game["away"])]
            assert sum(game["home"] == team for game in own) == hosted, (team, matchup)


@pytest.mark.parametrize("season,team,full_divisions,seventeenth", [
    (2022, "BUF", [("AFC", "North"), ("NFC", "North")], ("NFC", "West")),
    (2022, "BAL", [("AFC", "East"), ("NFC", "South")], ("NFC", "East")),
    (2023, "BUF", [("AFC", "West"), ("NFC", "East")], ("NFC", "South")),
    (2023, "KC", [("AFC", "East"), ("NFC", "North")], ("NFC", "East")),
    (2024, "BUF", [("AFC", "South"), ("NFC", "West")], ("NFC", "North")),
    (2024, "BAL", [("AFC", "West"), ("NFC", "East")], ("NFC", "South")),
    (2024, "JAX", [("AFC", "East"), ("NFC", "North")], ("NFC", "East")),
    (2024, "KC", [("AFC", "North"), ("NFC", "South")], ("NFC", "West")),
    (2025, "BUF", [("AFC", "North"), ("NFC", "South")], ("NFC", "East")),
    (2025, "BAL", [("AFC", "East"), ("NFC", "North")], ("NFC", "West")),
    (2025, "JAX", [("AFC", "West"), ("NFC", "West")], ("NFC", "South")),
    (2025, "KC", [("AFC", "South"), ("NFC", "East")], ("NFC", "North")),
])
def test_rotations_match_real_nfl_schedules(season, team, full_divisions, seventeenth, no_data):
    schedule = generate_schedule(FixedWinnerSimulator(), season, data_root=no_data)
    for conference, name in full_divisions:
        assert division(conference, name) <= opponents(schedule, team), (conference, name)
    assert opponents(schedule, team, "17th-same-finish") <= division(*seventeenth)


def test_seventeenth_game_is_against_the_division_played_in_full_two_years_later(no_data):
    for season in range(2021, 2030):
        now = generate_schedule(FixedWinnerSimulator(), season, data_root=no_data)
        later = generate_schedule(FixedWinnerSimulator(), season + 2, data_root=no_data)
        for game in now["games"]:
            if game["matchup"] != "17th-same-finish":
                continue
            afc, nfc = sorted((game["home"], game["away"]), key=lambda team: TEAM_DIVISION[team][0])
            assert later["interconference_rotation"][TEAM_DIVISION[afc][1]] == TEAM_DIVISION[nfc][1]
            # The AFC hosts in odd seasons.
            assert (game["home"] == afc) == (season % 2 == 1)


@pytest.mark.parametrize("season,window", [(2019, range(4, 14)), (2026, range(5, 15))])
def test_byes_fall_in_window_and_final_week_is_all_division_games(season, window, no_data):
    schedule = generate_schedule(FixedWinnerSimulator(), season, data_root=no_data, seed=23)
    final_week = 17 if season <= 2020 else 18

    assert len(schedule["bye_weeks"]) == 32
    assert set(schedule["bye_weeks"].values()) <= set(window)
    assert max(Counter(schedule["bye_weeks"].values()).values()) <= 4
    assert all(game["week"] <= final_week for game in schedule["games"])
    assert all(schedule["bye_weeks"][game["home"]] != game["week"]
               and schedule["bye_weeks"][game["away"]] != game["week"]
               for game in schedule["games"])
    last = [game for game in schedule["games"] if game["week"] == final_week]
    assert len(last) == 16 and all(game["matchup"] == "division" for game in last)


def test_seed_changes_the_weekly_layout(no_data):
    first = generate_schedule(FixedWinnerSimulator(), 2026, data_root=no_data, seed=1)["games"]
    second = generate_schedule(FixedWinnerSimulator(), 2026, data_root=no_data, seed=2)["games"]
    layout = lambda games: {(game["home"], game["away"], game["week"]) for game in games}
    assert layout(first) != layout(second)


def test_conference_division_rotation_cycles_every_three_years(no_data):
    pairings = []
    for season in range(2024, 2028):
        schedule = generate_schedule(FixedWinnerSimulator(), season, data_root=no_data)
        pairings.append(tuple(tuple(pair) for pair in schedule["division_rotation"]["AFC"]))

    assert len(set(pairings[:3])) == 3
    assert pairings[0] == pairings[3]


def test_interconference_division_rotation_cycles_every_four_years(no_data):
    rotations = []
    for season in range(2024, 2029):
        schedule = generate_schedule(FixedWinnerSimulator(), season, data_root=no_data)
        rotations.append(tuple(schedule["interconference_rotation"].items()))

    assert len(set(rotations[:4])) == 4
    assert rotations[0] == rotations[4]


# ---------------------------------------------------------------- previous-season finish


def _results(schedule, season, home_wins=lambda game: True):
    return [{"game_id": f"{season}-{index}", "season": season, "game_type": "REG", "week": game["week"],
             "home": game["home"], "away": game["away"],
             "home_final": 20 if home_wins(game) else 10, "away_final": 10 if home_wins(game) else 20}
            for index, game in enumerate(schedule)]


def test_finish_positions_ignore_an_incomplete_previous_season(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    partial = [game for game in full_schedule(2026) if game["week"] <= 8]
    pd.DataFrame(_results(partial, 2026)).to_parquet(processed / "games.parquet")

    schedule = generate_schedule(FixedWinnerSimulator(), 2027, data_root=tmp_path)
    assert schedule["position_source"].startswith("checkpoint preseason Elo proxy")


def test_finish_positions_use_complete_processed_or_raw_results(tmp_path):
    processed, raw = tmp_path / "processed", tmp_path / "raw"
    processed.mkdir()
    raw.mkdir()
    complete = _results(full_schedule(2025), 2025)
    # Processed data dropped one game (replay check), so it is incomplete...
    pd.DataFrame(complete[1:]).to_parquet(processed / "games.parquet")
    # ...but the raw play-by-play has every game (several rows per game, as in nflverse).
    raw_rows = [{"game_id": game["game_id"], "season_type": "REG", "home_team": game["home"],
                 "away_team": game["away"], "home_score": game["home_final"], "away_score": game["away_final"]}
                for game in complete for _ in range(3)]
    pd.DataFrame(raw_rows).to_parquet(raw / "pbp_2025.parquet")

    schedule = generate_schedule(FixedWinnerSimulator(), 2026, data_root=tmp_path)
    assert schedule["position_source"] == "play-by-play results for 2025"

    pd.DataFrame(complete).to_parquet(processed / "games.parquet")
    schedule = generate_schedule(FixedWinnerSimulator(), 2026, data_root=tmp_path)
    assert schedule["position_source"] == "processed results for 2025"


# ---------------------------------------------------------------- tiebreakers


def _game(home, away, home_score, away_score):
    winner = home if home_score > away_score else away if away_score > home_score else None
    return {"stage": "Regular Season", "home": home, "away": away, "home_score": home_score,
            "away_score": away_score, "winner": winner}


def test_division_tie_goes_to_head_to_head_before_division_record():
    games = [
        _game("BUF", "MIA", 20, 17), _game("MIA", "BUF", 10, 13),  # BUF sweeps MIA
        _game("NE", "BUF", 24, 0), _game("BUF", "NE", 0, 24),
        _game("BUF", "NYJ", 21, 3), _game("NYJ", "BUF", 21, 3),     # BUF 3-3 in the division
        _game("MIA", "NE", 30, 0), _game("NE", "MIA", 0, 30),
        _game("MIA", "NYJ", 30, 0), _game("NYJ", "MIA", 0, 30),     # MIA 4-2, with a better margin
        _game("BUF", "KC", 20, 10), _game("KC", "MIA", 20, 10),     # both finish 4-3
    ]
    east = _standings(games, 7)["conferences"]["AFC"]["divisions"]["East"]
    assert [row["team"] for row in east][:2] == ["BUF", "MIA"]


def test_wild_card_tie_keeps_only_each_divisions_top_club():
    games = [
        _game("PIT", "CLE", 20, 10),   # PIT beat CLE: PIT ranks above CLE in the North
        _game("CLE", "MIA", 20, 10),   # CLE beat MIA
        _game("MIA", "PIT", 20, 10),   # MIA beat PIT
    ]
    records = _standings(games, 7)["records"]
    breaker = _Tiebreaker(games, records, "test")
    breaker.division_order.update({"PIT": 2, "CLE": 3, "MIA": 2})
    # All three are 1-1. CLE is not PIT's division leader, so it is set aside and PIT and MIA
    # are compared head-to-head: MIA wins.
    assert breaker.best(["PIT", "CLE", "MIA"], "wildcard") == "MIA"


def test_coin_toss_is_deterministic_for_a_seed():
    breaker = _Tiebreaker([], {team: {"wins": 0, "losses": 0, "ties": 0, "win_pct": 0.0} for team in TEAMS}, "x")
    breaker.division_order.update({team: 1 for team in TEAMS})
    picks = {breaker.best(["BUF", "KC"], "wildcard") for _ in range(3)}
    assert len(picks) == 1


def test_season_module_exposes_era_helpers():
    assert season_module.games_per_team(2020) == 16 and season_module.games_per_team(2021) == 17
    assert season_module.playoff_game_count(2019) == 11 and season_module.playoff_game_count(2020) == 13
