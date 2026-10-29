import json

import pandas as pd
import pytest

from runitback.nfl.engine.rules import Outcome
from runitback.nfl.sim.season import generate_schedule, simulate_season
from runitback.nfl.vocab import TEAMS


class FixedWinnerSimulator:
	ratings = None

	def ratings_for(self, *args, **kwargs):
		return None

	def rollout(self, start, n, **kwargs):
		final = start.copy()
		final.score = [21, 10]
		final.over = True
		return [final], [[(start, Outcome("kickoff", "kick_received", 75, 0))]]


def full_schedule(season=2026):
	weeks = 16 if season <= 2020 else 17
	rotating = list(TEAMS)
	schedule = []
	for week in range(weeks):
		schedule.extend({"week": week + 1, "home": rotating[index], "away": rotating[-index - 1]}
						for index in range(16))
		rotating = [rotating[0], rotating[-1], *rotating[1:-1]]
	return schedule


def test_simulate_full_season_and_playoffs_exports_training_data(tmp_path):
	progress = []
	dataset_dir = tmp_path / "dataset"
	generated = generate_schedule(FixedWinnerSimulator(), 2026, games_path=tmp_path / "missing", seed=7)

	result = simulate_season(FixedWinnerSimulator(), generated["games"], 2026, seed=7,
							 progress=lambda stage, completed, total: progress.append((stage, completed, total)),
							 dataset_dir=dataset_dir, bye_weeks=generated["bye_weeks"],
							 position_source=generated["position_source"])

	assert result["regular_game_count"] == 272
	assert result["playoff_game_count"] == 13
	assert len(result["games"]) == 285
	assert [len(round_result["games"]) for round_result in result["playoffs"]] == [6, 4, 2, 1]
	assert result["champion"] == result["super_bowl"]["winner"]
	assert len(result["standings"]["AFC"]["seeds"]) == 7
	assert len(result["standings"]["NFC"]["seeds"]) == 7
	assert result["bye_weeks"] == generated["bye_weeks"]
	assert all(team["bye_week"] == result["bye_weeks"][team["team"]]
			   for conference in result["standings"].values()
			   for division in conference["divisions"].values() for team in division)
	assert progress[-1] == ("Super Bowl", 285, 285)

	games = pd.read_parquet(dataset_dir / "games.parquet")
	plays = pd.read_parquet(dataset_dir / "plays.parquet")
	assert len(games) == len(plays) == 285
	assert set(games.game_type) == {"REG", "POST"}
	metadata = json.loads((dataset_dir / "generation.json").read_text())
	assert metadata["synthetic"] and metadata["bye_weeks"] == generated["bye_weeks"]


def test_simulate_season_rejects_incomplete_schedule():
	with pytest.raises(ValueError, match="requires 17 games per team"):
		simulate_season(FixedWinnerSimulator(), full_schedule()[:16], 2026)


def test_pre_2020_season_uses_12_team_playoffs():
	result = simulate_season(FixedWinnerSimulator(), full_schedule(2019), 2019, seed=7)

	assert result["regular_game_count"] == 256
	assert result["playoff_team_count"] == 12
	assert result["playoff_game_count"] == 11
	assert len(result["games"]) == 267
	assert [len(round_result["games"]) for round_result in result["playoffs"]] == [4, 4, 2, 1]


def test_generate_schedule_uses_nfl_opponent_formula(tmp_path):
	schedule = generate_schedule(FixedWinnerSimulator(), 2026, games_path=tmp_path / "missing")
	games = schedule["games"]
	counts = {team: 0 for team in TEAMS}
	home_counts = {team: 0 for team in TEAMS}
	division_home = {team: 0 for team in TEAMS}
	division_away = {team: 0 for team in TEAMS}
	week_teams = {}
	matchup_counts = {}
	for game in games:
		counts[game["home"]] += 1
		counts[game["away"]] += 1
		home_counts[game["home"]] += 1
		week_teams.setdefault(game["week"], set())
		assert game["home"] not in week_teams[game["week"]]
		assert game["away"] not in week_teams[game["week"]]
		week_teams[game["week"]].update((game["home"], game["away"]))
		matchup_counts[game["matchup"]] = matchup_counts.get(game["matchup"], 0) + 1
		if game["matchup"] == "division":
			division_home[game["home"]] += 1
			division_away[game["away"]] += 1

	assert len(games) == 272
	assert set(counts.values()) == {17}
	assert set(home_counts.values()) <= {8, 9}
	assert set(division_home.values()) == {3}
	assert set(division_away.values()) == {3}
	assert matchup_counts == {"division": 96, "conference-division": 64,
							  "interconference-division": 64, "same-finish-conference": 32,
							  "17th-same-finish": 16}
	assert schedule["position_source"].startswith("checkpoint preseason Elo proxy")


def test_sixteen_game_era_assigns_byes_in_historical_window(tmp_path):
	schedule = generate_schedule(FixedWinnerSimulator(), 2019, games_path=tmp_path / "missing", seed=23)

	assert len(schedule["games"]) == 256
	assert len(schedule["bye_weeks"]) == 32
	assert set(schedule["bye_weeks"].values()) <= set(range(4, 14))
	assert all(game["week"] <= 17 for game in schedule["games"])
	assert all(schedule["bye_weeks"][game["home"]] != game["week"]
			   and schedule["bye_weeks"][game["away"]] != game["week"]
			   for game in schedule["games"])


def test_conference_division_rotation_cycles_every_three_years(tmp_path):
	pairings = []
	for season in range(2024, 2028):
		schedule = generate_schedule(FixedWinnerSimulator(), season, games_path=tmp_path / "missing")
		pairings.append(tuple(tuple(pair) for pair in schedule["division_rotation"]["AFC"]))

	assert len(set(pairings[:3])) == 3
	assert pairings[0] == pairings[3]


def test_interconference_division_rotation_cycles_every_four_years(tmp_path):
	rotations = []
	for season in range(2024, 2029):
		schedule = generate_schedule(FixedWinnerSimulator(), season, games_path=tmp_path / "missing")
		rotations.append(tuple(schedule["interconference_rotation"].items()))

	assert len(set(rotations[:4])) == 4
	assert rotations[0] == rotations[4]
