"""Simulate a regular season and playoffs, optionally exporting training data."""
from __future__ import annotations

import hashlib
import json
import random
import uuid
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pandas as pd

from ..data.ratings import NEUTRAL, RATING_COLS
from ..engine.rules import AWAY, HOME, GameState, new_game
from ..vocab import TEAM_ALIASES, TEAMS
from .sampler import Simulator

DIVISIONS = {
	"AFC": {
		"East": ("BUF", "MIA", "NE", "NYJ"),
		"North": ("BAL", "CIN", "CLE", "PIT"),
		"South": ("HOU", "IND", "JAX", "TEN"),
		"West": ("DEN", "KC", "LAC", "LV"),
	},
	"NFC": {
		"East": ("DAL", "NYG", "PHI", "WAS"),
		"North": ("CHI", "DET", "GB", "MIN"),
		"South": ("ATL", "CAR", "NO", "TB"),
		"West": ("ARI", "LA", "SEA", "SF"),
	},
}
TEAM_DIVISION = {team: (conference, division)
				 for conference, divisions in DIVISIONS.items()
				 for division, teams in divisions.items() for team in teams}
ProgressCallback = Callable[[str, int, int], None]


def normalize_schedule(schedule: Sequence[Mapping] | pd.DataFrame, season: int) -> list[dict]:
	"""Normalize and validate a regular-season schedule with all 32 NFL teams."""
	if not 1999 <= season <= 2040:
		raise ValueError("season must be between 1999 and 2040")
	rows = schedule.to_dict("records") if isinstance(schedule, pd.DataFrame) else list(schedule)
	if not rows:
		raise ValueError("schedule contains no games")

	fixtures = []
	seen_by_week: set[tuple[int, str]] = set()
	participating_teams: set[str] = set()
	for index, row in enumerate(rows, start=1):
		if "season" in row and int(row["season"]) != season:
			raise ValueError(f"schedule row {index} season must equal {season}")
		try:
			week_value = float(row["week"])
		except (KeyError, TypeError, ValueError):
			raise ValueError(f"schedule row {index} has an invalid or missing week") from None
		if not week_value.is_integer() or not 1 <= week_value <= 18:
			raise ValueError(f"schedule row {index} week must be an integer from 1 to 18")
		week = int(week_value)

		teams = []
		for field in ("home", "home_team", "away", "away_team"):
			if field in row:
				value = row[field]
				if pd.isna(value):
					continue
				team = TEAM_ALIASES.get(str(value).strip().upper(), str(value).strip().upper())
				teams.append((field, team))
		values = {field: team for field, team in teams}
		home, away = values.get("home", values.get("home_team")), values.get("away", values.get("away_team"))
		if home not in TEAM_DIVISION or away not in TEAM_DIVISION:
			raise ValueError(f"schedule row {index} must contain valid home and away NFL teams")
		if home == away:
			raise ValueError(f"schedule row {index} has the same home and away team")
		game_type = str(row.get("game_type", row.get("season_type", "REG"))).strip().upper()
		if game_type not in ("REG", "REGULAR", "REGULAR SEASON"):
			raise ValueError(f"schedule row {index} must be a regular-season game")
		game_type = "REG"
		for team in (home, away):
			slot = (week, team)
			if slot in seen_by_week:
				raise ValueError(f"team {team} appears more than once in week {week}")
			seen_by_week.add(slot)
			participating_teams.add(team)
		fixtures.append({"week": week, "home": home, "away": away,
						 "game_type": game_type, "input_order": index - 1})

	missing = sorted(set(TEAMS) - participating_teams)
	if missing:
		raise ValueError(f"schedule must include all 32 teams; missing: {', '.join(missing)}")
	games_per_team = 16 if season <= 2020 else 17
	counts = {team: 0 for team in TEAMS}
	for fixture in fixtures:
		counts[fixture["home"]] += 1
		counts[fixture["away"]] += 1
	wrong_counts = {team: count for team, count in counts.items() if count != games_per_team}
	if wrong_counts:
		team, count = next(iter(wrong_counts.items()))
		raise ValueError(f"season {season} requires {games_per_team} games per team; {team} has {count}")
	return sorted(fixtures, key=lambda row: (row["week"], row["input_order"]))


def _prior_finish_ranks(sim: Simulator, season: int,
						games_path: Path = Path("data/nfl/processed/games.parquet")) -> tuple[dict[str, int], str]:
	previous_season = season - 1
	if games_path.is_file():
		try:
			frame = pd.read_parquet(games_path)
		except (OSError, ValueError):
			frame = pd.DataFrame()
		if not frame.empty and {"season", "week", "game_type", "home", "away", "home_final", "away_final"} <= set(frame.columns):
			previous = frame[(frame.season == previous_season) & (frame.game_type == "REG")]
			historical_games = []
			for row in previous.itertuples(index=False):
				winner = row.home if row.home_final > row.away_final else row.away if row.away_final > row.home_final else None
				historical_games.append({"stage": "Regular Season", "home": row.home, "away": row.away,
									 "home_score": int(row.home_final), "away_score": int(row.away_final),
									 "winner": winner})
			if historical_games:
				standings = _standings(historical_games, 7 if previous_season >= 2020 else 6)
				played = {team for game in historical_games for team in (game["home"], game["away"])}
				if played == set(TEAMS):
					ranks = {row["team"]: row["division_rank"]
							 for conference in standings["conferences"].values()
							 for division in conference["divisions"].values() for row in division}
					return ranks, f"processed results for {previous_season}"

	profiles = {team: sim.ratings.latest(team, season, week=1) if sim.ratings is not None else NEUTRAL
				for team in TEAMS}
	ranks = {}
	for divisions in DIVISIONS.values():
		for teams in divisions.values():
			ordered = sorted(teams, key=lambda team: (-profiles[team]["elo"], team))
			ranks.update({team: position for position, team in enumerate(ordered, start=1)})
	return ranks, f"checkpoint preseason Elo proxy for {season}"


def _schedule_weeks(fixtures: list[dict]) -> list[dict]:
	max_weeks = 16 if len(fixtures) == 256 else 17
	incident: dict[str, list[int]] = {team: [] for team in TEAMS}
	for index, fixture in enumerate(fixtures):
		incident[fixture["home"]].append(index)
		incident[fixture["away"]].append(index)
	adjacent = [set() for _ in fixtures]
	for indices in incident.values():
		for index in indices:
			adjacent[index].update(other for other in indices if other != index)
	team_games = [(fixture["home"], fixture["away"]) for fixture in fixtures]
	colors: list[int | None] = [None] * len(fixtures)
	team_colors: dict[str, set[int]] = {team: set() for team in TEAMS}
	ordered = sorted(range(len(fixtures)), key=lambda index: -len(adjacent[index]))
	for index in ordered:
		home, away = team_games[index]
		blocked = team_colors[home] | team_colors[away]
		week = next((week for week in range(max_weeks) if week not in blocked), None)
		if week is None:
			raise RuntimeError(f"could not fit the generated schedule into {max_weeks} weeks")
		colors[index] = week
		team_colors[home].add(week)
		team_colors[away].add(week)
	return [fixtures[index] | {"week": colors[index] + 1, "input_order": index}
			for index in range(len(fixtures))]


def _assign_byes(fixtures: list[dict], season: int, seed: int) -> dict[str, int]:
	games_per_team = 16 if season <= 2020 else 17
	final_week = games_per_team + 1
	bye_window = range(4, 14) if season <= 2020 else range(5, 15)
	valid_bye_weeks = set(bye_window)
	by_team: dict[str, list[tuple[int, str, int]]] = {team: [] for team in TEAMS}
	for index, game in enumerate(fixtures):
		by_team[game["home"]].append((index, game["away"], game["week"]))
		by_team[game["away"]].append((index, game["home"], game["week"]))
	remaining = set(TEAMS)
	week_counts: Counter[int] = Counter()
	selected: list[tuple[int, int]] = []
	rng = random.Random((season << 12) + seed)

	def choose_byes() -> bool:
		if not remaining:
			return True
		options = {
			team: [game for game in by_team[team]
				   if game[1] in remaining and game[2] in valid_bye_weeks and week_counts[game[2]] < 2]
			for team in remaining
		}
		team = min(remaining, key=lambda value: (len(options[value]), value))
		candidates = options[team]
		rng.shuffle(candidates)
		candidates.sort(key=lambda game: week_counts[game[2]])
		for index, opponent, week in candidates:
			remaining.remove(team)
			remaining.remove(opponent)
			week_counts[week] += 1
			selected.append((index, week))
			if choose_byes():
				return True
			selected.pop()
			week_counts[week] -= 1
			remaining.update((team, opponent))
		return False

	if not choose_byes():
		raise RuntimeError("could not assign one bye week to every team")
	bye_weeks = {}
	for index, original_week in selected:
		game = fixtures[index]
		game["bye_week"] = original_week
		game["week"] = final_week
		bye_weeks[game["home"]] = original_week
		bye_weeks[game["away"]] = original_week
	return bye_weeks


def _balance_home_games(fixtures: list[dict], season: int, seed: int) -> None:
	games_per_team = 16 if season <= 2020 else 17
	home_target = games_per_team // 2
	division_home = Counter(game["home"] for game in fixtures if game["matchup"] == "division")
	home_target_by_team = {team: home_target for team in TEAMS}
	if games_per_team % 2:
		extra_home_teams = sorted(TEAMS, key=lambda team: (TEAMS.index(team) - seed) % len(TEAMS))[:len(TEAMS) // 2]
		for team in extra_home_teams:
			home_target_by_team[team] += 1
	demand = {team: home_target_by_team[team] - division_home[team] for team in TEAMS}
	if any(value < 0 for value in demand.values()):
		raise RuntimeError("divisional home games exceed a team's home-game target")

	slots_by_team = {}
	slot_teams = []
	for team in TEAMS:
		first_slot = len(slot_teams)
		slot_teams.extend([team] * demand[team])
		slots_by_team[team] = list(range(first_slot, len(slot_teams)))
	slot_games: list[int | None] = [None] * len(slot_teams)
	nondivision_games = [index for index, game in enumerate(fixtures) if game["matchup"] != "division"]
	random.Random((season << 16) + seed).shuffle(nondivision_games)

	def place(game_index: int, visited: set[int]) -> bool:
		game = fixtures[game_index]
		for team in sorted((game["home"], game["away"]), key=lambda value: (-demand[value], value)):
			for slot in slots_by_team[team]:
				if slot in visited:
					continue
				visited.add(slot)
				other_game = slot_games[slot]
				if other_game is None or place(other_game, visited):
					slot_games[slot] = game_index
					return True
		return False

	for game_index in nondivision_games:
		if not place(game_index, set()):
			raise RuntimeError("could not balance home and away games")

	home_by_game = {game_index: slot_teams[slot]
					for slot, game_index in enumerate(slot_games) if game_index is not None}
	for game_index, home in home_by_game.items():
		game = fixtures[game_index]
		if home not in (game["home"], game["away"]):
			raise RuntimeError("home-game balancing assigned a team to the wrong matchup")
		if game["home"] != home:
			game["home"], game["away"] = game["away"], game["home"]


def generate_schedule(sim: Simulator, season: int, *, seed: int = 0,
					  games_path: Path = Path("data/nfl/processed/games.parquet")) -> dict:
	"""Build a complete schedule using NFL division rotations and same-finish matchups."""
	if not 1999 <= season <= 2040:
		raise ValueError("season must be between 1999 and 2040")
	finish_rank, rank_source = _prior_finish_ranks(sim, season, games_path)
	division_names = ("East", "North", "South", "West")
	division_index = {name: index for index, name in enumerate(division_names)}
	division_rotation = (
		(("East", "South"), ("North", "West")),
		(("East", "West"), ("North", "South")),
		(("East", "North"), ("South", "West")),
	)
	rotation_pairs = division_rotation[(season - 2024) % len(division_rotation)]
	conference_pairs = {conference: {frozenset(pair) for pair in rotation_pairs} for conference in DIVISIONS}
	fixtures = []

	def add(home: str, away: str, matchup: str) -> None:
		fixtures.append({"home": home, "away": away, "game_type": "REG", "stage": "Regular Season",
						"round": "Regular Season", "matchup": matchup})

	for conference, divisions in DIVISIONS.items():
		for division, members in divisions.items():
			ordered = sorted(members, key=lambda team: finish_rank[team])
			for index, team in enumerate(ordered):
				for opponent in ordered[index + 1:]:
					add(team, opponent, "division")
					add(opponent, team, "division")

	for conference, divisions in DIVISIONS.items():
		for pair_index, (first, second) in enumerate(rotation_pairs):
			first_teams = sorted(divisions[first], key=lambda team: finish_rank[team])
			second_teams = sorted(divisions[second], key=lambda team: finish_rank[team])
			for first_rank, team in enumerate(first_teams, start=1):
				for second_rank, opponent in enumerate(second_teams, start=1):
					home, away = (team, opponent) if (season + first_rank + second_rank + pair_index) % 2 == 0 else (opponent, team)
					add(home, away, "conference-division")

	division_offset = (2027 - season) % 4
	interconference_rotation = {
		first: division_names[(index + division_offset) % 4]
		for index, first in enumerate(division_names)
	}
	for first_index, first in enumerate(division_names):
		second = interconference_rotation[first]
		first_teams = sorted(DIVISIONS["AFC"][first], key=lambda team: finish_rank[team])
		second_teams = sorted(DIVISIONS["NFC"][second], key=lambda team: finish_rank[team])
		for first_rank, team in enumerate(first_teams, start=1):
			for second_rank, opponent in enumerate(second_teams, start=1):
				home, away = (team, opponent) if (season + first_rank + second_rank + first_index) % 2 == 0 else (opponent, team)
				add(home, away, "interconference-division")

	for conference, divisions in DIVISIONS.items():
		played_pairs = conference_pairs[conference]
		for first_index, first in enumerate(division_names):
			for second in division_names[first_index + 1:]:
				if frozenset((first, second)) in played_pairs:
					continue
				first_teams = sorted(divisions[first], key=lambda team: finish_rank[team])
				second_teams = sorted(divisions[second], key=lambda team: finish_rank[team])
				for rank, (team, opponent) in enumerate(zip(first_teams, second_teams), start=1):
					home, away = (team, opponent) if (season + rank + first_index) % 2 == 0 else (opponent, team)
					add(home, away, "same-finish-conference")

	if season >= 2021:
		position_offset = (season - 2024) % 4
		for first_index, first in enumerate(division_names):
			second = division_names[(first_index + position_offset) % 4]
			first_teams = sorted(DIVISIONS["AFC"][first], key=lambda team: finish_rank[team])
			second_teams = sorted(DIVISIONS["NFC"][second], key=lambda team: finish_rank[team])
			for rank, (team, opponent) in enumerate(zip(first_teams, second_teams), start=1):
				home, away = (team, opponent) if (season + rank + first_index) % 2 == 0 else (opponent, team)
				add(home, away, "17th-same-finish")

	expected = 272 if season >= 2021 else 256
	if len(fixtures) != expected:
		raise RuntimeError(f"schedule construction generated {len(fixtures)} games, expected {expected}")
	_balance_home_games(fixtures, season, seed)
	weekly_games = _schedule_weeks(fixtures)
	bye_weeks = _assign_byes(weekly_games, season, seed)
	return {"games": weekly_games, "bye_weeks": bye_weeks, "position_source": rank_source,
			"division_rotation": {conference: [list(pair) for pair in rotation_pairs] for conference in DIVISIONS},
			"interconference_rotation": interconference_rotation}


def _randomized_strengths(sim: Simulator, season: int, seed: int) -> dict[str, dict]:
	if sim.ratings is None:
		raise ValueError("randomized strength mode requires checkpoint ratings")
	profiles = [sim.ratings.latest(team, season, week=1) for team in TEAMS]
	random.Random(seed).shuffle(profiles)
	return dict(zip(sorted(TEAMS), profiles))


def _game_ratings(sim: Simulator, fixture: dict, season: int, strength_mode: str,
				  randomized_strengths: dict[str, dict] | None) -> tuple[dict, dict] | None:
	override = None
	if strength_mode == "neutral":
		override = {fixture["home"]: NEUTRAL, fixture["away"]: NEUTRAL}
	elif randomized_strengths is not None:
		override = {team: randomized_strengths[team] for team in (fixture["home"], fixture["away"])}
	week = fixture.get("week")
	return sim.ratings_for(fixture["home"], fixture["away"], season, week, override=override)


def _simulate_game(sim: Simulator, fixture: dict, season: int, *, game_id: str,
				   strength_mode: str, randomized_strengths: dict[str, dict] | None,
				   rng: random.Random, temperature: float,
				   inference_lock=None) -> tuple[dict, list[dict], list[dict]]:
	ratings = _game_ratings(sim, fixture, season, strength_mode, randomized_strengths)
	opening_kicker = rng.choice((HOME, AWAY))
	start = new_game(fixture["home"], fixture["away"], season, fixture["game_type"],
					 opening_kicker=opening_kicker)
	context = inference_lock if inference_lock is not None else nullcontext()
	with context:
		finals, logs = sim.rollout(start, 1, seed=rng.randrange(2**32), temperature=temperature,
								   keep_plays=1, ratings=ratings)
	if not finals or not finals[0].over:
		raise RuntimeError(f"simulation did not finish {game_id}")
	if len(logs) != 1 or not logs[0]:
		raise RuntimeError(f"simulation produced no play log for {game_id}")

	final = finals[0]
	winner = final.home if final.score[HOME] > final.score[AWAY] else final.away if final.score[AWAY] > final.score[HOME] else None
	game = {
		"game_id": game_id, "season": season, "week": fixture["week"],
		"game_type": fixture["game_type"], "stage": fixture["stage"],
		"round": fixture["round"], "home": fixture["home"], "away": fixture["away"],
		"home_score": final.score[HOME], "away_score": final.score[AWAY], "winner": winner,
		"home_seed": fixture.get("home_seed"), "away_seed": fixture.get("away_seed"),
		"site": fixture.get("site", "home"), "bye_week": fixture.get("bye_week"),
	}
	play_rows, rating_rows = [], []
	for play_index, (state, outcome) in enumerate(logs[0]):
		play_rows.append({
			"game_id": game_id, "idx": play_index, "qtr": state.qtr, "qsec": state.qsec,
			"phase": state.phase, "pos": state.pos, "down": state.down, "ytg": state.ytg,
			"yl": state.yl, "score_h": state.score[HOME], "score_a": state.score[AWAY],
			"to_h": state.timeouts[HOME], "to_a": state.timeouts[AWAY],
			"play_type": outcome.play_type, "result": outcome.result, "yards": outcome.yards,
			"elapsed": outcome.elapsed,
		})
	if ratings is not None:
		for team, side, values in zip((fixture["home"], fixture["away"]), ("home", "away"), ratings):
			rating_rows.append({"game_id": game_id, "season": season, "week": fixture["week"],
								"team": team, "side": side,
								**{f"pre_{key}": values[key] for key in RATING_COLS}})
	return game, play_rows, rating_rows


def _record_sort(teams: Sequence[str], records: dict[str, dict], games: Sequence[dict]) -> list[str]:
	pool = set(teams)

	def key(team: str) -> tuple:
		h2h_wins = h2h_games = 0.0
		for game in games:
			if game["home"] not in pool or game["away"] not in pool:
				continue
			if team not in (game["home"], game["away"]):
				continue
			h2h_games += 1
			if game["winner"] == team:
				h2h_wins += 1
			elif game["winner"] is None:
				h2h_wins += 0.5
		h2h_pct = h2h_wins / h2h_games if h2h_games else 0.0
		record = records[team]
		conf_games = record["conference_wins"] + record["conference_losses"] + record["conference_ties"]
		conf_pct = ((record["conference_wins"] + 0.5 * record["conference_ties"]) / conf_games
					if conf_games else 0.0)
		return (-record["win_pct"], -h2h_pct, -conf_pct, -record["point_diff"], -record["points_for"], team)

	return sorted(teams, key=key)


def _standings(games: Sequence[dict], playoff_seed_count: int,
				bye_weeks: Mapping[str, int] | None = None) -> dict:
	regular = [game for game in games if game["stage"] == "Regular Season"]
	records = {
		team: {"team": team, "conference": TEAM_DIVISION[team][0], "division": TEAM_DIVISION[team][1],
			   "wins": 0, "losses": 0, "ties": 0, "points_for": 0, "points_against": 0,
			   "conference_wins": 0, "conference_losses": 0, "conference_ties": 0}
		for team in TEAMS
	}
	for game in regular:
		home, away = records[game["home"]], records[game["away"]]
		home["points_for"] += game["home_score"]
		home["points_against"] += game["away_score"]
		away["points_for"] += game["away_score"]
		away["points_against"] += game["home_score"]
		conf_game = home["conference"] == away["conference"]
		if game["winner"] == game["home"]:
			home["wins"] += 1
			away["losses"] += 1
			if conf_game:
				home["conference_wins"] += 1
				away["conference_losses"] += 1
		elif game["winner"] == game["away"]:
			away["wins"] += 1
			home["losses"] += 1
			if conf_game:
				away["conference_wins"] += 1
				home["conference_losses"] += 1
		else:
			home["ties"] += 1
			away["ties"] += 1
			if conf_game:
				home["conference_ties"] += 1
				away["conference_ties"] += 1

	for record in records.values():
		record["bye_week"] = (bye_weeks or {}).get(record["team"])
		total = record["wins"] + record["losses"] + record["ties"]
		record["win_pct"] = (record["wins"] + 0.5 * record["ties"]) / total if total else 0.0
		record["point_diff"] = record["points_for"] - record["points_against"]

	conferences = {}
	seed_map = {}
	for conference, divisions in DIVISIONS.items():
		division_rows = {}
		division_winners = []
		conference_teams = [team for teams in divisions.values() for team in teams]
		for division, teams in divisions.items():
			ordered = _record_sort(teams, records, regular)
			division_rows[division] = []
			for rank, team in enumerate(ordered, start=1):
				row = records[team] | {"division_rank": rank, "seed": None}
				division_rows[division].append(row)
			division_winners.append(ordered[0])
		wildcard_teams = [team for team in conference_teams if team not in division_winners]
		wildcard_count = playoff_seed_count - len(division_winners)
		seeded = (_record_sort(division_winners, records, regular)
				  + _record_sort(wildcard_teams, records, regular)[:wildcard_count])
		seed_map[conference] = {team: seed for seed, team in enumerate(seeded, start=1)}
		for rows in division_rows.values():
			for row in rows:
				row["seed"] = seed_map[conference].get(row["team"])
				row["playoff_team"] = row["seed"] is not None
		conferences[conference] = {
			"divisions": division_rows,
			"seeds": [records[team] | {"seed": seed} for seed, team in enumerate(seeded, start=1)],
		}
	return {"conferences": conferences, "records": records, "seed_map": seed_map}


def _playoff_game(sim: Simulator, season: int, home: str, away: str, home_seed: int,
				  away_seed: int, round_name: str, week: int, game_number: int, *,
				  strength_mode: str, randomized_strengths: dict[str, dict] | None,
				  rng: random.Random, temperature: float, inference_lock=None) -> tuple[dict, list[dict], list[dict]]:
	return _simulate_game(
		sim,
		{"week": week, "game_type": "POST", "stage": round_name, "round": round_name,
		 "home": home, "away": away, "home_seed": home_seed, "away_seed": away_seed,
		 "site": "neutral" if round_name == "Super Bowl" else "home"},
		season, game_id=f"playoff-{season}-{round_name.lower().replace(' ', '-')}-{game_number:02d}",
		strength_mode=strength_mode, randomized_strengths=randomized_strengths, rng=rng,
		temperature=temperature, inference_lock=inference_lock,
	)


def simulate_season(sim: Simulator, schedule: Sequence[Mapping] | pd.DataFrame, season: int, *,
					seed: int = 0, temperature: float = 1.0, strength_mode: str = "checkpoint",
					progress: ProgressCallback | None = None, inference_lock=None,
					dataset_dir: Path | None = None, run_id: str | None = None,
					checkpoint: str | None = None, position_source: str | None = None,
					bye_weeks: Mapping[str, int] | None = None) -> dict:
	"""Simulate all scheduled regular games and the NFL's 14-team playoff bracket."""
	if temperature <= 0:
		raise ValueError("temperature must be greater than zero")
	if strength_mode not in ("checkpoint", "neutral", "randomized"):
		raise ValueError("strength_mode must be checkpoint, neutral, or randomized")
	fixtures = normalize_schedule(schedule, season)
	randomized_strengths = (_randomized_strengths(sim, season, seed)
							if strength_mode == "randomized" else None)
	rng = random.Random(seed)
	run_id = run_id or uuid.uuid4().hex[:8]
	playoff_seed_count = 7 if season >= 2020 else 6
	playoff_game_count = 13 if playoff_seed_count == 7 else 11
	total = len(fixtures) + playoff_game_count
	completed = 0
	games, plays, ratings_rows = [], [], []
	if progress:
		progress("Regular Season", completed, total)

	for game_number, fixture in enumerate(fixtures, start=1):
		game_id = f"season-{season}-{run_id}-reg-{game_number:03d}"
		game_fixture = fixture | {"stage": "Regular Season", "round": "Regular Season"}
		game, game_plays, game_ratings = _simulate_game(
			sim, game_fixture, season, game_id=game_id, strength_mode=strength_mode,
			randomized_strengths=randomized_strengths, rng=rng, temperature=temperature,
			inference_lock=inference_lock,
		)
		games.append(game)
		plays.extend(game_plays)
		ratings_rows.extend(game_ratings)
		completed += 1
		if progress:
			progress("Regular Season", completed, total)

	standings = _standings(games, playoff_seed_count, bye_weeks)
	playoff_results = []

	def play_round(round_name: str, matchups: list[tuple[str, int, str, int]], week: int) -> list[dict]:
		nonlocal completed
		round_games = []
		for index, (home, home_seed, away, away_seed) in enumerate(matchups, start=1):
			game, game_plays, game_ratings = _playoff_game(
				sim, season, home, away, home_seed, away_seed, round_name, week, index,
				strength_mode=strength_mode, randomized_strengths=randomized_strengths,
				rng=rng, temperature=temperature, inference_lock=inference_lock,
			)
			games.append(game)
			round_games.append(game)
			plays.extend(game_plays)
			ratings_rows.extend(game_ratings)
			completed += 1
			if progress:
				progress(round_name, completed, total)
		playoff_results.append({"round": round_name, "games": round_games})
		return round_games

	seed_map = standings["seed_map"]
	teams_by_seed = {conference: {seed: team for team, seed in seeds.items()}
					 for conference, seeds in seed_map.items()}
	wild_card_winners: dict[str, list[tuple[str, int]]] = {"AFC": [], "NFC": []}
	wild_card_matchups = []
	wild_card_seedings = ((2, 7), (3, 6), (4, 5)) if playoff_seed_count == 7 else ((3, 6), (4, 5))
	for conference in ("AFC", "NFC"):
		seeds = teams_by_seed[conference]
		wild_card_matchups.extend((conference, high_seed, seeds[high_seed], low_seed, seeds[low_seed])
							  for high_seed, low_seed in wild_card_seedings)
	wild_card_games = play_round("Wild Card", [
		(home, high_seed, away, low_seed)
		for _, high_seed, home, low_seed, away in wild_card_matchups
	], week=19)
	for (conference, _, _, _, _), game in zip(wild_card_matchups, wild_card_games):
		wild_card_winners[conference].append((game["winner"], seed_map[conference][game["winner"]]))

	divisional_matchups = []
	for conference in ("AFC", "NFC"):
		surviving = sorted((seed, team) for team, seed in wild_card_winners[conference])
		if playoff_seed_count == 7:
			surviving.append((1, teams_by_seed[conference][1]))
			surviving.sort()
			high, low = surviving[0], surviving[-1]
			middle_high, middle_low = surviving[1], surviving[2]
			divisional_matchups.append((conference, high[1], high[0], low[1], low[0]))
			divisional_matchups.append((conference, middle_high[1], middle_high[0], middle_low[1], middle_low[0]))
		else:
			divisional_matchups.append((conference, teams_by_seed[conference][1], 1,
										surviving[-1][1], surviving[-1][0]))
			divisional_matchups.append((conference, teams_by_seed[conference][2], 2,
									surviving[0][1], surviving[0][0]))
	divisional_games = play_round("Divisional", [
		(home, home_seed, away, away_seed)
		for _, home, home_seed, away, away_seed in divisional_matchups
	], week=20)
	conference_finalists: dict[str, list[tuple[str, int]]] = {"AFC": [], "NFC": []}
	for (conference, _, _, _, _), game in zip(divisional_matchups, divisional_games):
		conference_finalists[conference].append((game["winner"], seed_map[conference][game["winner"]]))

	conference_matchups = []
	for conference in ("AFC", "NFC"):
		finalists = sorted(conference_finalists[conference], key=lambda item: item[1])
		conference_matchups.append((conference, finalists[0], finalists[1]))
	conference_games = play_round("Conference Championship", [
		(first[0], first[1], second[0], second[1])
		for _, first, second in conference_matchups
	], week=21)
	champions = {}
	for (conference, _, _), game in zip(conference_matchups, conference_games):
		champions[conference] = (game["winner"], seed_map[conference][game["winner"]])

	afc_champion, afc_seed = champions["AFC"]
	nfc_champion, nfc_seed = champions["NFC"]
	# The NFL alternates the designated home team; the game is still at a neutral site.
	designated_home, designated_away = (
		(afc_champion, afc_seed), (nfc_champion, nfc_seed)
	) if season % 2 == 0 else (
		(nfc_champion, nfc_seed), (afc_champion, afc_seed)
	)
	super_bowl = play_round("Super Bowl", [
		(designated_home[0], designated_home[1], designated_away[0], designated_away[1])
	], week=22)[0]
	champion = super_bowl["winner"]

	result = {
		"season": season, "seed": seed, "strength_mode": strength_mode,
		"position_source": position_source,
		"bye_weeks": dict(bye_weeks or {}),
		"regular_game_count": len(fixtures), "playoff_game_count": playoff_game_count,
		"playoff_team_count": playoff_seed_count * 2,
		"games": games, "standings": standings["conferences"],
		"playoffs": playoff_results, "champion": champion,
		"super_bowl": super_bowl,
	}

	if dataset_dir is not None:
		dataset_dir.mkdir(parents=True, exist_ok=True)
		games_df, plays_df = pd.DataFrame(games), pd.DataFrame(plays)
		games_df.to_parquet(dataset_dir / "games.parquet", index=False)
		plays_df.to_parquet(dataset_dir / "plays.parquet", index=False)
		if ratings_rows:
			pd.DataFrame(ratings_rows).to_parquet(dataset_dir / "ratings.parquet", index=False)
		checkpoint_sha256 = None
		if checkpoint is not None and Path(checkpoint).is_file():
			digest = hashlib.sha256()
			with Path(checkpoint).open("rb") as checkpoint_file:
				for chunk in iter(lambda: checkpoint_file.read(1 << 20), b""):
					digest.update(chunk)
			checkpoint_sha256 = digest.hexdigest()
		metadata = {"format": "runitback-nfl-processed-v1", "synthetic": True,
					"season": season, "seed": seed, "strength_mode": strength_mode,
					"position_source": position_source,
					"bye_weeks": dict(bye_weeks or {}),
					"generator_checkpoint": checkpoint,
					"generator_checkpoint_sha256": checkpoint_sha256,
					"temperature": temperature, "games": len(games_df), "plays": len(plays_df),
					"ratings_included": bool(ratings_rows)}
		(dataset_dir / "generation.json").write_text(json.dumps(metadata, indent=2) + "\n")
		result["training_data_path"] = str(dataset_dir)
	return result
