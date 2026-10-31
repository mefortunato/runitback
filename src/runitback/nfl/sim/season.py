"""Simulate a regular season and playoffs, optionally exporting training data."""
from __future__ import annotations

import hashlib
import json
import random
import uuid
from collections import Counter
from pathlib import Path
from typing import Callable, ContextManager, Mapping, Sequence

import pandas as pd

from ..data.ratings import NEUTRAL, RATING_COLS, scale_ratings
from ..engine.rules import AWAY, HOME, new_game
from ..paths import data_dir
from ..vocab import TEAM_ALIASES, TEAMS
from .sampler import Simulator

# The 32-team, eight-division alignment began in 2002; earlier seasons had 31 teams and six divisions.
FIRST_SEASON = 2002
LAST_SEASON = 2040
DIVISION_NAMES = ("East", "North", "South", "West")
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

# Same-conference divisions played in full: a three-year cycle (2020 East-West, 2021 East-South,
# 2022 East-North, ...), identical in both conferences.
INTRACONFERENCE_ROTATION = (
    (("East", "West"), ("North", "South")),
    (("East", "South"), ("North", "West")),
    (("East", "North"), ("South", "West")),
)
# AFC division -> NFC division played in full: a four-year cycle, starting with 2023.
INTERCONFERENCE_ROTATION = (
    {"East": "East", "North": "West", "South": "South", "West": "North"},  # 2023
    {"East": "West", "North": "East", "South": "North", "West": "South"},  # 2024
    {"East": "South", "North": "North", "South": "West", "West": "East"},  # 2025
    {"East": "North", "North": "South", "South": "East", "West": "West"},  # 2026
)
MAX_GAME_ATTEMPTS = 5
BYE_LAYOUT_ATTEMPTS = 20

ProgressCallback = Callable[[str, int, int], None]
InferenceContext = Callable[[], ContextManager[None]]


def games_per_team(season: int) -> int:
    return 16 if season <= 2020 else 17


def playoff_seed_count(season: int) -> int:
    return 7 if season >= 2020 else 6


def playoff_game_count(season: int) -> int:
    return 13 if playoff_seed_count(season) == 7 else 11


def first_playoff_week(season: int) -> int:
    """Wild-card week: the week after the regular season (one bye per team)."""
    return games_per_team(season) + 2


def intraconference_pairs(season: int) -> tuple[tuple[str, str], tuple[str, str]]:
    return INTRACONFERENCE_ROTATION[(season - 2020) % 3]


def interconference_rotation(season: int) -> dict[str, str]:
    return dict(INTERCONFERENCE_ROTATION[(season - 2023) % 4])


def seventeenth_game_rotation(season: int) -> dict[str, str]:
    """The 17th game is against the NFC division an AFC division will play in full two seasons later."""
    return interconference_rotation(season + 2)


def _check_season(season: int) -> None:
    if not FIRST_SEASON <= season <= LAST_SEASON:
        raise ValueError(f"season must be between {FIRST_SEASON} and {LAST_SEASON}")


def normalize_schedule(schedule: Sequence[Mapping] | pd.DataFrame, season: int) -> list[dict]:
    """Normalize and validate a regular-season schedule with all 32 NFL teams."""
    _check_season(season)
    rows = schedule.to_dict("records") if isinstance(schedule, pd.DataFrame) else list(schedule)
    if not rows:
        raise ValueError("schedule contains no games")
    last_week = games_per_team(season) + 1

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
        if not week_value.is_integer() or not 1 <= week_value <= last_week:
            raise ValueError(f"schedule row {index} week must be an integer from 1 to {last_week}")
        week = int(week_value)

        values = {}
        for field in ("home", "home_team", "away", "away_team"):
            if field in row and not pd.isna(row[field]):
                raw = str(row[field]).strip().upper()
                values[field] = TEAM_ALIASES.get(raw, raw)
        home, away = values.get("home", values.get("home_team")), values.get("away", values.get("away_team"))
        if home not in TEAM_DIVISION or away not in TEAM_DIVISION:
            raise ValueError(f"schedule row {index} must contain valid home and away NFL teams")
        if home == away:
            raise ValueError(f"schedule row {index} has the same home and away team")
        game_type = str(row.get("game_type", row.get("season_type", "REG"))).strip().upper()
        if game_type not in ("REG", "REGULAR", "REGULAR SEASON"):
            raise ValueError(f"schedule row {index} must be a regular-season game")
        for team in (home, away):
            slot = (week, team)
            if slot in seen_by_week:
                raise ValueError(f"team {team} appears more than once in week {week}")
            seen_by_week.add(slot)
            participating_teams.add(team)
        fixtures.append({"week": week, "home": home, "away": away, "game_type": "REG",
                         "input_order": index - 1})

    missing = sorted(set(TEAMS) - participating_teams)
    if missing:
        raise ValueError(f"schedule must include all 32 teams; missing: {', '.join(missing)}")
    expected = games_per_team(season)
    counts = Counter(team for fixture in fixtures for team in (fixture["home"], fixture["away"]))
    wrong_counts = {team: counts[team] for team in TEAMS if counts[team] != expected}
    if wrong_counts:
        team, count = next(iter(wrong_counts.items()))
        raise ValueError(f"season {season} requires {expected} games per team; {team} has {count}")
    return sorted(fixtures, key=lambda row: (row["week"], row["input_order"]))


# ---------------------------------------------------------------- previous-season finish


def _complete_season(games: list[dict], season: int) -> bool:
    counts = Counter(team for game in games for team in (game["home"], game["away"]))
    return set(counts) == set(TEAMS) and all(counts[team] == games_per_team(season) for team in TEAMS)


def _result_rows(frame: pd.DataFrame, home: str, away: str, home_score: str, away_score: str) -> list[dict]:
    rows = []
    for h, a, hs, as_ in zip(frame[home], frame[away], frame[home_score], frame[away_score]):
        h, a = TEAM_ALIASES.get(h, h), TEAM_ALIASES.get(a, a)
        winner = h if hs > as_ else a if as_ > hs else None
        rows.append({"stage": "Regular Season", "home": h, "away": a,
                     "home_score": int(hs), "away_score": int(as_), "winner": winner})
    return rows


def _processed_results(path: Path, season: int) -> list[dict]:
    try:
        frame = pd.read_parquet(path, columns=["season", "game_type", "home", "away", "home_final", "away_final"])
    except (OSError, ValueError, KeyError):
        return []
    frame = frame[(frame.season == season) & (frame.game_type == "REG")]
    return _result_rows(frame, "home", "away", "home_final", "away_final")


def _raw_results(path: Path, season: int) -> list[dict]:
    try:
        frame = pd.read_parquet(path, columns=["game_id", "season_type", "home_team", "away_team",
                                               "home_score", "away_score"])
    except (OSError, ValueError, KeyError):
        return []
    frame = frame[frame.season_type == "REG"].drop_duplicates("game_id").dropna(
        subset=["home_score", "away_score"])
    return _result_rows(frame, "home_team", "away_team", "home_score", "away_score")


def _prior_finish_ranks(sim: Simulator, season: int, data_root: Path | None = None) -> tuple[dict[str, int], str]:
    """Division finish (1-4) for each team in the previous season, from complete results when available.

    Processed data drops games that fail the replay check, and the latest season may still be in
    progress, so a season is only used when every team has its full slate of regular-season games.
    """
    root = data_dir() if data_root is None else data_root
    previous = season - 1
    if previous >= FIRST_SEASON:
        sources = ((f"processed results for {previous}", _processed_results, root / "processed" / "games.parquet"),
                   (f"play-by-play results for {previous}", _raw_results, root / "raw" / f"pbp_{previous}.parquet"))
        for label, load, path in sources:
            if not path.is_file():
                continue
            games = load(path, previous)
            if _complete_season(games, previous):
                standings = _standings(games, playoff_seed_count(previous), tiebreak_seed=str(previous))
                ranks = {row["team"]: row["division_rank"]
                         for conference in standings["conferences"].values()
                         for division in conference["divisions"].values() for row in division}
                return ranks, label

    profiles = {team: sim.ratings.latest(team, season, week=1) if sim.ratings is not None else NEUTRAL
                for team in TEAMS}
    ranks = {}
    for divisions in DIVISIONS.values():
        for teams in divisions.values():
            ordered = sorted(teams, key=lambda team: (-profiles[team]["elo"], team))
            ranks.update({team: position for position, team in enumerate(ordered, start=1)})
    return ranks, f"checkpoint preseason Elo proxy for {season} (no complete {previous} results)"


# ---------------------------------------------------------------- schedule construction


def _block_rounds(games: list[dict]) -> list[list[dict]]:
    """Split one division-pair block into rounds in which each of its teams plays once."""
    first = sorted({game["home"] for game in games} | {game["away"] for game in games})
    if len(games) == 4:  # same-finish block: already one game per team
        return [games]
    if all(TEAM_DIVISION[game["home"]] == TEAM_DIVISION[game["away"]] for game in games):
        # Division: three one-factors of the four teams, each played in two legs.
        p, q, r, s = first
        rounds = []
        for pairs in (((p, q), (r, s)), ((p, r), (q, s)), ((p, s), (q, r))):
            for leg in (0, 1):
                rounds.append([next(game for game in games if {game["home"], game["away"]} == {x, y}
                                    and (game["home"] == x) == (leg == 0)) for x, y in pairs])
        return rounds
    # Full 4x4 block: round k pairs the i-th team of one division with the (i+k)-th of the other.
    division = TEAM_DIVISION[first[0]]
    left = sorted(team for team in first if TEAM_DIVISION[team] == division)
    right = sorted(team for team in first if TEAM_DIVISION[team] != division)
    by_pair = {frozenset((game["home"], game["away"])): game for game in games}
    return [[by_pair[frozenset((left[i], right[(i + k) % 4]))] for i in range(4)] for k in range(4)]


def _schedule_weeks(fixtures: list[dict], weeks: int, seed: int) -> list[dict]:
    """Give every game a week so that every team plays exactly once per week.

    Games are grouped into rounds between pairs of divisions (or within one division), the
    division-level rounds are edge-coloured into weeks, and each division plays its own division
    rounds in the weeks it is not paired with another division. This spreads division games
    through the season.
    """
    blocks: dict[frozenset, list[dict]] = {}
    for fixture in fixtures:
        key = frozenset((TEAM_DIVISION[fixture["home"]], TEAM_DIVISION[fixture["away"]]))
        blocks.setdefault(key, []).append(fixture)
    rng = random.Random((weeks << 20) + seed)
    rounds = {key: _block_rounds(games) for key, games in blocks.items()}
    edges = [key for key, block_rounds in rounds.items() if len(key) == 2 for _ in block_rounds]
    for _ in range(500):
        rng.shuffle(edges)
        busy: dict[tuple[str, str], set[int]] = {division: set() for division in set(TEAM_DIVISION.values())}
        colors: list[int] = []
        for edge in edges:
            a, b = tuple(edge)
            free = [week for week in range(weeks) if week not in busy[a] and week not in busy[b]]
            if not free:
                break
            week = rng.choice(free)
            colors.append(week)
            busy[a].add(week)
            busy[b].add(week)
        else:
            break
    else:
        raise RuntimeError(f"could not fit the generated schedule into {weeks} weeks")

    week_of: dict[int, int] = {}
    pending: dict[frozenset, list[int]] = {}
    for edge, week in zip(edges, colors):
        pending.setdefault(edge, []).append(week)
    for key, block_rounds in rounds.items():
        if len(key) == 2:
            block_weeks = pending[key]
        else:
            (division,) = key
            block_weeks = [week for week in range(weeks) if week not in busy[division]]
        if len(block_weeks) != len(block_rounds):
            raise RuntimeError("schedule rounds do not match the available weeks")
        rng.shuffle(block_weeks)
        for week, games in zip(block_weeks, block_rounds):
            for game in games:
                week_of[id(game)] = week
    ordered = sorted(fixtures, key=lambda game: (week_of[id(game)], game["home"]))
    return [fixture | {"week": week_of[id(fixture)] + 1, "input_order": index}
            for index, fixture in enumerate(ordered)]


def _assign_byes(fixtures: list[dict], season: int, seed: int, division_only: bool) -> dict[str, int] | None:
    """Move one game per team from the bye window to the final week, which leaves each team one bye.

    With `division_only`, only division games may move, which makes the final week all division
    games, as the NFL has scheduled it since 2010. Returns None when no assignment exists.
    """
    final_week = games_per_team(season) + 1
    valid_bye_weeks = set(range(4, 14) if season <= 2020 else range(5, 15))
    rng = random.Random((season << 12) + seed)

    def search(allowed: Callable[[dict], bool]) -> list[tuple[int, int]] | None:
        by_team: dict[str, list[tuple[int, str, int]]] = {team: [] for team in TEAMS}
        for index, game in enumerate(fixtures):
            if allowed(game) and game["week"] in valid_bye_weeks:
                by_team[game["home"]].append((index, game["away"], game["week"]))
                by_team[game["away"]].append((index, game["home"], game["week"]))
        remaining = set(TEAMS)
        week_counts: Counter[int] = Counter()
        selected: list[tuple[int, int]] = []

        def choose() -> bool:
            if not remaining:
                return True
            options = {team: [game for game in by_team[team] if game[1] in remaining and week_counts[game[2]] < 2]
                       for team in remaining}
            team = min(remaining, key=lambda value: (len(options[value]), value))
            candidates = options[team]
            rng.shuffle(candidates)
            candidates.sort(key=lambda game: week_counts[game[2]])
            for index, opponent, week in candidates:
                remaining.difference_update((team, opponent))
                week_counts[week] += 1
                selected.append((index, week))
                if choose():
                    return True
                selected.pop()
                week_counts[week] -= 1
                remaining.update((team, opponent))
            return False

        return selected if choose() else None

    selected = search(lambda game: game["matchup"] == "division" or not division_only)
    if selected is None:
        return None
    bye_weeks = {}
    for index, original_week in selected:
        game = fixtures[index]
        game["bye_week"] = original_week
        game["week"] = final_week
        bye_weeks[game["home"]] = original_week
        bye_weeks[game["away"]] = original_week
    return bye_weeks


def generate_schedule(sim: Simulator, season: int, *, seed: int = 0, data_root: Path | None = None) -> dict:
    """Build a complete schedule from the NFL scheduling formula.

    Every team plays its division home and away, one same-conference division and one
    other-conference division in full (on the league's three- and four-year rotations), the
    same-place finishers of the two remaining same-conference divisions and, from 2021, the
    same-place finisher of the other-conference division it will play in full two years later.
    Home sites follow the formula too: 2 of 4 in each full-division block, one of the two
    same-finish conference games, and the 17th game hosted by the AFC in odd years.
    """
    _check_season(season)
    finish_rank, rank_source = _prior_finish_ranks(sim, season, data_root)
    flip = season % 2
    fixtures: list[dict] = []

    def add(home: str, away: str, matchup: str) -> None:
        fixtures.append({"home": home, "away": away, "game_type": "REG", "stage": "Regular Season",
                         "round": "Regular Season", "matchup": matchup})

    def ranked(conference: str, division: str) -> list[str]:
        return sorted(DIVISIONS[conference][division], key=lambda team: finish_rank[team])

    def full_division(first: list[str], second: list[str], matchup: str) -> None:
        # Each team hosts exactly two of its four opponents from the other division.
        for first_rank, team in enumerate(first, start=1):
            for second_rank, opponent in enumerate(second, start=1):
                home_first = (first_rank + second_rank + flip) % 2 == 0
                add(*((team, opponent) if home_first else (opponent, team)), matchup)

    for conference, divisions in DIVISIONS.items():
        for division in divisions:
            members = ranked(conference, division)
            for index, team in enumerate(members):
                for opponent in members[index + 1:]:
                    add(team, opponent, "division")
                    add(opponent, team, "division")

    pairs = intraconference_pairs(season)
    for conference in DIVISIONS:
        for first, second in pairs:
            full_division(ranked(conference, first), ranked(conference, second), "conference-division")

    rotation = interconference_rotation(season)
    for afc_division, nfc_division in rotation.items():
        full_division(ranked("AFC", afc_division), ranked("NFC", nfc_division), "interconference-division")

    # Same-finish conference games: with full pairs (A, B) and (C, D), the cycle A->C->B->D->A gives
    # every division (and so every team) one home and one away game.
    (a, b), (c, d) = pairs
    cycle = [(a, c), (c, b), (b, d), (d, a)]
    for conference in DIVISIONS:
        for host_division, visitor_division in cycle:
            if flip:
                host_division, visitor_division = visitor_division, host_division
            for host, visitor in zip(ranked(conference, host_division), ranked(conference, visitor_division)):
                add(host, visitor, "same-finish-conference")

    if season >= 2021:
        for afc_division, nfc_division in seventeenth_game_rotation(season).items():
            for afc_team, nfc_team in zip(ranked("AFC", afc_division), ranked("NFC", nfc_division)):
                add(*((afc_team, nfc_team) if season % 2 else (nfc_team, afc_team)), "17th-same-finish")

    expected = 16 * games_per_team(season)
    if len(fixtures) != expected:
        raise RuntimeError(f"schedule construction generated {len(fixtures)} games, expected {expected}")
    home_counts = Counter(game["home"] for game in fixtures)
    if not set(home_counts.values()) <= {games_per_team(season) // 2, (games_per_team(season) + 1) // 2}:
        raise RuntimeError("schedule construction produced unbalanced home games")

    for attempt in range(BYE_LAYOUT_ATTEMPTS + 1):
        # Prefer a layout whose final week is all division games; fall back to any final week.
        division_only = attempt < BYE_LAYOUT_ATTEMPTS
        weekly_games = _schedule_weeks(fixtures, games_per_team(season), seed + 7919 * attempt)
        bye_weeks = _assign_byes(weekly_games, season, seed, division_only)
        if bye_weeks is not None:
            break
    else:
        raise RuntimeError("could not assign one bye week to every team")
    return {"games": weekly_games, "bye_weeks": bye_weeks, "position_source": rank_source,
            "division_rotation": {conference: [list(pair) for pair in pairs] for conference in DIVISIONS},
            "interconference_rotation": rotation}


# ---------------------------------------------------------------- simulation


def _strength_profiles(sim: Simulator, season: int, strength_mode: str, seed: int) -> dict[str, dict] | None:
    """Unscaled pre-game ratings for every team, held fixed for the whole simulated season.

    Checkpoint mode uses each team's preseason (week 1) rating, so simulating a past season does
    not leak that season's real in-season results into the synthetic games.
    """
    if sim.ratings is None:
        if strength_mode == "randomized":
            raise ValueError("randomized strength mode requires checkpoint ratings")
        return None
    if strength_mode == "neutral":
        return {team: dict(NEUTRAL) for team in TEAMS}
    preseason = [sim.ratings.latest(team, season, week=1) for team in TEAMS]
    if strength_mode == "randomized":
        random.Random(seed).shuffle(preseason)
    return dict(zip(TEAMS, preseason))


def _simulate_game(sim: Simulator, fixture: dict, season: int, *, game_id: str,
                   profiles: dict[str, dict] | None, rng: random.Random, temperature: float,
                   inference_context: InferenceContext | None = None) -> tuple[dict, list[dict], list[dict]]:
    home, away = fixture["home"], fixture["away"]
    scale = float(getattr(sim, "strength_scale", 1.0))
    ratings = None if profiles is None else (scale_ratings(profiles[home], scale),
                                             scale_ratings(profiles[away], scale))
    for _ in range(MAX_GAME_ATTEMPTS):
        opening_kicker = rng.choice((HOME, AWAY))
        start = new_game(home, away, season, fixture["game_type"], opening_kicker=opening_kicker)
        finals, logs = sim.rollout(start, 1, seed=rng.randrange(2**32), temperature=temperature,
                                   keep_plays=1, ratings=ratings, inference_context=inference_context)
        # A rollout can hit the play/context cap before the game ends (e.g. a long playoff
        # overtime); resample it rather than discard the whole season.
        if finals and finals[0].over and len(logs) == 1 and logs[0]:
            break
    else:
        raise RuntimeError(f"simulation did not finish {game_id} in {MAX_GAME_ATTEMPTS} attempts")

    final = finals[0]
    winner = home if final.score[HOME] > final.score[AWAY] else away if final.score[AWAY] > final.score[HOME] else None
    game = {
        "game_id": game_id, "season": season, "week": fixture["week"],
        "game_type": fixture["game_type"], "stage": fixture["stage"],
        "round": fixture["round"], "home": home, "away": away,
        "home_score": final.score[HOME], "away_score": final.score[AWAY], "winner": winner,
        "home_seed": fixture.get("home_seed"), "away_seed": fixture.get("away_seed"),
        "site": fixture.get("site", "home"), "bye_week": fixture.get("bye_week"),
        "opening_kicker": opening_kicker,
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
    if profiles is not None:
        # Same schema as data/nfl/processed/ratings.parquet. Ratings are unscaled and held constant
        # for the simulated season, so pre- and post-game values are equal.
        for team, side in ((home, "home"), (away, "away")):
            values = profiles[team]
            rating_rows.append({"game_id": game_id, "season": season, "week": fixture["week"],
                                "team": team, "side": side,
                                **{f"pre_{key}": float(values[key]) for key in RATING_COLS},
                                **{f"post_{key}": float(values[key]) for key in RATING_COLS}})
    return game, play_rows, rating_rows


# ---------------------------------------------------------------- standings and tiebreakers


class _Tiebreaker:
    """NFL tie-breaking procedures for division ranking and wild-card / seeding order.

    Implements head-to-head (and the three-club sweep rule for wild cards), division record,
    common games (minimum four for wild cards), conference record, strength of victory, strength
    of schedule, net points in conference games, net points in all games and a seeded coin toss.
    The "combined ranking in points scored and allowed" steps are omitted. When a step separates
    some clubs from a tied group, the procedure restarts with the clubs still tied.
    """

    def __init__(self, games: Sequence[dict], records: dict[str, dict], tiebreak_seed: str):
        self.records = records
        self.results: dict[str, list[tuple[str, float, int]]] = {team: [] for team in TEAMS}
        for game in games:
            home, away = game["home"], game["away"]
            margin = game["home_score"] - game["away_score"]
            home_result = 1.0 if game["winner"] == home else 0.0 if game["winner"] == away else 0.5
            self.results[home].append((away, home_result, margin))
            self.results[away].append((home, 1.0 - home_result, -margin))
        order = list(TEAMS)
        random.Random(f"coin-toss-{tiebreak_seed}").shuffle(order)
        self.coin = {team: index for index, team in enumerate(order)}
        self.division_order: dict[str, int] = {}

    def pct(self, team: str, opponents: set[str] | None = None) -> float | None:
        results = [r for opp, r, _ in self.results[team] if opponents is None or opp in opponents]
        return sum(results) / len(results) if results else None

    def _combined(self, team: str, beaten_only: bool) -> float:
        points = games = 0.0
        for opponent, result, _ in self.results[team]:
            if beaten_only and result != 1.0:
                continue
            record = self.records[opponent]
            points += record["wins"] + 0.5 * record["ties"]
            games += record["wins"] + record["losses"] + record["ties"]
        return points / games if games else 0.0

    def _net(self, team: str, opponents: set[str] | None = None) -> float:
        return float(sum(margin for opp, _, margin in self.results[team]
                         if opponents is None or opp in opponents))

    # Each step returns the clubs that survive it, or None when it does not separate the group.
    def _by_value(self, group: list[str], value: Callable[[str], float | None]) -> list[str] | None:
        values = {team: value(team) for team in group}
        if any(v is None for v in values.values()):
            return None
        best = max(values.values())
        survivors = [team for team in group if values[team] >= best - 1e-9]
        return survivors if len(survivors) < len(group) else None

    def _head_to_head(self, group: list[str]) -> list[str] | None:
        others = set(group)
        return self._by_value(group, lambda team: self.pct(team, others - {team}))

    def _sweep(self, group: list[str]) -> list[str] | None:
        """Wild-card rule for three or more clubs: head-to-head applies only to a sweep."""
        def record_vs(team: str) -> dict[str, list[float]]:
            out: dict[str, list[float]] = {}
            for opp, result, _ in self.results[team]:
                if opp in group and opp != team:
                    out.setdefault(opp, []).append(result)
            return out

        for team in group:
            versus = record_vs(team)
            if len(versus) == len(group) - 1 and all(all(r == 1.0 for r in rs) for rs in versus.values()):
                return [team]
        for team in group:
            versus = record_vs(team)
            if len(versus) == len(group) - 1 and all(all(r == 0.0 for r in rs) for rs in versus.values()):
                return [other for other in group if other != team]
        return None

    def _common(self, group: list[str], minimum: int) -> list[str] | None:
        common = set.intersection(*({opp for opp, _, _ in self.results[team]} for team in group)) - set(group)
        if not common:
            return None
        counts = [sum(opp in common for opp, _, _ in self.results[team]) for team in group]
        if min(counts) < minimum:
            return None
        return self._by_value(group, lambda team: self.pct(team, common))

    def _same_group(self, team: str, index: int) -> set[str]:
        return {other for other in TEAMS if TEAM_DIVISION[other][:index + 1] == TEAM_DIVISION[team][:index + 1]
                and other != team}

    def _steps(self, group: list[str], kind: str) -> list[Callable[[list[str]], list[str] | None]]:
        division = lambda g: self._by_value(g, lambda t: self.pct(t, self._same_group(t, 1)))
        conference = lambda g: self._by_value(g, lambda t: self.pct(t, self._same_group(t, 0)))
        victory = lambda g: self._by_value(g, lambda t: self._combined(t, True))
        schedule = lambda g: self._by_value(g, lambda t: self._combined(t, False))
        conference_net = lambda g: self._by_value(g, lambda t: self._net(t, self._same_group(t, 0)))
        net = lambda g: self._by_value(g, self._net)
        if kind == "division":
            return [self._head_to_head, division, lambda g: self._common(g, 1), conference,
                    victory, schedule, conference_net, net]
        head = self._head_to_head if len(group) == 2 else self._sweep
        return [head, conference, lambda g: self._common(g, 4), victory, schedule, conference_net, net]

    def best(self, group: Sequence[str], kind: str) -> str:
        group = sorted(group)
        while len(group) > 1:
            if kind == "wildcard":
                # Only the highest-ranked club of each division stays in a wild-card tie.
                leaders: dict[str, str] = {}
                for team in sorted(group, key=lambda t: self.division_order[t]):
                    leaders.setdefault(TEAM_DIVISION[team][1], team)
                if len(leaders) < len(group):
                    group = sorted(leaders.values())
                    continue
            for step in self._steps(group, kind):
                survivors = step(group)
                if survivors is not None:
                    group = survivors
                    break
            else:
                return min(group, key=lambda team: self.coin[team])
        return group[0]

    def rank(self, teams: Sequence[str], kind: str) -> list[str]:
        remaining, ordered = list(teams), []
        while remaining:
            top = max(self.records[team]["win_pct"] for team in remaining)
            tied = [team for team in remaining if self.records[team]["win_pct"] >= top - 1e-9]
            winner = self.best(tied, kind)
            ordered.append(winner)
            remaining.remove(winner)
        return ordered


def _standings(games: Sequence[dict], seed_count: int, bye_weeks: Mapping[str, int] | None = None,
               tiebreak_seed: str = "0") -> dict:
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
            outcome = ((home, "wins"), (away, "losses"))
        elif game["winner"] == game["away"]:
            outcome = ((away, "wins"), (home, "losses"))
        else:
            outcome = ((home, "ties"), (away, "ties"))
        for record, key in outcome:
            record[key] += 1
            if conf_game:
                record[f"conference_{key}"] += 1

    for record in records.values():
        record["bye_week"] = (bye_weeks or {}).get(record["team"])
        total = record["wins"] + record["losses"] + record["ties"]
        record["win_pct"] = (record["wins"] + 0.5 * record["ties"]) / total if total else 0.0
        record["point_diff"] = record["points_for"] - record["points_against"]

    breaker = _Tiebreaker(regular, records, tiebreak_seed)
    division_orders = {}
    for conference, divisions in DIVISIONS.items():
        for division, teams in divisions.items():
            ordered = breaker.rank(teams, "division")
            division_orders[(conference, division)] = ordered
            breaker.division_order.update({team: rank for rank, team in enumerate(ordered, start=1)})

    conferences = {}
    seed_map = {}
    for conference, divisions in DIVISIONS.items():
        division_winners = [division_orders[(conference, division)][0] for division in divisions]
        wildcard_teams = [team for division in divisions for team in division_orders[(conference, division)][1:]]
        seeded = (breaker.rank(division_winners, "wildcard")
                  + breaker.rank(wildcard_teams, "wildcard")[:seed_count - len(division_winners)])
        seed_map[conference] = {team: seed for seed, team in enumerate(seeded, start=1)}
        division_rows = {}
        for division in divisions:
            division_rows[division] = [
                records[team] | {"division_rank": rank, "seed": seed_map[conference].get(team),
                                 "playoff_team": team in seed_map[conference]}
                for rank, team in enumerate(division_orders[(conference, division)], start=1)
            ]
        conferences[conference] = {
            "divisions": division_rows,
            "seeds": [records[team] | {"seed": seed} for seed, team in enumerate(seeded, start=1)],
        }
    return {"conferences": conferences, "records": records, "seed_map": seed_map}


# ---------------------------------------------------------------- season


def simulate_season(sim: Simulator, schedule: Sequence[Mapping] | pd.DataFrame, season: int, *,
                    seed: int = 0, temperature: float = 1.0, strength_mode: str = "checkpoint",
                    progress: ProgressCallback | None = None,
                    inference_context: InferenceContext | None = None,
                    dataset_dir: Path | None = None, run_id: str | None = None,
                    checkpoint: str | None = None, position_source: str | None = None,
                    bye_weeks: Mapping[str, int] | None = None) -> dict:
    """Simulate all scheduled regular-season games and the era's 12- or 14-team playoff bracket."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if strength_mode not in ("checkpoint", "neutral", "randomized"):
        raise ValueError("strength_mode must be checkpoint, neutral, or randomized")
    fixtures = normalize_schedule(schedule, season)
    profiles = _strength_profiles(sim, season, strength_mode, seed)
    rng = random.Random(seed)
    run_id = run_id or uuid.uuid4().hex[:8]
    seed_count = playoff_seed_count(season)
    total = len(fixtures) + playoff_game_count(season)
    completed = 0
    games, plays, ratings_rows = [], [], []
    if progress:
        progress("Regular Season", completed, total)

    def play(fixture: dict, game_id: str) -> dict:
        nonlocal completed
        game, game_plays, game_ratings = _simulate_game(
            sim, fixture, season, game_id=game_id, profiles=profiles, rng=rng,
            temperature=temperature, inference_context=inference_context,
        )
        games.append(game)
        plays.extend(game_plays)
        ratings_rows.extend(game_ratings)
        completed += 1
        if progress:
            progress(fixture["stage"], completed, total)
        return game

    for game_number, fixture in enumerate(fixtures, start=1):
        play(fixture | {"stage": "Regular Season", "round": "Regular Season"},
             f"season-{season}-{run_id}-reg-{game_number:03d}")

    standings = _standings(games, seed_count, bye_weeks, tiebreak_seed=f"{season}-{seed}")
    seed_map = standings["seed_map"]
    teams_by_seed = {conference: {s: team for team, s in seeds.items()} for conference, seeds in seed_map.items()}
    playoff_results = []
    week = first_playoff_week(season)

    def play_round(round_name: str, matchups: list[tuple[str, int, str, int]]) -> list[dict]:
        nonlocal week
        round_games = []
        for index, (home, home_seed, away, away_seed) in enumerate(matchups, start=1):
            fixture = {"week": week, "game_type": "POST", "stage": round_name, "round": round_name,
                       "home": home, "away": away, "home_seed": home_seed, "away_seed": away_seed,
                       "site": "neutral" if round_name == "Super Bowl" else "home"}
            game_id = f"season-{season}-{run_id}-{round_name.lower().replace(' ', '-')}-{index:02d}"
            round_games.append(play(fixture, game_id))
        playoff_results.append({"round": round_name, "games": round_games})
        week += 1
        return round_games

    def winner_seed(conference: str, game: dict) -> tuple[int, str]:
        return seed_map[conference][game["winner"]], game["winner"]

    wild_card_pairs = ((2, 7), (3, 6), (4, 5)) if seed_count == 7 else ((3, 6), (4, 5))
    wild_card = [(conference, high, low) for conference in ("AFC", "NFC") for high, low in wild_card_pairs]
    wild_card_games = play_round("Wild Card", [
        (teams_by_seed[c][high], high, teams_by_seed[c][low], low) for c, high, low in wild_card])
    survivors: dict[str, list[tuple[int, str]]] = {"AFC": [], "NFC": []}
    for (conference, _, _), game in zip(wild_card, wild_card_games):
        survivors[conference].append(winner_seed(conference, game))

    divisional = []
    for conference in ("AFC", "NFC"):
        teams = sorted(survivors[conference] + [(1, teams_by_seed[conference][1])]
                       + ([(2, teams_by_seed[conference][2])] if seed_count == 6 else []))
        # Top seed hosts the lowest remaining seed; the other two remaining seeds meet.
        divisional += [(conference, teams[0], teams[3]), (conference, teams[1], teams[2])]
    divisional_games = play_round("Divisional", [
        (high[1], high[0], low[1], low[0]) for _, high, low in divisional])
    finalists: dict[str, list[tuple[int, str]]] = {"AFC": [], "NFC": []}
    for (conference, _, _), game in zip(divisional, divisional_games):
        finalists[conference].append(winner_seed(conference, game))

    championship = [(conference, *sorted(finalists[conference])) for conference in ("AFC", "NFC")]
    championship_games = play_round("Conference Championship", [
        (high[1], high[0], low[1], low[0]) for _, high, low in championship])
    champions = {conference: winner_seed(conference, game)
                 for (conference, _, _), game in zip(championship, championship_games)}

    # The NFL alternates the designated home team (AFC after even seasons); the site is neutral.
    first, second = ("AFC", "NFC") if season % 2 == 0 else ("NFC", "AFC")
    super_bowl = play_round("Super Bowl", [
        (champions[first][1], champions[first][0], champions[second][1], champions[second][0])])[0]

    result = {
        "season": season, "seed": seed, "strength_mode": strength_mode,
        "position_source": position_source,
        "bye_weeks": dict(bye_weeks or {}),
        "regular_game_count": len(fixtures), "playoff_game_count": playoff_game_count(season),
        "playoff_team_count": seed_count * 2,
        "games": games, "standings": standings["conferences"],
        "playoffs": playoff_results, "champion": super_bowl["winner"],
        "super_bowl": super_bowl,
    }
    if dataset_dir is not None:
        _export(dataset_dir, games, plays, ratings_rows, {
            "season": season, "seed": seed, "strength_mode": strength_mode,
            "position_source": position_source, "bye_weeks": dict(bye_weeks or {}),
            "temperature": temperature, "generator_checkpoint": checkpoint,
            "generator_strength_scale": float(getattr(sim, "strength_scale", 1.0)),
        }, checkpoint)
        result["training_data_path"] = str(dataset_dir)
    return result


def _export(dataset_dir: Path, games: list[dict], plays: list[dict], ratings_rows: list[dict],
            metadata: dict, checkpoint: str | None) -> None:
    """Write games/plays/ratings in the data/nfl/processed schema, plus generation metadata."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    games_df = pd.DataFrame(games)
    games_df["home_final"] = games_df["home_score"]
    games_df["away_final"] = games_df["away_score"]
    plays_df = pd.DataFrame(plays)
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
    metadata = {"format": "runitback-nfl-processed-v1", "synthetic": True, **metadata,
                "generator_checkpoint_sha256": checkpoint_sha256,
                "games": len(games_df), "plays": len(plays_df), "ratings_included": bool(ratings_rows)}
    (dataset_dir / "generation.json").write_text(json.dumps(metadata, indent=2) + "\n")

