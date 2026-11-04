"""Simulate an NBA regular season, the play-in tournament and the best-of-seven playoffs.

Team strength is held fixed for the whole simulated season (each team's preseason rating, the
league average, or preseason ratings shuffled between teams), so the games are independent of
one another. That lets the simulator batch games that are played on different days, and lets a
playoff series sample all seven games at once and stop counting when a team reaches four wins:
with fixed ratings this is the same distribution as playing the games one after another.
"""
from __future__ import annotations

import random
from collections import Counter
from datetime import date, timedelta
from typing import Callable, ContextManager, Mapping, Sequence

from ..data.ratings import NEUTRAL, scale_ratings
from ..engine.rules import AWAY, HOME, GameState, new_game
from ..vocab import TEAMS
from .sampler import Simulator

# The 30-team, six-division alignment began with the 2004-05 season (seasons are named by the
# year they end in, as everywhere in the NBA package).
FIRST_SEASON = 2005
LAST_SEASON = 2040
FIRST_PLAY_IN_SEASON = 2021  # the play-in tournament began in 2020-21
CONFERENCES = ("East", "West")
DIVISIONS = {
    "East": {
        "Atlantic": ("BOS", "BKN", "NYK", "PHI", "TOR"),
        "Central": ("CHI", "CLE", "DET", "IND", "MIL"),
        "Southeast": ("ATL", "CHA", "MIA", "ORL", "WAS"),
    },
    "West": {
        "Northwest": ("DEN", "MIN", "OKC", "POR", "UTA"),
        "Pacific": ("GSW", "LAC", "LAL", "PHX", "SAC"),
        "Southwest": ("DAL", "HOU", "MEM", "NOP", "SAS"),
    },
}
TEAM_DIVISION = {team: (conference, division) for conference, divisions in DIVISIONS.items()
                 for division, teams in divisions.items() for team in teams}
GAMES_PER_TEAM = 82
REGULAR_SEASON_GAMES = 30 * GAMES_PER_TEAM // 2  # 1,230
PLAYOFF_ROUNDS = ("First Round", "Conference Semifinals", "Conference Finals", "NBA Finals")
SERIES_LENGTH, WINS_NEEDED = 7, 4
# Home court in a best-of-seven: the higher seed hosts games 1, 2, 5 and 7.
HIGHER_SEED_HOME = (True, True, False, False, True, False, True)
# First-round pairings by seed; the bracket is fixed after that (1/8 meets 4/5, 2/7 meets 3/6).
FIRST_ROUND = ((1, 8), (4, 5), (3, 6), (2, 7))
MAX_GAMES_PER_DAY = 9
MAX_GAME_ATTEMPTS = 5

ProgressCallback = Callable[[str, int, int], None]
InferenceContext = Callable[[], ContextManager[None]]


def has_play_in(season: int) -> bool:
    return season >= FIRST_PLAY_IN_SEASON


def play_in_game_count(season: int) -> int:
    return 6 if has_play_in(season) else 0


def total_game_slots(season: int) -> int:
    """Games the simulator samples for a season: every regular-season and play-in game, plus all
    seven possible games of each of the 15 playoff series (unused ones are discarded)."""
    return REGULAR_SEASON_GAMES + play_in_game_count(season) + 15 * SERIES_LENGTH


def _check_season(season: int) -> None:
    if not FIRST_SEASON <= season <= LAST_SEASON:
        raise ValueError(f"season must be between {FIRST_SEASON} and {LAST_SEASON}")


# ---------------------------------------------------------------- schedule


def opening_day(season: int) -> date:
    """Opening night: the fourth Tuesday of October of the year the season starts."""
    first = date(season - 1, 10, 1)
    first_tuesday = first + timedelta(days=(1 - first.weekday()) % 7)
    return first_tuesday + timedelta(weeks=3)


def generate_schedule(season: int, *, seed: int = 0) -> list[dict]:
    """An 82-game schedule from the NBA formula, laid out over the days of a season.

    Every team plays its four division rivals four times (16 games), six of the other ten
    conference teams four times and the remaining four three times (36), and each team in the
    other conference twice (30). Which conference opponents are played three times rotates by
    season (and `seed`). Every team gets 41 home games.
    """
    _check_season(season)
    rng = random.Random(f"nba-schedule-{season}-{seed}")
    fixtures: list[dict] = []

    def add(home: str, away: str, matchup: str, times: int = 1) -> None:
        fixtures.extend({"home": home, "away": away, "matchup": matchup} for _ in range(times))

    for conference, divisions in DIVISIONS.items():
        names = list(divisions)
        for teams in divisions.values():
            for i, team in enumerate(teams):
                for opponent in teams[i + 1:]:
                    add(team, opponent, "division", 2)
                    add(opponent, team, "division", 2)
        # Non-division conference games, for each of the three pairs of divisions. Team i of the
        # first division plays the second division's teams i+shift and i+shift+1 three times and
        # the other three four times; the first of the two three-game sets is hosted twice by the
        # first division's team, the second by the other team, so home games stay balanced.
        for a in range(3):
            for b in range(a + 1, 3):
                first, second = divisions[names[a]], divisions[names[b]]
                shift = rng.randrange(5)
                for i, team in enumerate(first):
                    for j, opponent in enumerate(second):
                        offset = (j - i - shift) % 5
                        if offset == 0:
                            add(team, opponent, "conference", 2)
                            add(opponent, team, "conference")
                        elif offset == 1:
                            add(team, opponent, "conference")
                            add(opponent, team, "conference", 2)
                        else:
                            add(team, opponent, "conference", 2)
                            add(opponent, team, "conference", 2)
    for east in (team for teams in DIVISIONS["East"].values() for team in teams):
        for west in (team for teams in DIVISIONS["West"].values() for team in teams):
            add(east, west, "interconference")
            add(west, east, "interconference")

    if len(fixtures) != REGULAR_SEASON_GAMES:
        raise RuntimeError(f"schedule construction generated {len(fixtures)} games, expected {REGULAR_SEASON_GAMES}")
    counts = Counter(team for game in fixtures for team in (game["home"], game["away"]))
    homes = Counter(game["home"] for game in fixtures)
    if set(counts.values()) != {GAMES_PER_TEAM} or set(homes.values()) != {GAMES_PER_TEAM // 2}:
        raise RuntimeError("schedule construction produced an unbalanced schedule")
    return _assign_days(fixtures, season, rng)


def _assign_days(fixtures: list[dict], season: int, rng: random.Random) -> list[dict]:
    """Spread games over days: nobody plays twice in a day, back-to-backs are avoided when
    possible, and teams with the most games left are scheduled first so the season ends evenly."""
    remaining = list(fixtures)
    rng.shuffle(remaining)
    left = Counter(team for game in remaining for team in (game["home"], game["away"]))
    last_played: dict[str, int] = {}
    start = opening_day(season)
    scheduled: list[dict] = []
    day = 0
    while remaining:
        day += 1
        busy: set[str] = set()
        picked: list[int] = []
        order = sorted(range(len(remaining)), key=lambda k: -(left[remaining[k]["home"]] + left[remaining[k]["away"]]))
        for allow_back_to_back in (False, True):
            for k in order:
                if len(picked) >= MAX_GAMES_PER_DAY:
                    break
                game = remaining[k]
                teams = (game["home"], game["away"])
                if k in picked or any(t in busy for t in teams):
                    continue
                if not allow_back_to_back and any(last_played.get(t) == day - 1 for t in teams):
                    continue
                picked.append(k)
                busy.update(teams)
            if picked:
                break
        for k in picked:
            game = remaining[k]
            for team in (game["home"], game["away"]):
                left[team] -= 1
                last_played[team] = day
            scheduled.append(game | {"day": day, "date": (start + timedelta(days=day - 1)).isoformat()})
        chosen = set(picked)
        remaining = [game for k, game in enumerate(remaining) if k not in chosen]
    scheduled.sort(key=lambda game: (game["day"], game["home"]))
    return [game | {"game_number": index} for index, game in enumerate(scheduled, start=1)]


# ---------------------------------------------------------------- simulation


def _strength_profiles(sim: Simulator, season: int, strength_mode: str, seed: int) -> dict[str, dict] | None:
    """Unscaled ratings for every team, held fixed for the season. Checkpoint mode uses each
    team's preseason rating, so simulating a past season does not see that season's results."""
    if sim.ratings is None:
        if strength_mode == "randomized":
            raise ValueError("randomized strength mode requires checkpoint ratings")
        return None
    if strength_mode == "neutral":
        return {team: dict(NEUTRAL) for team in TEAMS}
    preseason = [sim.ratings.latest(team, season, before=f"{season - 1}-08-01") for team in TEAMS]
    if strength_mode == "randomized":
        random.Random(seed).shuffle(preseason)
    return dict(zip(TEAMS, preseason))


class _GameRunner:
    """Simulates batches of fixtures and keeps the season's progress count."""

    def __init__(self, sim: Simulator, season: int, profiles: dict[str, dict] | None, seed: int,
                 temperature: float, progress: ProgressCallback | None,
                 inference_context: InferenceContext | None, batch_size: int):
        self.sim, self.season, self.profiles = sim, season, profiles
        self.rng = random.Random(seed)
        self.temperature, self.progress = temperature, progress
        self.inference_context, self.batch_size = inference_context, batch_size
        self.scale = float(getattr(sim, "strength_scale", 1.0))
        self.completed, self.total = 0, total_game_slots(season)

    def _ratings(self, home: str, away: str) -> tuple[dict, dict] | None:
        if self.profiles is None:
            return None
        return scale_ratings(self.profiles[home], self.scale), scale_ratings(self.profiles[away], self.scale)

    def report(self, stage: str) -> None:
        if self.progress:
            self.progress(stage, self.completed, self.total)

    def play(self, fixtures: list[dict], stage: str) -> list[dict]:
        """Simulate every fixture; a game that hits the event cap unfinished is resampled."""
        finals: list[GameState | None] = [None] * len(fixtures)
        pending = list(range(len(fixtures)))
        for attempt in range(MAX_GAME_ATTEMPTS):
            if not pending:
                break
            starts = [new_game(fixtures[k]["home"], fixtures[k]["away"], self.season, fixtures[k]["game_type"])
                      for k in pending]
            ratings = [self._ratings(fixtures[k]["home"], fixtures[k]["away"]) for k in pending]

            def on_batch(count: int) -> None:
                if attempt == 0:
                    self.completed += count
                    self.report(stage)

            states = self.sim.rollout_games(starts, ratings, seed=self.rng.randrange(2**32),
                                            temperature=self.temperature, batch_size=self.batch_size,
                                            inference_context=self.inference_context, on_batch=on_batch)
            for k, state in zip(pending, states):
                if state.over:
                    finals[k] = state
            pending = [k for k in pending if finals[k] is None]
        if pending:
            raise RuntimeError(f"simulation did not finish {len(pending)} {stage} games in {MAX_GAME_ATTEMPTS} attempts")
        return [_game_row(fixture, final, self.season) for fixture, final in zip(fixtures, finals)]


def _game_row(fixture: dict, final: GameState, season: int) -> dict:
    home, away = fixture["home"], fixture["away"]
    home_score, away_score = final.score[HOME], final.score[AWAY]
    return {
        "game_id": fixture["game_id"], "season": season, "game_type": fixture["game_type"],
        "stage": fixture["stage"], "round": fixture["round"], "day": fixture.get("day"),
        "date": fixture.get("date"), "game_in_series": fixture.get("game_in_series"),
        "home": home, "away": away, "home_score": home_score, "away_score": away_score,
        "winner": home if home_score > away_score else away,
        "home_seed": fixture.get("home_seed"), "away_seed": fixture.get("away_seed"),
        "overtimes": max(0, final.qtr - 4),
    }


# ---------------------------------------------------------------- standings and tiebreakers


class _Tiebreaker:
    """NBA tie-breaking order for ranking teams with the same winning percentage.

    Two teams: head-to-head, division winner, division record (same division), conference
    record, point differential, then a seeded coin flip. Three or more: division winner,
    head-to-head among the tied teams, division record (all in one division), conference record,
    point differential, coin flip. When a step separates some teams, the rest start over. The
    "record against playoff teams" steps are omitted.
    """

    def __init__(self, games: Sequence[dict], records: dict[str, dict], tiebreak_seed: str):
        self.records = records
        self.results: dict[str, list[tuple[str, int, int]]] = {team: [] for team in TEAMS}
        for game in games:
            home, away = game["home"], game["away"]
            margin = game["home_score"] - game["away_score"]
            self.results[home].append((away, int(game["winner"] == home), margin))
            self.results[away].append((home, int(game["winner"] == away), -margin))
        order = list(TEAMS)
        random.Random(f"coin-flip-{tiebreak_seed}").shuffle(order)
        self.coin = {team: index for index, team in enumerate(order)}
        self.division_winners: set[str] = set()

    def pct(self, team: str, opponents: set[str]) -> float | None:
        results = [won for opp, won, _ in self.results[team] if opp in opponents]
        return sum(results) / len(results) if results else None

    @staticmethod
    def _by_value(group: list[str], value: Callable[[str], float | None]) -> list[str] | None:
        values = {team: value(team) for team in group}
        if any(v is None for v in values.values()):
            return None
        best = max(values.values())
        survivors = [team for team in group if values[team] >= best - 1e-9]
        return survivors if len(survivors) < len(group) else None

    def _peers(self, team: str, level: int) -> set[str]:
        return {other for other in TEAMS if other != team and TEAM_DIVISION[other][:level] == TEAM_DIVISION[team][:level]}

    def _steps(self, group: list[str]) -> list[Callable[[list[str]], list[str] | None]]:
        head_to_head = lambda g: self._by_value(g, lambda t: self.pct(t, set(g) - {t}))
        division_winner = lambda g: self._by_value(g, lambda t: float(t in self.division_winners))
        same_division = len({TEAM_DIVISION[t] for t in group}) == 1
        division = lambda g: self._by_value(g, lambda t: self.pct(t, self._peers(t, 2))) if same_division else None
        conference = lambda g: self._by_value(g, lambda t: self.pct(t, self._peers(t, 1)))
        net = lambda g: self._by_value(g, lambda t: float(sum(m for _, _, m in self.results[t])))
        if len(group) == 2:
            return [head_to_head, division_winner, division, conference, net]
        return [division_winner, head_to_head, division, conference, net]

    def best(self, group: Sequence[str]) -> str:
        group = sorted(group)
        while len(group) > 1:
            for step in self._steps(group):
                survivors = step(group)
                if survivors is not None:
                    group = survivors
                    break
            else:
                return min(group, key=lambda team: self.coin[team])
        return group[0]

    def rank(self, teams: Sequence[str]) -> list[str]:
        remaining, ordered = list(teams), []
        while remaining:
            top = max(self.records[team]["win_pct"] for team in remaining)
            tied = [team for team in remaining if self.records[team]["win_pct"] >= top - 1e-9]
            winner = self.best(tied)
            ordered.append(winner)
            remaining.remove(winner)
        return ordered


def _empty_record(team: str) -> dict:
    conference, division = TEAM_DIVISION[team]
    return {"team": team, "conference": conference, "division": division, "wins": 0, "losses": 0,
            "home_wins": 0, "home_losses": 0, "away_wins": 0, "away_losses": 0,
            "conference_wins": 0, "conference_losses": 0, "division_wins": 0, "division_losses": 0,
            "points_for": 0, "points_against": 0, "streak": "", "last_ten": ""}


def standings(games: Sequence[dict], season: int, tiebreak_seed: str = "0") -> dict:
    """Division and conference standings, with seeds 1-10 (1-8 before the play-in era)."""
    records = {team: _empty_record(team) for team in TEAMS}
    results: dict[str, list[bool]] = {team: [] for team in TEAMS}
    for game in sorted(games, key=lambda g: (g.get("day") or 0, g["game_id"])):
        for team, opp, side, pf, pa in ((game["home"], game["away"], "home", game["home_score"], game["away_score"]),
                                        (game["away"], game["home"], "away", game["away_score"], game["home_score"])):
            record, won = records[team], game["winner"] == team
            key = "wins" if won else "losses"
            record[key] += 1
            record[f"{side}_{key}"] += 1
            if TEAM_DIVISION[opp][0] == record["conference"]:
                record[f"conference_{key}"] += 1
                if TEAM_DIVISION[opp][1] == record["division"]:
                    record[f"division_{key}"] += 1
            record["points_for"] += pf
            record["points_against"] += pa
            results[team].append(won)
    for team, record in records.items():
        played = record["wins"] + record["losses"]
        record["win_pct"] = record["wins"] / played if played else 0.0
        record["point_diff"] = record["points_for"] - record["points_against"]
        history = results[team]
        if history:
            run = next((k for k, won in enumerate(reversed(history)) if won != history[-1]), len(history))
            record["streak"] = f"{'W' if history[-1] else 'L'}{run}"
            last = history[-10:]
            record["last_ten"] = f"{sum(last)}-{len(last) - sum(last)}"

    breaker = _Tiebreaker(games, records, tiebreak_seed)
    division_orders: dict[tuple[str, str], list[str]] = {}
    for conference, divisions in DIVISIONS.items():
        for division, teams in divisions.items():
            division_orders[(conference, division)] = breaker.rank(teams)
            breaker.division_winners.add(division_orders[(conference, division)][0])

    play_in = has_play_in(season)
    out, seed_map = {}, {}
    for conference, divisions in DIVISIONS.items():
        ordered = breaker.rank([team for teams in divisions.values() for team in teams])
        seed_map[conference] = {team: seed for seed, team in enumerate(ordered, start=1)}
        leader = records[ordered[0]]

        def row(team: str) -> dict:
            seed = seed_map[conference][team]
            status = "playoffs" if seed <= (6 if play_in else 8) else "play_in" if play_in and seed <= 10 else "out"
            r = records[team]
            gb = ((leader["wins"] - r["wins"]) + (r["losses"] - leader["losses"])) / 2
            return r | {"seed": seed, "status": status, "games_behind": gb,
                        "division_winner": team in breaker.division_winners}

        out[conference] = {
            "seeds": [row(team) for team in ordered],
            "divisions": {division: [row(team) | {"division_rank": rank}
                                     for rank, team in enumerate(division_orders[(conference, division)], start=1)]
                          for division in divisions},
        }
    return {"conferences": out, "seed_map": seed_map, "records": records, "breaker": breaker}


# ---------------------------------------------------------------- playoffs


def _series_fixtures(prefix: str, round_name: str, high: tuple[int, str], low: tuple[int, str]) -> list[dict]:
    fixtures = []
    for number, high_home in enumerate(HIGHER_SEED_HOME, start=1):
        (home_seed, home), (away_seed, away) = (high, low) if high_home else (low, high)
        fixtures.append({"game_id": f"{prefix}-g{number}", "game_type": "POST", "stage": "Playoffs",
                         "round": round_name, "game_in_series": number, "home": home, "away": away,
                         "home_seed": home_seed, "away_seed": away_seed})
    return fixtures


def _series_result(conference: str, high: tuple[int, str], low: tuple[int, str], games: list[dict]) -> dict:
    wins = {high[1]: 0, low[1]: 0}
    played = []
    for game in games:
        played.append(game)
        wins[game["winner"]] += 1
        if max(wins.values()) == WINS_NEEDED:
            break
    winner = max(wins, key=wins.get)
    return {"conference": conference, "high": {"team": high[1], "seed": high[0]},
            "low": {"team": low[1], "seed": low[0]}, "wins": wins, "winner": winner,
            "loser": low[1] if winner == high[1] else high[1], "games": played,
            "summary": f"{winner} wins {wins[winner]}-{min(wins.values())}"}


def simulate_season(sim: Simulator, season: int, *, schedule: Sequence[Mapping] | None = None, seed: int = 0,
                    temperature: float = 1.0, strength_mode: str = "checkpoint",
                    progress: ProgressCallback | None = None,
                    inference_context: InferenceContext | None = None, batch_size: int = 128) -> dict:
    """Simulate a full season: 1,230 regular-season games, the play-in (from 2020-21) and four
    best-of-seven playoff rounds. `schedule` defaults to `generate_schedule(season, seed=seed)`."""
    _check_season(season)
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if strength_mode not in ("checkpoint", "neutral", "randomized"):
        raise ValueError("strength_mode must be checkpoint, neutral, or randomized")
    fixtures = [dict(game) for game in (schedule if schedule is not None else generate_schedule(season, seed=seed))]
    counts = Counter(team for game in fixtures for team in (game["home"], game["away"]))
    if set(counts) != set(TEAMS):
        raise ValueError("schedule must include all 30 NBA teams")
    profiles = _strength_profiles(sim, season, strength_mode, seed)
    runner = _GameRunner(sim, season, profiles, seed, temperature, progress, inference_context, batch_size)
    runner.report("Regular Season")

    for number, fixture in enumerate(fixtures, start=1):
        fixture.update(game_id=f"nba-{season}-reg-{number:04d}", game_type="REG",
                       stage="Regular Season", round="Regular Season")
    games = runner.play(fixtures, "Regular Season")
    table = standings(games, season, tiebreak_seed=f"{season}-{seed}")
    seeds = {conference: {s: team for team, s in m.items()} for conference, m in table["seed_map"].items()}

    # Play-in: 7 hosts 8 (winner is the 7 seed), 9 hosts 10 (loser is out), then the 7/8 loser
    # hosts the 9/10 winner for the 8 seed.
    play_in_rounds = []
    if has_play_in(season):
        first = []
        for conference in CONFERENCES:
            for high, low in ((7, 8), (9, 10)):
                first.append({"game_id": f"nba-{season}-playin-{conference.lower()}-{high}v{low}",
                              "game_type": "POST", "stage": "Play-In", "round": f"{conference} {high} vs {low}",
                              "conference": conference, "home": seeds[conference][high],
                              "away": seeds[conference][low], "home_seed": high, "away_seed": low})
        first_games = runner.play(first, "Play-In")
        second = []
        for conference in CONFERENCES:
            g78, g910 = [g for f, g in zip(first, first_games) if f["conference"] == conference]
            loser78 = g78["away"] if g78["winner"] == g78["home"] else g78["home"]
            seeds[conference][7] = g78["winner"]
            second.append({"game_id": f"nba-{season}-playin-{conference.lower()}-8th", "game_type": "POST",
                           "stage": "Play-In", "round": f"{conference} 8th seed", "conference": conference,
                           "home": loser78, "away": g910["winner"],
                           "home_seed": table["seed_map"][conference][loser78],
                           "away_seed": table["seed_map"][conference][g910["winner"]]})
        second_games = runner.play(second, "Play-In")
        for fixture, game in zip(second, second_games):
            seeds[fixture["conference"]][8] = game["winner"]
        play_in_rounds = [{"round": "Play-In", "games": first_games + second_games}]
    playoff_seeds = {conference: {s: seeds[conference][s] for s in range(1, 9)} for conference in CONFERENCES}

    rounds: list[dict] = []
    # Each conference bracket is a list of (seed, team) slots in bracket order.
    bracket = {conference: [(s, playoff_seeds[conference][s]) for pair in FIRST_ROUND for s in pair]
               for conference in CONFERENCES}
    breaker = table["breaker"]
    for round_index, round_name in enumerate(PLAYOFF_ROUNDS):
        matchups = []  # (conference, higher, lower)
        if round_name == "NBA Finals":
            (_, east), (_, west) = bracket["East"][0], bracket["West"][0]
            records = table["records"]
            # Home court goes to the better regular-season record (tiebreakers if level).
            higher = breaker.rank([east, west])[0] if records[east]["win_pct"] == records[west]["win_pct"] else \
                max((east, west), key=lambda t: records[t]["win_pct"])
            lower = west if higher == east else east
            matchups.append(("Finals", (bracket[TEAM_DIVISION[higher][0]][0][0], higher),
                             (bracket[TEAM_DIVISION[lower][0]][0][0], lower)))
        else:
            for conference in CONFERENCES:
                slots = bracket[conference]
                for k in range(0, len(slots), 2):
                    a, b = slots[k], slots[k + 1]
                    matchups.append((conference, *((a, b) if a[0] < b[0] else (b, a))))
        fixtures = []
        for k, (conference, high, low) in enumerate(matchups, start=1):
            fixtures += _series_fixtures(f"nba-{season}-r{round_index + 1}-s{k}", round_name, high, low)
        series_games = runner.play(fixtures, round_name)
        series = [_series_result(conference, high, low, series_games[k * SERIES_LENGTH:(k + 1) * SERIES_LENGTH])
                  for k, (conference, high, low) in enumerate(matchups)]
        rounds.append({"round": round_name, "series": series})
        if round_name != "NBA Finals":
            for conference in CONFERENCES:
                winners = [s for s in series if s["conference"] == conference]
                bracket[conference] = [(s["high"]["seed"] if s["winner"] == s["high"]["team"] else s["low"]["seed"],
                                        s["winner"]) for s in winners]

    finals = rounds[-1]["series"][0]
    runner.report("Complete")
    return {
        "season": season, "seed": seed, "strength_mode": strength_mode, "temperature": temperature,
        "regular_game_count": len(games), "play_in": has_play_in(season),
        "playoff_game_count": sum(len(s["games"]) for r in rounds for s in r["series"])
        + sum(len(r["games"]) for r in play_in_rounds),
        "playoff_team_count": 16, "games": games, "standings": table["conferences"],
        "play_in_rounds": play_in_rounds, "playoffs": rounds,
        "champion": finals["winner"], "finals": finals,
    }
