"""Generate synthetic NFL seasons in the processed training-data format."""
from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path

import pandas as pd

from ..data.ratings import NEUTRAL, RATING_COLS
from ..engine.rules import AWAY, HOME, new_game
from ..vocab import TEAM_ALIASES, TEAMS
from .sampler import Simulator

log = logging.getLogger(__name__)


def _column(frame: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"schedule must include one of these columns: {', '.join(names)}")


def _normalize_schedule(schedule: pd.DataFrame, season: int) -> list[dict]:
    if not 1999 <= season <= 2040:
        raise ValueError("season must be between 1999 and 2040")
    week_col = _column(schedule, "week")
    home_col = _column(schedule, "home", "home_team")
    away_col = _column(schedule, "away", "away_team")
    type_col = next((name for name in ("game_type", "season_type") if name in schedule), None)

    if "season" in schedule and not schedule.empty:
        seasons = pd.to_numeric(schedule["season"], errors="coerce")
        if seasons.isna().any() or not (seasons == season).all():
            raise ValueError(f"all schedule season values must equal {season}")

    rows = []
    for index, row in schedule.reset_index(drop=True).iterrows():
        try:
            week_value = float(row[week_col])
        except (TypeError, ValueError):
            raise ValueError(f"schedule row {index + 1} has an invalid week") from None
        if not week_value.is_integer() or week_value < 1:
            raise ValueError(f"schedule row {index + 1} has an invalid week")

        teams = []
        for column in (home_col, away_col):
            if pd.isna(row[column]):
                raise ValueError(f"schedule row {index + 1} is missing a team")
            team = str(row[column]).strip().upper()
            teams.append(TEAM_ALIASES.get(team, team))
        home, away = teams
        if home not in TEAMS or away not in TEAMS:
            raise ValueError(f"schedule row {index + 1} has an unknown team: {home} vs {away}")
        if home == away:
            raise ValueError(f"schedule row {index + 1} has the same home and away team")

        game_type = "REG" if type_col is None or pd.isna(row[type_col]) else str(row[type_col]).strip().upper()
        if game_type not in ("REG", "POST"):
            raise ValueError(f"schedule row {index + 1} has invalid game_type {game_type!r}; use REG or POST")
        rows.append({"week": int(week_value), "home": home, "away": away, "game_type": game_type,
                     "input_order": index})

    if not rows:
        raise ValueError("schedule contains no games")
    return sorted(rows, key=lambda item: (item["week"], item["input_order"]))


def _randomized_strengths(sim: Simulator, fixtures: list[dict], season: int, seed: int) -> dict[str, dict]:
    if sim.ratings is None:
        raise ValueError("randomized strength mode requires checkpoint ratings")
    teams = sorted({team for fixture in fixtures for team in (fixture["home"], fixture["away"])})
    profiles = [sim.ratings.latest(team, season, week=1) for team in teams]
    random.Random(seed).shuffle(profiles)
    return dict(zip(teams, profiles))


def generate_season(sim: Simulator, schedule: pd.DataFrame, season: int, out_dir: Path, *,
                    seed: int = 0, temperature: float = 1.0, checkpoint: str | None = None,
                    strength_mode: str = "checkpoint", force: bool = False) -> dict:
    """Simulate a schedule and save games/plays/ratings Parquet files for model training."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if strength_mode not in ("checkpoint", "neutral", "randomized"):
        raise ValueError("strength_mode must be checkpoint, neutral, or randomized")
    games_path = out_dir / "games.parquet"
    plays_path = out_dir / "plays.parquet"
    ratings_path = out_dir / "ratings.parquet"
    metadata_path = out_dir / "generation.json"
    targets = [games_path, plays_path, ratings_path, metadata_path]
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output already exists ({names}); pass force=True to replace it")

    fixtures = _normalize_schedule(schedule, season)
    randomized_strengths = (_randomized_strengths(sim, fixtures, season, seed)
                            if strength_mode == "randomized" else None)
    checkpoint_sha256 = None
    if checkpoint is not None and Path(checkpoint).is_file():
        digest = hashlib.sha256()
        with Path(checkpoint).open("rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1 << 20), b""):
                digest.update(chunk)
        checkpoint_sha256 = digest.hexdigest()

    rng = random.Random(seed)
    games, plays, ratings_rows = [], [], []
    for game_number, fixture in enumerate(fixtures, start=1):
        game_id = f"synthetic-{season}-s{seed}-{fixture['week']:02d}-{game_number:04d}"
        opening_kicker = rng.choice((HOME, AWAY))
        start = new_game(fixture["home"], fixture["away"], season, fixture["game_type"],
                         opening_kicker=opening_kicker)
        override = None
        if strength_mode == "neutral":
            override = {fixture["home"]: NEUTRAL, fixture["away"]: NEUTRAL}
        elif randomized_strengths is not None:
            override = {team: randomized_strengths[team] for team in (fixture["home"], fixture["away"])}
        game_ratings = sim.ratings_for(fixture["home"], fixture["away"], season, fixture["week"],
                                       override=override)
        finals, logs = sim.rollout(start, 1, seed=rng.randrange(2**32), temperature=temperature,
                                   keep_plays=1, ratings=game_ratings)
        if not finals[0].over:
            raise RuntimeError(f"simulation did not finish {game_id}; no output was written")
        if len(logs) != 1 or not logs[0]:
            raise RuntimeError(f"simulation produced no play log for {game_id}; no output was written")

        final = finals[0]
        games.append({"game_id": game_id, "season": season, "game_type": fixture["game_type"],
                      "week": fixture["week"], "home": fixture["home"], "away": fixture["away"],
                      "home_final": final.score[HOME], "away_final": final.score[AWAY],
                      "opening_kicker": opening_kicker})
        for play_index, (state, outcome) in enumerate(logs[0]):
            plays.append({
                "game_id": game_id, "idx": play_index, "qtr": state.qtr, "qsec": state.qsec,
                "phase": state.phase, "pos": state.pos, "down": state.down, "ytg": state.ytg,
                "yl": state.yl, "score_h": state.score[HOME], "score_a": state.score[AWAY],
                "to_h": state.timeouts[HOME], "to_a": state.timeouts[AWAY],
                "play_type": outcome.play_type, "result": outcome.result, "yards": outcome.yards,
                "elapsed": outcome.elapsed,
            })

        if game_ratings is not None:
            for team, side, team_ratings in zip((fixture["home"], fixture["away"]), ("home", "away"),
                                                game_ratings):
                ratings_rows.append({"game_id": game_id, "season": season, "week": fixture["week"],
                                     "team": team, "side": side,
                                     **{f"pre_{key}": team_ratings[key] for key in RATING_COLS}})
        if game_number % 25 == 0 or game_number == len(fixtures):
            log.info("simulated %d/%d games", game_number, len(fixtures))

    out_dir.mkdir(parents=True, exist_ok=True)
    games_df, plays_df = pd.DataFrame(games), pd.DataFrame(plays)
    games_df.to_parquet(games_path, index=False)
    plays_df.to_parquet(plays_path, index=False)
    if ratings_rows:
        pd.DataFrame(ratings_rows).to_parquet(ratings_path, index=False)
    elif force:
        ratings_path.unlink(missing_ok=True)

    metadata = {"format": "runitback-nfl-processed-v1", "synthetic": True,
                "generator_checkpoint": checkpoint, "generator_checkpoint_sha256": checkpoint_sha256,
                "season": season, "seed": seed, "strength_mode": strength_mode,
                "temperature": temperature, "games": len(games_df), "plays": len(plays_df),
                "ratings_included": bool(ratings_rows)}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata