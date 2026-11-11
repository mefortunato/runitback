"""College basketball player tables from sportsdataverse's ESPN player box scores.

One release file per season and gender (``espn_<mens|womens>_college_basketball_player_boxscores``).
Each row is a player listed for a team in a game, with ``did_not_play`` for players who were listed
but didn't get in, the minutes and the box-score counts. ESPN gives **no did-not-play reason** for
college games (unlike the NBA's), and no history of injury reports.

Player ids are ESPN athlete ids, which stay the same when a player transfers, so every table here is
keyed by player and a transfer's history follows him to his new team.

Minutes are missing for most women's games before 2014 (and women's seasons before 2017 have only
about 1,700 games), so :data:`FIRST_SEASON` starts the women's tables in 2014.

Outputs (``processed`` directory):

- ``player_games.parquet``: one row per (game, player): ``game_id, season, game_type, game_date,
  team_id, team, opponent_id, home, player_id, name, position, group, starter, dnp, minutes`` and the
  box-score counts in :data:`STATS`.
- ``team_games.parquet``: one row per (game, team) with both teams' points and the estimated
  possessions (:func:`team_games`).
- ``players.parquet``: the player directory (name, position group, first / last season, last team).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RELEASES = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"


@dataclass(frozen=True)
class Gender:
    name: str               # cbb / wcbb (the wager sport)
    release: str            # sportsdataverse release name
    first_season: int       # first season with minutes in (nearly) every box score


MEN = Gender("cbb", "espn_mens_college_basketball_player_boxscores", 2008)
WOMEN = Gender("wcbb", "espn_womens_college_basketball_player_boxscores", 2014)
GENDERS = {"cbb": MEN, "wcbb": WOMEN, "men": MEN, "women": WOMEN}

GAME_TYPES = {2: "REG", 3: "POST"}       # 1 (preseason exhibitions) is dropped
GROUPS = ("G", "F", "C")
GROUP = {"G": "G", "PG": "G", "SG": "G", "G-F": "G", "F": "F", "SF": "F", "PF": "F", "F-C": "F", "C": "C"}
DEFAULT_GROUP = "F"                      # ATH, NA and missing positions
REGULATION_MINUTES = 40.0

# box-score counts kept per player-game (output name: ESPN column)
STATS = {
    "pts": "points", "fgm": "field_goals_made", "fga": "field_goals_attempted",
    "fg3m": "three_point_field_goals_made", "fg3a": "three_point_field_goals_attempted",
    "ftm": "free_throws_made", "fta": "free_throws_attempted", "oreb": "offensive_rebounds",
    "dreb": "defensive_rebounds", "ast": "assists", "stl": "steals", "blk": "blocks",
    "tov": "turnovers", "pf": "fouls",
}
BOX_COLUMNS = ["game_id", "season", "season_type", "game_date", "athlete_id", "athlete_display_name",
               "team_id", "team_location", "opponent_team_id", "home_away", "minutes", "starter",
               "did_not_play", "athlete_position_abbreviation", "team_score", "opponent_team_score",
               *STATS.values()]
FTA_WEIGHT = 0.475                       # free-throw trips per attempt in the possession estimate (college)


def gender(name: str) -> Gender:
    try:
        return GENDERS[name]
    except KeyError:
        raise ValueError(f"unknown gender {name!r}: one of {sorted(GENDERS)}") from None


def box_url(g: Gender, season: int) -> str:
    return f"{RELEASES}/{g.release}/player_box_{season}.parquet"


def download_box(g: Gender, season: int, raw_dir: Path, force: bool = False) -> Path | None:
    """One season's ESPN player box scores to ``raw_dir/player_box_<season>.parquet``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"player_box_{season}.parquet"
    if path.exists() and not force:
        return path
    try:
        df = pd.read_parquet(box_url(g, season))
    except Exception as e:  # noqa: BLE001 - not published (a season that hasn't started)
        log.warning("could not download the %s %d player box scores (%s)", g.name, season, e)
        return None
    df.to_parquet(path)
    return path


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq
    available = set(pq.read_schema(path).names)
    return pd.read_parquet(path, columns=[c for c in columns if c in available])


def position_group(pos) -> str:
    return GROUP.get(pos.upper(), DEFAULT_GROUP) if isinstance(pos, str) else DEFAULT_GROUP


def build_player_games(box: pd.DataFrame) -> pd.DataFrame:
    """Raw box rows (:data:`BOX_COLUMNS`) -> ``player_games`` (see the module docstring). Rows without
    an athlete id and preseason games are dropped; a player listed twice in a game keeps one row."""
    b = box[box.athlete_id.notna() & box.season_type.isin(list(GAME_TYPES))]
    dnp = b.did_not_play.fillna(False).astype(bool).to_numpy()
    minutes = pd.to_numeric(b.minutes, errors="coerce").fillna(0.0).to_numpy()
    pg = pd.DataFrame({
        "game_id": b.game_id.astype("int64").astype(str).to_numpy(),
        "season": b.season.astype(int).to_numpy(),
        "game_type": b.season_type.map(GAME_TYPES).to_numpy(),
        "game_date": b.game_date.astype(str).str[:10].to_numpy(),
        "team_id": b.team_id.astype("int64").to_numpy(),
        "team": b.team_location.astype(str).to_numpy(),
        "opponent_id": b.opponent_team_id.astype("int64").to_numpy(),
        "home": b.home_away.eq("home").to_numpy(),
        "player_id": b.athlete_id.astype("int64").to_numpy(),
        "name": b.athlete_display_name.to_numpy(),
        "position": b.athlete_position_abbreviation.to_numpy(),
        "starter": b.starter.fillna(False).astype(bool).to_numpy(),
        "dnp": dnp | (minutes <= 0),
        "minutes": np.where(dnp, 0.0, minutes),
        "team_score": pd.to_numeric(b.team_score, errors="coerce").to_numpy(),
        "opp_score": pd.to_numeric(b.opponent_team_score, errors="coerce").to_numpy(),
    })
    pg["group"] = pg.position.map(position_group)
    for k, c in STATS.items():
        v = pd.to_numeric(b[c], errors="coerce").fillna(0.0).to_numpy() if c in b else 0.0
        pg[k] = np.where(pg.dnp, 0.0, v).astype(np.float32)
    pg = pg.drop_duplicates(["game_id", "player_id"]).sort_values(
        ["game_date", "game_id", "home", "starter", "minutes"], ascending=[True, True, False, False, False],
        kind="stable").reset_index(drop=True)
    return pg


def team_games(pg: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team): ``game_id, season, game_date, team_id, opponent_id, home, points,
    opp_points, poss`` (possessions per team, the mean of both teams' ``FGA - OREB + TOV + 0.475 FTA``
    from the summed player box scores), ``minutes`` (the team's player-minutes) and ``players``.
    Games without exactly two teams in ``pg`` are dropped."""
    t = pg.groupby(["game_id", "team_id"], sort=False).agg(
        season=("season", "first"), game_date=("game_date", "first"), game_type=("game_type", "first"),
        opponent_id=("opponent_id", "first"), home=("home", "first"), points=("team_score", "first"),
        opp_points=("opp_score", "first"), fga=("fga", "sum"), oreb=("oreb", "sum"), tov=("tov", "sum"),
        fta=("fta", "sum"), minutes=("minutes", "sum"), players=("dnp", lambda s: int((~s).sum())),
    ).reset_index()
    t["poss_own"] = t.fga - t.oreb + t.tov + FTA_WEIGHT * t.fta
    two = t.groupby("game_id").team_id.transform("size") == 2
    t = t[two].copy()
    t["poss"] = t.groupby("game_id").poss_own.transform("mean")
    return t.drop(columns=["fga", "oreb", "tov", "fta", "poss_own"]).reset_index(drop=True)


def build_player_directory(pg: pd.DataFrame) -> pd.DataFrame:
    pg = pg.sort_values(["game_date", "game_id"], kind="stable")
    played = pg[~pg.dnp]
    last = pg.groupby("player_id").tail(1).set_index("player_id")
    pos = pg[pg.position.notna()].groupby("player_id").position.last()
    d = pd.DataFrame({
        "player_id": last.index, "name": last.name.to_numpy(),
        "position": pos.reindex(last.index).to_numpy(),
        "first_season": pg.groupby("player_id").season.min().reindex(last.index).to_numpy(),
        "last_season": last.season.to_numpy(), "team_id": last.team_id.to_numpy(), "team": last.team.to_numpy(),
        "teams": pg.groupby("player_id").team_id.nunique().reindex(last.index).to_numpy(),
        "games": played.groupby("player_id").size().reindex(last.index).fillna(0).astype(int).to_numpy(),
    })
    d["group"] = d.position.map(position_group)
    return d.reset_index(drop=True)


def latest_team_names(pg: pd.DataFrame) -> pd.Series:
    """Each ESPN team id's school name in its latest game (as the wager model's college team keys,
    :func:`runitback.wager.espn.latest_names`), indexed by id."""
    from ..wager.espn import latest_names

    rows = pg[["team_id", "team", "season", "game_date"]].rename(columns={"team_id": "id", "team": "school",
                                                                          "game_date": "day"})
    return latest_names(rows.drop_duplicates(["id", "season", "day"]), "school")


def build_players(g: Gender, seasons: list[int], raw_dir: Path, out_dir: Path) -> dict:
    """Write ``player_games``, ``team_games`` and ``players`` parquet files for the seasons whose box
    scores are in ``raw_dir``. Returns a per-season summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parts, summary = [], {}
    for s in seasons:
        path = raw_dir / f"player_box_{s}.parquet"
        if not path.exists():
            log.warning("%s %d: %s missing (download first); skipped", g.name, s, path.name)
            continue
        pg = build_player_games(_read(path, BOX_COLUMNS))
        played = pg[~pg.dnp]
        summary[s] = {"games": int(pg.game_id.nunique()), "players": int(pg.player_id.nunique()),
                      "player_games": len(played), "dnp_share": round(float(pg.dnp.mean()), 3)}
        parts.append(pg)
    if not parts:
        raise FileNotFoundError(f"no player box scores in {raw_dir}")
    pg = pd.concat(parts, ignore_index=True)
    names = latest_team_names(pg)
    pg["team"] = pg.team_id.map(names).fillna(pg.team)
    pg.to_parquet(out_dir / "player_games.parquet", index=False)
    team_games(pg).to_parquet(out_dir / "team_games.parquet", index=False)
    build_player_directory(pg).to_parquet(out_dir / "players.parquet", index=False)
    return summary
