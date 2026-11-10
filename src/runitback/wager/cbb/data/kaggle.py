"""March Machine Learning Mania (Kaggle) data, shared by men's and women's.

Not the ``cbb`` / ``wcbb`` download any more (that is ESPN's, with upcoming
games and betting lines); kept for Kaggle's tournament seeds and slots,
which :mod:`.bracket` simulates, keyed by Kaggle's team ids.
"""

import pathlib
import tempfile

import pandas as pd

TEAM_STATS_COLUMNS = [
    "Win",
    "Score",
    "PlusMinus",
    "FGM",
    "FGA",
    "FGM3",
    "FGA3",
    "FTM",
    "FTA",
    "OR",
    "DR",
    "Ast",
    "TO",
    "Stl",
    "Blk",
    "PF",
]

OPP_TEAM_STATS_COLUMNS = [f"{col}_OPP" for col in TEAM_STATS_COLUMNS]


def _read(directory: str, name: str) -> pd.DataFrame:
    path = pathlib.Path(directory) / name
    if not path.exists() and path.with_name(name + ".zip").exists():
        path = path.with_name(name + ".zip")
    return pd.read_csv(path)


def download_kaggle(prefix: str, year: int, output_file: str = None) -> pd.DataFrame:
    """Regular season + tournament detailed results with team names.

    Args:
        prefix: ``"M"`` (men) or ``"W"`` (women).
        year: Competition year, e.g. 2026 for ``march-machine-learning-mania-2026``.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    competition = f"march-machine-learning-mania-{year}"
    files = [
        f"{prefix}RegularSeasonDetailedResults.csv",
        f"{prefix}NCAATourneyDetailedResults.csv",
        f"{prefix}Teams.csv",
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        for name in files:
            api.competition_download_file(competition, file_name=name, path=temp_dir)
        results = pd.concat([_read(temp_dir, name) for name in files[:2]])
        teams = _read(temp_dir, files[2])
    games = format_games(results, teams)

    if output_file:
        pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        games.to_csv(output_file, index=False)
    return games


def format_games(games: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Kaggle winner/loser rows -> one row per team per game."""
    games = games.copy()
    games["WPlusMinus"] = games["WScore"] - games["LScore"]
    games["WWin"] = (games["WPlusMinus"] > 0).astype(int)
    games["LPlusMinus"] = games["LScore"] - games["WScore"]
    games["LWin"] = (games["LPlusMinus"] > 0).astype(int)

    col_renamer = {col: col[1:] for col in games.columns if col.startswith("W")}
    col_renamer.update(
        {col: col[1:] + "_OPP" for col in games.columns if col.startswith("L")}
    )

    games = games.rename(columns=col_renamer)
    games["Loc"] = games["Loc"].map({"H": 1, "A": -1, "N": 0})

    opp_games = games.copy()[["Season", "DayNum", "NumOT"]]
    opp_games["Loc"] = -1 * games["Loc"]
    opp_games[["TeamID", "TeamID_OPP"]] = games[["TeamID_OPP", "TeamID"]].values
    opp_games[TEAM_STATS_COLUMNS] = games[OPP_TEAM_STATS_COLUMNS].values
    opp_games[OPP_TEAM_STATS_COLUMNS] = games[TEAM_STATS_COLUMNS].values

    games = pd.concat([games, opp_games], ignore_index=True)

    names = teams.set_index("TeamID")["TeamName"]
    games["Team"] = games["TeamID"].map(names)
    games["Team_OPP"] = games["TeamID_OPP"].map(names)

    # sorted team ids so both rows of a game share an id
    low = games[["TeamID", "TeamID_OPP"]].min(axis=1)
    high = games[["TeamID", "TeamID_OPP"]].max(axis=1)
    games["_id"] = (
        games["Season"].astype(str)
        + "."
        + games["DayNum"].astype(str)
        + "."
        + low.astype(str)
        + "."
        + high.astype(str)
    )
    return games
