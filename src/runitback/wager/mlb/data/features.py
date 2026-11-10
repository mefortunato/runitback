from ...core.schema import GameSchema
from .download import BOX_COLUMNS, STARTER_COLUMNS

STATS_COLUMNS = ["runs", *(c for c in BOX_COLUMNS if c != "runs")]

OPP_STATS_COLUMNS = [f"opp_{col}" for col in STATS_COLUMNS]

OPP_STARTER_COLUMNS = [f"opp_{col}" for col in STARTER_COLUMNS]

# The betting market from ESPN's odds API (from 2013). There is no point
# spread: the run line is a fixed 1.5-run handicap priced like a moneyline,
# so the market's view of a game is its moneylines (and its expected margin
# the one they imply; see backtest.market_predictions). Like the NFL's they
# are not default features.
MARKET_COLUMNS = [
    "moneyline",
    "opp_moneyline",
    "total_line",
    "over_odds",
    "under_odds",
    "run_line",
    "run_line_odds",
    "opp_run_line_odds",
]

SCHEMA = GameSchema(
    season="season",
    # days since the season's first game: teams play almost every day
    date="day",
    team="team",
    opp="opp_team",
    game_id="game_id",
    target="result",
    stats=tuple(STATS_COLUMNS + OPP_STATS_COLUMNS),
    location="location",
    # the announced starting pitchers' records (see download.starter_features)
    context=(
        "rest",
        "opp_rest",
        "postseason",
        *STARTER_COLUMNS,
        *OPP_STARTER_COLUMNS,
    ),  # fmt: skip
    categorical=("game_type",),
    paired=(
        ("runs", "opp_runs"),
        ("rest", "opp_rest"),
        ("moneyline", "opp_moneyline"),
        ("run_line_odds", "opp_run_line_odds"),
        ("sp_id", "opp_sp_id"),
        ("sp_name", "opp_sp_name"),
        *zip(STARTER_COLUMNS, OPP_STARTER_COLUMNS),
    ),
    negated=("run_line",),
    team_name="team_name",
    opp_name="opp_name",
    display=(
        "game_date",
        "start_date",
        "week",
        "game_number",
        "sp_name",
        "opp_sp_name",
        *MARKET_COLUMNS,
    ),  # fmt: skip
    market_moneyline=("moneyline", "opp_moneyline"),
    total="total",
    market_total="total_line",
    market_total_odds=("over_odds", "under_odds"),
)
