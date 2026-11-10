from ...core.schema import GameSchema

STATS_COLUMNS = [
    "goals",
    "shots",
    "missed_shots",
    "blocked_shots",
    "hits",
    "giveaways",
    "takeaways",
    "faceoff_pct",
    "pp_opportunities",
    "pp_goals",
    "pp_toi",
    "pp_pct",
    "pk_pct",
    "sh_goals_against",
    "empty_net_goals",
]

OPP_STATS_COLUMNS = [f"opp_{col}" for col in STATS_COLUMNS]

# The betting market from ESPN's odds API (from 2019-20). There is no point
# spread: the puck line is a fixed 1.5-goal handicap priced like a
# moneyline, so the market's view of a game is its moneylines (and its
# expected margin the one they imply; see backtest.market_predictions).
# Like the NFL's they are not default features.
MARKET_COLUMNS = [
    "moneyline",
    "opp_moneyline",
    "total_line",
    "over_odds",
    "under_odds",
    "puck_line",
    "puck_line_odds",
    "opp_puck_line_odds",
]

SCHEMA = GameSchema(
    season="season",
    # days since the season's first game: teams play every 1-3 days
    date="day",
    team="team",
    opp="opp_team",
    game_id="game_id",
    target="result",
    stats=tuple(STATS_COLUMNS + OPP_STATS_COLUMNS),
    location="location",
    context=("rest", "opp_rest", "postseason"),
    categorical=("game_type",),
    paired=(
        ("goals", "opp_goals"),
        ("rest", "opp_rest"),
        ("moneyline", "opp_moneyline"),
        ("puck_line_odds", "opp_puck_line_odds"),
    ),
    negated=("puck_line",),
    team_name="team_name",
    opp_name="opp_name",
    display=("game_date", "start_date", "week", "periods", *MARKET_COLUMNS),
    market_moneyline=("moneyline", "opp_moneyline"),
    total="total",
    market_total="total_line",
    market_total_odds=("over_odds", "under_odds"),
)
