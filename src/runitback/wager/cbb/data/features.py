from ...core.schema import GameSchema
from ...nba.data.features import STATS_COLUMNS as NBA_STATS_COLUMNS

# ESPN's box-score stats that start in 2023 outside the NBA (college
# basketball and the WNBA): a model trained on earlier seasons would see
# them for the first time in the season it predicts.
RECENT_STATS = ("paint_points", "fast_break_points", "turnover_points")

# Men's and women's college basketball share the NBA's ESPN box score,
# without the recent stats.
STATS_COLUMNS = [c for c in NBA_STATS_COLUMNS if c not in RECENT_STATS]

OPP_STATS_COLUMNS = [f"opp_{col}" for col in STATS_COLUMNS]

# The betting market from ESPN's odds API. Upcoming games carry lines once
# ESPN posts them (about a day ahead); like the NFL's they are not default
# features.
MARKET_COLUMNS = [
    "spread_line",
    "total_line",
    "moneyline",
    "opp_moneyline",
    "over_odds",
    "under_odds",
    "spread_odds",
    "opp_spread_odds",
]

SCHEMA = GameSchema(
    season="season",
    # days since the season's first game: teams play every 2-4 days, so team
    # state is refreshed (and the backtest replayed) one game day at a time
    date="day",
    team="team",
    opp="opp_team",
    game_id="game_id",
    target="result",
    # points come from the schedule; every other stat from the box score
    stats=tuple(STATS_COLUMNS + OPP_STATS_COLUMNS),
    location="location",
    context=("rest", "opp_rest", "conference_game", "postseason"),
    categorical=("game_type", "conference", "opp_conference"),
    paired=(
        ("points", "opp_points"),
        ("rest", "opp_rest"),
        ("conference", "opp_conference"),
        ("moneyline", "opp_moneyline"),
        ("spread_odds", "opp_spread_odds"),
    ),
    negated=("spread_line",),
    team_name="team_name",
    opp_name="opp_name",
    display=("game_date", "start_date", "week", "notes", *MARKET_COLUMNS),
    market_spread="spread_line",
    market_moneyline=("moneyline", "opp_moneyline"),
    total="total",
    market_total="total_line",
    market_total_odds=("over_odds", "under_odds"),
)
