from ...core.schema import GameSchema

STATS_COLUMNS = [
    "goals",
    "half_time_goals",
    "shots",
    "shots_on_target",
    "corners",
    "fouls",
    "yellow_cards",
    "red_cards",
]

OPP_STATS_COLUMNS = [f"opp_{col}" for col in STATS_COLUMNS]

# The betting market from football-data.co.uk (see download.py for which
# bookmaker each comes from). Football has three outcomes: ``moneyline`` /
# ``opp_moneyline`` are the home / away win prices and ``draw_moneyline`` the
# draw's, all American; a win probability needs all three (a draw is a quarter
# of the games), so the two-way moneyline columns alone overstate it. The
# total is always the 2.5-goal line. ``spread_line`` is the Asian handicap
# turned into the team's expected margin (positive: favored).
MARKET_COLUMNS = [
    "moneyline",
    "opp_moneyline",
    "draw_moneyline",
    "total_line",
    "over_odds",
    "under_odds",
    "spread_line",
]

SCHEMA = GameSchema(
    season="season",
    # days since the season's first game: teams play every 3-7 days
    date="day",
    team="team",
    opp="opp_team",
    game_id="game_id",
    target="result",
    stats=tuple(STATS_COLUMNS + OPP_STATS_COLUMNS),
    location="location",
    context=("rest", "opp_rest"),
    paired=(
        ("goals", "opp_goals"),
        ("rest", "opp_rest"),
        ("moneyline", "opp_moneyline"),
    ),
    negated=("spread_line",),
    team_name="team_name",
    opp_name="opp_name",
    display=(
        "game_date",
        "start_date",
        "week",
        "referee",
        "odds_source",
        *MARKET_COLUMNS,
    ),
    market_spread="spread_line",
    market_moneyline=("moneyline", "opp_moneyline"),
    total="total",
    market_total="total_line",
    market_total_odds=("over_odds", "under_odds"),
)
