from ...core.schema import GameSchema

# ESPN's team box score (every FBS game since 2004)
STATS_COLUMNS = [
    "points",
    "firstDowns",
    "totalYards",
    "netPassingYards",
    "passingCompletions",
    "passingAttempts",
    "yardsPerPass",
    "rushingYards",
    "rushingAttempts",
    "yardsPerRushAttempt",
    "thirdDownConv",
    "thirdDownAtt",
    "fourthDownConv",
    "fourthDownAtt",
    "totalPenalties",
    "totalPenaltiesYards",
    "turnovers",
    "fumblesLost",
    "interceptions",
    "possessionTime",
]

OPP_STATS_COLUMNS = [f"opp_{col}" for col in STATS_COLUMNS]

# The betting market from ESPN's odds API. Lines for upcoming games are as of
# the last download (ESPN posts them about a week ahead); like the NFL's they
# are not default features.
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
    # chronological week of the season; bowls follow the last regular week,
    # one week per 7 days of the bowl season (see download.chronological_weeks)
    date="week",
    team="team",
    opp="opp_team",
    game_id="game_id",
    target="result",
    # points come from the schedule; every other stat from the box score
    stats=tuple(STATS_COLUMNS + OPP_STATS_COLUMNS),
    location="location",
    context=("rest", "opp_rest", "conference_game", "postseason"),
    categorical=("conference", "opp_conference"),
    paired=(
        ("conference", "opp_conference"),
        ("rest", "opp_rest"),
        ("moneyline", "opp_moneyline"),
        ("spread_odds", "opp_spread_odds"),
    ),
    negated=("spread_line",),
    team_name="team_name",
    opp_name="opp_name",
    display=("start_date", "start_time_tbd", "game_type", "notes", *MARKET_COLUMNS),
    market_spread="spread_line",
    market_moneyline=("moneyline", "opp_moneyline"),
    total="total",
    market_total="total_line",
    market_total_odds=("over_odds", "under_odds"),
)
