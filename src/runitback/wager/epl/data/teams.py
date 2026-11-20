"""Premier League clubs: football-data.co.uk's names -> games-table keys.

The key is a three-letter code (the usual broadcast one); ``team_name`` is the
club's full name. Every club of the Premier League era (1992-93 on) that
football-data.co.uk has under its short name is listed, so a newly promoted
club is the only thing that can be missing (``team_codes`` raises with the
names to add).
"""

# football-data.co.uk name -> (code, full name)
CLUBS = {
    "Arsenal": ("ARS", "Arsenal"),
    "Aston Villa": ("AVL", "Aston Villa"),
    "Barnsley": ("BAR", "Barnsley"),
    "Birmingham": ("BIR", "Birmingham City"),
    "Blackburn": ("BLB", "Blackburn Rovers"),
    "Blackpool": ("BLP", "Blackpool"),
    "Bolton": ("BOL", "Bolton Wanderers"),
    "Bournemouth": ("BOU", "AFC Bournemouth"),
    "Bradford": ("BRD", "Bradford City"),
    "Brentford": ("BRE", "Brentford"),
    "Brighton": ("BHA", "Brighton & Hove Albion"),
    "Burnley": ("BUR", "Burnley"),
    "Cardiff": ("CAR", "Cardiff City"),
    "Charlton": ("CHA", "Charlton Athletic"),
    "Chelsea": ("CHE", "Chelsea"),
    "Coventry": ("COV", "Coventry City"),
    "Crystal Palace": ("CRY", "Crystal Palace"),
    "Derby": ("DER", "Derby County"),
    "Everton": ("EVE", "Everton"),
    "Fulham": ("FUL", "Fulham"),
    "Huddersfield": ("HUD", "Huddersfield Town"),
    "Hull": ("HUL", "Hull City"),
    "Ipswich": ("IPS", "Ipswich Town"),
    "Leeds": ("LEE", "Leeds United"),
    "Leicester": ("LEI", "Leicester City"),
    "Liverpool": ("LIV", "Liverpool"),
    "Luton": ("LUT", "Luton Town"),
    "Man City": ("MCI", "Manchester City"),
    "Man United": ("MUN", "Manchester United"),
    "Middlesbrough": ("MID", "Middlesbrough"),
    "Newcastle": ("NEW", "Newcastle United"),
    "Norwich": ("NOR", "Norwich City"),
    "Nott'm Forest": ("NFO", "Nottingham Forest"),
    "Oldham": ("OLD", "Oldham Athletic"),
    "Portsmouth": ("POR", "Portsmouth"),
    "QPR": ("QPR", "Queens Park Rangers"),
    "Reading": ("REA", "Reading"),
    "Sheffield United": ("SHU", "Sheffield United"),
    "Sheffield Weds": ("SHW", "Sheffield Wednesday"),
    "Southampton": ("SOU", "Southampton"),
    "Stoke": ("STK", "Stoke City"),
    "Sunderland": ("SUN", "Sunderland"),
    "Swansea": ("SWA", "Swansea City"),
    "Swindon": ("SWI", "Swindon Town"),
    "Tottenham": ("TOT", "Tottenham Hotspur"),
    "Watford": ("WAT", "Watford"),
    "West Brom": ("WBA", "West Bromwich Albion"),
    "West Ham": ("WHU", "West Ham United"),
    "Wigan": ("WIG", "Wigan Athletic"),
    "Wimbledon": ("WIM", "Wimbledon"),
    "Wolves": ("WOL", "Wolverhampton Wanderers"),
}

# Team names used by sportsbooks / The Odds API -> games-table team keys.
odds_map = {full: code for code, full in CLUBS.values()} | {
    "Brighton and Hove Albion": "BHA",
    "Bournemouth": "BOU",
    "Manchester Utd": "MUN",
    "Newcastle": "NEW",
    "Tottenham": "TOT",
    "Wolverhampton": "WOL",
    "West Ham": "WHU",
    "Nottingham Forest": "NFO",
    "Leeds": "LEE",
    "Leicester": "LEI",
}


def team_codes(names) -> dict[str, tuple[str, str]]:
    """``name -> (code, full name)`` for every name in ``names``; raises on
    a club :data:`CLUBS` does not have (a newly promoted one)."""
    names = set(names)
    unknown = sorted(names - set(CLUBS))
    if unknown:
        raise ValueError(
            f"unknown football-data.co.uk team names {unknown}; add them to "
            "runitback.wager.epl.data.teams.CLUBS"
        )
    return {name: CLUBS[name] for name in names}
