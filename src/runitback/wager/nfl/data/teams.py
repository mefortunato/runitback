import pandas as pd

from .names import move_map

# nflverse lists the Rams twice ("LA" and "LAR"); games use "LA".
DUPLICATES = {"LAR"}


def load_teams() -> pd.DataFrame:
    """Current teams' display metadata from nflverse (name, colors, logo)."""
    import nflreadpy

    teams = nflreadpy.load_teams().to_pandas()
    teams = teams[~teams["team_abbr"].isin(set(move_map) | DUPLICATES)]
    return pd.DataFrame(
        {
            "team": teams["team_abbr"],
            "name": teams["team_name"],
            "nickname": teams["team_nick"],
            "color": teams["team_color"],
            "color2": teams["team_color2"],
            "logo": teams["team_logo_espn"],
            "conference": teams["team_conf"],
            "division": teams["team_division"],
        }
    ).reset_index(drop=True)
