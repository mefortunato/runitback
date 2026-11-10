import pandas as pd

from ...cbb.data.download import WOMEN, download_college


def download(
    output_file: str | None = None,
    first_season: int = WOMEN.first_season,
    last_season: int | None = None,
    cache_dir: str | None = "data/wcbb/espn_cache",
) -> pd.DataFrame:
    """Download every season's women's games, box scores and lines (ESPN; see
    :mod:`runitback.wager.cbb.data.download`).

    Args:
        output_file: Also write the games table here.
        first_season / last_season: Seasons to download, by ending year
            (default: 2005-06 through the current season). ESPN's betting
            lines start in 2021-22.
        cache_dir: Reuse finished seasons' files and finished games' lines
            from here (``None``: always download).
    """
    return download_college(WOMEN, output_file, first_season, last_season, cache_dir)
