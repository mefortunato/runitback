"""Web views of the sports added after the NFL (ported from sportsml's tests/test_views.py)."""

import pytest

from runitback.wager.core.process import process
from runitback.wager.web.views import BUILTIN_VIEWS, get_view


def fixture_games(sport: str):
    if sport == "nhl":
        from runitback.wager.nhl.data.download import build_games

        from .test_nhl import LINES, REPORTS, SCHEDULE, TEAMS, TODAY

        return build_games(SCHEDULE, REPORTS, TEAMS, LINES, today=TODAY)
    raise ValueError(sport)


@pytest.mark.parametrize("sport", ["nhl"])
def test_views_show_their_games(sport):
    assert sport in BUILTIN_VIEWS
    view = get_view(sport)
    games = process(sport, fixture_games(sport))
    played = games[games["result"].notna()]
    for stat in view.stats:
        values = stat.value(played)
        assert len(values) == len(played), stat.key
    assert set(view.score) <= set(games.columns)
    row = games.iloc[-1]
    assert view.kickoff(row)[:4] in ("2024", "2025")
    assert view.date_name(row)
    assert isinstance(view.playoffs(played), dict)
