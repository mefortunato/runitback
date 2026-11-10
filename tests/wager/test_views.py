"""Web views of the sports ported from sportsml after the NFL."""

import pytest

from runitback.wager.core.process import process
from runitback.wager.web.views import BUILTIN_VIEWS, get_view


def fixture_games(sport: str):
    from runitback.wager.mlb.data.download import build_games

    from .test_mlb import LINES, LOGS, PEOPLE, SCHEDULE, TEAMS, TODAY

    return build_games([SCHEDULE], LOGS, PEOPLE, TEAMS, LINES, today=TODAY)


@pytest.mark.parametrize("sport", ["mlb"])
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
