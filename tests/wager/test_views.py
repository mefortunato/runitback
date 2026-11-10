"""Web views of the sports ported from sportsml after the NFL."""

import pandas as pd
import pytest

from runitback.wager.core.process import process
from runitback.wager.web.views import BUILTIN_VIEWS, get_view


def fixture_games(sport: str) -> pd.DataFrame:
    if sport in ("cbb", "wcbb"):
        from runitback.wager.cbb.data.download import build_games

        from .test_college import BOX, SCHEDULE, TODAY

        return build_games(SCHEDULE, BOX, {}, TODAY)
    if sport == "wnba":
        from runitback.wager.wnba.data.download import build_games

        from .test_wnba import TODAY, schedule

        return build_games(schedule(), pd.DataFrame(), today=TODAY)
    from runitback.wager.mlb.data.download import build_games

    from .test_mlb import LINES, LOGS, PEOPLE, SCHEDULE, TEAMS, TODAY

    return build_games([SCHEDULE], LOGS, PEOPLE, TEAMS, LINES, today=TODAY)


@pytest.mark.parametrize("sport", ["cbb", "wcbb", "wnba", "mlb"])  # nhl is ported later
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


def bracket(rows):
    out = []
    for gid, (team, opp, result, day) in enumerate(rows):
        for t, o, r in ((team, opp, result), (opp, team, -result)):
            out.append({"season": 2025, "day": day, "game_id": gid, "team": t,
                        "opp_team": o, "result": r, "game_type": "ncaa"})  # fmt: skip
    return pd.DataFrame(out)


def test_college_champion_runner_up_and_final_four():
    from runitback.wager.web.sports.cbb import playoffs

    games = bracket(
        [
            ("Florida", "Auburn", 6, 1),
            ("Houston", "Duke", 3, 1),
            ("Florida", "Houston", 2, 3),
        ]
    )
    assert playoffs(games) == {
        2025: {
            "Florida": "champion",
            "Houston": "runner_up",
            "Auburn": "semifinal",
            "Duke": "semifinal",
        }
    }


def test_wnba_finals_lengths():
    from runitback.wager.web.sports.wnba import finals_wins, playoffs

    assert (finals_wins(2004), finals_wins(2024), finals_wins(2025)) == (2, 3, 4)
    games = bracket([("LV", "NY", 5, d) for d in range(1, 4)])
    games["game_type"] = "playoffs"
    assert playoffs(games) == {}  # 2025: best of seven, three wins are not enough
    games["season"] = 2024
    assert playoffs(games)[2024] == {"LV": "champion", "NY": "runner_up"}
