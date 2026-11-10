import datetime
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from runitback.wager.nfl.data.names import odds_map
from runitback.wager.odds.providers import ESPN, OddsAPI
from runitback.wager.web.odds import build_odds, load_previous, match_games, refresh_odds

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def fake_get(name, headers=None):
    payload = json.loads((FIXTURES / name).read_text())

    def get(url, params, *rest):
        get.calls.append((url, params))
        return payload, headers or {}

    get.calls = []
    return get


def nfl_lines():
    get = fake_get("oddsapi_nfl.json", {"x-requests-remaining": "147"})
    provider = OddsAPI("nfl", names=odds_map, api_key="k", get=get)
    return provider, provider.fetch()


def test_odds_api_lines_from_the_home_side(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    provider, lines = nfl_lines()
    assert provider.remaining == 147
    url, params = provider._get.calls[0]
    assert url.endswith("/sports/americanfootball_nfl/odds")
    assert params["markets"] == "h2h,spreads,totals" and params["regions"] == "us"
    kc = lines[lines.home == "KC"].set_index("book")
    assert set(kc.index) == {"DraftKings", "FanDuel"}
    # KC -2.5: home favored by 2.5
    assert kc.loc["DraftKings", "spread"] == 2.5
    assert kc.loc["FanDuel", ["spread_home_odds", "spread_away_odds"]].tolist() == [
        -115,
        -105,
    ]
    dk = kc.loc["DraftKings", ["ml_home", "ml_away", "total", "over_odds"]]
    assert dk.tolist() == [-142, 120, 46.5, -108]
    dal = lines[lines.home == "DAL"].iloc[0]
    assert pd.isna(dal.spread) and dal.ml_home == 130
    with pytest.raises(RuntimeError, match="ODDS_API_KEY"):
        OddsAPI("nfl", api_key="", get=fake_get("oddsapi_nfl.json")).fetch()


def espn_provider(**kw):
    from .cfb_fixtures import ODDS, SCHEDULE, item

    odds = ODDS | {
        "11": [
            item("DraftKings", -3.0, 52.5, -150, 130, home_price=-105,
                 away_price=-115, over=-110, under=-110,
                 home_open={"pointSpread": {"american": "-2.5"}},
                 open={"total": {"american": "51.5"}}),
            item("Opening", -1.0, 50.0),
            item("DraftKings - Live Odds", 7.0, 70.5),
        ]
    }  # fmt: skip
    calls = []

    def get(game_id):
        calls.append(game_id)
        return odds.get(game_id, [])

    today = datetime.datetime(2099, 12, 28, tzinfo=datetime.timezone.utc)
    provider = ESPN(schedule=SCHEDULE, odds=get, today=today, **kw)
    return provider, calls


def test_espn_lines_of_the_coming_games_from_the_home_side():
    provider, calls = espn_provider()
    lines = provider.fetch()
    assert calls == ["11"]  # only the upcoming FBS game within 8 days
    assert lines["book"].tolist() == ["DraftKings"]  # no opening or live lines
    dk = lines.iloc[0]
    assert (dk.home, dk.away, dk.spread, dk.spread_open) == (
        "Georgia",
        "Auburn",
        3.0,
        2.5,
    )
    assert (dk.spread_home_odds, dk.spread_away_odds, dk.ml_home, dk.ml_away) == (
        -105,
        -115,
        -150,
        130,
    )
    assert (dk.total, dk.total_open, dk.over_odds) == (52.5, 51.5, -110)
    assert dk.commence == pd.Timestamp("2099-12-31T20:00Z")
    provider, calls = espn_provider(days=2)
    assert provider.fetch().empty and calls == []


def test_espn_pools_fcs_sides_like_the_games_table():
    from .cfb_fixtures import SCHEDULE, item

    game = SCHEDULE.iloc[[0]].assign(
        completed=False, home_points=None, away_points=None,
        start_date="2024-11-23T19:00:00.000Z",
    )  # fmt: skip
    provider = ESPN(
        schedule=game,
        odds=lambda gid: [item("DraftKings", -40.5, 55.5)],
        today=datetime.datetime(2024, 11, 20, tzinfo=datetime.timezone.utc),
    )
    lines = provider.fetch()
    assert list(lines[["home", "away"]].iloc[0]) == ["Alabama", "FCS"]
    assert lines["spread"].iloc[0] == 40.5


def card(game_id, home, away, kickoff):
    return {
        "game_id": game_id,
        "home": {"team": home},
        "away": {"team": away},
        "kickoff": kickoff,
    }


CARDS = [
    card("2024_09_BUF_KC", "KC", "BUF", "2024-11-03 20:20"),
    # listed the other way round from the provider: lines must be flipped
    card("2024_09_DAL_ATL", "ATL", "DAL", "2024-11-03 13:00"),
    card("2024_10_KC_BUF", "BUF", "KC", "2024-11-17 16:25"),  # rematch, later
]


def test_lines_match_the_nearest_game_and_flip_to_its_home_side():
    _, lines = nfl_lines()
    matched = match_games(lines, CARDS)
    assert set(matched.game_id) == {"2024_09_BUF_KC", "2024_09_DAL_ATL"}
    atl = matched[matched.game_id == "2024_09_DAL_ATL"].iloc[0]
    assert (atl.home, atl.ml_home, atl.ml_away) == ("DAL", -150, 130)  # flipped prices


def test_lines_match_on_display_names_without_a_name_map():
    """College basketball: sportsbooks name schools "Duke Blue Devils"."""
    lines = pd.DataFrame(
        [{"home": "North Carolina Tar Heels", "away": "Duke Blue Devils",
          "commence": pd.Timestamp("2026-03-07T23:30Z"), "spread": -2.5,
          "spread_open": np.nan, "spread_home_odds": -110, "spread_away_odds": -110,
          "ml_home": -140, "ml_away": 120}]
    )  # fmt: skip
    cards = [
        {"game_id": "1", "kickoff": "2026-03-07 18:30",
         "home": {"team": "Duke", "name": "Duke Blue Devils"},
         "away": {"team": "North Carolina", "name": "North Carolina Tar Heels"}}
    ]  # fmt: skip
    row = match_games(lines, cards).iloc[0]
    assert (row.game_id, row.spread, row.ml_home, row.ml_away) == ("1", 2.5, 120, -140)


def test_hockey_and_baseball_request_no_spreads():
    get = fake_get("oddsapi_nfl.json")
    OddsAPI("nhl", api_key="k", get=get).fetch()
    OddsAPI("nba", api_key="k", get=get).fetch()
    assert get.calls[0][0].endswith("/sports/icehockey_nhl/odds")
    assert [params["markets"] for _, params in get.calls] == [
        "h2h,totals",
        "h2h,spreads,totals",
    ]


def test_odds_payload_has_consensus_best_prices_and_history():
    _, lines = nfl_lines()
    previous = {
        "games": {
            "2024_09_BUF_KC": {
                "history": [
                    {"t": "2024-11-01T12:00:00+00:00", "spread": 1.5, "total": 47.0}
                ]
            },
            "2024_10_KC_BUF": {"books": [], "history": [{"t": "x", "spread": -1.0}]},
            "2023_18_OLD": {"history": []},
        }
    }
    odds = build_odds(
        lines,
        CARDS,
        previous,
        now="2024-11-03T15:05:00+00:00",
        provider="oddsapi",
        remaining=147,
    )
    assert odds["provider"] == "oddsapi" and odds["remaining"] == 147
    kc = odds["games"]["2024_09_BUF_KC"]
    assert kc["consensus"]["spread"] == 2.75 and kc["consensus"]["total"] == 46.25
    best = kc["best"]
    # KC needs the fewest points (DraftKings 2.5), BUF the most (FanDuel +3)
    assert best["spread"]["home"] == {"book": "DraftKings", "odds": -110, "points": 2.5}
    assert best["spread"]["away"] == {"book": "FanDuel", "odds": -105, "points": 3.0}
    assert best["moneyline"]["home"]["book"] == "DraftKings"  # -142 beats -148
    assert best["moneyline"]["away"]["book"] == "FanDuel"  # +125 beats +120
    assert (
        best["total"]["over"]["points"] == 46.0
        and best["total"]["under"]["points"] == 46.5
    )
    # the history grows from the published file; the first point opens
    assert [p["spread"] for p in kc["history"]] == [1.5, 2.75]
    assert kc["open"]["spread"] == 1.5
    # a game this refresh missed keeps its lines; finished games are dropped
    assert "2024_10_KC_BUF" in odds["games"] and "2023_18_OLD" not in odds["games"]
    atl = odds["games"]["2024_09_DAL_ATL"]
    assert atl["consensus"]["spread"] is None and atl["best"]["spread"]["home"] is None


def test_refresh_skips_without_games_soon_and_survives_a_failed_fetch(tmp_path):
    provider, _ = nfl_lines()
    previous = {"games": {}, "updated": "earlier"}
    far = [card("g", "KC", "BUF", "2024-12-25 16:00")]
    now = datetime.datetime(2024, 11, 3, 12, tzinfo=datetime.timezone.utc)
    calls = len(provider._get.calls)
    assert refresh_odds(provider, far, previous, now=now) is previous
    assert len(provider._get.calls) == calls  # no request spent off-season

    class Broken:
        name, remaining = "broken", None

        def fetch(self):
            raise ConnectionError("down")

    assert refresh_odds(Broken(), CARDS, previous, now=now) is previous
    fresh = refresh_odds(provider, CARDS, None, now=now)
    assert set(fresh["games"]) == {"2024_09_BUF_KC", "2024_09_DAL_ATL"}

    (tmp_path / "nfl").mkdir()
    (tmp_path / "nfl" / "odds.json").write_text(json.dumps(fresh))
    assert load_previous(str(tmp_path), "nfl") == fresh
    assert load_previous(str(tmp_path), "cfb") is None
    assert load_previous("http://127.0.0.1:9/none", "nfl") is None


def test_export_writes_odds_when_asked(webdata_for_odds, tmp_path, monkeypatch):
    from runitback.wager.web import export as export_module

    seen = {}

    def fake_refresh(provider, cards, previous):
        seen["provider"], seen["cards"] = provider, cards
        return {"updated": "now", "provider": "oddsapi", "remaining": 99, "games": {}}

    monkeypatch.setattr(export_module, "refresh_odds", fake_refresh)
    monkeypatch.setattr(
        export_module, "provider_for", lambda sport, name: (sport, name)
    )
    export_module.export(
        data_dir=str(webdata_for_odds),
        output_dir=str(tmp_path),
        odds={"toy": "oddsapi"},
    )
    saved = json.loads((tmp_path / "data" / "toy" / "odds.json").read_text())
    assert saved["remaining"] == 99 and seen["provider"] == ("toy", "oddsapi")
    assert seen["cards"] and all("game_id" in c for c in seen["cards"])


@pytest.fixture(scope="module")
def webdata_for_odds(tmp_path_factory):
    from runitback.wager.web.build import build

    from .conftest import make_games

    out = tmp_path_factory.mktemp("webdata")
    build(
        sports=["toy"],
        output_dir=str(out),
        games=make_games(),
        backtest_seasons=1,
        max_epochs=1,
    )
    return out
