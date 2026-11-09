import numpy as np
import pandas as pd
import pytest
import scipy.stats

from runitback.wager.core.schema import GameSchema
from runitback.wager.core.sport import Sport, register_sport

TOY_SCHEMA = GameSchema(
    season="season",
    date="week",
    team="team",
    opp="opp",
    game_id="game_id",
    target="margin",
    stats=("points", "yards", "opp_points", "opp_yards"),
    location="location",
    context=("rest", "opp_rest"),
    categorical=("surface",),
    paired=(("rest", "opp_rest"), ("moneyline", "opp_moneyline")),
    negated=("spread",),
    market_spread="spread",
    market_moneyline=("moneyline", "opp_moneyline"),
    total="total",
    market_total="total_line",
    market_total_odds=("over_odds", "under_odds"),
)


def american(prob: float) -> float:
    """American odds for ``prob`` with a ~4.5% bookmaker margin."""
    prob = min(max(prob * 1.0225, 0.01), 0.99)
    return -100 * prob / (1 - prob) if prob >= 0.5 else 100 * (1 - prob) / prob


def make_games(
    n_teams: int = 12,
    seasons=(2021, 2022, 2023, 2024),
    weeks: int = 12,
    upcoming_weeks: int = 1,
    home_advantage: float = 3.0,
    noise: float = 6.0,
    seed: int = 0,
    persistence: float = 0.0,
) -> pd.DataFrame:
    """Toy games with known per-season offense/defense ratings.

    ``persistence`` is the season-to-season correlation of team ratings (real
    leagues are well above 0; the default 0 redraws them every season).

    One row per team per game. The last ``upcoming_weeks`` of the last season
    are unplayed (missing target and stats).
    """
    rng = np.random.default_rng(seed)
    rows = []
    off = dfn = None
    carry = np.sqrt(1 - persistence**2)
    for season in seasons:
        new_off, new_dfn = rng.normal(0, 5, n_teams), rng.normal(0, 5, n_teams)
        if off is None:
            off, dfn = new_off, new_dfn
        else:
            off = persistence * off + carry * new_off
            dfn = persistence * dfn + carry * new_dfn
        for week in range(1, weeks + 1):
            order = rng.permutation(n_teams)
            for a, b in order.reshape(-1, 2):
                loc = rng.choice([1, 1, 1, 0])
                rest_a, rest_b = rng.integers(5, 9, 2)
                pts_a = 21 + off[a] - dfn[b] + home_advantage * loc / 2
                pts_b = 21 + off[b] - dfn[a] - home_advantage * loc / 2
                # a market that knows the true expected margin and total,
                # to the half point
                spread = round(2 * (pts_a - pts_b)) / 2
                total_line = round(2 * (pts_a + pts_b)) / 2
                p_a = scipy.stats.norm.cdf((pts_a - pts_b) / (noise * np.sqrt(2)))
                ml_a, ml_b = american(p_a), american(1 - p_a)
                pts_a += rng.normal(0, noise)
                pts_b += rng.normal(0, noise)
                yds_a = 300 + 12 * (off[a] - dfn[b]) + rng.normal(0, 40)
                yds_b = 300 + 12 * (off[b] - dfn[a]) + rng.normal(0, 40)
                game_id = f"{season}_{week:02d}_{a}_{b}"
                surface = rng.choice(["grass", "turf"])
                for team, opp, pts, opts, yds, oyds, sign, rest, orest, ml, oml in [
                    (a, b, pts_a, pts_b, yds_a, yds_b, 1, rest_a, rest_b, ml_a, ml_b),
                    (b, a, pts_b, pts_a, yds_b, yds_a, -1, rest_b, rest_a, ml_b, ml_a),
                ]:
                    rows.append(
                        {
                            "season": season,
                            "week": week,
                            "game_id": game_id,
                            "team": f"T{team:02d}",
                            "opp": f"T{opp:02d}",
                            "location": sign * loc,
                            "rest": rest,
                            "opp_rest": orest,
                            "surface": surface,
                            "points": round(pts),
                            "opp_points": round(opts),
                            "yards": yds,
                            "opp_yards": oyds,
                            "margin": round(pts) - round(opts),
                            "total": round(pts) + round(opts),
                            "spread": sign * spread,
                            "total_line": total_line,
                            "over_odds": -110.0,
                            "under_odds": -110.0,
                            "moneyline": ml,
                            "opp_moneyline": oml,
                        }
                    )
    games = pd.DataFrame(rows)
    last = games["season"] == max(seasons)
    unplayed = last & (games["week"] > weeks - upcoming_weeks)
    games.loc[unplayed, ["margin", "total", *TOY_SCHEMA.stats]] = np.nan
    return games


@pytest.fixture(scope="session", autouse=True)
def mlflow_tracking(tmp_path_factory):
    """Keep MLflow's tracking database out of the working directory."""
    path = tmp_path_factory.mktemp("mlflow") / "mlflow.db"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{path}")
        yield


@pytest.fixture(scope="session", autouse=True)
def toy_sport():
    return register_sport(
        Sport(
            name="toy",
            schema=TOY_SCHEMA,
            download=lambda output_file=None: make_games(),
            gnn={"hidden_dim": 16, "out_dim": 8, "num_layers": 2},
            transformer={
                "history": 8,
                "d_model": 16,
                "nhead": 2,
                "num_layers": 1,
                "total_weight": 1.0,
            },
        )
    )


@pytest.fixture(scope="session")
def toy_games():
    return make_games()


@pytest.fixture(scope="session", autouse=True)
def toy_view(toy_sport):
    from runitback.wager.web.views import SportView, Stat, column, register_view

    teams = pd.DataFrame(
        {
            "team": [f"T{i:02d}" for i in range(12)],
            "name": [f"Team {i}" for i in range(12)],
            "nickname": [f"T{i}" for i in range(12)],
            "color": "#123456",
            "color2": "#abcdef",
            "logo": "https://example.invalid/logo.png",
            "conference": "A",
            "division": "A1",
        }
    )
    return register_view(
        SportView(
            sport="toy",
            label="Toy",
            date_label="Week",
            stats=(
                Stat("points", "Points", column("points")),
                Stat("allowed", "Allowed", column("opp_points"), 1, False),
                Stat("yards", "Yards", column("yards"), 0),
            ),
            load_teams=lambda: teams,
            score=("points", "opp_points"),
        )
    )
