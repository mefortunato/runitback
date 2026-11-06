"""Per-game efficiency (rate) stats derived from each sport's box score.

A history token already carries the raw box score, but counts mix how well
a team played with how many chances it had (pace, plays, plate
appearances). The rates the analytics community found most predictive of
future results are hard for a network to rebuild from standardized counts,
so each sport derives them here:

- **NFL**: play-by-play EPA per play, success rate, early-down and neutral
  game-state EPA, explosive plays, special teams and drive scoring
  (:mod:`runitback.wager.nfl.data.pbp`), plus yards per play, net yards per
  dropback, sack and turnover rates from the box score.
- **College football**: yards per play, points per play, third / fourth
  down conversion, turnover, first-down and pass rates.
- **Basketball** (NBA, WNBA, men's and women's college): possessions, pace,
  offensive / defensive / net rating per 100 possessions and Dean Oliver's
  four factors (eFG%, turnover %, offensive rebound %, free-throw rate),
  plus 3-point attempt rate, assist rate and steal / block rates.
- **NHL**: shot-attempt (Corsi) and unblocked-attempt (Fenwick) shares,
  shot share, shooting and save percentage without empty-net goals, PDO and
  power-play conversion.
- **MLB**: OBP, SLG, OPS, ISO, strikeout / walk / home-run rates, BABIP
  and a FIP-style pitching line (from the opponent's batting).

Each function takes a games table (one row per team per game, the
opponent's box score in ``opp_*`` columns) and returns new columns, all
prefixed ``_eff_``: ``_eff_<stat>`` for the team's offense and
``_eff_opp_<stat>`` for the same stat by the opponent (what the team's
defense allowed). Stats that are the same from both sides (pace, shares)
appear once. Missing inputs give NaN, which the models treat as average.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

# side -> {stat: Series}; ``own(col)`` / ``opp(col)`` read one side's column
Getter = Callable[[str], pd.Series]


def _getter(games: pd.DataFrame, prefix: str) -> Getter:
    def get(col: str) -> pd.Series:
        name = prefix + col
        if name not in games:
            return pd.Series(np.nan, index=games.index)
        return pd.to_numeric(games[name], errors="coerce").astype(float)

    return get


def ratio(num, den) -> pd.Series:
    """``num / den``, NaN where ``den`` is not positive."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den.where(den > 0)


def both_sides(
    games: pd.DataFrame,
    side: Callable[[Getter, Getter], dict[str, pd.Series]],
    shared: Callable[[Getter, Getter], dict[str, pd.Series]] | None = None,
) -> pd.DataFrame:
    """``side(own, opp)`` for the team (``_eff_<stat>``) and mirrored for
    the opponent (``_eff_opp_<stat>``), plus ``shared(own, opp)`` once."""
    team, opp = _getter(games, ""), _getter(games, "opp_")
    out = {f"_eff_{k}": v for k, v in side(team, opp).items()}
    out |= {f"_eff_opp_{k}": v for k, v in side(opp, team).items()}
    if shared is not None:
        out |= {f"_eff_{k}": v for k, v in shared(team, opp).items()}
    frame = pd.DataFrame(out, index=games.index)
    return frame.replace([np.inf, -np.inf], np.nan)


# --- NFL --------------------------------------------------------------------
NFL_PBP = [
    "pbp_epa_play", "pbp_success", "pbp_pass_epa", "pbp_rush_epa",
    "pbp_early_epa", "pbp_early_pass_rate", "pbp_neutral_epa",
    "pbp_explosive", "pbp_turnover_epa", "pbp_st_epa", "pbp_drive_score",
]  # fmt: skip


def _nfl_side(own: Getter, opp: Getter) -> dict[str, pd.Series]:
    dropbacks = own("attempts") + own("sacks_suffered")
    plays = dropbacks + own("carries")
    net_pass = own("passing_yards") - own("sack_yards_lost")
    turnovers = (
        own("passing_interceptions")
        + own("sack_fumbles_lost")
        + own("rushing_fumbles_lost")
        + own("receiving_fumbles_lost").fillna(0)
    )
    out = {
        "yards_play": ratio(net_pass + own("rushing_yards"), plays),
        "net_yards_dropback": ratio(net_pass, dropbacks),
        "rush_yards_carry": ratio(own("rushing_yards"), own("carries")),
        "sack_rate": ratio(own("sacks_suffered"), dropbacks),
        "turnover_rate": ratio(turnovers, plays),
        "first_down_rate": ratio(
            own("passing_first_downs") + own("rushing_first_downs"), plays
        ),
        "box_epa_play": ratio(own("passing_epa") + own("rushing_epa"), plays),
        "pass_rate": ratio(dropbacks, plays),
        "points_play": ratio(own("score"), plays),
    }
    return out | {stat: own(stat) for stat in NFL_PBP}


def nfl_efficiency(games: pd.DataFrame) -> pd.DataFrame:
    return both_sides(games, _nfl_side)


# --- College football -------------------------------------------------------
def _cfb_side(own: Getter, opp: Getter) -> dict[str, pd.Series]:
    plays = own("passingAttempts") + own("rushingAttempts")
    return {
        "yards_play": ratio(own("totalYards"), plays),
        "points_play": ratio(own("points"), plays),
        "pass_yards_att": ratio(own("netPassingYards"), own("passingAttempts")),
        "rush_yards_carry": ratio(own("rushingYards"), own("rushingAttempts")),
        "third_down_rate": ratio(own("thirdDownConv"), own("thirdDownAtt")),
        "fourth_down_rate": ratio(own("fourthDownConv"), own("fourthDownAtt")),
        "turnover_rate": ratio(own("turnovers"), plays),
        "first_down_rate": ratio(own("firstDowns"), plays),
        "pass_rate": ratio(own("passingAttempts"), plays),
        "penalty_yards_play": ratio(own("totalPenaltiesYards"), plays),
    }


def cfb_efficiency(games: pd.DataFrame) -> pd.DataFrame:
    def shared(own: Getter, opp: Getter):
        plays = own("passingAttempts") + own("rushingAttempts")
        opp_plays = opp("passingAttempts") + opp("rushingAttempts")
        return {
            "plays": plays + opp_plays,
            "yards_play_net": ratio(own("totalYards"), plays)
            - ratio(opp("totalYards"), opp_plays),
        }

    return both_sides(games, _cfb_side, shared)


# --- Basketball -------------------------------------------------------------
def possessions(own: Getter, opp: Getter) -> pd.Series:
    """Average of both teams' ``FGA - OREB + TOV + 0.44 FTA``."""

    def one(side: Getter) -> pd.Series:
        return side("fga") - side("oreb") + side("tov") + 0.44 * side("fta")

    return (one(own) + one(opp)) / 2


def _basketball_side(own: Getter, opp: Getter) -> dict[str, pd.Series]:
    poss = possessions(own, opp)
    return {
        "rating": 100 * ratio(own("points"), poss),
        "efg": ratio(own("fgm") + 0.5 * own("fg3m"), own("fga")),
        "tov_pct": ratio(own("tov"), poss),
        "orb_pct": ratio(own("oreb"), own("oreb") + opp("dreb")),
        "ft_rate": ratio(own("ftm"), own("fga")),
        "three_rate": ratio(own("fg3a"), own("fga")),
        "three_pct": ratio(own("fg3m"), own("fg3a")),
        "two_pct": ratio(own("fgm") - own("fg3m"), own("fga") - own("fg3a")),
        "ast_rate": ratio(own("ast"), own("fgm")),
        "stl_pct": ratio(own("stl"), poss),
        "blk_pct": ratio(own("blk"), opp("fga") - opp("fg3a")),
    }


def basketball_efficiency(games: pd.DataFrame) -> pd.DataFrame:
    def shared(own: Getter, opp: Getter):
        poss = possessions(own, opp)
        return {
            "pace": poss,
            "net_rating": 100 * ratio(own("points") - opp("points"), poss),
        }

    return both_sides(games, _basketball_side, shared)


# --- NHL --------------------------------------------------------------------
def _nhl_side(own: Getter, opp: Getter) -> dict[str, pd.Series]:
    # ``blocked_shots`` are the shots a team blocked, i.e. the opponent's
    goals = own("goals") - own("empty_net_goals").fillna(0)
    attempts = own("shots") + own("missed_shots") + opp("blocked_shots")
    return {
        "shot_attempts": attempts,
        "shooting_pct": ratio(goals, own("shots")),
        "goals_attempt": ratio(goals, attempts),
        "pp_conversion": ratio(own("pp_goals"), own("pp_opportunities")),
        "pp_opportunities": own("pp_opportunities"),
        "hits": own("hits"),
        "giveaways": own("giveaways"),
    }


def nhl_efficiency(games: pd.DataFrame) -> pd.DataFrame:
    def shared(own: Getter, opp: Getter):
        cf = own("shots") + own("missed_shots") + opp("blocked_shots")
        ca = opp("shots") + opp("missed_shots") + own("blocked_shots")
        ff = own("shots") + own("missed_shots")
        fa = opp("shots") + opp("missed_shots")
        sh = ratio(own("goals") - own("empty_net_goals").fillna(0), own("shots"))
        sv = 1 - ratio(opp("goals") - opp("empty_net_goals").fillna(0), opp("shots"))
        return {
            "corsi_pct": ratio(cf, cf + ca),
            "fenwick_pct": ratio(ff, ff + fa),
            "shot_share": ratio(own("shots"), own("shots") + opp("shots")),
            "save_pct": sv,
            "pdo": sh + sv,
        }

    return both_sides(games, _nhl_side, shared)


# --- MLB --------------------------------------------------------------------
def _mlb_side(own: Getter, opp: Getter) -> dict[str, pd.Series]:
    ab, pa = own("at_bats"), own("plate_appearances")
    on_base = own("hits") + own("walks") + own("hit_by_pitch")
    obp = ratio(on_base, ab + own("walks") + own("hit_by_pitch") + own("sac_flies"))
    slg = ratio(own("total_bases"), ab)
    in_play = ab - own("strikeouts") - own("home_runs") + own("sac_flies")
    return {
        "obp": obp,
        "slg": slg,
        "ops": obp + slg,
        "iso": ratio(own("total_bases") - own("hits"), ab),
        "k_pct": ratio(own("strikeouts"), pa),
        "bb_pct": ratio(own("walks"), pa),
        "hr_rate": ratio(own("home_runs"), pa),
        "babip": ratio(own("hits") - own("home_runs"), in_play),
        "runs_pa": ratio(own("runs"), pa),
        # FIP-style line of the pitchers who faced these batters
        "fip_pa": ratio(
            13 * own("home_runs")
            + 3 * (own("walks") + own("hit_by_pitch"))
            - 2 * own("strikeouts"),
            pa,
        ),
        "pitches_pa": ratio(opp("pitches"), pa),
        "errors": own("errors"),
    }


def mlb_efficiency(games: pd.DataFrame) -> pd.DataFrame:
    return both_sides(games, _mlb_side)
