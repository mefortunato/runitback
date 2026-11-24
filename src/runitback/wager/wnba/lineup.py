"""The WNBA's player-value lineup layout (``runitback wnba player-values`` writes its ``_pv_*`` columns;
see :class:`~runitback.wager.core.lineup.LineupSpec`).

The NBA's layout (:mod:`runitback.wager.nba.lineup`), because the WNBA's values come from the same
pipeline (``runitback.nba.player_values`` with the WNBA's settings): each position group (G / F / C, from
the ESPN position) has an offense and a defense column; offense adds to the total, defense takes it away.
Registered on the WNBA ``SPORT`` (``Sport.lineup``) for the ``pv`` / ``pva`` / ``pvc`` lineup sets (W2,
experiments/wnba). Defined here rather than imported, so the WNBA's backtest code hash covers it.
"""

from ..core.lineup import LineupSpec

POSITIONS = ("G", "F", "C")
OFFENSE = tuple(f"{g}_OFF" for g in POSITIONS)
DEFENSE = tuple(f"{g}_DEF" for g in POSITIONS)

LINEUP = LineupSpec(
    groups=(*OFFENSE, *DEFENSE),
    # better offense: more points; better defense: fewer
    total_sign=(1.0,) * len(OFFENSE) + (-1.0,) * len(DEFENSE),
    # _pv_6 offense / _pv_7 defense change vs the previous game's lineup
    summaries=(("offense", OFFENSE), ("defense", DEFENSE)),
    # a quarter of the wager model's 40-game history (as the NBA's, so runitback.nba.player_values' RECENT)
    recent=10,
    # the lineup editor, as the NBA's: one ranked rotation (the model's slots are team-wide minutes ranks,
    # not per position); each player adds her offense and defense into her G / F / C columns
    editor_groups=("ROT",),
    labels={"ROT": "Rotation"},
    rotation_groups=("ROT",),
)
