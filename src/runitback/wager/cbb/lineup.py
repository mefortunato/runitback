"""College basketball's player-value lineup layout, shared by men's (``cbb``) and women's (``wcbb``)
(``runitback college player-values`` writes its ``_pv_*`` columns; see
:class:`~runitback.wager.core.lineup.LineupSpec`).

As the NBA's (:mod:`runitback.wager.nba.lineup`): each position group (G / F / C, from the ESPN
position) has an offense and a defense column; offense adds to the total, defense takes it away. The
layout is not on ``SPORT`` yet: C4 (the college wager experiments) registers it with the ``pvc`` arms.
"""

from ..core.lineup import LineupSpec

POSITIONS = ("G", "F", "C")
OFFENSE = tuple(f"{g}_OFF" for g in POSITIONS)
DEFENSE = tuple(f"{g}_DEF" for g in POSITIONS)

LINEUP = LineupSpec(
    groups=(*OFFENSE, *DEFENSE),
    total_sign=(1.0,) * len(OFFENSE) + (-1.0,) * len(DEFENSE),
    # _pv_6 offense / _pv_7 defense change vs the previous game's lineup
    summaries=(("offense", OFFENSE), ("defense", DEFENSE)),
    # a quarter of the wager model's 32-game history
    recent=8,
)
