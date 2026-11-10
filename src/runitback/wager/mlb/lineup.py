"""The MLB's player-value lineup layout (``runitback mlb player-values`` writes its ``_pv_*`` columns;
see :class:`~runitback.wager.core.lineup.LineupSpec`).

Three groups, all in runs per game over replacement: the batting lineup (the nine starters, each
weighted by his batting slot's plate appearances), the starting pitcher and the bullpen. A better
lineup scores more runs (total +), better pitching allows fewer (total -). The starting pitcher
keeps a term of his own, as the plan's M5 asks. The layout is not on ``SPORT`` yet: M5 (the MLB
wager experiments) registers it when it adds the ``pvc`` arms.
"""

from ..core.lineup import LineupSpec

GROUPS = ("BAT", "SP", "BP")

LINEUP = LineupSpec(
    groups=GROUPS,
    total_sign=(1.0, -1.0, -1.0),
    # _pv_3 lineup / _pv_4 pitching (SP + bullpen) change vs the previous game
    summaries=(("offense", ("BAT",)), ("pitching", ("SP", "BP"))),
    # two turns through a five-man rotation (the wager model's history is 50 games)
    recent=10,
)
