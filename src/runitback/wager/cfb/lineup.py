"""College football's player-value lineup layout (``runitback cfb player-values`` writes its
``_pv_*`` columns; see :class:`~runitback.wager.core.lineup.LineupSpec`).

College play-by-play has no on-field personnel, so the only player value is the starting
quarterback's (``runitback.cfb.player_values``): EPA per quarterback play over replacement, in
points per game. The layout is not on ``SPORT`` yet: F3 (the CFB wager experiments) registers it when
it adds the ``pbp32_qb`` arm.
"""

from ..core.lineup import LineupSpec

LINEUP = LineupSpec(
    groups=("QB",),
    # a better quarterback scores more
    total_sign=(1.0,),
    # _pv_1: change vs the previous game's starter
    summaries=(("qb", ("QB",)),),
    # a quarter of the wager model's 28-game history
    recent=7,
    editor_groups=("QB",),
    labels={"QB": "Quarterback"},
)
