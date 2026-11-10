"""The NHL's player-value lineup layout (``runitback nhl player-values`` writes its ``_pv_*`` columns;
see :class:`~runitback.wager.core.lineup.LineupSpec`).

Every skater plays both ends, so forwards and defensemen each have an offense column (expected goals
for) and a defense column (expected goals against prevented), all in goals per game over
replacement. The starting goalie has his own column (goals saved above expected). Offense adds to
the total; defense and the goalie take it away. The goalie also has his own previous-game change
column, since the starter is the largest swing between games. The layout is not on ``SPORT`` yet:
H6 (the NHL wager experiments) registers it when it adds the ``pvc`` arms.
"""

from ..core.lineup import LineupSpec

POSITIONS = ("F", "D")
OFFENSE = tuple(f"{g}_OFF" for g in POSITIONS)
DEFENSE = tuple(f"{g}_DEF" for g in POSITIONS)
GOALIE = "G"

LINEUP = LineupSpec(
    groups=(*OFFENSE, *DEFENSE, GOALIE),
    # better offense: more goals; better defense or goalie: fewer
    total_sign=(1.0,) * len(OFFENSE) + (-1.0,) * len(DEFENSE) + (-1.0,),
    # _pv_5 offense / _pv_6 defense / _pv_7 goalie change vs the previous game's lineup
    summaries=(("offense", OFFENSE), ("defense", DEFENSE), ("goalie", (GOALIE,))),
    # a quarter of the wager model's 40-game history
    recent=10,
)
