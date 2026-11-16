"""The NBA's player-value lineup layout (``runitback nba player-values`` writes its ``_pv_*`` columns;
see :class:`~runitback.wager.core.lineup.LineupSpec`).

Every NBA player plays both ends, so each position group (G / F / C, from the ESPN position) has an
offense and a defense column. Offense adds to the total, defense takes it away. Registered on the
NBA ``SPORT`` (``Sport.lineup``) for the ``pv`` / ``pva`` / ``pvc`` lineup sets (experiments/nba).
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
    # a quarter of the wager model's 40-game history
    recent=10,
    # the lineup editor: one ranked rotation (the model's slots are team-wide minutes ranks, not per
    # position); each player adds his offense and defense into his G / F / C columns
    editor_groups=("ROT",),
    labels={"ROT": "Rotation"},
    rotation_groups=("ROT",),
)
