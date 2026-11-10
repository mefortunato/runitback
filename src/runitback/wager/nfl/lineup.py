"""The NFL's player-value lineup layout (``runitback nfl player-values`` writes its ``_pv_*``
columns; see :class:`~runitback.wager.core.lineup.LineupSpec`)."""

from ..core.lineup import LineupSpec

OFFENSE = ("QB", "RB", "WR", "TE", "OL")
DEFENSE = ("DL", "LB", "DB")

LINEUP = LineupSpec(
    # kicker and punter are summed into ST
    groups=(*OFFENSE, *DEFENSE, "ST"),
    # better offense: more points; better defense: fewer; kicking: neither way
    total_sign=(1.0,) * len(OFFENSE) + (-1.0,) * len(DEFENSE) + (0.0,),
    # _pv_9 offense / _pv_10 defense change vs the previous game's lineup
    summaries=(("offense", OFFENSE), ("defense", DEFENSE)),
    recent=8,
    editor_groups=(*OFFENSE, *DEFENSE, "K", "P"),
    labels={"QB": "Quarterback", "RB": "Running back", "WR": "Wide receiver", "TE": "Tight end",
            "OL": "Offensive line", "DL": "Defensive line", "LB": "Linebacker", "DB": "Defensive back",
            "K": "Kicker", "P": "Punter"},
    editor_columns={"K": "ST", "P": "ST"},
)
