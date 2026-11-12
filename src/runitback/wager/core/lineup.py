"""A sport's player-value lineup layout: the position groups behind the wager model's ``_pv_*``
columns, how the additive lineup terms read them, and what the game pages' lineup editor shows.

A sport's player-value table (``pv.parquet``, one row per team-game) has these columns, for ``n``
position groups and ``k`` previous-game summaries:

    _pv_0 .. _pv_{n-1}            each group's projected-lineup value (points over replacement)
    _pv_{n} .. _pv_{n+k-1}        each summary's change vs the previous game's lineup
    _pv_{n+k} .. _pv_{2n+k-1}     each group's change vs the mean of the team's last ``recent`` lineups

The lineup sets read the first ``n + k`` (``pv``) or all ``2n + k`` (``pva``, ``pvc``) of them. In the
sign-constrained additive terms (:class:`~runitback.wager.models.transformer.model.PairHead`) a better
lineup in any group always moves the team's margin up, and the total by ``total_sign`` (+1 a group
that scores, -1 one that prevents scoring, 0 neither).
"""

from __future__ import annotations

from dataclasses import dataclass, field

PV_PREFIX = "_pv_"


@dataclass(frozen=True)
class LineupSpec:
    """Attributes:
        groups: The value columns' position groups, in ``_pv_`` order.
        total_sign: Each group's sign on the total (same order as ``groups``).
        summaries: The previous-game change columns after the levels: ``(name, groups summed)``.
        recent: Lineups behind each group's change column (about the wager model's history).
        editor_groups: The lineup editor's groups (roster positions), in display order.
        labels: Each editor group's display name.
        editor_columns: Editor groups whose values add into another group's column
            (``{"K": "ST"}``); the rest map to the column of the same name.
        rotation_groups: Editor groups that are one ranked rotation for the whole team (the NBA's):
            each player adds a value into several columns (his position's offense and defense),
            at the slot his rank gives, so the group maps to no single column.
    """

    groups: tuple[str, ...]
    total_sign: tuple[float, ...]
    summaries: tuple[tuple[str, tuple[str, ...]], ...] = ()
    recent: int = 8
    editor_groups: tuple[str, ...] = ()
    labels: dict[str, str] = field(default_factory=dict)
    editor_columns: dict[str, str] = field(default_factory=dict)
    rotation_groups: tuple[str, ...] = ()

    def __post_init__(self):
        if len(self.total_sign) != len(self.groups):
            raise ValueError("total_sign needs one sign per group")
        unknown = {self.column_of(g) for g in self.editor_groups if g not in self.rotation_groups} - set(self.groups)
        if unknown:
            raise ValueError(f"editor groups map to unknown columns {sorted(unknown)}")

    @property
    def levels(self) -> tuple[int, ...]:
        """``_pv_`` indices of the groups' values."""
        return tuple(range(len(self.groups)))

    @property
    def changes(self) -> tuple[int, ...]:
        """``_pv_`` indices of the groups' changes vs the team's recent lineups."""
        start = len(self.groups) + len(self.summaries)
        return tuple(range(start, start + len(self.groups)))

    def width(self, mode: str) -> int:
        """Columns a player-value lineup set reads: ``pv`` levels and summaries, ``pva`` / ``pvc``
        every column."""
        if mode == "pv":
            return len(self.groups) + len(self.summaries)
        if mode in ("pva", "pvc"):
            return 2 * len(self.groups) + len(self.summaries)
        raise ValueError(f"{mode!r} is not a player-value lineup set")

    @property
    def columns(self) -> list[str]:
        """Every ``_pv_`` column of the sport's player-value table."""
        return [f"{PV_PREFIX}{i}" for i in range(self.width("pvc"))]

    def group_of_index(self, index: int) -> int:
        """The group (position in ``groups``) of a level or change column index."""
        if index in self.levels:
            return index
        if index in self.changes:
            return index - self.changes[0]
        raise ValueError(f"_pv_{index} is not a group column")

    def column_of(self, editor_group: str) -> str:
        """The value column an editor group's players add into."""
        return self.editor_columns.get(editor_group, editor_group)
