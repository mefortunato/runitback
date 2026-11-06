from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SeasonSplit:
    """Seasons used to train, validate (calibrate / early-stop) and test."""

    train: list[int]
    val: list[int] = field(default_factory=list)
    test: list[int] = field(default_factory=list)

    def __post_init__(self):
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            overlap = set(getattr(self, a)) & set(getattr(self, b))
            if overlap:
                raise ValueError(f"{a} and {b} seasons overlap: {sorted(overlap)}")
        if not self.train:
            raise ValueError("no training seasons")

    @property
    def all(self) -> list[int]:
        return sorted(set(self.train) | set(self.val) | set(self.test))


def split_seasons(
    seasons,
    train: list[int] | None = None,
    val: list[int] | None = None,
    test: list[int] | None = None,
) -> SeasonSplit:
    """Default split: latest season is test, the one before is val, rest train.

    Pass an empty list to disable val or test.
    """
    seasons = sorted({int(s) for s in seasons}, reverse=True)
    if test is None:
        test = seasons[:1]
    if val is None:
        val = [s for s in seasons if s not in test][:1]
    if train is None:
        train = [s for s in seasons if s not in set(test) | set(val)]
    return SeasonSplit(
        train=sorted(int(s) for s in train),
        val=sorted(int(s) for s in val),
        test=sorted(int(s) for s in test),
    )
