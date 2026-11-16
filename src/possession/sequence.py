"""Prepare the same categorical frame evidence for either decoder."""

from collections.abc import Mapping, Sequence

from possession import baseline
from possession.labels import DEAD


def prepare(
    nearest: list[str | None], dead: Sequence[bool] | None, *, anchors: Mapping[int, str],
    hold_frames: int, place_gaps: bool = True,
) -> tuple[list[str], int]:
    """Apply team confirmation, gap placement, and the dead-ball mask."""
    labels = baseline.sticky_labels(nearest, hold_frames=hold_frames, restarts=anchors)
    moved = 0
    if place_gaps:
        labels, moved = baseline.place_switches_in_gaps(labels, nearest, restart_frames=anchors)
    if dead is not None:
        labels = [DEAD if stopped else label for label, stopped in zip(labels, dead, strict=True)]
    return labels, moved
