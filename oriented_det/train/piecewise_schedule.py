"""Piecewise-constant schedules indexed by 0-based training epoch."""

from __future__ import annotations

from typing import Optional, Sequence


def resolve_piecewise_schedule(
    epoch: int,
    boundaries: Optional[Sequence[int]],
    values: Optional[Sequence[float]],
    default: float,
) -> float:
    """Return ``values[i]`` for the segment containing ``epoch``.

    Segments are defined by strictly increasing ``boundaries`` (0-based epoch
    indices, same convention as ``freeze_backbone_epochs`` and
    ``final_nms_iou_schedule_epochs``). Example::

        boundaries=[20, 24], values=[0.1, 0.05, 0.0]
        epoch 0..19 -> 0.1, 20..23 -> 0.05, 24+ -> 0.0

    When ``boundaries`` or ``values`` is empty/None, returns ``default``.
    """
    if boundaries is None or values is None or not boundaries or not values:
        return float(default)
    idx = 0
    for boundary in boundaries:
        if epoch < boundary:
            break
        idx += 1
    idx = min(idx, len(values) - 1)
    return float(values[idx])
