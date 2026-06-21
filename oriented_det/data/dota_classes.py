"""DOTA v1.0 canonical class list (aligned with MMRotate / common DOTA LE90 configs)."""

from __future__ import annotations

from typing import FrozenSet

# Official 15 foreground categories; order matches MMRotate DOTADataset / typical 1x DOTA configs.
DOTA_V1_CLASSES: list[str] = [
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
]

DOTA_V1_CLASS_SET: FrozenSet[str] = frozenset(DOTA_V1_CLASSES)
