"""FAIR1M canonical 37 fine-grained classes and 5 coarse groups.

Class order matches the ai4rs / MMRotate FAIR1M community recipes (including
``other-ship`` / ``other-vehicle`` / ``other-airplane``). Keep this list stable:
DOTA tile discovery sorts unique names alphabetically, so recipes and exports
must pin ``FAIR1M_CLASSES`` for a reproducible 37-way ``class_map``.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List

# ai4rs FAIRDataset order (ship → vehicle → airplane → court → road).
FAIR1M_CLASSES: list[str] = [
    # ship (9)
    "Passenger Ship",
    "Liquid Cargo Ship",
    "Dry Cargo Ship",
    "Motorboat",
    "Fishing Boat",
    "Warship",
    "Engineering Ship",
    "other-ship",
    "Tugboat",
    # vehicle (10)
    "Small Car",
    "Cargo Truck",
    "Van",
    "Trailer",
    "other-vehicle",
    "Dump Truck",
    "Bus",
    "Tractor",
    "Excavator",
    "Truck Tractor",
    # airplane (11)
    "Boeing737",
    "Boeing747",
    "Boeing777",
    "Boeing787",
    "other-airplane",
    "C919",
    "A220",
    "A321",
    "A330",
    "A350",
    "ARJ21",
    # court (4)
    "Tennis Court",
    "Football Field",
    "Basketball Court",
    "Baseball Field",
    # road (3)
    "Intersection",
    "Bridge",
    "Roundabout",
]

FAIR1M_CLASS_SET: FrozenSet[str] = frozenset(FAIR1M_CLASSES)

# Coarse groups for optional ``loss.roi_grouped_ce_groups`` / ``dataset.map_labels``.
FAIR1M_GROUPS: Dict[str, List[str]] = {
    "ship": [
        "Passenger Ship",
        "Liquid Cargo Ship",
        "Dry Cargo Ship",
        "Motorboat",
        "Fishing Boat",
        "Warship",
        "Engineering Ship",
        "other-ship",
        "Tugboat",
    ],
    "vehicle": [
        "Small Car",
        "Cargo Truck",
        "Van",
        "Trailer",
        "other-vehicle",
        "Dump Truck",
        "Bus",
        "Tractor",
        "Excavator",
        "Truck Tractor",
    ],
    "airplane": [
        "Boeing737",
        "Boeing747",
        "Boeing777",
        "Boeing787",
        "other-airplane",
        "C919",
        "A220",
        "A321",
        "A330",
        "A350",
        "ARJ21",
    ],
    "court": [
        "Tennis Court",
        "Football Field",
        "Basketball Court",
        "Baseball Field",
    ],
    "road": [
        "Intersection",
        "Bridge",
        "Roundabout",
    ],
}

# Exact map_labels dict: fine name → coarse group name (optional ablation).
FAIR1M_FINE_TO_COARSE: Dict[str, str] = {
    name: group for group, names in FAIR1M_GROUPS.items() for name in names
}

assert len(FAIR1M_CLASSES) == 37
assert len(FAIR1M_FINE_TO_COARSE) == 37
assert set(FAIR1M_FINE_TO_COARSE) == FAIR1M_CLASS_SET

__all__ = [
    "FAIR1M_CLASSES",
    "FAIR1M_CLASS_SET",
    "FAIR1M_FINE_TO_COARSE",
    "FAIR1M_GROUPS",
]
