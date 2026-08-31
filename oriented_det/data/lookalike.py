"""Reserved lookalike (confuser) routing labels for hard-negative training.

``lookalike`` is never a semantic classifier class. Boxes with this name (or
configured aliases) are kept on the sample for assignment as background and are
excluded from ``class_map`` / ``num_classes``.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Set

# Exact match, case-sensitive. Funnel tags via dataset.map_labels, e.g. Confuser -> lookalike.
LOOKALIKE_CLASS_NAME = "lookalike"


def resolve_lookalike_label_set(
    extra_labels: Optional[Sequence[str]] = None,
) -> Set[str]:
    """Return the set of class names treated as lookalike hard negatives.

    ``LOOKALIKE_CLASS_NAME`` is always included. ``extra_labels`` may add aliases
    (useful when labels are already stored under another name without remapping).
    """
    names: Set[str] = {LOOKALIKE_CLASS_NAME}
    if extra_labels:
        for name in extra_labels:
            text = str(name).strip()
            if text:
                names.add(text)
    return names


def is_lookalike_class_name(
    class_name: str,
    lookalike_labels: Optional[Iterable[str]] = None,
) -> bool:
    """True if ``class_name`` is the reserved token or a configured alias."""
    if isinstance(lookalike_labels, set):
        look_set = lookalike_labels
    else:
        look_set = resolve_lookalike_label_set(
            list(lookalike_labels) if lookalike_labels is not None else None
        )
    return class_name in look_set


def filter_semantic_class_names(
    class_names: Iterable[str],
    lookalike_labels: Optional[Iterable[str]] = None,
) -> list[str]:
    """Drop lookalike routing names from a class-name list (sorted unique)."""
    look_set = (
        lookalike_labels
        if isinstance(lookalike_labels, set)
        else resolve_lookalike_label_set(list(lookalike_labels) if lookalike_labels else None)
    )
    return sorted({name for name in class_names if name not in look_set})


__all__ = [
    "LOOKALIKE_CLASS_NAME",
    "resolve_lookalike_label_set",
    "is_lookalike_class_name",
    "filter_semantic_class_names",
]
