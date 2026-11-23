"""Score prefilter + NMS (numpy; no torch)."""

from __future__ import annotations

import numpy as np
import pytest

from export.postprocess import (
    any_pair_iou_at_least,
    finalize_detections_numpy,
    max_pairwise_rbox_iou,
    meta_to_finalize_kwargs,
)


def _kwargs(**overrides):
    base = {
        "nms_class_agnostic": True,
        "final_nms_iou_threshold": 0.5,
        "max_detections_per_image": 100,
        "final_nms_use_cpu": True,
        "score_threshold": 0.2,
        "per_class_score_threshold": None,
        "class_id_to_name": {1: "a", 2: "b"},
        "max_output_slots": 100,
    }
    base.update(overrides)
    return base


def test_finalize_prefilter_matches_postfilter_only_path():
    """Pre-NMS score floor must not change greedy-NMS results vs post-only filter."""
    rng = np.random.RandomState(0)
    n = 200
    boxes = rng.rand(n, 5).astype(np.float32)
    boxes[:, 2:4] = boxes[:, 2:4] * 40 + 5
    scores = rng.rand(n).astype(np.float32)
    scores[:80] = scores[:80] * 0.15  # below 0.2
    scores[80:] = scores[80:] * 0.8 + 0.2
    labels = rng.randint(1, 3, size=(n,)).astype(np.int64)

    out_new, num_new = finalize_detections_numpy(boxes, scores, labels, n, **_kwargs())

    out_all, num_all = finalize_detections_numpy(
        boxes, scores, labels, n, **_kwargs(score_threshold=0.0)
    )
    keep = out_all[:num_all, 5] >= 0.2
    ref = out_all[:num_all][keep]
    num_ref = int(keep.sum())

    assert num_new == num_ref
    if num_new:
        assert np.allclose(out_new[:num_new], ref, atol=1e-5)


def test_max_pairwise_rbox_iou_identical_and_separated() -> None:
    same = np.array([[10.0, 10.0, 8.0, 4.0, 0.0], [10.0, 10.0, 8.0, 4.0, 0.0]], dtype=np.float32)
    assert any_pair_iou_at_least(same, 0.5)
    assert max_pairwise_rbox_iou(same) > 0.99
    sep = np.array([[10.0, 10.0, 8.0, 4.0, 0.0], [80.0, 80.0, 8.0, 4.0, 0.0]], dtype=np.float32)
    assert not any_pair_iou_at_least(sep, 0.5)
    assert max_pairwise_rbox_iou(sep) < 0.05
    assert max_pairwise_rbox_iou(same[:1]) == 0.0


def test_output_classes_filters_after_nms_not_before() -> None:
    """Excluded classes still win NMS; then the allowlist drops them.

    Overlapping aircraft (0.54) + drone (0.47): a pre-NMS aircraft score floor
    of 1.1 would leave the drone. Post-NMS ``output_classes=["drone"]`` drops
    the aircraft winner and yields no box.
    """
    boxes = np.array(
        [
            [10.0, 10.0, 20.0, 8.0, 0.0],
            [10.0, 10.0, 20.0, 8.0, 0.0],
            [80.0, 80.0, 20.0, 8.0, 0.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.54, 0.47, 0.90], dtype=np.float32)
    labels = np.array([1, 2, 2], dtype=np.int64)  # aircraft, drone, drone
    id2n = {1: "aircraft", 2: "drone", 3: "helicopter"}

    out_all, n_all = finalize_detections_numpy(
        boxes,
        scores,
        labels,
        3,
        **_kwargs(
            score_threshold=0.35,
            final_nms_iou_threshold=0.1,
            nms_class_agnostic=True,
            class_id_to_name=id2n,
            per_class_score_threshold=None,
        ),
    )
    assert n_all == 2
    kept_all = {(int(out_all[i, 6]), round(float(out_all[i, 5]), 2)) for i in range(n_all)}
    assert kept_all == {(2, 0.90), (1, 0.54)}  # far drone + aircraft winner

    out_allow, n_allow = finalize_detections_numpy(
        boxes,
        scores,
        labels,
        3,
        **_kwargs(
            score_threshold=0.35,
            final_nms_iou_threshold=0.1,
            nms_class_agnostic=True,
            class_id_to_name=id2n,
            per_class_score_threshold=None,
            output_classes=["Drone"],
        ),
    )
    assert n_allow == 1
    assert int(out_allow[0, 6]) == 2
    assert float(out_allow[0, 5]) == pytest.approx(0.90)
    # Overlap drone 0.47 stayed suppressed; plane location is empty.

    # Score-floor hack before NMS leaves the weaker drone on the plane.
    out_hack, n_hack = finalize_detections_numpy(
        boxes,
        scores,
        labels,
        3,
        **_kwargs(
            score_threshold=0.35,
            final_nms_iou_threshold=0.1,
            nms_class_agnostic=True,
            class_id_to_name=id2n,
            per_class_score_threshold={"aircraft": 1.1},
        ),
    )
    assert n_hack == 2
    kept_hack = {(int(out_hack[i, 6]), round(float(out_hack[i, 5]), 2)) for i in range(n_hack)}
    assert kept_hack == {(2, 0.90), (2, 0.47)}


def test_meta_to_finalize_kwargs_reads_output_classes() -> None:
    fk = meta_to_finalize_kwargs(
        {
            "class_names": ["aircraft", "drone"],
            "production": {
                "score_threshold": 0.3,
                "output_classes": ["drone", ""],
                "final_nms_iou_threshold": 0.1,
                "max_detections_per_image": 10,
            },
        }
    )
    assert fk["output_classes"] == ["drone"]
