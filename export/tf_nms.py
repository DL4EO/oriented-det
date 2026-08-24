"""Pure-TensorFlow rotated NMS and detect finalization (SavedModel-safe).

These ops are traced into a SavedModel so reload needs only TensorFlow — not
ONNX Runtime, PyTorch, or oriented-det. IoU is convex-quad clipping (same
geometry as CPU polygon NMS, with small float drift).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

_MAX_VERTS = 8
_EPS = 1e-8
_MIN_BOX_SIZE = 1.0


def _cross(u: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def rboxes_to_corners(boxes: tf.Tensor) -> tf.Tensor:
    """Convert ``[N, 5]`` ``(cx, cy, w, h, angle)`` to ``[N, 4, 2]`` corners.

    Corner order matches ``oriented_det.geometry.RBox.corners`` (CCW in y-up).
    """
    cx, cy, w, h, ang = tf.unstack(tf.cast(boxes, tf.float32), axis=-1)
    cos_a = tf.cos(ang)
    sin_a = tf.sin(ang)
    w2 = w * 0.5
    h2 = h * 0.5
    local_x = tf.stack([-w2, w2, w2, -w2], axis=-1)
    local_y = tf.stack([-h2, -h2, h2, h2], axis=-1)
    rx = local_x * cos_a[:, None] - local_y * sin_a[:, None]
    ry = local_x * sin_a[:, None] + local_y * cos_a[:, None]
    return tf.stack([cx[:, None] + rx, cy[:, None] + ry], axis=-1)


def _signed_area_quad(poly: tf.Tensor) -> tf.Tensor:
    x = poly[..., 0]
    y = poly[..., 1]
    x_next = tf.roll(x, shift=-1, axis=-1)
    y_next = tf.roll(y, shift=-1, axis=-1)
    return 0.5 * tf.reduce_sum(x * y_next - x_next * y, axis=-1)


def _ensure_ccw(poly: tf.Tensor) -> tf.Tensor:
    sa = _signed_area_quad(poly)
    return tf.where(sa[:, None, None] < 0.0, poly[:, ::-1, :], poly)


def _is_inside(p: tf.Tensor, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    return _cross(b - a, p - a) >= -1e-6


def _line_intersect(p: tf.Tensor, q: tf.Tensor, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    s = q - p
    t = b - a
    cross = _cross(s, t)
    ap = a - p
    denom = tf.where(tf.abs(cross) < _EPS, tf.ones_like(cross), cross)
    u = _cross(ap, t) / denom
    return p + u[..., None] * s


def _clip_one_edge(
    verts: tf.Tensor,
    counts: tf.Tensor,
    a: tf.Tensor,
    b: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Sutherland–Hodgman clip of batched polygons against edge ``a→b``."""
    n_batch = tf.shape(verts)[0]
    max_v = verts.shape[1]
    new_verts = tf.zeros_like(verts)
    new_counts = tf.zeros([n_batch], dtype=tf.int32)

    def _append(
        pts: tf.Tensor, cnt: tf.Tensor, point: tf.Tensor, mask: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        write = mask & (cnt < max_v)
        idx = tf.stack(
            [tf.range(n_batch), tf.clip_by_value(cnt, 0, max_v - 1)], axis=1
        )
        current = tf.gather_nd(pts, idx)
        pts = tf.tensor_scatter_nd_update(
            pts, idx, tf.where(write[:, None], point, current)
        )
        cnt = cnt + tf.cast(write, tf.int32)
        return pts, cnt

    def body(i, pts, cnt):
        active = tf.cast(i, tf.int32) < counts
        p = verts[:, i]
        next_i = tf.math.floormod(i + 1, tf.maximum(counts, 1))
        q = tf.gather(verts, next_i, batch_dims=1)
        p_in = _is_inside(p, a, b)
        q_in = _is_inside(q, a, b)
        inter = _line_intersect(p, q, a, b)

        pts, cnt = _append(pts, cnt, inter, active & (~p_in) & q_in)
        pts, cnt = _append(pts, cnt, q, active & q_in)
        pts, cnt = _append(pts, cnt, inter, active & p_in & (~q_in))
        return i + 1, pts, cnt

    _, new_verts, new_counts = tf.while_loop(
        lambda i, *_: i < max_v,
        body,
        loop_vars=(tf.constant(0, tf.int32), new_verts, new_counts),
        maximum_iterations=max_v,
    )
    return new_verts, new_counts


def _clip_polygon(subject: tf.Tensor, n_subj: tf.Tensor, clipper: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    clipper = _ensure_ccw(clipper)
    verts = subject
    counts = n_subj
    for e in range(4):
        a = clipper[:, e]
        b = clipper[:, (e + 1) % 4]
        verts, counts = _clip_one_edge(verts, counts, a, b)
    return verts, counts


def _polygon_area(verts: tf.Tensor, counts: tf.Tensor) -> tf.Tensor:
    max_v = verts.shape[1]
    x = verts[..., 0]
    y = verts[..., 1]
    idx = tf.range(max_v)
    nxt = tf.math.floormod(idx[None, :] + 1, tf.maximum(counts[:, None], 1))
    x_next = tf.gather(x, nxt, batch_dims=1)
    y_next = tf.gather(y, nxt, batch_dims=1)
    cross = x * y_next - x_next * y
    valid = idx[None, :] < counts[:, None]
    return tf.abs(0.5 * tf.reduce_sum(tf.where(valid, cross, tf.zeros_like(cross)), axis=1))


def _pad_quads(corners: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    n = tf.shape(corners)[0]
    pad = tf.zeros([n, _MAX_VERTS - 4, 2], dtype=corners.dtype)
    verts = tf.concat([corners, pad], axis=1)
    counts = tf.fill([n], 4)
    return verts, counts


def rbox_iou_one_to_many(box: tf.Tensor, boxes: tf.Tensor) -> tf.Tensor:
    """IoU of one rbox ``[5]`` against ``[N, 5]``."""
    n = tf.shape(boxes)[0]
    poly_a = tf.tile(rboxes_to_corners(box[None, :]), [n, 1, 1])
    poly_b = rboxes_to_corners(boxes)
    verts_a, n_a = _pad_quads(poly_a)
    clipped, n_c = _clip_polygon(verts_a, n_a, poly_b)
    inter = _polygon_area(clipped, n_c)
    area_a = tf.maximum(box[2] * box[3], 0.0)
    area_b = tf.maximum(boxes[:, 2] * boxes[:, 3], 0.0)
    union = area_a + area_b - inter
    return tf.where(union > _EPS, inter / union, tf.zeros_like(inter))


def pairwise_rbox_iou(boxes1: tf.Tensor, boxes2: tf.Tensor) -> tf.Tensor:
    """Return ``[N, M]`` IoU matrix (used by tests)."""
    n = tf.shape(boxes1)[0]

    def body(i, acc):
        row = rbox_iou_one_to_many(boxes1[i], boxes2)
        acc = acc.write(i, row)
        return i + 1, acc

    m = tf.shape(boxes2)[0]
    ta = tf.TensorArray(tf.float32, size=n)
    _, ta = tf.while_loop(
        lambda i, _: i < n,
        body,
        (tf.constant(0, tf.int32), ta),
        maximum_iterations=n,
    )
    out = ta.stack()
    out.set_shape([None, None])
    tf.debugging.assert_equal(tf.shape(out)[1], m)
    return out


def rotated_nms_tf(
    boxes: tf.Tensor,
    scores: tf.Tensor,
    labels: tf.Tensor,
    *,
    iou_threshold: float,
    max_detections: int,
    class_agnostic: bool,
) -> tf.Tensor:
    """Greedy rotated NMS. Returns keep indices in score order."""
    n = tf.shape(boxes)[0]
    max_det = tf.constant(int(max_detections), tf.int32)
    iou_thr = tf.constant(float(iou_threshold), tf.float32)

    def empty() -> tf.Tensor:
        return tf.zeros([0], dtype=tf.int32)

    def nonempty() -> tf.Tensor:
        order = tf.argsort(scores, direction="DESCENDING")
        keep_mask = tf.ones([n], dtype=tf.bool)
        keep_ta = tf.TensorArray(tf.int32, size=max_det, dynamic_size=False)

        def cond(i, keep_mask, n_kept, keep_ta):
            return (i < n) & (n_kept < max_det)

        def body(i, keep_mask, n_kept, keep_ta):
            idx = order[i]
            take = keep_mask[idx]

            def do_take():
                ious = rbox_iou_one_to_many(boxes[idx], boxes)
                same_class = (
                    tf.ones([n], dtype=tf.bool)
                    if class_agnostic
                    else tf.equal(labels, labels[idx])
                )
                rng = tf.range(n)
                suppress = (ious > iou_thr) & same_class & tf.not_equal(rng, idx)
                new_mask = keep_mask & ~suppress
                new_ta = keep_ta.write(n_kept, idx)
                return new_mask, n_kept + 1, new_ta

            def skip():
                return keep_mask, n_kept, keep_ta

            keep_mask, n_kept, keep_ta = tf.cond(take, do_take, skip)
            return i + 1, keep_mask, n_kept, keep_ta

        _, _, n_kept, keep_ta = tf.while_loop(
            cond,
            body,
            loop_vars=(tf.constant(0, tf.int32), keep_mask, tf.constant(0, tf.int32), keep_ta),
            maximum_iterations=n,
        )
        stacked = keep_ta.stack()
        return stacked[:n_kept]

    return tf.cond(n > 0, nonempty, empty)


def _unbatch_pre_nms(
    boxes: tf.Tensor,
    scores: tf.Tensor,
    labels: tf.Tensor,
    count: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Drop a leading batch-1 axis when static rank shows onnx2tf-style batching."""
    if boxes.shape.rank == 3:
        boxes = boxes[0]
    if scores.shape.rank == 2:
        scores = scores[0]
    if labels.shape.rank == 2:
        labels = labels[0]
    if count.shape.rank == 1:
        count = count[0]
    return boxes, scores, labels, count


def _score_threshold_vector(
    labels: tf.Tensor,
    *,
    score_threshold: float,
    per_class_score_threshold: Optional[Dict[str, float]],
    class_id_to_name: Dict[int, str],
) -> tf.Tensor:
    n = tf.shape(labels)[0]
    default = tf.constant(float(score_threshold), tf.float32)
    if not class_id_to_name:
        return tf.fill([n], default)
    max_id = max(int(k) for k in class_id_to_name)
    table = [float(score_threshold)] * (max_id + 1)
    if per_class_score_threshold:
        for lid, name in class_id_to_name.items():
            table[int(lid)] = float(per_class_score_threshold.get(name, score_threshold))
    lut = tf.constant(table, dtype=tf.float32)
    clipped = tf.clip_by_value(labels, 0, max_id)
    gathered = tf.gather(lut, clipped)
    return tf.where(labels > max_id, tf.fill([n], default), gathered)


def tf_finalize_detections(
    boxes: tf.Tensor,
    scores: tf.Tensor,
    labels: tf.Tensor,
    count: tf.Tensor,
    *,
    nms_class_agnostic: bool,
    final_nms_iou_threshold: float,
    max_detections_per_image: int,
    score_threshold: float,
    per_class_score_threshold: Optional[Dict[str, float]],
    class_id_to_name: Dict[int, str],
    max_output_slots: int,
    **_: Any,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Score-filter + rotated NMS → padded ``[max_output_slots, 7]`` and count."""
    boxes = tf.cast(boxes, tf.float32)
    scores = tf.cast(scores, tf.float32)
    labels = tf.cast(labels, tf.int32)
    boxes, scores, labels, count = _unbatch_pre_nms(boxes, scores, labels, count)
    count = tf.cast(tf.reshape(count, []), tf.int32)

    p = tf.shape(boxes)[0]
    n = tf.minimum(tf.maximum(count, 0), p)
    idx = tf.range(p)
    valid = idx < n
    valid = valid & (boxes[:, 2] >= _MIN_BOX_SIZE) & (boxes[:, 3] >= _MIN_BOX_SIZE)
    thr = _score_threshold_vector(
        labels,
        score_threshold=score_threshold,
        per_class_score_threshold=per_class_score_threshold,
        class_id_to_name=class_id_to_name,
    )
    valid = valid & (scores >= thr)
    valid_idx = tf.reshape(tf.where(valid), [-1])

    def _run_nms() -> Tuple[tf.Tensor, tf.Tensor]:
        b = tf.gather(boxes, valid_idx)
        s = tf.gather(scores, valid_idx)
        lab = tf.gather(labels, valid_idx)
        keep = rotated_nms_tf(
            b,
            s,
            lab,
            iou_threshold=final_nms_iou_threshold,
            max_detections=int(max_detections_per_image),
            class_agnostic=bool(nms_class_agnostic),
        )
        kb = tf.gather(b, keep)
        ks = tf.gather(s, keep)
        kl = tf.cast(tf.gather(lab, keep), tf.float32)
        size_ok = (kb[:, 2] >= _MIN_BOX_SIZE) & (kb[:, 3] >= _MIN_BOX_SIZE)
        kb = tf.boolean_mask(kb, size_ok)
        ks = tf.boolean_mask(ks, size_ok)
        kl = tf.boolean_mask(kl, size_ok)
        det = tf.concat([kb, ks[:, None], kl[:, None]], axis=1)
        m = tf.minimum(tf.shape(det)[0], int(max_output_slots))
        det = det[:m]
        pad_n = int(max_output_slots) - m
        det = tf.pad(det, [[0, pad_n], [0, 0]])
        det.set_shape([int(max_output_slots), 7])
        return det, tf.cast(m, tf.int32)

    def _empty() -> Tuple[tf.Tensor, tf.Tensor]:
        z = tf.zeros([int(max_output_slots), 7], dtype=tf.float32)
        return z, tf.constant(0, tf.int32)

    return tf.cond(tf.size(valid_idx) > 0, _run_nms, _empty)


class DetectPostprocessLayer(tf.keras.layers.Layer):
    """Keras wrapper so finalize kwargs are baked into a SavedModel."""

    def __init__(self, finalize_kwargs: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.finalize_kwargs = dict(finalize_kwargs)

    def call(self, inputs):
        boxes, scores, labels, count = inputs
        detections, num = tf_finalize_detections(
            boxes,
            scores,
            labels,
            count,
            nms_class_agnostic=bool(self.finalize_kwargs.get("nms_class_agnostic", False)),
            final_nms_iou_threshold=float(
                self.finalize_kwargs.get("final_nms_iou_threshold", 0.1)
            ),
            max_detections_per_image=int(
                self.finalize_kwargs.get("max_detections_per_image")
                or self.finalize_kwargs.get("max_output_slots")
                or 3000
            ),
            score_threshold=float(self.finalize_kwargs.get("score_threshold", 0.05)),
            per_class_score_threshold=self.finalize_kwargs.get("per_class_score_threshold"),
            class_id_to_name={
                int(k): str(v)
                for k, v in (self.finalize_kwargs.get("class_id_to_name") or {}).items()
            },
            max_output_slots=int(self.finalize_kwargs.get("max_output_slots") or 3000),
        )
        return detections, num

    def compute_output_shape(self, input_shape):
        slots = int(self.finalize_kwargs.get("max_output_slots") or 3000)
        return (tf.TensorShape([slots, 7]), tf.TensorShape([]))

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg["finalize_kwargs"] = self.finalize_kwargs
        return cfg
