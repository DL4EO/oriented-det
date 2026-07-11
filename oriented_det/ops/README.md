# Ops

User guide: [Operations](../../docs/user-guide/operations.md).

GPU-oriented geometry helpers live here, including oriented IoU, anchor matching,
and NMS utilities used by the detector models.

## Rotated Backend Policy

`rotated_ops.py` is the public switch for rotated IoU/NMS:

- `ORIENTED_DET_ROTATED_BACKEND=gpu_sample` (default) uses this repo's parallel
  tensor/GPU sampling implementation.
- `ORIENTED_DET_ROTATED_BACKEND=cpu` is for debugging/reference checks only.

For **final detection NMS only**, callers can pass **`rotated_nms(..., force_cpu=True)`**
(or set **`model.final_nms_use_cpu`** / **`production.final_nms_use_cpu`** in JSON) so that
step always uses **exact polygon IoU on CPU** ( **Shapely** when installed, else
Sutherland–Hodgman with a prominent install warning), regardless of
``ORIENTED_DET_ROTATED_BACKEND``. RPN proposal NMS stays horizontal ``torchvision.ops.nms``.

**mAP / AP matching** defaults to exact CPU polygon IoU
(``compute_oriented_map(..., use_exact_rotated_iou=True)``). In training, set
``evaluation.use_exact_rotated_iou: false`` in JSON (or pass ``use_exact_rotated_iou=False``)
for faster GPU sampling IoU (approximate). GT-cover / accuracy on mAP epochs use the same flag.
Final detection NMS is unaffected.

Sampling-based GPU IoU/NMS use a regular grid of interior points per box (local
coordinates **−0.495 … +0.495** times half-width and half-height per axis, i.e. **99%**
of each dimension—slightly inset from the true corners). The grid size must be a
**perfect square** (e.g. `25`, `36`, `49`, `64`, `81`, `100`).

Environment variables (all optional):

Geometry-based sizing (**on by default** via
``ORIENTED_DET_GPU_ORIENTED_IOU_SAMPLE_BY_MAX_SIDE``). One square grid is chosen
for the whole IoU/NMS call from the **largest** requirement among all boxes in
the batch (anchors, GT, proposals, …):

- Target **~2 px** spacing between grid points along each box axis (configurable).
- **Minimum 3 points** along the shorter side (thin ships / slivers).
- **Extra refinement** on the long axis when aspect ratio > 2 (elongated OBBs).
- Clamped to **[25, 1024]** samples (5×5 … 32×32) by default.

Small compact boxes (e.g. 10×10 px cars/trucks) use at least **25** samples (5×5
floor); a 5×100 px box uses a much finer grid (up to the cap).

- `ORIENTED_DET_GPU_ORIENTED_IOU_TARGET_SPACING_PX` — target sample spacing in
  image pixels (default `2.0`).
- `ORIENTED_DET_GPU_ORIENTED_IOU_MIN_SAMPLES` — lower clamp, perfect square
  (default `25`).
- `ORIENTED_DET_GPU_ORIENTED_IOU_MAX_SAMPLES` — upper clamp, perfect square
  (default `1024`).
- `ORIENTED_DET_GPU_ORIENTED_IOU_SAMPLE_BY_MAX_SIDE` — **on by default** (unset =
  geometry enabled). Set to `0` / `false` / `no` / `off` for a flat **100**-sample
  grid (debug only).
- `ORIENTED_DET_GPU_NMS_IOU_SAMPLES` — optional floor for oriented NMS when
  geometry is enabled; flat count (default `100`) when geometry is off.
- `ORIENTED_DET_GPU_NMS_IOU_SAMPLE_BY_MAX_SIDE` — same as IoU toggle, NMS-only.

Debug: ``resolve_oriented_iou_sample_count(boxes1, boxes2)`` in [gpu_ops.py](gpu_ops.py).
Benchmark vs Shapely: ``python tools/measure_sampled_riou_error.py`` (see [tools/README.md](../../tools/README.md)).

### Geometry-based rIoU sampling — rationale & metrics

Training **anchor/proposal matching** uses GPU sampling IoU (`oriented_box_iou_gpu`).
That path is **approximate**; **mAP and optional final NMS** use **exact Shapely polygon IoU**
on CPU. The geometry defaults below were chosen so matching IoU is close enough to
polygon IoU across DOTA-like scales without paying a fixed 10×10 (100-point) grid on
every tiny anchor.

#### Problem

Sampling IoU places a **√S × √S** grid in each box’s local frame, then estimates
intersection from point counts. Error grows when:

1. **Grid spacing in image pixels is too coarse** along an axis (elongated ships, or
   small cars/trucks at 10–25 px).
2. **Partial overlap at high IoU** (0.7–1): a coarse grid systematically **underestimates**
   intersection.

A single global sample count wastes work on 10×10 px boxes (100 samples ≈ 1 px spacing)
while still under-serving 5×100 px ships unless spacing is aspect-aware.

#### Design

| Mechanism | Rationale |
|-----------|-----------|
| **Geometry-driven S** | Grid side from each box `w`, `h` and aspect ratio; one **max** S for the whole IoU call (batch of anchors + GT). Small squares get fewer samples; elongated boxes get more. |
| **`target_spacing_px = 2`** | Target **~2 px** between grid points along each axis. Chosen from sweeps on a **vehicle** stratum (10–25 px, car/truck scale): spacing 4 px gave ~10% of pairs with \|error\| > 10%; spacing 2 px cut that to **0.2%** at ~3× sample cost vs spacing 4. |
| **`min_samples = 25` (5×5)** | Floor for sub-10 px objects where geometry alone would use 3×3 (9). Does not affect 10–25 px vehicles when spacing is 2 (geometry already requests ~13×13). |
| **`max_samples = 1024` (32×32)** | Cap for extreme elongated or very large boxes; avoids unbounded cost. |
| **Aspect > 2 long-axis boost** | Tightens spacing on the long edge for thin ships / bridges. |
| **≥ 3 points on short side** | Ensures at least minimal coverage across the thin dimension. |

Default constants live in [gpu_ops.py](gpu_ops.py) (``_DEFAULT_TARGET_SPACING_PX``,
``_DEFAULT_MIN_SAMPLES``, ``_DEFAULT_MAX_SAMPLES``, ``_MIN_POINTS_ALONG_SHORT_SIDE``).

#### Benchmark methodology

Script: [tools/measure_sampled_riou_error.py](../../tools/measure_sampled_riou_error.py)

- **Exact:** Shapely polygon IoU (`rbox_iou(..., intersection_backend="shapely")`).
- **Sampled:** `oriented_box_iou_gpu` with the same geometry params.
- **Pairs:** Synthetic stratified boxes (tiny/small/medium/large squares, **vehicle**
  10–25 px, elongated / thin ships), random overlap via center offsets.
- **Seed 0, 1400 pairs** (200 per stratum) unless noted.

Reproduce:

```bash
python tools/measure_sampled_riou_error.py --pairs 1400 --seed 0
python tools/measure_sampled_riou_error.py --pairs 600 --seed 0 --categories vehicle
python tools/measure_sampled_riou_error.py --pairs 600 --seed 0 --categories vehicle --sweep-spacing 2 3 4
```

#### Measured error (seed 0)

**Full mix (1400 pairs)** — current defaults (spacing **2 px**, min **25**, max **1024**):

| Stratum | mean \|err\| | p90 \|err\| | fraction \|err\| > 10% |
|---------|-------------|-----------|-------------------------|
| ALL | 0.006 | 0.017 | 0.1% |
| vehicle (10–25 px) | 0.012 | 0.033 | 0.5% |
| tiny_square (4–16 px) | 0.013 | 0.038 | 0.0% |
| small_square | 0.010 | 0.025 | 0.0% |
| medium / large square | ~0.003 | ~0.010 | 0.0% |
| elongated / thin | ~0.001 | ~0.004 | 0.0% |

**Exact IoU bin (all strata):**

| Exact IoU | mean \|err\| | fraction \|err\| > 10% |
|-----------|-------------|-------------------------|
| 0 | 0 | 0% |
| 0–0.3 | 0.003 | 0% |
| 0.3–0.7 | 0.014 | 0% |
| 0.7–1 | 0.036 | 2.7% |

Mean grid size across pairs: **~727** samples (vs ~670 with spacing 4 px / min 25).

**Vehicle-only (600 pairs)** — spacing sweep at min = 25:

| `target_spacing_px` | mean \|err\| | p90 \|err\| | > 10% | ~grid |
|---------------------|-------------|-----------|-------|-------|
| 4 (old default) | 0.033 | 0.097 | 10.1% | 49 |
| 3 | 0.024 | 0.074 | 3.3% | 81 |
| **2 (current)** | **0.016** | **0.047** | **0.2%** | 169 |

High-IoU vehicle pairs (exact 0.7–1): mean \|err\| **0.059** at spacing 2 vs **0.130** at spacing 4;
fraction \|err\| > 10% **3.2%** vs **87%**.

#### Tradeoffs

- **Cost:** ~2×–3× more samples than spacing 4 for small/medium boxes; large and
  elongated boxes still hit the **1024** cap.
- **Not exact:** Sampling remains an approximation; do not use for published mAP
  (use exact CPU path) or for a principled IoU **loss** (prefer `kfiou` / `probiou`,
  or future exact differentiable IoU).
- **One S per kernel call:** A single large GT in a chunk raises S for all pairs in
  that `oriented_box_iou_gpu` invocation (conservative, simpler GPU code).

The first release intentionally avoids MMCV/MMDet/MMRotate runtime dependencies
and does not add custom CUDA kernels. If profiling shows a large enough win later,
add in-repo CUDA kernels as a new backend behind `rotated_ops.py`.

`gpu_ops.match_anchors_to_gt_gpu` supports an HBB assignment mode for RPN
training. In that mode, oriented anchors and GT boxes are converted to enclosing
axis-aligned boxes. The matcher computes HBB IoU in large anchor chunks and keeps
only the best GT per anchor and best anchor per GT. This keeps MMRotate-style HBB
assignment semantics while avoiding thousands of tiny GPU launches and avoiding a
full `anchors x GT` matrix on large P2 grids.

`gpu_ops.hbb_nms_for_oriented_boxes_gpu` performs fast proposal pruning by
converting oriented proposals to HBB boxes and using `torchvision.ops.nms`.

`gpu_ops.oriented_nms_gpu` (final detection NMS) avoids both the dense
`[M, M]` sampled-IoU matrix and a per-box Python greedy loop:

1. **Pair pruning:** candidate pairs are limited to those whose AABB-based IoU
   upper bound `I_aabb / (area_i + area_j - I_aabb)` exceeds half the NMS
   threshold (the 0.5 factor absorbs sampling-grid noise). Sampled rotated IoU
   is then computed only for that sparse pair list (`O(P*S)` instead of
   `O(M^2 * S)`), with the same estimator as `oriented_box_iou_gpu`.
2. **Fixpoint suppression (Cluster-NMS):** instead of a sequential greedy loop
   with one host sync per box, `keep[j] = not any_i(suppress[i, j] and keep[i])`
   is iterated until stable. The unique fixpoint equals the sequential greedy
   NMS result and convergence takes at most the longest suppression-chain depth
   (typically < 10 iterations).

This was the dominant cost of per-epoch validation for early-training
RetinaNet checkpoints (hundreds of weakly-suppressing candidates per class).

## Auxiliary decoded-box losses (`kfiou.py`, `probiou.py`)

When **`model.roi_box_reg_iou_weight`** > 0 and **`model.roi_box_reg_main_loss_type`** is `smooth_l1` (default), training adds a scalar decoded-box term on ROI positives. When main is `probiou` / `riou` / `kfiou`, that decoded metric is the **primary** loss instead; use **`model.roi_box_reg_smooth_l1_aux_weight`** for encoded Smooth L1 aux.
**`model.roi_box_reg_iou_loss_type`**: **`riou`** (default), **`kfiou`**, or
**`probiou`**. All three are wired through
`mean_auxiliary_box_reg_loss` in [kfiou.py](kfiou.py).

### KFIoU (`kfiou.py`)

[kfiou.py](kfiou.py) implements the Kalman Filter IoU surrogate (Gaussian overlap +
center Smooth L1), aligned with MMRotate’s `kf_iou_loss`. Optional
**`model.roi_box_reg_kfiou_fun`** selects the overlap transform: omit /
**`none`** → `1 - KFIoU`, **`ln`** → `-log(KFIoU)`, **`exp`** → `exp(1 - KFIoU) - 1`.

The fused-covariance step regularizes `sigma_p + sigma_t` with a small diagonal
jitter and uses a batched ``pinv`` for the Kalman gain so near-degenerate boxes
or unstable forward values (LR sweeps) do not trip ``linalg.solve`` on CUDA.

### ProbIoU (`probiou.py`)

[probiou.py](probiou.py) implements Probabilistic IoU (Gaussian bounding boxes),
following the [reference implementation](https://github.com/ProbIOU/probiou-sample/blob/main/probiou_pytorch.py).
Use **`model.roi_box_reg_iou_loss_type: "probiou"`**. Optional
**`model.roi_box_reg_probiou_mode`**: **`l1`** (bounded, default) or **`l2`**
(`-log(1 - l1²)`).
