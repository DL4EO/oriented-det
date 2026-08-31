"""Per-window centroid margin during sliding-window inference (runtime.inference)."""

from oriented_det.runtime.inference import (
    _centroid_in_sliding_window_interior,
    count_sliding_window_positions,
    resolve_sliding_window_margin_pixels,
    uses_native_sliding_window,
)


def test_resolve_margin_from_overlap_pixels():
    mx, my = resolve_sliding_window_margin_pixels(
        overlap_pixels=256, slice_h=1024, slice_w=1024
    )
    assert mx == 128.0 and my == 128.0


def test_explicit_window_margin_overrides_overlap():
    mx, my = resolve_sliding_window_margin_pixels(
        window_margin_pixels=64, overlap_pixels=256, slice_h=1024, slice_w=1024
    )
    assert mx == 64.0 and my == 64.0


def test_interior_window_rejects_edge_centroid():
    # 1024 crop, 128 margin, window not on image border
    assert not _centroid_in_sliding_window_interior(
        50.0, 512.0, 1024, 1024, 128.0, 128.0, 512, 0, 4096, 4096
    )
    assert _centroid_in_sliding_window_interior(
        200.0, 512.0, 1024, 1024, 128.0, 128.0, 512, 0, 4096, 4096
    )


def test_image_border_skips_margin_on_that_side():
    # Left edge of image: centroid in left band is kept
    assert _centroid_in_sliding_window_interior(
        50.0, 512.0, 1024, 1024, 128.0, 128.0, 0, 0, 1024, 2048
    )
    # Interior right edge of window (not image border): right band rejected
    assert not _centroid_in_sliding_window_interior(
        980.0, 512.0, 1024, 1024, 128.0, 128.0, 0, 0, 4096, 2048
    )


def test_zero_margin_keeps_all_in_crop():
    assert _centroid_in_sliding_window_interior(
        0.0, 0.0, 512, 512, 0.0, 0.0, 256, 256, 2048, 2048
    )


def test_pad_mode_never_uses_native_sliding_window():
    pad = {"resize_mode": "pad", "target_size": [800, 800]}
    assert uses_native_sliding_window(824, 1234, pad) is False
    assert count_sliding_window_positions(824, 1234, pad, overlap_pixels=0) == 1
    assert count_sliding_window_positions(400, 300, pad, overlap_pixels=0) == 1
    keep = {"resize_mode": "keep_ratio", "target_size": [800, 800], "pad_size_divisor": 32}
    assert uses_native_sliding_window(824, 1234, keep) is False
    assert count_sliding_window_positions(824, 1234, keep, overlap_pixels=0) == 1


def test_fixed_mode_still_tiles_oversized_dota_rasters():
    fixed = {"resize_mode": "fixed", "target_size": [1024, 1024]}
    assert uses_native_sliding_window(800, 800, fixed) is False
    assert uses_native_sliding_window(2048, 2048, fixed) is True
    n = count_sliding_window_positions(2048, 2048, fixed, overlap_pixels=0)
    assert n > 1
    crop = {"resize_mode": "crop", "target_size": [800, 800]}
    assert uses_native_sliding_window(824, 1234, crop) is True

