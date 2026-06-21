"""Per-window centroid margin during sliding-window inference (runtime.inference)."""

from oriented_det.runtime.inference import (
    _centroid_in_sliding_window_interior,
    resolve_sliding_window_margin_pixels,
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
