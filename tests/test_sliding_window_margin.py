"""Per-window centroid margin during sliding-window inference (runtime.inference)."""

from oriented_det.runtime.inference import (
    _centroid_in_sliding_window_interior,
    _sliding_window_grid,
    count_sliding_window_positions,
    resolve_sliding_window_margin_pixels,
    uses_native_sliding_window,
)


def test_resolve_margin_defaults_to_zero():
    mx, my = resolve_sliding_window_margin_pixels(
        overlap_pixels=256, slice_h=1024, slice_w=1024
    )
    assert mx == 0.0 and my == 0.0
    mx, my = resolve_sliding_window_margin_pixels()
    assert mx == 0.0 and my == 0.0


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


def test_sliding_window_grid_flushes_last_tile_to_image_edge():
    # 2000×2000, 1024 / overlap 200 → stride 824. Last origin is 2000-1024=976, not 1648.
    grid = _sliding_window_grid(2000, 2000, 1024, 1024, overlap_pixels=200)
    xs = sorted({x for x, _ in grid})
    ys = sorted({y for _, y in grid})
    assert xs == [0, 824, 976]
    assert ys == [0, 824, 976]
    assert (976, 976) in grid
    assert (1648, 1648) not in grid


def test_sliding_window_grid_exact_stride_has_no_padded_leftover():
    # 1848 = 1024 + 824: flush last origin onto 824 and drop the 200px leftover window.
    grid = _sliding_window_grid(1848, 1848, 1024, 1024, overlap_pixels=200)
    xs = sorted({x for x, _ in grid})
    assert xs == [0, 824]
    assert len(grid) == 4


def test_sliding_window_grid_fits_canvas_is_single_origin():
    assert _sliding_window_grid(800, 800, 1024, 1024, overlap_pixels=200) == [(0, 0)]
    assert _sliding_window_grid(1024, 1024, 1024, 1024, overlap_pixels=200) == [(0, 0)]


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


def test_resolve_window_batch_size_default_skips_probe(monkeypatch):
    import oriented_det.runtime.inference as inf

    monkeypatch.delenv("ORIENTED_DET_WINDOW_BATCH_SIZE", raising=False)
    called = []

    def _boom(*_a, **_k):
        called.append(1)
        return 99

    monkeypatch.setattr(inf, "_probe_max_window_batch_size", _boom)
    inf._WINDOW_BATCH_SIZE_CACHE.clear()
    dummy = object()
    assert inf.resolve_window_batch_size(dummy, "cuda", {}) == 8
    assert inf.resolve_window_batch_size(dummy, "cpu", {}) == 4
    assert called == []


def test_resolve_window_batch_size_env_int_and_auto(monkeypatch):
    import oriented_det.runtime.inference as inf

    monkeypatch.setenv("ORIENTED_DET_WINDOW_BATCH_SIZE", "16")
    inf._WINDOW_BATCH_SIZE_CACHE.clear()
    dummy = object()
    assert inf.resolve_window_batch_size(dummy, "cuda", {}) == 16
    assert inf.resolve_window_batch_size(dummy, "cuda", {}, override=3) == 3

    monkeypatch.setenv("ORIENTED_DET_WINDOW_BATCH_SIZE", "auto")
    monkeypatch.setattr(inf, "get_model_size", lambda _p=None: (1024, 1024))
    monkeypatch.setattr(inf, "_probe_max_window_batch_size", lambda *_a, **_k: 32)
    inf._WINDOW_BATCH_SIZE_CACHE.clear()
    assert inf.resolve_window_batch_size(dummy, "cuda", {"target_size": [1024, 1024]}) == 32

