import pytest

pytest.importorskip("gradio")

from gradio.components.slider import Slider

import tools.app  # noqa: F401 - importing applies Gradio compatibility patches


def test_slider_preprocess_accepts_none_payload() -> None:
    slider = Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.01)

    assert slider.preprocess(None) == 0.3


def test_outline_thickness_stays_in_screen_pixels() -> None:
    """Zoom scales the image, not the stroke. A 1px outline stays one pixel wide at 4x."""
    import numpy as np

    from tools.app import _draw_outlines, _zoom_rgb

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    square = np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=np.int32)
    zoomed = _zoom_rgb(image, 4.0)
    red_bgr = (0, 0, 255)

    thin = _draw_outlines(zoomed, [(square, red_bgr)], 1, 4.0)
    # Top edge lands on y = 2 * 4. Sample the middle of that edge.
    assert thin[8, 16, 0] == 255
    assert thin[7, 16, 0] == 0
    assert thin[9, 16, 0] == 0

    thick = _draw_outlines(zoomed, [(square, red_bgr)], 3, 4.0)
    assert thick[7, 16, 0] == 255
    assert thick[8, 16, 0] == 255
    assert thick[9, 16, 0] == 255
