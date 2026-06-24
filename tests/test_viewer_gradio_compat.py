from gradio.components.slider import Slider

import tools.app  # noqa: F401 - importing applies Gradio compatibility patches


def test_slider_preprocess_accepts_none_payload() -> None:
    slider = Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.01)

    assert slider.preprocess(None) == 0.3
