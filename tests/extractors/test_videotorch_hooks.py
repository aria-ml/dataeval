"""A `VideoTorchExtractor` with a `layer_name` hooks the layer for each clip and removes the hook afterwards.

Fixtures: random float32 videos of eight 8x8x3 frames in [0, 1), the mock processor and mock video model of
`test_videotorch_extractor.py`, and a variant of that model whose encoder ends in an in-place ReLU after a linear layer
with random weights, so the ReLU's input has negative values.
"""

import numpy as np
import pytest
import torch

from dataeval.extractors import VideoTorchExtractor
from tests.extractors.test_videotorch_extractor import (
    FRAME_SIZE,
    HIDDEN_SIZE,
    NUM_FRAMES,
    MockProcessor,
    MockVideoModel,
    make_video,
)

pytestmark = pytest.mark.required


class _InPlaceVideoModel(MockVideoModel):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(NUM_FRAMES * FRAME_SIZE, HIDDEN_SIZE),
            torch.nn.ReLU(inplace=True),
        )


def _hooks(module: torch.nn.Module) -> int:
    return len(module._forward_hooks) + len(module._forward_pre_hooks)


def test_constructing_registers_no_hook():
    model = MockVideoModel()
    for _ in range(3):
        VideoTorchExtractor(model, processor=MockProcessor(), layer_name="encoder.1", device="cpu")
    assert _hooks(model.encoder[1]) == 0


@pytest.mark.parametrize("use_output", [True, False])
def test_a_call_leaves_no_hook(use_output):
    model = MockVideoModel()
    extractor = VideoTorchExtractor(
        model, processor=MockProcessor(), layer_name="encoder.1", use_output=use_output, device="cpu"
    )
    extractor([make_video(8)])
    assert _hooks(model.encoder[1]) == 0


def test_input_capture_on_an_in_place_layer_keeps_its_negatives():
    extractor = VideoTorchExtractor(
        _InPlaceVideoModel(), processor=MockProcessor(), layer_name="encoder.2", use_output=False, device="cpu"
    )
    assert (np.asarray(extractor([make_video(8)])) < 0).any()


class _ReturnsNone(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> None:
        return None


class _VideoModelWithALayerThatReturnsNone(MockVideoModel):
    def __init__(self) -> None:
        super().__init__()
        self.nothing = _ReturnsNone()

    def forward(self, pixel_values):
        self.nothing(pixel_values)
        return super().forward(pixel_values)


def test_a_layer_that_returns_none_is_named():
    extractor = VideoTorchExtractor(
        _VideoModelWithALayerThatReturnsNone(), processor=MockProcessor(), layer_name="nothing", device="cpu"
    )
    with pytest.raises(RuntimeError, match="'nothing' gave no tensor"):
        extractor([make_video(8)])
