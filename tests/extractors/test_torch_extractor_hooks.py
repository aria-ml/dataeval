"""A `TorchExtractor` with a `layer_name` hooks the layer for each call and removes the hook afterwards.

Fixtures: ten random float32 images of shape (1, 2, 2), so a `Flatten` gives four features; tiny CPU models whose
linear layers have random weights, so a linear output has negative values; and three custom modules: one that returns a
tuple, one that is called with a keyword argument only, and one that is never called during the forward pass.
"""

import numpy as np
import pytest
import torch

from dataeval.extractors import TorchExtractor

pytestmark = pytest.mark.required


@pytest.fixture
def images() -> list[np.ndarray]:
    return list(np.random.default_rng(0).random((10, 1, 2, 2), dtype=np.float32) - 0.5)


def _stacked(images: list[np.ndarray]) -> torch.Tensor:
    return torch.as_tensor(np.stack(images))


def _hooks(module: torch.nn.Module) -> int:
    return len(module._forward_hooks) + len(module._forward_pre_hooks)


class _Pair(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, 2 * x


class _ReturnsATuple(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flat = torch.nn.Flatten()
        self.pair = _Pair()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.pair(self.flat(x))
        return a + b


class _Inner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class _CallsByKeyword(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flat = torch.nn.Flatten()
        self.inner = _Inner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x=self.flat(x))


class _SkipsALayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flat = torch.nn.Flatten()
        self.unused = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flat(x)


def _model(inplace: bool = False) -> torch.nn.Sequential:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 4), torch.nn.ReLU(inplace=inplace))


def test_constructing_registers_no_hook():
    model = _model()
    for _ in range(3):
        TorchExtractor(model, layer_name="1", device="cpu")
    assert _hooks(model[1]) == 0


@pytest.mark.parametrize("use_output", [True, False])
def test_a_call_leaves_no_hook(images, use_output):
    model = _model()
    TorchExtractor(model, layer_name="1", use_output=use_output, device="cpu", batch_size=4)(images)
    assert _hooks(model[1]) == 0


def test_output_capture_matches_a_manual_forward(images):
    model = _model()
    result = TorchExtractor(model, layer_name="1", device="cpu")(images)
    with torch.no_grad():
        expected = model[1](model[0](_stacked(images))).numpy()
    assert np.array_equal(result, expected)


def test_input_capture_on_an_in_place_layer_keeps_its_negatives(images):
    model = _model(inplace=True)
    result = np.asarray(TorchExtractor(model, layer_name="2", use_output=False, device="cpu")(images))
    with torch.no_grad():
        expected = model[1](model[0](_stacked(images))).numpy()
    assert (result < 0).any()
    assert np.array_equal(result, expected)


def test_a_tuple_output_gives_its_first_tensor(images):
    model = _ReturnsATuple()
    result = TorchExtractor(model, layer_name="pair", device="cpu")(images)
    assert np.array_equal(result, model.flat(_stacked(images)).numpy())


@pytest.mark.parametrize("use_output", [True, False])
def test_a_kwargs_only_call_is_captured(images, use_output):
    torch.manual_seed(0)
    model = _CallsByKeyword()
    result = TorchExtractor(model, layer_name="inner", use_output=use_output, device="cpu")(images)
    with torch.no_grad():
        flat = model.flat(_stacked(images))
        expected = (model.inner(flat) if use_output else flat).numpy()
    assert np.array_equal(result, expected)


def test_a_layer_that_does_not_run_raises(images):
    with pytest.raises(RuntimeError, match="did not run"):
        TorchExtractor(_SkipsALayer(), layer_name="unused", device="cpu")(images)


class _ReturnsNone(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> None:
        return None


class _CallsALayerThatReturnsNone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flat = torch.nn.Flatten()
        self.nothing = _ReturnsNone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.nothing(x)
        return self.flat(x)


def test_a_layer_that_returns_none_is_named(images):
    with pytest.raises(RuntimeError, match="'nothing' gave no tensor"):
        TorchExtractor(_CallsALayerThatReturnsNone(), layer_name="nothing", device="cpu")(images)


class _TakesNothing(torch.nn.Module):
    def forward(self) -> None:
        return None


class _CallsALayerWithNoArguments(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flat = torch.nn.Flatten()
        self.nothing = _TakesNothing()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.nothing()
        return self.flat(x)


def test_a_layer_that_receives_none_is_named(images):
    with pytest.raises(RuntimeError, match="'nothing' received no tensor"):
        TorchExtractor(_CallsALayerWithNoArguments(), layer_name="nothing", use_output=False, device="cpu")(images)
