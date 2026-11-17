"""`TorchLayers` reads several layers of one torch model in one forward pass, each as a solo extractor reads it.

Fixtures: ten random float32 images of shape (1, 2, 2) and a tiny CPU model with fixed seeds, `Flatten`, `Linear(4, 4)`,
`ReLU`, `Linear(4, 3)`, wrapped in a module that counts its forward passes, so each layer gives a different, fixed array
and the number of forward passes can be read off.
"""

import numpy as np
import pytest
import torch

from dataeval import Embeddings
from dataeval.extractors import TorchExtractor, TorchLayers
from dataeval.protocols import FeatureExtractor

pytestmark = [pytest.mark.required, pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")]

LAYERS = ["net.1", "net.3"]


class _Counted(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 3)
        )
        self.forwards = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forwards += 1
        return self.net(x)


@pytest.fixture
def images() -> list[np.ndarray]:
    return list(np.random.default_rng(0).random((10, 1, 2, 2), dtype=np.float32) - 0.5)


def _hooks(model: torch.nn.Module) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def test_each_layer_equals_a_solo_extractor(images):
    model = _Counted()
    read = TorchLayers(model, LAYERS, device="cpu", batch_size=4).extract(images)
    assert list(read) == LAYERS
    for layer in LAYERS:
        solo = TorchExtractor(model, layer_name=layer, device="cpu", batch_size=4)(images)
        assert np.array_equal(read[layer], np.asarray(solo)), layer


def test_one_forward_per_batch(images):
    model = _Counted()
    TorchLayers(model, LAYERS, device="cpu", batch_size=4).extract(images)
    assert model.forwards == 3


def test_no_hook_is_left(images):
    model = _Counted()
    TorchLayers(model, LAYERS, device="cpu", batch_size=4).extract(images)
    assert _hooks(model) == 0


def test_no_images_give_empty_rows_as_a_solo_extractor_does():
    model = _Counted()
    read = TorchLayers(model, LAYERS, device="cpu").extract([])
    solo = np.asarray(TorchExtractor(model, layer_name=LAYERS[0], device="cpu")([]))
    for layer in LAYERS:
        rows = np.asarray(read[layer])
        assert rows.shape == solo.shape
        assert rows.dtype == solo.dtype


def test_solo_returns_a_plain_extractor_of_one_layer():
    layers = TorchLayers(_Counted(), LAYERS, device="cpu", batch_size=4)
    solo = layers.solo("net.1")
    assert type(solo) is TorchExtractor
    assert solo.layer_name == "net.1"
    assert solo.batch_size == 4
    with pytest.raises(ValueError, match="not one of"):
        layers.solo("net.2")


def test_it_is_not_a_feature_extractor(images):
    layers = TorchLayers(_Counted(), LAYERS, device="cpu")
    assert not callable(layers)
    assert not isinstance(layers, FeatureExtractor)

    class _Images:
        def __len__(self) -> int:
            return len(images)

        def __getitem__(self, i: int) -> np.ndarray:
            return images[i]

    with pytest.raises(TypeError, match="compute_multi"):
        Embeddings(_Images(), extractor=layers)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("names", "match"), [([], "at least one"), (["net.1", "net.1"], "more than once"), (["nope"], "Invalid layer")]
)
def test_unusable_layer_names_raise(names, match):
    with pytest.raises(ValueError, match=match):
        TorchLayers(_Counted(), names, device="cpu")


def test_it_is_exported_and_experimental():
    import dataeval.extractors

    assert "TorchLayers" in dataeval.extractors.__all__
    assert "TorchLayers" in getattr(TorchLayers, "__experimental__", "")


def test_solo_keeps_transforms_given_as_a_generator(images):
    model = _Counted()
    layers = TorchLayers(model, LAYERS, transforms=(t for t in [lambda x: x * 2]), device="cpu")
    for layer in LAYERS:
        assert np.array_equal(np.asarray(layers.solo(layer)(images)), np.asarray(layers.extract(images)[layer])), layer


class _NestedTuple(torch.nn.Module):
    """A module whose `pair` layer returns ((a, b), c), as some recurrent and attention layers do."""

    def __init__(self) -> None:
        super().__init__()
        self.flat = torch.nn.Flatten()
        self.pair = _Nested()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (a, _), _ = self.pair(self.flat(x))
        return a


class _Nested(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return (x, 2 * x), 3 * x


def test_a_nested_tuple_output_gives_its_first_tensor(images):
    model = _NestedTuple()
    expected = model.flat(torch.as_tensor(np.stack(images))).numpy()
    solo = np.asarray(TorchExtractor(model, layer_name="pair", device="cpu")(images))
    read = np.asarray(TorchLayers(model, ["pair"], device="cpu").extract(images)["pair"])
    assert np.array_equal(solo, expected)
    assert np.array_equal(read, expected)


def test_a_single_string_of_layer_names_is_refused():
    with pytest.raises(TypeError, match="sequence of layer names"):
        TorchLayers(_Counted(), "net.1", device="cpu")
