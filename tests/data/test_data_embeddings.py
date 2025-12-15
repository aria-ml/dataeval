import logging
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from dataeval.data import Embeddings
from dataeval.protocols import Array, DatasetMetadata, DatumMetadata


@dataclass
class ObjectDetectionTarget:
    boxes: Any
    labels: Any
    scores: Any


class MockDataset:
    metadata = DatasetMetadata({"id": "mock_dataset"})

    def __init__(self, data, targets, metadata=None):
        self.data = data
        self.targets = targets
        self.datum_metadata = metadata

    def __getitem__(self, idx) -> tuple[Array, Any, DatumMetadata]:
        return self.data[idx], self.targets[idx], self.datum_metadata[idx] if self.datum_metadata else {"id": idx}

    def __len__(self) -> int:
        return len(self.data)


class TorchDataset(torch.utils.data.Dataset):
    metadata = DatasetMetadata({"id": "torch_dataset", "index2label": {k: str(k) for k in range(10)}})

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], {"meta": idx}

    def __len__(self):
        return len(self.data)


class IdentityModel(torch.nn.Module):
    def forward(self, x):
        return x

    def encode(self, x):
        return x


def get_dataset(size: int = 10):
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = size
    mock_dataset.__getitem__.side_effect = lambda _: (np.zeros((3, 16, 16)), [], {})
    return mock_dataset


@pytest.fixture(scope="module")
def torch_ic_ds():
    return TorchDataset(torch.ones((10, 1, 3, 3)), torch.nn.functional.one_hot(torch.arange(10)))


@pytest.fixture(scope="module")
def sequential_model():
    """Fixture providing a multi-layer sequential model for layer extraction tests"""
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, 3, padding=1),  # layer "0"
        torch.nn.ReLU(),  # layer "1"
        torch.nn.Flatten(),  # layer "2"
        torch.nn.Linear(36, 10),  # layer "3"
    )


@pytest.fixture(scope="module")
def model_with_functional_ops():
    """Model where functional operations exist between named layers (output != input case)"""

    class ModelWithFunctionalOps(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 4, 3, padding=1)
            self.relu = torch.nn.ReLU()
            self.linear = torch.nn.Linear(36, 10)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = x.flatten(1)  # Functional op between layers
            x = self.linear(x)
            return x

    return ModelWithFunctionalOps()


@pytest.mark.required
class TestEmbeddings:
    """
    Test collate aggregates MAITE style data into separate collections from tuple return
    """

    @pytest.mark.parametrize(
        "data, labels, metadata",
        [
            [[0, 1, 2], [3, 4, 5], None],
            [np.ones((10, 3, 3)), np.ones((10,)), None],
            [np.ones((10, 3, 3)), np.ones((10, 3, 3)), None],
            [np.ones((10, 3, 3)), np.ones((10, 3)), [{i: i} for i in range(10)]],
            [
                np.ones((10, 3, 3)),
                [ObjectDetectionTarget([[0, 1, 2, 3], [4, 5, 6, 7]], [0, 1], [1, 0]) for _ in range(10)],
                [{i: i} for i in range(10)],
            ],
        ],
    )
    def test_mock_inputs(self, data, labels, metadata):
        """Tests common (input, target, metadata) dataset output"""
        ds = MockDataset(data, labels, metadata)
        em = Embeddings(ds, batch_size=64)

        assert len(ds) == len(em)

    @pytest.mark.parametrize(
        "data, targets",
        [
            [
                torch.ones((10, 1, 3, 3)),
                torch.nn.functional.one_hot(torch.arange(10)),
            ],
            [
                torch.ones((10, 1, 3, 3)),
                [ObjectDetectionTarget(torch.ones(10, 4), torch.zeros(10), torch.zeros(10)) for _ in range(10)],
            ],
        ],
    )
    def test_with_model_encode(self, data, targets):
        """Tests with basic identity model"""
        ds = TorchDataset(data, targets)
        em = Embeddings(ds, batch_size=64, model=IdentityModel(), device="cpu")

        assert len(ds) == len(em)
        assert len(em) == len(ds)

        for i in range(len(ds)):
            np.testing.assert_allclose(em[i], data[i])

        for idx, e in enumerate(em):
            np.testing.assert_allclose(ds[idx][0], e)

    def test_embeddings(self):
        embs = Embeddings(get_dataset(), 10, model=torch.nn.Identity(), transforms=lambda x: x + 1)
        assert len(embs[0:3]) == 3

        embs_tt = embs.to_tensor()
        assert isinstance(embs_tt, torch.Tensor)
        assert len(embs_tt) == len(embs)

        embs_np = np.asarray(embs)
        assert isinstance(embs_np, np.ndarray)
        assert len(embs_np) == len(embs)

        for emb in embs:
            assert np.array_equal(emb, np.ones((3, 16, 16)))

        with pytest.raises(TypeError):
            embs["string"]  # type: ignore

    def test_embeddings_getitem_types(self):
        embs = Embeddings(get_dataset(), 10, model=torch.nn.Identity(), transforms=lambda x: x + 1)
        assert len(embs[0])
        assert len(embs[0:2]) == 2
        assert len(embs[[0, 1]]) == 2
        assert len(embs[range(2)]) == 2
        assert len(embs[(i for i in [0, 1])]) == 2
        assert len(embs[np.array([0, 1])]) == 2

    def test_embeddings_getitem_types_raises(self):
        embs = Embeddings(get_dataset(), 10, model=torch.nn.Identity(), transforms=lambda x: x + 1)
        with pytest.raises(TypeError):
            embs["1"]  # type: ignore
        with pytest.raises(TypeError):
            embs["str"]  # type: ignore
        with pytest.raises(TypeError):
            embs[[[0, 1]]]  # type: ignore
        with pytest.raises(TypeError):
            embs[np.array([[0, 1]])]

    def test_embeddings_from_array(self):
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        embs = Embeddings.from_array(arr)
        assert isinstance(embs, Embeddings)
        assert len(embs) == arr.shape[0]
        assert np.array_equal(embs.to_tensor().numpy(), arr)

    def test_embeddings_embeddings_only_no_embeddings(self):
        embs = Embeddings([], 1)
        embs._embeddings_only = True
        with pytest.raises(ValueError):
            embs[0]

    def test_embeddings_new(self, torch_ic_ds):
        embs = Embeddings(torch_ic_ds, batch_size=64, model=IdentityModel(), device="cpu", transforms=lambda x: x + 1)
        mini_ds = TorchDataset(torch.ones((5, 1, 3, 3)), torch.nn.functional.one_hot(torch.arange(5)))
        mini_embs = embs.new(mini_ds)
        assert mini_embs.batch_size == embs.batch_size
        assert mini_embs.device == embs.device
        assert len(mini_embs) == 5
        assert mini_embs._dataset != embs._dataset
        assert mini_embs._transforms == embs._transforms
        assert mini_embs._model == embs._model

    def test_embeddings_new_embeddings_only_raises(self):
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        embs = Embeddings.from_array(arr)
        with pytest.raises(ValueError):
            embs.new([])

    @patch("dataeval.data._embeddings.np.save", side_effect=OSError())
    def test_embeddings_save_failure(self, tmp_path):
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        embs = Embeddings.from_array(arr)
        with pytest.raises(OSError):
            embs.save(tmp_path)

    def test_embeddings_layer_name_extraction(self, torch_ic_ds, sequential_model):
        """Test that layer_name correctly extracts embeddings from specified layer"""
        # Test extracting from the flatten layer (layer "2")
        embs = Embeddings(torch_ic_ds, batch_size=64, model=sequential_model, layer_name="2", device="cpu")

        # Get embeddings
        result = embs[0]

        # The flatten layer should output a 1D tensor with 36 elements (4 channels * 3*3 spatial)
        assert result.shape == (36,)
        assert isinstance(result, np.ndarray)

    def test_embeddings_layer_name_not_found(self, torch_ic_ds, sequential_model):
        """Test that layer_name raises ValueError when layer doesn't exist"""
        # Try to use a non-existent layer name
        with pytest.raises(ValueError, match="Invalid layer.*nonexistent"):
            Embeddings(torch_ic_ds, batch_size=64, model=sequential_model, layer_name="nonexistent", device="cpu")

    def test_embeddings_layer_name_vs_normal_output(self, torch_ic_ds, sequential_model):
        """Test that layer_name extraction gives different results than normal model output"""
        # Normal embeddings (final output)
        normal_embs = Embeddings(torch_ic_ds, batch_size=64, model=sequential_model, device="cpu")
        normal_result = normal_embs[0]

        # Layer embeddings (from flatten layer)
        layer_embs = Embeddings(torch_ic_ds, batch_size=64, model=sequential_model, layer_name="2", device="cpu")
        layer_result = layer_embs[0]

        # Results should have different shapes and values
        assert normal_result.shape != layer_result.shape
        assert normal_result.shape == (10,)  # Final linear layer output
        assert layer_result.shape == (36,)  # Flatten layer output
        assert not np.allclose(normal_result, layer_result[:10])  # Different values

    def test_embeddings_use_output_parameter(self, torch_ic_ds, sequential_model):
        """Test that use_output parameter correctly captures input vs output of layer"""
        # Capture output from flatten layer (default behavior)
        embs_output = Embeddings(
            torch_ic_ds, batch_size=64, model=sequential_model, layer_name="2", use_output=True, device="cpu"
        )

        # Capture input to flatten layer
        embs_input = Embeddings(
            torch_ic_ds, batch_size=64, model=sequential_model, layer_name="2", use_output=False, device="cpu"
        )

        output_result = embs_output[0]
        input_result = embs_input[0]

        # Output should be flattened (1D)
        assert output_result.shape == (36,)
        # Input should be the pre-flattened tensor (3D)
        assert input_result.shape == (4, 3, 3)

        # Verify they contain the same data, just in different shapes
        assert np.allclose(output_result, input_result.flatten())

    def test_hook_fn_captures_output(self, torch_ic_ds, sequential_model):
        """Test that _hook_fn correctly captures output when use_output=True"""
        embs = Embeddings(
            torch_ic_ds, batch_size=1, model=sequential_model, layer_name="2", use_output=True, device="cpu"
        )

        # Simulate what happens during forward pass
        sample_input = torch.ones(1, 4, 3, 3)
        sample_output = torch.ones(1, 36)  # Flattened output

        # Call hook function directly
        embs._hook_fn(torch.nn.Identity(), (sample_input,), sample_output)

        # Verify output was captured
        assert embs.captured_output is not None
        assert embs.captured_output.shape == sample_output.shape
        assert torch.allclose(embs.captured_output, sample_output)

        # Verify it's detached (no gradient tracking)
        assert not embs.captured_output.requires_grad

    def test_hook_fn_captures_input(self, torch_ic_ds, sequential_model):
        """Test that _hook_fn correctly captures input when use_output=False"""
        embs = Embeddings(
            torch_ic_ds, batch_size=1, model=sequential_model, layer_name="2", use_output=False, device="cpu"
        )

        # Simulate what happens during forward pass
        sample_input = torch.ones(1, 4, 3, 3)
        sample_output = torch.ones(1, 36)

        # Call hook function directly
        embs._hook_fn(torch.nn.Identity(), (sample_input,), sample_output)

        # Verify input was captured, not output
        assert embs.captured_output is not None
        assert embs.captured_output.shape == sample_input.shape
        assert torch.allclose(embs.captured_output, sample_input)
        assert not embs.captured_output.requires_grad

    def test_get_valid_layer_selection_returns_module(self, sequential_model):
        """Test that _get_valid_layer_selection returns correct module"""
        embs = Embeddings([], 1)

        # Get the flatten layer
        layer = embs._get_valid_layer_selection("2", sequential_model)

        assert isinstance(layer, torch.nn.Flatten)

        # Get the conv layer
        layer = embs._get_valid_layer_selection("0", sequential_model)
        assert isinstance(layer, torch.nn.Conv2d)

    def test_get_valid_layer_selection_non_pytorch_model(self):
        """Test that _get_valid_layer_selection raises TypeError for non-PyTorch model"""
        embs = Embeddings([], 1)

        # Try with a non-PyTorch model
        with pytest.raises(TypeError, match="Expected PyTorch model.*torch.nn.Module"):
            embs._get_valid_layer_selection("some_layer", "not_a_model")  # pyright: ignore[reportArgumentType]

    def test_get_valid_layer_selection_invalid_layer(self, sequential_model):
        """Test that _get_valid_layer_selection raises ValueError for invalid layer"""
        embs = Embeddings([], 1)

        with pytest.raises(ValueError, match="Invalid layer 'invalid_layer'.*Available layers"):
            embs._get_valid_layer_selection("invalid_layer", sequential_model)

    def test_encode_without_layer_name(self, torch_ic_ds):
        """Test that _encode returns normal model output when layer_name is None"""
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(9, 5))
        embs = Embeddings(torch_ic_ds, batch_size=1, model=model, layer_name=None, device="cpu")

        # Create sample images
        images = [torch.ones(1, 3, 3) for _ in range(2)]

        # Encode without layer extraction
        result = embs._encode(images)

        # Should be output of final linear layer
        assert result.shape == (2, 5)
        assert isinstance(result, np.ndarray)

    def test_encode_with_layer_name(self, torch_ic_ds, sequential_model):
        """Test that _encode returns hooked output when layer_name is set"""
        embs = Embeddings(
            torch_ic_ds, batch_size=1, model=sequential_model, layer_name="2", use_output=True, device="cpu"
        )

        # Create sample images (matching input shape for the model)
        images = [torch.ones(1, 3, 3) for _ in range(2)]

        # Encode with layer extraction
        result = embs._encode(images)

        # Should be output of flatten layer (layer "2"), not final linear layer
        assert result.shape == (2, 36)  # Flattened: 4 channels * 3 * 3
        assert isinstance(result, np.ndarray)

        # Verify captured_output was populated
        assert embs.captured_output is not None

    def test_encode_preserves_batch_dimension(self, torch_ic_ds, sequential_model):
        """Test that _encode correctly handles different batch sizes"""
        embs = Embeddings(torch_ic_ds, batch_size=1, model=sequential_model, layer_name="2", device="cpu")

        # Test with different batch sizes
        for batch_size in [1, 3, 5]:
            images = [torch.ones(1, 3, 3) for _ in range(batch_size)]
            result = embs._encode(images)

            assert result.shape[0] == batch_size
            assert result.shape[1:] == (36,)

    @pytest.mark.parametrize(
        "model_fixture, output_layer, input_layer, expected_output_shape, expected_input_shape, shapes_match",
        [
            (
                "sequential_model",
                "1",
                "2",
                (4, 3, 3),
                (4, 3, 3),
                True,
            ),  # ReLU output → Flatten input: direct connection
            (
                "model_with_functional_ops",
                "relu",
                "linear",
                (4, 3, 3),
                (36,),
                False,
            ),  # relu output → linear input: functional op between
        ],
    )
    def test_layer_output_vs_next_layer_input(
        self,
        torch_ic_ds,
        model_fixture,
        output_layer,
        input_layer,
        expected_output_shape,
        expected_input_shape,
        shapes_match,
        request,
    ):
        """Test both cases: where layer[i] output equals and doesn't equal layer[i+1] input"""
        model = request.getfixturevalue(model_fixture)

        # Capture output of layer N
        embs_output = Embeddings(
            torch_ic_ds, batch_size=1, model=model, layer_name=output_layer, use_output=True, device="cpu"
        )

        # Capture input of layer N+1
        embs_input = Embeddings(
            torch_ic_ds, batch_size=1, model=model, layer_name=input_layer, use_output=False, device="cpu"
        )

        output_result = embs_output[0]
        input_result = embs_input[0]

        assert output_result.shape == expected_output_shape
        assert input_result.shape == expected_input_shape
        assert (output_result.shape == input_result.shape) == shapes_match

    def test_embeddings_layer_logging(self, torch_ic_ds, sequential_model, caplog):
        with caplog.at_level(logging.DEBUG):
            _ = Embeddings(
                torch_ic_ds,
                batch_size=64,
                model=sequential_model,
                layer_name="2",
                use_output=True,
                device="cpu",
            )

            assert "Capturing output data from layer 2" in caplog.text

    def test_empty_dataset_shape(self):
        """Test that shape property handles empty dataset (line 149)"""
        empty_ds = MockDataset([], [], [])
        embs = Embeddings(empty_ds, batch_size=1, model=IdentityModel(), device="cpu")

        # Shape should be (0,) for empty dataset
        assert embs.shape == (0,)

    def test_hash_embeddings_only(self):
        """Test __hash__ for embeddings-only instance (lines 225-227)"""
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        embs = Embeddings.from_array(arr)

        # Should hash based on embeddings array data
        hash1 = hash(embs)
        assert isinstance(hash1, int)

        # Same data should give same hash
        embs2 = Embeddings.from_array(arr.copy())
        hash2 = hash(embs2)
        assert hash1 == hash2

        # Different data should give different hash
        arr_different = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        embs3 = Embeddings.from_array(arr_different)
        hash3 = hash(embs3)
        assert hash1 != hash3

    def test_hash_with_dataset_model_transforms(self, torch_ic_ds):
        """Test __hash__ for regular embeddings with dataset, model, and transforms (lines 228-231)"""
        model = IdentityModel()

        def transform(x):
            return x + 1

        embs = Embeddings(torch_ic_ds, batch_size=64, model=model, transforms=transform, device="cpu")

        # Should hash based on dataset, model, and transforms
        hash1 = hash(embs)
        assert isinstance(hash1, int)

        # Same configuration should give same hash
        embs2 = Embeddings(torch_ic_ds, batch_size=64, model=model, transforms=transform, device="cpu")
        hash2 = hash(embs2)
        assert hash1 == hash2

    def test_hash_different_transforms(self, torch_ic_ds):
        """Test __hash__ with different transforms produces different hash"""
        model = IdentityModel()

        embs1 = Embeddings(torch_ic_ds, batch_size=64, model=model, transforms=lambda x: x + 1, device="cpu")
        embs2 = Embeddings(torch_ic_ds, batch_size=64, model=model, transforms=lambda x: x + 2, device="cpu")

        hash1 = hash(embs1)
        hash2 = hash(embs2)

        # Different transforms should produce different hashes
        assert hash1 != hash2


class TestPathProperty:
    """Test suite for path property getter and setter"""

    def test_path_getter(self, tmp_path):
        """Test path property getter (line 237)"""
        from pathlib import Path

        path = tmp_path / "test_embeddings.npy"
        ds = get_dataset(10)
        embs = Embeddings(ds, batch_size=1, path=path, device="cpu")

        # Path property should return the configured path
        assert embs.path == path
        assert isinstance(embs.path, Path)

    def test_path_getter_none(self):
        """Test path property getter when path is None"""
        ds = get_dataset(10)
        embs = Embeddings(ds, batch_size=1, path=None, device="cpu")

        # Path property should return None
        assert embs.path is None

    def test_path_setter_none_converts_memmap_to_array(self, tmp_path):
        """Test path setter with None converts memmap to in-memory array (lines 241-247)"""
        path = tmp_path / "test.npy"
        arr = np.random.randn(100, 128).astype(np.float32)
        np.save(path, arr)

        # Load as memmap
        loaded = np.load(path, mmap_mode="r+")
        embs = Embeddings.from_array(loaded)
        assert isinstance(embs._embeddings, np.memmap)

        # Set path to None
        embs.path = None

        # Should convert to regular array
        assert not isinstance(embs._embeddings, np.memmap)
        assert isinstance(embs._embeddings, np.ndarray)
        assert embs._use_memmap is False
        np.testing.assert_array_equal(embs._embeddings, arr)

    def test_path_setter_new_path_saves_embeddings(self, tmp_path):
        """Test path setter with new path saves current embeddings (lines 248-254)"""
        old_path = tmp_path / "old.npy"
        new_path = tmp_path / "new.npy"

        # Create embeddings with old path and compute
        ds = get_dataset(10)
        embs = Embeddings(ds, batch_size=2, path=old_path, device="cpu")
        embs.compute()

        # Set new path
        embs.path = new_path

        # New path should be set and file should exist
        assert embs.path == new_path
        assert new_path.exists()

    def test_path_setter_same_path_no_save(self, tmp_path):
        """Test path setter with same path doesn't trigger save"""
        path = tmp_path / "test.npy"

        ds = get_dataset(10)
        embs = Embeddings(ds, batch_size=2, path=path, device="cpu")

        # Set same path (should be no-op)
        embs.path = path

        assert embs.path == path


class TestResolvePath:
    """Test suite for _resolve_path method"""

    def test_resolve_path_string_to_path(self, tmp_path):
        """Test _resolve_path converts string to absolute Path (line 257-258)"""
        ds = get_dataset(10)
        embs = Embeddings(ds, batch_size=1, device="cpu")

        str_path = str(tmp_path / "test.npy")
        resolved = embs._resolve_path(str_path)

        assert isinstance(resolved, type(tmp_path))  # Path type
        assert resolved.is_absolute()

    def test_resolve_path_directory_adds_filename(self, tmp_path):
        """Test _resolve_path adds filename for directory path (lines 259-260)"""
        ds = TorchDataset(torch.ones((10, 1, 3, 3)), torch.nn.functional.one_hot(torch.arange(10)))
        embs = Embeddings(ds, batch_size=1, device="cpu")

        # Pass directory path
        resolved = embs._resolve_path(tmp_path)

        # Should add filename based on hash
        assert resolved.parent == tmp_path
        assert resolved.suffix == ".npy"
        assert "emb-" in resolved.name

    def test_resolve_path_no_suffix_adds_filename(self, tmp_path):
        """Test _resolve_path adds filename for path without suffix (line 259-260)"""
        ds = get_dataset(10)
        embs = Embeddings(ds, batch_size=1, device="cpu")

        path_no_suffix = tmp_path / "embeddings"
        resolved = embs._resolve_path(path_no_suffix)

        # Should add filename based on hash
        assert resolved.parent == path_no_suffix
        assert resolved.suffix == ".npy"
        assert "emb-" in resolved.name


class TestProgressCallback:
    """Test suite for progress_callback functionality"""

    def test_progress_callback_called_during_compute(self):
        """Test that progress_callback is called during embedding computation"""
        ds = get_dataset(10)
        callback_calls = []

        def callback(step: int, *, total: int | None = None, desc: str | None = None, extra_info: dict | None = None):
            callback_calls.append({"step": step, "total": total, "desc": desc, "extra_info": extra_info})

        embs = Embeddings(ds, batch_size=2, model=torch.nn.Identity(), device="cpu", progress_callback=callback)
        _ = embs[:]  # Trigger computation

        # Callback should have been called
        assert len(callback_calls) > 0
        # Verify callback was called with proper parameters
        for call in callback_calls:
            assert "step" in call
            assert "total" in call
            assert call["total"] == 10  # Total should be the dataset length

    def test_progress_callback_not_called_when_none(self):
        """Test that no error occurs when progress_callback is None"""
        ds = get_dataset(10)
        embs = Embeddings(ds, batch_size=2, model=torch.nn.Identity(), device="cpu", progress_callback=None)
        _ = embs[:]  # Should work without error

    def test_progress_callback_with_getitem(self):
        """Test that progress_callback works with __getitem__"""
        ds = get_dataset(10)
        callback_calls = []

        def callback(step: int, *, total: int | None = None, desc: str | None = None, extra_info: dict | None = None):
            callback_calls.append({"step": step, "total": total})

        embs = Embeddings(ds, batch_size=3, model=torch.nn.Identity(), device="cpu", progress_callback=callback)
        _ = embs[0:5]  # Get first 5 items

        # Callback should have been called
        assert len(callback_calls) > 0

    def test_progress_callback_with_compute(self):
        """Test that progress_callback works with compute method"""
        ds = get_dataset(10)
        callback_calls = []

        def callback(step: int, *, total: int | None = None, desc: str | None = None, extra_info: dict | None = None):
            callback_calls.append({"step": step, "total": total})

        embs = Embeddings(ds, batch_size=2, model=torch.nn.Identity(), device="cpu", progress_callback=callback)
        embs.compute()

        # Callback should have been called
        assert len(callback_calls) > 0
        # Total should be the full dataset length
        for call in callback_calls:
            assert call["total"] == 10

    def test_progress_callback_preserved_in_new(self, torch_ic_ds):
        """Test that progress_callback is preserved when creating new embeddings"""
        callback_calls = []

        def callback(step: int, *, total: int | None = None, desc: str | None = None, extra_info: dict | None = None):
            callback_calls.append({"step": step, "total": total})

        embs = Embeddings(torch_ic_ds, batch_size=2, model=IdentityModel(), device="cpu", progress_callback=callback)
        mini_ds = TorchDataset(torch.ones((5, 1, 3, 3)), torch.nn.functional.one_hot(torch.arange(5)))
        mini_embs = embs.new(mini_ds)

        # Clear previous calls
        callback_calls.clear()

        # Use the new embeddings
        _ = mini_embs[:]

        # Callback should have been called with new dataset
        assert len(callback_calls) > 0
        for call in callback_calls:
            assert call["total"] == 5  # New dataset size
