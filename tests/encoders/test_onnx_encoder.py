"""Tests for OnnxEncoder."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from dataeval.encoders import OnnxEncoder
from dataeval.protocols import ArrayLike, Dataset


class MockDataset(Dataset[ArrayLike]):
    """Simple dataset for testing."""

    def __init__(self, images: np.ndarray):
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> ArrayLike:
        return self.images[index]


class MockDatasetWithLabels(Dataset[tuple[ArrayLike, Any, Any]]):
    """Simple dataset with labels for testing."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[ArrayLike, Any, Any]:
        return self.images[index], self.labels[index], {}


def _create_simple_onnx_model(model_path: Path, input_dim: int = 768, output_dim: int = 128) -> Path:
    """Create a simple ONNX model with Flatten + Linear layers.

    The model takes input of shape (batch, 3, 16, 16), flattens to (batch, 768),
    then applies a linear layer to produce (batch, output_dim).
    """
    from onnx import TensorProto, helper, numpy_helper, save

    # Input: (batch, 3, 16, 16)
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 16, 16])

    # Output: (batch, output_dim)
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", output_dim])

    # Flatten node: (batch, 3, 16, 16) -> (batch, 768)
    flatten_node = helper.make_node("Flatten", inputs=["input"], outputs=["flattened"], axis=1)

    # Linear weights and bias
    rng = np.random.default_rng(42)
    W = rng.standard_normal((input_dim, output_dim)).astype(np.float32)
    b = rng.standard_normal((output_dim,)).astype(np.float32)

    W_init = numpy_helper.from_array(W, "W")
    b_init = numpy_helper.from_array(b, "b")

    # MatMul node: (batch, 768) @ (768, output_dim) -> (batch, output_dim)
    matmul_node = helper.make_node("MatMul", inputs=["flattened", "W"], outputs=["matmul_out"])

    # Add bias: (batch, output_dim) + (output_dim,) -> (batch, output_dim)
    add_node = helper.make_node("Add", inputs=["matmul_out", "b"], outputs=["output"])

    # Create the graph
    graph = helper.make_graph(
        nodes=[flatten_node, matmul_node, add_node],
        name="simple_model",
        inputs=[X],
        outputs=[Y],
        initializer=[W_init, b_init],
    )

    # Create the model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    # Save the model
    save(model, str(model_path))
    return model_path


def _create_multi_output_onnx_model(model_path: Path) -> Path:
    """Create an ONNX model with multiple outputs.

    The model has two outputs:
    - embeddings: (batch, 128)
    - output: (batch, 64)
    """
    from onnx import TensorProto, helper, numpy_helper, save

    # Input
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 16, 16])

    # Outputs
    Y1 = helper.make_tensor_value_info("embeddings", TensorProto.FLOAT, ["batch", 128])
    Y2 = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 64])

    # Flatten
    flatten_node = helper.make_node("Flatten", inputs=["input"], outputs=["flattened"], axis=1)

    # First linear: 768 -> 128
    rng = np.random.default_rng(42)
    W1 = rng.standard_normal((768, 128)).astype(np.float32)
    b1 = rng.standard_normal((128,)).astype(np.float32)
    W1_init = numpy_helper.from_array(W1, "W1")
    b1_init = numpy_helper.from_array(b1, "b1")

    matmul1 = helper.make_node("MatMul", inputs=["flattened", "W1"], outputs=["matmul1_out"])
    add1 = helper.make_node("Add", inputs=["matmul1_out", "b1"], outputs=["embeddings"])

    # Second linear: 128 -> 64
    W2 = rng.standard_normal((128, 64)).astype(np.float32)
    b2 = rng.standard_normal((64,)).astype(np.float32)
    W2_init = numpy_helper.from_array(W2, "W2")
    b2_init = numpy_helper.from_array(b2, "b2")

    matmul2 = helper.make_node("MatMul", inputs=["embeddings", "W2"], outputs=["matmul2_out"])
    add2 = helper.make_node("Add", inputs=["matmul2_out", "b2"], outputs=["output"])

    # Create the graph
    graph = helper.make_graph(
        nodes=[flatten_node, matmul1, add1, matmul2, add2],
        name="multi_output_model",
        inputs=[X],
        outputs=[Y1, Y2],
        initializer=[W1_init, b1_init, W2_init, b2_init],
    )

    # Create the model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    save(model, str(model_path))
    return model_path


@pytest.fixture
def simple_onnx_model(tmp_path: Path) -> Path:
    """Create a simple ONNX model for testing."""
    return _create_simple_onnx_model(tmp_path / "test_model.onnx")


@pytest.fixture
def multi_output_onnx_model(tmp_path: Path) -> Path:
    """Create an ONNX model with multiple outputs for testing."""
    return _create_multi_output_onnx_model(tmp_path / "multi_output_model.onnx")


@pytest.mark.optional
class TestOnnxEncoderInit:
    """Test OnnxEncoder initialization."""

    def test_init_basic(self, simple_onnx_model: Path) -> None:
        """Test basic initialization."""
        encoder = OnnxEncoder(simple_onnx_model)
        assert encoder.batch_size > 0
        assert encoder.output_name is None

    def test_init_with_batch_size(self, simple_onnx_model: Path) -> None:
        """Test initialization with custom batch size."""
        encoder = OnnxEncoder(simple_onnx_model, batch_size=64)
        assert encoder.batch_size == 64

    def test_init_with_string_path(self, simple_onnx_model: Path) -> None:
        """Test initialization with string path."""
        encoder = OnnxEncoder(str(simple_onnx_model))
        assert encoder._model_path == simple_onnx_model

    def test_init_with_output_name(self, simple_onnx_model: Path) -> None:
        """Test initialization with output name."""
        encoder = OnnxEncoder(simple_onnx_model, output_name="output")
        assert encoder.output_name == "output"

    def test_init_with_transforms(self, simple_onnx_model: Path) -> None:
        """Test initialization with transforms."""

        class MockTransform:
            def __call__(self, x: np.ndarray) -> np.ndarray:
                return x * 2

        encoder = OnnxEncoder(simple_onnx_model, transforms=MockTransform())
        assert len(encoder._transforms) == 1

    def test_init_with_multiple_transforms(self, simple_onnx_model: Path) -> None:
        """Test initialization with multiple transforms."""

        class ScaleTransform:
            def __call__(self, x: np.ndarray) -> np.ndarray:
                return x * 2

        class ShiftTransform:
            def __call__(self, x: np.ndarray) -> np.ndarray:
                return x + 1

        encoder = OnnxEncoder(simple_onnx_model, transforms=[ScaleTransform(), ShiftTransform()])
        assert len(encoder._transforms) == 2


@pytest.mark.optional
class TestOnnxEncoderEncode:
    """Test OnnxEncoder.encode method."""

    @pytest.fixture
    def encoder(self, simple_onnx_model: Path) -> OnnxEncoder:
        """Create an encoder for testing."""
        return OnnxEncoder(simple_onnx_model, batch_size=10)

    @pytest.fixture
    def dataset(self) -> MockDatasetWithLabels:
        """Create a simple dataset for testing."""
        images = np.random.randn(20, 3, 16, 16).astype(np.float32)
        labels = np.arange(20)
        return MockDatasetWithLabels(images, labels)

    def test_encode_all_indices(self, encoder: OnnxEncoder, dataset: MockDatasetWithLabels) -> None:
        """Test encoding all indices."""
        indices = list(range(len(dataset)))
        result = encoder.encode(dataset, indices)
        assert result.shape[0] == 20
        assert result.shape[1] == 128  # Output dimension of the model

    def test_encode_subset_indices(self, encoder: OnnxEncoder, dataset: MockDatasetWithLabels) -> None:
        """Test encoding subset of indices."""
        indices = [0, 5, 10, 15]
        result = encoder.encode(dataset, indices)
        assert result.shape[0] == 4

    def test_encode_empty_indices(self, encoder: OnnxEncoder, dataset: MockDatasetWithLabels) -> None:
        """Test encoding with empty indices."""
        result = encoder.encode(dataset, [])
        assert result.shape[0] == 0

    def test_encode_single_index(self, encoder: OnnxEncoder, dataset: MockDatasetWithLabels) -> None:
        """Test encoding single index."""
        result = encoder.encode(dataset, [0])
        assert result.shape[0] == 1
        assert result.shape[1] == 128

    def test_encode_out_of_range_raises(self, encoder: OnnxEncoder, dataset: MockDatasetWithLabels) -> None:
        """Test that out-of-range indices raise IndexError."""
        with pytest.raises(IndexError, match="out of range"):
            encoder.encode(dataset, [0, 100])

    def test_encode_streaming_mode(self, encoder: OnnxEncoder, dataset: MockDatasetWithLabels) -> None:
        """Test encoding in streaming mode."""
        indices = list(range(15))
        batches = list(encoder.encode(dataset, indices, stream=True))

        # Should have multiple batches (batch_size=10, 15 items = 2 batches)
        assert len(batches) == 2

        # Check batch structure
        batch_indices, batch_embeddings = batches[0]
        assert len(batch_indices) == 10
        assert batch_embeddings.shape[0] == 10

        # Last batch should have remaining items
        batch_indices, batch_embeddings = batches[1]
        assert len(batch_indices) == 5
        assert batch_embeddings.shape[0] == 5

    def test_encode_streaming_empty(self, encoder: OnnxEncoder, dataset: MockDatasetWithLabels) -> None:
        """Test streaming mode with empty indices."""
        batches = list(encoder.encode(dataset, [], stream=True))
        assert batches == []

    def test_encode_dataset_without_labels(self, encoder: OnnxEncoder) -> None:
        """Test encoding dataset that returns only images."""
        images = np.random.randn(10, 3, 16, 16).astype(np.float32)
        dataset = MockDataset(images)
        result = encoder.encode(dataset, list(range(10)))
        assert result.shape[0] == 10
        assert result.shape[1] == 128


@pytest.mark.optional
class TestOnnxEncoderModelLoading:
    """Test model loading behavior."""

    def test_model_not_found_raises(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for missing model."""
        encoder = OnnxEncoder(tmp_path / "nonexistent.onnx")
        images = np.random.randn(5, 3, 16, 16).astype(np.float32)
        dataset = MockDataset(images)

        with pytest.raises(FileNotFoundError, match="Model not found"):
            encoder.encode(dataset, [0])

    def test_lazy_loading(self, simple_onnx_model: Path) -> None:
        """Test that model is loaded lazily."""
        encoder = OnnxEncoder(simple_onnx_model)
        # Session should not be loaded yet
        assert encoder._session is None

        # Trigger loading
        images = np.random.randn(5, 3, 16, 16).astype(np.float32)
        dataset = MockDataset(images)
        encoder.encode(dataset, [0])

        # Session should now be loaded
        assert encoder._session is not None

    def test_invalid_output_name_raises(self, simple_onnx_model: Path) -> None:
        """Test that invalid output_name raises ValueError."""
        encoder = OnnxEncoder(simple_onnx_model, output_name="nonexistent")
        images = np.random.randn(5, 3, 16, 16).astype(np.float32)
        dataset = MockDataset(images)

        with pytest.raises(ValueError, match="not found in model outputs"):
            encoder.encode(dataset, [0])


@pytest.mark.optional
class TestOnnxEncoderMultiOutput:
    """Test multi-output model handling."""

    def test_multi_output_requires_output_name(self, multi_output_onnx_model: Path) -> None:
        """Test that multi-output models require output_name."""
        encoder = OnnxEncoder(multi_output_onnx_model)
        images = np.random.randn(5, 3, 16, 16).astype(np.float32)
        dataset = MockDataset(images)

        with pytest.raises(ValueError, match="Specify 'output_name'"):
            encoder.encode(dataset, [0])

    def test_multi_output_with_output_name(self, multi_output_onnx_model: Path) -> None:
        """Test encoding with specified output from multi-output model."""
        encoder = OnnxEncoder(multi_output_onnx_model, output_name="embeddings", batch_size=10)
        images = np.random.randn(5, 3, 16, 16).astype(np.float32)
        dataset = MockDataset(images)

        result = encoder.encode(dataset, list(range(5)))
        assert result.shape == (5, 128)  # embeddings output shape

    def test_multi_output_select_second_output(self, multi_output_onnx_model: Path) -> None:
        """Test selecting second output from multi-output model."""
        encoder = OnnxEncoder(multi_output_onnx_model, output_name="output", batch_size=10)
        images = np.random.randn(5, 3, 16, 16).astype(np.float32)
        dataset = MockDataset(images)

        result = encoder.encode(dataset, list(range(5)))
        assert result.shape == (5, 64)  # output shape


@pytest.mark.optional
class TestOnnxEncoderTransforms:
    """Test transform functionality."""

    def test_transforms_applied(self, simple_onnx_model: Path) -> None:
        """Test that transforms are applied during encoding."""

        class DoubleTransform:
            def __call__(self, x: np.ndarray) -> np.ndarray:
                return x * 2

        encoder_no_transform = OnnxEncoder(simple_onnx_model, batch_size=10)
        encoder_with_transform = OnnxEncoder(simple_onnx_model, batch_size=10, transforms=DoubleTransform())

        images = np.ones((3, 3, 16, 16), dtype=np.float32)
        dataset = MockDataset(images)

        result_no_transform = encoder_no_transform.encode(dataset, [0])
        result_with_transform = encoder_with_transform.encode(dataset, [0])

        # Results should be different due to transform
        assert not np.allclose(result_no_transform, result_with_transform)


@pytest.mark.optional
class TestOnnxEncoderRepr:
    """Test __repr__ method."""

    def test_repr_basic(self, simple_onnx_model: Path) -> None:
        """Test basic repr."""
        encoder = OnnxEncoder(simple_onnx_model, batch_size=32)
        repr_str = repr(encoder)
        assert "OnnxEncoder" in repr_str
        assert "batch_size=32" in repr_str

    def test_repr_with_output_name(self, simple_onnx_model: Path) -> None:
        """Test repr includes output name when set."""
        encoder = OnnxEncoder(simple_onnx_model, batch_size=32, output_name="output")
        repr_str = repr(encoder)
        assert "output_name='output'" in repr_str


@pytest.mark.optional
class TestOnnxEncoderProtocol:
    """Test that OnnxEncoder conforms to EmbeddingEncoder protocol."""

    def test_protocol_conformance(self, simple_onnx_model: Path) -> None:
        """Test that OnnxEncoder has required protocol methods."""
        encoder = OnnxEncoder(simple_onnx_model)

        # Check required properties
        assert hasattr(encoder, "batch_size")
        assert isinstance(encoder.batch_size, int)

        # Check required methods
        assert hasattr(encoder, "encode")
        assert callable(encoder.encode)


@pytest.mark.optional
class TestOnnxEncoderBatching:
    """Test batching behavior."""

    def test_batching_with_small_batch(self, simple_onnx_model: Path) -> None:
        """Test encoding with small batch size."""
        encoder = OnnxEncoder(simple_onnx_model, batch_size=2)
        images = np.random.randn(10, 3, 16, 16).astype(np.float32)
        dataset = MockDataset(images)

        result = encoder.encode(dataset, list(range(10)))
        assert result.shape == (10, 128)

    def test_batching_with_large_batch(self, simple_onnx_model: Path) -> None:
        """Test encoding with batch size larger than dataset."""
        encoder = OnnxEncoder(simple_onnx_model, batch_size=100)
        images = np.random.randn(5, 3, 16, 16).astype(np.float32)
        dataset = MockDataset(images)

        result = encoder.encode(dataset, list(range(5)))
        assert result.shape == (5, 128)

    def test_batching_streaming_batch_count(self, simple_onnx_model: Path) -> None:
        """Test that streaming yields correct number of batches."""
        encoder = OnnxEncoder(simple_onnx_model, batch_size=3)
        images = np.random.randn(10, 3, 16, 16).astype(np.float32)
        dataset = MockDataset(images)

        batches = list(encoder.encode(dataset, list(range(10)), stream=True))

        # With batch_size=3 and 10 items, we expect 4 batches: 3, 3, 3, 1
        assert len(batches) == 4
        assert len(batches[0][0]) == 3
        assert len(batches[1][0]) == 3
        assert len(batches[2][0]) == 3
        assert len(batches[3][0]) == 1


@pytest.mark.optional
class TestOnnxEncoderImportError:
    """Test import error handling."""

    def test_import_error_message(self, simple_onnx_model: Path) -> None:
        """Test that helpful error message is shown when onnxruntime is missing."""
        with patch("dataeval.encoders._onnx._get_inference_session") as mock_get:
            mock_get.side_effect = ImportError(
                "onnxruntime is required for OnnxEncoder. "
                "Install it with: pip install onnxruntime (CPU) or pip install onnxruntime-gpu (GPU)"
            )

            encoder = OnnxEncoder(simple_onnx_model)
            images = np.random.randn(5, 3, 16, 16).astype(np.float32)
            dataset = MockDataset(images)

            with pytest.raises(ImportError, match="onnxruntime is required"):
                encoder.encode(dataset, [0])


@pytest.mark.optional
class TestOnnxEncoderBytesInput:
    """Test loading models from bytes."""

    def test_init_with_bytes(self, simple_onnx_model: Path) -> None:
        """Test initialization with model bytes."""
        import onnx

        model = onnx.load(str(simple_onnx_model))
        model_bytes = model.SerializeToString()

        encoder = OnnxEncoder(model_bytes, batch_size=10)
        assert encoder._model_bytes is not None
        assert encoder._model_path is None

    def test_encode_with_bytes(self, simple_onnx_model: Path) -> None:
        """Test encoding using model bytes."""
        import onnx

        model = onnx.load(str(simple_onnx_model))
        model_bytes = model.SerializeToString()

        encoder = OnnxEncoder(model_bytes, batch_size=10)
        images = np.random.randn(5, 3, 16, 16).astype(np.float32)
        dataset = MockDataset(images)

        result = encoder.encode(dataset, list(range(5)))
        assert result.shape == (5, 128)

    def test_repr_with_bytes(self, simple_onnx_model: Path) -> None:
        """Test repr shows bytes info."""
        import onnx

        model = onnx.load(str(simple_onnx_model))
        model_bytes = model.SerializeToString()

        encoder = OnnxEncoder(model_bytes, batch_size=10)
        repr_str = repr(encoder)
        assert "bytes" in repr_str
        assert "batch_size=10" in repr_str
