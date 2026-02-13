"""Tests for OnnxExtractor."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from dataeval.extractors import OnnxExtractor
from dataeval.protocols import FeatureExtractor


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
class TestOnnxExtractorInit:
    """Test OnnxExtractor initialization."""

    def test_init_basic(self, simple_onnx_model: Path) -> None:
        """Test basic initialization."""
        extractor = OnnxExtractor(simple_onnx_model)
        assert extractor.output_name is None

    def test_init_with_string_path(self, simple_onnx_model: Path) -> None:
        """Test initialization with string path."""
        extractor = OnnxExtractor(str(simple_onnx_model))
        assert extractor._model_path == simple_onnx_model

    def test_init_with_output_name(self, simple_onnx_model: Path) -> None:
        """Test initialization with output name."""
        extractor = OnnxExtractor(simple_onnx_model, output_name="output")
        assert extractor.output_name == "output"

    def test_init_with_transforms(self, simple_onnx_model: Path) -> None:
        """Test initialization with transforms."""

        class MockTransform:
            def __call__(self, x: np.ndarray) -> np.ndarray:
                return x * 2

        extractor = OnnxExtractor(simple_onnx_model, transforms=MockTransform())
        assert len(extractor._transforms) == 1

    def test_init_with_multiple_transforms(self, simple_onnx_model: Path) -> None:
        """Test initialization with multiple transforms."""

        class ScaleTransform:
            def __call__(self, x: np.ndarray) -> np.ndarray:
                return x * 2

        class ShiftTransform:
            def __call__(self, x: np.ndarray) -> np.ndarray:
                return x + 1

        extractor = OnnxExtractor(simple_onnx_model, transforms=[ScaleTransform(), ShiftTransform()])
        assert len(extractor._transforms) == 2


@pytest.mark.optional
class TestOnnxExtractorCall:
    """Test OnnxExtractor.__call__ method."""

    @pytest.fixture
    def extractor(self, simple_onnx_model: Path) -> OnnxExtractor:
        """Create an extractor for testing."""
        return OnnxExtractor(simple_onnx_model)

    def test_call_batch_of_images(self, extractor: OnnxExtractor) -> None:
        """Test extracting features from a batch of images."""
        images = [np.random.randn(3, 16, 16).astype(np.float32) for _ in range(5)]
        result = extractor(images)
        assert result.shape[0] == 5
        assert result.shape[1] == 128

    def test_call_single_image(self, extractor: OnnxExtractor) -> None:
        """Test extracting features from a single image."""
        images = [np.random.randn(3, 16, 16).astype(np.float32)]
        result = extractor(images)
        assert result.shape[0] == 1
        assert result.shape[1] == 128

    def test_call_empty_list(self, extractor: OnnxExtractor) -> None:
        """Test extracting features from empty list."""
        result = extractor([])
        assert result.shape[0] == 0


@pytest.mark.optional
class TestOnnxExtractorModelLoading:
    """Test model loading behavior."""

    def test_model_not_found_raises(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for missing model."""
        extractor = OnnxExtractor(tmp_path / "nonexistent.onnx")
        images = [np.random.randn(3, 16, 16).astype(np.float32)]

        with pytest.raises(FileNotFoundError, match="Model not found"):
            extractor(images)

    def test_lazy_loading(self, simple_onnx_model: Path) -> None:
        """Test that model is loaded lazily."""
        extractor = OnnxExtractor(simple_onnx_model)
        assert extractor._session is None

        # Trigger loading
        images = [np.random.randn(3, 16, 16).astype(np.float32)]
        extractor(images)

        assert extractor._session is not None

    def test_invalid_output_name_raises(self, simple_onnx_model: Path) -> None:
        """Test that invalid output_name raises ValueError."""
        extractor = OnnxExtractor(simple_onnx_model, output_name="nonexistent")
        images = [np.random.randn(3, 16, 16).astype(np.float32)]

        with pytest.raises(ValueError, match="not found in model outputs"):
            extractor(images)


@pytest.mark.optional
class TestOnnxExtractorMultiOutput:
    """Test multi-output model handling."""

    def test_multi_output_requires_output_name(self, multi_output_onnx_model: Path) -> None:
        """Test that multi-output models require output_name."""
        extractor = OnnxExtractor(multi_output_onnx_model)
        images = [np.random.randn(3, 16, 16).astype(np.float32)]

        with pytest.raises(ValueError, match="Specify 'output_name'"):
            extractor(images)

    def test_multi_output_with_output_name(self, multi_output_onnx_model: Path) -> None:
        """Test extraction with specified output from multi-output model."""
        extractor = OnnxExtractor(multi_output_onnx_model, output_name="embeddings")
        images = [np.random.randn(3, 16, 16).astype(np.float32) for _ in range(5)]

        result = extractor(images)
        assert result.shape == (5, 128)

    def test_multi_output_select_second_output(self, multi_output_onnx_model: Path) -> None:
        """Test selecting second output from multi-output model."""
        extractor = OnnxExtractor(multi_output_onnx_model, output_name="output")
        images = [np.random.randn(3, 16, 16).astype(np.float32) for _ in range(5)]

        result = extractor(images)
        assert result.shape == (5, 64)


@pytest.mark.optional
class TestOnnxExtractorTransforms:
    """Test transform functionality."""

    def test_transforms_applied(self, simple_onnx_model: Path) -> None:
        """Test that transforms are applied during extraction."""

        class DoubleTransform:
            def __call__(self, x: np.ndarray) -> np.ndarray:
                return x * 2

        extractor_no_transform = OnnxExtractor(simple_onnx_model)
        extractor_with_transform = OnnxExtractor(simple_onnx_model, transforms=DoubleTransform())

        images = [np.ones((3, 16, 16), dtype=np.float32) for _ in range(3)]

        result_no_transform = extractor_no_transform(images)
        result_with_transform = extractor_with_transform(images)

        # Results should be different due to transform
        assert not np.allclose(result_no_transform, result_with_transform)


@pytest.mark.optional
class TestOnnxExtractorRepr:
    """Test __repr__ method."""

    def test_repr_basic(self, simple_onnx_model: Path) -> None:
        """Test basic repr."""
        extractor = OnnxExtractor(simple_onnx_model)
        repr_str = repr(extractor)
        assert "OnnxExtractor" in repr_str

    def test_repr_with_output_name(self, simple_onnx_model: Path) -> None:
        """Test repr includes output name when set."""
        extractor = OnnxExtractor(simple_onnx_model, output_name="output")
        repr_str = repr(extractor)
        assert "output_name='output'" in repr_str


@pytest.mark.optional
class TestOnnxExtractorProtocol:
    """Test that OnnxExtractor conforms to FeatureExtractor protocol."""

    def test_protocol_conformance(self, simple_onnx_model: Path) -> None:
        """Test that OnnxExtractor implements FeatureExtractor protocol."""
        extractor = OnnxExtractor(simple_onnx_model)
        assert isinstance(extractor, FeatureExtractor)
        assert callable(extractor)


@pytest.mark.optional
class TestOnnxExtractorImportError:
    """Test import error handling."""

    def test_import_error_message(self, simple_onnx_model: Path) -> None:
        """Test that helpful error message is shown when onnxruntime is missing."""
        with patch("dataeval.extractors._onnx._get_inference_session") as mock_get:
            mock_get.side_effect = ImportError(
                "onnxruntime is required for OnnxExtractor. "
                "Install it with: pip install onnxruntime (CPU) or pip install onnxruntime-gpu (GPU)",
            )

            extractor = OnnxExtractor(simple_onnx_model)
            images = [np.random.randn(3, 16, 16).astype(np.float32)]

            with pytest.raises(ImportError, match="onnxruntime is required"):
                extractor(images)


@pytest.mark.optional
class TestOnnxExtractorBytesInput:
    """Test loading models from bytes."""

    def test_init_with_bytes(self, simple_onnx_model: Path) -> None:
        """Test initialization with model bytes."""
        import onnx

        model = onnx.load(str(simple_onnx_model))
        model_bytes = model.SerializeToString()

        extractor = OnnxExtractor(model_bytes)
        assert extractor._model_bytes is not None
        assert extractor._model_path is None

    def test_call_with_bytes(self, simple_onnx_model: Path) -> None:
        """Test extraction using model bytes."""
        import onnx

        model = onnx.load(str(simple_onnx_model))
        model_bytes = model.SerializeToString()

        extractor = OnnxExtractor(model_bytes)
        images = [np.random.randn(3, 16, 16).astype(np.float32) for _ in range(5)]

        result = extractor(images)
        assert result.shape == (5, 128)

    def test_repr_with_bytes(self, simple_onnx_model: Path) -> None:
        """Test repr shows bytes info."""
        import onnx

        model = onnx.load(str(simple_onnx_model))
        model_bytes = model.SerializeToString()

        extractor = OnnxExtractor(model_bytes)
        repr_str = repr(extractor)
        assert "bytes" in repr_str
