"""Tests for ONNX utilities."""

from pathlib import Path

import numpy as np
import pytest


def _create_simple_onnx_model(model_path: Path) -> Path:
    """Create a simple ONNX classification model for testing."""
    from onnx import TensorProto, helper, numpy_helper, save

    # Input: (batch, 3, 16, 16)
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 16, 16])
    # Output: (batch, 10) - classification logits
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 10])

    # Flatten node: (batch, 3, 16, 16) -> (batch, 768)
    flatten_node = helper.make_node("Flatten", inputs=["input"], outputs=["flattened"], axis=1)

    # Gemm (FC layer): 768 -> 10
    rng = np.random.default_rng(42)
    W = rng.standard_normal((768, 10)).astype(np.float32)
    b = rng.standard_normal((10,)).astype(np.float32)

    W_init = numpy_helper.from_array(W, "W")
    b_init = numpy_helper.from_array(b, "b")

    # Gemm node
    gemm_node = helper.make_node(
        "Gemm",
        inputs=["flattened", "W", "b"],
        outputs=["output"],
        alpha=1.0,
        beta=1.0,
        transB=0,
    )

    graph = helper.make_graph(
        nodes=[flatten_node, gemm_node],
        name="simple_classifier",
        inputs=[X],
        outputs=[Y],
        initializer=[W_init, b_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    save(model, str(model_path))
    return model_path


@pytest.fixture
def simple_classifier_model(tmp_path: Path) -> Path:
    """Create a simple ONNX classifier for testing."""
    return _create_simple_onnx_model(tmp_path / "classifier.onnx")


@pytest.mark.optional
class TestFindEmbeddingLayer:
    """Test find_embedding_layer function."""

    def test_finds_gemm_input(self, simple_classifier_model: Path) -> None:
        """Test that it finds the input to the Gemm layer."""
        from dataeval.utils.onnx import find_embedding_layer

        layer = find_embedding_layer(simple_classifier_model)
        assert layer == "flattened"

    def test_raises_on_nonexistent_file(self, tmp_path: Path) -> None:
        """Test that it raises FileNotFoundError for missing file."""
        from dataeval.utils.onnx import find_embedding_layer

        with pytest.raises(Exception):  # Could be FileNotFoundError or onnx error
            find_embedding_layer(tmp_path / "nonexistent.onnx")


@pytest.mark.optional
class TestToEncodingModel:
    """Test to_encoding_model function."""

    def test_returns_bytes_and_layer_name(self, simple_classifier_model: Path) -> None:
        """Test that it returns serialized model bytes and layer name."""
        from dataeval.utils.onnx import to_encoding_model

        model_bytes, layer_name = to_encoding_model(simple_classifier_model)
        assert isinstance(model_bytes, bytes)
        assert len(model_bytes) > 0
        assert isinstance(layer_name, str)
        assert layer_name == "flattened"

    def test_adds_embedding_output(self, simple_classifier_model: Path) -> None:
        """Test that the returned model has the embedding layer as output."""
        import onnx

        from dataeval.utils.onnx import to_encoding_model

        model_bytes, layer_name = to_encoding_model(simple_classifier_model)

        # Load the modified model from bytes
        model = onnx.load_from_string(model_bytes)
        output_names = [out.name for out in model.graph.output]

        assert layer_name in output_names

    def test_with_explicit_layer(self, simple_classifier_model: Path) -> None:
        """Test that it works with an explicitly specified layer."""
        from dataeval.utils.onnx import find_embedding_layer, to_encoding_model

        embedding_layer = find_embedding_layer(simple_classifier_model)
        model_bytes, layer_name = to_encoding_model(simple_classifier_model, embedding_layer=embedding_layer)

        assert layer_name == embedding_layer
        assert isinstance(model_bytes, bytes)

    def test_works_with_onnx_encoder(self, simple_classifier_model: Path) -> None:
        """Test that the result can be used with OnnxEncoder."""
        from dataeval.encoders import OnnxEncoder
        from dataeval.utils.onnx import to_encoding_model

        model_bytes, layer_name = to_encoding_model(simple_classifier_model)

        # Create encoder with bytes
        encoder = OnnxEncoder(model_bytes, batch_size=10, output_name=layer_name)

        # Create a simple dataset
        class SimpleDataset:
            def __init__(self, images: np.ndarray):
                self.images = images

            def __len__(self) -> int:
                return len(self.images)

            def __getitem__(self, idx: int) -> np.ndarray:
                return self.images[idx]

        images = np.random.randn(5, 3, 16, 16).astype(np.float32)
        dataset = SimpleDataset(images)

        # Encode
        result = encoder.encode(dataset, list(range(5)))

        # Should get embeddings with dimension 768 (flattened 3*16*16)
        assert result.shape == (5, 768)


@pytest.mark.optional
class TestToEncodingModelWithFile:
    """Test to_encoding_model function with file output."""

    def test_creates_file(self, simple_classifier_model: Path, tmp_path: Path) -> None:
        """Test that it creates a new model file."""
        from dataeval.utils.onnx import to_encoding_model

        output_path = tmp_path / "embedding_model.onnx"
        result_path, layer_name = to_encoding_model(simple_classifier_model, output_path)

        assert output_path.exists()
        assert result_path == str(output_path)
        assert isinstance(layer_name, str)

    def test_adds_embedding_output_to_file(self, simple_classifier_model: Path, tmp_path: Path) -> None:
        """Test that the created model has the embedding layer as output."""
        import onnx

        from dataeval.utils.onnx import find_embedding_layer, to_encoding_model

        embedding_layer = find_embedding_layer(simple_classifier_model)
        output_path = tmp_path / "embedding_model.onnx"
        _, layer_name = to_encoding_model(simple_classifier_model, output_path, embedding_layer)

        assert layer_name == embedding_layer

        model = onnx.load(str(output_path))
        output_names = [out.name for out in model.graph.output]

        assert embedding_layer in output_names
