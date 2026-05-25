"""Shared test helpers for verification tests.

Provides lightweight mock objects that satisfy DataEval protocols without
depending on the unit test suite or heavyweight dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import DatasetMetadata


@dataclass
class SimpleMetadata:
    """Minimal implementation of the Metadata protocol for verification tests."""

    class_labels: NDArray[np.intp]
    factor_data: NDArray[np.int64]
    factor_names: list[str]
    is_discrete: list[bool]
    index2label: dict[int, str] = field(default_factory=dict)


def make_metadata(
    n_samples: int = 60,
    n_factors: int = 3,
    n_classes: int = 3,
    seed: int = 42,
) -> SimpleMetadata:
    """Create a simple metadata object for bias evaluation tests."""
    rng = np.random.default_rng(seed)
    class_labels = np.tile(np.arange(n_classes, dtype=np.intp), n_samples // n_classes + 1)[:n_samples]
    factor_data = rng.integers(0, 4, size=(n_samples, n_factors), dtype=np.int64)
    factor_names = [f"factor_{i}" for i in range(n_factors)]
    is_discrete = [True] * n_factors
    index2label = {i: f"class_{i}" for i in range(n_classes)}

    return SimpleMetadata(
        class_labels=class_labels,
        factor_data=factor_data,
        factor_names=factor_names,
        is_discrete=is_discrete,
        index2label=index2label,
    )


@dataclass
class SimpleImageDataset:
    """Minimal dataset satisfying the AnnotatedDataset protocol for image data."""

    images: NDArray[np.floating]
    _metadata: DatasetMetadata = field(default_factory=lambda: DatasetMetadata(id="verification-test"))

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    def __getitem__(self, idx: int):
        return self.images[idx]

    def __len__(self) -> int:
        return len(self.images)


def build_simple_onnx_model(model_path: Path, output_dim: int = 64) -> Path:
    """Build a small ONNX encoder: ``(batch, 3, 16, 16) -> (batch, output_dim)``.

    Uses a Flatten + MatMul + Add (linear layer) graph. Sufficient for
    end-to-end interoperability tests that need a real ONNX file without
    pulling a heavyweight pretrained model.
    """
    from onnx import TensorProto, helper, numpy_helper, save

    input_dim = 3 * 16 * 16
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 16, 16])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", output_dim])

    rng = np.random.default_rng(0)
    w = rng.standard_normal((input_dim, output_dim)).astype(np.float32)
    b = rng.standard_normal((output_dim,)).astype(np.float32)

    graph = helper.make_graph(
        nodes=[
            helper.make_node("Flatten", inputs=["input"], outputs=["flat"], axis=1),
            helper.make_node("MatMul", inputs=["flat", "W"], outputs=["matmul_out"]),
            helper.make_node("Add", inputs=["matmul_out", "b"], outputs=["output"]),
        ],
        name="simple_encoder",
        inputs=[x],
        outputs=[y],
        initializer=[numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    save(model, str(model_path))
    return model_path


@dataclass
class SimpleAnnotatedDataset:
    """Minimal dataset satisfying the AnnotatedDataset protocol."""

    images: NDArray[np.floating]
    labels: NDArray[np.integer]
    _metadata: DatasetMetadata = field(default_factory=lambda: DatasetMetadata(id="verification-test"))

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    def __getitem__(self, idx: int):
        return self.images[idx], int(self.labels[idx]), {}

    def __len__(self) -> int:
        return len(self.images)
