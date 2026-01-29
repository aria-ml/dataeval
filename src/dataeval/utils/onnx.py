"""
Utilities for working with ONNX models.

These utilities help prepare ONNX models for embedding extraction by
identifying and exposing intermediate layers that produce embeddings.
"""

__all__ = ["find_embedding_layer", "to_encoding_model"]

from pathlib import Path
from types import ModuleType
from typing import overload


def _get_onnx() -> ModuleType:
    """Import onnx and return the module."""
    try:
        import onnx

        return onnx
    except ImportError as e:
        raise ImportError("onnx is required for ONNX model utilities. Install it with: pip install onnx") from e


def find_embedding_layer(model_path: str | Path) -> str:
    """
    Find the embedding layer name in an ONNX classification model.

    For classification models like ResNet, this identifies the layer that
    produces embeddings (feature vectors) before the final classification
    layer. This is typically the input to the last fully-connected (Gemm)
    layer or the output of a GlobalAveragePool layer.

    Parameters
    ----------
    model_path : str or Path
        Path to the ONNX model file.

    Returns
    -------
    str
        Name of the embedding layer output.

    Raises
    ------
    ImportError
        If the onnx package is not installed.
    ValueError
        If the embedding layer cannot be identified.
    """
    onnx = _get_onnx()
    model = onnx.load(str(model_path))

    # Look for the last Gemm (fully-connected) layer
    # The input to this layer is the embedding
    last_gemm_node = None
    for node in model.graph.node:
        if node.op_type == "Gemm":
            last_gemm_node = node

    if last_gemm_node:
        return last_gemm_node.input[0]

    # Fallback: look for GlobalAveragePool
    for node in reversed(model.graph.node):
        if node.op_type == "GlobalAveragePool":
            return node.output[0]

    raise ValueError(
        "Could not identify the embedding layer. "
        "The model must have a Gemm (fully-connected) or GlobalAveragePool layer."
    )


@overload
def to_encoding_model(
    model_path: str | Path,
    output_path: None = None,
    embedding_layer: str | None = None,
) -> tuple[bytes, str]: ...


@overload
def to_encoding_model(
    model_path: str | Path,
    output_path: str | Path,
    embedding_layer: str | None = None,
) -> tuple[str, str]: ...


def to_encoding_model(
    model_path: str | Path,
    output_path: str | Path | None = None,
    embedding_layer: str | None = None,
) -> tuple[bytes, str] | tuple[str, str]:
    """
    Modify an ONNX model to output embeddings from an intermediate layer.

    This function modifies a classification model to expose an intermediate
    layer as an additional output, allowing extraction of embeddings instead
    of (or in addition to) classification logits.

    Parameters
    ----------
    model_path : str or Path
        Path to the original ONNX model file.
    output_path : str, Path, or None, default None
        Path where the modified model will be saved. If None, returns
        serialized model bytes instead of writing to disk.
    embedding_layer : str or None, default None
        Name of the layer to expose as output. If None, automatically
        detected using :func:`find_embedding_layer`.

    Returns
    -------
    tuple[bytes, str] or tuple[str, str]
        If output_path is None: (model_bytes, layer_name)
        If output_path is provided: (output_path, layer_name)

    Raises
    ------
    ImportError
        If the onnx package is not installed.
    ValueError
        If the embedding layer cannot be identified (when embedding_layer is None).
    """
    onnx = _get_onnx()
    model = onnx.load(str(model_path))

    # Find embedding layer if not provided
    if embedding_layer is None:
        embedding_layer = find_embedding_layer(model_path)

    model = onnx.shape_inference.infer_shapes(model)

    # Locate the layer's info (Input, Output, or ValueInfo)
    target_info = next((o for o in model.graph.output if o.name == embedding_layer), None)

    # If not an output, check internal value_info
    if target_info is None:
        target_info = next((vi for vi in model.graph.value_info if vi.name == embedding_layer), None)

        if target_info is None:
            raise ValueError(f"Layer '{embedding_layer}' not found in graph shapes.")

        # Promote to output since it wasn't one already
        model.graph.output.extend([target_info])

    model_bytes = model.SerializeToString()

    if output_path is not None:
        Path(output_path).write_bytes(model_bytes)
        return str(output_path), embedding_layer

    return model_bytes, embedding_layer
