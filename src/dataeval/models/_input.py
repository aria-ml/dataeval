"""Build the opinionated `image` input tensor for CV models (IR-3.1-S-1 / S-4)."""

__all__ = ["build_model_input"]

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval.extractors._resize import resize_chw
from dataeval.models._metadata import ModelIOSpec
from dataeval.utils._internal import as_numpy


def _to_channels(img: NDArray[Any], channels: str) -> NDArray[Any]:
    c = img.shape[0]
    target = 3 if channels == "RGB" else 1
    if c == target:
        return img
    if target == 1 and c == 3:
        return img.mean(axis=0, keepdims=True)
    if target == 3 and c == 1:
        return np.repeat(img, 3, axis=0)
    raise ValueError(f"cannot convert image with {c} channels to {channels}")


def _normalize(img: NDArray[Any]) -> NDArray[np.float32]:
    # Integer pixels are assumed 8-bit [0, 255] and scaled to [0, 1]. Float inputs
    # are assumed already normalized and passed through unchanged -- dividing a
    # float image by 255 on a value-magnitude heuristic silently blacks out a
    # legitimately normalized image that slightly overshoots 1.0.
    arr = img.astype(np.float32)
    if np.issubdtype(img.dtype, np.integer):
        arr = arr / 255.0
    return arr


def build_model_input(
    images: Sequence[ArrayLike],
    spec: ModelIOSpec,
    *,
    height: int | None = None,
    width: int | None = None,
) -> NDArray[np.float32]:
    """
    Build the opinionated ``image`` input tensor for a CV model.

    Each input image is normalized, converted to the color layout declared by
    ``spec``, resized to the target ``(height, width)``, and stacked into a single
    batched FP32 tensor. Integer pixels are assumed 8-bit ``[0, 255]`` and scaled
    to ``[0, 1]``; float images are assumed already normalized and passed through
    unchanged.

    Parameters
    ----------
    images : Sequence[ArrayLike]
        Images in CHW (channels, height, width) layout. Each may have a different
        spatial size; all are resized to the resolved target size.
    spec : ModelIOSpec
        Model input/output contract, typically from :func:`~dataeval.models.read_model_metadata`.
        Supplies the target color layout and the default ``height``/``width``.
    height : int or None, default None
        Target height override. Required when ``spec.height`` is ``-1`` (variable);
        otherwise overrides the spec value when set.
    width : int or None, default None
        Target width override. Required when ``spec.width`` is ``-1`` (variable);
        otherwise overrides the spec value when set.

    Returns
    -------
    NDArray[np.float32]
        Batched tensor of shape ``(B, C, H, W)`` with values in ``[0, 1]``, where
        ``B`` is ``len(images)`` and ``C`` is 3 for ``"RGB"`` or 1 for
        ``"GRAYSCALE"``.

    Raises
    ------
    ValueError
        If the resolved height or width is variable (``-1``) with no override, if
        an image is not 3-dimensional (CHW), or if its channel count cannot be
        converted to the target layout.

    See Also
    --------
    ~dataeval.models.ModelIOSpec : The input/output contract consumed here.

    Examples
    --------
    >>> import numpy as np
    >>> spec = ModelIOSpec(
    ...     task="IMAGE_CLASSIFICATION",
    ...     channels="RGB",
    ...     height=8,
    ...     width=8,
    ...     batch_size=-1,
    ...     n_classes=4,
    ... )
    >>> images = [np.zeros((3, 16, 16), dtype=np.uint8), np.full((3, 12, 20), 255, dtype=np.uint8)]
    >>> tensor = build_model_input(images, spec)
    >>> tensor.shape
    (2, 3, 8, 8)
    >>> tensor.dtype
    dtype('float32')
    """
    h = height if height is not None else spec.height
    w = width if width is not None else spec.width
    if h is None or h < 1:
        raise ValueError("model input height is variable (-1); set an explicit height override")
    if w is None or w < 1:
        raise ValueError("model input width is variable (-1); set an explicit width override")

    built: list[NDArray[np.float32]] = []
    for image in images:
        arr = as_numpy(image)
        if arr.ndim != 3:
            raise ValueError(f"model input expects CHW images; got shape {arr.shape}")
        # Normalize first, while the original dtype is intact: channel conversion
        # (mean) and resize both produce float arrays, which would defeat a
        # dtype-gated normalization if done afterward.
        arr = _normalize(arr)
        arr = _to_channels(arr, spec.channels)
        arr = resize_chw(arr, (h, w))
        built.append(arr)
    return np.stack(built).astype(np.float32)
