"""Array conversion, shape and dtype utilities across NumPy/PyTorch/MAITE."""

__all__ = []

import logging
from collections.abc import Iterable, Iterator
from typing import Any, Literal, TypeVar, overload

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from dataeval._log import LogMessage, get_logger
from dataeval.exceptions import ShapeMismatchError
from dataeval.protocols import Array, SequenceLike
from dataeval.utils._internal import try_import

_logger = get_logger(__name__)


np_dtype = TypeVar("np_dtype", bound=np.generic)


def as_numpy(
    array: ArrayLike | SequenceLike[Any] | None,
    *,
    dtype: type[np_dtype] | None = None,
    required_ndim: int | Iterable[int] | None = None,
    required_shape: tuple[int, ...] | None = None,
) -> NDArray[np_dtype]:
    """
    Convert an ArrayLike to Numpy array without copying (if possible).

    Parameters
    ----------
    array : ArrayLike or SequenceLike or None
        Input array-like object
    dtype : numpy dtype or None, default None
        Desired output dtype
    required_ndim : int or Iterable[int] or None, default None
        Required number of dimensions (or set of valid dimensions)
    required_shape : tuple[int, ...] or None, default None
        Required shape of output

    Returns
    -------
    NDArray
        NumPy array
    """
    return to_numpy(array, dtype=dtype, required_ndim=required_ndim, required_shape=required_shape, copy=False)


def argmax_label(target: ArrayLike) -> int:
    """Return the predicted class index (top-1) of an image-classification score vector."""
    return int(np.argmax(as_numpy(target)))


def to_numpy(  # noqa: C901
    array: ArrayLike | SequenceLike[Any] | None,
    *,
    dtype: type[np_dtype] | None = None,
    required_ndim: int | Iterable[int] | None = None,
    required_shape: tuple[int, ...] | None = None,
    copy: bool = True,
) -> NDArray[np_dtype]:
    """
    Convert an ArrayLike to new Numpy array.

    Parameters
    ----------
    array : ArrayLike or SequenceLike or None
        Input array-like object
    dtype : numpy dtype or None, default None
        Desired output dtype
    required_ndim : int or Iterable[int] or None, default None
        Required number of dimensions (or set of valid dimensions)
    required_shape : tuple[int, ...] or None, default None
        Required shape of output
    copy : bool, default True
        Whether to copy the array

    Returns
    -------
    NDArray
        NumPy array

    Raises
    ------
    ShapeMismatchError
        If required_ndim or required_shape constraints are not met
    """
    _array: NDArray[np_dtype] | None = None

    if array is None:
        _array = np.array([], dtype=dtype)
    elif isinstance(array, np.ndarray | np.memmap):
        numpy = array.copy() if copy else array
        _array = numpy.astype(dtype) if dtype is not None else numpy
    elif array.__class__.__module__.startswith("tensorflow"):  # pragma: no cover - removed tf from deps
        tf = try_import("tensorflow")
        if tf and tf.is_tensor(array):
            _logger.log(logging.INFO, "Converting Tensorflow array to NumPy array.")
            numpy = array.numpy().copy() if copy else array.numpy()  # type: ignore
            _array = numpy.astype(dtype) if dtype is not None else numpy
    elif array.__class__.__module__.startswith("torch"):
        torch = try_import("torch")
        if torch and isinstance(array, torch.Tensor):
            _logger.log(logging.INFO, "Converting PyTorch array to NumPy array.")
            numpy = array.detach().cpu().numpy().copy() if copy else array.detach().cpu().numpy()  # type: ignore
            _logger.log(logging.DEBUG, LogMessage(lambda: f"{str(array)} -> {str(numpy)}"))
            _array = numpy.astype(dtype) if dtype is not None else numpy

    # If the array was not converted yet, let numpy create the array directly
    if _array is None:
        _array = np.array(array, dtype=dtype) if copy else np.asarray(array, dtype=dtype)

    required_ndims = (required_ndim,) if isinstance(required_ndim, int) else required_ndim
    if required_ndims is not None and _array.ndim not in required_ndims:
        raise ShapeMismatchError(f"Array has {_array.ndim} dimensions, expected {required_ndim}.")

    if required_shape is not None and _array.shape != required_shape:
        raise ShapeMismatchError(f"Array has shape {_array.shape}, expected {required_shape}.")

    return _array


def to_numpy_iter(iterable: Iterable[ArrayLike]) -> Iterator[NDArray[Any]]:
    """
    Yield an iterator of numpy arrays from an ArrayLike iterable.

    Parameters
    ----------
    iterable : Iterable[ArrayLike]
        Iterable of array-like objects

    Yields
    ------
    NDArray
        NumPy arrays
    """
    for array in iterable:
        yield to_numpy(array)


@overload
def rescale_array(array: NDArray[np_dtype]) -> NDArray[np_dtype]: ...
@overload
def rescale_array(array: torch.Tensor) -> torch.Tensor: ...
def rescale_array(array: Array | NDArray[np_dtype] | torch.Tensor) -> Array | NDArray[np_dtype] | torch.Tensor:
    """
    Rescale an array to the range [0, 1].

    Parameters
    ----------
    array : NDArray or torch.Tensor
        Input array

    Returns
    -------
    NDArray or torch.Tensor
        Rescaled array in range [0, 1]

    Raises
    ------
    TypeError
        If array type is not supported
    """
    if isinstance(array, np.ndarray | torch.Tensor):
        arr_min = array.min()
        arr_max = array.max()
        return (array - arr_min) / (arr_max - arr_min)
    raise TypeError(f"Unsupported type: {type(array)}")


TArray = TypeVar("TArray", Array, np.ndarray, torch.Tensor)


@overload
def ensure_embeddings(
    embeddings: TArray,
    dtype: torch.dtype,
    unit_interval: Literal[True, False, "force"] = False,
) -> torch.Tensor: ...


@overload
def ensure_embeddings(
    embeddings: TArray,
    dtype: type[np_dtype],
    unit_interval: Literal[True, False, "force"] = False,
) -> NDArray[np_dtype]: ...


@overload
def ensure_embeddings(
    embeddings: TArray,
    dtype: None = None,
    unit_interval: Literal[True, False, "force"] = False,
) -> TArray: ...


def ensure_embeddings(  # noqa: C901
    embeddings: TArray,
    dtype: type[np_dtype] | torch.dtype | None = None,
    unit_interval: Literal[True, False, "force"] = False,
) -> torch.Tensor | NDArray[np_dtype] | TArray:
    """
    Validate the embeddings array and convert it to the specified type.

    Parameters
    ----------
    embeddings : ArrayLike
        Embeddings array
    dtype : numpy dtype or torch dtype or None, default None
        The desired dtype of the output array, None to skip conversion
    unit_interval : bool or "force", default False
        Whether to validate or force the embeddings to unit interval

    Returns
    -------
    torch.Tensor or NDArray or other Array
        Converted embeddings array

    Raises
    ------
    ShapeMismatchError
        If the embeddings array is not 2D
    ShapeMismatchError
        If the embeddings array has a zero dimension
    ValueError
        If the embeddings array is not unit interval [0, 1] (when unit_interval=True)
    """
    if isinstance(dtype, torch.dtype):
        arr = torch.as_tensor(embeddings, dtype=dtype)
    else:
        arr = (
            embeddings.detach().cpu().numpy().astype(dtype)
            if isinstance(embeddings, torch.Tensor)
            else np.asarray(embeddings, dtype=dtype)
        )

    if arr.ndim != 2:
        raise ShapeMismatchError(f"Expected a 2D array, but got a {arr.ndim}D array.")

    if np.prod(arr.shape) == 0:
        raise ShapeMismatchError(f"Array has at least one zero dimension: {arr.shape}.")

    if unit_interval and (arr.min() < 0 or arr.max() > 1):
        if unit_interval == "force":
            _logger.warning("Embeddings are not unit interval [0, 1]. Forcing to unit interval.")
            arr = rescale_array(arr)
        else:
            raise ValueError("Embeddings must be unit interval [0, 1].")

    if dtype is None:
        return embeddings
    return arr


@overload
def flatten_samples(array: torch.Tensor) -> torch.Tensor: ...
@overload
def flatten_samples(array: SequenceLike[Any]) -> NDArray[Any]: ...


def flatten_samples(array: SequenceLike[Any] | torch.Tensor) -> NDArray[Any] | torch.Tensor:
    """
    Flattens input array from (N, ...) to (N, -1) where all samples N have all data in their last dimension.

    Parameters
    ----------
    array : ArrayLike
        Input array with shape (N, ...)

    Returns
    -------
    np.ndarray or torch.Tensor
        Flattened array with shape (N, -1)

    Raises
    ------
    TypeError
        If array type is not supported
    """
    if isinstance(array, torch.Tensor):
        return torch.flatten(array, start_dim=1)
    if isinstance(array, str | bytes):
        raise TypeError(f"Unsupported array type {type(array)}.")
    try:
        nparr = as_numpy(array)
        return nparr.reshape((nparr.shape[0], -1))
    except (TypeError, ValueError) as e:
        raise TypeError(f"Unsupported array type {type(array)}: {e}.") from e


def channels_first_to_last(array: TArray) -> TArray:
    """
    Convert array from channels first to channels last format.

    Parameters
    ----------
    array : ArrayLike
        Input array in CHW format

    Returns
    -------
    ArrayLike
        Converted array in HWC format

    Raises
    ------
    TypeError
        If array type is not supported
    """
    if isinstance(array, np.ndarray):
        return np.transpose(array, (1, 2, 0))
    if isinstance(array, torch.Tensor):
        return torch.permute(array, (1, 2, 0))
    raise TypeError(f"Unsupported array type {type(array)}.")


def resize_chw(image: NDArray[Any], size: tuple[int, int]) -> NDArray[np.floating[Any]]:
    """Bilinearly resize a CHW image to ``(height, width)`` (IR-3.1-S-4)."""
    height, width = size
    tensor = torch.as_tensor(np.asarray(image)).float()
    if tensor.ndim != 3:
        raise ValueError(f"resize_chw expects CHW images; got shape {tuple(tensor.shape)}")
    resized = torch.nn.functional.interpolate(
        tensor.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
    )
    return resized.squeeze(0).numpy()
