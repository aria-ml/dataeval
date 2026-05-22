"""MAITE dataset shape validation.

Public entry points that accept a :class:`~dataeval.protocols.AnnotatedDataset`
should fail fast when the dataset's datum shape does not match what they will
consume. This module provides both a callable helper (:func:`validate_dataset`)
and a decorator (:func:`requires_maite_dataset`) that wraps an ``__init__`` /
classmethod / method and validates a named dataset argument before the body
runs.

The validation probes ``dataset[0]`` (or the first item from an iterator-style
dataset) and checks:

1. The dataset is :class:`~collections.abc.Sized` and indexable.
2. For ``"image_only"`` kind: the datum is either a 3-tuple
   ``(image, target, metadata)`` *or* a bare image-like value
   (matches today's :func:`unwrap_image` behavior).
3. For target-consuming kinds: the datum is a length-3 tuple AND
   ``datum[1]`` satisfies the protocol the consumer requires
   (:class:`~dataeval.protocols.Array` for classification,
   :class:`~dataeval.protocols.ObjectDetectionTarget` for OD,
   :class:`~dataeval.protocols.SegmentationTarget` for segmentation,
   any of the three for ``"any_target"``).

On failure, raises :class:`~dataeval.exceptions.MaiteShapeError` with a
message that names the calling function, what was expected, and what was
observed.
"""

__all__ = ["DatasetKind", "requires_maite_dataset", "validate_dataset"]

import functools
import inspect
from collections.abc import Callable, Sized
from typing import Any, Literal, TypeVar, cast, get_args

from dataeval.exceptions import MaiteShapeError
from dataeval.protocols import (
    Array,
    Dataset,
    ObjectDetectionTarget,
    SegmentationTarget,
)

DatasetKind = Literal[
    "image_only",
    "classification",
    "object_detection",
    "segmentation",
    "any_target",
]
"""Kind of MAITE dataset a consumer requires.

- ``"image_only"`` — only the image (``datum[0]``) is read; tuple or bare image both OK.
- ``"classification"`` — full 3-tuple; ``datum[1]`` is an :class:`Array` of class scores/logits.
- ``"object_detection"`` — full 3-tuple; ``datum[1]`` is an :class:`ObjectDetectionTarget`.
- ``"segmentation"`` — full 3-tuple; ``datum[1]`` is a :class:`SegmentationTarget`.
- ``"any_target"`` — full 3-tuple; ``datum[1]`` matches *any* of the above.
"""

_KINDS: frozenset[str] = frozenset(get_args(DatasetKind))

_F = TypeVar("_F", bound=Callable[..., Any])


def _target_matches(target: Any, expected: DatasetKind) -> bool:
    if expected == "classification":
        return isinstance(target, Array)
    if expected == "object_detection":
        return isinstance(target, ObjectDetectionTarget)
    if expected == "segmentation":
        return isinstance(target, SegmentationTarget)
    if expected == "any_target":
        return isinstance(target, Array | ObjectDetectionTarget | SegmentationTarget)
    return False  # unreachable for valid kinds; "image_only" is short-circuited earlier


def _describe(value: Any) -> str:
    """Concise type description for error messages."""
    t = type(value).__name__
    if isinstance(value, tuple):
        return f"tuple of length {len(value)} ({', '.join(type(x).__name__ for x in value)})"
    if hasattr(value, "shape"):
        return f"{t} with shape {tuple(getattr(value, 'shape', ()))}"
    return t


def validate_dataset(  # noqa: C901
    dataset: Any,
    *,
    expected: DatasetKind = "any_target",
    arg_name: str = "dataset",
    caller: str | None = None,
) -> DatasetKind:
    """Validate that a dataset matches the expected MAITE datum shape.

    Parameters
    ----------
    dataset : Any
        The object passed in as the dataset. Must be :class:`Sized` and
        indexable by integer.
    expected : DatasetKind, default ``"any_target"``
        The shape the caller intends to consume. See :data:`DatasetKind`.
    arg_name : str, default ``"dataset"``
        The parameter name to use in error messages.
    caller : str, optional
        Name of the calling function/class — included in error messages
        for easier debugging. When ``None``, the caller is inferred from
        the stack.

    Returns
    -------
    DatasetKind
        The inferred concrete kind. For ``expected == "any_target"`` this
        will be one of ``"classification" | "object_detection" |
        "segmentation"``; for other inputs it echoes ``expected``.

    Raises
    ------
    MaiteShapeError
        If the dataset is empty, non-indexable, or its datum shape does
        not satisfy ``expected``.
    """
    if expected not in _KINDS:
        raise ValueError(f"validate_dataset: unknown expected={expected!r}. Must be one of {sorted(_KINDS)}.")

    where = caller or _infer_caller()

    if not isinstance(dataset, Sized):
        raise MaiteShapeError(
            f"{where}: argument {arg_name!r} is not Sized (has no __len__); got {type(dataset).__name__}."
        )

    if len(dataset) == 0:
        # Empty datasets are legal (e.g. fully filtered) — nothing to probe, nothing to reject.
        return "image_only" if expected == "any_target" else expected

    if not isinstance(dataset, Dataset):
        raise MaiteShapeError(f"{where}: argument {arg_name!r} is not a Dataset; got {type(dataset).__name__}.")

    datum = dataset[0]

    if expected == "image_only":
        # Image-only consumers accept either a bare value (treated as the image) or a
        # MAITE (image, target, metadata) tuple. Per the chosen probe depth we verify
        # tuple arity only — the image type itself is unwrapped by downstream code.
        if isinstance(datum, tuple) and len(datum) != 3:
            raise MaiteShapeError(
                f"{where}: argument {arg_name!r} returned a tuple of length {len(datum)} from dataset[0]; "
                f"expected either a bare image or a MAITE 3-tuple (image, target, metadata)."
            )
        return "image_only"

    # Target-consuming kinds: must be a 3-tuple.
    if not isinstance(datum, tuple) or len(datum) != 3:
        raise MaiteShapeError(
            f"{where}: argument {arg_name!r} requires a MAITE-protocol dataset "
            f"whose dataset[0] returns a 3-tuple (image, target, metadata); "
            f"got {_describe(datum)}. "
            f"If you only have images, wrap them so each item is (image, target, metadata)."
        )

    target = datum[1]
    if not _target_matches(target, expected):
        kind_label = {
            "classification": "an Array of class scores/logits",
            "object_detection": "an ObjectDetectionTarget (boxes/labels/scores)",
            "segmentation": "a SegmentationTarget (mask/labels/scores)",
            "any_target": "an Array, ObjectDetectionTarget, or SegmentationTarget",
        }[expected]
        raise MaiteShapeError(
            f"{where}: argument {arg_name!r} has dataset[0][1] of type {type(target).__name__}; "
            f"expected {kind_label} for {expected!r} consumers."
        )

    if expected == "any_target":
        if isinstance(target, ObjectDetectionTarget):
            return "object_detection"
        if isinstance(target, SegmentationTarget):
            return "segmentation"
        return "classification"
    return expected


def requires_maite_dataset(  # noqa: C901
    arg_name: str = "dataset",
    *,
    expected: DatasetKind = "any_target",
) -> Callable[[_F], _F]:
    """Validate a named dataset argument before the wrapped call runs (decorator).

    Resolves the dataset argument by name (works for positional or keyword
    passing) and calls :func:`validate_dataset`. Compatible with regular
    functions, methods (including ``__init__``), and ``classmethod``-wrapped
    constructors like :meth:`Embeddings.new`.

    Parameters
    ----------
    arg_name : str, default ``"dataset"``
        Name of the dataset parameter on the wrapped callable.
    expected : DatasetKind, default ``"any_target"``
        Forwarded to :func:`validate_dataset`.

    Examples
    --------
    >>> @requires_maite_dataset("dataset", expected="image_only")
    ... def fit(self, dataset): ...

    >>> @requires_maite_dataset(expected="object_detection")
    ... def evaluate(self, dataset): ...
    """

    def decorator(func: _F) -> _F:
        sig = inspect.signature(func)
        if arg_name not in sig.parameters:
            raise TypeError(f"requires_maite_dataset: {func.__qualname__} has no parameter named {arg_name!r}.")
        qualname = func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                bound = sig.bind_partial(*args, **kwargs)
            except TypeError:
                return func(*args, **kwargs)  # let the real call surface the error
            dataset = bound.arguments.get(arg_name, inspect.Parameter.empty)
            if dataset is not inspect.Parameter.empty and dataset is not None:
                validate_dataset(dataset, expected=expected, arg_name=arg_name, caller=qualname)
            return func(*args, **kwargs)

        return cast(_F, wrapper)

    return decorator


def _infer_caller() -> str:
    """Best-effort name of the function/method that called validate_dataset."""
    frame = inspect.currentframe()
    try:
        # 0: _infer_caller, 1: validate_dataset, 2: actual caller
        outer = frame.f_back.f_back if frame and frame.f_back else None  # type: ignore[union-attr]
        if outer is None:
            return "validate_dataset"
        return getattr(outer.f_code, "co_qualname", outer.f_code.co_name)
    finally:
        del frame
