"""Dataset validation helpers: what shape a dataset is, and refusing the wrong one.

The dataset *operations* that once lived here — :func:`split_dataset`,
:func:`unzip_dataset`, :class:`TrainValSplit`, :class:`DatasetSplits` — moved to
:mod:`dataeval.data` in v1.1 and stopped being importable from here in v1.2.0. The
validation helpers below never moved and were never deprecated, so this module remains
their public home.
"""

__all__ = ["DatasetKind", "validate_dataset"]


import functools
import inspect
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Sized
from typing import Any, Literal, TypeAlias, TypeVar, cast, get_args

from numpy.typing import NDArray

from dataeval.exceptions import MaiteShapeError
from dataeval.protocols import (
    Array,
    ArrayLike,
    Dataset,
    MultiobjectTrackingTarget,
    ObjectDetectionTarget,
    SegmentationTarget,
    _is_protocol_instance,
)
from dataeval.types._target import detection_count
from dataeval.utils._array import as_numpy

DatasetKind = Literal[
    "image_only",
    "classification",
    "object_detection",
    "segmentation",
    "multiobject_tracking",
    "any_target",
]
"""Kind of MAITE dataset a consumer requires.

- ``"image_only"`` — only the image (``datum[0]``) is read; tuple or bare image both OK.
- ``"classification"`` — full 3-tuple; ``datum[1]`` is an :class:`Array` of class scores/logits.
- ``"object_detection"`` — full 3-tuple; ``datum[1]`` is an :obj:`ObjectDetectionTarget`.
- ``"segmentation"`` — full 3-tuple; ``datum[1]`` is a :class:`SegmentationTarget`.
- ``"multiobject_tracking"`` — full 3-tuple; ``datum[1]`` is a :obj:`MultiobjectTrackingTarget`.
- ``"any_target"`` — full 3-tuple; ``datum[1]`` matches *any* of the above.
"""

_KINDS: frozenset[str] = frozenset(get_args(DatasetKind))

_F = TypeVar("_F", bound=Callable[..., Any])


# One predicate per target-consuming kind, **most specific first**. The order is what
# resolves ``any_target`` to a concrete kind, so a target satisfying more than one is
# reported as the most specific of them.
_TARGET_CHECKS: Mapping[str, Callable[[Any], bool]] = {
    "multiobject_tracking": lambda target: _is_protocol_instance(target, MultiobjectTrackingTarget),
    "object_detection": lambda target: _is_protocol_instance(target, ObjectDetectionTarget),
    "segmentation": lambda target: _is_protocol_instance(target, SegmentationTarget),
    "classification": lambda target: isinstance(target, Array),
}


def _target_matches(target: Any, expected: DatasetKind) -> bool:
    if expected == "any_target":
        return any(check(target) for check in _TARGET_CHECKS.values())
    check = _TARGET_CHECKS.get(expected)
    # None only for "image_only", which is short-circuited before this is reached.
    return check is not None and check(target)


def _describe(value: Any) -> str:
    """Concise type description for error messages."""
    t = type(value).__name__
    if isinstance(value, tuple):
        return f"tuple of length {len(value)} ({', '.join(type(x).__name__ for x in value)})"
    if hasattr(value, "shape"):
        return f"{t} with shape {tuple(getattr(value, 'shape', ()))}"
    return t


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
   :obj:`~dataeval.protocols.ObjectDetectionTarget` for OD,
   :class:`~dataeval.protocols.SegmentationTarget` for segmentation,
   :obj:`~dataeval.protocols.MultiobjectTrackingTarget` for multi-object
   tracking, any of the four for ``"any_target"``).
4. For multi-object tracking (including ``"any_target"`` resolving to it): each
   ``frame_tracks`` entry is checked against the invariant dataeval's consumers
   depend on -- ``labels`` establish the detection count and ``boxes`` must be
   reshapable to ``(N, 4)`` in x0, y0, x1, y1 order with one row per label. This is
   the permissive reading of maite's ``Is[...]`` predicates, which are inert without
   beartype: ``scores`` and ``track_ids`` are optional (NaN / ``-1`` fallbacks), so a
   short or missing ``track_ids`` -- an untracked detection -- is legitimate.

   Targets only: nothing here decodes a frame, and the scan is the probed sequence's
   alone. It is O(its frame count) -- about 0.5 microseconds per frame, or 50ms for a
   100k-frame sequence, and roughly 3x that through a masking proxy whose getters
   re-filter on every read. Deliberately not sampled: a scan that skipped frame 50,000
   would let exactly the defect this module exists to intercept reach the walk instead.

Whether a video stream yields one frame per frame target is **not** checked here. That
costs a decode, and the structuring walk behind :class:`~dataeval.Metadata` and
:class:`~dataeval.data.SequenceFrames` already checks it in both directions for *every*
sequence, from frames it was going to decode anyway -- where a probe of ``dataset[0]``
could only ever have checked the first.

On failure, raises :class:`~dataeval.exceptions.MaiteShapeError` with a
message that names the calling function, what was expected, and what was
observed.
"""


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
        "segmentation" | "multiobject_tracking"``; for other inputs it
        echoes ``expected``.

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
            "multiobject_tracking": "a MultiobjectTrackingTarget (frame_tracks)",
            "any_target": ("an Array, ObjectDetectionTarget, SegmentationTarget, or MultiobjectTrackingTarget"),
        }[expected]
        raise MaiteShapeError(
            f"{where}: argument {arg_name!r} has dataset[0][1] of type {type(target).__name__}; "
            f"expected {kind_label} for {expected!r} consumers."
        )

    if expected == "any_target":
        # _target_matches has already established that one of them matches, so the search
        # cannot come up empty. Driven by the same ordered table, so "most specific wins"
        # is declared once rather than restated here.
        kind = cast(DatasetKind, next(kind for kind, check in _TARGET_CHECKS.items() if check(target)))
    else:
        kind = expected

    if kind == "multiobject_tracking":
        _check_mot_coverage(datum[1], where, arg_name)
    return kind


# ---------- multi-object tracking coverage ----------


def _unreadable_array(exc: Exception) -> str:
    """Describe a per-frame array that could not be read, by why it could not be.

    Every read here goes through the same ``as_numpy`` the consumers use, so anything this
    catches is a target the consumer could not have read either. Deliberately broad: task
    dispatch answers on member *presence* without evaluating it (see
    ``protocols._is_protocol_instance``), which leaves this the first place a property
    getter is actually called -- and the place whose whole job is to turn a target malformed
    for its task into a MaiteShapeError rather than let the getter's own exception escape.
    """
    if isinstance(exc, AttributeError):
        return f"lacks a required per-frame array ({exc})"
    return f"has a per-frame array that cannot be read as an array ({exc})"


def _mot_frame_problem(frame: Any) -> str | None:
    """Return a per-frame target defect, or ``None`` when the frame is well-formed.

    Enforces the invariant dataeval's consumers actually depend on, which is the permissive
    reading of maite's ``Is[...]`` predicates (inert without beartype) and must stay so:
    ``labels`` establish the detection count and every other per-detection array is read
    against them -- ``instance_arrays`` reshapes ``boxes`` to ``(N, 4)``, ``track_ids_of``
    pads a short ``track_ids`` with ``-1``, ``own_class_scores`` reads ``scores`` as
    optional. A frame whose boxes cannot be read as one row of 4 columns per label is where
    a detection's position stops meaning the same thing to every reader, so that is what is
    rejected at the probe; ``scores`` and ``track_ids`` of any length (or absent) are not.

    Reads through ``detection_count`` and ``as_numpy`` -- the same readers
    ``instance_arrays`` will use next -- rather than a private equivalent, so the validator
    cannot come to a different verdict about a target than the consumer it is clearing the
    way for. Boxes are read only *after* the count is known, because a detection-free frame
    is never asked for them.
    """
    try:
        count = detection_count(frame)
    except Exception as exc:  # noqa: BLE001 - see _unreadable_array
        return _unreadable_array(exc)
    if not count:
        return None  # a detection-free frame needs no boxes, and is not asked for them
    try:
        boxes = as_numpy(frame.boxes)
    except Exception as exc:  # noqa: BLE001 - see _unreadable_array
        return _unreadable_array(exc)
    return _boxes_problem(boxes, count)


def _boxes_problem(boxes: NDArray[Any], count: int) -> str | None:
    """Whether ``boxes`` reads as one (x0, y0, x1, y1) row per label.

    Both halves are needed. The size test is the case where ``instance_arrays``'
    ``reshape(count, 4)`` would raise deep in the walk; the width test is the case where it
    would *succeed* and read every coordinate into the wrong slot, which no later reader
    can detect. A flat ``(4N,)`` buffer is exempt from the width test -- it reshapes
    row-major with coordinates intact -- which is why ndim 1 is not required to be 4 wide.
    """
    if boxes.size != 4 * count:
        return (
            f"has boxes with {boxes.size} value(s) for {count} label(s); boxes must be "
            "reshapable to (N, 4) in x0, y0, x1, y1 order with one row per label"
        )
    if boxes.ndim > 1 and boxes.shape[-1] != 4:
        return (
            f"has boxes of shape {tuple(boxes.shape)} for {count} label(s); boxes must be "
            "(N, 4) in x0, y0, x1, y1 order with one row per label, not transposed"
        )
    return None


def _frame_tracks(target: Any, where: str, arg_name: str) -> Sequence[Any]:
    """Target's per-frame targets, as a concrete sequence.

    The dispatch-level protocol check answers True on ``frame_tracks`` *presence* alone, so a
    value that is unreadable or not sequence-like reaches here; that is a shape defect for a
    tracking consumer and is reported rather than escaped as an AttributeError.
    """
    try:
        tracks = target.frame_tracks
    except AttributeError as exc:
        raise MaiteShapeError(
            f"{where}: argument {arg_name!r} has a MultiobjectTrackingTarget whose frame_tracks "
            f"is not readable ({exc})."
        ) from exc
    if not isinstance(tracks, Sequence):
        try:
            tracks = list(tracks)
        except TypeError:
            raise MaiteShapeError(
                f"{where}: argument {arg_name!r} has a MultiobjectTrackingTarget whose frame_tracks "
                f"is {_describe(tracks)}; expected a sequence of per-frame targets."
            ) from None
    return tracks


def _check_mot_coverage(target: Any, where: str, arg_name: str) -> None:
    """Entry-time MOT coverage check: every frame's per-detection arrays agree on a count.

    Targets only -- nothing here touches the stream, so no frame is decoded and a caller's
    stream is left exactly as it was found. The stream/target frame count is the walk's to
    check, not the probe's: the walk sees every sequence and already has the frames in hand,
    where a probe of ``dataset[0]`` would pay a full decode to check one.
    """
    tracks = _frame_tracks(target, where, arg_name)
    for i, frame in enumerate(tracks):
        problem = _mot_frame_problem(frame)
        if problem is not None:
            raise MaiteShapeError(
                f"{where}: argument {arg_name!r} has a multi-object tracking target whose frame {i} {problem}."
            )


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


AnyDatum: TypeAlias = ArrayLike | tuple[ArrayLike, Any, Any]


def unwrap_image(item: AnyDatum) -> ArrayLike:
    """Return ``item[0]`` if ``item`` is a MAITE-style ``(image, target, metadata)`` tuple, else ``item``.

    The first element of a MAITE tuple is the image by convention; remaining elements
    (target, metadata) are not type-checked here.
    """
    return item[0] if isinstance(item, tuple) else item


def iter_images(data: Iterable[AnyDatum] | Dataset[AnyDatum]) -> Iterator[ArrayLike]:
    """Yield images from an iterable or :class:`Dataset`, unwrapping MAITE-style tuples.

    MAITE datasets return ``(image, target, metadata)`` tuples per index; iterables of
    bare images pass through. Use in feature extractors so they accept both shapes
    uniformly.
    """
    for item in data:
        yield unwrap_image(item)
