"""Boolean masking utilities for per-detection attributes and metadata."""

__all__ = []

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import Array

T = TypeVar("T")


def try_mask_object(obj: T, mask: NDArray[np.bool_]) -> T:
    """Apply a boolean per-element mask to ``obj`` if it is a maskable sequence/array.

    Returns ``obj`` unchanged when it is not a per-element collection matching the
    mask length (e.g. a scalar or a string). Used to mask per-detection attributes
    (boxes, scores, labels) and metadata when dropping detections.
    """
    if not isinstance(obj, str | bytes | bytearray) and isinstance(obj, Sequence | Array) and len(obj) == len(mask):
        return obj[mask] if isinstance(obj, Array) else cast(T, [item for i, item in enumerate(obj) if mask[i]])
    return obj


def materialize_target_attrs(target: Any) -> dict[str, Any]:
    """Return a MAITE target's attributes as a plain dict for proxy ``__dict__`` materialization.

    Proxy targets (e.g. the masking wrappers in ClassFilter/Relabel) delegate reads through
    ``__getattribute__``, but ``@runtime_checkable`` protocol checks use ``getattr_static`` and
    bypass that, inspecting the instance ``__dict__`` directly. Copying these attrs in lets the
    proxy still satisfy ``isinstance(proxy, ObjectDetectionTarget)``. Handles both attribute-class
    targets (have ``__dict__``) and namedtuple targets (have ``_fields`` but no ``__dict__``).
    """
    source = getattr(target, "__dict__", None)
    if source:
        return dict(source)
    fields = getattr(target, "_fields", None)  # namedtuple (e.g. MAITE ObjectDetectionTargetTuple)
    if fields is not None:
        return {name: getattr(target, name) for name in fields}
    return {}


def mask_metadata(metadata: Mapping[str, Any], mask: NDArray[np.bool_]) -> dict[str, Any]:
    """Recursively mask per-detection arrays inside a datum metadata mapping.

    Walks nested dicts, applying :func:`try_mask_object` to each leaf so per-detection
    arrays (boxes, scores, ...) are filtered in step with a dropped-detection ``mask``.
    """
    return {k: mask_metadata(v, mask) if isinstance(v, dict) else try_mask_object(v, mask) for k, v in metadata.items()}


class MaskedTarget:
    """Proxy over a MAITE target that masks per-detection attributes by a boolean ``mask``.

    Each per-detection attribute read (boxes, scores, labels, ...) is filtered through
    :func:`try_mask_object`, dropping detections where ``mask`` is False. ``overrides``
    supplies replacement values for named attributes (e.g. relabeled ``labels``) that
    bypass masking.

    The target's attrs are surfaced in ``__dict__`` so ``@runtime_checkable`` protocol
    checks (which use ``getattr_static`` and bypass ``__getattribute__``) still see this
    proxy as the original target type; works for namedtuple targets too. Normal reads are
    still delegated/masked via ``__getattribute__`` below.
    """

    def __init__(self, target: Any, mask: NDArray[np.bool_], overrides: Mapping[str, Any] | None = None) -> None:
        self.__dict__.update(materialize_target_attrs(target))
        self._target = target
        self._mask = mask
        self._overrides = dict(overrides or {})

    def __getattribute__(self, name: str) -> Any:
        if name in ("_target", "_mask", "_overrides") or (name.startswith("__") and name.endswith("__")):
            return super().__getattribute__(name)
        if name in self._overrides:
            return self._overrides[name]
        return try_mask_object(getattr(self._target, name), self._mask)
