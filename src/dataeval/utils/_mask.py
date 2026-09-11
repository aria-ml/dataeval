"""Boolean masking utilities for per-detection attributes and metadata."""

__all__ = []

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import get_protocol_members

from dataeval.protocols import Array, ObjectDetectionTarget, SegmentationTarget

T = TypeVar("T")

_MASKABLE_TARGET_MEMBERS = get_protocol_members(ObjectDetectionTarget) | get_protocol_members(SegmentationTarget)


def try_mask_object(obj: T, mask: NDArray[np.bool_]) -> T:
    """Mask ``obj`` by ``mask`` if it is a sequence/array of matching length; otherwise return it unchanged.

    Used for per-detection attributes (boxes, scores, labels) and metadata when dropping detections.
    """
    if not isinstance(obj, str | bytes | bytearray) and isinstance(obj, Sequence | Array) and len(obj) == len(mask):
        return obj[mask] if isinstance(obj, Array) else cast(T, [item for i, item in enumerate(obj) if mask[i]])
    return obj


def materialize_target_attrs(target: Any) -> dict[str, Any]:
    """Collect a MAITE target's attributes into a plain dict for ``MaskedTarget.__dict__``.

    ``@runtime_checkable`` protocol checks use ``getattr_static``, which reads the instance
    ``__dict__`` directly and bypasses ``MaskedTarget.__getattribute__``. Copying attrs into
    ``__dict__`` here lets the proxy still satisfy
    ``isinstance(proxy, ObjectDetectionTarget | SegmentationTarget)``.

    Covers plain-attribute targets (via ``__dict__``), namedtuples (via ``_fields``), and
    ``@property``-backed targets. A property's value is a class descriptor, not present in
    the instance ``__dict__``, and is often shadowed there by a differently named private
    attribute (e.g. ``_boxes``). Those are read explicitly through the class instead.
    """
    source = getattr(target, "__dict__", None)
    attrs = dict(source) if source else {}
    fields = getattr(target, "_fields", None)  # namedtuple (e.g. MAITE ObjectDetectionTargetTuple)
    if fields is not None:
        attrs.update((name, getattr(target, name)) for name in fields)
    # hasattr on the class never invokes a property getter. It returns the
    # descriptor itself so this check is safe to run before reading the instance.
    for name in _MASKABLE_TARGET_MEMBERS:
        if name not in attrs and hasattr(type(target), name):
            attrs[name] = getattr(target, name)
    return attrs


def mask_metadata(metadata: Mapping[str, Any], mask: NDArray[np.bool_]) -> dict[str, Any]:
    """Apply :func:`try_mask_object` to every leaf of a nested metadata mapping."""
    return {k: mask_metadata(v, mask) if isinstance(v, dict) else try_mask_object(v, mask) for k, v in metadata.items()}


class MaskedTarget:
    """Proxy over a MAITE target that masks per-detection attributes by a boolean array.

    Each attribute read is filtered through :func:`try_mask_object`, dropping detections
    where ``mask`` is False. ``overrides`` supplies replacement values (e.g. relabeled
    ``labels``) that bypass masking.

    ``__init__`` copies the target's protocol-relevant attributes into ``__dict__`` so a
    ``getattr_static``-based ``isinstance`` check still recognizes the proxy as the wrapped
    target's type. Real attribute reads go through ``__getattribute__``, not this copy.
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
