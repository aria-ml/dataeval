"""Walking derived views down to the items they were derived from.

A derived view (frames of a video, crops of an image) holds many items that come from one item of its source. Each
such view exposes a private ``_source_items`` map from its items to its source's items; composing them down the
``.source`` chain gives every item its root item, the unit those items are not independent of. A merge exposes its
``_merged_parts`` instead, and each part is walked on its own, so a merge of views resolves as each view does.

Duck-typed on ``_source_items`` and ``_merged_parts``, so this package imports nothing from ``dataeval.data``.
"""

__all__ = []

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _hooked(dataset: Any) -> bool:
    """Whether `dataset` maps its items onward, looked up on the class as ``_structuring_source`` is."""
    kind = type(dataset)
    return getattr(kind, "_source_items", None) is not None or getattr(kind, "_merged_parts", None) is not None


def _root_index(roots: list[Any], dataset: Any) -> int:
    """Return `dataset`'s number among the roots met so far, matched by identity rather than equality."""
    for index, root in enumerate(roots):
        if root is dataset:
            return index
    roots.append(dataset)
    return len(roots) - 1


def _resolve(dataset: Any, positions: NDArray[np.intp], roots: list[Any]) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Resolve `positions` in `dataset` to (index into `roots`, position in that root) pairs.

    Roots are told apart by identity: two merged views built over one dataset object reach the same root, so their
    items from one video share it, while two separately loaded copies of the same data are two roots.
    """
    while getattr(type(dataset), "_source_items", None) is not None:
        positions = dataset._source_items[positions]
        dataset = dataset.source
    if getattr(type(dataset), "_merged_parts", None) is not None:
        return _resolve_parts(dataset._merged_parts, positions, roots)
    return np.full(len(positions), _root_index(roots, dataset), dtype=np.intp), positions


def _resolve_parts(
    parts: tuple[tuple[Any, int], ...], positions: NDArray[np.intp], roots: list[Any]
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Split merged `positions` among the parts they fall in, resolve each part on its own, and reassemble them.

    Split by the lengths the merge indexes by, so a part that has since changed length raises where reading it would,
    rather than leaving positions unassigned.
    """
    lengths = np.array([length for _, length in parts], dtype=np.intp)
    ends = np.cumsum(lengths)
    owner = np.searchsorted(ends, positions, side="right")
    which = np.empty(len(positions), dtype=np.intp)
    where = np.empty(len(positions), dtype=np.intp)
    for k, (part, _) in enumerate(parts):
        mine = owner == k
        if mine.any():
            which[mine], where[mine] = _resolve(part, positions[mine] - (ends[k] - lengths[k]), roots)
    return which, where


def root_items(dataset: Any) -> NDArray[np.intp] | None:
    """Return a group per item, shared by items with the same root item, or None when `dataset` is not derived.

    The walk stops at the first dataset with neither hook (a plain dataset, a stitch or a custom wrapper), whose items
    are the unit. With one root the groups are its item positions; with several, each root's positions are offset
    past the roots before it, so items share a group only when they reach the same item of the same root.
    """
    if not _hooked(dataset):
        return None
    roots: list[Any] = []
    which, where = _resolve(dataset, np.arange(len(dataset), dtype=np.intp), roots)
    offsets = np.concatenate(([0], np.cumsum([len(root) for root in roots[:-1]]))).astype(np.intp)
    groups = offsets[which] + where if len(roots) > 1 else where
    groups = groups.view()
    groups.flags.writeable = False
    return groups
