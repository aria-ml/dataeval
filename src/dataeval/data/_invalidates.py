"""Collect the statistics that a dataset's view operations have invalidated.

A view operation that rewrites image content makes some statistics describe the
transform rather than the source data — a resize makes ``width``/``height``/
``aspect_ratio`` report the resize target and makes ``sharpness`` measure the
interpolation kernel. Operations declare that as
:attr:`~dataeval.data.Operation.invalidates`; this module walks a dataset's operation
and wrapping chain and unions those declarations so a consumer can warn on the overlap
with the statistics it was asked to compute.
"""

__all__ = []

from typing import Any

from dataeval.flags import ImageStats

# NOTE: never import from dataeval.core in this module. core imports data
# (core/_compute_stats.py) and data imports core (data/_classbalance.py); that cycle
# holds only by import ordering. ImageStats comes from dataeval.flags, which is
# standalone, so this module stays importable from either side.


def _node_sources(node: Any) -> list[tuple[str, ImageStats]]:
    """Return the invalidation declared by a single link in a wrapping chain.

    A :class:`~dataeval.data.View`'s own ``invalidates`` attribute is deliberately not
    read: :meth:`View.__init__` copies the source's instance ``__dict__`` onto the view,
    so an instance-level declaration on a wrapped dataset would be misattributed to the
    wrapping view. A view speaks only through its operations.
    """
    from dataeval.data._view import View

    if isinstance(node, View):
        return [(repr(op), op.invalidates) for op in node._operations if op.invalidates]
    flags = getattr(node, "invalidates", ImageStats.NONE)
    return [(type(node).__name__, flags)] if flags else []


def invalidating_sources(dataset: Any) -> list[tuple[str, ImageStats]]:
    """Walk a dataset's operation and wrapping chain, collecting each invalidator.

    Parameters
    ----------
    dataset : Any
        Any dataset. A plain dataset, a bare sequence of images, or a
        :class:`~dataeval.data.View` with no invalidating operation yields ``[]``.

    Returns
    -------
    list[tuple[str, ImageStats]]
        ``(label, invalidated_stats)`` per invalidating source, outermost first. The
        label names the operation (``repr``, e.g. ``Resize(size=(224, 224))``) or, for
        an invalidating wrapper dataset, its class name.

    Notes
    -----
    The walk descends ``_dataset`` rather than reading
    :attr:`~dataeval.data.View.operation_groups` alone. ``operation_groups`` stops at the
    first non-:class:`~dataeval.data.View` in the chain, so an ``operation_groups``-only
    implementation silently misses the inner operations of
    ``View(DetectionCrops(View(base, [Resize()])))``. Descending also revisits operations
    that ``operation_groups`` already spans, which double-counts — harmless, because the
    caller only unions with ``|``.
    """
    sources: list[tuple[str, ImageStats]] = []
    current: Any = dataset
    while current is not None:
        sources.extend(_node_sources(current))
        current = getattr(current, "_dataset", None)
    return sources


def invalidated_stats(dataset: Any) -> ImageStats:
    """Return the union of statistics invalidated by a dataset's operation chain.

    Parameters
    ----------
    dataset : Any
        Any dataset; see :func:`invalidating_sources`.

    Returns
    -------
    ImageStats
        The OR of every invalidating operation's declaration, or ``ImageStats.NONE``
        when nothing in the chain rewrites image content.

    Examples
    --------
    >>> from dataeval.data._invalidates import invalidated_stats
    >>> from dataeval.flags import ImageStats
    >>> bool(invalidated_stats(dataset) & ImageStats.DIMENSION_WIDTH)
    False
    >>> bool(invalidated_stats(cropped_dataset) & ImageStats.DIMENSION_WIDTH)
    True
    """
    result = ImageStats.NONE
    for _, flags in invalidating_sources(dataset):
        result |= flags
    return result
