"""Populating a :class:`~dataeval.Metadata` from raw factor arrays, with no dataset.

The machinery behind :meth:`~dataeval.Metadata.from_factors`. Two ways in — arrays that
all describe one level, and arrays labelled by a :class:`~dataeval.types.SourceIndex` —
which differ only in how the rows are keyed and then converge on the same
``StructuredData`` bundle the dataset path produces, so the
reserved columns have exactly one producer.

Free functions taking the instance rather than methods on it: nothing here is polymorphic,
each one runs exactly once per construction, and keeping them out of the class keeps the
class's surface to what a caller can actually reach.
"""

__all__ = []

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from dataeval._metadata._columns import split_by_dimensionality
from dataeval._metadata._input import build_index2label, reject_length_mismatch
from dataeval._metadata._structurers import FactorsStructurer, SourceIndexRows
from dataeval.exceptions import ShapeMismatchError
from dataeval.types import Array1D, FactorLevel, SourceIndex
from dataeval.utils._internal import as_numpy

if TYPE_CHECKING:
    from dataeval._metadata._metadata import Metadata


def _load_factors(
    md: "Metadata",
    factors: Mapping[str, Array1D[Any]],
    class_labels: Array1D[Any] | None,
    *,
    index2label: Mapping[int, str] | None,
    item_indices: Array1D[Any] | None,
    level: FactorLevel | None = None,
    source_index: Sequence[SourceIndex] | None = None,
) -> None:
    """Populate structured state directly from raw factor arrays (see ``from_factors``)."""
    # Vector-valued statistics have no single-column form. Dropping them here rather
    # than letting a flatten silently produce a wrong-length column keeps a mapping
    # straight from compute_stats usable, and reports the same way add_factors reports
    # it. Column vectors keep working: _split_by_dimensionality flattens them.
    factor_arrays, skipped = split_by_dimensionality({str(k): v for k, v in factors.items()})

    if source_index is not None:
        _load_factors_by_source_index(
            md,
            factor_arrays,
            class_labels,
            index2label=index2label,
            level=level,
            item_indices=item_indices,
            source_index=source_index,
        )
    else:
        _load_factors_by_length(
            md,
            factor_arrays,
            class_labels,
            index2label=index2label,
            level=level,
            item_indices=item_indices,
        )
    # Recorded only once the structure exists, and only once every validation above has
    # passed: recording is a mutation, and a rejected call must leave no trace of itself
    # behind.
    md._record_multidimensional(skipped)


def _load_factors_by_length(  # noqa: C901
    md: "Metadata",
    factor_arrays: Mapping[str, NDArray[Any]],
    class_labels: Array1D[Any] | None,
    *,
    index2label: Mapping[int, str] | None,
    level: FactorLevel | None,
    item_indices: Array1D[Any] | None,
) -> None:
    """Populate structured state from factor arrays that all describe one level.

    Nothing in bare arrays distinguishes an item from a label, so every factor sits at
    the same level and the rows are numbered by position.

    Raises
    ------
    ShapeMismatchError
        When the factors, ``class_labels`` and ``item_indices`` do not agree on a
        single row count.
    """
    lengths = {len(v) for v in factor_arrays.values()}
    if len(lengths) > 1:
        raise ShapeMismatchError(f"All factor arrays must have the same length; got lengths {sorted(lengths)}.")
    factor_len = next(iter(lengths)) if factor_arrays else None

    if class_labels is not None:
        labels = as_numpy(class_labels, dtype=np.intp).reshape(-1)
        n = len(labels)
        if factor_len is not None and factor_len != n:
            raise ShapeMismatchError(f"class_labels length {n} does not match factor length {factor_len}.")
    elif factor_len is not None:
        n = factor_len
        labels = np.zeros(n, dtype=np.intp)
    else:
        n = 0
        labels = np.array([], dtype=np.intp)

    if item_indices is None:
        srcidx = np.arange(n, dtype=np.intp)
    else:
        srcidx = as_numpy(item_indices, dtype=np.intp).reshape(-1)
        if len(srcidx) != n:
            raise ShapeMismatchError(f"item_indices length {len(srcidx)} does not match row count {n}.")

    # A factors-only instance has a single level, which is therefore both the item
    # level and the label level. Structuring goes through the same StructuredData
    # bundle as the dataset path, so the reserved columns have exactly one producer
    # and cannot drift between the two constructors. No structurer instance exists yet
    # to resolve the level against — this call is what builds one — so it is declared
    # directly.
    requested = level or "unit"
    structurer = FactorsStructurer(requested)  # type: ignore[arg-type]
    data = structurer.build_from_arrays(factor_arrays, labels, srcidx)

    md._index2label = build_index2label(index2label, np.unique(labels))
    # Items, not rows, matching :attr:`item_count`'s contract and the source-index
    # path. ``item_indices`` exists so that several rows can share one item, and a
    # count of rows disagrees with it on exactly the tables it is for. It is also what
    # tells :func:`~dataeval.data.split_dataset` an object detection table from a
    # classification one — counted as rows, a table of detections never reaches the
    # grouped split and two detections of one image can land in different folds.
    md._count = int(len(np.unique(srcidx)))
    md._adopt(structurer, data)


def _load_factors_by_source_index(
    md: "Metadata",
    factor_arrays: Mapping[str, NDArray[Any]],
    class_labels: Array1D[Any] | None,
    *,
    index2label: Mapping[int, str] | None,
    level: FactorLevel | None,
    item_indices: Array1D[Any] | None,
    source_index: Sequence[SourceIndex],
) -> None:
    """Populate structured state from factor arrays labelled by a source index.

    The source index supplies what `level` and `item_indices` supply on the other
    path — which level each value belongs to and which item it came from — so all
    three together is a contradiction rather than a redundancy, and is rejected
    instead of one silently winning.

    Raises
    ------
    ValueError
        When `level` or `item_indices` is given alongside the source index, or when an
        address names a level this path cannot build — see
        :meth:`SourceIndexRows.reject_levels_beyond_two`.
    ShapeMismatchError
        When a factor does not have one value per source-index entry.
    """
    for name, value in (("level", level), ("item_indices", item_indices)):
        if value is not None:
            raise ValueError(
                f"`{name}` and `source_index` are mutually exclusive; the source index already "
                f"says which level each value sits at and which item it came from.",
            )

    reject_length_mismatch(factor_arrays, source_index)

    rows = SourceIndexRows.parse(source_index)
    rows.reject_levels_beyond_two()
    structurer = FactorsStructurer(rows=rows)
    labels = None if class_labels is None else as_numpy(class_labels, dtype=np.intp).reshape(-1)
    data = structurer.build_from_source_index(factor_arrays, labels)

    md._index2label = build_index2label(index2label, np.unique(data.class_labels))
    # Items, not rows: several labels can name the same item, and item_count that
    # counted rows would disagree with item_indices on the very datasets — one item,
    # several detections — this path exists to carry. Counted by adjacent change
    # rather than np.unique, which would re-sort what parse already left sorted.
    named_items = rows.item_ids if len(rows.item_positions) else rows.label_items
    md._count = int(np.count_nonzero(np.diff(named_items))) + 1 if len(named_items) else 0
    md._adopt(structurer, data)
