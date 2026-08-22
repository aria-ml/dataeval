"""The reserved column schema: which columns are structure rather than observation.

A dataframe row carries two kinds of column. Factors are observations — anything the
dataset or the caller measured — and are binned, correlated and reported on. The
reserved columns are the row's own identity: the level it belongs to, the item it came
from, and where it sits within each of its parents. A factor whose name would collide
with one of them is renamed rather than allowed to overwrite it, and so is one that
would be taken for a companion column binning writes — see :func:`safe_column_name`.

Sole producer of that layout: every structurer and
:meth:`~dataeval.Metadata.from_factors` builds its blocks through
:func:`reserved_block_columns`, so the schema cannot drift between the dataset path and
the raw-factor path.
"""

__all__ = []

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval._metadata._columns import is_companion_name
from dataeval.types import FactorLevel

# Columns the metadata dataframe has always carried. Retained verbatim because
# they are public surface: downstream code reads ``dataframe["item_index"]``.
LEGACY_COLUMNS: tuple[str, ...] = ("item_index", "target_index", "class_label", "score", "box")

# Columns introduced alongside the expanded level schema. ``level`` tags each row
# with the level it belongs to and replaces the old "target_index is null" test;
# the remaining columns carry the components of a level's compound key. Only
# columns a structurer actually writes belong here — a name reserved for a level
# that does not exist yet costs a user their metadata key for nothing.
#
# ``unit_index``, ``track_index`` and ``sequence_index`` exist for multi-object tracking,
# the only task whose item level is not ``unit``: there a frame's position within its
# sequence is not ``item_index`` (that identifies the video) and has nowhere else to go.
# ``unit_index`` and ``track_index`` are written on the tracking task's instance rows
# too, naming the frame a detection was observed in and the track it belongs to — without
# ``unit_index``, ``(item_index, instance_index)`` repeats across the frames of one
# sequence. ``track_index`` is ``-1`` on a detection no tracker linked. A task whose
# items are single units has no use for any of them, and writes none.
LEVEL_COLUMNS: tuple[str, ...] = (
    "level",
    "instance_index",
    "unit_index",
    "track_index",
    "sequence_index",
)

# Identifiers only some tasks produce. ``track_id`` names the track a tracking detection
# belongs to, ``-1`` when the detection is untracked.
#
# A reserved column rather than a factor, because it is an identifier: as a factor it
# would be binned and handed to bias and diversity analysis as though a track number
# were an observed property of the data, which is exactly what ``item_index`` and
# ``target_index`` are reserved to prevent. It stays fully queryable —
# ``rows_at("instance")["track_id"]`` — and this is also the column a future ``track``
# level would key its rows on, so nothing has to move when tracks become rows.
IDENTIFIER_COLUMNS: tuple[str, ...] = ("track_id",)

# Factor names colliding with any of these are prefixed with ``metadata_``. This is
# wider than the historical five-column set: the level key columns are just as
# load-bearing, so a metadata key named ``level`` or ``instance_index`` is renamed
# rather than allowed to clobber the column. :data:`LEGACY_COLUMNS` still holds the
# original tuple for callers that need it.
RESERVED_COLUMNS: tuple[str, ...] = LEGACY_COLUMNS + LEVEL_COLUMNS + IDENTIFIER_COLUMNS

# Reserved columns a block emits only when it actually has a value for them: the
# level-key columns, i.e. everything in LEVEL_COLUMNS that is not the level tag itself,
# plus the optional identifiers. A block carries the key components of its own level and
# of any ancestor its ``item_index`` does not already identify, and no others.
_OPTIONAL_COLUMNS: tuple[str, ...] = (
    *(name for name in LEVEL_COLUMNS if name != "level"),
    *IDENTIFIER_COLUMNS,
)


def reserved_block_columns(level: FactorLevel, size: int, **values: Any) -> dict[str, Sequence[Any] | NDArray[Any]]:
    """Build the reserved (non-factor) columns of a single row block.

    Sole producer of the reserved column layout: every structurer and
    :meth:`~dataeval.Metadata.from_factors` routes through here, so the schema
    cannot drift between the dataset path and the raw-factor path.

    Parameters
    ----------
    level : str
        Level tag written to every row of the block.
    size : int
        Number of rows in the block.
    **values : Any
        Values for individual reserved columns, as a sequence of length ``size``.
        Every column in :data:`LEGACY_COLUMNS` is emitted whether supplied or not,
        filled with null when omitted; the level-key columns of
        :data:`LEVEL_COLUMNS` and the identifiers of :data:`IDENTIFIER_COLUMNS` are
        emitted only when supplied, since a block carries its own level's key
        components and only the identifiers its task produces.

    Returns
    -------
    dict[str, Sequence[Any] | NDArray[Any]]
        Column name to values, ready to hand to :class:`RowBlock`. Array-valued
        columns stay arrays; see :func:`_as_column`.

    Raises
    ------
    ValueError
        When a supplied name is not a reserved column, or when a supplied value
        sequence is not ``size`` long.
    """
    unknown = sorted(set(values) - set(RESERVED_COLUMNS))
    if unknown:
        raise ValueError(f"Column(s) {unknown} are not reserved columns {list(RESERVED_COLUMNS)}.")

    columns: dict[str, Sequence[Any] | NDArray[Any]] = {"level": [level] * size}
    for name in LEGACY_COLUMNS:
        supplied = values.get(name)
        columns[name] = [None] * size if supplied is None else _as_column(supplied)
    columns.update({name: _as_column(values[name]) for name in _OPTIONAL_COLUMNS if values.get(name) is not None})
    _reject_ragged(level, size, columns)
    return columns


def _reject_ragged(level: FactorLevel, size: int, columns: Mapping[str, Sequence[Any] | NDArray[Any]]) -> None:
    """Reject a block whose columns disagree on how many rows it has.

    Ragged columns would otherwise surface as an opaque polars ``ShapeError`` at
    DataFrame construction, long after the structurer that produced them.
    """
    ragged = {name: len(column) for name, column in columns.items() if len(column) != size}
    if ragged:
        raise ValueError(
            f"Every column of a {level!r} block must have {size} values, one per row; got {ragged}.",
        )


def _as_column(values: Any) -> Sequence[Any] | NDArray[Any]:
    """Normalize a column's values to something polars can consume directly.

    An ndarray is passed through untouched. Polars builds a Series from a contiguous
    numeric array without copying it, whereas ``.tolist()`` first materializes one
    Python object per element — on the reference tracking workload that is roughly
    13.5M objects for the instance block's reserved columns alone, all built before
    polars sees any data. Anything else becomes a list, which is both what polars wants
    and what :func:`_reject_ragged` needs :func:`len` to work on.
    """
    return values if isinstance(values, np.ndarray) else list(values)


def safe_column_name(name: str) -> str:
    """Rename a factor that would be taken for a column DataEval owns.

    Two namespaces are reserved, and a factor is moved out of either rather than allowed
    to occupy it. :data:`RESERVED_COLUMNS` is the row's own identity, collided with head-on
    and escaped by prefix. The companion namespace — anything ending in one of
    ``COMPANION_SUFFIXES`` — is the one binning writes into, so a name lands in it by its
    *tail* and has to be escaped there; see ``is_companion_name`` in ``_metadata._columns``
    for what mistaking the two costs.

    Sole entry point for both: every factor name reaches a frame through here, whether it
    came from a dataset's metadata dictionaries, from
    :meth:`~dataeval.Metadata.from_factors`, or from
    :meth:`~dataeval.Metadata.add_factors` and :meth:`~dataeval.Metadata.agg` by way of
    ``_resolve_factor_name``. Placed here rather than in each caller because the check has
    to hold before anything is binned: a writer that resolves its name against the columns
    currently present cannot see a companion binning has not written yet.

    Parameters
    ----------
    name : str
        Factor name as supplied by the dataset or the caller.

    Returns
    -------
    str
        ``name`` unchanged; ``metadata_<name>`` when it is in :data:`RESERVED_COLUMNS`;
        or ``<name>_metadata`` when it ends in a companion suffix.
    """
    if name in RESERVED_COLUMNS:
        return f"metadata_{name}"
    return f"{name}_metadata" if is_companion_name(name) else name
