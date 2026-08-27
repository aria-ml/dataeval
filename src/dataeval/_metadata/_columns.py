"""How a factor's values become dataframe columns.

Two related jobs, kept together because both are answers to "which column holds this
factor, in which form": naming the companion column that binning writes, and deciding
which of a factor's columns a given reader should select. Neither knows anything about
levels, storage or the dataset — they are string and array functions, and are pure so
that a caller can resolve every column it intends to write before mutating anything.
"""

__all__ = []

from collections.abc import Mapping
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval.types import Array1D, FactorInfo, FactorLevel
from dataeval.utils._internal import as_numpy


def _flatten_column_vector(values: NDArray[np.generic]) -> NDArray[np.generic]:
    """Flatten an ``(N, 1)`` column of single values to ``(N,)``, leaving any other shape alone."""
    return values.reshape(-1) if values.ndim == 2 and values.shape[1] == 1 else values


def _holds_no_values(values: NDArray[np.generic]) -> bool:
    """Whether an array of a factor's values at one level holds no value at all.

    An empty array is not "no values" in this sense: a level that has no rows holds
    nothing for every factor alike, which says something about the dataset rather than
    about this factor.
    """
    if values.size == 0:
        return False
    if values.dtype.kind in "fc":
        return bool(np.isnan(values).all())
    if values.dtype.kind == "O":
        return all(value is None or (isinstance(value, float) and np.isnan(value)) for value in values)
    # Integer, boolean and fixed-width string arrays have no null to be made of.
    return False


def drop_vacuous_splits(
    columns: list[tuple[str, FactorLevel, NDArray[np.generic]]],
) -> tuple[list[tuple[str, FactorLevel, NDArray[np.generic]]], list[str]]:
    """Split one factor's level columns into those carrying values and those carrying none.

    A statistic that only applies to some of the levels a source index spans still has
    to be as long as the index to be placed at all, so it arrives padded with nulls at
    the levels it does not describe — ``compute_stats(per_background=True)`` measures the
    scene behind an image's annotations, which is a property of the image and of nothing
    inside it, and so has nulls on every instance row. Splitting that array by level
    yields one real column and one holding nothing. The empty one is not a factor: it
    cannot be binned, and left in place it reaches every evaluator that reads
    :attr:`~dataeval.Metadata.factor_names` — :class:`~dataeval.bias.Balance` reports it
    as a row of zero mutual information rather than omitting it.

    Only ever separates out a *part* of a split. A factor whose every level came back
    empty is kept whole, so that a factor a caller passed in can never vanish entirely —
    an all-null factor is then theirs to see in the frame rather than ours to hide.

    Pure, and returns the discarded names rather than recording them, so that
    :meth:`~dataeval.Metadata.add_factors` can still abandon the whole call on a later
    validation failure without having already written to
    :attr:`~dataeval.Metadata.dropped_factors`.
    """
    kept = [column for column in columns if not _holds_no_values(column[2])]
    if not kept:
        return columns, []
    return kept, [name for name, _, _ in columns if not any(name == kept_name for kept_name, _, _ in kept)]


def split_by_dimensionality(
    factors: Mapping[str, Array1D[Any]],
) -> tuple[dict[str, NDArray[Any]], list[str]]:
    """Separate the factors with a single-column form from those without one.

    A column vector has one: ``(N, 1)`` holds one value per row and is flattened to
    it. That is what a dataframe column or a ``reshape(-1, 1)`` pipeline produces, so
    rejecting it would drop real data over a shape carrying no extra information. Only
    an array that is genuinely several values per row — a histogram, a percentile
    vector, a centre coordinate — has nowhere to go.

    A *leading* singleton axis is left alone rather than flattened, and a 1-D array is
    passed through untouched however short it is. Both guard the same edge: on a
    one-row dataset ``(1, K)`` reads equally as one row of K values and as K rows of
    one value, and the first is what :func:`~dataeval.core.compute_stats` emits — so
    flattening would import ``center`` and ``histogram`` as K rows of data on the very
    dataset shape where they least resemble a real factor, while a blanket squeeze
    would additionally collapse a legitimate one-row ``mean`` to a scalar and drop it.

    Pure: it decides what will be dropped without recording it, so that
    :meth:`~dataeval.Metadata.add_factors` can still abandon the whole call on a later
    validation failure without having already written to
    :attr:`~dataeval.Metadata.dropped_factors`.
    """
    arrays = {name: _flatten_column_vector(as_numpy(values)) for name, values in factors.items()}
    kept = {name: values for name, values in arrays.items() if values.ndim == 1}
    return kept, [name for name in arrays if name not in kept]


# The two suffixes binning appends, and the namespace they define between them. Named
# rather than spelled inline because :func:`is_companion_name` has to answer for the
# same characters these build with, and a suffix that drifted between the two would
# reopen exactly the collision that function exists to close.
BINNED_SUFFIX = "↕"
DIGITIZED_SUFFIX = "#"
COMPANION_SUFFIXES: tuple[str, ...] = (BINNED_SUFFIX, DIGITIZED_SUFFIX)


def binned(name: str) -> str:
    """Name of the companion column holding ``name``'s bin indices."""
    return f"{name}{BINNED_SUFFIX}"


def digitized(name: str) -> str:
    """Name of the companion column holding ``name``'s category ordinals."""
    return f"{name}{DIGITIZED_SUFFIX}"


def is_companion_name(name: str) -> bool:
    """Whether ``name`` sits in the namespace binning writes its companion columns into.

    Every reader that resolves a companion does it by construction — ``binned(col)`` and
    ``digitized(col)`` over the columns actually present — so a *factor* holding one of
    those names is indistinguishable from the companion of its stem. A column named
    ``w#`` alongside a factor ``w`` makes ``Metadata._bin`` skip ``w`` as already binned,
    makes ``_reset_bins`` and the serializer's ``_without_companions`` drop the caller's
    values as derived, and leaves ``factor_names`` a name longer than ``factor_data`` is
    wide. Reserving the namespace is what keeps all three honest, so this is consulted
    wherever a factor is named — see ``safe_column_name`` in ``_structurers._reserved``.
    """
    return name.endswith(COMPANION_SUFFIXES)


def to_col(name: str, info: FactorInfo, is_binned: bool = True) -> str:
    """Column a reader of ``name`` should select, given what binning did to it."""
    if is_binned and info.is_binned:
        return binned(name)
    if info.is_digitized:
        return digitized(name)
    return name


def float_col(name: str, info: FactorInfo, schema: Mapping[str, pl.DataType]) -> str:
    """Column holding a float-castable representation of a factor.

    Raw values wherever they are numeric, so continuous factors keep their real
    values. A categorical factor's raw column holds strings, which have no float
    form at all, so it resolves to its digitized companion instead.
    """
    dtype = schema.get(name)
    return name if dtype is not None and dtype.is_numeric() else to_col(name, info, is_binned=False)
