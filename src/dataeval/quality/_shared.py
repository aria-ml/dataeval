"""Shared utilities for quality evaluators (duplicates and outliers)."""

__all__ = []

import warnings
from collections.abc import Sequence
from typing import Any

import polars as pl

from dataeval.core import StatsResult, compute_stats
from dataeval.data._invalidates import invalidating_sources
from dataeval.exceptions import StatsInvalidatedWarning
from dataeval.flags import ImageStats
from dataeval.types import FactorLevel, SourceIndex

# What :attr:`SourceIndex.kind` reports for a row that is one of an item's labels. An
# unstated level under a key resolves here on every task, so the evaluators can name the
# label end without a dataset to resolve against.
LABEL_KIND = "instance"


def _stat_names(stats: ImageStats) -> str:
    """Render a flag as the stat column names it produces, in declaration order.

    Every ``ImageStats`` member is named ``<CATEGORY>_<STAT>``, and ``<STAT>`` lowercased
    is the calculator's output name (``DIMENSION_ASPECT_RATIO`` -> ``aspect_ratio``), so
    the mapping is derived rather than duplicated from ``dataeval.core._calculators``.

    Walks the class's members rather than iterating ``stats`` itself: iterating a ``Flag``
    *value* to yield its constituent members only exists on Python 3.11+, and this package
    supports 3.10. Composite members (``PIXEL``, ``DIMENSION``, ...) are skipped by the
    power-of-two test so only individual stat columns are named.
    """
    return ", ".join(
        name.split("_", 1)[1].lower()
        for name, member in ImageStats.__members__.items()
        if name == member.name and member.value and not member.value & (member.value - 1) and member & stats
    )


def checked_compute_stats(
    datasets: Sequence[Any],
    *,
    stats: ImageStats,
    caller: str,
    **kwargs: Any,
) -> list[StatsResult]:
    """Compute stats per dataset, warning first about any that a view operation invalidated.

    Parameters
    ----------
    datasets : Sequence[Any]
        The datasets to compute over, always a sequence — ``Dataset | Sequence[Dataset]``
        is not discriminable at runtime, since a ``Dataset`` is itself sized and
        indexable. Single-dataset callers pass ``[data]`` and take ``[0]``.
    stats : ImageStats
        The statistics to compute. This must be the *effective* flags — what actually
        reaches :func:`~dataeval.core.compute_stats` — not what the evaluator was
        configured with. ``Duplicates`` computes ``flags & ImageStats.HASH``, so
        intersecting the invalidation against ``self.flags`` would warn about dimension
        stats it never computes.
    caller : str
        Name of the evaluator, used in the warning message.
    **kwargs : Any
        Forwarded to :func:`~dataeval.core.compute_stats` unchanged.

    Returns
    -------
    list[StatsResult]
        One result per dataset, in order.

    Warns
    -----
    StatsInvalidatedWarning
        Once per invalidating operation whose declaration overlaps ``stats``, unioned
        across ``datasets`` so N datasets sharing operations warn once, not N times.
    """
    invalidated: dict[str, ImageStats] = {}
    for dataset in datasets:
        for label, flags in invalidating_sources(dataset):
            invalidated[label] = invalidated.get(label, ImageStats.NONE) | flags

    for label, flags in invalidated.items():
        overlap = flags & stats
        if not overlap:
            continue
        warnings.warn(
            f"{label} invalidates statistics requested by {caller}: {_stat_names(overlap)}. "
            "These now describe the transform rather than the source data. "
            "If this is model preprocessing, move it to the extractor's transforms=; "
            "otherwise pass flags= to exclude them.",
            StatsInvalidatedWarning,
            # 1: here, 2: the evaluator's _evaluate_single/_evaluate_multi,
            # 3: the evaluator's evaluate(), 4: the set_metadata wrapper,
            # 5: the user's evaluate() call.
            stacklevel=5,
        )

    return [compute_stats(ds, stats=stats, **kwargs) for ds in datasets]


def get_dataset_step_from_idx(idx: int, dataset_steps: Sequence[int]) -> tuple[int, int]:
    """Map a global index to (dataset_index, local_index) using cumulative dataset_steps.

    Parameters
    ----------
    idx : int
        Global index in the combined array.
    dataset_steps : Sequence[int]
        Cumulative boundaries where each dataset ends.

    Returns
    -------
    tuple[int, int]
        (dataset_index, local_index) within that dataset.
        Returns (-1, idx) if the index is out of bounds.
    """
    last_step = 0
    for i, step in enumerate(dataset_steps):
        if idx < step:
            return i, idx - last_step
        last_step = step
    return -1, idx


def add_dataset_index(  # noqa: C901
    df: pl.DataFrame,
    dataset_steps: Sequence[int],
) -> pl.DataFrame:
    """Add a dataset_index column and remap item_index to local per-dataset indices.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with an ``item_index`` column containing global indices.
    dataset_steps : Sequence[int]
        Cumulative boundaries from :func:`combine_calculation_results`.

    Returns
    -------
    pl.DataFrame
        DataFrame with ``dataset_index`` prepended and ``item_index`` remapped
        to local per-dataset values, sorted by dataset_index then item_index.
    """
    if not dataset_steps:
        return df

    if df.shape[0] == 0:
        return pl.DataFrame(schema={"dataset_index": pl.Int64, **dict(df.schema)})

    dataset_indices: list[int] = []
    local_item_indices: list[int] = []
    for row in df.iter_rows(named=True):
        ds_idx, local_idx = get_dataset_step_from_idx(row["item_index"], dataset_steps)
        dataset_indices.append(ds_idx)
        local_item_indices.append(local_idx)

    existing_cols = [c for c in df.columns if c != "item_index"]
    sort_cols = ["dataset_index", "item_index"]
    if "target_index" in df.columns:
        sort_cols.append("target_index")
    # Part of a row's identity rather than a description of it, so two rows differing only
    # in level are ordered rather than left tied — polars does not promise a tie's order.
    if "level" in df.columns:
        sort_cols.append("level")
    if "metric_name" in df.columns:
        sort_cols.append("metric_name")

    return (
        df
        .with_columns(
            pl.Series("dataset_index", dataset_indices, dtype=pl.Int64),
            pl.Series("item_index", local_item_indices, dtype=pl.Int64),
        )
        .select(["dataset_index", "item_index"] + existing_cols)
        .sort(sort_cols)
    )


def reported_level(source_index: SourceIndex) -> FactorLevel | None:
    """Return the level to record for an address, or None where its key already says it.

    A result's ``level`` column exists to name the rows `target_index` cannot tell apart:
    a video frame and a track are both keyed, and nothing but the level separates them.
    The two *ends* need no such column — a null key is an item's own row and a key is one
    of its labels — so they are recorded as null and the column drops out entirely on any
    dataset that has only those two.

    That is what keeps the two spellings of one address producing one frame. The explicit
    ``SourceIndex(3, 7, "instance")`` and the minimal ``SourceIndex(3, 7)`` name the same
    row, are grouped and gated the same by :attr:`~dataeval.types.SourceIndex.kind`, and
    record the same nothing here.
    """
    kind = source_index.kind
    return None if kind == LABEL_KIND else kind


def selected_by_flags(source_index: SourceIndex, per_image: bool, per_target: bool) -> bool:
    """Whether `per_image` / `per_target` select an address.

    The two flags name the two ends of the level graph — an item's own row, and one of its
    labels — and :attr:`SourceIndex.kind` says which of those an address is, canonically,
    so the fully explicit spelling of a result is gated exactly as the minimal spelling of
    the same result is.

    A row *between* the two ends, such as a video frame or a track, is neither, so neither
    flag has a say over it and it is always selected. Nothing else could happen: there is no
    flag that names it, and dropping it would answer a caller who supplied per-frame
    statistics with an empty result and no reason.
    """
    kind = source_index.kind
    if kind is None:
        return per_image
    if kind == LABEL_KIND:
        return per_target
    return True


def drop_null_index_columns(df: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    """Drop specified columns if they contain no useful data.

    For scalar columns, checks whether all values are null.
    For list columns, checks whether all list elements across all rows are null.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to clean.
    columns : Sequence[str]
        Column names to check and potentially drop.

    Returns
    -------
    pl.DataFrame
        DataFrame with all-null columns removed.
    """
    for col in columns:
        if col not in df.columns:
            continue
        dtype = df[col].dtype
        if isinstance(dtype, pl.List):
            all_null = df[col].list.eval(pl.element().is_null().all()).list.first().all()
        else:
            all_null = df[col].null_count() == len(df)
        if all_null:
            df = df.drop(col)
    return df
