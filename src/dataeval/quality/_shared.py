"""Shared utilities for quality evaluators (duplicates and outliers)."""

__all__ = []

import warnings
from collections.abc import Sequence
from typing import Any, NamedTuple

import polars as pl

from dataeval.core import StatsResult, image_stats
from dataeval.data._invalidates import invalidating_sources
from dataeval.exceptions import StatsInvalidatedWarning
from dataeval.flags import ImageStats
from dataeval.protocols import MetadataLike, _is_protocol_instance
from dataeval.types import FactorLevel, SourceIndex

# What :attr:`SourceIndex.kind` reports for a row that is one of an item's labels. An
# unstated level under a key resolves here on every task, so the evaluators can name the
# label end without a dataset to resolve against.
LABEL_KIND = "instance"

#: Which level names each task understands. Image spellings are the ones that shipped; a tracking
#: dataset uses the names :class:`~dataeval.Metadata` gives its levels.
IMAGE_LEVEL_NAMES = ("item", "target")
TRACKING_LEVEL_NAMES = ("sequence", "unit", "track", "instance")

#: Levels measured per detection rather than per whole frame.
TARGET_LEVELS = frozenset({"target", "instance", "track"})


class LevelPlan(NamedTuple):
    """Which levels to report, and the statistics that reporting them needs.

    ``levels`` and ``per_image``/``per_target`` are two spellings of one thing -- one names the
    answers wanted, the other the measurements taken -- so this resolves whichever was given into
    both. Passing both spellings at once is refused rather than reconciled.
    """

    per_image: bool
    per_target: bool
    levels: frozenset[str]


def resolve_levels(
    levels: str | Sequence[str] | None,
    per_image: bool | None,
    per_target: bool | None,
    tracking: bool,
    *,
    caller: str,
    subject: str = "dataset",
) -> LevelPlan:
    """Resolve the requested levels and the statistics they need.

    Parameters
    ----------
    levels : str or Sequence[str] or None
        The levels asked for by name, or None to derive them from the flags.
    per_image, per_target : bool or None
        The flags spelling of the same request. None means not given.
    tracking : bool
        Whether the data is tracking data, which decides the names `levels` accepts.
    caller : str
        The ``Class.method`` that prefixes every message.
    subject : str, default "dataset"
        What the messages call the thing that was passed, for example "metadata".

    Returns
    -------
    LevelPlan
        The levels to report and the flags that measure them. Flags that name no level resolve
        to an empty plan, which callers refuse before resolving.

    Raises
    ------
    ValueError
        If both spellings are given, `levels` is empty, or a level name is not one this task has.
    """
    names = TRACKING_LEVEL_NAMES if tracking else IMAGE_LEVEL_NAMES
    if levels is not None and (per_image is not None or per_target is not None):
        raise ValueError(
            f"{caller}: pass either `levels` or `per_image`/`per_target`, not both -- "
            "they are two spellings of one thing, and reconciling them silently would hide "
            f"whichever lost. `levels` for this {subject} accepts {list(names)}."
        )
    if levels is None:
        return _implied_levels(bool(per_image if per_image is not None else True), bool(per_target), tracking)

    wanted = frozenset([levels] if isinstance(levels, str) else levels)
    unknown = sorted(wanted - set(names))
    if unknown:
        raise ValueError(
            f"{caller}: {unknown} {'is' if len(unknown) == 1 else 'are'} not a level of "
            f"{'a tracking' if tracking else 'an image'} {subject}; it accepts {list(names)}."
        )
    if not wanted:
        raise ValueError(f"{caller}: `levels` must name at least one level.")
    # Frames are the substrate every tracking relation is built on -- a track is found among the
    # detections of frames, and the frame map that places them is built either way -- so they are
    # always measured. What `levels` decides there is which of them are reported.
    return LevelPlan(
        per_image=True if tracking else bool(wanted - TARGET_LEVELS),
        per_target=bool(wanted & TARGET_LEVELS),
        levels=wanted,
    )


def _implied_levels(per_image: bool, per_target: bool, tracking: bool) -> LevelPlan:
    """Resolve the older spelling, reproducing what each combination reports today."""
    if tracking:
        levels = {"sequence", "unit"} if per_image else set()
        if per_target:
            levels |= {"instance", "track"}
    else:
        levels = {"item"} if per_image else set()
        if per_target:
            levels |= {"target"}
    return LevelPlan(per_image=per_image, per_target=per_target, levels=frozenset(levels))


def is_metadata(data: Any) -> bool:
    """Whether `data` is a metadata container rather than a dataset.

    Asked before any decode path: :class:`~dataeval.Metadata` structurally implements
    :class:`~dataeval.protocols.Dataset` as well.
    """
    return _is_protocol_instance(data, MetadataLike)


def checked_sole_metadata(data: Any, metadata: Any, caller: str) -> Any:
    """Return `data` when it is the only metadata container named.

    Raises
    ------
    ValueError
        If `metadata` names a different container than `data`.
    """
    if metadata is not None and metadata is not data:
        raise ValueError(
            f"{caller}.evaluate: `data` and `metadata` specify two different metadata containers. "
            "Pass a single metadata container."
        )
    return data


def _stat_names(stats: ImageStats) -> str:
    """Render a flag as the stat column names it produces, in declaration order.

    Every ``ImageStats`` member is named ``<CATEGORY>_<STAT>``, and ``<STAT>`` lowercased
    is the calculator's output name (``DIMENSION_ASPECT_RATIO`` -> ``aspect_ratio``), so
    the mapping is derived rather than duplicated from ``dataeval.core._calculators``.

    Iterating a ``Flag`` value yields only the single-bit members it holds, so composite
    members (``PIXEL``, ``DIMENSION``, ...) never appear and only individual stat columns
    are named. The ``name`` test only narrows ``Flag.name``, which is typed optional.
    """
    return ", ".join(member.name.split("_", 1)[1].lower() for member in stats if member.name)


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
        reaches :func:`~dataeval.core.image_stats` — not what the evaluator was
        configured with. ``Duplicates`` computes ``flags & ImageStats.HASH``, so
        intersecting the invalidation against ``self.flags`` would warn about dimension
        stats it never computes.
    caller : str
        Name of the evaluator, used in the warning message.
    **kwargs : Any
        Forwarded to :func:`~dataeval.core.image_stats` unchanged.

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

    return [image_stats(ds, stats=stats, **kwargs) for ds in datasets]


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


def row_kinds(frame: pl.DataFrame) -> pl.Series:
    """Return the kind of each row of a result frame, as :attr:`SourceIndex.kind` would report it.

    A result frame states a level only for the rows between an item's own row and its labels, so the
    two ends are told apart by ``target_index``: a key is one of an item's labels, no key is the item
    itself. A column the frame no longer holds, because it was entirely null, reads as all null.
    """
    level = pl.col("level").cast(pl.Utf8) if "level" in frame.columns else pl.lit(None, dtype=pl.Utf8)
    keyed = pl.col("target_index").is_not_null() if "target_index" in frame.columns else pl.lit(False)
    kind = (
        pl
        .when(level.is_not_null())
        .then(level)
        .when(keyed)
        .then(pl.lit(LABEL_KIND))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
    )
    return frame.with_columns(kind.alias("_kind"))["_kind"]


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
