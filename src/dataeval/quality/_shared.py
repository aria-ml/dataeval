"""Shared utilities for quality evaluators (duplicates and outliers)."""

__all__ = []

from collections.abc import Sequence

import polars as pl


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
