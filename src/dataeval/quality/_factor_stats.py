"""Decide which metadata factors can be thresholded, and lay them out as a stats result.

Eligibility depends on whether the raw values support ordering, so it is decided from the
column's polars dtype, not from :attr:`~dataeval.types.FactorInfo.factor_type` -- a binning
heuristic that marks integer counts and small samples discrete.
"""

__all__ = []

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import polars as pl

from dataeval._metadata._structurers._reserved import LEVEL_KEY_COLUMNS
from dataeval.core import StatsResult
from dataeval.types import FactorLevel, LevelSpec, SourceIndex

FactorKind = Literal["ordered", "categorical", "ineligible"]


def factor_levels(metadata: Any) -> dict[str, FactorLevel]:
    """Return every factor the metadata holds, keyed by name, mapped to the level it was measured at.

    The structure is read directly instead of through :meth:`~dataeval.Metadata.at` or
    :attr:`factor_info`, which bin every visible factor and warn about bin counts thresholding
    never reads. Each level's view is built the way :meth:`~dataeval.Metadata.at` builds it,
    minus the binning.
    """
    metadata._structure()
    found: dict[str, FactorLevel] = {}
    for level in metadata.levels:
        view = metadata._derived_copy()
        view._view = level
        view._build_factors()
        for name in view._visible_factors():
            found.setdefault(name, view._factor_level(name))
    return found


def factor_kind(metadata: Any, name: str) -> FactorKind:
    """Classify one factor by its raw column's dtype, not by the binning classifier.

    :attr:`FactorInfo.factor_type <dataeval.types.FactorInfo.factor_type>` decides how to bin;
    it marks integer counts and samples under 20 rows discrete. Eligibility depends on the
    raw values, so a vocabulary declared by the caller or a stats producer makes an integer
    factor categorical.
    """
    declared = metadata._encoding.get(name)
    if isinstance(declared, LevelSpec) and declared.provenance == "declared":
        return "categorical"
    dtype = metadata._store.dtype_of(name)
    if dtype == pl.Boolean or dtype == pl.String or isinstance(dtype, (pl.Categorical, pl.Enum)):
        return "categorical"
    if dtype.is_numeric() or dtype.is_temporal():
        return "ordered"
    return "ineligible"


def eligible_factors(metadata: Any, factors: Sequence[str] | None) -> dict[str, FactorLevel]:
    """Return the factors to threshold, each mapped to its level: the ones named, or every ordered one.

    Raises
    ------
    TypeError
        When `factors` is a single string rather than a sequence of names.
    KeyError
        When a named factor is not one of the metadata's.
    ValueError
        When `factors` is empty, when a named factor is categorical or ineligible, or when
        nothing is left to threshold.
    """
    levels = factor_levels(metadata)
    kinds: dict[str, FactorKind] = {name: factor_kind(metadata, name) for name in levels}
    if factors is None:
        ordered: dict[str, FactorLevel] = {name: level for name, level in levels.items() if kinds[name] == "ordered"}
        if not ordered:
            raise ValueError(
                "Outliers.evaluate: no ordered factors to threshold. Factors "
                f"{sorted(levels)} are categorical or ineligible. Categorical values are judged "
                "by rarity within a stratum, which is a separate test."
            )
        return ordered
    _check_named(metadata, factors, kinds)
    return {name: levels[name] for name in factors}


def _check_named(metadata: Any, factors: Sequence[str], kinds: Mapping[str, FactorKind]) -> None:
    """Validate `factors`: reject a single string, an empty list, and names that are not thresholdable."""
    if isinstance(factors, str):
        raise TypeError(f"Outliers.evaluate: `factors` takes a list of factor names; pass [{factors!r}].")
    if not factors:
        raise ValueError("Outliers.evaluate: `factors` names no factors; pass None to threshold every ordered one.")
    unknown = [name for name in factors if name not in kinds]
    if unknown:
        dropped = {name: list(metadata.dropped_factors[name]) for name in unknown if name in metadata.dropped_factors}
        raise KeyError(
            f"Outliers.evaluate: {unknown} are not factors of this metadata"
            + (f" (dropped: {dropped})" if dropped else "")
            + f". Its factors are {sorted(kinds)}."
        )
    refused = {name: kinds[name] for name in factors if kinds[name] != "ordered"}
    if refused:
        raise ValueError(
            f"Outliers.evaluate: {refused} cannot be thresholded: categorical factors have no "
            "location or scale, and ineligible columns are not quantities."
        )


def _address(metadata: Any, level: FactorLevel, row: Mapping[str, Any]) -> SourceIndex:
    """Return the :class:`~dataeval.types.SourceIndex` addressing a row at `level`.

    The item and label levels use unkeyed and keyed forms; levels between them name their
    level in the address.
    """
    if level == metadata.item_level:
        return SourceIndex(row["item_index"])
    if level == metadata.label_level:
        return SourceIndex(row["item_index"], row["target_index"])
    key_column = LEVEL_KEY_COLUMNS[level]
    if key_column is None:  # only "sequence" has none, and it is always the item level
        raise AssertionError(f"{level!r} has no key column and is not this metadata's item level")
    return SourceIndex(row["item_index"], row[key_column], level)


def factor_stats(metadata: Any, levels: Mapping[str, FactorLevel]) -> StatsResult:
    """Lay the factors out as a stats result: one row per entity at each level they live at.

    `levels` maps each factor to its own level, as :func:`eligible_factors` returns it. Each
    factor is read raw at its own level and filled with NaN on every other level's rows, so a
    threshold fitted within a level sees only that level's values. Times are converted to their
    epoch in the column's own unit.
    """
    metadata._structure()
    store = metadata._store
    source_index: list[SourceIndex] = []
    columns: dict[str, list[np.ndarray]] = {name: [] for name in levels}
    for level in (level for level in metadata.levels if level in levels.values()):
        rows = store.frame(level)
        # dict.fromkeys: the label level's own key is ``target_index``, named twice otherwise.
        keys = [
            col
            for col in dict.fromkeys(("item_index", "target_index", LEVEL_KEY_COLUMNS[level]))
            if col in rows.columns
        ]
        source_index.extend(_address(metadata, level, row) for row in rows.select(keys).iter_rows(named=True))
        for name, own in levels.items():
            if own == level:
                raw = rows[name].to_physical().cast(pl.Float64).fill_null(np.nan)
                columns[name].append(raw.to_numpy())
            else:
                columns[name].append(np.full(rows.height, np.nan))
    return StatsResult(
        stats={name: np.concatenate(parts) for name, parts in columns.items()},
        source_index=source_index,
        object_count=[],
        invalid_box_count=[],
        image_count=store.height(metadata.item_level),
    )
