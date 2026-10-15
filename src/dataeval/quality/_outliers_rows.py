"""Locate statistics rows in a Metadata, read each row's class at the metadata's view, and stratify the rows.

Also holds the findings and population schemas every Outliers check reports in, and the threshold fit with its floor,
shared by the threshold checks and the score checks without an import cycle.
"""

__all__ = []

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval._helpers import lexical_categorical
from dataeval._metadata._structurers._reserved import LEVEL_KEY_COLUMNS
from dataeval.core._combination_scores import _label
from dataeval.protocols import Threshold
from dataeval.quality._factor_stats import factor_kind, factor_levels
from dataeval.quality._shared import LABEL_KIND
from dataeval.types import FactorLevel, SourceIndex
from dataeval.utils.thresholds import AdaptiveThreshold, ConstantThreshold

# The index columns a stratum may name besides the factors: a video, a frame, a track. They do not propagate.
RESERVED_STRATA = ("item_index", "unit_index", "track_id")
# A frame or track id restarts in every video, so it names one frame or track only beside its item_index.
_PER_ITEM = ("unit_index", "track_id")

DEFAULT_OUTLIERS_OUTLIER_THRESHOLD: Threshold = AdaptiveThreshold()

# The fewest real values a statistical threshold is fitted to. Below 20 rows the default threshold flags clean
# normal data far more often (3.4 % of rows at 5, 0.17 % at 10, 0.05 % at 20), so a finding would be mostly noise.
_MIN_POPULATION = 20
# The fewest scored rows per dimension a multi-column score is judged from.
_MIN_ROWS_PER_DIMENSION = 10

_POPULATION_SCHEMA: dict[str, Any] = {
    "level": pl.Utf8,  # the level name, or null for an item's own row on the two-way split
    "metric_name": lexical_categorical(),
    "method": lexical_categorical(),  # null for a single-column threshold
    "population": lexical_categorical(),  # null = the whole level
    "n": pl.Int64,
    "judged": pl.Boolean,
    "reason": pl.Utf8,  # null when judged
}


class _Population(NamedTuple):
    """One population a check was asked to judge, and whether it was."""

    level: str | None
    metric_name: str
    method: str | None
    population: str | None
    n: int
    judged: bool
    reason: str | None


def _populations_frame(rows: Sequence[_Population]) -> pl.DataFrame:
    """Lay out population rows in the populations schema."""
    return pl.DataFrame([tuple(row) for row in rows], schema=_POPULATION_SCHEMA, orient="row")


# The frame of reference every flagged row carries, beside the value that was flagged. Declared
# once: the schema, the empty-frame schema and the cross-dataset merge all read it from here.
_CONTEXT_COLUMNS: dict[str, Any] = {
    "direction": lexical_categorical(),
    "bound": pl.Float64,
    "percentile": pl.Float64,
    "population_mean": pl.Float64,
    "population_std": pl.Float64,
}


# What judged a row: the check (null for a single-column threshold), the population it was judged in (null for its
# whole level) and the categorical value observed. Each is dropped from a frame where it is entirely null.
_EXTRA_COLUMNS: dict[str, Any] = {
    "method": lexical_categorical(),
    "population": lexical_categorical(),
    "observed": lexical_categorical(),
}

# Every column a check's findings frame holds before the all-null ones are dropped.
FINDINGS_SCHEMA: dict[str, Any] = {
    "item_index": pl.Int64,
    "target_index": pl.Int64,
    "level": lexical_categorical(),
    "metric_name": lexical_categorical(),
    "metric_value": pl.Float64,
    **_CONTEXT_COLUMNS,
    **_EXTRA_COLUMNS,
}


class _Strata(NamedTuple):
    """Which population each statistics row is judged in, aligned with the combined source index.

    Attributes
    ----------
    codes : NDArray[np.intp]
        Each row's population, -1 where the row is judged with the whole level.
    labels : Mapping[int, str or None]
        The label of each population code, as ``result.populations`` lists it.
    columns : tuple[str, ...]
        The strata names, ``("<class>",)`` for a classwise split.
    """

    codes: NDArray[np.intp]
    labels: Mapping[int, str | None]
    columns: tuple[str, ...]


def coded(labels: Sequence[str | None], columns: Sequence[str]) -> _Strata:
    """Give each entry a code for its population label, and -1 to an entry judged with its whole level (label None)."""
    numbers: dict[str, int] = {}
    codes = np.fromiter(
        (-1 if label is None else numbers.setdefault(label, len(numbers)) for label in labels),
        dtype=np.intp,
        count=len(labels),
    )
    return _Strata(codes, {code: label for label, code in numbers.items()}, tuple(columns))


def has_level_schema(metadata: Any) -> bool:
    """Whether `metadata` is a Metadata with a level schema, rather than a plain MetadataLike."""
    return all(hasattr(metadata, name) for name in ("_store", "levels", "view", "item_level", "label_level"))


def row_level(metadata: Any, si: SourceIndex) -> FactorLevel:
    """Return the level a source-index entry names, in the metadata's own level names."""
    kind: FactorLevel | None = si.kind
    if kind is None:
        return metadata.item_level
    return metadata.label_level if kind == LABEL_KIND else kind


def level_kind(metadata: Any, level: FactorLevel) -> FactorLevel | None:
    """Return the :attr:`SourceIndex.kind` of a row at `level`, the inverse of :func:`row_level`."""
    if level == metadata.item_level:
        return None
    return LABEL_KIND if level == metadata.label_level else level


def _key_columns(metadata: Any, level: FactorLevel) -> list[str]:
    """Return the columns that identify a row at `level`, as `_factor_stats._address` reads them."""
    if level == metadata.item_level:
        return ["item_index"]
    if level == metadata.label_level:
        return ["item_index", "target_index"]
    return ["item_index", str(LEVEL_KEY_COLUMNS[level])]


def _key(metadata: Any, level: FactorLevel, si: SourceIndex) -> tuple[Any, ...]:
    """Return the key tuple a source-index entry names at its level."""
    return (si.item,) if level == metadata.item_level else (si.item, si.key)


def row_positions(metadata: Any, source_index: Sequence[SourceIndex]) -> tuple[list[FactorLevel], NDArray[np.intp]]:
    """Each entry's level and its row position in that level's frame, -1 where the row is not found."""
    metadata._structure()
    levels: list[FactorLevel] = [row_level(metadata, si) for si in source_index]
    lookup: dict[FactorLevel, dict[tuple[Any, ...], int]] = {}
    for level in set(levels):
        keys = metadata._store.frame(level).select(_key_columns(metadata, level)).iter_rows()
        lookup[level] = {key: position for position, key in enumerate(keys)}
    positions = np.fromiter(
        (lookup[level].get(_key(metadata, level, si), -1) for level, si in zip(levels, source_index, strict=True)),
        dtype=np.intp,
        count=len(source_index),
    )
    return levels, positions


def classes_of(
    metadata: Any, levels: Sequence[FactorLevel], positions: NDArray[np.intp]
) -> list[tuple[int, ...] | None]:
    """Each entry's classes at the metadata's view: its ancestor's, or the distinct ones below it, or None."""
    labels = np.asarray(metadata.class_labels)
    per_level = {level: _classes_at(metadata, level, labels) for level in dict.fromkeys(levels)}
    found: list[tuple[int, ...] | None] = []
    for level, position in zip(levels, positions.tolist(), strict=True):
        classes = per_level[level]
        found.append(None if classes is None else classes[position] if position >= 0 else ())
    return found


def _classes_at(metadata: Any, level: FactorLevel, labels: NDArray[np.intp]) -> list[tuple[int, ...]] | None:
    """Classes of every row at `level`, or None when the view's classes do not reach it."""
    store = metadata._store
    view = metadata.view
    if level == view:
        return [(int(c),) for c in labels]
    if store.schema.is_ancestor(view, level):
        up = store.link(level, view).positions()
        return [(int(labels[p]),) if p >= 0 else () for p in up.tolist()]
    if store.schema.is_ancestor(level, view):
        return _distinct_below(store.link(view, level).positions(), labels, store.height(level))
    return None


def _distinct_below(up: NDArray[np.intp], labels: NDArray[np.intp], height: int) -> list[tuple[int, ...]]:
    """For each row above the view, the sorted distinct classes of its rows at the view."""
    found: list[set[int]] = [set() for _ in range(height)]
    for parent, label in zip(up.tolist(), labels.tolist(), strict=True):
        if parent >= 0:
            found[parent].add(int(label))
    return [tuple(sorted(classes)) for classes in found]


def _reserved_level(metadata: Any, name: str) -> FactorLevel:
    """Return the coarsest level whose own frame holds a reserved index column."""
    for level in metadata.levels:
        if name in metadata._store.frame(level).columns:
            return level
    raise KeyError(
        f"Outliers: stratum {name!r} is not an index column of this metadata's levels {list(metadata.levels)}."
    )


def check_strata(metadata: Any, columns: Sequence[str]) -> dict[str, FactorLevel]:
    """Validate strata names and return each one's own level.

    Raises
    ------
    KeyError
        When a name is neither a factor of the metadata nor a reserved index column it holds.
    ValueError
        When a named factor is not categorical.
    """
    levels = factor_levels(metadata)
    own: dict[str, FactorLevel] = {}
    for name in columns:
        if name in RESERVED_STRATA:
            own[name] = _reserved_level(metadata, name)
            continue
        if name not in levels:
            raise KeyError(
                f"Outliers: stratum {name!r} is not a factor of this metadata. Its factors are {sorted(levels)}."
                + (" Stratify by the class label with classwise()." if name == "class_label" else "")
            )
        # A column holding no value at all has the Null dtype; its rows are one population, name=null.
        if factor_kind(metadata, name) != "categorical" and metadata._store.dtype_of(name) != pl.Null:
            raise ValueError(
                f"Outliers: stratum {name!r} is continuous; strata come from categorical values. Judge one column "
                f"given another with given={{y: {name!r}}}, or declare a vocabulary for an integer id."
            )
        own[name] = levels[name]
    return own


def _reaches(metadata: Any, name: str, own: FactorLevel, level: FactorLevel) -> bool:
    """Whether a stratum has a value on `level`'s rows: a factor by propagation, a reserved column natively."""
    if name in RESERVED_STRATA:
        return name in metadata._store.frame(level).columns
    if own != level and metadata._levels.propagates_to(own, level) and not metadata.inherited:
        raise ValueError(
            f"Outliers: stratum {name!r} is defined at {own!r}, above the judged level {level!r}, and this metadata "
            "was built with inherited=False, which is what keeps ancestor values off these rows. Judge it at its "
            "own level, or set inherited=True to accept one value per descendant row."
        )
    return metadata._levels.propagates_to(own, level)


def _level_labels(metadata: Any, own: Mapping[str, FactorLevel], level: FactorLevel) -> list[str] | None:
    """Label every row at `level` with its population, then a row with no value; return None if unreached."""
    reached = [name for name, at in own.items() if _reaches(metadata, name, at, level)]
    if not reached:
        return None
    qualified = [col for name in reached for col in (["item_index", name] if name in _PER_ITEM else [name])]
    reached = list(dict.fromkeys(qualified))
    keys = list(metadata._store.select(level, reached).iter_rows())
    label_of = {key: _label(reached, key) for key in dict.fromkeys(keys)}
    return [label_of[key] for key in keys] + [_label(reached, [None] * len(reached))]


def level_strata(metadata: Any, columns: Sequence[str], level: FactorLevel) -> tuple[pl.Series | None, dict[int, str]]:
    """Return the codes (a Utf8 Series, one per row of `level`'s frame) and labels for one level.

    The Series is None where no stratum reaches the level. A null stratum value is its own population, labeled
    ``name=null``.
    """
    labels = _level_labels(metadata, check_strata(metadata, columns), level)
    if labels is None:
        return None, {}
    strata = coded(labels[:-1], columns)
    return pl.Series("stratum", strata.codes.astype(str), dtype=pl.Utf8), {
        code: str(label) for code, label in strata.labels.items()
    }


def reached_levels(metadata: Any, columns: Sequence[str], levels: Iterable[FactorLevel]) -> list[FactorLevel]:
    """Return the levels among `levels` that at least one stratum reaches."""
    own = check_strata(metadata, columns)
    return [level for level in levels if any(_reaches(metadata, name, at, level) for name, at in own.items())]


def strata_codes(
    metadata: Any,
    columns: Sequence[str],
    levels: Sequence[FactorLevel],
    positions: NDArray[np.intp],
    also: Sequence[FactorLevel] = (),
) -> _Strata:
    """One population code per entry: -1 where no stratum reaches the entry's level, else the crossed values' code.

    An entry the metadata has no row for has no stratum value, and joins the population whose values are all null.
    `also` names the levels other checks judge within the same strata, which count as judged levels.

    Raises
    ------
    ValueError
        When the strata reach none of the entries' levels nor any level in `also`.
    """
    own = check_strata(metadata, columns)
    per_level = {level: _level_labels(metadata, own, level) for level in dict.fromkeys(levels)}
    if all(labels is None for labels in per_level.values()) and not reached_levels(metadata, columns, also):
        raise ValueError(
            f"Outliers: strata {dict(own)} (each at its own level) reach none of the judged levels "
            f"{list(dict.fromkeys([*per_level, *also]))}. A stratum reaches its own level and the levels below it."
        )
    # Position -1, a row the metadata does not hold, reads the trailing all-null label.
    return coded(
        [
            None if (labels := per_level[level]) is None else labels[position]
            for level, position in zip(levels, positions.tolist(), strict=True)
        ],
        columns,
    )


class _Fit(NamedTuple):
    """One threshold fitted to one population, and how each value under it sits against the fit.

    Every field is an array aligned with the values the fit was asked about, so a level made of
    several class buckets assembles one of these by writing each bucket's fit into its own slots.

    Attributes
    ----------
    mask : NDArray[np.bool_]
        Whether each value is an outlier.
    bound : NDArray[np.float64]
        The limit each value crossed, NaN where it crossed none.
    above : NDArray[np.bool_]
        Whether the crossed limit was the upper one.
    percentile : NDArray[np.float64]
        Each value's rank within the population, as a percentage.
    mean, std : NDArray[np.float64]
        The population's own figures, repeated for every value fitted against it.
    """

    mask: NDArray[np.bool_]
    bound: NDArray[np.float64]
    above: NDArray[np.bool_]
    percentile: NDArray[np.float64]
    mean: NDArray[np.float64]
    std: NDArray[np.float64]


def _unfitted(count: int) -> _Fit:
    """Return a fit that flags nothing, for a population no threshold could be derived from."""
    return _Fit(
        mask=np.zeros(count, dtype=bool),
        bound=np.full(count, np.nan, dtype=np.float64),
        above=np.zeros(count, dtype=bool),
        percentile=np.full(count, np.nan, dtype=np.float64),
        mean=np.full(count, np.nan, dtype=np.float64),
        std=np.full(count, np.nan, dtype=np.float64),
    )


def _percentile_of(values: NDArray[Any], population: NDArray[Any]) -> NDArray[np.float64]:
    """Rank each value within an ascending population, as a percentage, splitting ties evenly.

    Reported instead of a z-score because it means the same thing under every threshold. A z is
    only the statistic that flagged the row under :class:`~dataeval.utils.thresholds.ZScoreThreshold`;
    beside an adaptive or IQR bound it is a second, unrelated summary of the same value. The
    population's ``mean`` and ``std`` are reported too, so a caller who wants a z can form one.
    """
    left = np.searchsorted(population, values, side="left")
    right = np.searchsorted(population, values, side="right")
    return 100.0 * (left + right) / (2 * len(population))


def _crossings(
    values: NDArray[np.float64],
    real: NDArray[np.bool_],
    lower: float | None,
    upper: float | None,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float64]]:
    """Return which values fell below, which rose above, and the limit each one crossed."""
    below = real & (values < lower) if lower is not None else np.zeros(len(values), dtype=bool)
    above = real & (values > upper) if upper is not None else np.zeros(len(values), dtype=bool)
    bound = np.full(len(values), np.nan, dtype=np.float64)
    bound[below] = lower
    bound[above] = upper
    return below, above, bound


def _floor_reason(real: int, threshold: Threshold, minimum: int = _MIN_POPULATION) -> str | None:
    """Why a population is too small for a threshold fitted to it, or None. A constant threshold needs no data."""
    if isinstance(threshold, ConstantThreshold) or real >= minimum:
        return None
    return f"below {minimum} rows"


def _fit_outliers(values: NDArray[Any], threshold: Threshold) -> tuple[_Fit, int, str | None]:
    """Fit a threshold to one population and describe every value against it.

    Parameters
    ----------
    values : NDArray
        1D array of metric values making up one population.
    threshold : Threshold
        Threshold instance to compute bounds.

    Returns
    -------
    tuple[_Fit, int, str or None]
        The outlier mask and the frame of reference behind it, the population's count of real
        values, and why the population was not judged (``"below 20 rows"``, ``"no spread"``) or None.

    Notes
    -----
    The population is the array's non-NaN values, which is exactly what the threshold is fitted
    to. A NaN is never an outlier: it records a measurement that was not made, not an unusual
    value, and it takes no part in the mean, the spread or the percentile ranking either.

    A statistical threshold is fitted only to a population of at least ``_MIN_POPULATION`` real
    values; a :class:`~dataeval.utils.thresholds.ConstantThreshold` needs no data and is exempt.
    """
    count = len(values)
    float_values = values.astype(np.float64)
    real = ~np.isnan(float_values)
    n = int(real.sum())
    # An empty or all-NaN population is one of zero real values, which no threshold judges.
    reason = f"below {_MIN_POPULATION} rows" if n == 0 else _floor_reason(n, threshold)
    if reason is not None:
        return _unfitted(count), n, reason

    lower, upper = threshold(float_values)
    # Both bounds None means the threshold could not be derived (e.g. zero variance).
    if lower is None and upper is None:
        return _unfitted(count), n, "no spread"

    below, above, bound = _crossings(float_values, real, lower, upper)
    population = np.sort(float_values[real])
    fit = _Fit(
        mask=below | above,
        bound=bound,
        above=above,
        percentile=_percentile_of(float_values, population),
        mean=np.full(count, float(population.mean()), dtype=np.float64),
        std=np.full(count, float(population.std()), dtype=np.float64),
    )
    return fit, n, None
