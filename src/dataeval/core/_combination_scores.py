"""Score how unusual each row of a table is as a combination of several of its columns."""

__all__ = []

import math
import warnings
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from typing import Any, Literal, NamedTuple, TypeAlias, TypedDict, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats
from sklearn.covariance import MinCovDet

from dataeval._metadata._structurers._reserved import RESERVED_COLUMNS
from dataeval.config import get_seed

ColumnKind: TypeAlias = Literal["numeric", "categorical", "vector", "sample", "tags"]
Method: TypeAlias = Literal["mahalanobis", "rarity", "residual", "distribution"]

REASONS: tuple[str, ...] = (
    "",
    "missing value",
    "missing stratum",
    "too few rows",
    "exact fit",
    "degenerate covariance",
    "constant residual",
)
_MISSING_VALUE, _MISSING_STRATUM, _TOO_FEW_ROWS, _EXACT_FIT, _DEGENERATE, _CONSTANT_RESIDUAL = range(1, 7)

_RESERVED = frozenset(RESERVED_COLUMNS) | {"dataset_index"}
_NUMERIC_FAMILY = frozenset({"numeric", "vector", "sample"})
_CATEGORICAL_FAMILY = frozenset({"categorical", "tags"})
_ABSENT = object()  # a value that keeps its row out of a group's population
# The finite-sample tail, with the consistency factor applied once, meets its calibration band (0.5 %-2 % of clean rows
# below 0.01) from 50 rows per dimension. Measured 2026-10-14 over 200 replicates: 2.4 % at 30 rows per dimension, 1.8 %
# at (100, 2), 1.4 % at (250, 5), 1.3 % at (500, 10).
_TAIL_ROWS_PER_DIMENSION = 50
# A spread from fewer neighbors is noisy enough to give the z heavy tails: at 10 neighbors the default
# threshold flags 0.5 % of clean rows at 50 to 100 rows, at 30 neighbors 0.04-0.08 %, as a single column does.
_MIN_WINDOW = 30
_GRID = (np.arange(100) + 0.5) / 100  # quantile levels at which samples are compared
_DRAWS = 50  # reference draws per sample length, for the expected distance at that length
_BLOCK_ROWS = 4096  # residual windows are built this many rows at a time, bounding peak memory to about block x window


class CombinationScoresResult(TypedDict):
    """
    How unusual each row is as a combination of columns, one entry per (row, test).

    Every per-entry array has the same length, ``len(data) * len(groups)``, sorted by ``group``
    then ``row``. Strings are stored once, in the label tuples, and the arrays hold indices into
    them.

    Attributes
    ----------
    groups : tuple[tuple[str, ...], ...]
        The columns of each test. A ``given={y: x}`` residual is ``(y, x)``.
    methods : tuple[str, ...]
        The method of each test: ``"mahalanobis"``, ``"rarity"``, ``"residual"`` or ``"distribution"``.
    dims : tuple[int, ...]
        Per test, the number of dimensions fitted for ``"mahalanobis"`` (a vector counts one per
        element, less one when its rows sum to a constant), otherwise the number of columns.
    kinds : Mapping[str, ColumnKind]
        The kind each scored column was read as.
    strata : tuple[str, ...]
        Stratum labels, such as ``"class_name=car"``. ``("",)`` without strata. Rows with a missing
        stratum value have the label ``"<missing>"``.
    observed : tuple[str, ...]
        Combination labels for ``"rarity"``, such as ``"weather=snow, time=night"``. Index 0 is ``""``.
    reasons : tuple[str, ...]
        Why a row was not scored. Index 0 is ``""``, a scored row.
    row : NDArray[np.int64]
        Position of the row in ``data``.
    group : NDArray[np.int32]
        Index into ``groups``.
    stratum : NDArray[np.int32]
        Index into ``strata``.
    score : NDArray[np.float64]
        The method's own number, NaN when the row was not scored. Higher is more unusual for
        ``"mahalanobis"`` and ``"distribution"``. For ``"rarity"`` it is the count of rows sharing the
        combination, so lower is rarer. For ``"residual"`` it is signed; the larger its magnitude, the
        more unusual.
    tail : NDArray[np.float64]
        For ``"mahalanobis"``, the finite-sample probability of a distance at least this large, reported
        where the population has at least 50 rows per dimension and no sample column is in the group.
        NaN otherwise.
    marginal : NDArray[np.float64]
        The row's largest single-column number. For ``"mahalanobis"`` and ``"residual"``, the largest
        absolute robust z of one column; for ``"rarity"``, the smallest count of any one column's value.
        A row unusual as a combination but ordinary in every column has a high ``score`` and a low
        ``marginal``. NaN for ``"distribution"``.
    n : NDArray[np.int64]
        Size of the population the row was compared with.
    combination : NDArray[np.int32]
        Index into ``observed``; 0 for methods other than ``"rarity"``.
    reason : NDArray[np.int8]
        Index into ``reasons``; 0 when the row was scored.
    """

    groups: tuple[tuple[str, ...], ...]
    methods: tuple[str, ...]
    dims: tuple[int, ...]
    kinds: Mapping[str, ColumnKind]
    strata: tuple[str, ...]
    observed: tuple[str, ...]
    reasons: tuple[str, ...]
    row: NDArray[np.int64]
    group: NDArray[np.int32]
    stratum: NDArray[np.int32]
    score: NDArray[np.float64]
    tail: NDArray[np.float64]
    marginal: NDArray[np.float64]
    n: NDArray[np.int64]
    combination: NDArray[np.int32]
    reason: NDArray[np.int8]


class _Test(NamedTuple):
    """One test: the columns it reads and the method that scores them."""

    columns: tuple[str, ...]
    method: Method


class _Scores(NamedTuple):
    """One test's numbers for every row of the data, filled population by population."""

    score: NDArray[np.float64]
    tail: NDArray[np.float64]
    marginal: NDArray[np.float64]
    n: NDArray[np.int64]
    combination: NDArray[np.int32]
    reason: NDArray[np.int8]

    @classmethod
    def empty(cls, height: int) -> "_Scores":
        """Return records for ``height`` rows, none scored yet."""
        return cls(
            np.full(height, np.nan),
            np.full(height, np.nan),
            np.full(height, np.nan),
            np.zeros(height, np.int64),
            np.zeros(height, np.int32),
            np.zeros(height, np.int8),
        )


_Scorer: TypeAlias = Callable[
    [pl.DataFrame, tuple[str, ...], Mapping[str, ColumnKind | None], list[NDArray[np.intp]], dict[Any, int]],
    tuple[_Scores, int],
]


# ----------------------------------------------------------------------------------------------
# Column kinds
# ----------------------------------------------------------------------------------------------


def _is_list(dtype: pl.DataType) -> bool:
    """Whether the column holds a list per row."""
    return isinstance(dtype, (pl.List, pl.Array))


def _is_label(dtype: pl.DataType) -> bool:
    """Whether the values name rather than measure: String, Boolean, Categorical or Enum."""
    return dtype == pl.String or dtype == pl.Boolean or isinstance(dtype, (pl.Categorical, pl.Enum))


def _is_quantity(dtype: pl.DataType) -> bool:
    """Whether the values are numbers or points in time."""
    return dtype.is_numeric() or dtype.is_temporal()


def _inner(dtype: pl.DataType) -> pl.DataType:
    """Return the element dtype of a list or array column."""
    return cast(pl.DataType, cast(pl.List, dtype).inner)


_FITS: Mapping[str, Callable[[pl.DataType], bool]] = {
    "numeric": _is_quantity,
    "categorical": lambda dtype: _is_quantity(dtype) or _is_label(dtype),
    "tags": lambda dtype: _is_list(dtype) and _is_label(_inner(dtype)),
    "vector": lambda dtype: _is_list(dtype) and _inner(dtype).is_numeric(),
    "sample": lambda dtype: _is_list(dtype) and _inner(dtype).is_numeric(),
}


def _default_kind(dtype: pl.DataType) -> ColumnKind | None:
    """Return the kind a dtype is read as without ``kinds=``, or None when it is not read at all."""
    if _is_quantity(dtype):
        return "numeric"
    if _is_label(dtype):
        return "categorical"
    return "tags" if _FITS["tags"](dtype) else None


def _resolve_kinds(frame: pl.DataFrame, kinds: Mapping[str, str]) -> dict[str, ColumnKind | None]:
    """Each column's kind, from its dtype unless ``kinds=`` overrides it."""
    resolved: dict[str, ColumnKind | None] = {name: _default_kind(dtype) for name, dtype in frame.schema.items()}
    for name, kind in kinds.items():
        if kind not in _FITS or not _FITS[kind](frame.schema[name]):
            raise ValueError(f"kinds: {name!r} holds {frame.schema[name]}, which cannot be read as {kind!r}.")
        resolved[name] = cast(ColumnKind, kind)
    return resolved


# ----------------------------------------------------------------------------------------------
# Validation and planning
# ----------------------------------------------------------------------------------------------


def _names(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Return a strata argument as a tuple of names."""
    if value is None:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


def _group(group: Sequence[str]) -> tuple[str, ...]:
    """One entry of ``groups`` as a tuple, refusing a bare string, an empty group and a repeated name."""
    if isinstance(group, str):
        raise TypeError(f"groups: a group is a sequence of column names, not the string {group!r}; write ({group!r},).")
    names = tuple(group)
    if not names or len(set(names)) != len(names):
        raise ValueError(f"groups: {names} must name at least one column, each once.")
    return names


def _named_groups(groups: Sequence[Sequence[str]] | None) -> list[tuple[str, ...]] | None:
    """Return the ``groups`` argument as tuples, or None for the pair search."""
    if groups is None:
        return None
    if isinstance(groups, str):
        raise TypeError("groups takes a sequence of column-name sequences, not a string.")
    return [_group(group) for group in groups]


def _check_names(
    frame: pl.DataFrame,
    groups: list[tuple[str, ...]] | None,
    strata: tuple[str, ...],
    given: Mapping[str, str],
    kinds: Mapping[str, str],
) -> None:
    """Refuse unknown names, reserved columns outside strata, and strata that are also scored."""
    scored = {name for group in groups or () for name in group} | set(given) | set(given.values())
    unknown = sorted((scored | set(strata) | set(kinds)) - set(frame.columns))
    if unknown:
        raise ValueError(f"{unknown} not a column of data, which has {frame.columns}.")
    reserved = sorted((scored | set(kinds)) & _RESERVED)
    if reserved:
        raise ValueError(
            f"{reserved} reserved: index columns address rows rather than describe them; use them as strata."
        )
    overlap = sorted(scored & set(strata))
    if overlap:
        raise ValueError(f"{overlap} named as strata, so cannot also be scored.")


def _check_stratum(name: str, dtype: pl.DataType, kind: ColumnKind | None) -> None:
    """Refuse a stratum that holds a list or an unread type, or takes continuous values."""
    if _is_list(dtype) or kind is None:
        raise ValueError(f"strata: {name!r} holds {dtype}, and a list or unread column cannot split the rows.")
    if not (dtype.is_integer() or dtype == pl.Date or kind == "categorical"):
        raise ValueError(
            f"strata: {name!r} holds {dtype}. Splitting on a continuous value needs bins, which make the result "
            f"depend on their edges. Condition on it with given=, or declare it kinds={{{name!r}: 'categorical'}}."
        )


def _check_strata(frame: pl.DataFrame, strata: tuple[str, ...], resolved: Mapping[str, ColumnKind | None]) -> None:
    """Refuse strata that cannot split the rows."""
    for name in strata:
        _check_stratum(name, frame.schema[name], resolved[name])


def _method(group: tuple[str, ...], resolved: Mapping[str, ColumnKind | None]) -> Method:
    """Return the method that scores a group, from the kinds of its columns."""
    kinds = {resolved[name] for name in group}
    if None in kinds:
        unread = [name for name in group if resolved[name] is None]
        raise ValueError(
            f"{unread} cannot be scored: a list of numbers needs kinds= ('vector' or 'sample'), and null, object, "
            "struct and nested columns are not read."
        )
    if kinds <= _CATEGORICAL_FAMILY:
        return "rarity"
    if kinds <= _NUMERIC_FAMILY:
        return "distribution" if len(group) == 1 and resolved[group[0]] == "sample" else "mahalanobis"
    raise ValueError(
        f"groups: {group} mixes numeric and categorical columns. To ask whether a value is unusual for its "
        "category, pass the category as strata."
    )


def _residual_tests(given: Mapping[str, str], resolved: Mapping[str, ColumnKind | None]) -> list[_Test]:
    """One residual test per ``given=`` entry."""
    tests = []
    for y, x in given.items():
        if y == x or resolved[y] != "numeric" or resolved[x] != "numeric":
            raise ValueError(f"given: {y!r} given {x!r} needs two different numeric columns.")
        tests.append(_Test((y, x), "residual"))
    return tests


def _same_family(pair: tuple[str, ...], resolved: Mapping[str, ColumnKind | None]) -> bool:
    """Whether both columns are numeric-family or both categorical-family."""
    kinds = {resolved[name] for name in pair}
    return kinds <= _NUMERIC_FAMILY or kinds <= _CATEGORICAL_FAMILY


def _pairs(
    frame: pl.DataFrame, strata: tuple[str, ...], resolved: Mapping[str, ColumnKind | None]
) -> list[tuple[str, ...]]:
    """Every pair of readable, unreserved, non-strata columns of one family, in column order."""
    eligible = [
        name for name in frame.columns if name not in strata and name not in _RESERVED and resolved[name] is not None
    ]
    return [pair for pair in combinations(eligible, 2) if _same_family(pair, resolved)]


def _plan(
    frame: pl.DataFrame,
    groups: Sequence[Sequence[str]] | None,
    strata: tuple[str, ...],
    given: Mapping[str, str],
    kinds: Mapping[str, str],
) -> tuple[list[_Test], dict[str, ColumnKind | None]]:
    """Validate the arguments and list the tests to run, with every column's kind."""
    named = _named_groups(groups)
    _check_names(frame, named, strata, given, kinds)
    resolved = _resolve_kinds(frame, kinds)
    _check_strata(frame, strata, resolved)
    searched = named if named is not None else _pairs(frame, strata, resolved)
    tests = [_Test(group, _method(group, resolved)) for group in searched] + _residual_tests(given, resolved)
    if named is None and not tests:
        raise ValueError("No pair of columns can be scored together: name groups=, or add given=.")
    return tests, resolved


# ----------------------------------------------------------------------------------------------
# Labels and strata
# ----------------------------------------------------------------------------------------------


def _format(value: Any) -> str:
    """One value for a label: null, a tag set joined by |, or the value, quoted when it holds a separator."""
    if value is None:
        return "null"
    if isinstance(value, tuple):
        return "|".join(_format(element) for element in value) if value else "{}"
    text = str(value)
    return repr(text) if any(mark in text for mark in ",=|") else text


def _label(names: Sequence[str], values: Sequence[Any]) -> str:
    """``name=value`` pairs joined by a comma, in the order the columns were given."""
    return ", ".join(f"{name}={_format(value)}" for name, value in zip(names, values, strict=True))


def _strata(
    frame: pl.DataFrame, strata: tuple[str, ...]
) -> tuple[NDArray[np.int32], tuple[str, ...], NDArray[np.bool_]]:
    """Each row's stratum code, the label of each code, and which rows have a missing stratum value."""
    if not strata:
        return np.zeros(frame.height, np.int32), ("",), np.zeros(frame.height, np.bool_)
    codes: dict[tuple[Any, ...] | None, int] = {}
    stratum = np.empty(frame.height, np.int32)
    for row, key in enumerate(frame.select(strata).iter_rows()):
        stratum[row] = codes.setdefault(None if None in key else key, len(codes))
    labels = tuple("<missing>" if key is None else _label(strata, key) for key in codes)
    missing = stratum == codes[None] if None in codes else np.zeros(frame.height, np.bool_)
    return stratum, labels, missing


def _populations(stratum: NDArray[np.int32], missing: NDArray[np.bool_], count: int) -> list[NDArray[np.intp]]:
    """Return the rows of each stratum, leaving out rows with a missing stratum."""
    rows = (np.flatnonzero((stratum == code) & ~missing) for code in range(count))
    return [population for population in rows if population.size]


# ----------------------------------------------------------------------------------------------
# rarity
# ----------------------------------------------------------------------------------------------


def _categories(frame: pl.DataFrame, name: str, kind: ColumnKind | None) -> list[Any]:
    """Return a categorical column's values (null is a level) or a tags column's sorted sets (a null list is absent)."""
    values = frame[name].to_list()
    if kind == "categorical":
        return values
    return [_ABSENT if tags is None else tuple(sorted({tag for tag in tags if tag is not None})) for tags in values]


def _count(
    rows: list[tuple[Any, ...]], present: list[int], columns: tuple[str, ...], scores: _Scores, observed: dict[Any, int]
) -> None:
    """Fill one population's counts: each row's combination, and the rarest of its single values."""
    joint = Counter(rows[row] for row in present)
    single = [Counter(rows[row][position] for row in present) for position in range(len(columns))]
    for row in present:
        value = rows[row]
        scores.score[row] = joint[value]
        scores.marginal[row] = min(counts[part] for counts, part in zip(single, value, strict=True))
        scores.n[row] = len(present)
        scores.combination[row] = observed.setdefault((columns, value), len(observed) + 1)


def _rarity(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    resolved: Mapping[str, ColumnKind | None],
    populations: list[NDArray[np.intp]],
    observed: dict[Any, int],
) -> tuple[_Scores, int]:
    """Count, within each population, the rows sharing each row's combination of categorical values."""
    # ponytail: a Python loop over rows; switch to a polars group_by if pair searches over large tables measure slow.
    rows = list(zip(*(_categories(frame, name, resolved[name]) for name in columns), strict=True))
    scores = _Scores.empty(frame.height)
    for population in populations:
        present = [int(row) for row in population if _ABSENT not in rows[row]]
        scores.reason[[int(row) for row in population if _ABSENT in rows[row]]] = _MISSING_VALUE
        _count(rows, present, columns, scores, observed)
    return scores, len(columns)


# ----------------------------------------------------------------------------------------------
# distribution
# ----------------------------------------------------------------------------------------------


def _samples(frame: pl.DataFrame, name: str) -> list[NDArray[np.float64] | None]:
    """Return a sample column as one array per row, non-finite elements dropped, None where nothing is left."""
    samples: list[NDArray[np.float64] | None] = []
    for values in frame[name].to_list():
        sample = np.array([np.nan if value is None else value for value in values or ()], np.float64)
        sample = sample[np.isfinite(sample)]
        samples.append(sample if sample.size else None)
    return samples


def _expected_distance(reference: NDArray[np.float64], lengths: NDArray[np.intp]) -> NDArray[np.float64]:
    """Return the mean distance to the reference of samples of each length drawn from it, by inverse-CDF sampling."""
    rng = np.random.default_rng(get_seed())
    expected = np.empty(lengths.size)
    for position, length in enumerate(lengths):
        draws = np.interp(rng.random((_DRAWS, int(length))), _GRID, reference)
        quantiles = np.quantile(draws, _GRID, axis=1, method="linear").T
        expected[position] = np.abs(quantiles - reference).mean()
    return expected


def _distribution(samples: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Return each sample's 1-Wasserstein distance to the median quantile function, over its length's typical one."""
    quantiles = np.stack([np.quantile(sample, _GRID, method="linear") for sample in samples])
    reference = np.median(quantiles, axis=0)
    distance = np.abs(quantiles - reference).mean(axis=1)
    lengths = np.array([sample.size for sample in samples])
    unique = np.unique(lengths)
    typical = _expected_distance(reference, unique)[np.searchsorted(unique, lengths)]
    return np.divide(distance, typical, out=distance.copy(), where=typical > 0)


def _distribution_test(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    _resolved: Mapping[str, ColumnKind | None],
    populations: list[NDArray[np.intp]],
    _observed: dict[Any, int],
) -> tuple[_Scores, int]:
    """Score each population's samples by their distance to the population's typical distribution."""
    samples = _samples(frame, columns[0])
    scores = _Scores.empty(frame.height)
    for population in populations:
        rows = np.array([row for row in population if samples[row] is not None], np.intp)
        scores.reason[np.setdiff1d(population, rows)] = _MISSING_VALUE
        scores.n[rows] = rows.size
        if rows.size < 3:
            scores.reason[rows] = _TOO_FEW_ROWS
            continue
        scores.score[rows] = _distribution([cast(NDArray[np.float64], samples[row]) for row in rows])
    return scores, 1


# ----------------------------------------------------------------------------------------------
# mahalanobis
# ----------------------------------------------------------------------------------------------


def _numeric(frame: pl.DataFrame, name: str) -> NDArray[np.float64]:
    """Return a numeric column as floats, NaN where missing; temporal values as offsets from the column minimum."""
    column = frame[name]
    if column.dtype.is_temporal():
        column = column.to_physical()
        column = column - column.min()
    values = column.cast(pl.Float64).to_numpy().astype(np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def _drop_compositional(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Drop the last element when every complete row sums to one value: the elements span one dimension less."""
    sums = matrix[~np.isnan(matrix).any(axis=1)].sum(axis=1)
    if matrix.shape[1] > 1 and sums.size and np.allclose(sums, sums[0], rtol=1e-9, atol=0):
        return matrix[:, :-1]
    return matrix


def _vector(frame: pl.DataFrame, name: str) -> NDArray[np.float64]:
    """Return a vector column as a (rows, width) array, a whole row NaN where it is null, empty or incomplete."""
    lists = frame[name].to_list()
    widths = {len(values) for values in lists if values}
    if len(widths) > 1:
        raise ValueError(
            f"{name!r} holds lists of lengths {sorted(widths)}; a vector has one length. Read it as a sample."
        )
    # A column with no list, or only empty ones, keeps one all-NaN element so that every row is missing.
    matrix = np.full((len(lists), (widths.pop() if widths else 0) or 1), np.nan)
    for row, values in enumerate(lists):
        if values:
            matrix[row] = [np.nan if value is None else value for value in values]
    matrix[~np.isfinite(matrix).all(axis=1)] = np.nan
    return _drop_compositional(matrix)


_Block: TypeAlias = "NDArray[np.float64] | list[NDArray[np.float64] | None]"


def _block(frame: pl.DataFrame, name: str, kind: ColumnKind | None) -> _Block:
    """Return a numeric-family column as what it adds to a fit: a 2-D array, or a sample per row."""
    if kind == "sample":
        return _samples(frame, name)
    return _numeric(frame, name)[:, None] if kind == "numeric" else _vector(frame, name)


def _absent(block: _Block) -> NDArray[np.bool_]:
    """Return which rows a block has no value for."""
    if isinstance(block, np.ndarray):
        return np.asarray(np.isnan(block).any(axis=1))
    return np.array([sample is None for sample in block], np.bool_)


def _design(block: _Block, rows: NDArray[np.intp]) -> NDArray[np.float64]:
    """Return a block's columns for one population's rows; a sample becomes log1p of its distribution score there."""
    if isinstance(block, np.ndarray):
        return block[rows]
    if not rows.size:
        return np.empty((0, 1))
    return np.log1p(_distribution([cast(NDArray[np.float64], block[row]) for row in rows]))[:, None]


def _width(block: _Block) -> int:
    """Return the dimensions a block adds to a fit."""
    return block.shape[1] if isinstance(block, np.ndarray) else 1


def _robust_z(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the absolute robust z of each column, |x - median| over 1.4826 MAD; NaN where the MAD is zero."""
    median = np.median(x, axis=0)
    spread = 1.4826 * np.median(np.abs(x - median), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(x - median) / spread
    z[:, spread == 0] = np.nan
    return z


def _largest_robust_z(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return each row's largest absolute robust z over the columns; NaN when no column has a spread."""
    z = _robust_z(x)
    usable = ~np.isnan(z).all(axis=0)
    return np.max(z[:, usable], axis=1) if usable.any() else np.full(len(x), np.nan)


def _tail(squared: NDArray[np.float64], support: NDArray[np.bool_], d: int) -> NDArray[np.float64]:
    """Return the finite-sample tail of reweighted MCD distances, Beta in the support and F outside (Cerioli 2010)."""
    w = int(support.sum())
    inside = stats.beta.sf(squared * w / (w - 1) ** 2, d / 2, (w - d - 1) / 2)
    outside = stats.f.sf(squared * w * (w - d) / ((w + 1) * (w - 1) * d), d, w - d)
    return np.where(support, inside, outside)


def _support_scatter(x: NDArray[np.float64], support: NDArray[np.bool_]) -> tuple[NDArray[np.float64], ...]:
    """Return the mean and consistency-scaled ddof=1 covariance of the support rows, taken without sklearn's own."""
    d = x.shape[1]
    k = 0.975 / stats.chi2.cdf(stats.chi2.ppf(0.975, d), d + 2)
    return x[support].mean(axis=0), k * np.cov(x[support], rowvar=False, ddof=1).reshape(d, d)


def _unfittable(x: NDArray[np.float64]) -> int:
    """Return why a population cannot be fitted before trying: too few rows, or too many sharing one value."""
    n, d = x.shape
    if n <= d + 1:
        return _TOO_FEW_ROWS
    shared = max(int(np.unique(column, return_counts=True)[1].max()) for column in x.T)
    return _EXACT_FIT if shared >= (n + d + 2) // 2 else 0


def _fit(x: NDArray[np.float64], rows: NDArray[np.intp], scores: _Scores, *, calibrated: bool) -> None:
    """Score one population's rows by robust Mahalanobis distance, or record why they cannot be."""
    n, d = x.shape
    scores.n[rows] = n
    scores.reason[rows] = _unfittable(x)
    if n == 0 or scores.reason[rows[0]]:
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The covariance matrix associated")
        fit = MinCovDet(random_state=get_seed()).fit(x)
    mean, scatter = _support_scatter(x, fit.support_)
    eigenvalues = np.linalg.eigvalsh(scatter)
    if eigenvalues.min() <= eigenvalues.max() * d * np.finfo(np.float64).eps:
        scores.reason[rows] = _DEGENERATE
        return
    centered = x - mean
    squared = np.einsum("ij,ji->i", centered, np.linalg.solve(scatter, centered.T))
    scores.score[rows] = np.sqrt(squared)
    scores.marginal[rows] = _largest_robust_z(x)
    if calibrated and n >= _TAIL_ROWS_PER_DIMENSION * d:
        scores.tail[rows] = _tail(squared, fit.support_, d)


def _mahalanobis(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    resolved: Mapping[str, ColumnKind | None],
    populations: list[NDArray[np.intp]],
    _observed: dict[Any, int],
) -> tuple[_Scores, int]:
    """Score each population's rows by robust Mahalanobis distance over the group's numeric dimensions."""
    # ponytail: one MCD fit per group per stratum, run in sequence, about 0.1 ms per row per fit (300k rows x 190 pairs
    # is ~1.7 h); parallelize per group, or fit on a seeded subsample and score every row.
    blocks = [_block(frame, name, resolved[name]) for name in columns]
    absent = np.logical_or.reduce([_absent(block) for block in blocks])
    calibrated = all(resolved[name] != "sample" for name in columns)
    scores = _Scores.empty(frame.height)
    for population in populations:
        scores.reason[population[absent[population]]] = _MISSING_VALUE
        rows = population[~absent[population]]
        _fit(np.hstack([_design(block, rows) for block in blocks]), rows, scores, calibrated=calibrated)
    return scores, sum(_width(block) for block in blocks)


# ----------------------------------------------------------------------------------------------
# residual
# ----------------------------------------------------------------------------------------------


def _windows(n: int, width: int, start: int = 0, stop: int | None = None) -> NDArray[np.intp]:
    """For sorted positions ``start`` to ``stop`` of ``n``, the ``width`` nearest others, shifted inward at the ends."""
    position = np.arange(start, n if stop is None else stop)
    first = np.clip(position - width // 2, 0, n - width - 1)
    span = first[:, None] + np.arange(width + 1)
    return span[span != position[:, None]].reshape(position.size, width)


def _block_z(ordered: NDArray[np.float64], width: int, start: int, stop: int) -> NDArray[np.float64]:
    """Return the robust z of sorted positions ``start`` to ``stop`` against their windows, NaN where flat."""
    neighbors = ordered[_windows(ordered.size, width, start, stop)]
    median = np.median(neighbors, axis=1)
    spread = 1.4826 * np.median(np.abs(neighbors - median[:, None]), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(spread == 0, np.nan, (ordered[start:stop] - median) / spread)


def _local_fit(y: NDArray[np.float64], x: NDArray[np.float64], rows: NDArray[np.intp], scores: _Scores) -> None:
    """Score one population's rows by the robust z of ``y`` against its nearest rows in ``x``."""
    n = rows.size
    width = max(_MIN_WINDOW, math.ceil(math.sqrt(n)))
    scores.n[rows] = n
    if n < width + 1:
        scores.reason[rows] = _TOO_FEW_ROWS
        return
    order = rows[np.argsort(x[rows], kind="stable")]
    ordered = y[order]
    blocks = [_block_z(ordered, width, start, min(start + _BLOCK_ROWS, n)) for start in range(0, n, _BLOCK_ROWS)]
    z = np.concatenate(blocks)
    scores.score[order] = z
    scores.reason[order[np.isnan(z)]] = _CONSTANT_RESIDUAL
    scores.marginal[rows] = _robust_z(y[rows][:, None])[:, 0]


def _residual(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    _resolved: Mapping[str, ColumnKind | None],
    populations: list[NDArray[np.intp]],
    _observed: dict[Any, int],
) -> tuple[_Scores, int]:
    """Score ``y`` given ``x`` within each population by a local robust fit, following curved relationships."""
    # ponytail: n * width medians (about n^1.5), memory bounded by _BLOCK_ROWS * width; a running median if slow.
    y, x = (_numeric(frame, name) for name in columns)
    absent = np.isnan(y) | np.isnan(x)
    scores = _Scores.empty(frame.height)
    for population in populations:
        scores.reason[population[absent[population]]] = _MISSING_VALUE
        _local_fit(y, x, population[~absent[population]], scores)
    return scores, 2


_SCORERS: dict[str, _Scorer] = {
    "rarity": _rarity,
    "mahalanobis": _mahalanobis,
    "residual": _residual,
    "distribution": _distribution_test,
}


# ----------------------------------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------------------------------


def _stack(parts: list[_Scores], field: str, dtype: type[np.generic]) -> NDArray[Any]:
    """One field of every test's records, end to end."""
    return np.concatenate([getattr(part, field) for part in parts]) if parts else np.empty(0, dtype)


def _assemble(
    height: int,
    tests: list[_Test],
    resolved: Mapping[str, ColumnKind | None],
    stratum: NDArray[np.int32],
    strata_labels: tuple[str, ...],
    scored: list[tuple[_Scores, int]],
    observed: dict[Any, int],
) -> CombinationScoresResult:
    """Stack the tests' records into one columnar result."""
    parts = [scores for scores, _ in scored]
    count = len(tests)
    return CombinationScoresResult(
        groups=tuple(test.columns for test in tests),
        methods=tuple(test.method for test in tests),
        dims=tuple(dims for _, dims in scored),
        kinds={name: cast(ColumnKind, resolved[name]) for test in tests for name in test.columns},
        strata=strata_labels,
        observed=("", *(_label(columns, value) for columns, value in observed)),
        reasons=REASONS,
        row=np.tile(np.arange(height, dtype=np.int64), count),
        group=np.repeat(np.arange(count, dtype=np.int32), height),
        stratum=np.tile(stratum, count),
        score=_stack(parts, "score", np.float64),
        tail=_stack(parts, "tail", np.float64),
        marginal=_stack(parts, "marginal", np.float64),
        n=_stack(parts, "n", np.int64),
        combination=_stack(parts, "combination", np.int32),
        reason=_stack(parts, "reason", np.int8),
    )


def combination_scores(
    data: pl.DataFrame | Mapping[str, Sequence[Any] | NDArray[Any]],
    groups: Sequence[Sequence[str]] | None = None,
    *,
    strata: str | Sequence[str] | None = None,
    given: Mapping[str, str] | None = None,
    kinds: Mapping[str, ColumnKind] | None = None,
) -> CombinationScoresResult:
    """
    Score how unusual each row is as a combination of several columns.

    A value can be ordinary in every column on its own and still be improbable as a combination: a
    large box near the horizon, a bright frame stamped 02:00, a fast track with low straightness. This scores
    every row of ``data`` once per test, within its stratum, and reports the numbers behind each
    score. It never decides what is an outlier; an evaluator such as
    :class:`~dataeval.quality.Outliers` applies thresholds to these numbers.

    Each column is read by the kind its dtype implies, and each test is scored by the method that
    fits its columns:

    ============================  ==================  ================================================
    Columns of the test           Method              ``score``
    ============================  ==================  ================================================
    numeric, vectors and samples  ``mahalanobis``     robust Mahalanobis distance, higher is rarer
    categorical and tag sets      ``rarity``          rows sharing the combination, lower is rarer
    one sample column             ``distribution``    length-adjusted Wasserstein distance
    ``given={y: x}``              ``residual``        local robust z of ``y`` given ``x``, signed
    ============================  ==================  ================================================

    Parameters
    ----------
    data : pl.DataFrame or Mapping[str, Sequence or NDArray]
        The table. A mapping is read as a :class:`polars.DataFrame`. Integer, float, decimal and
        temporal columns are numeric; string, categorical, enum and boolean columns are categorical;
        lists of strings or booleans are tag sets. A list of numbers is read only when ``kinds=`` says
        whether it is a ``"vector"`` or a ``"sample"``.
    groups : Sequence[Sequence[str]] or None, default None
        The columns to score together, one test per group. A group of one column scores that column
        alone. A group may not mix numeric and categorical columns; pass the category as ``strata``
        instead. None tests every pair of readable columns of one family, leaving out strata and
        reserved index columns.
    strata : str, Sequence[str] or None, default None
        Columns whose values split the rows into populations; each row is compared only with its own.
        Categorical, boolean, integer and date columns, and reserved index columns such as
        ``item_index``, may be strata. A float or datetime column may not, since splitting on it needs
        bins.
    given : Mapping[str, str] or None, default None
        ``{y: x}`` adds a ``residual`` test of numeric column ``y`` given numeric column ``x``.
    kinds : Mapping[str, ColumnKind] or None, default None
        Overrides of the kind a column is read as, such as ``{"speeds": "sample"}`` or
        ``{"camera_id": "categorical"}``.

    Returns
    -------
    CombinationScoresResult
        One entry per (row, test), with the score, the finite-sample tail where one applies, the row's
        largest single-column number, the population size, and why a row was not scored.

    Raises
    ------
    ValueError
        When a name is not a column; a reserved index column is scored; a strata column is also
        scored or holds a list or continuous values; a group is empty, repeats a column, mixes numeric and
        categorical columns, or holds a column that cannot be read; a ``given=`` entry is not two
        different numeric columns; a ``kinds=`` override does not fit the column's dtype; or
        ``groups=None`` finds no pair and ``given`` is empty.
    TypeError
        When ``groups``, or one of its entries, is a bare string.

    Notes
    -----
    A row that cannot be scored keeps its entry, with ``score`` NaN and a ``reason``: a missing
    value, a missing stratum, too few rows (a ``given=`` residual needs at least 31 rows in each
    population), more rows sharing one value than a robust fit can use
    (``exact fit``), a covariance without full rank, or a constant residual. Missing means null,
    NaN or infinite for numbers; a null category is a level of its own.

    Examples
    --------
    >>> from dataeval.core import combination_scores
    >>> result = combination_scores({"weather": ["clear"] * 9 + ["snow"]}, groups=[("weather",)])
    >>> result["score"]
    array([9., 9., 9., 9., 9., 9., 9., 9., 9., 1.])
    >>> result["observed"][result["combination"][-1]]
    'weather=snow'
    """
    frame = data if isinstance(data, pl.DataFrame) else pl.DataFrame(dict(data))
    frame = frame.with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None))  # NaN is the null level, not a unique key
    strata_names = _names(strata)
    tests, resolved = _plan(frame, groups, strata_names, dict(given or {}), dict(kinds or {}))
    stratum, strata_labels, missing = _strata(frame, strata_names)
    populations = _populations(stratum, missing, len(strata_labels))
    observed: dict[Any, int] = {}
    scored = [_SCORERS[test.method](frame, test.columns, resolved, populations, observed) for test in tests]
    for scores, _ in scored:
        scores.reason[missing] = _MISSING_STRATUM
    return _assemble(frame.height, tests, resolved, stratum, strata_labels, scored, observed)
