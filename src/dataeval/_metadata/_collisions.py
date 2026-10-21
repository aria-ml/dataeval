"""Comparing an added factor's values with those of a factor of the same name already present.

Backs ``Metadata.add_factors(return_collisions=True)``. Pure: the caller reads both columns
on the incoming factor's rows before anything is written, and this says which rows differ.
"""

__all__ = []

import math
from collections.abc import Mapping

import polars as pl


def reject_invalid_tolerance(tolerance: float | Mapping[str, float], names: set[str]) -> None:
    """Refuse a tolerance that is not finite and 0 or more, or that names a factor not being added.

    The second mirrors ``how=``, which refuses a factor the result does not declare.
    """
    values = tolerance.values() if isinstance(tolerance, Mapping) else [tolerance]
    if bad := [value for value in values if not (math.isfinite(value) and value >= 0)]:
        raise ValueError(f"collision_tolerance must be finite and 0 or more; got {bad}.")
    if isinstance(tolerance, Mapping) and (unknown := sorted(set(tolerance) - names)):
        raise ValueError(
            f"collision_tolerance names {unknown}, which this call does not add; it adds {sorted(names)}.",
        )


def tolerance_for(tolerance: float | Mapping[str, float], passed: str, placed: str) -> float:
    """One factor's tolerance: its placed name's entry, else its passed name's, else exact."""
    if not isinstance(tolerance, Mapping):
        return float(tolerance)
    return float(tolerance.get(placed, tolerance.get(passed, 0.0)))


def _present(values: pl.Series) -> pl.Series:
    """Whether each value is there: not null, and not NaN in a float column."""
    return (values.fill_nan(None) if values.dtype.is_float() else values).is_not_null()


def loses_values(prior: pl.Series, result: pl.Series) -> bool:
    """Whether writing `result` over `prior`, row for row, would lose a value `prior` holds.

    A value is lost when its row ends up missing or different. A row `prior` holds nothing
    on, null or NaN, has nothing to lose, so filling it loses nothing. A change of type
    counts as a change: ``1`` read back as ``1.0`` is not the value that was there.
    """
    held = _present(prior)
    if prior.dtype != result.dtype:
        return bool(held.any())
    kept = _present(result) & (prior == result).fill_null(False)
    return bool((held & ~kept).any())


def differing_rows(
    rows: pl.DataFrame, existing: pl.Series, incoming: pl.Series, tolerance: float
) -> tuple[int, pl.DataFrame]:
    """Compare two columns row by row, where both have a value.

    Parameters
    ----------
    rows : pl.DataFrame
        The columns that name each row, one row per value.
    existing, incoming : pl.Series
        The values already present and the ones being added, on those rows.
    tolerance : float
        Relative tolerance for numeric pairs.

    Returns
    -------
    tuple[int, pl.DataFrame]
        How many rows were compared, and ``rows`` narrowed to the ones that differ, with
        ``existing``, ``incoming`` and ``relative_difference`` beside them.
    """
    both = _present(existing) & _present(incoming)
    compared = rows.filter(both).with_columns(existing=existing.filter(both), incoming=incoming.filter(both))
    if existing.dtype.is_integer() and incoming.dtype.is_integer():
        # Python ints, exact at any size and any mix of signs. Cast to Float64 first, 2**53 + 1
        # and 2**53 read as one value, and polars 1.0 subtracts Int64 from UInt64 in Float64.
        # ponytail: a Python loop over the compared rows; vectorize in Int128 once the polars floor has it.
        pairs = zip(compared["existing"], compared["incoming"], strict=True)
        relative = pl.Series([abs(old - new) / max(abs(old), abs(new)) if old != new else 0.0 for old, new in pairs])
        return _differing(compared.with_columns(relative_difference=relative.cast(pl.Float64)), tolerance)
    if existing.dtype.is_numeric() and incoming.dtype.is_numeric():
        old, new = pl.col("existing").cast(pl.Float64), pl.col("incoming").cast(pl.Float64)
        # Equal values first: 0/0 and inf/inf are NaN, and NaN is what flags a difference below.
        relative = (
            pl
            .when(pl.col("existing") == pl.col("incoming"))
            .then(0.0)
            .otherwise((old - new).abs() / pl.max_horizontal(old.abs(), new.abs()))
        )
        return _differing(compared.with_columns(relative_difference=relative), tolerance)
    # Python equality, since polars refuses to compare a string column with a number column.
    unequal = pl.Series([old != new for old, new in zip(compared["existing"], compared["incoming"], strict=True)])
    differing = compared.filter(unequal) if len(unequal) else compared
    return compared.height, differing.with_columns(relative_difference=pl.lit(None, dtype=pl.Float64))


def _differing(compared: pl.DataFrame, tolerance: float) -> tuple[int, pl.DataFrame]:
    """Keep the numeric rows whose ratio is over `tolerance`, reading a NaN ratio as null.

    `not <=` rather than `>`, so a NaN ratio -- an infinity against a finite value, or
    against the other infinity -- differs. It is reported as null rather than NaN, since a
    filter on the ratio would otherwise drop it.
    """
    differing = compared.filter(~(pl.col("relative_difference") <= tolerance))
    return compared.height, differing.with_columns(pl.col("relative_difference").fill_nan(None))
