"""Checks of a factor's values against the known-factor registry.

The checks mutate nothing they are given, so a caller can judge every column it intends to
write before writing any of them.
"""

__all__ = []

import dataclasses
import math
from collections.abc import Iterable
from numbers import Real
from typing import Any

import numpy as np

from dataeval._log import get_logger
from dataeval._metadata._structurers._data import StructuredData
from dataeval.types import KNOWN_FACTORS, FactorLevel
from dataeval.utils._internal import is_absent

_logger = get_logger(__name__)


def _is_valid(value: Any, lower: float | None, upper: float | None, lower_open: bool) -> bool:
    # A boolean is a Real to Python and a 0 or 1 to a reader, which is not a measurement.
    if not isinstance(value, Real) or isinstance(value, bool | np.bool_) or not math.isfinite(value):
        return False
    if lower is not None and (value <= lower if lower_open else value < lower):
        return False
    return upper is None or value <= upper


def known_violation(name: str, level: FactorLevel, values: Any) -> str | None:
    """Say how a factor's data breaks its known definition, if it does.

    Parameters
    ----------
    name : str
        The factor name. One that is not in ``KNOWN_FACTORS`` is never a violation.
    level : FactorLevel
        The level the values are defined at, checked before the values are.
    values : Any
        The factor's values at that level.

    Returns
    -------
    str or None
        A sentence naming the factor and the rule, or ``None`` when the data is valid.
    """
    known = KNOWN_FACTORS.get(name)
    if known is None:
        return None
    if level not in known.levels:
        allowed = " or ".join(repr(lv) for lv in known.levels)
        return f"{name!r} is defined at the {allowed} level, not {level!r}"

    # Absence is the store's own: None, or NaN as a float column spells a missing number.
    # Being known adds no null rule, so only what remains is held to the definition.
    present = [v for v in np.asarray(values, dtype=object).reshape(-1).tolist() if not is_absent(v)]
    bad = [v for v in present if not _is_valid(v, known.lower, known.upper, known.lower_open)]
    if not bad:
        return None

    if known.lower is None and known.upper is None:
        rule = "a finite number"
    else:
        lo = "(" if known.lower_open else "["
        low = "-inf" if known.lower is None else f"{known.lower:g}"
        high = "inf" if known.upper is None else f"{known.upper:g}"
        close = ")" if known.upper is None else "]"
        rule = f"a finite number in {lo}{low}, {high}{close}"
    return f"{name!r} must be {rule} ({known.unit}); {len(bad)} of {len(present)} values are not, e.g. {bad[0]!r}"


def reject_invalid_known(columns: Iterable[tuple[str, FactorLevel, Any]]) -> None:
    """Refuse the first column whose data breaks its known definition.

    Raises
    ------
    ValueError
        When a known factor is at a level it is not defined at, or holds an invalid value.
    """
    for name, level, values in columns:
        if (violation := known_violation(name, level, values)) is not None:
            # A timestamp is most often refused for being a date, which converts rather than renames.
            convert = "Pass seconds since the Unix epoch (UTC), or rename" if name == "timestamp" else "Rename"
            raise ValueError(f"{violation}. {convert} the factor if it means something else.")


def hold_back_invalid_known(data: StructuredData) -> StructuredData:
    """Hold back each walked factor whose data breaks its known definition.

    Nobody asked for the walk's columns, so a failing one is not refused: it is set aside
    the way a column mixing types is, kept as the walk read it for a repair to read.
    A column already held back is left alone; its first problem is the one to report.

    Returns
    -------
    StructuredData
        ``data`` itself when nothing fails, otherwise a copy with the failing columns moved
        from ``factors`` to ``unusable`` and ``"invalid_known_factor"`` recorded against each.
    """
    factors = {level: dict(columns) for level, columns in data.factors.items()}
    unusable = {level: dict(columns) for level, columns in data.unusable.items()}
    dropped = {name: list(reasons) for name, reasons in data.dropped_factors.items()}
    held = False
    for level, columns in data.factors.items():
        for name, values in columns.items():
            if (violation := known_violation(name, level, values)) is None:
                continue
            _logger.warning(violation)
            del factors[level][name]
            unusable.setdefault(level, {})[name] = values.tolist() if isinstance(values, np.ndarray) else list(values)
            dropped.setdefault(name, []).append("invalid_known_factor")
            held = True
    if not held:
        return data
    return dataclasses.replace(data, factors=factors, unusable=unusable, dropped_factors=dropped)
