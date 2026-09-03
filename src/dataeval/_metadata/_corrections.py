"""Applying declared corrections to the values of a column nobody could read.

A column no reading could make a factor of is held back rather than coerced, and its values
are kept exactly as the dataset wrote them. This is where a caller's decision about how to
read them is carried out: the corrections are applied to those values, and a column that
comes out agreeing with itself, and grouping its rows rather than naming them, becomes a
factor.

Two kinds of column arrive here. One whose values **disagree about what they are** — a
sentinel letter among coordinates, a number wearing a unit — is answered by
:class:`~dataeval.types.Remap` or :class:`~dataeval.types.ParseValue`. One whose values agree
perfectly but **name their rows instead of grouping them**, which is what a column of
timestamps does, is answered by :class:`~dataeval.types.ParseDateTime`: reading a timestamp
as the period it falls in is what gives the column a vocabulary to group by.

The application is deliberately separate from the declarations. Those are records —
storable, comparable, written into the descriptor and read back — and keeping the
arithmetic out of them is what lets one be reviewed in a pull request without running it.
"""

__all__ = []

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timezone
from types import MappingProxyType
from typing import Any

import numpy as np

from dataeval.types import ParseDateTime, ParseValue, Remap, Rescale
from dataeval.types._factors import EPOCH_SECONDS
from dataeval.utils._internal import simplify_type

Correction = ParseValue | ParseDateTime | Remap | Rescale


def is_absent(value: Any) -> bool:
    """Whether a row recorded no value at all.

    Checked before any correction is consulted, which is what keeps "not recorded" and
    "a value the mapping does not name" two different answers. A catch-all that swallowed
    absence would collapse them, and the reserved missing code exists to hold them apart.
    """
    return value is None or (isinstance(value, float | np.floating) and bool(np.isnan(value)))


def for_factor(factor: str, corrections: Sequence[Correction]) -> list[Correction]:
    """Select the corrections that name one factor, in the order they were declared."""
    return [correction for correction in corrections if correction.factor == factor]


def apply(values: Sequence[Any], corrections: Sequence[Correction]) -> list[Any]:
    """Apply one factor's corrections to its values, in declaration order.

    Each value is carried through the list: a :class:`~dataeval.types.Remap` that matches
    replaces it, a :class:`~dataeval.types.ParseValue` strips its decoration, a
    :class:`~dataeval.types.ParseDateTime` reads it as a time, and a
    :class:`~dataeval.types.Rescale` whose range contains it transforms it. Only the
    **first** matching rescale applies, so two overlapping ranges resolve to the one
    declared first rather than compounding into a number neither of them describes. The
    others still apply before or after one, which is what lets a column be read into numbers
    and then converted in a single declaration.

    A value no correction matches is left exactly as it was, so a partial mapping is a
    partial mapping: a column its leftovers keep mixed simply stays unusable and says so,
    rather than being quietly completed by a rule nobody wrote.
    """
    return [value if is_absent(value) else _corrected(value, corrections) for value in values]


def _corrected(value: Any, corrections: Sequence[Correction]) -> Any:
    """Carry one present value through the corrections that name its factor.

    A rescale is the one kind not read off the table, because whether it applies is a
    question about the *list* rather than about the value: only the first whose range
    contains it does, so two overlapping ranges resolve to the one declared first rather
    than compounding into a number neither describes.
    """
    rescaled = False
    for correction in corrections:
        if isinstance(correction, Rescale):
            if not rescaled and _within(value, correction.over):
                value = _as_number(value) * correction.multiply + correction.add
                rescaled = True
        else:
            # Indexed rather than fetched with a default: a correction the table has no
            # reader for is a correction nobody wrote code to apply, and silently returning
            # the value unchanged would read it as a rule that happened to match nothing.
            value = _VALUE_READERS[type(correction)](value, correction)
    return value


def _parsed_value(value: Any, parse: ParseValue) -> Any:
    """Remove a value's decoration, so the number underneath can be read.

    Only text is touched: a row that already holds a number has nothing wearing decoration,
    and stripping substrings from its spelling would be a second reading of a value that was
    never in doubt. The cleaned text is handed back as text, because what it *becomes* is
    decided where every other column's type is.
    """
    if not isinstance(value, str):
        return value
    for substring in parse.drop:
        value = value.replace(substring, "")
    return value.replace(parse.decimal, ".") if parse.decimal != "." else value


def _read_as_time(value: Any, reading: ParseDateTime) -> Any:
    """Read one value as a timestamp: as the period it falls in, or as the instant itself.

    A timestamp reaches a column in more than one spelling, and all of them mean the same
    moment: text under a format, a number counting the epoch, or a ``datetime`` that is
    already one. Reading only the text would leave a declaration against either of the
    others recorded, replayed, and doing nothing to any row.

    A value no reading understands is returned exactly as it was, which is what leaves a
    column its leftovers keep mixed unusable and saying so, rather than completed by a rule
    nobody wrote.
    """
    moment = _as_moment(value, reading)
    if moment is None:
        return value
    if reading.every is None:
        # Naive timestamps are read as UTC rather than through the local zone, so the same
        # declaration gives the same number wherever it is replayed.
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.timestamp()
    return _period_label(moment, reading.every)


def _as_moment(value: Any, reading: ParseDateTime) -> datetime | None:
    """Read one value as a moment, whichever of its three spellings it arrived in.

    A ``datetime`` is separated from a bare ``date`` inside the one check, because it is a
    subclass of one and would otherwise be truncated to its midnight by the wrong branch.
    """
    if isinstance(value, date):
        return value if isinstance(value, datetime) else datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        return _as_datetime(value, reading.format)
    return _as_epoch_moment(value, reading.epoch)


def _as_epoch_moment(value: Any, unit: str) -> datetime | None:
    """Read a number as a count of the epoch, or None where the value is not one.

    Booleans are refused before numbers because ``bool`` is a subclass of ``int``: ``True``
    is not one second past the epoch, it is a value this reading has nothing to say about.
    """
    if isinstance(value, bool | np.bool_):
        return None
    if isinstance(value, int | float | np.integer | np.floating):
        return _from_epoch(float(value), unit)
    return None


def _from_epoch(count: float, unit: str) -> datetime | None:
    """Read a count of the epoch in the declared unit, or None where it names no moment.

    A count far enough out to leave the range ``datetime`` can hold is the usual sign that
    the unit was misdeclared -- milliseconds read as seconds land tens of thousands of years
    out -- so it is returned unread rather than raised over, and the column says so by
    staying mixed.
    """
    try:
        return datetime.fromtimestamp(count * EPOCH_SECONDS[unit], tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _as_datetime(value: str, pattern: str | None) -> datetime | None:
    """Read text as a datetime under a declared format, or as ISO 8601 where none is given."""
    try:
        if pattern is not None:
            return datetime.strptime(value, pattern)
        # ``fromisoformat`` learned to read a trailing 'Z' in 3.11; on the 3.10 floor it
        # raises, and a timestamp that has been through JSON is very often spelled that way.
        return datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        return None


def _iso_week(moment: datetime) -> str:
    """Label a moment's ISO week, which belongs to the year holding its Thursday.

    Read off ``isocalendar`` rather than counted, which is what makes a year's first days
    land in the previous year's final week where the calendar says they do.
    """
    year, week, _ = moment.isocalendar()
    return f"{year:04d}-W{week:02d}"


# How each period is spelled. An absolute period leads with its coarsest field and pads
# every one, so the order the labels take in a vocabulary is the order the periods happened
# in. A recurring position is spelled as its number alone, which is read back as the number
# it is and cut into bins like any other ordered reading -- 14:00 is later in the day than
# 10:00, and a label that sorted as text would only accidentally agree. ISO throughout, so
# Monday is 1 and a week is numbered as its year's, rather than a second convention here.
_PERIOD_LABELS: Mapping[str, Callable[[datetime], str]] = MappingProxyType({
    "year": lambda m: f"{m.year:04d}",
    "quarter": lambda m: f"{m.year:04d}-Q{(m.month - 1) // 3 + 1}",
    "month": lambda m: f"{m.year:04d}-{m.month:02d}",
    "week": _iso_week,
    "day": lambda m: f"{m.year:04d}-{m.month:02d}-{m.day:02d}",
    "hour": lambda m: f"{m.year:04d}-{m.month:02d}-{m.day:02d}T{m.hour:02d}",
    "month_of_year": lambda m: f"{m.month:d}",
    "day_of_week": lambda m: f"{m.isoweekday():d}",
    "hour_of_day": lambda m: f"{m.hour:d}",
})


def _period_label(moment: datetime, every: str) -> str:
    """Name the period a moment falls in, as a label that reads the same in any locale.

    Raises
    ------
    ValueError
        When the table has no spelling for a period :class:`~dataeval.types.ParseDateTime`
        accepts -- the two vocabularies having drifted, which is the one way this is reached
        with a period the record already validated.
    """
    spelling = _PERIOD_LABELS.get(every)
    if spelling is None:
        raise ValueError(
            f"ParseDateTime accepts the period {every!r} but this has no spelling for it. "
            f"It spells {', '.join(_PERIOD_LABELS)}.",
        )
    return spelling(moment)


def _remapped(value: Any, remap: Remap) -> Any:
    """Replace one value against a mapping: by name, then by range, then by catch-all.

    In that order because it is the order of decreasing specificity, and a value named
    outright should not be answered by a range that happens to contain it.
    """
    mapping = remap.mapping
    # ``type(...) is type(...)`` alongside equality, because ``True == 1`` and ``1 == 1.0``
    # in Python: a mapping keyed on the number 1 must not answer for a boolean, and a
    # caller who wrote both keys means both.
    for key, replacement in mapping.items():
        if not isinstance(key, tuple) and key is not None and type(key) is type(value) and key == value:
            return replacement
    for key, replacement in mapping.items():
        if isinstance(key, tuple) and _within(value, key):
            return replacement
    return mapping.get(None, value) if None in mapping else value


def _within(value: Any, over: tuple[float | None, float | None]) -> bool:
    """Whether a value falls in a half-open ``[low, high)`` range.

    Half-open to match the convention binning already uses, so a value on a boundary
    belongs to exactly one of two adjacent ranges however they are written. A value that
    does not read as a number is in no range at all: a range is an interval of numbers, and
    text has no place on it.
    """
    number = _as_number(value)
    if number is None:
        return False
    low, high = over
    return (low is None or number >= low) and (high is None or number < high)


def _as_number(value: Any) -> Any:
    """Read the value as a number, or None where it does not read as one.

    Read through ``simplify_type``, the same conversion that
    decides whether a column mixes types, so a numeral is a number whichever way it is
    spelled -- metadata that has been through JSON is all text, and a range written over
    numbers has to reach it.
    """
    simplified = simplify_type(value)
    return None if isinstance(simplified, str) else simplified


# Each correction that reads one value into another, wherever it is declared. Rescale is
# deliberately absent: see :func:`_corrected`.
_VALUE_READERS: Mapping[type, Callable[[Any, Any], Any]] = MappingProxyType({
    Remap: _remapped,
    ParseValue: _parsed_value,
    ParseDateTime: _read_as_time,
})
