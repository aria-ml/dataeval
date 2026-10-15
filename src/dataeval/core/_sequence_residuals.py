"""Score how far each value of a table sits from its neighbors along an ordered sequence."""

__all__ = []

import bisect
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, TypedDict

import numpy as np
import polars as pl
from numpy.typing import NDArray

REASONS: tuple[str, ...] = ("", "missing value", "missing order", "missing sequence", "short sequence")
_MISSING_VALUE, _MISSING_ORDER, _MISSING_SEQUENCE, _SHORT_SEQUENCE = range(1, 5)
_MIN_ROWS = 3  # a row is told apart from its neighbors only when at least two others can agree
_SHRINK_ROWS = 30  # the weight, in rows, of the pooled spread in each sequence's shrunk spread
_MAD_TO_SIGMA = 1.4826
_RELATIVE_TOLERANCE = 1e-9  # of a monotonic column's interval: rounding left by the detrending
_EPSILON_TOLERANCE = 16  # machine epsilons of a sequence's largest value: rounding at the value's magnitude


class SequenceResidualsResult(TypedDict):
    r"""
    How far each value sits from its neighbors along its sequence, one entry per (row, column).

    Every per-entry array has the same length, ``len(data) * len(columns)``, sorted by ``column`` then ``row``.

    Attributes
    ----------
    columns : tuple[str, ...]
        The scored columns, in the order given.
    monotonic : tuple[bool, ...]
        Per column, whether it was read as never decreasing along a sequence.
    window : int
        How many neighbors on each side a value was compared with.
    reasons : tuple[str, ...]
        Why a row was not scored. Index 0 is ``""``, a scored row.
    row : NDArray[np.int64]
        Position of the row in ``data``.
    column : NDArray[np.int32]
        Index into ``columns``.
    sequence : NDArray[np.int64]
        The row's sequence, numbered in order of first appearance; -1 where the key is missing.
    value : NDArray[np.float64]
        The value as a float, temporal values in their physical units; NaN where missing.
    expected : NDArray[np.float64]
        Where the neighbors put the value, in the column's units; NaN where unscored.
    residual : NDArray[np.float64]
        ``value - expected``; NaN where unscored.
    spread : NDArray[np.float64]
        The sequence's spread shrunk toward the pooled spread, which ``score`` divides by; 1.0 where the score is the
        residual undivided; NaN where unscored.
    score : NDArray[np.float64]
        ``residual / spread``; NaN where unscored.
    reason : NDArray[np.int8]
        Index into ``reasons``.
    out_of_order : NDArray[np.bool\_]
        For a monotonic column, whether the row breaks its sequence's order; False otherwise.
    order_bound : NDArray[np.float64]
        For an out-of-order row, the value it is out of order against; NaN otherwise.
    """

    columns: tuple[str, ...]
    monotonic: tuple[bool, ...]
    window: int
    reasons: tuple[str, ...]
    row: NDArray[np.int64]
    column: NDArray[np.int32]
    sequence: NDArray[np.int64]
    value: NDArray[np.float64]
    expected: NDArray[np.float64]
    residual: NDArray[np.float64]
    spread: NDArray[np.float64]
    score: NDArray[np.float64]
    reason: NDArray[np.int8]
    out_of_order: NDArray[np.bool_]
    order_bound: NDArray[np.float64]


class _Layout(NamedTuple):
    """The rows of one column that can be scored, sorted along their sequences."""

    rows: NDArray[np.intp]  # positions in data, sorted by sequence then order
    starts: NDArray[np.intp]  # per sorted row, the position in `rows` where its sequence starts
    stops: NDArray[np.intp]  # per sorted row, the position in `rows` just past its sequence's end
    sequence: NDArray[np.intp]  # per sorted row, its sequence's index in `begin` and `end`
    begin: NDArray[np.intp]  # per sequence, the position in `rows` where it starts
    end: NDArray[np.intp]  # per sequence, the position in `rows` just past its end


class _Column(NamedTuple):
    """One column's entries, aligned with the rows of data."""

    expected: NDArray[np.float64]
    residual: NDArray[np.float64]
    spread: NDArray[np.float64]
    score: NDArray[np.float64]
    reason: NDArray[np.int8]
    out_of_order: NDArray[np.bool_]
    order_bound: NDArray[np.float64]


def _is_quantity(dtype: pl.DataType) -> bool:
    """Whether a column holds numbers or times, the values a residual can be taken of."""
    return dtype.is_numeric() or dtype.is_temporal()


def _check_arguments(columns: Sequence[str], keys: Sequence[str], monotonic: Sequence[str], window: int) -> None:
    """Refuse a window below 1, no columns, no sequence key, and a monotonic name that is not scored."""
    if window < 1:
        raise ValueError(f"window must be at least 1, not {window}.")
    if not columns:
        raise ValueError("columns names no column to score.")
    if not keys:
        raise ValueError("sequence names no column to identify a sequence.")
    stray = sorted(set(monotonic) - set(columns))
    if stray:
        raise ValueError(f"monotonic names {stray}, which columns does not score.")


def _check_columns(frame: pl.DataFrame, columns: Sequence[str], keys: Sequence[str], order: str) -> None:
    """Refuse names that are not columns of data, and scored or ordering columns that hold neither numbers nor times."""
    unknown = sorted({*columns, *keys, order} - set(frame.columns))
    if unknown:
        raise ValueError(f"{unknown} not a column of data, which has {frame.columns}.")
    unreadable = sorted(name for name in {*columns, order} if not _is_quantity(frame.schema[name]))
    if unreadable:
        raise ValueError(f"{unreadable} hold neither numbers nor times, so they have no residual or order.")


def _floats(series: pl.Series) -> NDArray[np.float64]:
    """Return a numeric or temporal column as floats in its physical units, NaN where missing or not finite."""
    values = series.to_physical().cast(pl.Float64).to_numpy().astype(np.float64)
    return np.where(np.isfinite(values), values, np.nan)


def _sequence_codes(frame: pl.DataFrame, keys: Sequence[str]) -> NDArray[np.int64]:
    """Return each row's sequence number, in order of first appearance; -1 where any key is missing."""
    keyed = frame.select(keys).with_row_index("__row")
    codes = keyed.drop("__row").drop_nulls().unique(maintain_order=True).with_row_index("__code")
    joined = keyed.join(codes, on=list(keys), how="left").sort("__row")
    return joined["__code"].fill_null(-1).cast(pl.Int64).to_numpy()


def _layout(present: NDArray[np.bool_], sequence: NDArray[np.int64], order: NDArray[np.float64]) -> _Layout:
    """Sort the rows that can be scored along their sequences; equal orders keep their row order."""
    rows = np.flatnonzero(present)
    rows = rows[np.lexsort((order[rows], sequence[rows]))]
    codes = sequence[rows]
    edges = np.flatnonzero(np.diff(codes)) + 1
    # No scorable row means no sequence, not one empty sequence at position 0.
    begin = (np.r_[0, edges] if rows.size else edges).astype(np.intp)
    end = (np.r_[edges, rows.size] if rows.size else edges).astype(np.intp)
    lengths = end - begin
    sequences = np.repeat(np.arange(begin.size, dtype=np.intp), lengths)
    return _Layout(rows, np.repeat(begin, lengths), np.repeat(end, lengths), sequences, begin, end)


def _medians(values: NDArray[np.float64], group: NDArray[np.intp], size: int) -> NDArray[np.float64]:
    """Return ``np.median`` of the values in each group ``0 .. size - 1``; NaN for a group with none."""
    lengths = np.bincount(group, minlength=size)
    ranked = np.append(values[np.lexsort((values, group))], np.nan)  # the NaN is an empty group's middle
    first = np.cumsum(lengths) - lengths
    low = np.where(lengths > 0, first + (lengths - 1) // 2, -1)
    high = np.where(lengths > 0, first + lengths // 2, -1)
    median = np.where(low == high, ranked[low], (ranked[low] + ranked[high]) / 2)
    return np.where(np.bincount(group[np.isnan(values)], minlength=size) > 0, np.nan, median)  # as np.median


def _side(values: NDArray[np.float64], layout: _Layout, offsets: NDArray[np.intp]) -> NDArray[np.float64]:
    """Gather each row's neighbors at `offsets` within its own sequence, NaN past the sequence's ends."""
    at = np.arange(values.size)[:, None] + offsets[None, :]
    inside = (at >= layout.starts[:, None]) & (at < layout.stops[:, None])
    return np.where(inside, values[np.clip(at, 0, values.size - 1)], np.nan)


def _residuals(
    values: NDArray[np.float64], order: NDArray[np.float64], layout: _Layout, window: int
) -> NDArray[np.float64]:
    """Return each sorted row's residual against the nearest of its candidates."""
    before = -np.arange(window, 0, -1)
    after = np.arange(1, window + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an empty side is NaN on purpose
        left = np.nanmedian(_side(values, layout, before), axis=1)
        right = np.nanmedian(_side(values, layout, after), axis=1)
        left_at = np.nanmedian(_side(order, layout, before), axis=1)
        right_at = np.nanmedian(_side(order, layout, after), axis=1)
    span = right_at - left_at
    line = left + (right - left) * (order - left_at) / np.where(span > 0, span, np.nan)
    # Candidates: the left median, the right median, the line between them, the end line. Ties go to the earlier one.
    candidates = values[:, None] - np.column_stack([left, right, line, _end_lines(values, order, layout, window)])
    pick = np.argmin(np.where(np.isnan(candidates), np.inf, np.abs(candidates)), axis=1)
    return candidates[np.arange(values.size), pick]


def _half_medians(
    source: NDArray[np.float64], at: NDArray[np.intp], near: NDArray[np.bool_], far: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the medians of `source` over each row's near and far half of the positions `at`; NaN for an empty one."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an empty half is NaN on purpose
        taken = source[at]
        return np.nanmedian(np.where(near, taken, np.nan), axis=1), np.nanmedian(np.where(far, taken, np.nan), axis=1)


def _end_lines(
    values: NDArray[np.float64], order: NDArray[np.float64], layout: _Layout, window: int
) -> NDArray[np.float64]:
    """Return, for the first and last row of each sequence, the line through the two halves of its one side."""
    # An end row has no side to interpolate across, and its one side's median lags any trend; this line follows it.
    line = np.full(values.size, np.nan)
    size = np.minimum(window, layout.end - layout.begin - 1)[:, None]  # the rows on an end row's one side
    column = np.arange(window)[None, :]  # nearest first
    near, far = column < size // 2, (column >= size // 2) & (column < size)
    for row, step in ((layout.begin, 1), (layout.end - 1, -1)):
        at = np.clip(row[:, None] + step * (column + 1), 0, max(values.size - 1, 0))
        near_at, far_at = _half_medians(order, at, near, far)
        near_value, far_value = _half_medians(values, at, near, far)
        span = far_at - near_at
        line[row] = near_value + (far_value - near_value) * (order[row] - near_at) / np.where(span != 0, span, np.nan)
    return line


def _intervals(values: NDArray[np.float64], order: NDArray[np.float64], layout: _Layout) -> NDArray[np.float64]:
    """Return each sorted row's sequence median change per unit of order, over steps of rising order; 0 without one."""
    step = np.diff(order)
    rising = (step > 0) & (layout.sequence[1:] == layout.sequence[:-1])
    group = layout.sequence[1:][rising]
    medians = _medians(np.diff(values)[rising] / step[rising], group, layout.begin.size)
    counts = np.bincount(group, minlength=layout.begin.size)
    return np.where(counts > 0, medians, 0.0)[layout.sequence]


def _tolerances(
    values: NDArray[np.float64], intervals: NDArray[np.float64], layout: _Layout, integral: bool
) -> NDArray[np.float64]:
    """Return each sorted row's residual up to which a monotonic column's deviation is rounding or its own tick."""
    largest = np.maximum.reduceat(np.abs(values), layout.begin)[layout.sequence]
    floor = np.maximum(_RELATIVE_TOLERANCE * np.abs(intervals), 1.0 if integral else 0.0)
    return np.maximum(floor, _EPSILON_TOLERANCE * np.finfo(np.float64).eps * largest)


def _off_run(values: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Mark the rows not on one longest non-decreasing run, found by patience sorting with ``bisect_right``."""
    tails: list[float] = []
    ends: list[int] = []
    before = [-1] * values.size
    for position, value in enumerate(values.tolist()):
        slot = bisect.bisect_right(tails, value)
        before[position] = ends[slot - 1] if slot else -1
        if slot == len(tails):
            tails.append(value)
            ends.append(position)
        else:
            tails[slot], ends[slot] = value, position
    off = np.ones(values.size, dtype=np.bool_)
    position = ends[-1] if ends else -1
    while position >= 0:
        off[position] = False
        position = before[position]
    return off


def _run_bounds(values: NDArray[np.float64], off: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Return, for each row off the run, the run value it is out of order against."""
    run = np.flatnonzero(~off)
    bound = np.full(values.size, np.nan)
    for position in np.flatnonzero(off).tolist():
        later = run[run > position]
        earlier = run[run < position]
        # The next run value when the row is above it, else the previous one. A row before the whole run and not
        # above its first value would extend the longest run, so one with no earlier run value is always above.
        if later.size and values[position] > values[later[0]]:
            bound[position] = values[later[0]]
        else:
            bound[position] = values[earlier[-1]]
    return bound


def _out_of_order(
    values: NDArray[np.float64], residual: NDArray[np.float64], window: int
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Return the rows of one sequence that break a monotonic column's order, and what each breaks it against."""
    # The fewest rows to remove is found forwards and on the reversed, negated sequence. The set whose rows sit
    # further from their neighbors is kept, and a tie keeps the forward one.
    forward = _off_run(values)
    backward = _off_run(-values[::-1])[::-1]
    off = backward if np.abs(residual[backward]).sum() > np.abs(residual[forward]).sum() else forward
    # A set of at most `window` rows is a glitch and is returned whole. A larger one is a reset, and only the rows
    # below the row before them are returned.
    if off.sum() <= window:
        return off, _run_bounds(values, off)
    descent = np.r_[False, values[1:] < values[:-1]]
    return descent, np.where(descent, np.r_[np.nan, values[:-1]], np.nan)


def _spread(residual: NDArray[np.float64]) -> float:
    """Robust standard deviation: 1.4826 times the median absolute deviation about the median."""
    return float(_MAD_TO_SIGMA * np.median(np.abs(residual - np.median(residual))))


def _shrunk(residual: NDArray[np.float64], layout: _Layout) -> NDArray[np.float64]:
    """Return each sorted row's sequence spread, shrunk toward the pooled spread with a weight of 30 rows."""
    pooled = _spread(residual) if residual.size else 0.0
    if pooled == 0:  # every spread is 1.0, so the column's scores are its residuals undivided
        return np.ones(residual.size)
    count = layout.begin.size
    deviation = np.abs(residual - _medians(residual, layout.sequence, count)[layout.sequence])
    own = _MAD_TO_SIGMA * _medians(deviation, layout.sequence, count)
    # Squared with pow, as Python squares the pooled spread; numpy's ``own**2`` multiplies, which can round 1 ulp apart.
    squared = np.power(own, np.full_like(own, 2.0))
    n = layout.end - layout.begin
    return np.sqrt((n * squared + _SHRINK_ROWS * pooled**2) / (n + _SHRINK_ROWS))[layout.sequence]


def _reasons(value: NDArray[np.float64], order: NDArray[np.float64], sequence: NDArray[np.int64]) -> NDArray[np.int8]:
    """Return the first reason each row cannot be scored, before short sequences are known."""
    reason = np.zeros(value.size, dtype=np.int8)
    reason[np.isnan(value)] = _MISSING_VALUE
    reason[sequence < 0] = _MISSING_SEQUENCE
    reason[np.isnan(order)] = _MISSING_ORDER
    return reason


def _drop_short(reason: NDArray[np.int8], sequence: NDArray[np.int64]) -> None:
    """Mark the rows of sequences with fewer than three scorable rows as ``short sequence``."""
    present = reason == 0
    counts = np.bincount(sequence[present])
    short = np.zeros_like(present)
    short[present] = counts[sequence[present]] < _MIN_ROWS
    reason[short] = _SHORT_SEQUENCE


def _score_column(
    value: NDArray[np.float64],
    order: NDArray[np.float64],
    sequence: NDArray[np.int64],
    monotonic: bool,
    integral: bool,
    window: int,
) -> _Column:
    """Score one column: residuals, spreads and scores for its scorable rows, and its order breaks."""
    height = value.size
    reason = _reasons(value, order, sequence)
    _drop_short(reason, sequence)
    layout = _layout(reason == 0, sequence, order)
    values, positions = value[layout.rows], order[layout.rows]
    intervals = np.zeros(values.size)
    if monotonic:  # detrended by each sequence's median interval per unit of order
        intervals = _intervals(values, positions, layout)
        values = values - positions * intervals
    residual = _residuals(values, positions, layout, window)
    breaks = np.zeros(height, dtype=np.bool_)
    bound = np.full(height, np.nan)
    if monotonic:
        _settle(value[layout.rows], residual, intervals, layout, integral, window, breaks, bound)
    spread = _shrunk(residual, layout)
    column = _Column(
        np.full(height, np.nan),
        np.full(height, np.nan),
        np.full(height, np.nan),
        np.full(height, np.nan),
        reason,
        breaks,
        bound,
    )
    column.expected[layout.rows] = value[layout.rows] - residual
    column.residual[layout.rows] = residual
    column.spread[layout.rows] = spread
    column.score[layout.rows] = residual / spread
    return column


def _settle(
    raw: NDArray[np.float64],
    residual: NDArray[np.float64],
    intervals: NDArray[np.float64],
    layout: _Layout,
    integral: bool,
    window: int,
    breaks: NDArray[np.bool_],
    bound: NDArray[np.float64],
) -> None:
    """For a monotonic column, zero the residuals within tolerance and mark each sequence's order breaks."""
    residual[np.abs(residual) <= _tolerances(raw, intervals, layout, integral)] = 0.0
    # A sequence that never goes down breaks no order, so only the ones that do are searched.
    down = np.unique(layout.sequence[1:][(raw[1:] < raw[:-1]) & (layout.sequence[1:] == layout.sequence[:-1])])
    # ponytail: a Python loop per sequence that goes down, monotonic columns only; find the longest non-decreasing
    # runs of every sequence at once if many sequences break order.
    for start, stop in zip(layout.begin[down].tolist(), layout.end[down].tolist(), strict=True):
        part = slice(start, stop)
        off, against = _out_of_order(raw[part], residual[part], window)
        breaks[layout.rows[part]] = off
        bound[layout.rows[part]] = against


def sequence_residuals(
    data: pl.DataFrame | Mapping[str, Sequence[Any] | NDArray[Any]],
    columns: Sequence[str],
    *,
    sequence: str | Sequence[str],
    order: str,
    monotonic: Sequence[str] = (),
    window: int = 5,
) -> SequenceResidualsResult:
    """
    Score how far each value sits from its neighbors along an ordered sequence.

    Some wrong values are ordinary over the whole table and only stand out against the rows next to them: a frame's
    timestamp three hours off, a GPS fix that jumps for one frame, a box whose size doubles for one frame of its track.
    This sorts each sequence by ``order``, compares every value with its neighbors on each side, and reports how far it
    sits from where they put it. It never decides what is an outlier; an evaluator such as
    :class:`~dataeval.quality.Outliers` applies thresholds to these numbers.

    Parameters
    ----------
    data : pl.DataFrame or Mapping[str, Sequence or NDArray]
        The table. A mapping is read as a :class:`polars.DataFrame`.
    columns : Sequence[str]
        The columns to score, each numeric or temporal. A temporal column is read in its physical units, such as
        microseconds for ``Datetime("us")``.
    sequence : str or Sequence[str]
        The columns whose values identify one sequence, such as ``"item_index"`` for the frames of a video or
        ``("item_index", "track_id")`` for the detections of a track. A row with a missing key is not scored.
    order : str
        The numeric or temporal column that orders the rows of a sequence, such as ``"unit_index"``. Rows with equal
        order keep their row order.
    monotonic : Sequence[str], default ()
        Columns that should never decrease along a sequence, such as a timestamp. Equal consecutive values are
        allowed; a value below an earlier one is out of order. Each must also be in ``columns``.
    window : int, default 5
        How many neighbors on each side a value is compared with.

    Returns
    -------
    SequenceResidualsResult
        One entry per (row, column): the value, where its neighbors put it, the residual, the spread it was divided
        by, the score, why a row was not scored, and for a monotonic column whether the row is out of order.

    Raises
    ------
    ValueError
        When ``window`` is below 1; ``columns`` is empty; ``sequence`` is empty; a name in ``columns``, ``sequence`` or
        ``order`` is not a column of ``data``; a column of ``columns``, or ``order``, holds neither numbers nor times;
        or ``monotonic`` names a column not in ``columns``.

    Notes
    -----
    **The expected value.** Each row is compared with up to ``window`` rows before it and up to ``window`` rows after
    it in its sequence. The candidates for where the row should sit are:

    - the median of the rows before it;
    - the median of the rows after it;
    - the line through those two medians, each placed at its side's median ``order``, read at the row's ``order``;
    - for the first and last row of a sequence, which have one side, the line through the medians of that side's near
      half and far half.

    The expected value is the candidate nearest the row's value, ties going to the earlier candidate in that list.
    A scene cut matches one side. A fade, a slow drift or a change of frame rate matches the line. A trend at a
    sequence's end matches the end line. A glitch matches none of them, so its residual stays large.

    **Why the line and not each side's trend.** Extrapolating each side's own slope into the row needs a slope from a
    few noisy steps, and projected several rows that error is as large as the glitches the check looks for: in
    measurements a single spike of 8 standard deviations was caught in 62 to 100 % of runs, against 100 % for the
    line on most shapes, and a clock rounded to milliseconds flagged 15 % of clean rows. The line through the two side
    medians uses no slope estimated from steps.

    **The scale.** A sequence's spread is 1.4826 times the median absolute deviation, about the median, of its
    residuals. It is shrunk toward the spread of every residual of the column, with a weight of 30 rows, so a short
    track is judged mostly by the pooled spread and a quiet video is not judged by a noisy one's jitter. The score is
    the residual divided by that spread. When the pooled spread is 0, as on an exactly regular clock, the score is the
    residual itself and ``spread`` is 1.0.

    **Monotonic columns.** Before the comparison, each sequence is detrended by its median change per unit of
    ``order``, so a timestamp's steady rise, a varying frame rate and a dropped frame do not read as departures. A
    residual no larger than the column's own rounding is set to 0: one physical unit for an integer or temporal column,
    1e-9 of the interval, or 16 machine epsilons of the sequence's largest value, whichever is largest. Order is
    checked separately. The fewest rows to remove for the sequence to be non-decreasing are found from both ends, and
    the set whose rows sit further from their neighbors is kept. A set of at most ``window`` rows is marked whole, so
    one bad stamp marks only itself. A larger set is a reset, and only the rows below the row before them are marked:
    the frame where the clock went back. Only the columns in ``monotonic`` are read this way, so pass every timestamp
    there: one left out is scored as an ordinary column, neither detrended nor checked for order.

    **Rows not scored.** A row with a missing ``order``, a missing ``sequence`` key, or a missing, NaN or infinite
    value is not scored and is no row's neighbor. A sequence with fewer than 3 such rows is not scored.

    **Limits.** A run of more than ``window`` departing rows (more than 5 by default) is read as a segment, not a
    glitch; a run of exactly ``window`` rows can still flag a row in its middle. On a steep ramp, a glitch in the
    ramp's own direction must exceed the ramp's rise over about three rows. On a steep trend, a glitch can also flag
    the clean rows next to it, because it shifts the median of each nearby row's side. A clock that runs backwards
    throughout marks every row but the first.

    Examples
    --------
    One frame is far brighter than the frames around it:

    >>> from dataeval.core import sequence_residuals
    >>> brightness = [10.0, 11.0, 10.5, 11.2, 10.8, 30.0, 11.1, 10.6, 11.3, 10.9]
    >>> result = sequence_residuals(
    ...     {"brightness": brightness, "video": [0] * 10, "frame": list(range(10))},
    ...     ["brightness"],
    ...     sequence="video",
    ...     order="frame",
    ... )
    >>> int(np.argmax(result["score"]))
    5
    >>> float(result["expected"][5]), float(result["residual"][5])
    (11.0, 19.0)

    A timestamp out of order is marked, with the value it is out of order against:

    >>> stamps = [0.0, 0.5, 1.0, 1.5, 9.0, 2.5, 3.0, 3.5]
    >>> result = sequence_residuals(
    ...     {"time_s": stamps, "video": [0] * 8, "frame": list(range(8))},
    ...     ["time_s"],
    ...     sequence="video",
    ...     order="frame",
    ...     monotonic=["time_s"],
    ... )
    >>> np.flatnonzero(result["out_of_order"]).tolist()
    [4]
    >>> float(result["order_bound"][4])
    2.5
    """
    frame = data if isinstance(data, pl.DataFrame) else pl.DataFrame(dict(data))
    keys = [sequence] if isinstance(sequence, str) else list(sequence)
    names = list(columns)
    _check_arguments(names, keys, monotonic, window)
    _check_columns(frame, names, keys, order)
    codes = _sequence_codes(frame, keys)
    positions = _floats(frame[order])
    values = [_floats(frame[name]) for name in names]
    scored = [
        _score_column(
            value,
            positions,
            codes,
            name in monotonic,
            frame.schema[name].is_integer() or frame.schema[name].is_temporal(),
            window,
        )
        for name, value in zip(names, values, strict=True)
    ]
    height = frame.height
    return SequenceResidualsResult(
        columns=tuple(names),
        monotonic=tuple(name in monotonic for name in names),
        window=window,
        reasons=REASONS,
        row=np.tile(np.arange(height, dtype=np.int64), len(names)),
        column=np.repeat(np.arange(len(names), dtype=np.int32), height),
        sequence=np.tile(codes, len(names)),
        value=np.concatenate(values),
        expected=np.concatenate([part.expected for part in scored]),
        residual=np.concatenate([part.residual for part in scored]),
        spread=np.concatenate([part.spread for part in scored]),
        score=np.concatenate([part.score for part in scored]),
        reason=np.concatenate([part.reason for part in scored]),
        out_of_order=np.concatenate([part.out_of_order for part in scored]),
        order_bound=np.concatenate([part.order_bound for part in scored]),
    )
