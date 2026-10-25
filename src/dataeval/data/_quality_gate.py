"""Keeping the frames whose value on one ``image_stats`` statistic lies within declared bounds."""

__all__ = []

import math
import numbers
from collections.abc import Iterable, Iterator
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval.data._selectors import FrameCandidate, FrameInput, FrameSelector, FrameVerdict, SequenceInfo
from dataeval.flags import ImageStats

_logger = get_logger(__name__)

_PROBE = np.linspace(0.0, 1.0, 3 * 16 * 16).reshape(3, 16, 16)
"""The image a statistic is read off once, to learn whether its reading is one number."""


def _single_statistics() -> dict[str, tuple[type[Any], ImageStats]]:
    """Each statistic one :class:`~dataeval.flags.ImageStats` member produces alone, with its calculator.

    Read off the calculators' declared names, so nothing is computed.
    """
    from dataeval.core._calculators._registry import CalculatorRegistry

    statistics: dict[str, tuple[type[Any], ImageStats]] = {}
    for flag in ImageStats:
        calculators = CalculatorRegistry.get_calculators(flag)
        if len(calculators) == 1 and len(names := calculators[0][0].stat_names(flag)) == 1:
            statistics[names.pop()] = (calculators[0][0], flag)
    return statistics


def _reads_one_number(stat: str, calculator: tuple[type[Any], ImageStats]) -> bool:
    """Whether ``calculator`` reads the probe image as one real number, NaN included.

    A calculator that raises RuntimeError on the probe is unsupported: it may depend on an
    unavailable model. Other failures propagate so calculator bugs remain visible.
    """
    from dataeval.core._image_stats import _collect_calculator_stats

    try:
        stats, _, _ = _collect_calculator_stats([calculator], _PROBE, None)
    except RuntimeError:
        _logger.debug("QualityGate: %r could not be read off the probe image; it is unsupported.", stat, exc_info=True)
        return False
    values = stats[0].get(stat, [])
    if len(stats[0]) != 1 or len(values) != 1:
        return False
    value = values[0]
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


@cache
def _numeric_scalar_stats() -> tuple[str, ...]:
    """Every statistic :class:`QualityGate` accepts, sorted. Probes every calculator, so built only to refuse."""
    return tuple(sorted(name for name, calc in _single_statistics().items() if _reads_one_number(name, calc)))


class QualityGate(FrameSelector):
    """Keep the frames whose value on one ``image_stats`` statistic lies within declared bounds.

    A filter under a declared quality policy. It does not make a dataset better: low quality may
    itself be an operational condition, so what it drops is a choice the caller states, not a
    defect it finds. Values are in the units :func:`~dataeval.core.image_stats` reports for the
    statistic, with its default ``normalize_pixel_values=False``.

    An absolute bound states a minimum (or maximum) acceptable quality, decided one frame at a
    time. A quantile bound is relative to each video: ``min_quantile=0.1`` drops about the bottom
    tenth of *every* video, however good that video is. One side may be absolute and the other a
    quantile.

    Parameters
    ----------
    stat : str
        One numeric scalar ``image_stats`` statistic, such as ``"sharpness"``, ``"brightness"``,
        ``"contrast"`` or ``"darkness"``. Several statistics mean several gates in a
        :class:`~dataeval.data.Chain`. Gates with absolute bounds stream and compose freely, but a
        chain walks each sequence at most twice, so at most one of its gates may take a quantile
        bound.
    min : float or None, default None
        Keep frames whose value is at least this.
    max : float or None, default None
        Keep frames whose value is at most this.
    min_quantile : float or None, default None
        Drop frames strictly below this quantile of the video's finite values.
    max_quantile : float or None, default None
        Drop frames strictly above this quantile of the video's finite values.

    Raises
    ------
    ValueError
        If ``stat`` is not a numeric scalar statistic, if no bound is given, if a side has both an
        absolute and a quantile bound, if a quantile is outside ``[0, 1]``, if ``min`` or ``max`` is
        NaN, or if ``min > max`` or ``min_quantile > max_quantile``. An infinite bound is allowed,
        and bounds nothing.

    See Also
    --------
    :func:`~dataeval.core.image_stats` : The statistics and their units

    Notes
    -----
    Bounds are inclusive, and a frame's reading is rounded to float32 as ``image_stats`` reports
    it, so a bound copied from ``image_stats`` output keeps the frame it was read from. A quantile
    cutoff is an observed value (``numpy``'s ``method="inverted_cdf"``), and a frame equal to it is
    kept, so a video whose values are all equal drops nothing. With quantile bounds the gate is
    ``two_pass``: it buffers one float and one position per frame, never pixels.

    A frame whose value is NaN is kept, marked :attr:`~dataeval.data.FrameVerdict.unmeasured`,
    and never enters a quantile. A video with no finite value has no quantile cutoff to take, so
    a quantile bound keeps all of it, each frame marked unmeasured.

    Kept frames stand only for themselves (:attr:`~dataeval.data.FrameSelector.excludes`), and
    no factor is recorded: the statistic belongs to ``image_stats``.

    Examples
    --------
    >>> from dataeval.data import Chain, QualityGate, Redundancy, SequenceFrames
    >>> gate = QualityGate("sharpness", min_quantile=0.1)
    >>> gate.two_pass
    True
    >>> frames = SequenceFrames(mot_dataset, Chain(gate, Redundancy()))  # doctest: +SKIP

    An absolute floor, decided one frame at a time:

    >>> QualityGate("brightness", min=20.0).two_pass
    False
    """

    needs: FrameInput = FrameInput.PIXELS
    excludes = True

    def __init__(  # noqa: C901
        self,
        stat: str,
        *,
        min: float | None = None,  # noqa: A002
        max: float | None = None,  # noqa: A002
        min_quantile: float | None = None,
        max_quantile: float | None = None,
    ) -> None:
        if min is None and max is None and min_quantile is None and max_quantile is None:
            raise ValueError("QualityGate: give at least one bound: min, max, min_quantile or max_quantile.")
        if min is not None and min_quantile is not None:
            raise ValueError("QualityGate: give min or min_quantile, not both.")
        if max is not None and max_quantile is not None:
            raise ValueError("QualityGate: give max or max_quantile, not both.")
        for name, quantile in (("min_quantile", min_quantile), ("max_quantile", max_quantile)):
            if quantile is not None and not 0.0 <= quantile <= 1.0:  # `not` rather than `<`, so NaN is caught
                raise ValueError(f"QualityGate: {name} must lie in [0, 1]; got {quantile}.")
        for name, bound in (("min", min), ("max", max)):
            # NaN fails every comparison, so it would drop every measured frame; infinity is no bound on that side.
            if bound is not None and math.isnan(bound):
                raise ValueError(f"QualityGate: {name} must be a number, not NaN; use None or inf for no bound.")
        if min is not None and max is not None and min > max:
            raise ValueError(f"QualityGate: min must not exceed max; got min={min}, max={max}.")
        if min_quantile is not None and max_quantile is not None and min_quantile > max_quantile:
            raise ValueError(
                f"QualityGate: min_quantile must not exceed max_quantile; "
                f"got min_quantile={min_quantile}, max_quantile={max_quantile}."
            )

        calculator = _single_statistics().get(stat)
        if calculator is None or not _reads_one_number(stat, calculator):
            raise ValueError(
                f"QualityGate: stat must name a numeric scalar image_stats statistic, one of "
                f"{', '.join(_numeric_scalar_stats())}; got {stat!r}."
            )
        self.stat: str = stat
        self.min: float | None = None if min is None else float(min)
        self.max: float | None = None if max is None else float(max)
        self.min_quantile: float | None = None if min_quantile is None else float(min_quantile)
        self.max_quantile: float | None = None if max_quantile is None else float(max_quantile)
        self.two_pass: bool = min_quantile is not None or max_quantile is not None
        self._calculator = calculator

    def _measure(self, pixels: NDArray[Any]) -> float:
        """Return one frame's value, as ``image_stats`` reports it for that frame."""
        from dataeval.core._image_stats import _collect_calculator_stats

        stats, _, _ = _collect_calculator_stats([self._calculator], pixels, None)
        value = stats[0][self.stat][0]
        # image_stats reports a floating statistic as float32, so a bound read off its output meets
        # a frame's own reading exactly only when that reading is rounded the same way.
        return float(np.float32(value)) if isinstance(value, float | np.floating) else float(value)

    def _reading(self, frame: FrameCandidate) -> tuple[SequenceInfo, int, float]:
        return frame.sequence, frame.position, self._measure(frame.pixels)

    def select(self, frames: Iterator[FrameCandidate]) -> Iterator[FrameVerdict]:  # noqa: C901
        """Keep each frame whose value lies within the bounds, and each whose value is NaN, marked."""
        readings: Iterable[tuple[SequenceInfo, int, float]] = map(self._reading, frames)
        low, high = self.min, self.max
        no_cutoff = False
        if self.two_pass:
            readings = list(readings)  # positions and floats, never pixels
            values = np.array([value for _, _, value in readings], dtype=np.float64)
            finite = values[np.isfinite(values)]
            # With no finite value there is no cutoff to take: the quantile side is left unbounded,
            # and every frame kept is kept unjudged, so it is marked.
            no_cutoff = not finite.size
            if finite.size and self.min_quantile is not None:
                low = float(np.quantile(finite, self.min_quantile, method="inverted_cdf"))
            if finite.size and self.max_quantile is not None:
                high = float(np.quantile(finite, self.max_quantile, method="inverted_cdf"))

        sequence: SequenceInfo | None = None
        seen = unmeasured = 0
        for reading in readings:
            sequence, position, value = reading
            seen += 1
            if math.isnan(value):
                # No reading is no evidence of low quality, so the frame is kept rather than
                # dropped on an assumption, and marked as kept unmeasured.
                unmeasured += 1
                yield FrameVerdict(position, unmeasured=True)
            elif (low is None or value >= low) and (high is None or value <= high):
                unmeasured += no_cutoff
                yield FrameVerdict(position, unmeasured=no_cutoff)
        if unmeasured and sequence is not None:
            _logger.info(
                "QualityGate: %d of %d frame(s) in sequence %d could not be judged on %s and were kept unmeasured.",
                unmeasured,
                seen,
                sequence.index,
                self.stat,
            )
