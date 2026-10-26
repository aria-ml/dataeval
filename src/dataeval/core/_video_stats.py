"""Per-video statistics: one addressed row per video, from one walk of each video's frames."""

__all__ = []

import math
import warnings
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from typing import Any, NamedTuple, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

import dataeval.core._calculators._video_register  # noqa: F401
from dataeval._log import get_logger
from dataeval._metadata._aggregate import roll_up
from dataeval._metadata._reductions import coverage_for, expressions, identity_of, lookup
from dataeval.core import _ego_stats
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.core._calculators._video_base import ReadingCalculator, VideoCalculator
from dataeval.core._compute_stats import AddressedResult, _produced_stat_names, _reconcile_stats
from dataeval.core._temporal import DEFAULT_REDUNDANCY_RADIUS
from dataeval.core._video_runner import CLOCK, VideoClock, VideoOptions, run_videos, walk
from dataeval.flags import VideoStats
from dataeval.protocols import ArrayLike, MultiobjectTrackingDataset, ProgressCallback, VideoStream
from dataeval.types import Aggregator, SourceIndex

_logger = get_logger(__name__)

_Plan = Sequence[tuple[type[VideoCalculator], VideoStats]]
# Count statistics, declared so an empty or NaN-free column stays int64.
_INTEGER_STATS = frozenset({"n_frames", "longest_run"})
# Label statistics, declared so an empty column stays object and a missing label stays None.
_OBJECT_STATS = frozenset({"motion_mode"})


class VideoRow(NamedTuple):
    """One video's measurements, as a worker hands them back.

    Attributes
    ----------
    scalars : dict[str, list[Any]]
        One finished value per statistic, each wrapped in a one-element list.
    warnings : list[str]
        Every calculator's warnings for this video.
    """

    scalars: dict[str, list[Any]]
    warnings: list[str]


def _measure(stream: VideoStream, calculators: _Plan, options: VideoOptions) -> VideoRow:
    """Walk one video once through fresh calculators and finish every statistic, readings included."""
    if not calculators:
        return VideoRow({}, [])
    built = [(cls(options), flags) for cls, flags in calculators]
    clock = walk(stream, [calc for calc, _ in built], options)
    scalars = _reconcile_stats(
        [calc.compute(flags) for calc, flags in built if not isinstance(calc, ReadingCalculator)],
    )
    readings: dict[str, list[Any]] = {}
    for calc, _ in built:
        if isinstance(calc, ReadingCalculator):
            readings |= calc.readings()
    if readings:
        # Reduced here, in the worker, so a video's per-frame readings are dropped with it.
        scalars |= _roll_readings(readings, clock, calculators)
    return VideoRow(scalars, [w for calc, _ in built for w in calc.warnings])


def _column(name: str, values: list[Any]) -> NDArray[Any]:
    """One output column: int64 for counts, object for labels, float64 with None as NaN otherwise."""
    if name in _INTEGER_STATS:
        return np.asarray(values, dtype=np.int64)
    return np.asarray(values, dtype=object if name in _OBJECT_STATS else np.float64)


def _check_options(stats: VideoStats, options: VideoOptions) -> None:
    """Refuse a request that cannot run, before any video is read."""
    if options.redundancy_radius < 0:
        raise ValueError(f"redundancy_radius must be non-negative; got {options.redundancy_radius}.")
    if not stats & VideoStats.EGO:
        return
    bad = {
        name: value
        for name, value in (("ego_gap", options.ego_gap), ("ego_proc_short_side", options.ego_proc_short_side))
        if value < 1
    }
    if bad:
        raise ValueError(f"Parameters must be positive integers; got {bad}.")
    _ego_stats._check_proc_short_side(options.ego_proc_short_side, "ego_proc_short_side")
    # Imported here, on the caller's thread: a missing extra fails up front, and run_videos
    # caps OpenCV's threads only once cv2 is imported.
    _ego_stats._require_cv2()


def _selected(calculators: _Plan) -> dict[Aggregator, list[str]]:
    """Group the factors the selected reading flags name by the declaration that reduces them."""
    selected: dict[Aggregator, list[str]] = {}
    for cls, flags in calculators:
        if issubclass(cls, ReadingCalculator):
            for flag, (agg, factor) in cls.aggregations.items():
                if flag in flags:
                    selected.setdefault(agg, []).append(factor)
    return selected


def _roll_readings(readings: dict[str, list[Any]], clock: VideoClock, calculators: _Plan) -> dict[str, list[Any]]:
    """Reduce one video's readings to one value each, through the roll-up Metadata uses."""
    selected = _selected(calculators)
    frames = len(next(iter(readings.values())))
    columns = {factor: readings[factor] for factors in selected.values() for factor in factors}
    columns[CLOCK] = [math.nan] * frames if clock.order is None else clock.order.tolist()
    # A column with no values at all has no dtype, which abs_sum cannot read.
    frame = pl.DataFrame(columns).with_columns(pl.col(pl.Null).cast(pl.Float64))
    groups = np.zeros(frames, dtype=np.intp)

    values: dict[str, list[Any]] = {}
    for agg, factors in selected.items():
        temporal = lookup(agg.how).kind == "temporal"
        rolled = roll_up(
            frame,
            groups,
            1,
            expressions(replace(agg, factors=tuple(factors), order_by=CLOCK)),
            empty=identity_of(agg),
            min_coverage=coverage_for(agg),
            order=CLOCK if temporal else None,
        )
        values |= {column.name: column.to_list() for column in rolled.columns}
    return values


def video_stats(
    source: VideoStream | Sequence[VideoStream] | MultiobjectTrackingDataset,
    *,
    stats: VideoStats = VideoStats.NO_EGO,
    redundancy_radius: int = DEFAULT_REDUNDANCY_RADIUS,
    ego_gap: int = 1,
    ego_proc_short_side: int = 360,
    ego_mask: ArrayLike | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AddressedResult:
    """
    Compute statistics that describe each video as a whole, one row per video.

    Every selected statistic is computed in one walk of each video's frames, so a video is
    decoded once however many statistics are asked for. Rows are addressed by item, as
    :func:`image_stats` rows are, so ``md.add_factors(video_stats(ds))`` attaches them to
    :class:`~dataeval.Metadata` at the sequence level.

    Parameters
    ----------
    source : VideoStream, Sequence[VideoStream] or MultiobjectTrackingDataset
        One video's frames (item 0), a sequence of videos, or a tracking dataset whose
        items carry their video first.
    stats : VideoStats, default VideoStats.NO_EGO
        Which statistics to compute. See :class:`~dataeval.flags.VideoStats`.
    redundancy_radius : int, default 4
        Largest Hamming distance, in bits, between neighboring frames' perceptual hashes for the
        later frame to count as redundant, the same default as :class:`~dataeval.quality.Duplicates`.
    ego_gap : int, default 1
        Frames between the two frames of each pair the `EGO` statistics measure motion across,
        as ``gap`` in :func:`ego_stats`.
    ego_proc_short_side : int, default 360
        Short side, in pixels, frames are downscaled to before the `EGO` optical flow, as
        ``proc_short_side`` in :func:`ego_stats`.
    ego_mask : ArrayLike or None, default None
        A 2D (height, width) image of a burned-in overlay such as a HUD, where nonzero marks the
        overlay, as ``mask`` in :func:`ego_stats`. The `EGO` statistics read the camera from the rest
        of the frame. Without it, a dense overlay can read a pan as slower or still, and a light one
        gets pans refused, which skews the `EGO` summaries or leaves them NaN. A mask also drops
        samples within 32 px of the overlay, at `ego_proc_short_side`, and movers are judged against
        what is left, so one covering about a fifth of the frame can be read as the camera. Ignored
        unless `stats` selects an `EGO` statistic.
    progress_callback : ProgressCallback or None, default None
        Called once per finished video with the count so far and the total.

    Returns
    -------
    AddressedResult
        ``stats`` maps each selected statistic to one value per video: float64, int64 for
        counts, and object for ``motion_mode``, with None where it is missing.
        ``source_index`` holds one :class:`~dataeval.types.SourceIndex` per video, in
        item order.

    Raises
    ------
    ImportError
        If `stats` selects an `EGO` statistic and OpenCV is not installed. Install it with the
        ``dataeval[opencv]`` extra.
    ValueError
        If `redundancy_radius` is negative, or if `stats` selects an `EGO` statistic and
        `ego_gap` or `ego_proc_short_side` is not positive or `ego_proc_short_side` is under
        32 px, or if `ego_mask` is not a 2D image or does not fit the frames (see :func:`ego_stats`).

    Warns
    -----
    UserWarning
        If `stats` selects no statistics. One row per video is still returned, with no
        columns.

    See Also
    --------
    :func:`image_stats` : Over :class:`~dataeval.data.SequenceFrames`, per-frame statistics
        to drill down into a video
    :func:`ego_stats` : Per-frame camera ego-motion

    Notes
    -----
    **Timing is only as good as** ``time_s``. Duration and frame rate are read from each
    frame's ``time_s``, never from the container. A loader that synthesizes ``time_s`` from
    the frame index reports one frame per second. A video with any frame lacking ``time_s``,
    or with fewer than two frames, gets NaN for both, and for the two `EGO` variabilities.
    Where ``time_s`` restarts, as at the clip boundaries of stitched video, frames keep stream
    order and each backward step counts as one typical frame interval.

    **Ego-motion.** The `EGO` summaries are the roll-ups :func:`ego_stats` declares, reduced
    exactly as :meth:`~dataeval.Metadata.add_factors` reduces them, so the two agree wherever
    timestamps only increase. A summary is NaN where fewer than a quarter of the video's frames
    recorded a value for its field. The camera kinematics record one only on frames whose
    estimate is trusted.

    **Parallelism.** Videos are measured concurrently, one per worker thread, when
    :func:`~dataeval.config.set_max_processes` allows more than one worker. Each worker
    fetches and decodes its own video, so a dataset whose items share one open decoder is not
    safe to measure this way. Set ``max_processes`` to 1 for such a dataset.

    Examples
    --------
    Two in-memory videos at 25 frames per second, two seconds and one second long:

    >>> from types import SimpleNamespace
    >>> from dataeval.core import video_stats
    >>> def video(n_frames):
    ...     return [SimpleNamespace(pixels=np.zeros((3, 8, 8)), time_s=k / 25) for k in range(n_frames)]
    >>> result = video_stats([video(50), video(25)])
    >>> result["stats"]["n_frames"]
    array([50, 25])
    >>> result["stats"]["video_duration"]
    array([2., 1.])
    >>> result["stats"]["frame_rate"]
    array([25., 25.])
    >>> result["source_index"]
    [SourceIndex(0), SourceIndex(1)]

    Attach the rows to a tracking dataset's metadata, one per sequence:

    >>> metadata.add_factors(video_stats(tracking_dataset))  # doctest: +SKIP
    """
    if not stats:
        warnings.warn(
            "stats requests no statistics, so every row will be returned with no columns. "
            "VideoStats.NONE means this — name the statistics you want, or drop the video_stats call.",
            UserWarning,
            stacklevel=2,
        )

    options = VideoOptions(
        redundancy_radius=redundancy_radius,
        ego_gap=ego_gap,
        ego_proc_short_side=ego_proc_short_side,
        ego_mask=_ego_stats._as_mask(ego_mask, "ego_mask") if stats & VideoStats.EGO else None,
    )
    _check_options(stats, options)
    calculators = cast(_Plan, CalculatorRegistry.get_calculators(stats))
    names = _produced_stat_names(calculators)
    rows = run_videos(source, partial(_measure, calculators=calculators, options=options), progress_callback)

    grouped_warnings: dict[str, list[str]] = {}
    for i, row in enumerate(rows):
        for msg in row.warnings:
            grouped_warnings.setdefault(msg, []).append(str(i))
    for msg, indices in grouped_warnings.items():
        _logger.warning("%s — indices: %s", msg, ", ".join(indices))

    columns = {name: _column(name, [row.scalars[name][0] for row in rows]) for name in sorted(names)}
    return AddressedResult(stats=columns, source_index=[SourceIndex(i) for i in range(len(rows))])
