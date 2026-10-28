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
from dataeval._experimental import warn_experimental_once
from dataeval._log import get_logger
from dataeval._metadata._aggregate import roll_up
from dataeval._metadata._reductions import coverage_for, expressions, identity_of, lookup
from dataeval.core import _ego_stats
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.core._calculators._video_base import ReadingCalculator, VideoCalculator
from dataeval.core._compute_stats import AddressedResult, _produced_stat_names, _reconcile_stats
from dataeval.core._persistence import REPORT_GRID, edge_sharpness
from dataeval.core._temporal import DEFAULT_REDUNDANCY_RADIUS
from dataeval.core._video_runner import CLOCK, VideoClock, VideoOptions, run_videos, walk
from dataeval.flags import VideoStats
from dataeval.protocols import (
    ArrayLike,
    MultiobjectTrackingDataset,
    ProgressCallback,
    SharpnessMeasure,
    VideoStream,
)
from dataeval.types import Aggregator, SourceIndex

_logger = get_logger(__name__)

_Plan = Sequence[tuple[type[VideoCalculator], VideoStats]]
# Count statistics, declared so an empty or NaN-free column stays int64.
_INTEGER_STATS = frozenset({"n_frames", "longest_run"})
# Label statistics, declared so an empty column stays object and a missing label stays None.
_OBJECT_STATS = frozenset({"motion_mode"})
# Map statistics: one 2D map per video, stacked to (n_videos, *REPORT_GRID).
_MAP_STATS = frozenset({"ego_map", "blur_map", "blur_map_raw"})
_BLUR = VideoStats.PERSISTENCE_BLUR | VideoStats.PERSISTENCE_BLUR_RAW


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
    """One output column: int64 for counts, object for labels, stacked maps, float64 with None as NaN otherwise."""
    if name in _MAP_STATS:
        return np.stack(values).astype(np.float64) if values else np.empty((0, *REPORT_GRID))
    if name in _INTEGER_STATS:
        return np.asarray(values, dtype=np.int64)
    return np.asarray(values, dtype=object if name in _OBJECT_STATS else np.float64)


def _check_ego(options: VideoOptions) -> None:
    """Refuse `EGO` settings that cannot run, naming the keyword."""
    bad = {
        name: value
        for name, value in (("ego_gap", options.ego_gap), ("ego_proc_short_side", options.ego_proc_short_side))
        if value < 1
    }
    if bad:
        raise ValueError(f"Parameters must be positive integers; got {bad}.")
    _ego_stats._check_proc_short_side(options.ego_proc_short_side, "ego_proc_short_side")


def _check_persistence(options: VideoOptions) -> None:
    """Refuse `PERSISTENCE` settings that cannot run, naming the keyword."""
    if not (math.isfinite(options.blur_sigma) and options.blur_sigma > 0):
        raise ValueError(f"blur_sigma must be a positive, finite number of cells; got {options.blur_sigma}.")
    if options.persistence_samples < 1:
        raise ValueError(f"persistence_samples must be a positive integer; got {options.persistence_samples}.")
    if options.blur_measure is not None and not callable(options.blur_measure):
        raise TypeError(
            f"blur_measure must be a SharpnessMeasure, a callable; got {type(options.blur_measure).__name__}."
        )


def _check_options(stats: VideoStats, options: VideoOptions) -> None:
    """Refuse a request that cannot run, before any video is read."""
    if options.redundancy_radius < 0:
        raise ValueError(f"redundancy_radius must be non-negative; got {options.redundancy_radius}.")
    if stats & VideoStats.EGO:
        _check_ego(options)
    if stats & VideoStats.PERSISTENCE:
        _check_persistence(options)
    if stats & (VideoStats.EGO | VideoStats.PERSISTENCE):
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


def video_stats(  # noqa: C901
    source: VideoStream | Sequence[VideoStream] | MultiobjectTrackingDataset,
    *,
    stats: VideoStats = VideoStats.NO_EGO,
    redundancy_radius: int = DEFAULT_REDUNDANCY_RADIUS,
    ego_gap: int = 1,
    ego_proc_short_side: int = 360,
    ego_mask: ArrayLike | None = None,
    blur_measure: SharpnessMeasure | None = None,
    blur_sigma: float = 6.0,
    persistence_samples: int = 64,
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
        of the frame. Without it, a dense overlay drags a pan's reading slower, or splits it into two
        motions that are refused, and a light one gets pans refused, which skews the `EGO` summaries
        or leaves them NaN. A mask also drops samples within 32 px of the overlay, at
        `ego_proc_short_side`, and movers are judged against what is left, so one holding most of
        what is left can be read as the camera. Ignored unless `stats` selects an `EGO` statistic.
    blur_measure : SharpnessMeasure or None, default None
        The per-cell sharpness the blur maps compare, see
        :class:`~dataeval.protocols.SharpnessMeasure`. None uses the standard deviation of
        :func:`~dataeval.utils.preprocessing.edge_filter` per cell, the sharpness
        :func:`image_stats` reports. Ignored unless `stats` selects a `PERSISTENCE` blur
        statistic.
    blur_sigma : float, default 6.0
        Width, in grid cells, of the trend ``blur_map`` removes from ``blur_map_raw``. Each map
        is solved on 48 rows, so 6 cells is about an eighth of the frame height. Ignored unless
        `stats` selects a `PERSISTENCE` statistic.
    persistence_samples : int, default 64
        Most start frames the `PERSISTENCE` maps keep per video, spread evenly over its whole
        length. Each start pairs with the frames 10 and 30 frames after it. Ignored unless
        `stats` selects a `PERSISTENCE` statistic.
    progress_callback : ProgressCallback or None, default None
        Called once per finished video with the count so far and the total.

    Returns
    -------
    AddressedResult
        ``stats`` maps each selected statistic to one value per video: float64, int64 for
        counts, and object for ``motion_mode``, with None where it is missing. Each
        `PERSISTENCE` statistic is instead a float64 map per video, shape ``(n_videos, 48, 85)``.
        ``source_index`` holds one :class:`~dataeval.types.SourceIndex` per video, in
        item order.

    Raises
    ------
    ImportError
        If `stats` selects an `EGO` or `PERSISTENCE` statistic and OpenCV is not installed.
        Install it with the ``dataeval[opencv]`` extra.
    ValueError
        If `redundancy_radius` is negative, or if `stats` selects an `EGO` statistic and
        `ego_gap` or `ego_proc_short_side` is not positive or `ego_proc_short_side` is under
        32 px, or if `ego_mask` is not a 2D image or does not fit the frames (see :func:`ego_stats`),
        or if `stats` selects a `PERSISTENCE` statistic and `blur_sigma` is not positive and
        finite or `persistence_samples` is not positive, or if `blur_measure` returns other than
        one finite, non-negative value per cell.
    TypeError
        If `stats` selects a `PERSISTENCE` blur statistic and `blur_measure` is not callable.

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

    **Persistence (experimental).** Selecting any `PERSISTENCE` statistic, directly, through
    the group or through `ALL`, raises :class:`~dataeval.exceptions.ExperimentalWarning` once
    per process. The map shape, the column names and the `blur_measure`, `blur_sigma` and
    `persistence_samples` keywords may change in a later release. Each `PERSISTENCE` statistic
    is a map of 48 by 85 cells per video, read from frame pairs 10 and 30 frames apart while
    the scene moves behind the camera.
    ``ego_map`` is the share of each cell's textured pixels that stayed unchanged while the
    camera moved them: near 1 on a textured burned-in overlay or part of the vehicle, near 0 on
    the scene. It does not tell the two apart, and a flat region fixed to the camera, such as a
    black bar, reads NaN. A frame pair counts only when one camera motion, fitted to what
    changed, explains more of the frame than "nothing moved" does, and a pixel counts only where
    that motion moved it at least 2 pixels on the frame shrunk to a 360-pixel short side (4
    pixels at 720p). Movers before a still camera, even heavy traffic, therefore count for
    nothing, and a camera that moves less than that reads NaN rather than reading its scenery as
    fixed. One mover that fills most of the frame and moves as one can still pass for the
    camera, and then the still rest of the frame reads near 1. ``blur_map_raw`` is the natural
    log of how much each cell softens the same content, so ``log(0.5)`` means half the
    sharpness. Both blur maps are relative to the video's own average, so a lens that is soft
    everywhere alike reads 0. ``blur_map`` subtracts the map's own trend over `blur_sigma`
    cells, which removes blur wider than about `blur_sigma` cells, such as haze across half the
    lens, and part of a defect about that wide. On clean real footage, the raw map of one video
    read large areas as soft that no lens defect caused, so compare raw maps across many videos
    from one camera rather than reading one alone. A loss of sharpness that grows at a steady
    rate, in log, across the frame reads 0 in both maps, because each pair's correction for
    frame-wide sharpness changes removes it too. A cell is NaN where too few frame pairs moved
    it with the camera or too few of its pixels have texture, and the blur maps are also NaN
    where ``ego_map`` is 0.1 or more. ``ego_map`` needs at least 31 frames of camera motion, and
    a textureless video, a still camera or one moving less than 2 pixels a pair reads NaN
    everywhere, with a warning naming the reason. The pair gaps are counted in frames, so they
    span different times at different frame rates. Each video keeps at most
    `persistence_samples` start frames, and at least half that many once it is long enough to
    fill them, spread evenly over its whole length, so memory and the solve stay bounded however
    long the video is. :meth:`~dataeval.Metadata.add_factors` skips the maps, as it skips every
    statistic with more than one value per row.

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
        blur_measure=(edge_sharpness if blur_measure is None else blur_measure) if stats & _BLUR else None,
        blur_sigma=blur_sigma,
        persistence_samples=persistence_samples,
    )
    _check_options(stats, options)
    if stats & VideoStats.PERSISTENCE:
        warn_experimental_once(
            "VideoStats.PERSISTENCE",
            details="The map shape, column names and the blur_measure, blur_sigma and persistence_samples keywords "
            "may change.",
        )
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
