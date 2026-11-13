"""Per-detection steps: how far, how fast, how much it scaled and overlapped since its track was last seen."""

__all__ = []

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from dataeval._experimental import experimental
from dataeval._metadata._structurers._tracking import MOTStructurer
from dataeval.core._compute_stats import AddressedResult
from dataeval.protocols import MultiobjectTrackingDataset
from dataeval.types import MeasurementSpec, SourceIndex
from dataeval.types._target import detection_count, refuse_repeated_track_ids
from dataeval.utils._array import as_numpy
from dataeval.utils.data import validate_dataset

# A row's status is the first of these that applies, so a code is also its rank.
_STATUS = ("untracked", "first_observation", "gap_exceeds_limit", "invalid_box", "geometry_change", "ok")
_UNTRACKED, _FIRST, _GAP, _INVALID, _GEOMETRY, _OK = range(len(_STATUS))
_NAMES = np.array(_STATUS, dtype=object)

_FACTORS = (
    "step_frame_delta",
    "step_elapsed_time_s",
    "step_displacement_norm",
    "step_speed_norm",
    "step_speed_norm_per_s",
    "step_iou_with_previous",
    "step_log_scale_change",
    "step_log_scale_rate",
)
_DETAILS = (
    "step_status",
    "step_dx_px",
    "step_dy_px",
    "step_previous_target_index",
    "step_frame_key",
    "step_previous_frame_key",
)


# (units, normalization) of each factor. The algorithm and version are shared: one change to how a step is measured
# changes every factor, and a result is comparable with another only where they agree.
_UNITS = {
    "step_frame_delta": ("frames", None),
    "step_elapsed_time_s": ("seconds", None),
    "step_displacement_norm": ("diagonals of the previous box", "previous box diagonal"),
    "step_speed_norm": ("diagonals of the previous box per frame", "previous box diagonal"),
    "step_speed_norm_per_s": ("diagonals of the previous box per second", "previous box diagonal"),
    "step_iou_with_previous": ("fraction", None),
    "step_log_scale_change": ("natural log of the geometric mean of the side ratios", None),
    "step_log_scale_rate": ("natural log of the geometric mean of the side ratios per frame", None),
}

# The descriptor says what a step is not conditioned on, so that two results that disagree about it are not combined.
_CONDITIONING = "none: a step is not conditioned on shot boundaries or camera motion"


class _Item(NamedTuple):
    """One sequence's detections in order, a row each, with what its frame says about them."""

    position: NDArray[np.int64]  # the frame's position in the stream
    key: NDArray[np.int64]  # the frame's own number
    time_s: NDArray[np.float64]  # NaN where the frame declares no time
    width: NDArray[np.int64]  # -1 where the frame's size cannot be read
    height: NDArray[np.int64]
    boxes: NDArray[np.float64]
    ids: NDArray[np.int64]


_NO_ROWS = _Item(
    *(np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float64), np.empty(0, np.int64)),
    *(np.empty(0, np.int64), np.empty((0, 4), np.float64), np.empty(0, np.int64)),
)


def _check_max_frame_delta(value: Any) -> None:
    """Refuse a limit that is not an integer of at least 1, naming the keyword."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"max_frame_delta must be an integer of at least 1; got {value!r}.")


def _check_boxes(item: int, frame_tracks: Sequence[Any]) -> None:
    """Refuse a frame whose boxes are not four numbers for each detection it holds."""
    for position, target in enumerate(frame_tracks):
        count, size = detection_count(target), as_numpy(target.boxes).size
        if count and size != 4 * count:
            raise ValueError(
                f"The target of frame {position} of item {item} holds {count} detection(s) and {size} box value(s). "
                "Each box needs four numbers, (x0, y0, x1, y1).",
            )


def _seconds(value: Any) -> float:
    return np.nan if value is None else float(value)


def _extent(value: int | None) -> int:
    return -1 if value is None else int(value)


def _read_item(item: int, stream: Any, frame_tracks: Sequence[Any]) -> _Item:
    """Pair a sequence's stream with its target and read every detection, frame by frame."""
    _check_boxes(item, frame_tracks)
    frames: list[_Item] = []
    for position, rows in enumerate(MOTStructurer._frames_of(stream, frame_tracks)):  # pyright: ignore[reportPrivateUsage]
        refuse_repeated_track_ids(rows.track_ids, f"frame {position} of item {item}")
        n = len(rows.labels)
        frames.append(
            _Item(
                np.full(n, position, np.int64),
                np.full(n, rows.frame_index, np.int64),
                np.full(n, _seconds(rows.time_s), np.float64),
                np.full(n, _extent(rows.width), np.int64),
                np.full(n, _extent(rows.height), np.int64),
                rows.boxes.astype(np.float64),
                rows.track_ids.astype(np.int64),
            )
        )
    return _Item(*(np.concatenate(column) for column in zip(*frames, strict=True))) if frames else _NO_ROWS


def _previous_observation(ids: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return each row's previous observation of its track in this sequence, or -1.

    Rows are in frame order, so a stable sort by id keeps each track's observations in frame order too, and a
    row's predecessor is its neighbor when the ids match. A negative id belongs to no track.
    """
    previous = np.full(ids.size, -1, np.int64)
    tracked = np.flatnonzero(ids >= 0)
    order = tracked[np.argsort(ids[tracked], kind="stable")]
    same = ids[order[1:]] == ids[order[:-1]]
    previous[order[1:][same]] = order[:-1][same]
    return previous


def _valid_boxes(boxes: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Whether each box is finite, ordered and of positive width and height, as read from the stream."""
    x0, y0, x1, y1 = boxes.T
    with np.errstate(invalid="ignore"):
        return np.isfinite(boxes).all(axis=1) & (x1 > x0) & (y1 > y0)


def _status_codes(
    item: _Item, previous: NDArray[np.int64], limit: int
) -> tuple[NDArray[np.int8], NDArray[np.intp], NDArray[np.int64]]:
    """Assign every row its status, lowest precedence first so that a later rule overrides an earlier one."""
    code = np.full(item.ids.size, _OK, np.int8)
    now = np.flatnonzero(previous >= 0)
    before = previous[now]
    readable = item.width >= 0
    same_size = readable[now] & readable[before] & (item.width[now] == item.width[before])
    same_size &= item.height[now] == item.height[before]
    code[now[~same_size]] = _GEOMETRY
    valid = _valid_boxes(item.boxes)
    code[now[~(valid[now] & valid[before])]] = _INVALID
    delta = item.position[now] - item.position[before]
    code[now[delta > limit]] = _GAP
    code[previous < 0] = _FIRST
    code[item.ids < 0] = _UNTRACKED
    return code, now, before


def _scatter(size: int, rows: NDArray[np.intp], values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a column of `size` values, NaN except at `rows`."""
    column = np.full(size, np.nan)
    column[rows] = values
    return column


def _centers(boxes: NDArray[np.float64]) -> NDArray[np.float64]:
    return (boxes[:, :2] + boxes[:, 2:]) / 2


def _diagonal(boxes: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.hypot(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])


def _displacement(now: NDArray[np.float64], before: NDArray[np.float64]) -> NDArray[np.float64]:
    """Distance between the box centers, in diagonals of the previous box."""
    return np.hypot(*(_centers(now) - _centers(before)).T) / _diagonal(before)


def _iou(now: NDArray[np.float64], before: NDArray[np.float64]) -> NDArray[np.float64]:
    """Intersection over union of unclipped boxes, 0.0 where they do not overlap.

    Worked in coordinates translated to the previous box's center and divided by its diagonal, which keeps every
    product near 1 whatever the box's size or position.
    """
    shift, scale = np.tile(_centers(before), 2), _diagonal(before)[:, None]
    a, b = (before - shift) / scale, (now - shift) / scale
    inside = np.clip(np.minimum(a[:, 2:], b[:, 2:]) - np.maximum(a[:, :2], b[:, :2]), 0, None).prod(axis=1)
    areas = ((r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1]) for r in (a, b))
    return inside / (sum(areas) - inside)


def _log_scale(now: NDArray[np.float64], before: NDArray[np.float64]) -> NDArray[np.float64]:
    """Half the log of the area ratio, written as logs of the sides so that no product of sides is formed."""
    sides = np.log(np.stack([now[:, 2] - now[:, 0], now[:, 3] - now[:, 1]])) - np.log(
        np.stack([before[:, 2] - before[:, 0], before[:, 3] - before[:, 1]])
    )
    return 0.5 * sides.sum(axis=0)


def _measures(
    item: _Item, previous: NDArray[np.int64], code: NDArray[np.int8], delta: NDArray[np.float64], seconds: Any
) -> dict[str, NDArray[np.float64]]:
    """Return the measurements of the rows with status ok, which are the only rows that have any."""
    ok = np.flatnonzero(code == _OK)
    now, before = item.boxes[ok], item.boxes[previous[ok]]
    displacement, change, k = _displacement(now, before), _log_scale(now, before), delta[ok]
    size = item.ids.size
    with np.errstate(over="ignore"):
        per_second = displacement / seconds[ok]
    per_second[~np.isfinite(per_second)] = np.nan  # a step of a subnormal fraction of a second overflows the rate
    return {
        "step_displacement_norm": _scatter(size, ok, displacement),
        "step_speed_norm": _scatter(size, ok, displacement / k),
        "step_speed_norm_per_s": _scatter(size, ok, per_second),
        "step_iou_with_previous": _scatter(size, ok, _iou(now, before)),
        "step_log_scale_change": _scatter(size, ok, change),
        "step_log_scale_rate": _scatter(size, ok, change / k),
    }


def _pixel_difference(item: _Item, now: NDArray[np.intp], before: NDArray[np.int64]) -> dict[str, NDArray[np.float64]]:
    """Raw center differences in pixels, wherever there is a previous observation and both centers are finite."""
    with np.errstate(invalid="ignore", over="ignore"):
        c1, c0 = _centers(item.boxes[now]), _centers(item.boxes[before])
        step = c1 - c0
    finite = np.isfinite(c1).all(axis=1) & np.isfinite(c0).all(axis=1)
    return {
        "step_dx_px": _scatter(item.ids.size, now[finite], step[finite, 0]),
        "step_dy_px": _scatter(item.ids.size, now[finite], step[finite, 1]),
    }


def _frame_columns(
    item: _Item, previous: NDArray[np.int64], now: NDArray[np.intp], before: NDArray[np.int64]
) -> dict[str, NDArray[Any]]:
    """Return what the stream says about each step: the frames between, the seconds between and their own numbers."""
    size = item.ids.size
    with np.errstate(invalid="ignore", over="ignore"):  # a time that gives no step is NaN, not a warning
        step = item.time_s[now] - item.time_s[before]
    usable = np.isfinite(step) & (step > 0)
    previous_key = np.full(size, -1, np.int64)
    previous_key[now] = item.key[before]
    return {
        "step_frame_delta": _scatter(size, now, (item.position[now] - item.position[before]).astype(np.float64)),
        "step_elapsed_time_s": _scatter(size, now[usable], step[usable]),
        "step_previous_target_index": previous,
        "step_frame_key": item.key,
        "step_previous_frame_key": previous_key,
    }


def _item_columns(item: _Item, limit: int) -> dict[str, NDArray[Any]]:
    """Every column of one sequence, a value per detection."""
    previous = _previous_observation(item.ids)
    code, now, before = _status_codes(item, previous, limit)
    columns = _frame_columns(item, previous, now, before)
    columns |= _pixel_difference(item, now, before)
    columns |= _measures(item, previous, code, columns["step_frame_delta"], columns["step_elapsed_time_s"])
    columns["step_status"] = _NAMES[code]
    return columns


def _specs(limit: int) -> dict[str, MeasurementSpec]:
    """Return what each factor was measured with, which for all of them is one limit and one conditioning."""
    config = {"max_frame_delta": limit, "conditioning": _CONDITIONING}
    return {
        name: MeasurementSpec("track_step", "1", units, "box", normalization, config)
        for name, (units, normalization) in _UNITS.items()
    }


def _stack(items: list[dict[str, NDArray[Any]]], name: str) -> NDArray[Any]:
    """One column for the whole dataset, which is empty when there are no sequences."""
    if not items:
        return np.empty(0, object if name == "step_status" else np.float64 if name in _FACTORS else np.int64)
    return np.concatenate([columns[name] for columns in items])


@experimental
def track_steps(source: MultiobjectTrackingDataset, *, max_frame_delta: int = 1) -> AddressedResult[Any]:
    """Measure the step each detection takes from where its track was last seen.

    A tracker's identity can be wrong without any single box looking wrong: an id that jumps to another object, a box
    that doubles in size between frames, a track that races across the frame. Each of these is a large step, and no
    whole-track statistic shows which detection took it. A step runs from the previous observation of a detection's
    track, the latest earlier detection with the same track id in the same sequence, to the detection itself.

    .. versionadded:: 1.3

    Parameters
    ----------
    source : MultiobjectTrackingDataset
        A tracking dataset. Each item's stream is paired with its target frame by frame, and a stream and a target
        that disagree on how many frames the item has raise.
    max_frame_delta : int, default 1
        The most frames a step may span, at least 1. A step spans ``k`` frames, the difference between the two
        frames' positions in the stream, so a frame with no detection counts. The default admits consecutive frames
        only. A row whose previous observation lies further back has status ``"gap_exceeds_limit"`` and no
        measurements; pass 2 to bridge a single missed frame.

    Returns
    -------
    AddressedResult
        One row per detection of every item, untracked ones included, addressed by
        ``SourceIndex(item, target_index)``, where ``target_index`` counts a item's detections over all its frames
        as :class:`~dataeval.Metadata` does. :meth:`~dataeval.Metadata.add_factors` attaches the rows to the instance
        rows with no further argument. The result declares no roll-ups.

        ``stats`` holds float64 columns. Every measurement is NaN unless the row's status is ``"ok"``:

        - ``step_frame_delta``: ``k``, the frames between the two observations. NaN where there is no previous
          observation.
        - ``step_elapsed_time_s``: the seconds between the two frames' reported ``time_s``. NaN where there is no
          previous observation, where either time is missing or not finite, and where the step is zero or negative.
          A missing time never changes the status.
        - ``step_displacement_norm``: the distance between the two box centers, in diagonals of the previous box, so
          that it compares across resolutions and box sizes.
        - ``step_speed_norm``: ``step_displacement_norm / k``.
        - ``step_speed_norm_per_s``: ``step_displacement_norm / step_elapsed_time_s``. NaN also where the time is, and
          where the rate is not finite because the step is a vanishing fraction of a second.
        - ``step_iou_with_previous``: the intersection over union of the two boxes, unclipped. 0.0 where they do not
          overlap.
        - ``step_log_scale_change``: ``0.5 * (ln w1 - ln w0 + ln h1 - ln h0)``, the natural log of the ratio of the
          boxes' geometric-mean sides. Positive where the box grew.
        - ``step_log_scale_rate``: ``step_log_scale_change / k``.

        ``details`` holds one value per row, and :meth:`~dataeval.Metadata.add_factors` does not attach it:

        - ``step_status``: why a row is or is not measured, the first of ``"untracked"`` (a negative track id, or a
          frame that carries none), ``"first_observation"`` (the id has no earlier detection in this item),
          ``"gap_exceeds_limit"`` (``k`` exceeds `max_frame_delta`), ``"invalid_box"`` (either box is not finite, or
          has a reversed or zero-width or zero-height side), ``"geometry_change"`` (the two frames' pixel sizes are
          not both readable and equal) and ``"ok"``.
        - ``step_dx_px``, ``step_dy_px``: the raw difference between the two box centers in pixels, wherever a
          previous observation exists and both centers are finite, whatever the status.
        - ``step_previous_target_index``: the ``target_index`` of the previous observation, -1 for none.
        - ``step_frame_key``, ``step_previous_frame_key``: the frames' own numbers, ``frame_index`` where a frame
          declares one and its position otherwise. -1 for no previous observation.

        ``measurement_specs`` records ``max_frame_delta`` for every factor, so results measured with different limits
        are not combined by :func:`~dataeval.core.combine_stats_results`.

    Raises
    ------
    ValueError
        If `max_frame_delta` is not an integer of at least 1, which is checked before any item is read. If a stream
        yields more or fewer frames than its target holds, if a frame's boxes are not four numbers for each of its
        detections, or if a frame gives one track id to two detections.
    MaiteShapeError
        If `source` is not a multi-object tracking dataset.

    See Also
    --------
    :func:`~dataeval.core.track_stats` : One row per track, measured as a whole
    :func:`~dataeval.core.frame_stats` : One row per frame, with the time between frames and exact repeats

    Notes
    -----
    **What a step is not conditioned on.** A step is not conditioned on shot boundaries or on camera motion. A cut
    between two shots, or a camera that pans, moves every box and reads as a large step. A stitched video with
    ``track_ids="preserve"`` steps across the seams of its segments. Where the camera moves,
    ``step_displacement_norm`` measures the camera as much as the object.

    **A large step is not an error.** The function does not judge identity: a large step can be an id switch, a
    reused id or a fast object. A small one on a still object includes the jitter of its box. The rows of one track
    are not independent observations of anything, so count tracks, not rows, when a sample size matters.

    **The frame-size check.** ``"geometry_change"`` compares the pixel sizes a stream reports for the two frames, read
    from the shape of each frame's ``pixels``. A lazy stream reports the size in its header for every frame, so a
    resolution change inside one file is not seen until a frame is decoded. A zoom, a crop or a rotation inside a
    constant frame passes unseen. A size that cannot be read counts as a change, so a stream whose frames carry no
    pixels array has no ``"ok"`` row.

    **Precision.** Boxes are read as :class:`~dataeval.Metadata` holds them, as float32, and every measurement is
    computed in float64 from those values. A coordinate beyond about 8 million pixels loses sub-pixel resolution, and
    one beyond the float32 range becomes infinite and reads as an invalid box.

    **After an invalid box.** The previous observation is never skipped over an invalid box. The detection that
    follows one has status ``"invalid_box"`` though its own box is fine, and ``step_previous_target_index`` says which
    detection it was compared with.

    **At the default.** ``k`` is 1 for every ``"ok"`` row, so ``step_speed_norm`` equals ``step_displacement_norm`` and
    ``step_log_scale_rate`` equals ``step_log_scale_change``. They differ where `max_frame_delta` is raised, and that is
    where they compare across gap sizes. ``k`` counts the frames the stream yields, not the difference between frame
    numbers, so a stream that skips numbers is still one frame apart.

    **Cost.** Each frame's size is read from the shape of its pixels and no pixel is converted. A stream whose
    ``pixels`` getter decodes does so once per frame. Memory grows with the number of detections, because each holds
    one :class:`~dataeval.types.SourceIndex`.

    Examples
    --------
    Four videos of sixty frames, each holding one tracked detection that does not move. A detection's first
    observation has no step, so its measurements are NaN and its status says why:

    >>> from dataeval.core import track_steps
    >>> result = track_steps(train_videos)
    >>> result["details"]["step_status"][:3].tolist()
    ['first_observation', 'ok', 'ok']
    >>> result["stats"]["step_displacement_norm"][:3]
    array([nan,  0.,  0.])
    >>> result["stats"]["step_iou_with_previous"][:3]
    array([nan,  1.,  1.])

    Attach the steps to a tracking dataset's metadata:

    >>> metadata.add_factors(track_steps(tracking_dataset))  # doctest: +SKIP
    """
    _check_max_frame_delta(max_frame_delta)
    validate_dataset(source, expected="multiobject_tracking", arg_name="source", caller="track_steps")
    items = []
    for index in range(len(source)):
        stream, target, _ = source[index]
        items.append(_item_columns(_read_item(index, stream, target.frame_tracks), int(max_frame_delta)))
    # ponytail: one SourceIndex object per detection, about 120 B each, because the result contract addresses every
    # value by one. An array-backed address column would remove the cost.
    source_index = [SourceIndex(i, j) for i, columns in enumerate(items) for j in range(columns["step_frame_key"].size)]
    return AddressedResult(
        stats={name: _stack(items, name) for name in _FACTORS},
        source_index=source_index,
        measurement_specs=_specs(int(max_frame_delta)),
        details={name: _stack(items, name) for name in _DETAILS},
    )
