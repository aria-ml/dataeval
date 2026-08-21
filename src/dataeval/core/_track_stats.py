"""Statistics for characterizing object tracks in multi-object tracking datasets."""

from __future__ import annotations

__all__ = ["TrackStatsResult", "track_stats"]

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, cast, overload

import numpy as np
from typing_extensions import NotRequired

from dataeval._log import get_logger
from dataeval.data import build_tracks
from dataeval.protocols import Dataset, MultiobjectTrackingDataset
from dataeval.types import Track
from dataeval.types._track import frame_size
from dataeval.utils._internal import EPSILON

_logger = get_logger(__name__)


class TrackStatsResult(TypedDict):
    """
    Compute per-track statistics for one video sequence.

    Attributes
    ----------
    track_ids : Sequence[int]
        Sorted list of track IDs present in the input. ``track_ids[i]``
        gives the track ID whose stats are at index *i* in every other
        sequence field. Track IDs restart in every sequence, so this
        identifies a track only together with ``item_index``.
    item_index : Sequence[int], optional
        Which dataset item each entry describes, present only when a whole
        dataset was measured. Paired with ``track_ids`` it names exactly one
        track, which is what lets the whole result be handed to
        :meth:`~dataeval.Metadata.add_factors` with ``key="track_id"``.
        Absent when a single sequence's tracks were passed, since there is
        then no item to name.
    labels : Sequence[int]
        Representative class label for each track, taken as the most frequent
        per-frame label. A track should be single-class in practice, but the
        raw per-frame labels may disagree; this reports the majority label so
        each track has exactly one class for downstream analysis.
    label_confidence : Sequence[float]
        Confidence in each track's assigned ``labels`` entry: the assigned
        label's share of total detection score across the track's
        observations, ``sum(scores[labels == label]) / sum(scores)``. 1.0 for
        a single-class track; lower when observations disagree. With uniform
        scores (e.g. ground truth) this is the fraction of frames carrying the
        label — a purity / label-consistency signal.
    mean_score : Sequence[float]
        Mean detection confidence over the track's observations. 1.0 for
        ground-truth tracks whose scores are all 1.0. NaN when scores are
        absent.
    n_appearances : Sequence[int]
        Number of frames in which the track ID appears. A track present
        in every frame of a 300-frame video has ``n_appearances = 300``;
        a track present only in the first and last frames has
        ``n_appearances = 2``.
    track_duration : Sequence[int]
        Total number of frames between a track's first and last appearance,
        inclusive. Equal to ``n_appearances`` for contiguous tracks; greater
        than ``n_appearances`` when gaps are present. A track present only
        in the first and last frames of a 300-frame video has
        ``track_duration = 300``.
    n_gaps : Sequence[int]
        Number of contiguous gap runs between a track's first and last
        appearance. Zero for fully contiguous tracks.
    total_gap_length : Sequence[int]
        Total number of frames between a track's first and last appearance
        where the track ID is absent. Equal to
        ``track_duration - n_appearances``.
    mean_speed : Sequence[float]
        Mean per-frame displacement of the bounding-box center in pixels,
        computed over consecutive observed frame pairs and normalised by
        the inter-frame delta so gaps do not inflate the estimate.
    speed_variance : Sequence[float]
        Variance of per-frame center displacement across consecutive
        observed frame pairs. Zero for single-observation tracks.
    net_displacement : Sequence[float]
        Euclidean distance in pixels between the box center at the first
        and last observation, regardless of the path taken.
    straightness_index : Sequence[float]
        Ratio of net displacement to total path length (sum of per-step
        Euclidean distances between consecutive observed centers). Ranges
        from 0 (maximally tortuous / looping) to 1 (perfectly straight).
        NaN for single-observation tracks or perfectly stationary tracks
        (zero path length).
    jitter_rate : Sequence[float]
        Smoothness of the track's speed profile, computed using SPARC
        (SPectral ARC length). Unlike the reference paper, the result is not
        negated, so higher = more jittery. NaN for tracks shorter than
        ``jitter_min_frames`` (default 10). Presented as a rate per track length.
    entry_at_edge : Sequence[bool | None]
        True when the track's first bounding box touches or crosses the
        frame border (within ``edge_threshold`` pixels on any side), and
        None when the frame dimensions are unknown. None rather than False,
        because "the border was not reached" and "where the border is was
        never established" are different answers, and only one of them is a
        measurement.
    exit_at_edge : Sequence[bool | None]
        True when the track's last bounding box touches or crosses the
        frame border (within ``edge_threshold`` pixels on any side), and
        None when the frame dimensions are unknown, as for ``entry_at_edge``.

    Notes
    -----
    1. Every sequence field is indexed by position in sorted track ID order,
    not by track ID directly. Use ``track_ids[i]`` to recover the original
    track ID for position *i*.

    References
    ----------
    Balasubramanian, S., Melendez-Calderon, A., & Burdet, E. (2012).
    A robust and sensitive metric for quantifying movement smoothness.
    IEEE Transactions on Biomedical Engineering, 59(8), 2126-2136.

    Batschelet, E. (1981). Circular Statistics in Biology. Academic Press.

    Benhamou, S. (2004). How to reliably estimate the tortuosity of an
    animal's path: straightness, sinuosity, or fractal dimension?
    Journal of Theoretical Biology, 229(2), 209-220.
    """

    track_ids: Sequence[int]
    item_index: NotRequired[Sequence[int]]
    labels: Sequence[int]
    label_confidence: Sequence[float]
    mean_score: Sequence[float]
    n_appearances: Sequence[int]
    track_duration: Sequence[int]
    n_gaps: Sequence[int]
    total_gap_length: Sequence[int]
    mean_speed: Sequence[float]
    speed_variance: Sequence[float]
    net_displacement: Sequence[float]
    straightness_index: Sequence[float]
    jitter_rate: Sequence[float]
    entry_at_edge: Sequence[bool | None]
    exit_at_edge: Sequence[bool | None]


def _centers(boxes: np.ndarray) -> np.ndarray:
    """Return shape-(T, 2) array of (cx, cy) centers for a (T, 4) box array."""
    return np.stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0],
        axis=1,
    ).astype(np.float64)


def _label_score_stats(labels: np.ndarray, scores: np.ndarray) -> tuple[int, float, float]:
    """Return a track's representative label, the confidence in it, and its mean score.

    The label is the most frequent per-frame class (ties broken by lowest
    index). Confidence is that label's share of the track's total detection
    score, ``sum(scores[labels == label]) / sum(scores)``: 1.0 for a
    single-class track, lower when other classes appear. When scores are
    absent or all zero, observations are weighted equally, so confidence
    becomes the fraction of frames carrying the label (its purity).
    ``mean_score`` is the mean detection confidence, or NaN when scores are
    absent.
    """
    vals, counts = np.unique(labels, return_counts=True)
    label = int(vals[np.argmax(counts)])

    scores = np.asarray(scores, dtype=np.float64)
    mean_score = float(scores.mean()) if scores.size else np.nan

    total = float(scores.sum())
    if scores.shape == labels.shape and total > EPSILON:
        weights, wsum = scores, total
    else:
        weights, wsum = np.ones(labels.shape), float(labels.size)
    confidence = float(weights[labels == label].sum() / wsum)
    return label, confidence, mean_score


def _at_edge(box: np.ndarray, frame_w: int, frame_h: int, threshold: float) -> bool:
    """Return True if *box* is within *threshold* pixels of any frame border."""
    x1, y1, x2, y2 = box
    return (
        bool(x1 <= threshold)
        or bool(y1 <= threshold)
        or bool(x2 >= frame_w - threshold)
        or bool(y2 >= frame_h - threshold)
    )


def _compute_appearances_and_duration(
    frames: np.ndarray,
) -> tuple[int, int]:
    tl = len(frames)
    duration = int(frames[-1] - frames[0] + 1) if tl > 0 else 0
    return tl, duration


def _compute_gaps(
    frames: np.ndarray,
    tl: int,
    duration: int,
) -> tuple[int, int]:
    if tl < 2:
        return 0, 0
    diffs = np.diff(frames)
    return int(np.sum(diffs > 1)), duration - tl


def _compute_step_speeds(
    ctrs: np.ndarray,
    frames: np.ndarray,
    tl: int,
) -> tuple[np.ndarray, np.ndarray]:
    if tl < 2:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, np.array([], dtype=np.float64)
    delta_pos = np.diff(ctrs, axis=0)
    delta_frames = np.diff(frames).astype(np.float64)
    step_speeds = np.linalg.norm(delta_pos, axis=1) / delta_frames
    return delta_pos, step_speeds


def _compute_speed_stats(
    step_speeds: np.ndarray,
) -> tuple[float, float]:
    if len(step_speeds) == 0:
        return 0.0, 0.0
    return float(np.mean(step_speeds)), float(np.var(step_speeds))


def _compute_straightness(
    net_disp: float,
    delta_pos: np.ndarray,
    tl: int,
) -> float:
    if tl < 2:
        return np.nan
    path_length = float(np.sum(np.linalg.norm(delta_pos, axis=1)))
    return np.nan if path_length == 0.0 else net_disp / path_length


def _compute_jitter_rate(
    speed_profile: np.ndarray,
    tl: int,
    jitter_min_frames: int,
    jitter_fc: float,
) -> float:
    """Per-frame jitter, comparable across clips of different lengths."""
    if jitter_min_frames > tl:
        return np.nan
    return _jitter_sparc(speed_profile, fc_norm=jitter_fc) / len(speed_profile)


def _compute_edge_flags(
    boxes: np.ndarray,
    has_frame_dims: bool,
    frame_width: int,
    frame_height: int,
    edge_threshold: float,
) -> tuple[bool | None, bool | None]:
    """Whether the track's first and last boxes touch the border, or None if unknowable.

    None rather than False without frame dimensions: False is the answer "this box is not
    at the border", which is a measurement, and there is none to report when where the
    border sits was never established.
    """
    if not has_frame_dims:
        return None, None
    return (
        _at_edge(boxes[0], frame_width, frame_height, edge_threshold),
        _at_edge(boxes[-1], frame_width, frame_height, edge_threshold),
    )


def _jitter_sparc(
    speed_profile: np.ndarray,
    padlevel: int = 4,
    fc_norm: float = 0.5,
    amp_th: float = 0.05,
) -> float:
    """
    Compute SPARC smoothness for a scalar speed profile, normalized to cycles-per-frame.

    Parameters
    ----------
    speed_profile : np.ndarray
        1-D array of per-frame scalar speeds (pixel displacement / frame).
    padlevel : int
        Zero-padding factor. FFT length = 2^(ceil(log2(N)) + padlevel).
        Default 4, matching the paper's recommendation.
    fc_norm : float
        Hard upper cutoff in cycles-per-frame. Must be in (0, 0.5], where
        0.5 is the Nyquist frequency (one cycle every two frames). Values
        below 0.5 exclude high-frequency content; e.g. 0.4 excludes the
        top 20% of the spectrum. Default 0.5 applies no hard cutoff beyond
        Nyquist (adaptive cutoff via amp_th still applies).
    amp_th : float
        Amplitude threshold for adaptive cutoff. The spectrum is truncated
        at the last frequency bin where normalized magnitude >= amp_th.
        The arc length is then normalized by the selected bandwidth, not
        fc_norm. Default 0.05, matching the reference implementation.

    Returns
    -------
    float
        SPARC value (>= 0). Higher = more jittery. Note: the reference
        implementation returns a negative value; this function returns the
        absolute value for convenience.

    Raises
    ------
    ValueError
        When fc_norm is outside of expected bounds.

    Notes
    -----
    1. The frequency axis uses cycles-per-frame rather than Hz, making the
    metric invariant to the true (or unknown) frame rate. The equivalent
    of the reference's fc parameter (in Hz) is fc_norm * fs, where fs is
    the frame rate.
    2. At least 2 frequency bins must survive both cutoffs for a meaningful
    result; fewer returns 0.0.
    3. 1.0 is subtracted from the final result because the minimum arc length
    for minimal jitter is 1.0, leaving an invisible gap between 0.0 and 1.0
    in the output range, that could easily be misinterpreted.

    References
    ----------
    Balasubramanian, S., Melendez-Calderon, A., & Burdet, E. (2012).
    A robust and sensitive metric for quantifying movement smoothness.
    IEEE Transactions on Biomedical Engineering, 59(8), 2126-2136.
    """
    if not (0 < fc_norm <= 0.5):
        raise ValueError("fc_norm must be in (0, 0.5]")

    n = len(speed_profile)
    nfft = int(2 ** (np.ceil(np.log2(n)) + padlevel))

    freq_cpf = np.fft.rfftfreq(nfft)
    mag = np.abs(np.fft.rfft(speed_profile, nfft))

    if mag[0] <= EPSILON:
        return 0.0
    mag = mag / mag[0]

    fc_mask = freq_cpf <= fc_norm
    freq_cpf = freq_cpf[fc_mask]
    mag = mag[fc_mask]

    above_threshold = np.where(mag >= amp_th)[0]
    if len(above_threshold) == 0:
        return 0.0
    fc_idx = above_threshold[-1]
    freq_cpf = freq_cpf[: fc_idx + 1]
    mag = mag[: fc_idx + 1]

    if len(freq_cpf) < 2:
        return 0.0

    bw = freq_cpf[-1] - freq_cpf[0]
    delta_f = np.diff(freq_cpf) / bw
    delta_m = np.diff(mag)

    arc_length = float(np.sum(np.sqrt(delta_f**2 + delta_m**2)))
    return max(0.0, arc_length - 1.0)


def _sequence_stats(
    tracks: Mapping[int, Track],
    frame_width: int | None = None,
    frame_height: int | None = None,
    edge_threshold: float = 5.0,
    jitter_min_frames: int = 10,
    jitter_fc: float = 0.5,
) -> TrackStatsResult:
    """Measure one sequence's tracks, in sorted track ID order.

    The whole computation; :func:`track_stats` is the entry point that decides what to
    hand it and how to label what comes back. Arguments are trusted, having been
    validated once at that entry point rather than once per sequence.
    """
    _logger.info(
        "Computing track stats for %d tracks",
        len(tracks),
    )

    has_frame_dims = frame_width is not None and frame_height is not None
    if not has_frame_dims:
        _logger.warning(
            "frame_width / frame_height not provided; entry_at_edge and exit_at_edge will be null.",
        )
        _frame_width: int = 0
        _frame_height: int = 0
    else:
        _frame_width = frame_width  # type: ignore[assignment]  # narrowed by has_frame_dims check
        _frame_height = frame_height  # type: ignore[assignment]

    # Output lists — one entry per track, in sorted track ID order.
    sorted_ids = sorted(tracks.keys())

    track_ids: list[int] = sorted_ids
    labels: list[int] = []
    label_confidence: list[float] = []
    mean_score: list[float] = []
    n_appearances: list[int] = []
    track_duration: list[int] = []
    n_gaps: list[int] = []
    total_gap_length: list[int] = []
    mean_speed: list[float] = []
    speed_variance: list[float] = []
    net_displacement: list[float] = []
    straightness_index: list[float] = []
    jitter_rate: list[float] = []
    entry_at_edge: list[bool | None] = []
    exit_at_edge: list[bool | None] = []

    if has_frame_dims and edge_threshold >= min(_frame_width, _frame_height) / 2:
        _logger.warning(
            "edge_threshold (%.1f) is >= half the smallest frame dimension; edge flags will be unreliable.",
            edge_threshold,
        )

    for tid in sorted_ids:
        track = tracks[tid]
        frames = track.frame_indices
        boxes = track.boxes

        if len(frames) == 0:
            continue

        label, confidence, score = _label_score_stats(track.labels, track.scores)
        labels.append(label)
        label_confidence.append(confidence)
        mean_score.append(score)

        ctrs = _centers(boxes)

        tl, duration = _compute_appearances_and_duration(frames)
        n_appearances.append(tl)
        track_duration.append(duration)

        gaps, gap_len = _compute_gaps(frames, tl, duration)
        n_gaps.append(gaps)
        total_gap_length.append(gap_len)

        delta_pos, step_speeds = _compute_step_speeds(ctrs, frames, tl)

        ms, sv = _compute_speed_stats(step_speeds)
        mean_speed.append(ms)
        speed_variance.append(sv)

        net_disp = float(np.linalg.norm(ctrs[-1] - ctrs[0]))
        net_displacement.append(net_disp)

        straightness_index.append(_compute_straightness(net_disp, delta_pos, tl))
        jitter_rate.append(_compute_jitter_rate(step_speeds, tl, jitter_min_frames, jitter_fc))

        entry, exit_ = _compute_edge_flags(boxes, has_frame_dims, _frame_width, _frame_height, edge_threshold)
        entry_at_edge.append(entry)
        exit_at_edge.append(exit_)

    _logger.info("Track stats complete.")

    return TrackStatsResult(
        track_ids=track_ids,
        labels=labels,
        label_confidence=label_confidence,
        mean_score=mean_score,
        n_appearances=n_appearances,
        track_duration=track_duration,
        n_gaps=n_gaps,
        total_gap_length=total_gap_length,
        mean_speed=mean_speed,
        speed_variance=speed_variance,
        net_displacement=net_displacement,
        straightness_index=straightness_index,
        jitter_rate=jitter_rate,
        entry_at_edge=entry_at_edge,
        exit_at_edge=exit_at_edge,
    )


def _dataset_stats(
    dataset: MultiobjectTrackingDataset,
    frame_width: int | None,
    frame_height: int | None,
    edge_threshold: float,
    jitter_min_frames: int,
    jitter_fc: float,
) -> TrackStatsResult:
    """Measure every sequence in a dataset, labelling each result with the item it came from.

    Items are walked by position, so ``item_index`` is the same index
    :class:`~dataeval.Metadata` keys its rows by. Reading them off
    :func:`~dataeval.data.build_tracks`'s dataset form instead would key on
    ``DatumMetadata['id']``, which is not required to be unique and silently collapses two
    sequences that share one.

    Frame dimensions are read from the first frame of each item's stream, so a dataset
    whose sequences differ in size gets each measured against its own borders — which the
    single ``frame_width`` / ``frame_height`` pair cannot express. Passing both explicitly
    overrides that for every sequence and skips the decode.
    """
    stated = frame_width is not None and frame_height is not None
    merged: dict[str, list[Any]] = {}
    count = len(dataset)

    for item in range(count):
        video_stream, target, _ = dataset[item]
        tracks = build_tracks(target)
        width, height = (frame_width, frame_height) if stated else frame_size(next(iter(video_stream), None))
        stats = _sequence_stats(tracks, width, height, edge_threshold, jitter_min_frames, jitter_fc)
        # Read as a plain mapping: a TypedDict types its values field by field, and this
        # walks them uniformly without caring which field it is holding.
        for name, values in cast("Mapping[str, Sequence[Any]]", stats).items():
            merged.setdefault(name, []).extend(values)
        merged.setdefault("item_index", []).extend([item] * len(stats["track_ids"]))

    _logger.info("Track stats complete for %d item(s).", count)
    # An empty dataset still has to answer with every field, so the keys come from the
    # result type rather than from whatever the walk happened to produce.
    return cast("TrackStatsResult", {name: merged.get(name, []) for name in TrackStatsResult.__annotations__})


@overload
def track_stats(
    source: Mapping[int, Track],
    frame_width: int | None = None,
    frame_height: int | None = None,
    edge_threshold: float = 5.0,
    jitter_min_frames: int = 10,
    jitter_fc: float = 0.5,
) -> TrackStatsResult: ...
@overload
def track_stats(
    source: MultiobjectTrackingDataset,
    frame_width: int | None = None,
    frame_height: int | None = None,
    edge_threshold: float = 5.0,
    jitter_min_frames: int = 10,
    jitter_fc: float = 0.5,
) -> TrackStatsResult: ...


def track_stats(
    source: Mapping[int, Track] | MultiobjectTrackingDataset,
    frame_width: int | None = None,
    frame_height: int | None = None,
    edge_threshold: float = 5.0,
    jitter_min_frames: int = 10,
    jitter_fc: float = 0.5,
) -> TrackStatsResult:
    """Compute per-track statistics for one video sequence or a whole dataset.

    Results are returned as lists indexed by **position in sorted track ID
    order**.  The ``track_ids`` field maps each position back to its original
    track ID::

        stats["track_ids"][i]  # the track ID at position i
        stats["mean_speed"][i]  # mean speed for that track

    Given a dataset, every sequence is measured and the results are concatenated,
    each labelled with the item it came from in ``item_index``. That pair —
    ``item_index`` and ``track_ids`` — is what names a row, so the whole result can
    be handed straight to :meth:`~dataeval.Metadata.add_factors`::

        metadata.add_factors(track_stats(dataset), level="track", key="track_id")

    Parameters
    ----------
    source : Mapping[int, Track] or MultiobjectTrackingDataset
        One sequence's tracks, as returned by :func:`~dataeval.data.build_tracks`, or a
        whole tracking dataset to walk. A dataset is the form that pairs with
        :class:`~dataeval.Metadata`: it produces the ``item_index`` a track row is
        identified by, and it reads each sequence's frame dimensions for itself.
    frame_width, frame_height : int or None, default None
        Frame dimensions in pixels, required for ``entry_at_edge`` and ``exit_at_edge``.
        Given a dataset both are read from each sequence's first frame, so passing them
        is only needed to override that — which also skips decoding a frame per item.
        Given one sequence's tracks there is nothing to read them from, and leaving them
        out makes both edge fields null, with a warning.
    edge_threshold : float, default 5.0
        A bounding box is considered to touch the frame border when any
        edge is within this many pixels of the frame boundary (inclusive).
    jitter_min_frames : int, default 10
        Minimum number of observed frames required to compute jitter.
    jitter_fc : float, default 0.5
        Hard upper frequency cutoff as a fraction of Nyquist (i.e. cycles-per-frame
        divided by 0.5). Must be in (0, 1]. The default applies no
        hard cutoff beyond Nyquist (use only adaptive cutoff). Set lower
        to exclude high-frequency noise, e.g. 0.4 excludes the top 20%
        of the spectrum.

    Returns
    -------
    TrackStatsResult
        A :class:`TrackStatsResult` dict where every stat field is a Sequence
        indexed by position in sorted track ID order.  Use ``track_ids[i]``
        to recover the original track ID for position *i*, and ``item_index[i]``
        the sequence it belongs to when a dataset was measured.

    Raises
    ------
    ValueError
        When frame dimensions are not positive.
    TypeError
        When `source` is neither a track mapping nor a tracking dataset.

    Notes
    -----
    Detections that no tracker linked — those carrying a negative track id — belong to no
    track and take no part in any statistic. ``Metadata`` builds its track rows by the same
    rule, so the tracks measured here are exactly the rows a keyed
    :meth:`~dataeval.Metadata.add_factors` writes to.

    Two of these quantities are ones ``Metadata`` already holds under other names:
    ``n_appearances`` is ``track_length`` and ``track_duration`` is ``frame_span``, both
    accumulated during structuring. Attaching the whole result therefore brings duplicates
    of those two; drop them from the mapping to leave metadata's own in place.
    """
    if frame_width is not None and frame_width <= 0:
        raise ValueError(f"frame_width must be positive, got {frame_width}")
    if frame_height is not None and frame_height <= 0:
        raise ValueError(f"frame_height must be positive, got {frame_height}")

    # Mapping first: a Mapping satisfies the Dataset protocol's __len__/__getitem__ too,
    # so checking the dataset side first would swallow the single-sequence form.
    if isinstance(source, Mapping):
        return _sequence_stats(source, frame_width, frame_height, edge_threshold, jitter_min_frames, jitter_fc)
    if isinstance(source, Dataset):
        return _dataset_stats(
            cast("MultiobjectTrackingDataset", source),
            frame_width,
            frame_height,
            edge_threshold,
            jitter_min_frames,
            jitter_fc,
        )
    raise TypeError(
        f"'source' must be a mapping of track id to Track or a MultiobjectTrackingDataset, "
        f"got {type(source).__name__}.",
    )
