"""Per-track path statistics: how far, and in which direction, a track's embeddings travel."""

__all__ = []

import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval._experimental import experimental
from dataeval.core._compute_stats import FactorResult
from dataeval.types import MeasurementSpec
from dataeval.types._target import whole_ids
from dataeval.utils._array import as_numpy

_UNITS = "distance between unit-length embeddings"
_GEOMETRY = "rows scaled to unit length, no video mean, first and last max(n // 3, 1) rows"


def _check_min_points(value: Any) -> None:
    """Refuse a floor that is not an integer of at least 3, naming the keyword."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 3:
        raise ValueError(
            f"min_points must be an integer of at least 3; got {value!r}. "
            "With two points the displacement equals its null for any data.",
        )


def _specs(min_points: int) -> dict[str, MeasurementSpec]:
    """Return what both measured factors were measured with: one floor and one geometry."""
    config = {"min_points": min_points, "geometry": _GEOMETRY}
    return {
        name: MeasurementSpec(algorithm, "1", _UNITS, "track", "unit length", config)
        for name, algorithm in (
            ("path_displacement", "track_path_displacement"),
            ("path_displacement_null", "track_path_displacement_null"),
        )
    }


def _read(
    embeddings: ArrayLike, track_ids: ArrayLike, frame_indices: ArrayLike, sequence_indices: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """Return the four inputs as arrays, one entry for each row, refusing what cannot be measured."""
    x = as_numpy(embeddings, dtype=np.float64, required_ndim=2)
    track = whole_ids(track_ids, "track_ids", "Give -1 to a row no tracker linked.")
    frame = whole_ids(frame_indices, "frame_indices")
    sequence = whole_ids(sequence_indices, "sequence_indices")
    for name, values in (("track_ids", track), ("frame_indices", frame), ("sequence_indices", sequence)):
        if len(values) != len(x):
            raise ValueError(f"{name} has {len(values)} entries and embeddings has {len(x)} rows. Give one per row.")
    if not np.isfinite(x).all():
        raise ValueError("embeddings must be finite, and some row is NaN or infinite. Check the extractor's output.")
    return x, track, frame, sequence


def _group(
    track: NDArray[np.intp], frame: NDArray[np.intp], sequence: NDArray[np.intp]
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """Return the tracked rows in ``(sequence, track, frame)`` order, and where each track starts and ends in it."""
    tracked = np.flatnonzero(track >= 0)
    if not len(tracked):
        raise ValueError(
            "No row has a track id. Pass the track ids of a tracking dataset, such as `DetectionCrops.track_ids` "
            "over a `SequenceFrames` view.",
        )
    # Sorted by a key no two rows share, so every later sum runs in one order whatever order the input had.
    order = tracked[np.lexsort((frame[tracked], track[tracked], sequence[tracked]))]
    s, t, f = sequence[order], track[order], frame[order]
    same_track = (s[1:] == s[:-1]) & (t[1:] == t[:-1])
    repeated = np.flatnonzero(same_track & (f[1:] == f[:-1]))
    if len(repeated):
        i = int(repeated[0])
        raise ValueError(
            f"track {t[i]} of sequence {s[i]} has two rows at frame {f[i]}. "
            "Give each detection of a track its own frame index, or drop the duplicate row.",
        )
    starts = np.flatnonzero(np.concatenate(([True], ~same_track)))
    ends = np.concatenate((starts[1:], [len(order)]))
    if (ends - starts).max() < 2:
        raise ValueError(
            "No track has two rows. `sequence_indices` is the video each row comes from, not its position in a frame "
            "view such as `DetectionCrops.item_indices`, which gives every frame its own sequence.",
        )
    return order, starts, ends


def _measure(rows: NDArray[np.float64]) -> tuple[float, float, NDArray[np.float64]]:
    """Return one track's path displacement, its null and its direction, from its unit-length rows in frame order."""
    n = len(rows)
    # ponytail: thirds counted in rows, not frames, so a gap in a track is not corrected for; split by frame span if
    # gapped tracks need it.
    k = max(n // 3, 1)
    step = rows[-k:].mean(axis=0) - rows[:k].mean(axis=0)
    length = float(np.linalg.norm(step))
    centered = rows - rows.mean(axis=0)
    null = np.sqrt(2.0 * float(np.sum(centered * centered)) / (k * (n - 1)))
    return length, float(null), (step / length if length > 0.0 else np.full(rows.shape[1], np.nan))


@experimental
def track_path_stats(
    embeddings: ArrayLike,
    track_ids: ArrayLike,
    frame_indices: ArrayLike,
    sequence_indices: ArrayLike,
    *,
    min_points: int = 60,
) -> FactorResult[Any]:
    """Measure how far, and in which direction, each track's embeddings travel.

    A track is a path through embedding space, not a point. This function reads two things off a path: the distance
    between where it starts and where it ends, and the direction from the first to the last. The embeddings are of one
    detection each, in any space, such as the crops of a tracking dataset. A direction describes the path. It does not
    state a cause: object appearance, the surroundings, object scale and camera motion can all move an embedding.

    .. versionadded:: 1.3

    Parameters
    ----------
    embeddings : ArrayLike or Embeddings
        One row per detection, shape ``(N, D)``. Pass the :class:`~dataeval.Embeddings` object itself when you have
        one.
    track_ids : ArrayLike
        The track of each row, shape ``(N,)``. A negative id marks a row no tracker linked, and the row is left out.
    frame_indices : ArrayLike
        The frame of each row, shape ``(N,)``. It orders the rows of a track and need not be consecutive. Two rows of
        one track cannot share a frame.
    sequence_indices : ArrayLike
        The video each row comes from, shape ``(N,)``, because a track id is unique only within one video. It is not
        ``DetectionCrops.item_indices``, which is a position in a frame view. Over a
        :class:`~dataeval.data.SequenceFrames` view, ``frames.frame_map[crops.item_indices][:, 0]`` is the video of
        each crop.
    min_points : int, default 60
        The fewest rows a track needs, at least 3. A shorter track gets NaN. The default is the shortest track the
        study behind this function covers. It is not tuned.

    Returns
    -------
    FactorResult
        One row per track, sorted by video and then track id. ``stats`` holds ``track_ids``, ``item_index`` (the
        video), ``n_points`` (the rows used), ``path_displacement`` and ``path_displacement_null``. ``details`` holds
        ``path_direction``, shape ``(n_tracks, D)``. ``measurement_specs`` records ``min_points`` and the geometry.
        :meth:`~dataeval.Metadata.add_factors` attaches the statistics with ``level="track", key="track_id"``, and
        ignores ``details``.

    Raises
    ------
    ValueError
        If the four inputs differ in length, ``embeddings`` is not two-dimensional or holds a value that is not
        finite, an id is not a whole number, ``min_points`` is not an integer of at least 3, no row has a track id, no
        track has two rows, or one track holds two rows at one frame.

    Warns
    -----
    UserWarning
        If a track has fewer than ``min_points`` rows. The warning counts them, and their statistics are NaN.

    See Also
    --------
    track_stats : Statistics of a track's boxes.
    track_steps : The step each detection takes from the previous one of its track.

    Notes
    -----
    Each row is scaled to unit length, and no video mean is subtracted, so the values of a track depend only on its
    own rows. For a track of ``n`` rows in frame order and ``k = max(n // 3, 1)``:

    - ``path_displacement`` is the Euclidean distance between the mean of the first ``k`` rows and the mean of the
      last ``k`` rows.
    - ``path_direction`` is the unit vector from the first mean to the last mean. It is NaN where
      ``path_displacement`` is 0.
    - ``path_displacement_null`` is the root-mean-square ``path_displacement`` over every order of the track's rows,
      ``sqrt(2 * S / (k * (n - 1)))``, where ``S`` is the sum of squared distances of the rows from their mean. It is
      the noise floor of ``path_displacement`` if the order carried no information. It is exact and uses no random
      numbers.

    ``path_displacement / path_displacement_null`` says how well the direction is determined at the track's length.
    The null falls as ``1 / sqrt(n)``, so the ratio rises with the number of rows. Compare ``path_displacement``
    across tracks. Compare the ratio only between tracks of equal length, such as the same track in two views. The
    ratio is not a score to rank tracks by, and a large ratio does not mean the track is wrong.

    A direction is noisy for one track. The directions of one track in two views, such as an object region and its
    surroundings, agree more than directions of different tracks, but the agreement of a single pair says little.

    The first and last thirds are counted in rows, not frames, so a gap in a track is not corrected for. The study
    behind these statistics covers the seven validation videos of one drone dataset, ground-truth tracks of at least 60
    rows and one ImageNet classifier, so its numbers describe those, and the constants above are provisional.

    Examples
    --------
    A track whose embedding moves along one axis, with a little noise:

    >>> from dataeval.core import track_path_stats
    >>> rng = np.random.default_rng(0)
    >>> rows = rng.normal(0.0, 0.02, (90, 8))
    >>> rows[:, 0] += 1.0
    >>> rows[:, 1] += np.linspace(-0.3, 0.3, 90)
    >>> result = track_path_stats(rows, np.zeros(90, int), np.arange(90), np.zeros(90, int))
    >>> result["stats"]["n_points"].tolist()
    [90]
    >>> round(float(result["stats"]["path_displacement"][0]), 1)
    0.4
    >>> int(np.argmax(result["details"]["path_direction"][0]))
    1
    """
    _check_min_points(min_points)
    x, track, frame, sequence = _read(embeddings, track_ids, frame_indices, sequence_indices)
    order, starts, ends = _group(track, frame, sequence)

    norms = np.linalg.norm(x[order], axis=1)
    unit = np.divide(x[order], norms[:, None], out=np.zeros((len(order), x.shape[1])), where=norms[:, None] > 0)
    displacement = np.full(len(starts), np.nan)
    null = np.full(len(starts), np.nan)
    direction = np.full((len(starts), x.shape[1]), np.nan)
    long_enough = (ends - starts) >= min_points
    for i in np.flatnonzero(long_enough):
        a, b = starts[i], ends[i]
        if (norms[a:b] > 0).all():  # a row of zero length has no direction, so neither does its track
            displacement[i], null[i], direction[i] = _measure(unit[a:b])
    n_short = int((~long_enough).sum())
    if n_short:
        warnings.warn(
            f"{n_short} of {len(starts)} tracks have fewer than {min_points} points and are NaN; "
            "pass `min_points=` to accept values below the studied range.",
            UserWarning,
            stacklevel=3,  # past the experimental wrapper, to the caller
        )

    stats: dict[str, NDArray[Any]] = {
        "track_ids": track[order][starts],
        "item_index": sequence[order][starts],
        "n_points": (ends - starts).astype(np.intp),
        "path_displacement": displacement,
        "path_displacement_null": null,
    }
    return {"stats": stats, "details": {"path_direction": direction}, "measurement_specs": _specs(int(min_points))}
