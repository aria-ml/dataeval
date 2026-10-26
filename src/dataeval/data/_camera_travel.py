"""Keeping a frame each time the estimated camera travel since the last kept frame reaches a set fraction of it."""

from __future__ import annotations

__all__ = []

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval.data._selectors import FrameCandidate, FrameInput, FrameSelector, FrameVerdict

if TYPE_CHECKING:
    from dataeval.core._ego_stats import _FittedMask, _FlowFrame
    from dataeval.protocols import ArrayLike

_logger = get_logger(__name__)


def _step(
    estimator: Any | None,
    previous: _FlowFrame | None,
    current: _FlowFrame | None,
    grid_target: int,
    mask: _FittedMask | None,
) -> tuple[float, float, int, int] | None:
    """Read a pair's translation in pixels with the frame's width and height, or None where it has no trusted reading.

    A non-finite translation counts as no reading, so a NaN never enters the running sum, where it
    would silently stop every later keep.
    """
    from dataeval.core._ego_stats import _measure_pair

    if previous is None or current is None:
        return None
    reading = _measure_pair(estimator, previous, current, grid_target, gap=1, mask=mask)
    if reading is None or not reading.trusted or not (math.isfinite(reading.t0x) and math.isfinite(reading.t0y)):
        return None
    width, height = current.full_wh  # a measured pair shares one original size
    return reading.t0x, reading.t0y, width, height


def _log_unmeasured(unmeasured: int, pairs: int, index: int) -> None:
    """Say once per sequence how many frames were kept for want of a trusted reading."""
    if unmeasured:
        _logger.info(
            "CameraTravel: %d of %d frame pair(s) in sequence %d had no trusted reading; "
            "their frames were kept unmeasured.",
            unmeasured,
            pairs,
            index,
        )


class CameraTravel(FrameSelector):
    """Keep a frame each time the camera's estimated travel since the last kept frame reaches ``distance``.

    Spaces frames by how far the camera has moved, as a fraction of the frame, rather than by time:
    a camera parked on a scene keeps almost nothing, and one sweeping across it keeps a frame every
    ``distance`` of the way. The first frame of a sequence is always kept.

    Parameters
    ----------
    distance : float
        Net image translation, as a fraction of the frame's width (horizontal) or height (vertical),
        at which the next frame is kept. Must be positive and finite. ``0.25`` keeps a frame each time
        the view has slid a quarter of the frame.
    mask : ArrayLike or None, default None
        A 2D (height, width) image of a burned-in overlay such as a HUD, where nonzero marks the
        overlay, as ``mask`` in :func:`~dataeval.core.ego_stats`. The camera is read from the rest of
        the frame. A pair whose samples outside the overlay do not reach three quadrants of the frame
        is kept unmeasured. None, or a mask with no nonzero pixel, reads the whole frame.

    Raises
    ------
    ValueError
        If ``distance`` is not a positive finite number, or ``mask`` is not a 2D image. Raised while
        selecting if ``mask`` does not fit a frame's size.
    ImportError
        If opencv is not installed.

    See Also
    --------
    :func:`~dataeval.core.ego_stats` : Measure camera motion across a whole video rather than select on it
    :class:`Redundancy` : Thin by how much a frame's content changed rather than how far the camera moved

    Notes
    -----
    **This measures apparent image motion, not new ground.** Zoom, rotation, parallax, independent
    movers and revisited places all complicate the reading, so a kept frame is one the view has
    slid away from, not necessarily one that shows somewhere new.

    **It is net displacement, not path length.** Reversals cancel while the travel is still short of
    ``distance``, so a camera sweeping back and forth within that distance keeps nothing new. Travel
    restarts at each kept frame, though, so a return to an earlier view, once kept, can be kept again.

    **Each adjacent pair goes through the same measurement** :func:`~dataeval.core.ego_stats` makes, at
    its defaults, which cannot be configured here: frames processed to a 360 px short side, DIS optical
    flow, and a grid target of 40. ``ego_stats`` also applies a temporal outlier gate, which compares
    every pair to the whole video's readings. A single-pass selector cannot see the whole video without
    buffering it, so that gate is skipped here.

    **A hard cut is refused, and the frame after it is kept.** A hard cut, or a large switch of field of view
    (about 2x), is a pair whose fitted motion does not explain the new frame, so it has no trusted reading.
    The first frame of the new shot is kept, marked unmeasured, as the new anchor, and the travel restarts
    there. A still shot after a hard cut therefore has its first frame kept. A gradual transition, such as a
    dissolve, is not refused, as the fitted motion explains each pair of it, so a still shot after one can
    still go unkept for its whole length. A smaller switch of field of view, such as 1.5x, can read as a zoom.

    **A burned-in overlay reads as scene content.** Without ``mask``, a dense one, such as a HUD of
    telemetry text, can drag a pan's reading slower, so frames are kept too sparsely across it, or
    split it into two motions, which is refused, so its frames are kept unmeasured; a light one gets
    pans refused too. ``mask`` excludes it, and also drops samples within 32 px of it at the 360 px
    processing size, so movers weigh more in what is left, and one holding most of what is left can
    be read as the camera.

    **No reading is no evidence of no motion.** These keep the frame, marked
    :attr:`~dataeval.data.FrameVerdict.unmeasured`, and restart the travel there:

    - a pair with no reading;
    - a pair that ``ego_stats``' per-pair trust flags refuse, including one whose fitted motion does not
      explain the new frame, as at a hard cut or a large switch of field of view, or where nothing in the
      frame confirms the fit, as on a nearly textureless noisy frame;
    - a pair whose frames differ in size, are too small for optical flow, hold a non-finite pixel, or
      include a flat frame, a single gray level such as a black dropout.

    An unreadable frame refuses both pairs it belongs to. A log line at info level counts them once
    per sequence.

    In a :class:`~dataeval.data.Chain` it reads the frames on either side of each one, so it must come
    first, or follow only :class:`~dataeval.data.AllFrames`.

    Examples
    --------
    >>> from dataeval.data import CameraTravel, SequenceFrames
    >>> frames = SequenceFrames(mot_dataset, CameraTravel(0.25))  # doctest: +SKIP
    """

    needs: FrameInput = FrameInput.PIXELS
    _needs_adjacent_frames: bool = True

    def __init__(self, distance: float, mask: ArrayLike | None = None) -> None:
        if not (math.isfinite(distance) and distance > 0):
            raise ValueError(f"CameraTravel: distance must be a positive finite fraction of the frame; got {distance}.")
        from dataeval.core._ego_stats import _as_mask, _require_cv2

        _require_cv2()
        self.distance: float = float(distance)
        self.mask: NDArray[np.bool_] | None = _as_mask(mask, "mask")

    def _repr_overrides(self) -> dict[str, str]:
        if self.mask is None:
            return {}
        height, width = self.mask.shape
        return {"mask": f"<{width}x{height}, coverage {self.mask.mean():.3f}>"}

    def select(self, frames: Iterator[FrameCandidate]) -> Iterator[FrameVerdict]:
        """Keep the first frame, then each frame the travel since the last kept one has brought to ``distance``."""
        from dataeval.core._ego_stats import _flow_estimator, _flow_frame, _OverlayMask
        from dataeval.core._video_runner import FrameView, VideoOptions

        options = VideoOptions(ego_mask=self.mask)
        estimator = _flow_estimator(options.ego_flow)
        overlay = None if self.mask is None else _OverlayMask(self.mask, "mask")
        previous: _FlowFrame | None = None
        travel_x = travel_y = 0.0
        pairs = unmeasured = 0
        index = 0
        for frame in frames:
            index = frame.sequence.index
            current = _flow_frame(FrameView(frame, frame.position, options))
            fitted = None if overlay is None or current is None else overlay.fit(current.full_wh, current.gray.shape)
            first = frame.position == 0
            step = None if first else _step(estimator, previous, current, options.ego_grid_target, fitted)
            previous = current  # the only frame that outlives this iteration
            if first:
                yield FrameVerdict(frame.position)
                continue
            pairs += 1
            if step is None:
                # No reading is no evidence that the camera stood still, so the frame is kept rather
                # than dropped on an assumption, marked as kept unmeasured, and anchors the travel.
                unmeasured += 1
                travel_x = travel_y = 0.0
                yield FrameVerdict(frame.position, unmeasured=True)
                continue
            # ponytail: translation only, zoom and rotation ignored; fold in the divergence ego_stats already estimates
            tx, ty, width, height = step
            travel_x += tx
            travel_y += ty
            if max(abs(travel_x) / width, abs(travel_y) / height) >= self.distance:
                travel_x = travel_y = 0.0
                yield FrameVerdict(frame.position)
        _log_unmeasured(unmeasured, pairs, index)
