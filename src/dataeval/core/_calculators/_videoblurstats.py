__all__ = []

import math
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from dataeval.core._blur import BlurValues, blur_stat
from dataeval.core._calculators._base import Handler
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.core._calculators._video_base import VideoCalculator
from dataeval.core._ego_stats import _downscale, _require_cv2
from dataeval.core._persistence import StartSampler
from dataeval.core._video_runner import FrameView, VideoClock, VideoOptions
from dataeval.flags import VideoStats

# Short side, in pixels, that a frame is shrunk to before it is measured. A video of 720 or more is measured at this
# one scale, but whether such videos of different native sizes then compare was not tested. A smaller video is
# measured at its own size. A video reports the short side it was measured at, so `video_stats` can warn when a call's
# videos were measured at more than one.
SHORT_SIDE = 720


def _median(values: Iterable[float]) -> float:
    """Return the median of the finite values, NaN when there are none."""
    finite = [v for v in values if math.isfinite(v)]
    return float(np.median(finite)) if finite else math.nan


@CalculatorRegistry.register(VideoStats)
class VideoBlurStatCalculator(VideoCalculator):
    """Per-video blur level: the median of the edge-width values of an evenly spread frame sample.

    A frame is measured as ``blur_stat`` measures an image, after shrinking it to a 720 short side if it is larger.
    The frames are the starts a ``StartSampler`` keeps, so the sample covers the whole video however long it turns
    out to be. Only frames of the first frame's size are measured, so every frame in the median is at one scale.
    """

    def __init__(self, options: VideoOptions) -> None:
        super().__init__(options)
        # Loads OpenCV for the shared downscale, which is why this is built only when a BLUR flag is selected.
        _require_cv2()
        self.sampler = StartSampler(options.blur_samples)
        self.values: dict[int, BlurValues] = {}
        self.level = math.nan
        self.shape: tuple[int, ...] | None = None
        # The short side the video was measured at, None until a frame has been; `_measure` hands it to `video_stats`.
        self.short_side: int | None = None

    def get_applicable_flags(self) -> VideoStats:
        """Return which flags this calculator handles."""
        return VideoStats.BLUR

    def update(self, view: FrameView) -> None:
        """Measure the frame if it becomes a kept start, dropping the starts the sample has outgrown."""
        # A frame off the current stride is refused here, before its pixels are read.
        if view.index % self.sampler.stride:
            return
        gray = view.gray
        # Refused before it is offered, so a frame of another size never takes a start's place.
        if not self._first_size(gray):
            return
        stride = self.sampler.stride
        offered = self.sampler.offer(view.index)
        if self.sampler.stride != stride:
            kept = set(self.sampler.kept)
            self.values = {k: v for k, v in self.values.items() if k in kept}
        if offered:
            # ponytail: a frame is computed when offered and dropped when the stride doubles, so more than
            # `blur_samples` frames are computed: up to half the ceiling per doubling, which is log growth in the
            # video's length. Decide the stride from the frame count when it is known, to compute only the kept.
            self.values[view.index] = blur_stat(_downscale(gray, SHORT_SIDE).astype(np.float32))

    def _first_size(self, gray: NDArray[np.uint8]) -> bool:
        """Return whether `gray` has the first frame's size, which it sets; else warn once per size."""
        first = self.shape
        if first is None:
            first = self.shape = gray.shape
            self.short_side = min(*first, SHORT_SIDE)
        if gray.shape == first:
            return True
        message = (
            f"Frames of {gray.shape[1]}x{gray.shape[0]} are left out of the BLUR median, which reads only "
            f"frames of the first frame's size, {first[1]}x{first[0]}."
        )
        if message not in self.warnings:
            self.warnings.append(message)
        return False

    def finish(self, clock: VideoClock) -> None:  # noqa: ARG002
        """Reduce the sample to the video's median, then let the sample go."""
        self.level = _median(v.blur for v in self.values.values())
        self.values = {}

    def get_handlers(self) -> dict[VideoStats, Handler]:
        """Return the value each flag produces."""
        return {
            VideoStats.BLUR_LEVEL: Handler("blur_level", lambda: [self.level]),
        }
