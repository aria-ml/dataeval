__all__ = []

import math

from dataeval.core._calculators._base import Handler
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.core._calculators._video_base import VideoCalculator
from dataeval.core._video_runner import FrameView, VideoClock, VideoOptions
from dataeval.flags import VideoStats


@CalculatorRegistry.register(VideoStats)
class DescriptiveStatCalculator(VideoCalculator):
    """Frame count, duration and frame rate, from the video clock alone. Never reads pixels."""

    def __init__(self, options: VideoOptions) -> None:
        super().__init__(options)
        self.count = 0
        self.clock = VideoClock(None, math.nan, math.nan)

    def get_applicable_flags(self) -> VideoStats:
        """Return which flags this calculator handles."""
        return VideoStats.DESCRIPTIVE

    def update(self, view: FrameView) -> None:  # noqa: ARG002
        """Count one frame."""
        self.count += 1

    def finish(self, clock: VideoClock) -> None:
        """Keep the video's clock."""
        self.clock = clock

    def _frame_rate(self) -> list[float]:
        duration = self.clock.duration
        return [self.count / duration if math.isfinite(duration) and duration > 0 else math.nan]

    def get_handlers(self) -> dict[VideoStats, Handler]:
        """Return the statistic each flag produces."""
        return {
            VideoStats.DESCRIPTIVE_FRAME_COUNT: Handler("n_frames", lambda: [self.count]),
            VideoStats.DESCRIPTIVE_DURATION: Handler("video_duration", lambda: [self.clock.duration]),
            VideoStats.DESCRIPTIVE_FRAME_RATE: Handler("frame_rate", self._frame_rate),
        }
