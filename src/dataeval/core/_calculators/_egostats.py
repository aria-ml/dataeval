__all__ = []

from collections.abc import Mapping
from typing import Any, ClassVar

from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.core._calculators._video_base import ReadingCalculator
from dataeval.core._ego_stats import EGO_AGGREGATIONS, _EgoEngine
from dataeval.core._video_runner import FrameView, VideoClock, VideoOptions
from dataeval.flags import VideoStats
from dataeval.types import Aggregator

# Each flag names its reduction and factor, and the declaration is looked up in EGO_AGGREGATIONS,
# so the output name is the one Metadata gives the same roll-up and the two cannot drift.
_FLAGS: dict[VideoStats, tuple[str, str]] = {
    VideoStats.EGO_PAN_SPEED: ("median", "pan_speed"),
    VideoStats.EGO_ZOOM_RATE: ("median", "zoom_rate"),
    VideoStats.EGO_ROLL_RATE: ("median", "roll_rate"),
    VideoStats.EGO_SHEAR_RATE: ("median", "shear_rate"),
    VideoStats.EGO_MOVER_FRAC: ("median", "mover_frac"),
    VideoStats.EGO_MOVER_SPEED: ("median", "mover_speed"),
    VideoStats.EGO_PAN_X_VARIABILITY: ("variability", "pan_x"),
    VideoStats.EGO_PAN_Y_VARIABILITY: ("variability", "pan_y"),
    VideoStats.EGO_ZOOM_EXCURSION: ("abs_sum", "zoom_rate"),
    VideoStats.EGO_ROLL_EXCURSION: ("abs_sum", "roll_rate"),
    VideoStats.EGO_MOTION: ("mode", "motion"),
}
_DECLARED = {(agg.how, factor): agg for agg in EGO_AGGREGATIONS for factor in agg.factors}


@CalculatorRegistry.register(VideoStats)
class EgoStatCalculator(ReadingCalculator):
    """Per-frame camera ego-motion, from one optical-flow fit per frame pair.

    Every flag reads the same fit, so selecting any of them pays for all of it.
    """

    aggregations: ClassVar[Mapping[VideoStats, tuple[Aggregator, str]]] = {
        flag: (_DECLARED[how, factor], factor) for flag, (how, factor) in _FLAGS.items()
    }

    def __init__(self, options: VideoOptions) -> None:
        super().__init__(options)
        # Loads OpenCV, which is why this calculator is built only when an EGO flag is selected.
        self.engine = _EgoEngine(options)

    def get_applicable_flags(self) -> VideoStats:
        """Return which flags this calculator handles."""
        return VideoStats.EGO

    def update(self, view: FrameView) -> None:
        """Measure the frame pair the frame closes."""
        self.engine.update(view)

    def finish(self, clock: VideoClock) -> None:
        """Close out the video."""
        self.engine.finish(clock)

    def readings(self) -> dict[str, list[Any]]:
        """Return the video's per-frame camera-motion rows."""
        return self.engine.frame_rows()
