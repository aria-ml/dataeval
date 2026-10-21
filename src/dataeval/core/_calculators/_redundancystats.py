__all__ = []

from dataeval.core._calculators._base import Handler
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.core._calculators._video_base import VideoCalculator
from dataeval.core._hash_index import pack_hashes
from dataeval.core._temporal import RedundantRunResult, redundant_runs
from dataeval.core._video_runner import FrameView, VideoClock, VideoOptions
from dataeval.flags import VideoStats


@CalculatorRegistry.register(VideoStats)
class RedundancyStatCalculator(VideoCalculator):
    """Share of frames, and longest stretch, that carry nothing new over the frame before."""

    def __init__(self, options: VideoOptions) -> None:
        super().__init__(options)
        self.hashes: list[str] = []
        self.runs: RedundantRunResult | None = None

    def get_applicable_flags(self) -> VideoStats:
        """Return which flags this calculator handles."""
        return VideoStats.REDUNDANCY

    def update(self, view: FrameView) -> None:
        """Hash one frame."""
        self.hashes.append(view.phash)

    def finish(self, clock: VideoClock) -> None:  # noqa: ARG002
        """Find the redundant runs among the video's hashes, in stream order."""
        codes, valid = pack_hashes(self.hashes)
        self.runs = redundant_runs(codes, self.options.redundancy_radius, valid=valid)

    def _fraction(self) -> list[float]:
        return [self.runs["redundant_fraction"] if self.runs else 0.0]

    def _longest(self) -> list[int]:
        return [int(self.runs["length"].max()) if self.runs and self.runs["length"].size else 0]

    def get_handlers(self) -> dict[VideoStats, Handler]:
        """Return the statistic each flag produces."""
        return {
            VideoStats.REDUNDANCY_FRACTION: Handler("redundant_fraction", self._fraction),
            VideoStats.REDUNDANCY_LONGEST_RUN: Handler("longest_run", self._longest),
        }
