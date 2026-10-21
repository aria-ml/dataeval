__all__ = []

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

from dataeval.core._calculators._base import Calculator, Handler
from dataeval.core._video_runner import FrameConsumer, VideoOptions
from dataeval.flags import VideoStats
from dataeval.types import Aggregator


class VideoCalculator(Calculator[VideoStats], FrameConsumer):
    """A calculator built once per video, fed every frame through ``update``, then ``finish``ed.

    Scalar calculators answer through ordinary :class:`Handler` s, one finished value per video,
    exactly as image calculators do.
    """

    def __init__(self, options: VideoOptions) -> None:
        self.options = options
        self.warnings: list[str] = []


class ReadingCalculator(VideoCalculator):
    """A calculator that records one reading per frame and declares how each reduces to one value.

    The runner reduces the readings, so the calculator has no handlers of its own.

    Attributes
    ----------
    aggregations : Mapping[VideoStats, tuple[Aggregator, str]]
        Per flag, the declaration that reduces a reading column and the column (factor) it reads.
    """

    aggregations: ClassVar[Mapping[VideoStats, tuple[Aggregator, str]]]

    @classmethod
    def stat_names(cls, flags: VideoStats) -> set[str]:
        """Return the output names of the aggregations `flags` selects."""
        return {agg.name_for(factor) for flag, (agg, factor) in cls.aggregations.items() if flag in flags}

    def get_handlers(self) -> dict[VideoStats, Handler]:
        """Return no handlers: the runner reduces the readings instead."""
        return {}

    @abstractmethod
    def readings(self) -> dict[str, list[Any]]:
        """Return this video's readings, one entry per frame per column."""
