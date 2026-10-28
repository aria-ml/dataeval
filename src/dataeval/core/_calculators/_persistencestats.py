__all__ = []

from functools import partial

import numpy as np
from numpy.typing import NDArray

from dataeval.core._calculators._base import Handler
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.core._calculators._video_base import VideoCalculator
from dataeval.core._ego_stats import _require_cv2
from dataeval.core._persistence import REPORT_GRID, CameraMaps, Sample, StartSampler, camera_maps, sample, solved_grid
from dataeval.core._video_runner import FrameView, VideoClock, VideoOptions
from dataeval.flags import VideoStats


@CalculatorRegistry.register(VideoStats)
class PersistenceStatCalculator(VideoCalculator):
    """Per-video maps of what is fixed to the camera, from an evenly spread, bounded sample of its frames.

    Every flag reads the same maps: selecting any computes ``ego_map``, and a blur flag adds the
    one solve that gives both blur maps.
    """

    def __init__(self, options: VideoOptions) -> None:
        super().__init__(options)
        # Loads OpenCV, which is why this calculator is built only when a PERSISTENCE flag is selected.
        _require_cv2()
        self.sampler = StartSampler(options.persistence_samples)
        self.samples: dict[int, Sample] = {}
        self.shape: tuple[int, ...] | None = None
        self.grid: tuple[int, int] | None = None
        self.maps: CameraMaps | None = None

    def get_applicable_flags(self) -> VideoStats:
        """Return which flags this calculator handles."""
        return VideoStats.PERSISTENCE

    def update(self, view: FrameView) -> None:
        """Keep the frame if it starts a frame pair or completes one, dropping starts the sample has outgrown."""
        blur = self.options.blur_measure is not None
        # Checked before the pixels are read, so a frame that can be neither a start nor a
        # partner is never converted.
        if view.index % self.sampler.stride and not self.sampler.needs(view.index, blur=blur):
            return
        gray = view.gray
        # Refused before it is offered, so a frame of another size never takes a start's place.
        grid = self._grid_for(gray)
        if grid is None:
            return
        stride = self.sampler.stride
        self.sampler.offer(view.index)
        if self.sampler.stride != stride:
            self.samples = {k: s for k, s in self.samples.items() if self.sampler.needs(k, blur=blur)}
        if self.sampler.needs(view.index, blur=blur):
            self.samples[view.index] = sample(gray, grid, self.options.blur_measure)

    def _grid_for(self, gray: NDArray[np.uint8]) -> tuple[int, int] | None:
        """Return the solved grid if `gray` has the first frame's size, which it sets; else warn once per size."""
        if self.grid is None:
            self.shape = gray.shape
            self.grid = solved_grid(*gray.shape)
        if gray.shape == self.shape:
            return self.grid
        height, width = self.shape or (0, 0)
        message = (
            f"Frames of {gray.shape[1]}x{gray.shape[0]} are left out of the PERSISTENCE maps, which read only "
            f"frames of the first frame's size, {width}x{height}."
        )
        if message not in self.warnings:
            self.warnings.append(message)
        return None

    def finish(self, clock: VideoClock) -> None:  # noqa: ARG002
        """Solve the video's maps from its samples, then let the samples go."""
        self.maps = camera_maps(
            self.samples,
            self.sampler.kept,
            self.grid,
            blur=self.options.blur_measure is not None,
            sigma=self.options.blur_sigma,
        )
        self.warnings.extend(self.maps.warnings)
        self.samples = {}

    def _map(self, name: str) -> list[NDArray[np.float64]]:
        return [getattr(self.maps, name) if self.maps else np.full(REPORT_GRID, np.nan)]

    def get_handlers(self) -> dict[VideoStats, Handler]:
        """Return the map each flag produces."""
        return {
            VideoStats.PERSISTENCE_EGO: Handler("ego_map", partial(self._map, "ego_map")),
            VideoStats.PERSISTENCE_BLUR: Handler("blur_map", partial(self._map, "blur_map")),
            VideoStats.PERSISTENCE_BLUR_RAW: Handler("blur_map_raw", partial(self._map, "blur_map_raw")),
        }
