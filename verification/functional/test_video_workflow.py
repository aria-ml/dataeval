"""Verify that the video workflow holds from a selected frame to a partition.

Each step of the workflow has its own tests. This one runs them in the order an integrator does, on one dataset: thin
the videos, attach statistics to the metadata, let ``Duplicates`` and ``Outliers`` find a planted copy and a planted
flash, follow each finding to the video and frame it came from, and split the frames without splitting a video.

Maps to meta repo test cases:
  - TC-13.1: Video dataset workflow
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

from dataeval import Metadata
from dataeval.core import image_stats, video_stats
from dataeval.data import SequenceFrames, SourceLocator, Stride, split_dataset
from dataeval.flags import ImageStats, VideoStats
from dataeval.quality import Duplicates, Outliers
from dataeval.types import SourceIndex
from verification.helpers import SimpleTrackingDataset, make_tracking_dataset

N_VIDEOS, N_FRAMES, STEP = 6, 24, 2
COPIED = ((0, 6), (0, 8), (0, 10))  # (video, frame) of the originals
COPIES = ((4, 12), (4, 14), (4, 16))  # where they were planted
FLASH = (2, 8)


@dataclass
class Workflow:
    """The dataset, its frame view and the metadata the statistics were attached to."""

    dataset: SimpleTrackingDataset
    frames: SequenceFrames
    metadata: Metadata
    flash_pixels: np.ndarray


@pytest.fixture(scope="module")
def workflow() -> Workflow:
    dataset = make_tracking_dataset((N_FRAMES,) * N_VIDEOS)
    for (source_video, source_frame), (video, frame) in zip(COPIED, COPIES, strict=True):
        dataset.plant_frame(video, frame, dataset.frame_pixels(source_video, source_frame).copy())
    flash = np.full((3, 16, 16), 255, dtype=np.uint8)
    dataset.plant_frame(*FLASH, flash)

    frames = SequenceFrames(dataset, Stride(STEP))
    metadata = Metadata(frames.source)
    metadata.add_factors(video_stats(dataset, stats=VideoStats.DESCRIPTIVE))
    stats = image_stats(frames, stats=ImageStats.VISUAL_BRIGHTNESS, per_target=False)
    metadata.add_factors(stats, source_index=frames.to_source(stats["source_index"]))
    return Workflow(dataset, frames, metadata, flash)


class TestVideoWorkflow:
    """Verify the chain from frame selection to a safe partition on one tracking dataset."""

    def test_selection_records_the_source_of_every_kept_frame(self, workflow: Workflow):
        frames = workflow.frames
        assert len(frames) == N_VIDEOS * N_FRAMES // STEP
        expected = [
            SourceIndex(video, frame, "unit") for video in range(N_VIDEOS) for frame in range(0, N_FRAMES, STEP)
        ]
        assert list(frames.source_indices) == expected
        represented = [cast(dict[str, Any], meta)["frames_represented"] for _, _, meta in frames.stream()]
        assert represented == [STEP] * len(frames)

    def test_statistics_attach_to_the_rows_they_describe(self, workflow: Workflow):
        sequences = workflow.metadata.rows_at("sequence")
        assert sequences["item_index"].to_list() == list(range(N_VIDEOS))
        assert sequences["n_frames"].to_list() == [N_FRAMES] * N_VIDEOS
        units = workflow.metadata.rows_at("unit")
        measured = units.filter(units["brightness"].is_not_null())
        kept = {(a.item, a.key) for a in workflow.frames.source_indices}
        assert {(row["item_index"], row["unit_index"]) for row in measured.iter_rows(named=True)} == kept

    def test_outliers_reports_the_flash_by_video_and_frame(self, workflow: Workflow):
        flagged = list(Outliers().evaluate(workflow.frames).frames.outliers)
        assert flagged == [SourceIndex(*FLASH, "unit")]

    def test_duplicates_reports_the_copies_by_video_and_frame(self, workflow: Workflow):
        found: Sequence[Any] = Duplicates().evaluate(workflow.frames).exact
        pairs = {frozenset(map(tuple, group)) for group in found}
        assert pairs == {frozenset({original, copy}) for original, copy in zip(COPIED, COPIES, strict=True)}

    def test_a_finding_resolves_to_the_frame_it_names(self, workflow: Workflow):
        locator = SourceLocator(workflow.dataset)
        flash = locator[SourceIndex(*FLASH, "unit")]
        assert np.array_equal(flash.pixels, workflow.flash_pixels)
        for (source_video, source_frame), (video, frame) in zip(COPIED, COPIES, strict=True):
            copy = locator[SourceIndex(video, frame, "unit")]
            assert np.array_equal(copy.pixels, workflow.dataset.frame_pixels(source_video, source_frame))

    def test_a_view_position_translates_to_its_source_address(self, workflow: Workflow):
        frames = workflow.frames
        assert frames.to_source([SourceIndex(len(frames) - 1)]) == [SourceIndex(N_VIDEOS - 1, N_FRAMES - STEP, "unit")]

    def test_splitting_on_sequence_keeps_every_video_in_one_partition(self, workflow: Workflow):
        splits = split_dataset(workflow.frames, split_on=["sequence"], test_frac=0.2, val_frac=0.2)
        fold = splits.folds[0]
        videos = [
            {workflow.frames.source_indices[int(i)].item for i in partition}
            for partition in (fold.train, fold.val, splits.test)
        ]
        assert all(videos)
        assert sum(len(v) for v in videos) == len(set().union(*videos)) == N_VIDEOS
        assert sum(len(p) for p in (fold.train, fold.val, splits.test)) == len(workflow.frames)
