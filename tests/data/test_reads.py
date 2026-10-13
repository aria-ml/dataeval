import numpy as np
import pytest
from polars.testing import assert_frame_equal

from dataeval import Metadata
from dataeval.data import ClassBalance, Indices, Limit, Resize, Shuffle, View
from dataeval.data._reads import Reads, skeleton


def _structured_alike(dataset) -> None:
    """Metadata over a dataset's skeletons is the Metadata over the dataset itself."""
    direct = Metadata(dataset)
    via_reads = Metadata(Reads([skeleton(dataset[i]) for i in range(len(dataset))], dataset.metadata))
    assert_frame_equal(direct.dataframe, via_reads.dataframe)
    assert list(direct.factor_names) == list(via_reads.factor_names)
    np.testing.assert_array_equal(direct.factor_data, via_reads.factor_data)
    assert dict(direct.dropped_factors) == dict(via_reads.dropped_factors)


@pytest.mark.required
class TestSkeleton:
    def test_an_image_is_dropped(self, factor_od):
        item, target, metadata = skeleton(factor_od[0])
        assert item is None
        assert target.labels.tolist() == [0, 1]
        assert metadata["tod"] == "night"

    def test_a_frame_keeps_its_number_and_shape_but_no_pixels(self, factor_videos):
        frames, _, _ = skeleton(factor_videos[0])
        assert [frame.frame_index for frame in frames] == [0, 1, 2]
        assert frames[0].pixels.shape == (3, 8, 8)
        assert not hasattr(frames[0].pixels, "dtype")
        assert not hasattr(frames[0], "time_s")

    def test_reads_is_a_dataset(self, factor_od):
        reads = Reads([skeleton(factor_od[i]) for i in range(2)], factor_od.metadata)
        assert len(reads) == 2
        assert reads[1][1].labels.tolist() == [1]
        assert reads.metadata is factor_od.metadata


@pytest.mark.required
class TestSkeletonsStructureLikeTheDataset:
    def test_classification(self, factor_ic):
        _structured_alike(factor_ic)

    def test_detection(self, factor_od):
        _structured_alike(factor_od)

    def test_tracking(self, factor_videos):
        _structured_alike(factor_videos)

    def test_views_with_reordering_duplicates_and_resizing(self, factor_od, factor_ic, factor_videos):
        _structured_alike(View(factor_od, [Shuffle(), Limit(4)]))
        _structured_alike(View(factor_od, [ClassBalance("interclass", num_samples=8), Resize((2, 2))]))
        _structured_alike(View(factor_ic, Indices([4, 0, 0, 2])))
        _structured_alike(View(factor_videos, [Indices([2, 0]), Limit(2)]))
