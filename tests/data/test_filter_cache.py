import pickle
from collections import Counter
from typing import Any

import numpy as np
import pytest
from polars.testing import assert_frame_equal

from dataeval import Metadata
from dataeval.data import ClassBalance, Filter, Indices, Limit, Relabel, Resize, Reverse, Shuffle, View
from tests.data.test_filter import _Segmentations


@pytest.mark.required
class TestViewReadCache:
    def test_a_view_starts_without_one(self, factor_od):
        assert View(factor_od, Limit(3))._reads is None

    def test_skeletons_start_it_in_selection_order(self, factor_od):
        view = View(factor_od, Limit(3))
        view.selection = [2, 0, 0]
        skeletons = view._skeletons()
        assert [s[1].labels.tolist() for s in skeletons] == [[2, 2], [0, 1], [0, 1]]
        assert view._reads is not None
        assert set(view._reads) == {0, 2}
        assert all(s[0] is None for s in skeletons)

    def test_structuring_source_is_the_view_until_there_are_reads(self, factor_od):
        view = View(factor_od, Limit(3))
        assert view._structuring_source() is view
        view._skeletons()
        assert len(view._structuring_source()) == 3

    def test_map_clears_it(self, factor_od):
        view = View(factor_od, Limit(3))
        view._skeletons()
        view.map(lambda datum: datum)
        assert view._reads is None

    def test_map_each_clears_it(self, factor_od):
        view = View(factor_od, Limit(3))
        view._skeletons()
        view.map_each({0: lambda datum: datum})
        assert view._reads is None

    def test_keep_reads_prunes_and_applies(self, factor_od):
        view = View(factor_od, Limit(3))
        view._skeletons()
        reads = view._reads
        assert reads is not None
        view.selection = [2]
        view._keep_reads(reads, {2: lambda s: (s[0], s[1], {**s[2], "seen": True})})
        assert view._reads is not None
        assert set(view._reads) == {2}
        assert view._reads[2][2]["seen"] is True

    def test_pickling_drops_it(self, factor_od):
        view = View(factor_od, Limit(3))
        view._skeletons()
        copy = pickle.loads(pickle.dumps(view))
        assert copy._reads is None
        assert view._reads is not None


class _Counting:
    """A dataset that counts how often each item is read."""

    def __init__(self, dataset: Any) -> None:
        self._dataset = dataset
        self.reads: Counter[int] = Counter()
        self.metadata = dataset.metadata

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Any:
        self.reads[index] += 1
        return self._dataset[index]


class _Forwarding:
    """A wrapper that forwards every attribute it lacks to the dataset inside it, as maite_datasets' wrappers do, and
    adds a factor to each datum it reads."""

    def __init__(self, dataset: Any) -> None:
        self._dataset = dataset

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)

    @property
    def metadata(self) -> Any:
        return self._dataset.metadata

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Any:
        image, target, metadata = self._dataset[index]
        return image, target, {**metadata, "wrapped": 1}


@pytest.mark.required
class TestFilterKeepsTheCache:
    def test_the_structured_path_starts_it_pruned_and_updated(self, factor_od):
        view = View(factor_od, Filter("occl < 0.5"))
        reads = view._reads
        assert reads is not None
        assert set(reads) == {0, 2, 5}
        assert reads[0][1].labels.tolist() == [0]

    def test_the_class_path_does_not_start_it_on_detections(self, factor_od):
        assert View(factor_od, Filter(classes=[0]))._reads is None

    def test_the_class_path_does_not_start_it_on_segmentation(self):
        assert View(_Segmentations([[0, 1], [2]]), Filter(classes=[0]))._reads is None

    def test_the_class_path_does_not_start_it_on_classification(self, factor_ic):
        assert View(factor_ic, Filter(classes=[0]))._reads is None

    def test_the_class_path_does_not_start_it_on_tracking(self, factor_videos):
        assert View(factor_videos, Filter(classes=[0]))._reads is None

    def test_the_class_path_maintains_one_on_detections(self, factor_od):
        view = View(factor_od, [Filter("tod = 'night'"), Filter(classes=[2])])
        reads = view._reads
        assert reads is not None
        assert set(reads) == {2, 5}
        assert reads[5][1].labels.tolist() == [2]

    def test_the_class_path_maintains_one_on_tracking(self, factor_videos):
        view = View(factor_videos, [Filter("weather = 'rain'"), Filter(classes=[1])])
        reads = view._reads
        assert reads is not None
        assert set(reads) == {0, 2}
        assert [f.track_ids.tolist() for f in reads[2][1].frame_tracks] == [[], [3], []]

    def test_chained_filters_read_each_item_once(self, factor_od):
        counting = _Counting(factor_od)
        View(counting, [Filter("tod = 'night'"), Limit(3), Filter(classes=[2])])
        expected = Counter(range(len(factor_od)))
        expected[0] += 1  # View.__init__ validates the dataset by reading its first item
        assert counting.reads == expected


def _same_with_and_without_cache(view: View[Any], partial_factors: bool = False) -> None:
    # Metadata structures on first use, so each is used while the view holds the cache it is meant to read.
    cached = Metadata(view, partial_factors=partial_factors)
    _ = cached.dataframe
    reads, view._reads = view._reads, None
    fresh = Metadata(view, partial_factors=partial_factors)
    _ = fresh.dataframe
    view._reads = reads
    assert_frame_equal(cached.dataframe, fresh.dataframe)
    assert list(cached.factor_names) == list(fresh.factor_names)
    np.testing.assert_array_equal(cached.factor_data, fresh.factor_data)
    assert dict(cached.dropped_factors) == dict(fresh.dropped_factors)


OD_SEQUENCES = {
    "image predicate": lambda: [Filter("tod = 'night'")],
    "detection predicate, limit": lambda: [Filter("occl < 0.5"), Limit(2)],
    "clean, shuffle": lambda: [Filter("occl < 0.5", drop="detections"), Shuffle()],
    "balance, classes": lambda: [ClassBalance("interclass", num_samples=8), Filter(classes=[0, 2])],
    "relabel, classes": lambda: [
        Relabel({"car": "vehicle", "person": "person", "boat": "vehicle"}, ["vehicle", "person"]),
        Filter(classes=[0]),
    ],
    "image predicate, classes": lambda: [Filter("tod = 'night'"), Filter(classes=[0, 2])],
    "filter, duplicate": lambda: [Filter("tod = 'night'"), Indices([3, 0, 0])],
    "filter, resize": lambda: [Filter("tod = 'night'"), Resize((2, 2))],
}
IC_SEQUENCES = {
    "image predicate": lambda: [Filter("tod = 'night'")],
    "classes, shuffle": lambda: [Filter(classes=[1]), Shuffle()],
    "image predicate, classes": lambda: [Filter("tod = 'night'"), Filter(classes=[1])],
    "duplicate, filter": lambda: [Indices([4, 0, 0, 2]), Filter("tod = 'night'")],
}
# Relabel, ClassBalance and Resize do not support tracking data, so they stay out of this matrix.
MOT_SEQUENCES = {
    "sequence predicate": lambda: [Filter("weather = 'rain'")],
    "clean tracks, limit": lambda: [Filter("track_length > 2", drop="detections"), Limit(2)],
    "detections, reverse": lambda: [Filter("class_label = 0"), Reverse()],
    "keep, untracked": lambda: [Indices([2, 0]), Filter("track_id >= 0 AND target_index >= 0", drop="detections")],
}


@pytest.mark.required
class TestMetadataFromTheCache:
    @pytest.mark.parametrize("name", list(OD_SEQUENCES))
    def test_detection(self, factor_od, name):
        _same_with_and_without_cache(View(factor_od, OD_SEQUENCES[name]()))

    @pytest.mark.parametrize("name", list(IC_SEQUENCES))
    def test_classification(self, factor_ic, name):
        _same_with_and_without_cache(View(factor_ic, IC_SEQUENCES[name]()))

    # The timed videos' frames declare time_s and pts, which a tracking skeleton keeps from the stream; one frame
    # declares no time_s, so time_s is a column only with partial_factors.
    @pytest.mark.parametrize(("videos", "partial"), [("factor_videos", False), ("timed_videos", True)])
    @pytest.mark.parametrize("name", list(MOT_SEQUENCES))
    def test_tracking(self, request, videos, partial, name):
        view = View(request.getfixturevalue(videos), MOT_SEQUENCES[name]())
        _same_with_and_without_cache(view, partial_factors=partial)

    def test_filter_then_metadata_reads_each_item_once(self, factor_od):
        counting = _Counting(factor_od)
        view = View(counting, [Filter("tod = 'night'"), Limit(3), Filter(classes=[2])])
        _ = Metadata(view).dataframe
        expected = Counter(range(len(factor_od)))
        expected[0] += 1  # View.__init__ validates the dataset by reading its first item
        expected[view.selection[0]] += 1  # Metadata.__init__ validates the view by reading its first item
        assert counting.reads == expected

    @pytest.mark.parametrize("operations", [[Filter("tod = 'night'")], [Limit(4)]], ids=["cached", "uncached"])
    def test_a_forwarding_wrapper_is_structured_not_the_view_inside_it(self, factor_od, operations):
        wrapper = _Forwarding(View(factor_od, operations))
        metadata = Metadata(wrapper)
        assert "wrapped" in metadata.factor_names
        assert metadata.item_count == len(wrapper)

    def test_metadata_never_starts_a_cache(self, factor_od):
        view = View(factor_od, Limit(3))
        _ = Metadata(view).dataframe
        assert view._reads is None

    def test_resize_after_filter_clears_it_and_metadata_sees_the_new_boxes(self, factor_od):
        view = View(factor_od, [Filter("tod = 'night'"), Resize((2, 2))])
        assert view._reads is None
        boxes = np.asarray(Metadata(view).rows_at("instance")["box"].to_list(), dtype=np.float32)
        read = np.concatenate([np.asarray(view[i][1].boxes, dtype=np.float32).reshape(-1, 4) for i in range(len(view))])
        np.testing.assert_allclose(boxes, read)
