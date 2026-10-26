"""Tests for walking derived views to each item's root item."""

import json
import zipfile
from typing import Any, cast

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval._helpers import independence_groups
from dataeval._metadata._source_groups import root_items
from dataeval.bias import Balance, Parity
from dataeval.data import (
    Chain,
    DetectionCrops,
    Filter,
    Indices,
    SequenceFrames,
    Stride,
    VideoSegments,
    VideoStitch,
    View,
    Window,
    merge_datasets,
)
from dataeval.exceptions import MetadataFormatError
from tests.data.conftest import MockICDataset
from tests.data.test_crops import _ODDataset, _ODTarget, _positional_image, _simple_dataset
from tests.data.test_frames import _Buffering, _ExcludeEven, make_dataset
from tests.data.test_stitch import make_mot
from tests.test_helpers import _counts


def _present(groups):
    """Narrow root items that the test has set up to exist."""
    assert groups is not None
    return groups


class TestRootItems:
    def test_plain_and_tracking_datasets_have_none(self):
        assert root_items(make_dataset()[0]) is None
        assert root_items(_simple_dataset()) is None

    def test_frames_resolve_to_their_video(self):
        frames = SequenceFrames(make_dataset()[0])
        list(frames.stream())
        assert _present(root_items(frames)).tolist() == [0] * 6 + [1] * 4

    def test_thinned_frames(self):
        frames = SequenceFrames(make_dataset()[0], Stride(2))
        list(frames.stream())
        assert _present(root_items(frames)).tolist() == [0, 0, 0, 1, 1]

    def test_view_over_frames_composes(self):
        frames = SequenceFrames(make_dataset()[0])
        view = View(frames, Indices([9, 0, 7]))
        list(frames.stream())
        assert _present(root_items(view)).tolist() == [1, 0, 1]

    def test_crops_resolve_to_their_image(self):
        crops = DetectionCrops(_simple_dataset())
        np.testing.assert_array_equal(root_items(crops), crops.item_indices)

    def test_frames_over_segments_reach_the_video(self):
        segments = VideoSegments(make_dataset()[0], Window(2))
        frames = SequenceFrames(segments)
        list(frames.stream())
        assert set(_present(root_items(frames)).tolist()) == {0, 1}

    def test_walk_stops_at_stitch(self):
        stitched = VideoStitch(make_mot([[[1], [1]], [[2], [2]]])[0], group_by=None)
        assert root_items(stitched) is None

    def test_empty_view(self):
        result = _present(root_items(View(_simple_dataset(), Indices([]))))
        assert result.tolist() == []
        assert result.dtype == np.intp

    def test_result_is_read_only(self):
        assert not _present(root_items(View(_simple_dataset(), Indices([0, 0])))).flags.writeable


def _rewrite_manifest(path, target, edit):
    """Copy an archive to `target`, letting `edit` change its manifest on the way."""
    with zipfile.ZipFile(path) as old, zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as new:
        for name in old.namelist():
            data = old.read(name)
            if name == "manifest.json":
                manifest = json.loads(data)
                edit(manifest)
                data = json.dumps(manifest).encode()
            new.writestr(name, data)


def _without_root_items(path, target):
    """Rewrite an archive as one written before `root_items` joined the manifest."""
    _rewrite_manifest(path, target, lambda manifest: manifest.pop("root_items"))


class TestMetadataRecords:
    def test_frames(self):
        md = Metadata(SequenceFrames(make_dataset()[0]))
        _ = md.item_indices
        assert _present(md._root_items).tolist() == [0] * 6 + [1] * 4

    def test_crops(self):
        crops = DetectionCrops(_simple_dataset())
        md = Metadata(crops)
        _ = md.item_indices
        np.testing.assert_array_equal(_present(md._root_items), crops.item_indices)

    def test_not_derived_is_none(self):
        for data in (_simple_dataset(), make_dataset()[0]):
            md = Metadata(data)
            _ = md.item_indices
            assert md._root_items is None
        assert Metadata.from_factors({"a": np.array([1, 2])})._root_items is None

    def test_none_before_structuring(self):
        assert Metadata(SequenceFrames(make_dataset()[0]))._root_items is None

    def test_derived_copies_keep_each_rows_video(self):
        md = Metadata(SequenceFrames(make_dataset()[0]))
        sub = md.where(pl.col("frame") < 2, level="unit")
        # Two detections per frame, so each kept frame reads as two rows at the instance level.
        assert sub.item_indices.tolist() == [0, 0, 1, 1, 6, 6, 7, 7]
        assert _present(sub._root_items)[sub.item_indices].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]

    def test_filtered_below_items(self):
        md = Metadata(SequenceFrames(make_dataset()[0]))
        sub = md.where(pl.col("target_index") == 0, level="instance")
        assert sub._cut_below_items
        assert sub.item_indices.tolist() == list(range(10))
        assert _present(sub._root_items)[sub.item_indices].tolist() == [0] * 6 + [1] * 4

    def test_filtered_below_items_after_dropping_a_frames_detections(self):
        md = Metadata(SequenceFrames(make_dataset()[0]))
        sub = md.where(pl.col("item_index") >= 6, level="instance")
        assert set(_present(sub._root_items)[sub.item_indices].tolist()) == {1}

    def test_rebinding_recomputes(self):
        md = Metadata(SequenceFrames(make_dataset()[0]))
        _ = md.item_indices
        md.bind(_simple_dataset())
        assert md._root_items is None
        _ = md.item_indices
        assert md._root_items is None
        md.bind(DetectionCrops(_simple_dataset()))
        _ = md.item_indices
        assert _present(md._root_items).tolist() == [0, 0, 1, 2, 2]

    def test_new_recomputes(self):
        md = Metadata(SequenceFrames(make_dataset()[0]))
        _ = md.item_indices
        fresh = md.new(DetectionCrops(_simple_dataset()))
        assert fresh._root_items is None
        _ = fresh.item_indices
        assert _present(fresh._root_items).tolist() == [0, 0, 1, 2, 2]

    def test_video_with_no_kept_frames(self):
        # The first video is one frame and `_ExcludeEven` removes it; the second keeps its odd frames, so no
        # remaining row may be read as video 0.
        md = Metadata(SequenceFrames(make_dataset(frame_counts=(1, 4))[0], _ExcludeEven()))
        _ = md.item_indices
        assert _present(md._root_items).tolist() == [1, 1]

    def test_view_keeping_nothing_of_one_video(self):
        frames = SequenceFrames(make_dataset()[0])
        md = Metadata(View(frames, Indices([6, 7, 9])))
        _ = md.item_indices
        assert _present(md._root_items).tolist() == [1, 1, 1]

    def test_two_pass_chain(self):
        frames = SequenceFrames(make_dataset()[0], Chain(Stride(2), _Buffering(two_pass=True)))
        md = Metadata(frames)
        _ = md.item_indices
        assert _present(md._root_items).tolist() == md._store.frame(md._item_level)["sequence"].to_list()

    def test_save_load_round_trip(self, tmp_path):
        md = Metadata(SequenceFrames(make_dataset()[0]))
        md.save(tmp_path / "md")
        back = Metadata.load(tmp_path / "md")
        assert _present(back._root_items).tolist() == [0] * 6 + [1] * 4
        assert not _present(back._root_items).flags.writeable

    def test_plain_save_load_is_none(self, tmp_path):
        Metadata(_simple_dataset()).save(tmp_path / "md")
        assert Metadata.load(tmp_path / "md")._root_items is None

    def test_filtered_save_load_keeps_each_rows_group(self, tmp_path):
        md = Metadata(SequenceFrames(make_dataset()[0]))
        sub = md.where(pl.col("frame") >= 3, level="unit")
        sub.save(tmp_path / "sub")
        back = Metadata.load(tmp_path / "sub")
        np.testing.assert_array_equal(back.item_indices, sub.item_indices)
        assert _present(back._root_items)[back.item_indices].tolist() == [0, 0, 0, 0, 0, 0, 1, 1]

    @pytest.mark.parametrize("view", ["frames", "crops"])
    def test_pre_change_derived_archive_is_refused(self, tmp_path, view):
        data = SequenceFrames(make_dataset()[0]) if view == "frames" else DetectionCrops(_simple_dataset())
        Metadata(data).save(tmp_path / "md")
        _without_root_items(tmp_path / "md", tmp_path / "old")
        with pytest.raises(MetadataFormatError, match="(?i)rebuild"):
            Metadata.load(tmp_path / "old")

    def test_pre_change_archive_is_refused_when_source_id_was_moved_aside(self, tmp_path):
        md = Metadata(DetectionCrops(_simple_dataset()))
        md.add_factors({"source_id": np.arange(5)}, level="unit", overwrite=True)
        assert md._moved_aside == {"source_id_original"}
        md.save(tmp_path / "md")
        _without_root_items(tmp_path / "md", tmp_path / "old")
        with pytest.raises(MetadataFormatError, match="(?i)rebuild"):
            Metadata.load(tmp_path / "old")

    def test_pre_change_plain_archive_loads(self, tmp_path):
        Metadata(_simple_dataset()).save(tmp_path / "md")
        _without_root_items(tmp_path / "md", tmp_path / "old")
        assert Metadata.load(tmp_path / "old")._root_items is None

    def test_pre_change_from_factors_archive_with_a_source_id_loads(self, tmp_path):
        # A factors container has no dataset to rebuild from, so the rebuild refusal cannot apply to it.
        Metadata.from_factors({"source_id": np.arange(5), "a": np.arange(5) % 2}).save(tmp_path / "md")
        _without_root_items(tmp_path / "md", tmp_path / "old")
        assert Metadata.load(tmp_path / "old")._root_items is None

    @pytest.mark.parametrize("filtered", [False, True])
    def test_root_items_of_the_wrong_length_are_refused(self, tmp_path, filtered):
        md = Metadata(SequenceFrames(make_dataset()[0]))
        if filtered:
            md = md.where(pl.col("frame") >= 3, level="unit")
        md.save(tmp_path / "md")
        # The filtered map must still reach its highest kept item (9); the unfiltered one must match `count`.
        _rewrite_manifest(tmp_path / "md", tmp_path / "bad", lambda m: m.update(root_items=m["root_items"][:9]))
        with pytest.raises(MetadataFormatError, match="root item"):
            Metadata.load(tmp_path / "bad")

    def test_filter_view_records_each_items_video(self):
        frames = SequenceFrames(make_dataset()[0])
        md = Metadata(View(frames, Filter("frame >= 3")))
        _ = md.item_indices
        # Frames 3-5 of video 0 and 3 of video 1, which has four frames.
        assert _present(md._root_items).tolist() == [0, 0, 0, 1]


def _detections_per_image(n=40):
    """An OD dataset of images with three detections each, all of one class per image."""
    images = np.stack([_positional_image(8, 8) for _ in range(n)])
    targets = [_ODTarget([[0, 0, 4, 4], [1, 1, 5, 5], [2, 2, 6, 6]], [i % 3] * 3) for i in range(n)]
    return _ODDataset(images, targets)


def _with_image_factor(data, indices=None):
    """Metadata over `data` with a per-image factor that tracks the class, so Balance has something to score."""
    md = Metadata(data)
    weather = np.arange(len(data)) if indices is None else np.asarray(indices)
    md.add_factors({"weather": weather % 3}, level=md.item_level, overwrite=True)
    return md


@pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")
class TestOneToOneViews:
    """A View that keeps each item at most once leaves every item its own unit, so wrapping changes nothing."""

    @pytest.mark.parametrize("indices", [list(range(40)), list(range(39, -1, -1)), list(range(0, 40, 2))])
    def test_od_view_matches_the_unwrapped_dataset(self, indices):
        plain = _detections_per_image()
        md = _with_image_factor(View(plain, Indices(indices)), indices)
        assert md._root_items is None
        base = _with_image_factor(plain)
        if len(indices) == 40:
            assert _counts(md, ["weather"]).tolist() == _counts(base, ["weather"]).tolist()  # type: ignore[union-attr]
        if indices == list(range(40)):
            assert Balance().evaluate(md).factors.equals(Balance().evaluate(base).factors)
        a, b = Parity().from_metadata(md), Parity().from_metadata(base)
        assert (a.group_source, a.group_level) == (b.group_source, b.group_level) == ("level", base.item_level)

    def test_tracking_view_matches_the_unwrapped_dataset(self):
        plain, _ = make_dataset()
        md = Metadata(View(plain, Indices([0, 1])))
        base = Metadata(plain)
        assert md._root_items is None
        names = ["height", "width"]
        assert _counts(md, names).tolist() == _counts(base, names).tolist()  # type: ignore[union-attr]

    def test_a_repeating_view_still_groups_its_copies(self):
        md = Metadata(View(_simple_dataset(), Indices([0, 0, 1])))
        _ = md.item_indices
        assert _present(md._root_items).tolist() == [0, 0, 1]

    def test_one_to_one_view_round_trips_a_filtered_save(self, tmp_path):
        md = _with_image_factor(View(_detections_per_image(), Indices(range(40))))
        sub = md.where(pl.col("weather") == 1, level=md.item_level)
        sub.save(tmp_path / "sub")
        back = Metadata.load(tmp_path / "sub")
        assert back._root_items is None
        np.testing.assert_array_equal(back.item_indices, sub.item_indices)


class TestMerges:
    def test_frame_views_over_different_sources_stay_apart(self):
        a, b = make_dataset()[0], make_dataset()[0]
        merged = merge_datasets(SequenceFrames(a), SequenceFrames(b))
        assert _present(root_items(merged)).tolist() == [0] * 6 + [1] * 4 + [2] * 6 + [3] * 4

    def test_views_over_one_source_share_its_videos(self):
        ds = make_dataset()[0]
        merged = merge_datasets(SequenceFrames(ds), SequenceFrames(ds, Stride(2)))
        assert _present(root_items(merged)).tolist() == [0] * 6 + [1] * 4 + [0, 0, 0, 1, 1]

    def test_view_over_a_merge_composes(self):
        ds = make_dataset()[0]
        merged = merge_datasets(SequenceFrames(ds), SequenceFrames(ds, Stride(2)))
        assert _present(root_items(View(merged, Indices([10, 0, 14])))).tolist() == [0, 0, 1]

    def test_crop_views_and_a_plain_part(self):
        plain = cast(Any, MockICDataset([0, 1], {0: "a", 1: "b", 2: "c"}))  # a plain dataset, untyped mock
        merged = merge_datasets(DetectionCrops(_simple_dataset()), plain)
        assert _present(root_items(merged)).tolist() == [0, 0, 1, 2, 2, 3, 4]

    def test_a_merge_of_distinct_plain_datasets_records_nothing(self):
        md = Metadata(merge_datasets(_simple_dataset(), _simple_dataset()))
        _ = md.item_indices
        assert md._root_items is None

    def test_a_dataset_merged_with_itself_groups_its_copies(self):
        ds = _simple_dataset()
        md = Metadata(merge_datasets(ds, ds))
        _ = md.item_indices
        assert _present(md._root_items).tolist() == [0, 1, 2, 0, 1, 2]

    def test_parity_groups_merged_frame_views_by_video(self):
        a, b = make_dataset()[0], make_dataset()[0]
        md = Metadata(merge_datasets(SequenceFrames(a), SequenceFrames(b, Stride(2))))
        unit = independence_groups(md, assume_independent=False)
        assert unit.source == "source"
        assert np.unique(_present(unit.groups)).size == 4

    def test_an_empty_middle_part(self):
        a, b, c = _simple_dataset(), _simple_dataset(), _simple_dataset()
        merged = merge_datasets(DetectionCrops(a), View(b, Indices([])), DetectionCrops(c))
        assert _present(root_items(merged)).tolist() == [0, 0, 1, 2, 2, 3, 3, 4, 5, 5]

    def test_a_merge_inside_a_merge(self):
        a, b = make_dataset()[0], make_dataset()[0]
        merged = merge_datasets(merge_datasets(SequenceFrames(a), SequenceFrames(b)), SequenceFrames(a, Stride(2)))
        assert _present(root_items(merged)).tolist() == [0] * 6 + [1] * 4 + [2] * 6 + [3] * 4 + [0, 0, 0, 1, 1]

    def test_frames_over_a_dataset_merged_with_itself_group_its_copies(self):
        a = make_dataset()[0]
        groups = _present(root_items(SequenceFrames(merge_datasets(a, a))))
        assert groups.tolist() == [0] * 6 + [1] * 4 + [0] * 6 + [1] * 4

    def test_the_split_follows_the_lengths_the_merge_reads_by(self):
        ds = _simple_dataset()
        view = View(ds, Indices([0, 1]))
        merged = merge_datasets(view, DetectionCrops(_simple_dataset()))
        view.selection = [2]  # the merge still reads two items from this part, so its second is gone
        with pytest.raises(IndexError):
            root_items(merged)  # loud, as merged[1] is, rather than a group read from uninitialized memory

    def test_a_merged_frame_view_round_trips_a_save(self, tmp_path):
        a, b = make_dataset()[0], make_dataset()[0]
        md = Metadata(merge_datasets(SequenceFrames(a), SequenceFrames(b, Stride(2))))
        md.save(tmp_path / "md")
        np.testing.assert_array_equal(_present(Metadata.load(tmp_path / "md")._root_items), _present(md._root_items))
