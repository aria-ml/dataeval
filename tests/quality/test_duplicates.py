import logging
from typing import Any

import numpy as np
import polars as pl
import pytest

from dataeval.core import align_subsequence, pack_hashes
from dataeval.core._clusterer import ClusterResult
from dataeval.core._compute_stats import compute_stats
from dataeval.data import FrameIndices, SequenceFrames, Stride
from dataeval.extractors import FlattenExtractor
from dataeval.flags import ImageStats
from dataeval.quality import Duplicates, DuplicatesOutput, _duplicates
from dataeval.quality._duplicates import (
    SourceIndex,
    _build_duplicates_dataframe,
    _dominant,
    _find_hash_groups,
    _merge_near_groups,
)


class MockDataset:
    def __len__(self):
        return 20

    def __iter__(self):
        for _ in range(20):
            yield np.random.random((3, 16, 16))

    def __getitem__(self, _):
        return np.random.random((3, 16, 16))


def _get_exact_groups(result: DuplicatesOutput, level: str = "item") -> pl.DataFrame:
    """Helper to filter exact duplicate groups at a given level."""
    return result.data().filter((pl.col("level") == level) & (pl.col("dup_type") == "exact"))


def _get_near_groups(result: DuplicatesOutput, level: str = "item") -> pl.DataFrame:
    """Helper to filter near duplicate groups at a given level."""
    return result.data().filter((pl.col("level") == level) & (pl.col("dup_type") == "near"))


@pytest.mark.required
class TestDuplicates:
    def test_duplicates(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates()
        results = dupes.evaluate(np.concatenate((data, data)))

        assert isinstance(results, DuplicatesOutput)
        assert isinstance(results.data(), pl.DataFrame)

        exact_items = _get_exact_groups(results, "item")
        assert exact_items.shape[0] == 20  # 20 exact duplicate groups

        # No target-level duplicates
        exact_targets = _get_exact_groups(results, "target")
        near_targets = _get_near_groups(results, "target")
        assert exact_targets.shape[0] == 0
        assert near_targets.shape[0] == 0

    def test_near_duplicates(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates()
        results = dupes.evaluate(np.concatenate((data, data + 0.001)))
        # Adding 0.001 to random data creates values that are NOT byte-identical,
        # so xxhash will NOT find them as exact duplicates. However, phash will
        # find them as near duplicates because the visual difference is minimal.
        near_items = _get_near_groups(results, "item")
        assert near_items.shape[0] > 0

        # No target-level duplicates
        exact_targets = _get_exact_groups(results, "target")
        near_targets = _get_near_groups(results, "target")
        assert exact_targets.shape[0] == 0
        assert near_targets.shape[0] == 0

    def test_duplicates_only_exact(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates(ImageStats.HASH_XXHASH)
        results = dupes.evaluate(np.concatenate((data, data, data + 0.001)))

        exact_items = _get_exact_groups(results, "item")
        assert exact_items.shape[0] == 20
        # near is empty because HASH_PHASH was not included in flags
        near_items = _get_near_groups(results, "item")
        assert near_items.shape[0] == 0

        exact_targets = _get_exact_groups(results, "target")
        near_targets = _get_near_groups(results, "target")
        assert exact_targets.shape[0] == 0
        assert near_targets.shape[0] == 0

    def test_duplicates_with_stats(self):
        data = np.random.random((20, 3, 16, 16))
        data = np.concatenate((data, data, data + 0.001))
        # Stats computed with full HASH (includes both xxhash and phash)
        stats = compute_stats(data, stats=ImageStats.HASH, per_image=True, per_target=False)
        # Detector configured for exact only - but from_stats uses what's in the stats
        dupes = Duplicates(ImageStats.HASH_XXHASH)
        results = dupes.from_stats(stats)

        exact_items = _get_exact_groups(results, "item")
        assert exact_items.shape[0] == 20
        # from_stats uses what's available in the stats, so phash results will be present
        near_items = _get_near_groups(results, "item")
        assert near_items.shape[0] > 0

        exact_targets = _get_exact_groups(results, "target")
        near_targets = _get_near_groups(results, "target")
        assert exact_targets.shape[0] == 0
        assert near_targets.shape[0] == 0

    def test_get_duplicates_multiple_stats(self):
        """Test cross-dataset duplicate detection with new API."""
        ones = np.ones((1, 16, 16))
        zeros = np.zeros((1, 16, 16))
        data1 = np.concatenate((ones, zeros, ones, zeros, ones))
        data2 = np.concatenate((zeros, ones, zeros))
        data3 = np.concatenate((zeros + 0.001, ones - 0.001))
        dupes1 = compute_stats(data1, stats=ImageStats.HASH, per_image=True, per_target=False)
        dupes2 = compute_stats(data2, stats=ImageStats.HASH, per_image=True, per_target=False)
        dupes3 = compute_stats(data3, stats=ImageStats.HASH, per_image=True, per_target=False)

        dupes = Duplicates()
        results = dupes.from_stats([dupes1, dupes2, dupes3])

        # Check items structure - multi-dataset has dataset_indices column
        assert "dataset_indices" in results.data().columns

        exact_items = _get_exact_groups(results, "item")
        assert exact_items.shape[0] == 2  # 2 exact duplicate groups

        # Check near duplicates
        near_items = _get_near_groups(results, "item")
        assert near_items.shape[0] >= 1

        # No targets in this test
        exact_targets = _get_exact_groups(results, "target")
        near_targets = _get_near_groups(results, "target")
        assert exact_targets.shape[0] == 0
        assert near_targets.shape[0] == 0

    def test_duplicates_invalid_stats(self):
        dupes = Duplicates()
        with pytest.raises((TypeError, KeyError)):
            dupes.from_stats(1234)  # type: ignore

    def test_duplicates_ignore_non_duplicate_too_small(self):
        dupes = Duplicates()
        images = [np.random.random((3, 16, 16)) for _ in range(20)]
        images[3] = np.zeros((3, 5, 5))
        images[5] = np.ones((3, 5, 5))
        results = dupes.evaluate(images)
        # The key assertion is that we don't crash on small images
        assert isinstance(results, DuplicatesOutput)

    def test_duplicates_ignore_duplicate_too_small(self):
        dupes = Duplicates()
        images = [np.random.random((3, 16, 16)) for _ in range(20)]
        images[3] = np.zeros((3, 5, 5))
        images[5] = np.zeros((3, 5, 5))
        results = dupes.evaluate(images)
        # Small images get hashed with xxhash but not phash
        # So they can still appear in exact duplicates
        assert isinstance(results, DuplicatesOutput)
        exact_items = _get_exact_groups(results, "item")
        if exact_items.shape[0] > 0:
            # Check that small duplicates were found
            found_small = False
            for row in exact_items.iter_rows(named=True):
                indices = row["item_indices"]
                if 3 in indices and 5 in indices:
                    found_small = True
                    break
            assert found_small

    def test_duplicates_dataset(self):
        dupes = Duplicates()
        results = dupes.evaluate(MockDataset())
        assert results is not None

    def test_duplicates_from_clusters_basic(self):
        """Test basic cluster-based duplicate detection."""
        mock_cluster_result: ClusterResult = {
            "mst": np.array(
                [[0, 1, 0.1], [1, 2, 0.05], [2, 3, 0.0], [3, 4, 0.2]],
                dtype=np.float32,
            ),
            "clusters": np.array([0, 0, 0, 0, 0], dtype=np.intp),
            "linkage_tree": np.array([], dtype=np.float32),
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int64),
            "k_distances": np.array([], dtype=np.float32),
        }

        detector = Duplicates()
        result = detector.from_clusters(mock_cluster_result)

        # Cluster-based detection never returns exact duplicates
        exact_items = _get_exact_groups(result, "item")
        assert exact_items.shape[0] == 0

        # All cluster-based duplicates are near duplicates
        near_items = _get_near_groups(result, "item")
        assert near_items.shape[0] > 0
        for row in near_items.iter_rows(named=True):
            assert "cluster" in row["methods"]

        # No target-level results
        exact_targets = _get_exact_groups(result, "target")
        near_targets = _get_near_groups(result, "target")
        assert exact_targets.shape[0] == 0
        assert near_targets.shape[0] == 0

    def test_duplicates_from_clusters_with_near(self):
        """Test cluster-based detection treats all duplicates as near duplicates."""
        mock_cluster_result: ClusterResult = {
            "mst": np.array([[0, 1, 0.0], [1, 2, 0.01], [2, 3, 0.05], [3, 4, 0.1]], dtype=np.float32),
            "clusters": np.array([0, 0, 0, 0, 0], dtype=np.intp),
            "linkage_tree": np.array([], dtype=np.float32),
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int64),
            "k_distances": np.array([], dtype=np.float32),
        }

        detector = Duplicates()
        result = detector.from_clusters(mock_cluster_result)

        exact_items = _get_exact_groups(result, "item")
        assert exact_items.shape[0] == 0

        near_items = _get_near_groups(result, "item")
        assert near_items.shape[0] > 0
        for row in near_items.iter_rows(named=True):
            assert "cluster" in row["methods"]

        exact_targets = _get_exact_groups(result, "target")
        near_targets = _get_near_groups(result, "target")
        assert exact_targets.shape[0] == 0
        assert near_targets.shape[0] == 0

    def test_duplicates_from_clusters_no_duplicates(self):
        """Test with data that has no duplicates."""
        mock_cluster_result: ClusterResult = {
            "mst": np.array([[0, 1, 0.5], [1, 2, 0.3], [2, 3, 0.4]], dtype=np.float32),
            "clusters": np.array([0, 0, 0, 0], dtype=np.intp),
            "linkage_tree": np.array([], dtype=np.float32),
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int64),
            "k_distances": np.array([], dtype=np.float32),
        }

        detector = Duplicates()
        result = detector.from_clusters(mock_cluster_result)

        # Near may be empty or contain groups with "cluster" method
        near_items = _get_near_groups(result, "item")
        if near_items.shape[0] > 0:
            for row in near_items.iter_rows(named=True):
                assert "cluster" in row["methods"]

        exact_targets = _get_exact_groups(result, "target")
        near_targets = _get_near_groups(result, "target")
        assert exact_targets.shape[0] == 0
        assert near_targets.shape[0] == 0

    def test_from_clusters_respects_merge_near_duplicates(self):
        """Test that from_clusters respects the merge_near_duplicates parameter."""
        mock_cluster_result: ClusterResult = {
            "mst": np.array([[0, 1, 0.0], [1, 2, 0.01], [2, 3, 0.5], [3, 4, 5.0]], dtype=np.float32),
            "clusters": np.array([0, 0, 0, 0, 0], dtype=np.intp),
            "linkage_tree": np.array([], dtype=np.float32),
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int64),
            "k_distances": np.array([], dtype=np.float32),
        }

        # With merge_near_duplicates=True (default)
        detector_merged = Duplicates(merge_near_duplicates=True)
        result_merged = detector_merged.from_clusters(mock_cluster_result)
        near_merged = _get_near_groups(result_merged, "item")
        assert near_merged.shape[0] > 0

        # With merge_near_duplicates=False
        detector_separate = Duplicates(merge_near_duplicates=False)
        result_separate = detector_separate.from_clusters(mock_cluster_result)
        near_separate = _get_near_groups(result_separate, "item")
        assert near_separate.shape[0] > 0

        # When merging, overlapping groups get combined so fewer or equal groups
        assert near_merged.shape[0] <= near_separate.shape[0]

    def test_hash_differs_for_full_image_vs_targets(self, get_mock_od_dataset):
        """Regression test: hash values should differ between full image and individual targets."""
        image = np.zeros((3, 100, 100), dtype=np.uint8)
        image[:, 0:50, 0:50] = 255
        image[:, 50:100, 50:100] = 0

        images = [image]
        labels = [[0, 1]]
        bboxes = [[[0, 0, 50, 50], [50, 50, 100, 100]]]

        dataset = get_mock_od_dataset(images, labels, bboxes)
        result = compute_stats(dataset, stats=ImageStats.HASH, per_image=True, per_target=True)

        assert len(result["source_index"]) == 3

        full_image_xxhash = result["stats"]["xxhash"][0]
        box0_xxhash = result["stats"]["xxhash"][1]
        box1_xxhash = result["stats"]["xxhash"][2]

        full_image_phash = result["stats"]["phash"][0]
        box0_phash = result["stats"]["phash"][1]
        box1_phash = result["stats"]["phash"][2]

        assert full_image_xxhash != box0_xxhash
        assert full_image_xxhash != box1_xxhash
        assert box0_xxhash != box1_xxhash

        assert full_image_phash != box0_phash
        assert full_image_phash != box1_phash
        assert box0_phash != box1_phash

    def test_duplicate_detection_with_items_and_targets(self, get_mock_od_dataset):
        """Test separating item and target duplicate detection."""
        image1 = np.zeros((3, 100, 100), dtype=np.uint8)
        image1[:, 0:50, 0:50] = 255

        image2 = np.zeros((3, 100, 100), dtype=np.uint8)
        image2[:, 0:50, 0:50] = 255
        image2[:, 50:100, 50:100] = 128

        image3 = image1.copy()

        images = [image1, image2, image3]
        labels = [[0], [0, 1], [0]]
        bboxes = [
            [[0, 0, 50, 50]],
            [[0, 0, 50, 50], [50, 50, 100, 100]],
            [[0, 0, 50, 50]],
        ]

        dataset = get_mock_od_dataset(images, labels, bboxes)
        detector = Duplicates()
        result = detector.evaluate(dataset, per_image=True, per_target=True)

        # Check item-level duplicates (images 0 and 2 are identical)
        exact_items = _get_exact_groups(result, "item")
        assert exact_items.shape[0] == 1
        group_indices = exact_items[0, "item_indices"].to_list()
        assert set(group_indices) == {0, 2}

        # Check target-level duplicates (all three white boxes should be duplicates)
        exact_targets = _get_exact_groups(result, "target")
        assert exact_targets.shape[0] >= 1
        # Find the group containing 3 white boxes
        found_white_box_group = False
        for row in exact_targets.iter_rows(named=True):
            if len(row["item_indices"]) == 3:
                # All should have target_index 0 (the white boxes)
                assert "target_indices" in result.data().columns
                targets = row["target_indices"]
                assert all(t == 0 for t in targets)
                found_white_box_group = True
                break
        assert found_white_box_group

    def test_per_image_only(self, get_mock_od_dataset):
        """Test evaluating with per_image=True, per_target=False."""
        images = [np.random.random((3, 100, 100)) for _ in range(2)]
        labels = [[0], [1]]
        bboxes = [[[10, 10, 50, 50]], [[20, 20, 60, 60]]]

        dataset = get_mock_od_dataset(images, labels, bboxes)
        detector = Duplicates()
        result = detector.evaluate(dataset, per_image=True, per_target=False)

        assert isinstance(result, DuplicatesOutput)
        # No target-level results
        exact_targets = _get_exact_groups(result, "target")
        near_targets = _get_near_groups(result, "target")
        assert exact_targets.shape[0] == 0
        assert near_targets.shape[0] == 0

    def test_per_target_only(self, get_mock_od_dataset):
        """Test evaluating with per_image=False, per_target=True."""
        images = [np.random.random((3, 100, 100)) for _ in range(2)]
        labels = [[0], [1]]
        bboxes = [[[10, 10, 50, 50]], [[20, 20, 60, 60]]]

        dataset = get_mock_od_dataset(images, labels, bboxes)
        detector = Duplicates()
        result = detector.evaluate(dataset, per_image=False, per_target=True)

        assert isinstance(result, DuplicatesOutput)
        # Item-level should be empty since per_image=False
        exact_items = _get_exact_groups(result, "item")
        assert exact_items.shape[0] == 0

    def test_cross_dataset_with_targets(self, get_mock_od_dataset):
        """Test cross-dataset duplicate detection with targets."""
        white_box = np.ones((3, 50, 50), dtype=np.uint8) * 255
        black_box = np.zeros((3, 50, 50), dtype=np.uint8)

        image1 = np.zeros((3, 100, 100), dtype=np.uint8)
        image1[:, 0:50, 0:50] = white_box

        image2 = np.zeros((3, 100, 100), dtype=np.uint8)
        image2[:, 0:50, 0:50] = white_box
        image2[:, 50:100, 50:100] = black_box

        dataset1 = get_mock_od_dataset([image1], [[0]], [[[0, 0, 50, 50]]])
        dataset2 = get_mock_od_dataset([image2], [[0, 1]], [[[0, 0, 50, 50], [50, 50, 100, 100]]])

        stats1 = compute_stats(dataset1, stats=ImageStats.HASH, per_image=True, per_target=True)
        stats2 = compute_stats(dataset2, stats=ImageStats.HASH, per_image=True, per_target=True)

        detector = Duplicates()
        result = detector.from_stats([stats1, stats2], per_target=True)

        # Multi-dataset should have dataset_indices column
        assert "dataset_indices" in result.data().columns

        # Check item-level duplicates
        exact_items = _get_exact_groups(result, "item")
        assert exact_items.shape[0] == 1

        # Check target-level duplicates
        exact_targets = _get_exact_groups(result, "target")
        assert exact_targets.shape[0] >= 1


@pytest.mark.required
class TestDuplicatesMultiDataset:
    """Tests for multi-dataset duplicate detection via evaluate(data, *other)."""

    def test_evaluate_multi_dataset_exact(self):
        """Evaluate with two datasets sharing exact duplicates."""
        data1 = np.random.random((10, 3, 16, 16))
        data2 = np.concatenate((data1[:5], np.random.random((5, 3, 16, 16))))

        dupes = Duplicates()
        result = dupes.evaluate(data1, data2)

        assert isinstance(result, DuplicatesOutput)
        assert "dataset_indices" in result.data().columns

        exact_items = _get_exact_groups(result, "item")
        assert exact_items.shape[0] >= 5  # at least 5 cross-dataset exact groups

        # dataset_indices lists should be present and align with item_indices
        for row in exact_items.iter_rows(named=True):
            assert len(row["dataset_indices"]) == len(row["item_indices"])

    def test_evaluate_multi_dataset_near(self):
        """Evaluate with two datasets sharing near duplicates."""
        data1 = np.random.random((10, 3, 16, 16))
        data2 = data1 + 0.001  # near-duplicates

        dupes = Duplicates()
        result = dupes.evaluate(data1, data2)

        assert "dataset_indices" in result.data().columns
        near_items = _get_near_groups(result, "item")
        assert near_items.shape[0] > 0

    def test_evaluate_multi_dataset_three_datasets(self):
        """Evaluate with three datasets."""
        ones = np.ones((3, 1, 16, 16))
        zeros = np.zeros((3, 1, 16, 16))
        mixed = np.concatenate((ones[:1], zeros[:1]))

        dupes = Duplicates()
        result = dupes.evaluate(ones, zeros, mixed)

        assert "dataset_indices" in result.data().columns
        exact_items = _get_exact_groups(result, "item")
        assert exact_items.shape[0] >= 2  # ones-group and zeros-group

    def test_evaluate_multi_dataset_no_duplicates(self):
        """Multi-dataset with no duplicates should return empty groups."""
        rng = np.random.default_rng(42)
        data1 = rng.random((5, 3, 16, 16))
        data2 = rng.random((5, 3, 16, 16))

        dupes = Duplicates(ImageStats.HASH_XXHASH)
        result = dupes.evaluate(data1, data2)

        exact_items = _get_exact_groups(result, "item")
        assert exact_items.shape[0] == 0
        assert len(result) == 0

    def test_evaluate_multi_dataset_exact_property(self):
        """exact property returns dict[int, list[list[int]]] for multi-dataset."""
        data1 = np.zeros((3, 1, 16, 16))
        data2 = np.zeros((2, 1, 16, 16))

        dupes = Duplicates(ImageStats.HASH_XXHASH)
        result = dupes.evaluate(data1, data2)

        exact = result.exact
        # Multi-dataset exact returns dict keyed by dataset_indices
        assert isinstance(exact, dict)
        for ds_idx, groups in exact.items():
            assert isinstance(ds_idx, int)
            assert isinstance(groups, list)
            for group in groups:
                assert isinstance(group, list)
                assert all(isinstance(i, int) for i in group)

    def test_evaluate_multi_dataset_near_property(self):
        """near property returns dict[int, list[tuple[list[int], list[str]]]] for multi-dataset."""
        data1 = np.random.random((10, 3, 16, 16))
        data2 = data1 + 0.001

        dupes = Duplicates()
        result = dupes.evaluate(data1, data2)

        near = result.near
        # Multi-dataset near returns dict keyed by dataset_indices
        assert isinstance(near, dict)
        for ds_idx, groups in near.items():
            assert isinstance(ds_idx, int)
            assert isinstance(groups, list)
            for indices, methods in groups:
                assert isinstance(indices, list)
                assert isinstance(methods, list)
                assert all(isinstance(m, str) for m in methods)

    def test_evaluate_multi_dataset_with_targets(self, get_mock_od_dataset):
        """Multi-dataset evaluate with per_target=True."""
        white_box = np.ones((3, 50, 50), dtype=np.uint8) * 255

        image1 = np.zeros((3, 100, 100), dtype=np.uint8)
        image1[:, 0:50, 0:50] = white_box

        image2 = np.zeros((3, 100, 100), dtype=np.uint8)
        image2[:, 0:50, 0:50] = white_box

        dataset1 = get_mock_od_dataset([image1], [[0]], [[[0, 0, 50, 50]]])
        dataset2 = get_mock_od_dataset([image2], [[0]], [[[0, 0, 50, 50]]])

        dupes = Duplicates()
        result = dupes.evaluate(dataset1, dataset2, per_image=True, per_target=True)

        assert "dataset_indices" in result.data().columns
        # Both images are identical, so should have exact item-level duplicates
        exact_items = _get_exact_groups(result, "item")
        assert exact_items.shape[0] >= 1
        # Targets are identical too
        exact_targets = _get_exact_groups(result, "target")
        assert exact_targets.shape[0] >= 1

    def test_evaluate_multi_dataset_with_threshold(self):
        """with_threshold works on multi-dataset results from evaluate."""
        data1 = np.random.random((10, 3, 16, 16))
        data2 = data1.copy()

        dupes = Duplicates(extractor=FlattenExtractor(), cluster_sensitivity=1.0)
        result = dupes.evaluate(data1, data2)

        assert "dataset_indices" in result.data().columns
        # Re-detect with tighter threshold
        strict = result.with_sensitivity(0.5)
        assert isinstance(strict, DuplicatesOutput)
        assert "dataset_indices" in strict.data().columns


@pytest.mark.required
class TestDuplicatesBackwardsCompat:
    def test_dataset_index_alias_is_removed(self):
        """The 'dataset_index' alias was deprecated in v1.0 and removed in v1.1.

        It redirected to 'dataset_indices' under a DeprecationWarning. Pinned as gone so
        the alias is not reintroduced by accident: the column has one name now.
        """
        data1 = np.zeros((3, 1, 16, 16))
        data2 = np.zeros((2, 1, 16, 16))

        result = Duplicates(ImageStats.HASH_XXHASH).evaluate(data1, data2)
        assert "dataset_indices" in result.data().columns

        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            result["dataset_index"]


@pytest.mark.required
class TestDuplicatesEdgeCases:
    def test_evaluate_invalid_config(self):
        """Covers ValueError when flags=NONE and no cluster-based detection configured."""
        detector = Duplicates(flags=ImageStats.NONE, extractor=None)
        data = np.zeros((1, 10, 10, 3))
        with pytest.raises(ValueError, match="Either flags must contain hash stats"):
            detector.evaluate(data)

    def test_build_dataframe_orientation_with_many_exact_rows(self):
        """Regression: ≥100 exact rows then a near row with string orientation must not
        trigger a polars schema-inference ComputeError."""
        item_exact = [[2 * i, 2 * i + 1] for i in range(101)]
        item_near = [((1000, 1001), "phash")]
        df = _build_duplicates_dataframe(
            item_exact=item_exact,
            item_near_method_groups=item_near,
            target_exact=None,
            target_near_method_groups=[],
            available_stats={"phash", "dhash", "phash_d4", "dhash_d4"},
            merge=True,
        )
        assert df.shape[0] == 102
        assert df.schema["orientation"] == pl.Utf8
        near = df.filter(pl.col("dup_type") == "near")
        assert near["orientation"].to_list() == ["same"]

    def test_merge_near_groups_logic(self):
        """Covers _merge_near_groups merging logic."""
        # Disjoint groups with merge
        groups = [([1, 2], "phash"), ([3, 4], "dhash")]
        result = _merge_near_groups(groups, {"phash", "dhash"}, merge=True)
        assert len(result) == 2

        # Overlapping groups with merge
        groups = [([1, 2], "phash"), ([2, 3], "dhash")]
        result = _merge_near_groups(groups, {"phash", "dhash"}, merge=True)
        assert len(result) == 1
        indices, methods, orientation = result[0]
        assert set(indices) == {1, 2, 3}

        # Complex overlap with merge
        groups = [([1, 2], "phash"), ([3, 4], "dhash"), ([2, 3], "phash")]
        result = _merge_near_groups(groups, {"phash", "dhash"}, merge=True)
        assert len(result) == 1
        indices, methods, orientation = result[0]
        assert set(indices) == {1, 2, 3, 4}

    def test_cluster_distance_factor_none_raises_error(self):
        """When flags=NONE and cluster_sensitivity=None, should raise ValueError."""

        class DummyExtractor:
            def __call__(self, data):
                return np.array([[0.1], [0.1], [0.9]])

        detector = Duplicates(
            flags=ImageStats.NONE,
            extractor=DummyExtractor(),
            cluster_sensitivity=None,
            cluster_algorithm="kmeans",
            n_clusters=2,
        )
        data = np.array([0.1, 0.1, 0.2, 0.2, 0.9])
        with pytest.raises(ValueError, match="Either flags must contain hash stats"):
            detector.evaluate(data)

    def test_find_hash_groups_empty_logic(self):
        """Covers _find_hash_groups filtering empty values."""
        stats = {"phash": np.array(["", "abc", "abc", ""])}
        source_index = [SourceIndex(i, None, None) for i in range(4)]
        indices = [0, 1, 2, 3]
        exact_groups: list[list[int]] = []

        groups = _find_hash_groups(stats, "phash", source_index, indices, exact_groups)
        assert groups == [[1, 2]]

    def test_evaluate_with_tuple_dataset(self, get_mock_ic_dataset):
        """Regression test: evaluate with cluster-based detection handles tuple datasets."""
        data = np.random.random((20, 3, 16, 16))
        data_with_dupes = np.concatenate([data, data])
        labels = list(range(len(data_with_dupes)))
        dataset = get_mock_ic_dataset(list(data_with_dupes), labels)

        detector = Duplicates(extractor=FlattenExtractor(), cluster_sensitivity=1.0)
        result = detector.evaluate(dataset)
        assert isinstance(result, DuplicatesOutput)

    def test_evaluate_with_tuple_dataset_cluster_only(self, get_mock_ic_dataset):
        """Regression test: cluster-only detection on tuple datasets."""
        data = np.random.random((20, 3, 16, 16))
        labels = list(range(len(data)))
        dataset = get_mock_ic_dataset(list(data), labels)

        detector = Duplicates(flags=ImageStats.NONE, extractor=FlattenExtractor(), cluster_sensitivity=1.0)
        result = detector.evaluate(dataset)
        assert isinstance(result, DuplicatesOutput)


@pytest.mark.required
class TestDuplicatesOutputAPI:
    """Tests for the new DataFrame-based DuplicatesOutput API."""

    def test_data_returns_dataframe(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates()
        result = dupes.evaluate(np.concatenate((data, data)))
        assert isinstance(result.data(), pl.DataFrame)

    def test_len_returns_group_count(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates()
        result = dupes.evaluate(np.concatenate((data, data)))
        assert len(result) == result.data().shape[0]
        assert len(result) > 0

    def test_dataframe_schema(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates()
        result = dupes.evaluate(np.concatenate((data, data)))
        df = result.data()

        # Check required columns
        assert "group_id" in df.columns
        assert "level" in df.columns
        assert "dup_type" in df.columns
        assert "item_indices" in df.columns
        assert "methods" in df.columns
        # orientation is only present when both basic and D4 hashes are computed
        assert "orientation" not in df.columns

        # Check types
        assert df.schema["group_id"] == pl.Int64
        assert df.schema["level"] == pl.Utf8
        assert df.schema["dup_type"] == pl.Utf8
        assert df.schema["item_indices"] == pl.List(pl.Int64)
        assert df.schema["methods"] == pl.List(pl.Utf8)

    def test_level_column_values(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates()
        result = dupes.evaluate(np.concatenate((data, data)))
        levels = result.data()["level"].unique().to_list()
        assert all(lvl in ("item", "target") for lvl in levels)

    def test_methods_is_list(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates()
        result = dupes.evaluate(np.concatenate((data, data)))
        # methods should be list[str], not comma-separated string
        for row in result.data().iter_rows(named=True):
            assert isinstance(row["methods"], list)
            assert all(isinstance(m, str) for m in row["methods"])

    def test_aggregate_by_image(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates()
        result = dupes.evaluate(np.concatenate((data, data)))

        by_image = result.aggregate_by_image()
        assert "item_index" in by_image.columns
        assert "group_count" in by_image.columns
        assert "dup_types" in by_image.columns
        assert "methods" in by_image.columns
        # Every image should appear at least once (they're all duplicates)
        assert by_image.shape[0] > 0

    def test_aggregate_by_group(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates()
        result = dupes.evaluate(np.concatenate((data, data)))

        by_group = result.aggregate_by_group()
        assert "group_id" in by_group.columns
        assert "level" in by_group.columns
        assert "member_count" in by_group.columns
        assert "methods" in by_group.columns
        # Should have same number of rows as groups
        assert by_group.shape[0] == len(result)

    def test_aggregate_by_method(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates()
        result = dupes.evaluate(np.concatenate((data, data)))

        by_method = result.aggregate_by_method()
        assert "method" in by_method.columns
        assert "group_count" in by_method.columns
        assert "total_members" in by_method.columns
        assert by_method.shape[0] > 0

    def test_aggregate_empty_result(self):
        """Test aggregation on empty results."""
        # Use random data with no duplicates and xxhash-only
        data = np.random.random((3, 3, 16, 16))
        dupes = Duplicates(flags=ImageStats.HASH_XXHASH)
        result = dupes.evaluate(data)

        # May be empty or not - test that aggregation doesn't crash
        by_image = result.aggregate_by_image()
        by_group = result.aggregate_by_group()
        by_method = result.aggregate_by_method()
        assert isinstance(by_image, pl.DataFrame)
        assert isinstance(by_group, pl.DataFrame)
        assert isinstance(by_method, pl.DataFrame)

    def test_with_threshold(self):
        """Test with_threshold for cluster-based redetection."""
        mock_cluster_result: ClusterResult = {
            "mst": np.array([[0, 1, 0.0], [1, 2, 0.01], [2, 3, 0.05], [3, 4, 0.1]], dtype=np.float32),
            "clusters": np.array([0, 0, 0, 0, 0], dtype=np.intp),
            "linkage_tree": np.array([], dtype=np.float32),
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int64),
            "k_distances": np.array([], dtype=np.float32),
        }

        detector = Duplicates()
        result = detector.from_clusters(mock_cluster_result)

        # Tighten threshold — should have fewer or equal near duplicates
        strict_result = result.with_sensitivity(0.1)
        assert isinstance(strict_result, DuplicatesOutput)
        strict_near = _get_near_groups(strict_result, "item")
        original_near = _get_near_groups(result, "item")
        # Stricter threshold means equal or fewer groups
        assert strict_near.shape[0] <= original_near.shape[0]

    def test_with_threshold_raises_without_clusters(self):
        """Test that with_threshold raises when no cluster results stored."""
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates()
        result = dupes.evaluate(np.concatenate((data, data)))

        with pytest.raises(ValueError, match="requires cluster results"):
            result.with_sensitivity(0.5)

    def test_aggregate_by_image_raises_multi_dataset(self):
        """aggregate_by_image should raise for multi-dataset output."""
        ones = np.ones((1, 16, 16))
        zeros = np.zeros((1, 16, 16))
        data1 = np.concatenate((ones, zeros, ones))
        data2 = np.concatenate((zeros, ones))
        dupes1 = compute_stats(data1, stats=ImageStats.HASH, per_image=True, per_target=False)
        dupes2 = compute_stats(data2, stats=ImageStats.HASH, per_image=True, per_target=False)

        dupes = Duplicates()
        result = dupes.from_stats([dupes1, dupes2])

        with pytest.raises(ValueError, match="aggregate_by_image only works"):
            result.aggregate_by_image()

    def test_exact_property_single_dataset(self):
        """exact property returns list[list[int]] for single-dataset."""
        data = np.random.random((10, 3, 16, 16))
        dupes = Duplicates()
        result = dupes.evaluate(np.concatenate((data, data)))

        exact = result.exact
        assert isinstance(exact, list)
        assert len(exact) > 0
        for group in exact:
            assert isinstance(group, list)
            assert len(group) >= 2
            assert all(isinstance(i, int) for i in group)

    def test_near_property_single_dataset(self):
        """near property returns list[tuple[list[int], list[str]]] for single-dataset."""
        data = np.random.random((10, 3, 16, 16))
        dupes = Duplicates()
        result = dupes.evaluate(np.concatenate((data, data + 0.001)))

        near = result.near
        assert isinstance(near, list)
        assert len(near) > 0
        for indices, methods in near:
            assert isinstance(indices, list)
            assert len(indices) >= 2
            assert all(isinstance(i, int) for i in indices)
            assert isinstance(methods, list)
            assert all(isinstance(m, str) for m in methods)

    def test_exact_property_multi_dataset(self):
        """exact property returns dict[int, list[list[int]]] for multi-dataset."""
        data = np.zeros((3, 1, 16, 16))
        dupes = Duplicates(ImageStats.HASH_XXHASH)
        result = dupes.evaluate(data, data)

        exact = result.exact
        assert isinstance(exact, dict)
        for ds_idx, groups in exact.items():
            assert isinstance(ds_idx, int)
            assert isinstance(groups, list)
            for group in groups:
                assert isinstance(group, list)
                assert all(isinstance(i, int) for i in group)

    def test_near_property_multi_dataset(self):
        """near property returns dict[int, list[tuple[list[int], list[str]]]] for multi-dataset."""
        data = np.random.random((10, 3, 16, 16))
        dupes = Duplicates()
        result = dupes.evaluate(data, data + 0.001)

        near = result.near
        assert isinstance(near, dict)
        for ds_idx, groups in near.items():
            assert isinstance(ds_idx, int)
            assert isinstance(groups, list)
            for indices, methods in groups:
                assert isinstance(indices, list)
                assert isinstance(methods, list)
                assert all(isinstance(m, str) for m in methods)


@pytest.mark.required
class TestUnmeasuredRegionsAreNotDuplicates:
    """An empty digest reports a region that could not be measured, not a picture.

    The hash calculator answers with the empty string for a view that holds no data — an
    out-of-bounds box, an image its boxes cover completely, or a band group the datum
    cannot supply. Grouping those together would call every unmeasured region an exact
    duplicate of every other, which is the failure the empty string exists to avoid, so
    exact grouping skips them exactly as near grouping already does.
    """

    def test_out_of_bounds_boxes_are_not_exact_duplicates(self):
        rng = np.random.default_rng(0)
        images = [rng.integers(0, 255, (3, 16, 16), dtype=np.uint8) for _ in range(4)]
        boxes = [[[100, 100, 110, 110]] for _ in range(4)]

        stats = compute_stats(
            images,
            boxes=boxes,
            stats=ImageStats.HASH_DUPLICATES_BASIC,
            per_image=True,
            per_target=True,
            normalize_pixel_values=False,
        )
        assert all(value == "" for value in stats["stats"]["xxhash"][1::2]), "boxes should be unmeasured"

        result = Duplicates().from_stats(stats, per_target=True)

        assert len(result) == 0

    def test_real_duplicates_are_still_found(self):
        """The skip must not swallow a genuine exact duplicate."""
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, (3, 16, 16), dtype=np.uint8)
        images = [image, image.copy(), rng.integers(0, 255, (3, 16, 16), dtype=np.uint8)]

        result = Duplicates().from_stats(
            compute_stats(images, stats=ImageStats.HASH_DUPLICATES_BASIC, normalize_pixel_values=False)
        )

        assert result.data()["item_indices"].to_list() == [[0, 1]]


@pytest.mark.required
class TestDuplicatesLevelViews:
    """`items` and `targets` narrow the result to one level, keeping the same surface."""

    @staticmethod
    def _results() -> DuplicatesOutput:
        data = np.random.random((6, 3, 16, 16))
        data = np.concatenate((data, data))
        stats = compute_stats(data, stats=ImageStats.HASH, per_image=True, per_target=False)
        return Duplicates(ImageStats.HASH_XXHASH).from_stats(stats)

    def test_items_keeps_only_item_level_rows(self):
        filtered = self._results().items
        assert isinstance(filtered, DuplicatesOutput)
        assert filtered.data()["level"].unique().to_list() in ([], ["item"])

    def test_targets_keeps_only_target_level_rows(self):
        filtered = self._results().targets
        assert isinstance(filtered, DuplicatesOutput)
        assert filtered.data()["level"].unique().to_list() in ([], ["target"])


@pytest.mark.required
def test_aggregate_by_group_carries_orientation_for_d4_hashes():
    """D4 hashing records which rotation/flip matched, and the summary keeps that column."""
    df = _build_duplicates_dataframe(
        item_exact=[[0, 1], [2, 3]],
        item_near_method_groups=[((4, 5), "phash_d4")],
        target_exact=None,
        target_near_method_groups=[],
        available_stats={"phash", "dhash", "phash_d4", "dhash_d4"},
        merge=True,
    )
    assert "orientation" in df.columns

    results = DuplicatesOutput(df, flags=ImageStats.HASH_DUPLICATES_D4)
    summary = results.aggregate_by_group()
    assert "orientation" in summary.columns
    assert summary.height == df.height


@pytest.mark.required
def test_aggregate_by_group_of_an_empty_d4_result_keeps_the_orientation_column():
    """The empty frame is built from the same schema, so readers see the same columns."""
    df = _build_duplicates_dataframe(
        item_exact=[[0, 1]],
        item_near_method_groups=[((2, 3), "phash_d4")],
        target_exact=None,
        target_near_method_groups=[],
        available_stats={"phash", "dhash", "phash_d4", "dhash_d4"},
        merge=True,
    )
    empty = DuplicatesOutput(df.clear(), flags=ImageStats.HASH_DUPLICATES_D4)
    assert "orientation" in empty.aggregate_by_group().columns


def _near_variants(seed: int = 42, size: int = 64):
    """A base image plus two mild photometric variants, as a re-encode would produce."""
    from dataeval.core import dhash, hamming_distance, phash

    rng = np.random.RandomState(seed)
    base = rng.randint(0, 256, (3, size, size)).astype(np.uint8)

    noisy = base
    for mag in range(1, 100):
        cand = np.clip(base.astype(np.int16) + rng.randint(-mag, mag + 1, base.shape), 0, 255).astype(np.uint8)
        if 0 < hamming_distance(phash(base), phash(cand)) <= 4 and 0 < hamming_distance(dhash(base), dhash(cand)) <= 4:
            noisy = cand
            break

    bright = base
    for mag in range(1, 100):
        cand = np.clip(base.astype(np.int16) + mag, 0, 255).astype(np.uint8)
        if 0 < hamming_distance(phash(base), phash(cand)) <= 4 and 0 < hamming_distance(dhash(base), dhash(cand)) <= 4:
            bright = cand
            break

    unrelated = [rng.randint(0, 256, (3, size, size)).astype(np.uint8) for _ in range(12)]
    return [base, noisy, bright, *unrelated]


@pytest.mark.required
class TestHashRadius:
    """Near-duplicate detection by Hamming distance rather than by digest equality."""

    def test_defaults_to_zero_for_exact_digest_grouping(self):
        assert Duplicates().hash_radius == 0
        assert Duplicates.Config().hash_radius == 0

    def test_set_by_argument_and_by_config(self):
        assert Duplicates(hash_radius=6).hash_radius == 6
        assert Duplicates(config=Duplicates.Config(hash_radius=4)).hash_radius == 4
        # An explicit argument still wins over the config it is given beside.
        assert Duplicates(hash_radius=2, config=Duplicates.Config(hash_radius=9)).hash_radius == 2

    def test_radius_zero_reproduces_digest_equality(self):
        """The compatibility contract: the default path is the behaviour that shipped."""
        images = _near_variants()
        assert Duplicates(hash_radius=0).evaluate(images).near == []

    def test_radius_finds_re_encodes_that_equality_misses(self):
        images = _near_variants()
        found = Duplicates(hash_radius=4).evaluate(images).near
        assert len(found) == 1
        members, methods = found[0]
        assert list(members) == [0, 1, 2]
        assert set(methods) == {"phash", "dhash"}

    def test_unrelated_images_do_not_group(self):
        rng = np.random.RandomState(7)
        images = [rng.randint(0, 256, (3, 64, 64)).astype(np.uint8) for _ in range(12)]
        assert Duplicates(hash_radius=6).evaluate(images).near == []

    def test_exact_duplicates_still_report_as_exact(self):
        """A radius widens the *near* relation; it must not reclassify identical images."""
        images = _near_variants()
        images.append(images[0].copy())
        result = Duplicates(hash_radius=6).evaluate(images)
        assert [sorted(g) for g in result.exact] == [[0, len(images) - 1]]

    def test_negative_radius_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            Duplicates(hash_radius=-1).evaluate(_near_variants())

    def test_from_stats_honours_the_radius(self):
        images = _near_variants()
        stats = compute_stats(images, stats=ImageStats.HASH_DUPLICATES_BASIC, normalize_pixel_values=False)
        assert Duplicates(hash_radius=0).from_stats(stats).near == []
        assert len(Duplicates(hash_radius=4).from_stats(stats).near) == 1

    def test_output_records_the_radius_it_used(self):
        assert Duplicates(hash_radius=5).evaluate(_near_variants()).hash_radius == 5


@pytest.mark.required
class TestWithRadius:
    def test_redetects_without_rehashing(self):
        images = _near_variants()
        strict = Duplicates(hash_radius=0).evaluate(images)
        assert strict.near == []
        relaxed = strict.with_radius(4)
        assert len(relaxed.near) == 1
        assert relaxed.hash_radius == 4
        # The original is untouched.
        assert strict.near == []
        assert strict.hash_radius == 0

    def test_round_trips_back_to_strict(self):
        relaxed = Duplicates(hash_radius=6).evaluate(_near_variants())
        assert relaxed.with_radius(0).near == []

    def test_agrees_with_evaluating_at_that_radius(self):
        images = _near_variants()
        redetected = Duplicates(hash_radius=0).evaluate(images).with_radius(4)
        direct = Duplicates(hash_radius=4).evaluate(images)
        assert redetected.near == direct.near
        assert redetected.exact == direct.exact

    def test_requires_stored_hash_statistics(self):
        cluster_result: ClusterResult = {
            "clusters": np.array([0, 0, 0], dtype=np.int64),
            "mst": np.array([[0, 1, 0.1], [1, 2, 0.2]], dtype=np.float32),
            "linkage_tree": np.array([], dtype=np.float32),
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int64),
            "k_distances": np.array([], dtype=np.float32),
        }
        output = Duplicates(cluster_sensitivity=1.0).from_clusters(cluster_result)
        with pytest.raises(ValueError, match="with_radius"):
            output.with_radius(4)

    def test_with_sensitivity_preserves_the_radius(self):
        """The two knobs are independent; changing one must not silently reset the other."""
        images = _near_variants()
        result = Duplicates(hash_radius=4, extractor=FlattenExtractor(), cluster_sensitivity=1.0).evaluate(images)
        assert result.with_sensitivity(2.0).hash_radius == 4


class _ReplayFrame:
    """A decoded frame standing in for :obj:`~dataeval.protocols.VideoFrame`."""

    def __init__(self, index: int, pixels: np.ndarray, timed: bool):
        self.frame_index = index
        self.pixels = pixels
        if timed:
            self.time_s = index / 30.0
            self.pts = index * 1001


class _ReplayStream:
    """A VideoStream replaying a fixed list of per-frame fill values or arrays."""

    def __init__(self, values, shape=(3, 24, 32), timed=True):
        self._values, self._shape, self._timed = values, shape, timed

    def __iter__(self):
        for index, value in enumerate(self._values):
            pixels = value if isinstance(value, np.ndarray) else np.full(self._shape, value, dtype=np.uint8)
            yield _ReplayFrame(index, pixels, self._timed)


def make_tracking_dataset(sequences, shape=(3, 24, 32), timed=True) -> Any:
    """A MAITE-shaped tracking dataset whose frames replay the given fill values."""
    from dataclasses import dataclass
    from typing import Any as _Any

    @dataclass
    class _FrameTarget:
        track_ids: np.ndarray
        boxes: np.ndarray
        scores: np.ndarray
        labels: np.ndarray

    @dataclass
    class _VideoTarget:
        frame_tracks: list

    class _Dataset:
        def __init__(self, data):
            self._data = data
            self.metadata = {"id": "videos", "index2label": {0: "thing"}}

        def __len__(self):
            return len(self._data)

        def __getitem__(self, index):
            return self._data[index]

    def target():
        return _FrameTarget(
            track_ids=np.zeros(1, dtype=np.int64),
            boxes=np.array([[1.0, 2.0, 9.0, 10.0]], dtype=np.float32),
            scores=np.ones(1, dtype=np.float32),
            labels=np.zeros(1, dtype=np.int64),
        )

    data: list[_Any] = []
    for index, values in enumerate(sequences):
        data.append((
            _ReplayStream(values, shape, timed),
            _VideoTarget(frame_tracks=[target() for _ in values]),
            {"id": f"v{index}"},
        ))
    return _Dataset(data)


class _DeterministicResizeImage:
    """Stand-in for ``PIL.Image`` that resizes by block-mean downsampling.

    Perceptual hashing routes through ``preprocessing.resize``, which uses PIL when it is
    importable and ``scipy.ndimage.zoom`` otherwise, and the two resamplers do not agree at
    the bit level. That makes near-duplicate expectations for a noisy frame pair
    backend-dependent: under scipy's resampler the pair's dhashes collapse to identical. Pinning
    this stub fixes the resampler for tests that assert on a specific Hamming distance, whether
    or not pillow happens to be installed.
    """

    class Resampling:
        LANCZOS = 1

    def __init__(self, array: np.ndarray):
        self._array = np.asarray(array)

    @classmethod
    def fromarray(cls, array):
        return cls(array)

    def resize(self, size, *args):
        rows = np.array_split(self._array, size[0], axis=0)
        blocks = [cols.mean() for row in rows for cols in np.array_split(row, size[1], axis=1)]
        return np.clip(np.rint(np.asarray(blocks).reshape(size)), 0, 255).astype(np.uint8)


@pytest.mark.required
class TestTrackingDispatch:
    """Duplicates over a multi-object-tracking dataset, one frame at a time."""

    def test_finds_a_clip_replayed_from_another_sequence(self):
        # Sequence 1 is frames 2-4 of sequence 0, re-encoded byte-identically.
        dataset = make_tracking_dataset([[10, 20, 30, 40, 50, 60], [30, 40, 50]])
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset)
        assert result.exact == [[(0, 2), (1, 0)], [(0, 3), (1, 1)], [(0, 4), (1, 2)]]

    def test_levels_use_the_tracking_vocabulary(self):
        dataset = make_tracking_dataset([[10, 20, 10]])
        frame = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset).data()
        assert set(frame["level"].to_list()) == {"unit"}
        assert "unit_indices" in frame.columns

    def test_members_name_the_sequence_and_the_frame(self):
        """Sequence index alone cannot name a frame; three groups would read identically."""
        dataset = make_tracking_dataset([[10, 20, 30], [10, 20, 30]])
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset)
        assert result.frames.exact == [[(0, 0), (1, 0)], [(0, 1), (1, 1)], [(0, 2), (1, 2)]]
        # A whole-sequence member is a sequence, so it needs no frame beside it.
        assert result.sequences.exact == [[0, 1]]

    def test_frames_and_items_are_the_same_rows(self):
        dataset = make_tracking_dataset([[10, 20, 10]])
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset)
        assert len(result.frames) == len(result.items) == len(result)
        assert len(result.detections) == 0

    def test_within_sequence_repeats_are_found(self):
        dataset = make_tracking_dataset([[10, 20, 10, 20, 10]])
        assert Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset).exact == [
            [(0, 0), (0, 2), (0, 4)],
            [(0, 1), (0, 3)],
        ]

    def test_frame_sample_thins_the_scan(self):
        dataset = make_tracking_dataset([list(range(10, 100, 10))])
        assert len(Duplicates(frame_sample=3).evaluate(dataset).data()) >= 0
        frames = SequenceFrames(dataset, Stride(3))
        assert len(frames) == 3

    def test_frame_sample_accepts_a_selector_directly(self):
        dataset = make_tracking_dataset([[10, 20, 30, 40]])
        result = Duplicates(flags=ImageStats.HASH_XXHASH, frame_sample=FrameIndices({0: [0, 2]})).evaluate(dataset)
        assert result.hash_radius == 0
        assert len(result) == 0  # frames 0 and 2 differ

    def test_an_already_wrapped_frame_view_passes_through(self):
        dataset = make_tracking_dataset([[10, 20, 10]])
        direct = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset)
        wrapped = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(SequenceFrames(dataset))
        assert direct.exact == wrapped.exact

    def test_hash_radius_applies_to_frames(self, monkeypatch):
        # The dhash expectation below holds for one resampler only; pin a deterministic one so
        # it holds whether or not pillow is installed.
        monkeypatch.setattr("dataeval.utils.preprocessing.Image", _DeterministicResizeImage)
        rng = np.random.RandomState(3)
        base = rng.randint(0, 256, (3, 64, 64)).astype(np.uint8)
        noisy = np.clip(base.astype(np.int16) + rng.randint(-12, 13, base.shape), 0, 255).astype(np.uint8)

        dataset = make_tracking_dataset([[base, noisy]], shape=(3, 64, 64))
        assert Duplicates(hash_radius=0).evaluate(dataset).near == []
        assert len(Duplicates(hash_radius=6).evaluate(dataset).near) == 1

    def test_mixing_tracking_and_image_datasets_is_refused(self):
        dataset = make_tracking_dataset([[10, 20]])
        images = [np.full((3, 24, 32), 10, dtype=np.uint8)]
        with pytest.raises(ValueError, match="cannot combine tracking datasets"):
            Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset, images)

    def test_cross_dataset_tracking(self):
        train = make_tracking_dataset([[10, 20, 30]])
        test = make_tracking_dataset([[30, 40]])
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(train, test)
        assert "dataset_indices" in result.data().columns
        # One group, whose members are split by the dataset each came from.
        assert result.exact == {0: [[(0, 2)]], 1: [[(0, 0)]]}

    def test_image_datasets_are_untouched(self):
        """The dispatch must be invisible to every path that existed before it."""
        images = _near_variants()
        frame = Duplicates(hash_radius=4).evaluate(images).data()
        assert set(frame["level"].to_list()) == {"item"}
        assert "unit_indices" not in frame.columns


@pytest.mark.required
class TestTemporalRedundancy:
    """Stretches of a video that repeat themselves -- the redundancy still imagery does not have."""

    def test_a_static_stretch_is_reported_as_one_redundant_group(self):
        dataset = make_tracking_dataset([[10, 10, 10, 10, 10, 40, 70, 100]])
        frame = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset).data()
        redundant = frame.filter(pl.col("dup_type") == "redundant")
        assert redundant.shape[0] == 1
        assert redundant["unit_indices"].to_list() == [[0, 1, 2, 3, 4]]
        assert redundant["item_indices"].to_list() == [[0, 0, 0, 0, 0]]

    def test_runs_never_cross_a_sequence_boundary(self):
        """Two videos are not adjacent in time, whatever their frames look like."""
        dataset = make_tracking_dataset([[10, 10], [10, 10]])
        redundant = (
            Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset).data().filter(pl.col("dup_type") == "redundant")
        )
        assert sorted(redundant["item_indices"].to_list()) == [[0, 0], [1, 1]]

    def test_a_changing_sequence_has_no_redundancy(self):
        dataset = make_tracking_dataset([[10, 40, 70, 100]])
        frame = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset).data()
        assert frame.filter(pl.col("dup_type") == "redundant").shape[0] == 0

    def test_radius_widens_what_counts_as_unchanged(self):
        from dataeval.core import hamming_distance, phash

        rng = np.random.RandomState(5)
        base = rng.randint(0, 256, (3, 64, 64)).astype(np.uint8)

        # Dynamically generate 4 frames where distance to predecessor is > 4
        # but distance to base frame is <= 12, so a radius 12 anchor drops them all.
        drifted = [base]
        for _ in range(4):
            for mag in range(1, 100):
                step = rng.randint(-mag, mag + 1, base.shape)
                cand = np.clip(drifted[-1].astype(np.int16) + step, 0, 255).astype(np.uint8)
                dist_prev = hamming_distance(phash(drifted[-1]), phash(cand))
                dist_base = hamming_distance(phash(base), phash(cand))
                if dist_prev > 4 and dist_base <= 12:
                    drifted.append(cand)
                    break

        dataset = make_tracking_dataset([drifted], shape=(3, 64, 64))
        strict = Duplicates(redundancy_radius=4).evaluate(dataset).data()
        loose = Duplicates(redundancy_radius=12).evaluate(dataset).data()
        assert strict.filter(pl.col("dup_type") == "redundant").shape[0] == 0
        assert loose.filter(pl.col("dup_type") == "redundant")["unit_indices"].to_list() == [[0, 1, 2, 3, 4]]

    def test_image_datasets_have_no_redundant_rows(self):
        """An image dataset has no temporal order for a run to span."""
        frame = Duplicates(redundancy_radius=8).evaluate(_near_variants()).data()
        assert "redundant" not in set(frame["dup_type"].to_list())

    def test_default_radius(self):
        assert Duplicates().redundancy_radius == 4
        assert Duplicates(redundancy_radius=2).redundancy_radius == 2


@pytest.mark.required
class TestAggregateBySequence:
    def test_reports_redundancy_and_duplication_per_sequence(self):
        dataset = make_tracking_dataset([[10, 10, 10, 10, 10, 40, 70, 100], [10, 50, 90]])
        summary = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset).aggregate_by_sequence()
        assert summary["sequence"].to_list() == [0, 1]
        assert summary["n_frames"].to_list() == [8, 3]
        assert summary["redundant_frames"].to_list() == [4, 0]
        assert summary["redundant_fraction"].to_list() == pytest.approx([0.5, 0.0])

    def test_a_sequence_with_nothing_found_still_gets_a_row(self):
        dataset = make_tracking_dataset([[10, 40, 70], [11, 41, 71]])
        summary = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset).aggregate_by_sequence()
        assert summary["sequence"].to_list() == [0, 1]
        assert summary["redundant_fraction"].to_list() == [0.0, 0.0]
        assert summary["group_count"].to_list() == [0, 0]

    def test_duplicate_frames_counts_each_frame_once(self):
        # Frame 0 of each sequence is in one group; sequence 0 also repeats it internally.
        dataset = make_tracking_dataset([[10, 40, 10], [10, 50]])
        summary = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset).aggregate_by_sequence()
        assert summary.filter(pl.col("sequence") == 0)["duplicate_frames"].item() == 2
        assert summary.filter(pl.col("sequence") == 1)["duplicate_frames"].item() == 1

    def test_fraction_is_over_the_frames_actually_measured(self):
        """Thinning changes the denominator, which is why n_frames is reported beside it."""
        dataset = make_tracking_dataset([[10] * 8])
        full = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset).aggregate_by_sequence()
        thinned = Duplicates(flags=ImageStats.HASH_XXHASH, frame_sample=2).evaluate(dataset).aggregate_by_sequence()
        assert full["n_frames"].item() == 8
        assert thinned["n_frames"].item() == 4
        assert full["redundant_fraction"].item() == pytest.approx(7 / 8)
        assert thinned["redundant_fraction"].item() == pytest.approx(3 / 4)

    def test_refused_for_image_datasets(self):
        result = Duplicates().evaluate(_near_variants())
        with pytest.raises(ValueError, match="requires results from a multi-object-tracking"):
            result.aggregate_by_sequence()

    def test_survives_a_result_with_no_groups(self):
        dataset = make_tracking_dataset([[10, 40, 70]])
        summary = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset).aggregate_by_sequence()
        assert summary.shape[0] == 1
        assert summary["group_count"].to_list() == [0]


@pytest.mark.required
class TestFrameViewIsCarriedThrough:
    """A frame view's coordinates have to survive every path that rebuilds the results."""

    def test_each_datasets_own_sequence_numbering_is_reported(self):
        """The frame maps are laid end to end, so a per-dataset index reads the wrong map."""
        train = make_tracking_dataset([[10, 20, 30]])
        test = make_tracking_dataset([[99], [10, 20, 30]])
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(train, test)
        # Dataset 1's copies live in its *second* sequence, not its first.
        assert result.frames.exact == {0: [[(0, 0)], [(0, 1)], [(0, 2)]], 1: [[(1, 0)], [(1, 1)], [(1, 2)]]}
        # And the two videos are the same video, which is its own relation.
        assert result.sequences.exact == {0: [[0]], 1: [[1]]}

    def test_cross_dataset_redundancy_is_found_and_never_crosses_a_dataset(self):
        first = make_tracking_dataset([[10, 10, 10]])
        second = make_tracking_dataset([[20, 20, 20]])
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(first, second)
        redundant = result.data().filter(pl.col("dup_type") == "redundant")
        assert redundant["unit_indices"].to_list() == [[0, 1, 2], [0, 1, 2]]
        assert redundant["dataset_indices"].to_list() == [[0, 0, 0], [1, 1, 1]]

    def test_aggregate_by_sequence_refuses_several_datasets(self):
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(
            make_tracking_dataset([[10, 20]]), make_tracking_dataset([[30, 40]])
        )
        with pytest.raises(ValueError, match="only works with output from a single dataset"):
            result.aggregate_by_sequence()

    def test_with_radius_keeps_the_frame_view(self):
        """Re-detection rebuilds the rows, and must rebuild them as frames rather than images."""
        dataset = make_tracking_dataset([[10, 20, 30, 40], [30, 40, 50]])
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset).with_radius(0)
        assert set(result.data()["level"].to_list()) == {"unit"}
        assert result.frame_map is not None
        assert result.exact == [[(0, 2), (1, 0)], [(0, 3), (1, 1)]]

    def test_with_radius_keeps_the_redundant_groups(self):
        dataset = make_tracking_dataset([[10, 10, 10, 10, 40, 70]])
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset)
        assert result.with_radius(0).aggregate_by_sequence().to_dicts() == result.aggregate_by_sequence().to_dicts()

    def test_aggregate_by_image_rows_are_frames_not_sequences(self):
        dataset = make_tracking_dataset([[10, 20, 30, 40], [30, 40, 50]])
        summary = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset).aggregate_by_image()
        assert summary.shape[0] == 4
        located = sorted(zip(summary["item_index"], summary["unit_index"], strict=True))
        assert located == [(0, 2), (0, 3), (1, 0), (1, 1)]

    def test_aggregate_by_image_is_unchanged_for_images(self):
        summary = Duplicates(hash_radius=4).evaluate(_near_variants()).aggregate_by_image()
        assert summary.columns == ["item_index", "group_count", "dup_types", "methods"]


def _fills(start: int, count: int, step: int = 3) -> list[int]:
    """Distinct per-frame fill values, so each frame hashes differently.

    Asserts the range, because a fill above 255 overflows the uint8 frames these build and the
    failure surfaces far from the fixture that caused it.
    """
    values = list(range(start, start + count * step, step))
    assert values[-1] < 256, f"fixture fill {values[-1]} exceeds uint8"
    return values


@pytest.mark.required
class TestSequenceRelations:
    """Whole-video relations: the same collect twice, and a clip cut from a longer source."""

    def test_two_encodes_of_one_collect_are_an_exact_sequence_match(self):
        source = _fills(10, 20)
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(make_tracking_dataset([source, list(source)]))
        assert result.sequences.exact == [[0, 1]]

    def test_a_reordered_video_is_not_an_exact_sequence_match(self):
        """The same frames in a different order are not the same video."""
        source = _fills(10, 20)
        shuffled = source[10:] + source[:10]
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(make_tracking_dataset([source, shuffled]))
        assert result.sequences.exact == []

    def test_a_contained_clip_reads_as_directed_containment(self):
        """The leakage signature: one direction near 1.0, the other near 0."""
        source = _fills(10, 40)
        dataset = make_tracking_dataset([source, source[18:30]])
        frame = Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=5).evaluate(dataset).data()
        segment = frame.filter(pl.col("dup_type") == "segment")
        assert segment.shape[0] == 1
        row = segment.row(0, named=True)
        assert row["item_indices"] == [0, 1]
        assert row["span_start"] == [18, 0]
        assert row["span_end"] == [29, 11]
        assert row["containment"] == pytest.approx([12 / 40, 1.0])

    def test_a_full_re_encode_is_contained_both_ways(self):
        source = _fills(10, 30)
        dataset = make_tracking_dataset([source, list(source)])
        segment = (
            Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=5)
            .evaluate(dataset)
            .data()
            .filter(pl.col("dup_type") == "segment")
        )
        assert segment.row(0, named=True)["containment"] == pytest.approx([1.0, 1.0])

    def test_two_clips_overlapping_in_the_middle(self):
        source = _fills(10, 60)
        dataset = make_tracking_dataset([source[0:30], source[20:50]])
        segment = (
            Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=5)
            .evaluate(dataset)
            .data()
            .filter(pl.col("dup_type") == "segment")
        )
        row = segment.row(0, named=True)
        assert row["span_start"] == [20, 0]
        assert row["span_end"] == [29, 9]

    def test_unrelated_videos_share_nothing(self):
        dataset = make_tracking_dataset([_fills(10, 30), _fills(11, 30)])
        frame = Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=5).evaluate(dataset).data()
        assert frame.filter(pl.col("level") == "sequence").shape[0] == 0

    def test_min_segment_frames_suppresses_a_shared_intro(self):
        """A shared title card is not a shared video."""
        intro = _fills(200, 4, step=10)
        dataset = make_tracking_dataset([intro + _fills(0, 30, 2), intro + _fills(61, 30, 2)])
        loose = Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=2).evaluate(dataset).data()
        strict = Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=10).evaluate(dataset).data()
        assert loose.filter(pl.col("dup_type") == "segment").shape[0] == 1
        assert strict.filter(pl.col("dup_type") == "segment").shape[0] == 0

    def test_max_segment_gap_bridges_a_dropped_frame(self):
        source = _fills(10, 40)
        clip = source[10:20] + [7] + source[21:30]
        dataset = make_tracking_dataset([source, clip])
        bridged = Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=15, max_segment_gap=2).evaluate(dataset)
        split = Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=15, max_segment_gap=0).evaluate(dataset)
        assert bridged.data().filter(pl.col("dup_type") == "segment").shape[0] == 1
        assert split.data().filter(pl.col("dup_type") == "segment").shape[0] == 0

    def test_image_datasets_get_no_sequence_rows(self):
        frame = Duplicates(hash_radius=4).evaluate(_near_variants()).data()
        assert "sequence" not in set(frame["level"].to_list())
        assert "span_start" not in frame.columns

    def test_the_method_column_names_the_hash_that_found_it(self):
        source = _fills(10, 30)
        dataset = make_tracking_dataset([source, list(source)])
        segment = (
            Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=5)
            .evaluate(dataset)
            .data()
            .filter(pl.col("dup_type") == "segment")
        )
        assert segment.row(0, named=True)["methods"] == ["xxhash"]

    def test_defaults(self):
        detector = Duplicates()
        assert detector.min_segment_frames == 30
        assert detector.max_segment_gap == 5
        assert detector.segment_offset_tolerance == 0


@pytest.mark.required
class TestSequenceRelationsSurviveRedetection:
    def test_with_radius_keeps_the_sequence_relations(self):
        """The whole point of storing the policy: a re-detection must not lose findings."""
        source = _fills(10, 40)
        dataset = make_tracking_dataset([source, source[18:30]])
        original = Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=5).evaluate(dataset)
        redetected = original.with_radius(0)
        for output in (original, redetected):
            frame = output.data()
            assert frame.filter(pl.col("dup_type") == "segment").shape[0] == 1
            assert frame.filter(pl.col("level") == "sequence").shape[0] >= 1
        assert redetected.min_segment_frames == 5

    def test_cross_dataset_leakage_is_found_and_directed(self):
        """A test clip cut from a training video -- the relation this whole tier exists for."""
        source = _fills(10, 40)
        train = make_tracking_dataset([source])
        test = make_tracking_dataset([source[18:30]])
        result = Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=5).evaluate(train, test)
        segment = result.data().filter(pl.col("dup_type") == "segment")
        assert segment.shape[0] == 1
        row = segment.row(0, named=True)
        assert row["dataset_indices"] == [0, 1]
        assert row["containment"] == pytest.approx([12 / 40, 1.0])


def _held(values: list[int]) -> list[int]:
    """Each frame held for two -- a copy played back at half speed."""
    return [value for value in values for _ in (0, 1)]


@pytest.mark.required
class TestAlignmentVerification:
    """Warped matching: the relation a constant offset cannot express."""

    def _rows(self, dataset, dup_type: str, **kwargs) -> pl.DataFrame:
        kwargs.setdefault("min_segment_frames", 5)
        result = Duplicates(flags=ImageStats.HASH_XXHASH, **kwargs).evaluate(dataset)
        return result.data().filter(pl.col("dup_type") == dup_type)

    def test_a_half_speed_copy_is_found_where_segments_find_nothing(self):
        """Every offset holds two frames, so the diagonals are two long and none survives."""
        source = _fills(10, 20)
        dataset = make_tracking_dataset([source, _held(source)])
        assert self._rows(dataset, "segment", verify_alignment=0).shape[0] == 0
        aligned = self._rows(dataset, "aligned", verify_alignment=0)
        assert aligned.shape[0] == 1
        row = aligned.row(0, named=True)
        assert row["item_indices"] == [0, 1]
        assert row["mean_distance"] == 0.0

    def test_warped_matching_is_off_unless_asked_for(self):
        source = _fills(10, 20)
        dataset = make_tracking_dataset([source, _held(source)])
        assert self._rows(dataset, "aligned").shape[0] == 0

    def test_the_aligned_span_names_source_frames(self):
        source = _fills(10, 20)
        dataset = make_tracking_dataset([source, _held(source)])
        row = self._rows(dataset, "aligned", verify_alignment=0).row(0, named=True)
        query_start, candidate_start = row["span_start"]
        query_end, candidate_end = row["span_end"]
        assert (query_start, query_end) == (0, 19)
        assert candidate_end - candidate_start >= 30

    def test_containment_says_how_much_of_the_candidate_took_part(self):
        """The query is aligned whole, so its own figure is 1.0 by construction."""
        source = _fills(10, 20)
        dataset = make_tracking_dataset([source, _held(source)])
        row = self._rows(dataset, "aligned", verify_alignment=0).row(0, named=True)
        assert row["containment"][0] == 1.0
        assert row["containment"][1] > 0.9

    def test_a_pair_the_segments_explain_is_not_warped_as_well(self):
        """Warping is quadratic; a pair already accounted for does not pay for it twice."""
        source = _fills(10, 40)
        dataset = make_tracking_dataset([source, source[18:30]])
        assert self._rows(dataset, "segment", verify_alignment=8).shape[0] == 1
        assert self._rows(dataset, "aligned", verify_alignment=8).shape[0] == 0

    def test_a_pair_sharing_one_frame_costs_too_much_to_report(self):
        source = _fills(10, 20)
        dataset = make_tracking_dataset([source, [source[0], *_fills(200, 19)]])
        assert self._rows(dataset, "aligned", verify_alignment=8).shape[0] == 0

    def test_a_zero_cost_alignment_collapsed_onto_one_frame_is_refused(self):
        """An unconstrained warp folds a run onto a single frame for free; a moment is not a stretch."""
        source = _fills(10, 20)
        repeated = [source[0]] * 6
        codes, _ = pack_hashes([f"{value:016x}" for value in repeated])
        against, _ = pack_hashes([f"{value:016x}" for value in source])
        collapsed = align_subsequence(codes, against)
        assert collapsed["normalized_cost"] == 0.0
        assert collapsed["end"] - collapsed["start"] + 1 == 1

        dataset = make_tracking_dataset([source, repeated])
        assert self._rows(dataset, "aligned", verify_alignment=0).shape[0] == 0

    def test_a_run_shorter_than_the_length_bar_is_not_warped(self):
        source = _fills(10, 20)
        dataset = make_tracking_dataset([source, _held(source)[:8]])
        assert self._rows(dataset, "aligned", verify_alignment=0, min_segment_frames=20).shape[0] == 0

    def test_a_negative_threshold_is_rejected_before_anything_is_measured(self):
        with pytest.raises(ValueError, match="verify_alignment must be non-negative"):
            Duplicates(flags=ImageStats.HASH_XXHASH, verify_alignment=-1).evaluate(MockDataset())

    def test_alignments_survive_a_redetection(self):
        source = _fills(10, 20)
        dataset = make_tracking_dataset([source, _held(source)])
        original = Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=5, verify_alignment=0).evaluate(dataset)
        redetected = original.with_radius(0)
        assert redetected.verify_alignment == 0
        for output in (original, redetected):
            assert output.data().filter(pl.col("dup_type") == "aligned").shape[0] == 1

    def test_cross_dataset_speed_edits_are_found(self):
        source = _fills(10, 20)
        train = make_tracking_dataset([source])
        test = make_tracking_dataset([_held(source)])
        result = Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=5, verify_alignment=0).evaluate(
            train, test
        )
        aligned = result.data().filter(pl.col("dup_type") == "aligned")
        assert aligned.shape[0] == 1
        assert aligned.row(0, named=True)["dataset_indices"] == [0, 1]


@pytest.mark.required
class TestMeanDistanceColumn:
    """How close the frames of a relation actually were, where the relation measures it."""

    def test_a_redundant_run_reports_how_close_its_frames_were(self):
        source = _fills(10, 6)
        dataset = make_tracking_dataset([[*source, source[-1], source[-1]]])
        rows = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset).data()
        redundant = rows.filter(pl.col("dup_type") == "redundant")
        assert redundant.shape[0] == 1
        assert redundant.row(0, named=True)["mean_distance"] == 0.0

    def test_a_segment_reports_the_mean_over_its_own_pairs(self):
        source = _fills(10, 40)
        dataset = make_tracking_dataset([source, source[18:30]])
        rows = Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=5).evaluate(dataset).data()
        segment = rows.filter(pl.col("dup_type") == "segment")
        assert segment.row(0, named=True)["mean_distance"] == 0.0

    def test_an_image_dataset_carries_no_distance_column(self):
        """Exact and near groups have no single distance to report, so the column is dropped."""
        images = [np.full((3, 16, 16), value, dtype=np.uint8) for value in _fills(10, 5)]
        rows = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate([*images, images[0]]).data()
        assert rows.shape[0] == 1
        assert "mean_distance" not in rows.columns


@pytest.mark.required
class TestSegmentPolicyValidation:
    """A nonsense policy is refused up front, whatever the data -- an image call included."""

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"min_segment_frames": 0}, "min_segment_frames must be at least 1"),
            ({"max_segment_gap": -1}, "must be non-negative"),
            ({"segment_offset_tolerance": -1}, "must be non-negative"),
            ({"verify_alignment": -1}, "verify_alignment must be non-negative"),
        ],
    )
    def test_a_nonsense_policy_is_refused_for_image_data(self, kwargs, match):
        """Image data never reaches the matcher, so an unchecked knob would be silently ignored."""
        with pytest.raises(ValueError, match=match):
            Duplicates(flags=ImageStats.HASH_XXHASH, **kwargs).evaluate(MockDataset())

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"min_segment_frames": 0}, "min_segment_frames must be at least 1"),
            ({"verify_alignment": -1}, "verify_alignment must be non-negative"),
        ],
    )
    def test_a_nonsense_policy_is_refused_for_tracking_data(self, kwargs, match):
        dataset = make_tracking_dataset([_fills(10, 6)])
        with pytest.raises(ValueError, match=match):
            Duplicates(flags=ImageStats.HASH_XXHASH, **kwargs).evaluate(dataset)

    def test_a_pair_too_large_to_warp_loses_its_alignment_and_says_so(self, monkeypatch, caplog):
        """One pair costing too much is skipped, not raised: the other relations still stand."""
        monkeypatch.setattr(_duplicates, "_ALIGNMENT_CELLS", 4)
        source = _fills(10, 20)
        dataset = make_tracking_dataset([source, _held(source)])
        caplog.set_level(logging.INFO, logger="dataeval.quality")
        result = Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=5, verify_alignment=0).evaluate(dataset)
        assert result.data().filter(pl.col("dup_type") == "aligned").shape[0] == 0
        assert "too large a problem" in caplog.text


def make_tracked_dataset(sequences, shape=(3, 24, 32), timed=True) -> Any:
    """A tracking dataset whose detections have distinguishable crops.

    Each sequence is a list of frames; each frame is a list of ``(track_id, fill)``. Detection *j*
    of a frame owns vertical strip *j* of the image and its strip is painted with that detection's
    fill, so a track's crops hash its own fills over time and two tracks in one frame are told
    apart -- which flat frames, whose every crop is the same colour, cannot do.
    """
    from dataclasses import dataclass

    @dataclass
    class _FrameTarget:
        track_ids: np.ndarray
        boxes: np.ndarray
        scores: np.ndarray
        labels: np.ndarray

    @dataclass
    class _VideoTarget:
        frame_tracks: list

    class _Dataset:
        def __init__(self, data):
            self._data = data
            self.metadata = {"id": "videos", "index2label": {0: "thing"}}

        def __len__(self):
            return len(self._data)

        def __getitem__(self, index):
            return self._data[index]

    channels, height, width = shape
    data: list[Any] = []
    for index, frames in enumerate(sequences):
        slots = max((len(frame) for frame in frames), default=1)
        strip = width // slots
        pixels, targets = [], []
        for frame in frames:
            image = np.zeros(shape, dtype=np.uint8)
            boxes = []
            for slot, (_, fill) in enumerate(frame):
                assert 0 <= fill < 256, f"fixture fill {fill} exceeds uint8"
                left, right = slot * strip, (slot + 1) * strip
                # A ramp inside the strip, so a crop's hash depends on its own content: a flat
                # patch has a degenerate perceptual hash and every flat patch shares it.
                image[:, :, left:right] = (fill + np.arange(strip, dtype=np.int64)[None, :] % 64).astype(np.uint8)
                boxes.append([float(left), 0.0, float(right), float(height)])
            pixels.append(image)
            targets.append(
                _FrameTarget(
                    track_ids=np.array([track for track, _ in frame], dtype=np.int64),
                    boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
                    scores=np.ones(len(frame), dtype=np.float32),
                    labels=np.zeros(len(frame), dtype=np.int64),
                )
            )
        data.append((_ReplayStream(pixels, shape, timed), _VideoTarget(frame_tracks=targets), {"id": f"v{index}"}))
    _ = channels
    return _Dataset(data)


def _track(track_id: int, fills: list[int]) -> list[tuple[int, int]]:
    """One track's per-frame ``(track_id, fill)`` entries."""
    return [(track_id, fill) for fill in fills]


@pytest.mark.required
class TestTrackDuplicates:
    """One object under two track ids, and a track carried along with a reused clip."""

    def _tracks(self, sequences, per_target: bool = True, **kwargs) -> pl.DataFrame:
        kwargs.setdefault("min_track_frames", 5)
        result = Duplicates(flags=ImageStats.HASH_XXHASH, **kwargs).evaluate(
            make_tracked_dataset(sequences), per_target=per_target
        )
        return result.data().filter(pl.col("level") == "track")

    def test_one_object_under_two_ids_is_an_exact_track_duplicate(self):
        fills = _fills(10, 8)
        rows = self._tracks([[[(0, fill), (1, fill)] for fill in fills]])
        exact = rows.filter(pl.col("dup_type") == "exact")
        assert exact.shape[0] == 1
        row = exact.row(0, named=True)
        assert row["item_indices"] == [0, 0]
        assert row["track_indices"] == [0, 1]

    def test_two_tracks_of_the_same_object_share_a_stretch(self):
        fills = _fills(10, 8)
        rows = self._tracks([[[(0, fill), (1, fill)] for fill in fills]])
        segment = rows.filter(pl.col("dup_type") == "segment")
        assert segment.shape[0] == 1
        row = segment.row(0, named=True)
        assert row["track_indices"] == [0, 1]
        assert row["span_start"] == [0, 0]
        assert row["span_end"] == [7, 7]
        assert row["containment"] == pytest.approx([1.0, 1.0])

    def test_two_different_objects_are_not_duplicates(self):
        fills = _fills(10, 8)
        assert self._tracks([[[(0, fill), (1, fill + 100)] for fill in fills]]).shape[0] == 0

    def test_a_track_reused_in_another_sequence_is_found(self):
        """The clip was copied and carried its annotations with it, under a different id."""
        fills = _fills(10, 8)
        rows = self._tracks([[[(0, fill)] for fill in fills], [[(7, fill)] for fill in fills]])
        assert rows.filter(pl.col("dup_type") == "exact").row(0, named=True)["item_indices"] == [0, 1]
        assert rows.filter(pl.col("dup_type") == "exact").row(0, named=True)["track_indices"] == [0, 7]

    def test_a_track_id_is_reported_as_the_annotation_gives_it(self):
        """Not a dense re-numbering: a reported track is one a reader can look up in their data."""
        fills = _fills(10, 8)
        rows = self._tracks([[[(41, fill), (99, fill)] for fill in fills]])
        assert rows.filter(pl.col("dup_type") == "exact").row(0, named=True)["track_indices"] == [41, 99]

    def test_unlinked_detections_are_left_out_rather_than_pooled(self):
        """Pooling every ``-1`` makes one phantom track that duplicates everything unassigned."""
        fills = _fills(10, 8)
        assert self._tracks([[[(-1, fill), (-1, fill)] for fill in fills]]).shape[0] == 0

    def test_a_mix_of_tracked_and_unlinked_keeps_only_the_tracked(self):
        fills = _fills(10, 8)
        frames = [[(0, fill), (1, fill), (-1, fill)] for fill in fills]
        rows = self._tracks([frames])
        assert rows.filter(pl.col("dup_type") == "exact").row(0, named=True)["track_indices"] == [0, 1]

    def test_track_relations_need_the_detection_hashes_they_read(self):
        """Off unless ``per_target`` asks for the crop hashes -- which is what makes them opt-in."""
        fills = _fills(10, 8)
        assert self._tracks([[[(0, fill), (1, fill)] for fill in fills]], per_target=False).shape[0] == 0

    def test_min_track_frames_gates_shared_stretches_but_not_exact_matches(self):
        fills = _fills(10, 8)
        rows = self._tracks([[[(0, fill), (1, fill)] for fill in fills]], min_track_frames=20)
        assert rows.filter(pl.col("dup_type") == "segment").shape[0] == 0
        assert rows.filter(pl.col("dup_type") == "exact").shape[0] == 1

    def test_a_partial_overlap_reads_as_directed_containment(self):
        """Track 1 covers the second half of track 0 -- a split annotated twice over its overlap."""
        fills = _fills(10, 12)
        frames = [[(0, fill)] if index < 6 else [(0, fill), (1, fill)] for index, fill in enumerate(fills)]
        rows = self._tracks([frames], min_track_frames=5)
        segment = rows.filter(pl.col("dup_type") == "segment")
        assert segment.shape[0] == 1
        assert segment.row(0, named=True)["containment"] == pytest.approx([0.5, 1.0])

    def test_an_image_dataset_has_no_track_rows(self):
        images = [np.full((3, 16, 16), value, dtype=np.uint8) for value in _fills(10, 5)]
        rows = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate([*images, images[0]]).data()
        assert "track_indices" not in rows.columns

    def test_a_nonsense_track_length_is_refused_up_front(self):
        with pytest.raises(ValueError, match="min_track_frames must be at least 1"):
            Duplicates(flags=ImageStats.HASH_XXHASH, min_track_frames=0).evaluate(MockDataset())

    def test_track_relations_survive_a_redetection(self):
        fills = _fills(10, 8)
        dataset = make_tracked_dataset([[[(0, fill), (1, fill)] for fill in fills]])
        original = Duplicates(flags=ImageStats.HASH_XXHASH, min_track_frames=5).evaluate(dataset, per_target=True)
        redetected = original.with_radius(0)
        assert redetected.min_track_frames == 5
        for output in (original, redetected):
            assert output.tracks.data().shape[0] == 2

    def test_the_tracks_accessor_selects_only_track_rows(self):
        fills = _fills(10, 8)
        dataset = make_tracked_dataset([[[(0, fill), (1, fill)] for fill in fills]])
        result = Duplicates(flags=ImageStats.HASH_XXHASH, min_track_frames=5).evaluate(dataset, per_target=True)
        assert set(result.tracks.data()["level"].to_list()) == {"track"}
        assert result.tracks.data().shape[0] < result.data().shape[0]

    def test_cross_dataset_track_reuse_is_found(self):
        fills = _fills(10, 8)
        train = make_tracked_dataset([[[(0, fill)] for fill in fills]])
        test = make_tracked_dataset([[[(7, fill)] for fill in fills]])
        result = Duplicates(flags=ImageStats.HASH_XXHASH, min_track_frames=5).evaluate(train, test, per_target=True)
        rows = result.data().filter(pl.col("level") == "track")
        assert rows.shape[0] == 2
        assert rows.row(0, named=True)["dataset_indices"] == [0, 1]
        assert rows.row(0, named=True)["track_indices"] == [0, 7]

    def test_aggregations_survive_track_rows(self):
        """A track row names neither a frame nor a sequence's frames, and must not be exploded."""
        fills = _fills(10, 8)
        dataset = make_tracked_dataset([[[(0, fill), (1, fill)] for fill in fills]])
        result = Duplicates(flags=ImageStats.HASH_XXHASH, min_track_frames=5).evaluate(dataset, per_target=True)
        assert result.aggregate_by_image().shape[0] > 0
        by_sequence = result.aggregate_by_sequence()
        assert by_sequence.shape[0] == 1
        assert by_sequence.row(0, named=True)["group_count"] > 0
        assert result.aggregate_by_group().shape[0] == result.data().shape[0]


def _segments(*spans: tuple[int, int, int, int, int, float]) -> Any:
    """Build a SegmentMatchResult from ``(q_start, q_end, c_start, c_end, n_matched, mean)`` rows."""
    fields = np.array(spans, dtype=np.float64)
    query_start, candidate_start = fields[:, 0].astype(np.intp), fields[:, 2].astype(np.intp)
    return {
        "query_start": query_start,
        "query_end": fields[:, 1].astype(np.intp),
        "candidate_start": candidate_start,
        "candidate_end": fields[:, 3].astype(np.intp),
        "offset": (candidate_start - query_start).astype(np.intp),
        "n_matched": fields[:, 4].astype(np.intp),
        "mean_distance": fields[:, 5],
        "density": np.ones(len(spans)),
    }


@pytest.mark.required
class TestDominantSegments:
    """One stretch, reported once -- not once per neighbouring diagonal."""

    def test_parallel_diagonals_collapse_to_the_best_one(self):
        """Periodic content matches its copy at several offsets; only one is the relation."""
        loop = _fills(10, 10) * 3
        dataset = make_tracking_dataset([loop, list(loop)])
        rows = Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=5).evaluate(dataset).data()
        segment = rows.filter(pl.col("dup_type") == "segment")
        assert segment.shape[0] == 1
        row = segment.row(0, named=True)
        assert (row["span_start"], row["span_end"]) == ([0, 0], [29, 29])
        assert row["mean_distance"] == 0.0

    def test_the_kept_segment_is_the_closest_of_the_diagonals(self):
        kept = _dominant(_segments((0, 59, 1, 60, 55, 4.2), (0, 59, 0, 59, 60, 1.0), (0, 59, 2, 61, 13, 5.3)))
        assert kept["offset"].tolist() == [0]
        assert kept["mean_distance"].tolist() == [1.0]

    def test_disjoint_stretches_are_both_kept(self):
        """Two different parts of one video appearing in another are two relations, not one."""
        kept = _dominant(_segments((0, 40, 100, 140, 41, 0.0), (200, 240, 500, 540, 41, 0.0)))
        assert len(kept["offset"]) == 2

    def test_content_repeated_twice_in_the_candidate_is_kept_twice(self):
        """Same query span, two different places in the candidate -- the query overlap alone would lose one."""
        kept = _dominant(_segments((0, 40, 100, 140, 41, 0.0), (0, 40, 900, 940, 41, 0.0)))
        assert sorted(kept["candidate_start"].tolist()) == [100, 900]

    def test_a_partial_overlap_below_the_bar_survives(self):
        kept = _dominant(_segments((0, 99, 0, 99, 100, 0.0), (90, 189, 90, 189, 100, 0.0)))
        assert len(kept["offset"]) == 2

    def test_a_lone_segment_is_returned_untouched(self):
        only = _segments((0, 40, 5, 45, 41, 1.5))
        assert _dominant(only)["offset"].tolist() == only["offset"].tolist()


@pytest.mark.required
class TestSequenceSummaryCounts:
    """`aggregate_by_sequence` separates a video repeating itself from a video copied elsewhere."""

    def test_a_video_that_only_repeats_itself_is_not_called_duplicated(self):
        """Consecutive frames resemble one another in any video; that is redundancy, not duplication."""
        held = [fill for fill in _fills(10, 8) for _ in (0, 1)]
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(make_tracking_dataset([held]))
        row = result.aggregate_by_sequence().row(0, named=True)
        assert row["redundant_frames"] == 8
        assert row["duplicate_frames"] == 0
        assert row["shared_with"] == 0

    def test_a_copied_video_reports_every_frame_and_one_partner(self):
        source = _fills(10, 12)
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(make_tracking_dataset([source, list(source)]))
        summary = result.aggregate_by_sequence()
        assert summary["duplicate_frames"].to_list() == [12, 12]
        assert summary["shared_with"].to_list() == [1, 1]

    def test_partners_are_counted_distinctly(self):
        source = _fills(10, 12)
        dataset = make_tracking_dataset([source, list(source), list(source), _fills(200, 12)])
        summary = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset).aggregate_by_sequence()
        assert summary["shared_with"].to_list() == [2, 2, 2, 0]
        assert summary["duplicate_frames"].to_list() == [12, 12, 12, 0]

    def test_an_unrelated_pair_shares_nothing(self):
        left, right = _fills(10, 10), _fills(150, 10)
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(make_tracking_dataset([left, right]))
        summary = result.aggregate_by_sequence()
        assert summary["duplicate_frames"].to_list() == [0, 0]
        assert summary["shared_with"].to_list() == [0, 0]

    def test_a_whole_sequence_match_counts_as_a_partner(self):
        """A sequence-level row names two videos and carries no frame index to explode."""
        source = _fills(10, 12)
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(make_tracking_dataset([source, list(source)]))
        assert result.data().filter(pl.col("level") == "sequence").shape[0] > 0
        assert result.aggregate_by_sequence()["shared_with"].to_list() == [1, 1]

    def test_the_columns_survive_an_empty_result(self):
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(make_tracking_dataset([_fills(10, 4)]))
        summary = result.aggregate_by_sequence()
        assert summary.shape[0] == 1
        assert summary.row(0, named=True)["shared_with"] == 0


@pytest.mark.required
class TestStrictRadiusNotice:
    """Video meeting the still-image default is said out loud, not silently under-reported."""

    def test_a_tracking_dataset_at_radius_zero_is_warned_about(self, caplog):
        caplog.set_level(logging.WARNING, logger="dataeval.quality")
        Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(make_tracking_dataset([_fills(10, 4)]))
        assert "hash_radius=6 for video" in caplog.text

    def test_a_radius_that_was_chosen_is_left_alone(self, caplog):
        caplog.set_level(logging.WARNING, logger="dataeval.quality")
        Duplicates(flags=ImageStats.HASH_XXHASH, hash_radius=6).evaluate(make_tracking_dataset([_fills(10, 4)]))
        assert "hash_radius" not in caplog.text

    def test_image_data_is_never_warned(self, caplog):
        """The default is right for stills; the notice is about video meeting it."""
        caplog.set_level(logging.WARNING, logger="dataeval.quality")
        images = [np.full((3, 16, 16), value, dtype=np.uint8) for value in _fills(10, 5)]
        Duplicates(flags=ImageStats.HASH_XXHASH).evaluate([*images, images[0]])
        assert "hash_radius" not in caplog.text

    def test_a_cross_dataset_call_warns_once_per_dataset(self, caplog):
        caplog.set_level(logging.WARNING, logger="dataeval.quality")
        Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(
            make_tracking_dataset([_fills(10, 4)]), make_tracking_dataset([_fills(10, 4)])
        )
        assert caplog.text.count("hash_radius=6 for video") == 2


@pytest.mark.required
class TestLevels:
    """`levels` names the answers wanted; per_image/per_target name the measurements taken."""

    def _dataset(self):
        fills = _fills(10, 8)
        return make_tracked_dataset([[[(0, fill), (1, fill)] for fill in fills]])

    def test_naming_one_level_reports_only_that_level(self):
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(self._dataset(), levels="sequence")
        assert set(result.data()["level"].to_list()) <= {"sequence"}

    def test_a_level_not_asked_for_is_not_searched_for(self):
        """The saving, not just the filtering: no frame-level grouping happens at all."""
        dataset = make_tracking_dataset([_fills(10, 12), _fills(10, 12)])
        sequence_only = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset, levels="sequence")
        assert sequence_only.data().filter(pl.col("dup_type") == "redundant").shape[0] == 0
        assert sequence_only.data().filter(pl.col("level") == "unit").shape[0] == 0
        assert sequence_only.data().filter(pl.col("level") == "sequence").shape[0] > 0

    def test_track_level_implies_the_detection_hashes_it_reads(self):
        result = Duplicates(flags=ImageStats.HASH_XXHASH, min_track_frames=5).evaluate(
            self._dataset(), levels=["track"]
        )
        assert result.tracks.data().shape[0] > 0
        assert set(result.data()["level"].to_list()) == {"track"}

    def test_several_levels_are_all_reported(self):
        fills = _fills(10, 8)
        frames = [[(0, fill), (1, fill)] for fill in fills]
        copied = make_tracked_dataset([frames, [list(frame) for frame in frames]])
        result = Duplicates(flags=ImageStats.HASH_XXHASH, min_track_frames=5).evaluate(
            copied, levels=("sequence", "track")
        )
        assert set(result.data()["level"].to_list()) == {"sequence", "track"}

    def test_the_default_matches_the_older_spelling(self):
        dataset = make_tracking_dataset([_fills(10, 8), _fills(10, 8)])
        implied = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset)
        named = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset, levels=("sequence", "unit"))
        assert implied.data().equals(named.data())

    def test_per_target_matches_naming_the_levels_it_implies(self):
        dataset = self._dataset()
        implied = Duplicates(flags=ImageStats.HASH_XXHASH, min_track_frames=5).evaluate(dataset, per_target=True)
        named = Duplicates(flags=ImageStats.HASH_XXHASH, min_track_frames=5).evaluate(
            dataset, levels=("sequence", "unit", "instance", "track")
        )
        assert implied.data().equals(named.data())

    def test_image_datasets_use_the_image_vocabulary(self):
        images = [np.full((3, 16, 16), value, dtype=np.uint8) for value in _fills(10, 5)]
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate([*images, images[0]], levels="item")
        assert set(result.data()["level"].to_list()) == {"item"}

    @pytest.mark.parametrize(("dataset_kind", "level"), [("image", "sequence"), ("tracking", "item")])
    def test_a_level_from_the_other_task_is_refused(self, dataset_kind, level):
        images = [np.full((3, 16, 16), value, dtype=np.uint8) for value in _fills(10, 5)]
        dataset = [*images, images[0]] if dataset_kind == "image" else make_tracking_dataset([_fills(10, 4)])
        with pytest.raises(ValueError, match="not a level of"):
            Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset, levels=level)

    @pytest.mark.parametrize("older", [{"per_image": True}, {"per_target": True}, {"per_image": False}])
    def test_both_spellings_at_once_is_refused(self, older):
        with pytest.raises(ValueError, match="not both"):
            Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(self._dataset(), levels="sequence", **older)

    def test_an_empty_level_list_is_refused(self):
        with pytest.raises(ValueError, match="at least one level"):
            Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(self._dataset(), levels=[])

    def test_a_frame_view_handed_in_directly_keeps_the_tracking_vocabulary(self):
        """A pre-wrapped SequenceFrames is not a tracking dataset, but it still yields frames."""
        dataset = make_tracking_dataset([[10, 20, 10]])
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(SequenceFrames(dataset), levels="unit")
        assert result.exact == [[(0, 0), (0, 2)]]

    def test_levels_survive_a_redetection(self):
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(self._dataset(), levels="sequence")
        redetected = result.with_radius(0)
        assert redetected.levels == frozenset({"sequence"})
        assert set(redetected.data()["level"].to_list()) <= {"sequence"}


@pytest.mark.required
class TestAggregateByPair:
    """The relation-shaped view: one row per pair, with each side's containment its own column."""

    def _copies(self):
        source = _fills(10, 12)
        return make_tracking_dataset([source, list(source), _fills(200, 12)])

    def test_one_row_per_related_pair(self):
        pairs = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(self._copies()).aggregate_by_pair("sequence")
        assert pairs.shape[0] == 1
        row = pairs.row(0, named=True)
        assert (row["item_a"], row["item_b"]) == (0, 1)
        assert row["relations"] == ["exact"]

    def test_frame_level_pairs_count_the_frames_they_share(self):
        pairs = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(self._copies()).aggregate_by_pair("unit")
        assert pairs.shape[0] == 1
        assert pairs.row(0, named=True)["n_groups"] == 12

    def test_a_pair_appears_once_however_it_was_ordered(self):
        pairs = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(self._copies()).aggregate_by_pair()
        keys = list(zip(pairs["level"], pairs["item_a"], pairs["item_b"], strict=True))
        assert len(keys) == len(set(keys))
        assert all(a <= b for _, a, b in keys)

    def test_containment_is_split_into_a_column_per_side(self):
        """The leakage read, without knowing that containment aligns positionally."""
        source = _fills(10, 40)
        dataset = make_tracking_dataset([source, source[18:30]])
        pairs = (
            Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=5)
            .evaluate(dataset)
            .aggregate_by_pair("sequence")
        )
        row = pairs.row(0, named=True)
        assert row["containment_a"] == pytest.approx(12 / 40)
        assert row["containment_b"] == 1.0

    def test_a_cross_dataset_pair_names_both_datasets(self):
        source = _fills(10, 40)
        train = make_tracking_dataset([source])
        test = make_tracking_dataset([source[18:30]])
        pairs = (
            Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=5)
            .evaluate(train, test)
            .aggregate_by_pair("sequence")
        )
        leaks = pairs.filter(pl.col("dataset_a") != pl.col("dataset_b"))
        assert leaks.shape[0] == 1
        assert leaks.row(0, named=True)["containment_b"] == 1.0

    def test_redundant_runs_name_no_pair(self):
        """A run of repeated frames relates a sequence to itself."""
        held = [fill for fill in _fills(10, 8) for _ in (0, 1)]
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(make_tracking_dataset([held]))
        assert result.data().filter(pl.col("dup_type") == "redundant").shape[0] > 0
        assert result.aggregate_by_pair().shape[0] == 0

    def test_a_group_holding_two_frames_of_one_video_names_no_pair(self):
        dataset = make_tracking_dataset([[10, 20, 10]])
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(dataset)
        assert result.exact == [[(0, 0), (0, 2)]]
        assert result.aggregate_by_pair().shape[0] == 0

    def test_track_pairs_carry_both_track_ids(self):
        fills = _fills(10, 8)
        dataset = make_tracked_dataset([[[(3, fill), (9, fill)] for fill in fills]])
        pairs = (
            Duplicates(flags=ImageStats.HASH_XXHASH, min_track_frames=5)
            .evaluate(dataset, levels="track")
            .aggregate_by_pair()
        )
        assert pairs.shape[0] == 1
        row = pairs.row(0, named=True)
        assert (row["track_a"], row["track_b"]) == (3, 9)

    def test_the_track_columns_are_dropped_when_no_track_took_part(self):
        pairs = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(self._copies()).aggregate_by_pair()
        assert "track_a" not in pairs.columns

    def test_an_unknown_level_is_refused_and_names_the_ones_present(self):
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(self._copies())
        with pytest.raises(ValueError, match="not among the levels"):
            result.aggregate_by_pair("track")

    def test_an_empty_result_keeps_the_schema(self):
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(make_tracking_dataset([_fills(10, 4)]))
        pairs = result.aggregate_by_pair()
        assert pairs.shape[0] == 0
        assert "containment_a" in pairs.columns

    def test_an_oversized_expansion_is_refused_and_names_the_way_through(self, monkeypatch):
        monkeypatch.setattr(_duplicates, "_MAX_PAIR_EXPANSION", 3)
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(self._copies())
        with pytest.raises(ValueError, match="Name a narrower `levels`"):
            result.aggregate_by_pair()

    def test_images_pair_at_the_item_level(self):
        images = [np.full((3, 16, 16), value, dtype=np.uint8) for value in _fills(10, 5)]
        pairs = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate([*images, images[0]]).aggregate_by_pair()
        assert pairs.shape[0] == 1
        row = pairs.row(0, named=True)
        assert (row["level"], row["item_a"], row["item_b"]) == ("item", 0, 5)


@pytest.mark.required
class TestCrossingView:
    """The split-boundary view: which relations have members on both sides."""

    def _split(self):
        """A training video that also repeats itself, so not every relation crosses."""
        source = _fills(10, 40)
        train = make_tracking_dataset([[*source, source[0]]])
        return train, make_tracking_dataset([source[18:30]])

    def test_it_keeps_only_relations_that_span_two_datasets(self):
        train, test = self._split()
        result = Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=5).evaluate(train, test)
        crossing = result.crossing.data()
        assert crossing.shape[0] > 0
        assert crossing.shape[0] < result.data().shape[0]
        assert all(len(set(row)) > 1 for row in crossing["dataset_indices"].to_list())

    def test_a_single_dataset_has_no_boundary_to_cross(self):
        source = _fills(10, 12)
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(make_tracking_dataset([source, list(source)]))
        assert result.data().shape[0] > 0
        assert result.crossing.data().shape[0] == 0

    def test_it_composes_with_the_level_views(self):
        train, test = self._split()
        result = Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=5).evaluate(train, test)
        assert result.crossing.sequences.data().shape[0] > 0
        assert set(result.crossing.sequences.data()["level"].to_list()) == {"sequence"}

    def test_leakage_reads_off_the_pair_view(self):
        """The headline question, without a filter that encodes domain knowledge."""
        train, test = self._split()
        result = Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=5).evaluate(train, test)
        pairs = result.crossing.aggregate_by_pair("sequence")
        assert pairs.shape[0] == 1
        assert pairs.row(0, named=True)["containment_b"] == 1.0

    def test_settings_travel_with_the_view(self):
        train, test = self._split()
        result = Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=5).evaluate(train, test)
        assert result.crossing.min_segment_frames == 5
        assert result.crossing.hash_radius == result.hash_radius

    def test_it_narrows_rows_rather_than_detection(self):
        """A row filter cannot survive a re-detection, which rebuilds the rows; the docs say so."""
        train, test = self._split()
        result = Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=5).evaluate(train, test)
        assert result.crossing.with_radius(0).data().shape[0] == result.with_radius(0).data().shape[0]
        assert result.with_radius(0).crossing.data().shape[0] < result.with_radius(0).data().shape[0]

    def test_a_level_view_does_survive_a_redetection(self):
        train, test = self._split()
        result = Duplicates(flags=ImageStats.HASH_XXHASH, min_segment_frames=5).evaluate(train, test)
        sequences = result.sequences
        assert sequences.levels == frozenset({"sequence"})
        assert set(sequences.with_radius(0).data()["level"].to_list()) <= {"sequence"}


@pytest.mark.required
class TestLongestRun:
    """A stare and a slow pan can score the same fraction; the longest run tells them apart."""

    def test_a_stare_reports_its_length(self):
        source = _fills(10, 10)
        stare = source[:4] + [source[4]] * 8 + source[5:]
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(make_tracking_dataset([stare]))
        # Positions 4..11 hold the same frame, so eight frames sit in one run.
        assert result.aggregate_by_sequence().row(0, named=True)["longest_run"] == 8

    def test_short_runs_stay_short(self):
        held = [fill for fill in _fills(10, 8) for _ in (0, 1)]
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(make_tracking_dataset([held]))
        row = result.aggregate_by_sequence().row(0, named=True)
        assert row["redundant_frames"] == 8
        assert row["longest_run"] == 2

    def test_a_sequence_with_no_redundancy_reports_zero(self):
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(make_tracking_dataset([_fills(10, 6)]))
        assert result.aggregate_by_sequence().row(0, named=True)["longest_run"] == 0


@pytest.mark.required
class TestDefaultRadiusWarning:
    """The image default is scheduled to change, and says so ahead of time."""

    def _images(self):
        images = [np.full((3, 16, 16), value, dtype=np.uint8) for value in _fills(10, 4)]
        return [*images, images[0]]

    def _futures(self, recorded) -> list[str]:
        return [str(item.message) for item in recorded if issubclass(item.category, FutureWarning)]

    def test_a_defaulted_radius_is_warned_about(self):
        with pytest.warns(FutureWarning, match="hash_radius will change"):
            Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(self._images())

    @pytest.mark.parametrize("radius", [0, 5])
    def test_a_chosen_radius_is_left_alone(self, radius, recwarn):
        """An explicit 0 goes on meaning 0 after the default moves, so nothing is coming for it."""
        Duplicates(flags=ImageStats.HASH_XXHASH, hash_radius=radius).evaluate(self._images())
        assert self._futures(recwarn) == []

    def test_a_radius_set_through_config_counts_as_chosen(self, recwarn):
        config = Duplicates.Config(flags=ImageStats.HASH_XXHASH, hash_radius=0)
        Duplicates(config=config).evaluate(self._images())
        assert self._futures(recwarn) == []

    def test_a_config_that_leaves_it_alone_is_still_warned_about(self):
        config = Duplicates.Config(flags=ImageStats.HASH_XXHASH)
        with pytest.warns(FutureWarning, match="hash_radius will change"):
            Duplicates(config=config).evaluate(self._images())

    def test_video_is_left_to_its_own_notice(self, recwarn):
        """For video the strict default does not miss relations, it measures them wrongly."""
        Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(make_tracking_dataset([_fills(10, 4)]))
        assert self._futures(recwarn) == []

    def test_the_warning_names_the_way_to_silence_it(self):
        with pytest.warns(FutureWarning) as recorded:
            Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(self._images())
        assert "Pass hash_radius explicitly" in str(recorded[0].message)
