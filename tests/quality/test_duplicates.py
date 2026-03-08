import numpy as np
import polars as pl
import pytest

from dataeval.core._clusterer import ClusterResult
from dataeval.core._compute_stats import compute_stats
from dataeval.extractors import FlattenExtractor
from dataeval.flags import ImageStats
from dataeval.quality import Duplicates, DuplicatesOutput
from dataeval.quality._duplicates import (
    SourceIndex,
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

        # Check items structure - multi-dataset has dataset_index column
        assert "dataset_index" in results.data().columns

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
        result = compute_stats(dataset, stats=ImageStats.HASH, per_image=True, per_target=True, per_channel=False)

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

        # Multi-dataset should have dataset_index column
        assert "dataset_index" in result.data().columns

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
        assert "dataset_index" in result.data().columns

        exact_items = _get_exact_groups(result, "item")
        assert exact_items.shape[0] >= 5  # at least 5 cross-dataset exact groups

        # dataset_index lists should be present and align with item_indices
        for row in exact_items.iter_rows(named=True):
            assert len(row["dataset_index"]) == len(row["item_indices"])

    def test_evaluate_multi_dataset_near(self):
        """Evaluate with two datasets sharing near duplicates."""
        data1 = np.random.random((10, 3, 16, 16))
        data2 = data1 + 0.001  # near-duplicates

        dupes = Duplicates()
        result = dupes.evaluate(data1, data2)

        assert "dataset_index" in result.data().columns
        near_items = _get_near_groups(result, "item")
        assert near_items.shape[0] > 0

    def test_evaluate_multi_dataset_three_datasets(self):
        """Evaluate with three datasets."""
        ones = np.ones((3, 1, 16, 16))
        zeros = np.zeros((3, 1, 16, 16))
        mixed = np.concatenate((ones[:1], zeros[:1]))

        dupes = Duplicates()
        result = dupes.evaluate(ones, zeros, mixed)

        assert "dataset_index" in result.data().columns
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
        # Multi-dataset exact returns dict keyed by dataset_index
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
        # Multi-dataset near returns dict keyed by dataset_index
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

        assert "dataset_index" in result.data().columns
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

        assert "dataset_index" in result.data().columns
        # Re-detect with tighter threshold
        strict = result.with_sensitivity(0.5)
        assert isinstance(strict, DuplicatesOutput)
        assert "dataset_index" in strict.data().columns


@pytest.mark.required
class TestDuplicatesEdgeCases:
    def test_evaluate_invalid_config(self):
        """Covers ValueError when flags=NONE and no cluster-based detection configured."""
        detector = Duplicates(flags=ImageStats.NONE, extractor=None)
        data = np.zeros((1, 10, 10, 3))
        with pytest.raises(ValueError, match="Either flags must contain hash stats"):
            detector.evaluate(data)

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
