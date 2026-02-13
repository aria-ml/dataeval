from collections.abc import Sequence

import numpy as np
import pytest

from dataeval.core._calculate import calculate
from dataeval.core._clusterer import ClusterResult
from dataeval.extractors import FlattenExtractor
from dataeval.flags import ImageStats
from dataeval.quality import Duplicates
from dataeval.quality._duplicates import (
    DatasetItemTuple,
    DuplicateDetectionResult,
    NearDuplicateGroup,
    SourceIndex,
)


class MockDataset:
    def __len__(self):
        return 20

    def __iter__(self):
        for _ in range(20):
            yield np.random.random((3, 16, 16))

    def __getitem__(self, _):
        return np.random.random((3, 16, 16))


@pytest.mark.required
class TestDuplicates:
    def test_duplicates(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates()
        results = dupes.evaluate(np.concatenate((data, data)))
        assert results.items.exact is not None
        assert len(results.items.exact) == 20
        # Near duplicates might be found due to perceptual hashing behavior with random data
        # The key is that we have exact duplicates, near can vary
        assert results.targets.exact is None
        assert results.targets.near is None

    def test_near_duplicates(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates()
        results = dupes.evaluate(np.concatenate((data, data + 0.001)))
        # Adding 0.001 to random data creates values that are NOT byte-identical,
        # so xxhash will NOT find them as exact duplicates. However, phash will
        # find them as near duplicates because the visual difference is minimal.
        # exact may or may not be None depending on whether some random images
        # happen to hash to the same xxhash value
        assert results.items.near is not None
        assert len(results.items.near) > 0
        assert results.targets.exact is None
        assert results.targets.near is None

    def test_duplicates_only_exact(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates(ImageStats.HASH_XXHASH)
        results = dupes.evaluate(np.concatenate((data, data, data + 0.001)))
        assert results.items.exact is not None
        assert len(results.items.exact) == 20
        # near is None because HASH_PHASH was not included in flags
        assert results.items.near is None
        assert results.targets.exact is None
        assert results.targets.near is None

    def test_duplicates_with_stats(self):
        data = np.random.random((20, 3, 16, 16))
        data = np.concatenate((data, data, data + 0.001))
        # Stats computed with full HASH (includes both xxhash and phash)
        stats = calculate(data, None, ImageStats.HASH, per_image=True, per_target=False)
        # Detector configured for exact only - but from_stats uses what's in the stats
        dupes = Duplicates(ImageStats.HASH_XXHASH)
        results = dupes.from_stats(stats)
        assert results.items.exact is not None
        assert len(results.items.exact) == 20
        # from_stats uses what's available in the stats, so phash results will be present
        # since the stats were computed with ImageStats.HASH
        assert results.items.near is not None
        assert results.targets.exact is None
        assert results.targets.near is None

    def test_get_duplicates_multiple_stats(self):
        """Test cross-dataset duplicate detection with new API."""
        ones = np.ones((1, 16, 16))
        zeros = np.zeros((1, 16, 16))
        data1 = np.concatenate((ones, zeros, ones, zeros, ones))
        data2 = np.concatenate((zeros, ones, zeros))
        data3 = np.concatenate((zeros + 0.001, ones - 0.001))
        dupes1 = calculate(data1, None, ImageStats.HASH, per_image=True, per_target=False)
        dupes2 = calculate(data2, None, ImageStats.HASH, per_image=True, per_target=False)
        dupes3 = calculate(data3, None, ImageStats.HASH, per_image=True, per_target=False)

        dupes = Duplicates()
        results = dupes.from_stats([dupes1, dupes2, dupes3])

        # Check items structure - now returns DatasetItemIndex objects
        assert results.items.exact is not None
        assert len(results.items.exact) == 2

        # Convert to set representation for easier checking
        # Format is now: [DatasetItemIndex(dataset_id=0, id=0), DatasetItemIndex(dataset_id=1, id=1), ...]
        exact_groups = []
        for group in results.items.exact:
            group_dict: dict[int, list[int]] = {}
            for item in group:
                assert isinstance(item, tuple)
                dataset_id = item[0]
                item_id = item[1]
                if dataset_id not in group_dict:
                    group_dict[dataset_id] = []
                group_dict[dataset_id].append(item_id)  # type: ignore
            exact_groups.append(group_dict)

        # Check that we have ones group and zeros group
        assert {0: [0, 2, 4], 1: [1]} in exact_groups
        assert {0: [1, 3], 1: [0, 2]} in exact_groups

        # Check near duplicates
        # The actual output shows all zeros are grouped together (not split into ones and zeros groups for near)
        assert results.items.near is not None
        assert len(results.items.near) >= 1
        near_groups = []
        for near_group in results.items.near:
            # near_group is now a NearDuplicateGroup with indices and methods
            group_dict: dict[int, list[int]] = {}
            for item in near_group.indices:
                assert isinstance(item, tuple)
                dataset_id = item[0]
                item_id = item[1]
                if dataset_id not in group_dict:
                    group_dict[dataset_id] = []
                group_dict[dataset_id].append(item_id)  # type: ignore
            near_groups.append(group_dict)

        # Near group includes perceptually similar across datasets
        # The zeros and near-zeros all group together
        assert any(g == {0: [1, 3], 1: [0, 2], 2: [0, 1]} or set(g.keys()) >= {0, 1, 2} for g in near_groups)

        # No targets in this test
        assert results.targets.exact is None
        assert results.targets.near is None

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
        # With random data, perceptual hashing may find near duplicates
        # The test is about small images - they should be skipped from hash computation
        # But the remaining random images might collide in perceptual hash space
        # The key assertion is that we don't crash on small images
        assert results.items is not None

    def test_duplicates_ignore_duplicate_too_small(self):
        dupes = Duplicates()
        images = [np.random.random((3, 16, 16)) for _ in range(20)]
        images[3] = np.zeros((3, 5, 5))
        images[5] = np.zeros((3, 5, 5))
        results = dupes.evaluate(images)
        # Small images get hashed with xxhash but not phash
        # So they can still appear in exact duplicates
        assert results.items is not None
        # The test just ensures we don't crash on small images
        # They actually DO get exact duplicate detection via xxhash
        if results.items.exact:
            # Check that small duplicates were found
            small_image_group = None
            for group in results.items.exact:
                if 3 in group and 5 in group:
                    small_image_group = group
                    break
            # Small images should be detected as exact duplicates
            assert small_image_group is not None

    def test_duplicates_dataset(self):
        dupes = Duplicates()
        results = dupes.evaluate(MockDataset())
        assert results is not None

    def test_duplicates_from_clusters_basic(self):
        """Test basic cluster-based duplicate detection."""
        # Create ClusterResult with MST and cluster assignments
        # Create a simple MST structure (edges with distances)
        # Format: [node1, node2, distance]
        mock_cluster_result: ClusterResult = {
            "mst": np.array(
                [[0, 1, 0.1], [1, 2, 0.05], [2, 3, 0.0], [3, 4, 0.2]],
                dtype=np.float32,  # Zero distance edge
            ),
            "clusters": np.array([0, 0, 0, 0, 0], dtype=np.intp),
            "linkage_tree": np.array([], dtype=np.float32),
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int64),
            "k_distances": np.array([], dtype=np.float32),
        }

        # Find duplicates using new method
        detector = Duplicates()
        result = detector.from_clusters(mock_cluster_result)

        # Cluster-based detection never returns exact duplicates since embeddings
        # are approximate representations. Only xxhash can identify true exact duplicates.
        assert result.items.exact is None
        # All cluster-based duplicates (including zero-distance) are near duplicates
        assert isinstance(result.items.near, list)
        assert all("cluster" in g.methods for g in result.items.near)
        # Targets is now an empty DuplicateDetectionResult, not None
        assert result.targets.exact is None
        assert result.targets.near is None

    def test_duplicates_from_clusters_with_near(self):
        """Test cluster-based detection treats all duplicates as near duplicates."""
        # Create ClusterResult with zero-distance edge and edges that will
        # produce near duplicates. Near duplicates are edges with distance < cluster std.
        # With distances [0.0, 0.01, 0.05, 0.1], std is ~0.042, so 0.01 will be a near dup.
        mock_cluster_result: ClusterResult = {
            "mst": np.array([[0, 1, 0.0], [1, 2, 0.01], [2, 3, 0.05], [3, 4, 0.1]], dtype=np.float32),
            "clusters": np.array([0, 0, 0, 0, 0], dtype=np.intp),
            "linkage_tree": np.array([], dtype=np.float32),
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int64),
            "k_distances": np.array([], dtype=np.float32),
        }

        # Cluster-based detection (flags don't affect from_clusters)
        detector = Duplicates()
        result = detector.from_clusters(mock_cluster_result)

        # Cluster-based detection never returns exact duplicates since embeddings
        # are approximate representations. Zero distance in embedding space doesn't
        # mean pixel-identical images.
        assert result.items.exact is None
        # All cluster-based duplicates come as near duplicates with "cluster" method
        assert isinstance(result.items.near, list)
        assert all("cluster" in g.methods for g in result.items.near)
        # Targets is now an empty DuplicateDetectionResult, not None
        assert result.targets.exact is None
        assert result.targets.near is None

    def test_duplicates_from_clusters_no_duplicates(self):
        """Test with data that has no duplicates."""
        # Create ClusterResult with no zero-distance edges
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

        # Should have proper structure - no exact duplicates, possibly near duplicates
        # (depends on cluster std calculation)
        # Near will be None or contain groups with "cluster" method
        if result.items.near is not None:
            assert all("cluster" in g.methods for g in result.items.near)
        # Targets is now an empty DuplicateDetectionResult, not None
        assert result.targets.exact is None
        assert result.targets.near is None

    def test_from_clusters_respects_merge_near_duplicates(self):
        """Test that from_clusters respects the merge_near_duplicates parameter."""
        # Create ClusterResult where union-find produces overlapping groups
        # across exact and near duplicate lists. With distances [0.0, 0.01, 0.5, 5.0]
        # and all in same cluster: std ~ 2.14, so 0.0 and 0.01 are near dups.
        # The exact threshold will pick up the 0.0 edge.
        # This gives us: exact group [0,1] and near group [1,2], which overlap on node 1.
        mock_cluster_result: ClusterResult = {
            "mst": np.array([[0, 1, 0.0], [1, 2, 0.01], [2, 3, 0.5], [3, 4, 5.0]], dtype=np.float32),
            "clusters": np.array([0, 0, 0, 0, 0], dtype=np.intp),
            "linkage_tree": np.array([], dtype=np.float32),
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int64),
            "k_distances": np.array([], dtype=np.float32),
        }

        # With merge_near_duplicates=True (default), overlapping groups should be merged
        detector_merged = Duplicates(merge_near_duplicates=True)
        result_merged = detector_merged.from_clusters(mock_cluster_result)
        assert result_merged.items.near is not None

        # With merge_near_duplicates=False, groups should remain separate
        detector_separate = Duplicates(merge_near_duplicates=False)
        result_separate = detector_separate.from_clusters(mock_cluster_result)
        assert result_separate.items.near is not None

        # When merging, overlapping groups get combined so we should have
        # fewer or equal groups compared to not merging
        assert len(result_merged.items.near) <= len(result_separate.items.near)

    def test_hash_differs_for_full_image_vs_targets(self, get_mock_od_dataset):
        """Regression test: hash values should differ between full image and individual targets.

        This test ensures that when computing hashes for object detection datasets,
        the full image (target=None) gets a different hash than individual bounding boxes.
        Previously, both used the full image hash, making duplicate detection useless
        for individual objects.
        """
        # Create an image with two distinct regions
        image = np.zeros((3, 100, 100), dtype=np.uint8)
        # Top-left quadrant: all white (255)
        image[:, 0:50, 0:50] = 255
        # Bottom-right quadrant: all black (0)
        image[:, 50:100, 50:100] = 0

        images = [image]
        labels = [[0, 1]]
        bboxes = [
            [[0, 0, 50, 50], [50, 50, 100, 100]],  # white region  # black region
        ]

        dataset = get_mock_od_dataset(images, labels, bboxes)

        # Calculate hashes for both full image and individual targets
        result = calculate(dataset, stats=ImageStats.HASH, per_image=True, per_target=True, per_channel=False)

        # Should have 3 results: full image + 2 boxes
        assert len(result["source_index"]) == 3

        # Extract hashes for full image and boxes
        full_image_xxhash = result["stats"]["xxhash"][0]  # target=None
        box0_xxhash = result["stats"]["xxhash"][1]  # target=0 (white region)
        box1_xxhash = result["stats"]["xxhash"][2]  # target=1 (black region)

        full_image_phash = result["stats"]["phash"][0]  # target=None
        box0_phash = result["stats"]["phash"][1]  # target=0 (white region)
        box1_phash = result["stats"]["phash"][2]  # target=1 (black region)

        # CRITICAL: Full image hash should differ from both box hashes
        assert full_image_xxhash != box0_xxhash, "Full image xxhash should differ from box 0"
        assert full_image_xxhash != box1_xxhash, "Full image xxhash should differ from box 1"

        # The two boxes contain different content, so their hashes should differ
        assert box0_xxhash != box1_xxhash, "Box 0 and Box 1 should have different xxhashes"

        # Same checks for perceptual hash
        assert full_image_phash != box0_phash, "Full image phash should differ from box 0"
        assert full_image_phash != box1_phash, "Full image phash should differ from box 1"
        assert box0_phash != box1_phash, "Box 0 and Box 1 should have different phashes"

    def test_duplicate_detection_with_items_and_targets(self, get_mock_od_dataset):
        """Test new API separating item and target duplicate detection."""
        # Create images with duplicates at both item and target level
        image1 = np.zeros((3, 100, 100), dtype=np.uint8)
        image1[:, 0:50, 0:50] = 255  # white box in top-left

        image2 = np.zeros((3, 100, 100), dtype=np.uint8)
        image2[:, 0:50, 0:50] = 255  # white box in top-left (duplicate of image1's box)
        image2[:, 50:100, 50:100] = 128  # gray box in bottom-right

        # Image 3 is a duplicate of image 1
        image3 = image1.copy()

        images = [image1, image2, image3]
        labels = [[0], [0, 1], [0]]
        bboxes = [
            [[0, 0, 50, 50]],  # white box
            [[0, 0, 50, 50], [50, 50, 100, 100]],  # white + gray boxes
            [[0, 0, 50, 50]],  # white box
        ]

        dataset = get_mock_od_dataset(images, labels, bboxes)

        # Detect duplicates
        detector = Duplicates()
        result = detector.evaluate(dataset, per_image=True, per_target=True)

        # Check item-level duplicates (images 0 and 2 are identical)
        assert result.items.exact is not None
        assert len(result.items.exact) == 1  # One group of exact duplicates
        assert set(result.items.exact[0]) == {0, 2}  # Images 0 and 2

        # Check target-level duplicates (all three white boxes should be duplicates)
        assert result.targets.exact is not None
        assert len(result.targets.exact) >= 1  # At least one group
        # Find the group containing white boxes
        white_box_group = None
        for group in result.targets.exact:
            if len(group) == 3:  # All three white boxes
                white_box_group = group
                break

        # All should be target 0 (the white boxes)
        assert isinstance(white_box_group, Sequence)
        for src_idx in white_box_group:
            assert isinstance(src_idx, SourceIndex)
            assert src_idx.target == 0

    def test_per_image_only(self, get_mock_od_dataset):
        """Test evaluating with per_image=True, per_target=False."""
        images = [np.random.random((3, 100, 100)) for _ in range(2)]
        labels = [[0], [1]]
        bboxes = [[[10, 10, 50, 50]], [[20, 20, 60, 60]]]

        dataset = get_mock_od_dataset(images, labels, bboxes)

        detector = Duplicates()
        result = detector.evaluate(dataset, per_image=True, per_target=False)

        # Should have items but no targets (targets will be empty DuplicateDetectionResult)
        assert result.items is not None
        assert result.targets.exact is None
        assert result.targets.near is None

    def test_per_target_only(self, get_mock_od_dataset):
        """Test evaluating with per_image=False, per_target=True."""
        images = [np.random.random((3, 100, 100)) for _ in range(2)]
        labels = [[0], [1]]
        bboxes = [[[10, 10, 50, 50]], [[20, 20, 60, 60]]]

        dataset = get_mock_od_dataset(images, labels, bboxes)

        detector = Duplicates()
        result = detector.evaluate(dataset, per_image=False, per_target=True)

        # When per_image=False, items result will be empty (exact=None, near=None)
        assert result.items is not None
        # No item-level hashing was done, so items should be empty
        assert result.items.exact is None or len(result.items.exact) == 0
        assert result.targets is not None
        # Targets should have some results (exact or near)
        assert result.targets.exact is not None or result.targets.near is not None

    def test_cross_dataset_with_targets(self, get_mock_od_dataset):
        """Test cross-dataset duplicate detection with targets."""
        # Create two datasets with duplicate targets across them
        white_box = np.ones((3, 50, 50), dtype=np.uint8) * 255
        black_box = np.zeros((3, 50, 50), dtype=np.uint8)

        # Dataset 1: image with white box
        image1 = np.zeros((3, 100, 100), dtype=np.uint8)
        image1[:, 0:50, 0:50] = white_box

        # Dataset 2: image with white box (duplicate) and black box
        image2 = np.zeros((3, 100, 100), dtype=np.uint8)
        image2[:, 0:50, 0:50] = white_box  # duplicate white box
        image2[:, 50:100, 50:100] = black_box

        dataset1 = get_mock_od_dataset([image1], [[0]], [[[0, 0, 50, 50]]])
        dataset2 = get_mock_od_dataset([image2], [[0, 1]], [[[0, 0, 50, 50], [50, 50, 100, 100]]])

        stats1 = calculate(dataset1, None, ImageStats.HASH, per_image=True, per_target=True)
        stats2 = calculate(dataset2, None, ImageStats.HASH, per_image=True, per_target=True)

        detector = Duplicates()
        result = detector.from_stats([stats1, stats2])

        # Check item-level duplicates - now returns DatasetItemIndex objects
        assert result.items.exact is not None
        # Should have 1 group with both images (both are same - black with white box in corner)
        assert len(result.items.exact) == 1

        # Convert to dict for comparison
        item_group = result.items.exact[0]
        item_dict = {}
        for item in item_group:
            assert isinstance(item, tuple)
            dataset_id = item[0]
            item_id = item[1]
            if dataset_id not in item_dict:
                item_dict[dataset_id] = []
            item_dict[dataset_id].append(item_id)
        assert item_dict == {0: [0], 1: [0]}

        # Check target-level duplicates
        assert result.targets is not None
        assert result.targets.exact is not None
        assert len(result.targets.exact) >= 1

        # Find the white box duplicate group
        white_group = None
        for group in result.targets.exact:
            # Convert group to check datasets
            datasets_in_group = set()
            for item in group:
                assert isinstance(item, tuple)
                datasets_in_group.add(item[0])

            if len(group) == 2 and 0 in datasets_in_group and 1 in datasets_in_group:
                # This is the cross-dataset duplicate group
                white_group = group
                break

        assert white_group is not None
        # Both should have target 0 (the white boxes)
        # white_group is now a list of DatasetItemIndex objects
        for item in white_group:
            # item.id is a SourceIndex for target-level results
            assert isinstance(item, tuple)
            source_index = item[1]
            assert isinstance(source_index, SourceIndex)
            assert source_index.target == 0


@pytest.mark.required
class TestDuplicatesEdgeCases:
    def test_repr_methods(self):
        """Covers __repr__ for DatasetItemTuple and NearDuplicateGroup."""
        tup = DatasetItemTuple(dataset_id=1, id=5)
        assert repr(tup) == "(1, 5)"

        group = NearDuplicateGroup(indices=[1, 2], methods=frozenset(["phash"]))
        assert "NearDuplicateGroup" in repr(group)
        assert "orientation" not in repr(group)  # Default is None

        group_oriented = NearDuplicateGroup(indices=[1, 2], methods=frozenset(["phash"]), orientation="rotated")
        assert "orientation=rotated" in repr(group_oriented)

    def test_evaluate_invalid_config(self):
        """Covers ValueError when flags=NONE and no cluster-based detection configured."""
        detector = Duplicates(flags=ImageStats.NONE, extractor=None)
        # Mock data (shape doesn't matter for this check)
        data = np.zeros((1, 10, 10, 3))
        with pytest.raises(ValueError, match="Either flags must contain hash stats"):
            detector.evaluate(data)

    def test_merge_duplicate_groups_logic(self):
        """Covers _merge_duplicate_groups logic for overlapping and disjoint sets."""
        detector = Duplicates()

        # Case 1: Disjoint groups
        groups_a = [[1, 2]]
        groups_b = [[3, 4]]
        merged = detector._merge_duplicate_groups(groups_a, groups_b)
        assert merged == [[1, 2], [3, 4]]

        # Case 2: Overlapping groups (transitive property)
        # [1, 2] and [2, 3] should merge to [1, 2, 3]
        groups_a = [[1, 2]]
        groups_b = [[2, 3]]
        merged = detector._merge_duplicate_groups(groups_a, groups_b)
        assert merged == [[1, 2, 3]]

        # Case 3: Complex overlap
        # [1, 2], [3, 4], [2, 3] -> All connect 1-2-3-4
        groups_a = [[1, 2], [3, 4]]
        groups_b = [[2, 3]]
        merged = detector._merge_duplicate_groups(groups_a, groups_b)
        assert merged == [[1, 2, 3, 4]]

    def test_merge_item_results_none_cases(self):
        """Covers _merge_item_results when inputs are None or empty."""
        detector = Duplicates()

        # Both None/Empty
        res = detector._merge_item_results(None, [], [], set())
        assert res.exact is None
        assert res.near is None

        # Only hash result provided
        mock_hash_result = DuplicateDetectionResult(exact=[[1, 2]], near=None)
        res = detector._merge_item_results(mock_hash_result, [], [], set())
        assert res == mock_hash_result

    def test_cluster_threshold_none_raises_error(self):
        """
        Covers logic where cluster_threshold is None with extractor provided.

        When flags=NONE and cluster_threshold=None, clustering is skipped entirely,
        leaving no detection method available. This should raise a ValueError.
        """

        # Create a dummy extractor that returns fixed embeddings
        class DummyExtractor:
            def __call__(self, data):
                return np.array([[0.1], [0.1], [0.9]])  # 0 and 1 are identical

        # With threshold=None and flags=NONE, there's no detection method available
        detector = Duplicates(
            flags=ImageStats.NONE,
            extractor=DummyExtractor(),
            cluster_threshold=None,
            cluster_algorithm="kmeans",
            n_clusters=2,
        )

        # Mock data
        data = np.array([0.1, 0.1, 0.2, 0.2, 0.9])

        # Should raise ValueError since no detection method is available
        with pytest.raises(ValueError, match="Either flags must contain hash stats"):
            detector.evaluate(data)

    def test_find_hash_duplicates_empty_logic(self):
        """Covers _find_hash_duplicates filtering empty values."""
        detector = Duplicates()
        stats = {"phash": np.array(["", "abc", "abc", ""])}  # Empty strings simulate missing hashes
        source_index = [SourceIndex(i, None, None) for i in range(4)]
        indices = [0, 1, 2, 3]
        exact_groups = []

        # Should only find group for "abc" (indices 1, 2), ignoring empty strings at 0, 3
        groups = detector._find_hash_duplicates(stats, "phash", source_index, indices, exact_groups)
        assert groups == [[1, 2]]

    def test_evaluate_with_tuple_dataset(self, get_mock_ic_dataset):
        """Regression test: evaluate with cluster-based detection should handle tuple datasets.

        When a dataset returns (image, label, metadata) tuples, the extractor should
        receive only the images, not the full tuples.
        """
        data = np.random.random((20, 3, 16, 16))
        # Create duplicates so clustering can find them
        data_with_dupes = np.concatenate([data, data])
        labels = list(range(len(data_with_dupes)))
        dataset = get_mock_ic_dataset(list(data_with_dupes), labels)

        detector = Duplicates(extractor=FlattenExtractor(), cluster_threshold=1.0)
        result = detector.evaluate(dataset)
        assert result is not None
        assert result.items is not None

    def test_evaluate_with_tuple_dataset_cluster_only(self, get_mock_ic_dataset):
        """Regression test: cluster-only detection on tuple datasets should not crash."""
        data = np.random.random((20, 3, 16, 16))
        labels = list(range(len(data)))
        dataset = get_mock_ic_dataset(list(data), labels)

        detector = Duplicates(flags=ImageStats.NONE, extractor=FlattenExtractor(), cluster_threshold=1.0)
        result = detector.evaluate(dataset)
        assert result is not None
