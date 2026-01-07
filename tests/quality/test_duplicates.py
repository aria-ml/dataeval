import numpy as np
import pytest

from dataeval.core._calculate import calculate
from dataeval.core._clusterer import ClusterResult
from dataeval.flags import ImageStats
from dataeval.quality import Duplicates


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
        assert len(results.exact) == 20
        assert len(results.near) == 0

    def test_near_duplicates(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates()
        results = dupes.evaluate(np.concatenate((data, data + 0.001)))
        assert len(results.exact) < 20
        assert len(results.near) > 0

    def test_duplicates_only_exact(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates(only_exact=True)
        results = dupes.evaluate(np.concatenate((data, data, data + 0.001)))
        assert len(results.exact) == 20
        assert len(results.near) == 0

    def test_duplicates_with_stats(self):
        data = np.random.random((20, 3, 16, 16))
        data = np.concatenate((data, data, data + 0.001))
        stats = calculate(data, None, ImageStats.HASH)
        dupes = Duplicates(only_exact=True)
        results = dupes.from_stats(stats)
        assert len(results.exact) == 20
        assert len(results.near) == 0

    def test_get_duplicates_multiple_stats(self):
        ones = np.ones((1, 16, 16))
        zeros = np.zeros((1, 16, 16))
        data1 = np.concatenate((ones, zeros, ones, zeros, ones))
        data2 = np.concatenate((zeros, ones, zeros))
        data3 = np.concatenate((zeros + 0.001, ones - 0.001))
        dupes1 = calculate(data1, None, ImageStats.HASH)
        dupes2 = calculate(data2, None, ImageStats.HASH)
        dupes3 = calculate(data3, None, ImageStats.HASH)

        dupes = Duplicates()
        results = dupes.from_stats((dupes1, dupes2, dupes3))
        assert len(results.exact) == 2
        assert results.exact[0] == {0: [0, 2, 4], 1: [1]}
        assert len(results.near) == 2
        assert results.near[0] == {0: [0, 2, 4], 1: [1], 2: [1]}

    def test_duplicates_invalid_stats(self):
        dupes = Duplicates()
        with pytest.raises(TypeError):
            dupes.from_stats(1234)  # type: ignore
        with pytest.raises(TypeError):
            dupes.from_stats([1234])  # type: ignore

    def test_duplicates_ignore_non_duplicate_too_small(self):
        dupes = Duplicates()
        images = [np.random.random((3, 16, 16)) for _ in range(20)]
        images[3] = np.zeros((3, 5, 5))
        images[5] = np.ones((3, 5, 5))
        results = dupes.evaluate(images)
        assert len(results.near) == 0

    def test_duplicates_ignore_duplicate_too_small(self):
        dupes = Duplicates()
        images = [np.random.random((3, 16, 16)) for _ in range(20)]
        images[3] = np.zeros((3, 5, 5))
        images[5] = np.zeros((3, 5, 5))
        results = dupes.evaluate(images)
        assert len(results.near) == 0

    def test_duplicates_dataset(self):
        dupes = Duplicates()
        results = dupes.evaluate(MockDataset())
        assert results is not None

    def test_duplicates_from_clusters_basic(self):
        """Test basic cluster-based duplicate detection"""

        # Create ClusterResult with MST and cluster assignments
        # Create a simple MST structure (edges with distances)
        # Format: [node1, node2, distance]
        mock_cluster_result: ClusterResult = {
            "mst": np.array(
                [[0, 1, 0.1], [1, 2, 0.05], [2, 3, 0.0], [3, 4, 0.2]],
                dtype=np.float32,  # Exact duplicate
            ),
            "clusters": np.array([0, 0, 0, 0, 0], dtype=np.intp),
            "linkage_tree": np.array([], dtype=np.float32),
            "condensed_tree": {
                "parent": np.array([]),
                "child": np.array([]),
                "lambda_val": np.array([]),
                "child_size": np.array([]),
            },
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int64),
            "k_distances": np.array([], dtype=np.float32),
        }

        # Find duplicates using new method
        detector = Duplicates()
        result = detector.from_clusters(mock_cluster_result)

        # Should return proper structure
        assert isinstance(result.exact, list)
        assert isinstance(result.near, list)

    def test_duplicates_from_clusters_only_exact(self):
        """Test cluster-based detection with only_exact=True"""

        # Create ClusterResult
        mock_cluster_result: ClusterResult = {
            "mst": np.array([[0, 1, 0.0], [1, 2, 0.05]], dtype=np.float32),
            "clusters": np.array([0, 0, 0], dtype=np.intp),
            "linkage_tree": np.array([], dtype=np.float32),
            "condensed_tree": {
                "parent": np.array([]),
                "child": np.array([]),
                "lambda_val": np.array([]),
                "child_size": np.array([]),
            },
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int64),
            "k_distances": np.array([], dtype=np.float32),
        }

        # Only exact duplicates
        detector = Duplicates(only_exact=True)
        result = detector.from_clusters(mock_cluster_result)

        # Should find exact duplicates
        assert isinstance(result.exact, list)
        # Near duplicates should be empty
        assert isinstance(result.near, list)
        assert len(result.near) == 0

    def test_duplicates_from_clusters_no_duplicates(self):
        """Test with data that has no duplicates"""

        # Create ClusterResult with no zero-distance edges
        mock_cluster_result: ClusterResult = {
            "mst": np.array([[0, 1, 0.5], [1, 2, 0.3], [2, 3, 0.4]], dtype=np.float32),
            "clusters": np.array([0, 0, 0, 0], dtype=np.intp),
            "linkage_tree": np.array([], dtype=np.float32),
            "condensed_tree": {
                "parent": np.array([]),
                "child": np.array([]),
                "lambda_val": np.array([]),
                "child_size": np.array([]),
            },
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int64),
            "k_distances": np.array([], dtype=np.float32),
        }

        detector = Duplicates()
        result = detector.from_clusters(mock_cluster_result)

        # Should have proper structure (may be empty)
        assert isinstance(result.exact, list)
        assert isinstance(result.near, list)
