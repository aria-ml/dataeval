import numpy as np
import pytest

from dataeval.core._compute_stats import StatsResult, combine_stats_results
from dataeval.quality._shared import get_dataset_step_from_idx
from dataeval.types import SourceIndex


@pytest.mark.required
class TestResultsEdgeCases:
    def test_combine_results_empty_seq(self):
        """Covers TypeError when combining empty sequence."""
        with pytest.raises(TypeError, match="Cannot combine empty sequence"):
            combine_stats_results([])

    def test_get_dataset_step_oob(self):
        """Covers get_dataset_step_from_idx when index is out of bounds."""
        steps = [10, 20]  # Dataset 0 ends at 10, Dataset 1 ends at 20
        idx = 25  # Out of bounds

        d_idx, local_idx = get_dataset_step_from_idx(idx, steps)

        assert d_idx == -1
        assert local_idx == 25  # Should return original idx

    def test_combine_applies_offsets(self):
        """Verify item offsets are applied when combining multiple results."""
        r1: StatsResult = {
            "source_index": [SourceIndex(0), SourceIndex(1)],
            "stats": {"x": np.array([1.0, 2.0])},
            "object_count": [1, 1],
            "invalid_box_count": [0, 0],
            "image_count": 2,
        }
        r2: StatsResult = {
            "source_index": [SourceIndex(0), SourceIndex(1)],
            "stats": {"x": np.array([3.0, 4.0])},
            "object_count": [1, 1],
            "invalid_box_count": [0, 0],
            "image_count": 2,
        }
        stats, source_index, dataset_steps = combine_stats_results([r1, r2])
        assert [s.item for s in source_index] == [0, 1, 2, 3]
        assert dataset_steps == [2, 4]
        np.testing.assert_array_equal(stats["x"], [1.0, 2.0, 3.0, 4.0])

    def test_combine_single_result(self):
        """Single StatsResult returns stats directly with empty dataset_steps."""
        r: StatsResult = {
            "source_index": [SourceIndex(0), SourceIndex(1)],
            "stats": {"x": np.array([1.0, 2.0])},
            "object_count": [1, 1],
            "invalid_box_count": [0, 0],
            "image_count": 2,
        }
        stats, source_index, dataset_steps = combine_stats_results([r])
        assert [s.item for s in source_index] == [0, 1]
        assert dataset_steps == []
        np.testing.assert_array_equal(stats["x"], [1.0, 2.0])
