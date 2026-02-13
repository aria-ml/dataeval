import pytest

from dataeval.quality._results import combine_results, get_dataset_step_from_idx


@pytest.mark.required
class TestResultsEdgeCases:
    def test_combine_results_empty_seq(self):
        """Covers TypeError when combining empty sequence."""
        with pytest.raises(TypeError, match="Cannot combine empty sequence"):
            combine_results([])

    def test_get_dataset_step_oob(self):
        """Covers get_dataset_step_from_idx when index is out of bounds."""
        steps = [10, 20]  # Dataset 0 ends at 10, Dataset 1 ends at 20
        idx = 25  # Out of bounds

        d_idx, local_idx = get_dataset_step_from_idx(idx, steps)

        assert d_idx == -1
        assert local_idx == 25  # Should return original idx
