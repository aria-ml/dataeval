import numpy as np
import pytest

from dataeval.core._label_errors import _compute_label_scores, _suggest_labels, label_errors


class TestSuggestLabels:
    """Tests for the weighted voting logic."""

    @pytest.mark.parametrize(
        "nbr_labels, result",
        [
            (np.array([[2, 1, 2, 3, 2], [0, 3, 3, 3, 3]]), [[2], [3]]),
            (np.array([[1, 2, 1, 2, 1, 2], [0, 1, 0, 1, 3, 4]]), [[1, 2], [0, 1]]),
            (np.array([[0, 1, 2, 3, 4], [4, 0, 1, 3, 2]]), [[], []]),
        ],
    )
    def test_label_suggestion(self, nbr_labels, result):
        """Test that all variants are returned, single value, two values and empty list."""
        recs = _suggest_labels(nbr_labels, 5)

        assert recs[0] == result[0]
        assert recs[1] == result[1]

    def test_positional_weighting(self):
        """Test that combinations of distance and counts."""
        neighbor_labels = np.array([[1, 2, 1, 2, 2, 1, 2], [1, 2, 2, 1, 2, 1, 0]])
        num_classes = 3

        recs = _suggest_labels(neighbor_labels, num_classes)
        assert recs[0] == [2, 1]
        assert recs[1] == [1, 2]


class TestComputeLabelScores:
    """Tests for the distance ratio calculation."""

    def test_perfect_separation(self):
        """Test distinct clusters, score << 1."""
        data = np.vstack([np.zeros((5, 2)), np.ones((5, 2)) * 10])
        labels = np.array([0] * 5 + [1] * 5)

        scores, _ = _compute_label_scores(data, labels, k=3)

        # Scores should be very low (near 0)
        assert np.all(scores < 0.05)
        assert len(scores) == 10

    def test_obvious_mislabel(self):
        """Test a point physically located in the wrong cluster."""
        # Sample 0 is labeled '0', but located at (10,10) with Class 1.
        data = np.vstack([np.zeros((5, 2)), np.ones((5, 2)) * 10])
        data[0] = [9, 9]
        data += (np.arange(10) / 10)[:, None]
        labels = np.array([0] * 5 + [1] * 5)

        scores, potential_labels = _compute_label_scores(data, labels, k=3)
        assert scores[0] > 1.0
        assert scores[-1] < 1.0
        assert potential_labels[0].tolist() == [1, 1, 1]
        assert potential_labels[-1].tolist() == [0, 0, 0]

    def test_single_class_edge_case(self):
        """If dataset has only 1 class, we cannot calculate extra-class distance."""
        data = np.random.rand(5, 2)
        labels = np.zeros(5, dtype=int)

        scores, potential_labels = _compute_label_scores(data, labels, k=2)

        # Should return zeros and avoid crashing
        assert np.all(scores == 0.0)
        assert np.all(potential_labels == -1)


class TestLabelErrors:
    """Integration tests for the main entry point."""

    def test_integration_flow(self):
        """Verify the final dictionary structure and ranking."""
        data = np.vstack([np.zeros((5, 2)), np.ones((5, 2)) * 10])
        data[0] = [9, 9]
        data += (np.arange(10) / 10)[:, None]
        labels = np.array([0] * 5 + [1] * 5)

        result = label_errors(data, labels, k=3)

        assert isinstance(result, dict)
        assert "errors" in result
        assert "error_rank" in result
        assert "scores" in result

        assert len(result["errors"]) == 1
        assert 0 in result["errors"]

        original_lbl, suggested = result["errors"][0]
        assert original_lbl == 0
        assert len(suggested) == 1
        assert suggested[0] == 1

        assert result["error_rank"][0] == 0
        assert result["scores"][0] > 1

    def test_k_automatic_adjustment(self):
        """Verify k is reduced if it exceeds class count."""
        # Class 0 has only 3 samples. User asks for k=50.
        data = np.vstack([np.zeros((5, 2)), np.ones((5, 2)) * 10])
        labels = np.array([0] * 5 + [1] * 5)

        # Should execute without index out of bounds error
        result = label_errors(data, labels, k=50)

        assert len(result["errors"]) == 0
        assert len(result["scores"]) == 10
