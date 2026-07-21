"""Coverage.evaluate accepts embeddings + a raw label array, with no dataset/Metadata."""

import numpy as np
import pytest

from dataeval.scope._coverage import Coverage, CoverageOutput


@pytest.mark.required
class TestCoverageRawLabels:
    def test_embeddings_and_raw_labels_without_dataset(self):
        """The minimal call: pre-computed embeddings + a raw integer label array."""
        rng = np.random.default_rng(0)
        embeddings = rng.random((60, 8))
        # Three classes of 20 samples each: labels 0, 1, 2.
        class_labels = np.repeat([0, 1, 2], 20).astype(np.intp)

        result = Coverage(num_observations=5, min_class_samples=10).evaluate(class_labels, embeddings=embeddings)

        assert isinstance(result, CoverageOutput)
        # With no index2label, classes name themselves by their integer index.
        assert set(result.data()["class"].to_list()) == {"0", "1", "2"}
        assert result.data()["count"].to_list() == [20, 20, 20] or sorted(result.data()["count"].to_list()) == [
            20,
            20,
            20,
        ]
        assert result.coverage_radius > 0
        assert len(result.critical_value_radii) == len(embeddings)
        assert all(0 <= i < len(embeddings) for i in result.uncovered_indices)

    def test_raw_labels_accept_plain_list(self):
        """A plain Python list of ints works as the raw label form too."""
        rng = np.random.default_rng(1)
        embeddings = rng.random((40, 6))
        class_labels = [0] * 20 + [1] * 20

        result = Coverage(num_observations=5, min_class_samples=10).evaluate(class_labels, embeddings=embeddings)

        assert isinstance(result, CoverageOutput)
        assert set(result.data()["class"].to_list()) == {"0", "1"}

    def test_raw_labels_require_embeddings(self):
        """Raw labels are only accepted alongside pre-computed embeddings."""
        class_labels = np.repeat([0, 1], 10).astype(np.intp)
        with pytest.raises(ValueError, match="embeddings"):
            Coverage(num_observations=5, min_class_samples=5).evaluate(class_labels)

    def test_raw_labels_shape_mismatch_raises(self):
        """A label array that does not match the embeddings count is rejected."""
        from dataeval.exceptions import ShapeMismatchError

        rng = np.random.default_rng(2)
        embeddings = rng.random((60, 8))
        class_labels = np.repeat([0, 1, 2], 5).astype(np.intp)  # only 15 labels
        with pytest.raises(ShapeMismatchError, match="one embedding per"):
            Coverage(num_observations=5, min_class_samples=5).evaluate(class_labels, embeddings=embeddings)
