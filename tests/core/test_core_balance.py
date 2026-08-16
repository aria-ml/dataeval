import logging

import numpy as np
import pytest

from dataeval.core._mutual_info import (
    _merge_labels_and_factors,
    _validate_num_neighbors,
    mutual_info,
)

CLASS_LABELS = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
FACTOR_DATA = np.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 2],
        [1, 0, 3],
        [2, 0, 4],
        [2, 1, 5],
        [3, 1, 6],
        [3, 1, 7],
        [4, 1, 8],
        [4, 1, 9],
    ],
)


@pytest.mark.required
class TestBalanceValidateNumNeighbors:
    @pytest.mark.parametrize(
        ("test_param", "expected_exception", "err_msg"),
        [
            (
                "7",
                pytest.raises(TypeError, match="not real-valued numeric type"),
                "Variable 7 is not real-valued numeric type.",
            ),
            (0, pytest.raises(ValueError, match="Invalid value for"), "Invalid value for 0."),
        ],
    )
    def test_validate_num_neighbors_type_errors(self, test_param, expected_exception, err_msg):
        with expected_exception as e:
            _validate_num_neighbors(test_param)
        assert err_msg in str(e.value)

    def test_validate_num_neighbors_warning(self, caplog):
        err_msg = "Variable 4 is currently type float and will be truncated to type int."
        with caplog.at_level(logging.WARNING):
            _validate_num_neighbors(4.0)  # type: ignore
        assert err_msg in caplog.text

    def test_validate_num_neighbors_pass(self):
        _validate_num_neighbors(10)
        pass


@pytest.mark.required
class TestBalanceMergeLabelsAndFactors:
    def test_without_discrete_features(self):
        data, discrete_features, declared = _merge_labels_and_factors(CLASS_LABELS, FACTOR_DATA, None)
        assert data.shape == (FACTOR_DATA.shape[0], FACTOR_DATA.shape[1] + 1)
        assert discrete_features == [True, True, True, False]
        # `is_continuous` calls all three factors discrete; only the sklearn-facing list
        # demotes the all-distinct third one.
        assert declared == [True, True, True, True]

    def test_provided_discrete_features(self):
        provided_discrete_features = [False, True, False]
        expected_discrete_features = [True] + provided_discrete_features

        data, discrete_features, declared = _merge_labels_and_factors(
            CLASS_LABELS, FACTOR_DATA, provided_discrete_features
        )
        assert data.shape == (FACTOR_DATA.shape[0], FACTOR_DATA.shape[1] + 1)
        assert discrete_features == expected_discrete_features
        assert declared == expected_discrete_features

    def test_provided_discrete_features_override_unique(self):
        provided_discrete_features = [False, True, True]
        expected_discrete_features = [True, False, True, False]

        data, discrete_features, declared = _merge_labels_and_factors(
            CLASS_LABELS, FACTOR_DATA, provided_discrete_features
        )
        assert data.shape == (FACTOR_DATA.shape[0], FACTOR_DATA.shape[1] + 1)
        # A column of distinct values is presented to sklearn as continuous, but the
        # caller's word is kept so the chance correction still applies to it.
        assert discrete_features == expected_discrete_features
        assert declared == [True, *provided_discrete_features]


@pytest.mark.required
class TestBalanceFunctional:
    def test_balance(self):
        """Test the balance function with TypedDict return."""
        result = mutual_info(CLASS_LABELS, FACTOR_DATA)

        # Test that result is a dict with the expected keys
        assert "class_to_factor" in result
        assert "interfactor" in result

        # Test factors array. Over ten samples none of these factors clears what its own
        # cardinality would produce by chance: factor 0 spreads five values across ten
        # rows, factor 2 holds a distinct value per row, and the observed mutual
        # information of each sits at or below the expectation under a random assignment
        # with the same margins. The class row is chance-corrected, so all three score 0.
        assert result["class_to_factor"].ndim == 1
        assert len(result["class_to_factor"]) == FACTOR_DATA.shape[1] + 1
        np.testing.assert_allclose(
            result["class_to_factor"],
            np.array([1.0, 0.0, 0.0, 0.0]),
            atol=1e-6,
        )

        # Test interfactor matrix
        assert result["interfactor"].ndim == 2
        assert result["interfactor"].shape == (FACTOR_DATA.shape[1], FACTOR_DATA.shape[1])
        np.testing.assert_allclose(result["interfactor"], result["interfactor"].T, atol=1e-6)
        # The factor-to-factor block is chance-corrected too, so factor 2 -- a distinct
        # value per row -- no longer reports an association with the other two. Its
        # diagonal entry is left at the estimator's own value, since a factor is not
        # scored against itself here.
        np.testing.assert_allclose(
            result["interfactor"],
            np.array([[1.0, 0.64, 0.0], [0.64, 1.0, 0.0], [0.0, 0.0, 0.621398]]),
            atol=1e-6,
        )

    def test_class_row_scores_the_share_of_class_entropy_explained(self):
        """The class row ranks factors by how much of the class label they account for."""
        rng = np.random.default_rng(0)
        n, k = 8000, 20
        labels = rng.integers(0, k, size=n)
        factors = np.column_stack([
            labels,  # determines the class outright
            labels // 4,  # a coarsening: implied by the class, but only part of it
            np.where(rng.random(n) < 0.5, labels, rng.integers(0, k, size=n)),  # half signal
            np.arange(n),  # a distinct value per row, generalizing to nothing
        ])
        row = mutual_info(labels, factors, discrete_features=[True] * 4)["class_to_factor"]

        assert row[1] == pytest.approx(1.0, abs=1e-3)
        # A coarsening carries exactly the share of the class entropy it retains, rather
        # than the 1.0 that normalizing by the factor's own entropy would report.
        assert row[2] == pytest.approx(np.log(5) / np.log(k), abs=0.01)
        assert 0.2 < row[3] < 0.5
        assert row[4] == pytest.approx(0.0, abs=0.01)
        assert row[1] > row[2] > row[3] > row[4]

    @pytest.mark.parametrize("ordering", ["shuffled", "class_grouped"])
    def test_identifier_scores_zero_however_the_rows_are_ordered(self, ordering):
        """A per-row identifier generalizes to nothing, whatever order it was assigned in.

        The row order is the whole test: a dataset collected or stored class by class --
        ImageFolder, one capture session per class -- numbers its identifiers in class
        order, so the column tracks the label perfectly while predicting nothing about an
        unseen sample. Declared discrete, it must reach the chance correction rather than
        the estimator, which reads the monotone column as near-perfect information.
        """
        rng = np.random.default_rng(0)
        n, k = 4000, 10
        labels = rng.integers(0, k, size=n)
        if ordering == "class_grouped":
            labels = np.sort(labels)
        identifier = np.arange(n)
        row = mutual_info(labels, identifier.reshape(-1, 1), discrete_features=[True])["class_to_factor"]
        assert row[1] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("cardinality", [5, 20, 184, 1000, 5000])
    def test_unrelated_factor_scores_zero_at_any_cardinality(self, cardinality):
        """Mutual information rises with cardinality by chance; the class row corrects for it."""
        rng = np.random.default_rng(0)
        n = 8000
        labels = rng.integers(0, 20, size=n)
        unrelated = rng.integers(0, cardinality, size=n)  # independent of labels
        row = mutual_info(labels, unrelated.reshape(-1, 1), discrete_features=[True])["class_to_factor"]
        assert row[1] < 0.01

    def test_constant_factor_zero_norm(self):
        """A constant factor has zero entropy, so its normalization factor is 0 and MI is 0.0."""
        # First factor is constant (single value) -> entropy 0 -> norm_factor 0.
        factor_data = np.column_stack([np.full(10, 7), np.arange(10) % 3])
        result = mutual_info(CLASS_LABELS, factor_data, discrete_features=[True, True])

        # The constant factor is index 0 in the interfactor matrix.
        assert result["interfactor"][0, 0] == 0.0
        # class-to-constant-factor MI is also driven to 0.0.
        assert result["class_to_factor"][1] == 0.0
