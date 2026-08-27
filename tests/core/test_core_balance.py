import logging

import numpy as np
import pytest

from dataeval.core._mutual_info import (
    _merge_labels_and_factors,
    _validate_num_neighbors,
    mutual_info,
    mutual_info_classwise,
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
    """Two questions per column, answered from two different places.

    ``coded`` is read from the values -- can this column be tabulated -- and ``declared``
    is the caller's word on whether the column's alphabet belongs to the variable. Every
    column of ``FACTOR_DATA`` holds integers, so ``coded`` is True throughout however the
    caller declares them.
    """

    def test_provided_discrete_features(self):
        data, sklearn_list, coded, declared = _merge_labels_and_factors(CLASS_LABELS, FACTOR_DATA, [False, True, False])
        assert data.shape == (FACTOR_DATA.shape[0], FACTOR_DATA.shape[1] + 1)
        # The declaration is the caller's, verbatim, with the label axis prepended.
        assert declared == [True, False, True, False]
        # It does not reach `coded`, which describes the columns rather than the variables.
        assert coded == [True, True, True, True]
        assert sklearn_list == [True, True, True, False]

    def test_declaration_does_not_move_the_sklearn_list(self):
        """Declaring a column's alphabet an artifact does not change which estimator reads it.

        The two lists were one flag before, so a caller declaring a binned factor
        continuous also pushed it onto the neighbor-based estimator -- which then read bin
        indices as measurements. Now the estimator follows the values.
        """
        _, sklearn_list, coded, declared = _merge_labels_and_factors(CLASS_LABELS, FACTOR_DATA, [False, True, True])
        assert declared == [True, False, True, True]
        assert coded == [True, True, True, True]
        # A column of distinct values is still presented to sklearn as continuous, so the
        # estimator does not treat every value as its own category. `coded` keeps it
        # tabulable, since a per-row identifier is what the chance correction is for.
        assert sklearn_list == [True, True, True, False]


@pytest.mark.required
class TestBalanceFunctional:
    def test_balance(self):
        """Test the balance function with TypedDict return."""
        result = mutual_info(CLASS_LABELS, FACTOR_DATA, discrete_features=[True, True, True])

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
        # value per row -- no longer reports an association with the other two. The
        # diagonal is 1.0 throughout: a factor shares all of itself with itself, whatever
        # it is made of.
        np.testing.assert_allclose(
            result["interfactor"],
            np.array([[1.0, 0.64, 0.0], [0.64, 1.0, 0.0], [0.0, 0.0, 1.0]]),
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

    @staticmethod
    def _quantile_bins(values, count):
        """Cut `values` into `count` equal-occupancy bins."""
        return np.digitize(values, np.quantile(values, np.linspace(0, 1, count + 1)[1:-1]))

    def _correlated_pair(self, rho=0.9, n=20000, seed=4):
        rng = np.random.default_rng(seed)
        first = rng.normal(size=n)
        second = rho * first + np.sqrt(1 - rho**2) * rng.normal(size=n)
        return rng.integers(0, 2, size=n), first, second

    @pytest.mark.parametrize("bin_count", [8, 16, 32, 64])
    def test_binned_pair_holds_still_as_the_cut_changes(self, bin_count):
        """A pair of binned factors scores the same however finely the same data is cut.

        :class:`~dataeval.Metadata` derives a factor's bin count from the data rather than
        taking it as a setting, so the same factor measured twice can arrive cut into
        different numbers of bins. Dividing by a binned factor's entropy made the reported
        association a function of that count -- 0.40 at four bins down to 0.14 at 128 on
        this data -- which put the correlation threshold at the mercy of the draw.

        For a bivariate normal pair the Linfoot transformation has an exact target: it
        equals rho^2, here 0.81. Binning still costs resolution at the coarse end, so the
        floor is generous; what this pins is that the value converges rather than decays.
        """
        labels, first, second = self._correlated_pair()
        factors = np.column_stack([
            self._quantile_bins(first, bin_count),
            self._quantile_bins(second, bin_count),
        ])
        # Declared False: these hold bin indices, so their alphabet is the cut's, not theirs.
        score = mutual_info(labels, factors, discrete_features=[False, False])["interfactor"][0, 1]
        assert 0.72 < score < 0.85

    @pytest.mark.parametrize("bin_count", [8, 64, 256, 1024])
    def test_independent_binned_pair_scores_zero_at_any_cut(self, bin_count):
        """Dropping the entropy ceiling must not drop the chance correction with it.

        The correction removes the mutual information a pair's cardinality produces on
        average under independence, not the sampling noise around it, so the residual is
        small rather than exactly zero -- and unlike the floor it does not grow with the
        bin count. The same bound the class row is held to; without the correction this
        pair reads as a genuine association at the finer cuts.
        """
        rng = np.random.default_rng(11)
        n = 20000
        labels = rng.integers(0, 2, size=n)
        factors = np.column_stack([
            self._quantile_bins(rng.normal(size=n), bin_count),
            self._quantile_bins(rng.normal(size=n), bin_count),
        ])
        score = mutual_info(labels, factors, discrete_features=[False, False])["interfactor"][0, 1]
        assert score < 0.01

    @pytest.mark.parametrize("cardinality", [2, 8, 40])
    def test_identical_factors_with_their_own_alphabet_score_one(self, cardinality):
        """A factor whose alphabet is its own keeps the entropy ceiling, so a duplicate is 1.0."""
        rng = np.random.default_rng(4)
        n = 20000
        values = rng.integers(0, cardinality, size=n)
        factors = np.column_stack([values, values])
        score = mutual_info(rng.integers(0, 2, size=n), factors, discrete_features=[True, True])["interfactor"][0, 1]
        assert score == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("bin_count", [2, 3, 8, 32])
    def test_identical_binned_factors_also_score_one(self, bin_count):
        """A duplicate reads 1.0 on the Linfoot branch too, however coarse the cut.

        Mutual information is capped by the smaller of the two entropies whatever produced
        the codes, so the transformation alone tops out at ``1 - exp(-2 * min(H1, H2))`` --
        0.75 for a binary pair. Dividing by that reachable maximum is what puts a coarsely
        cut pair and a finely cut one on one scale, and is why a fixed correlation threshold
        can mean the same thing for both.
        """
        rng = np.random.default_rng(4)
        n = 20000
        codes = self._quantile_bins(rng.normal(size=n), bin_count)
        factors = np.column_stack([codes, codes])
        score = mutual_info(rng.integers(0, 2, size=n), factors, discrete_features=[False, False])
        assert score["interfactor"][0, 1] == pytest.approx(1.0, abs=1e-6)

    def test_lopsided_binary_split_is_not_penalized_for_its_shape(self):
        """The reachable maximum follows occupancy, not just level count.

        A 90/10 split carries far less entropy than an even one, so the transformation alone
        caps it near 0.47. Reading the ceiling off the observed counts rather than off the
        number of levels is what keeps a duplicate at 1.0 here as well.
        """
        rng = np.random.default_rng(7)
        n = 20000
        codes = np.digitize(rng.normal(size=n), [1.2816])
        assert 0.05 < codes.mean() < 0.15  # the split really is lopsided
        factors = np.column_stack([codes, codes])
        score = mutual_info(rng.integers(0, 2, size=n), factors, discrete_features=[False, False])
        assert score["interfactor"][0, 1] == pytest.approx(1.0, abs=1e-6)

    def test_a_coarse_cut_still_reports_less_than_a_fine_one(self):
        """Normalizing the ceiling must not hand back the resolution the cut destroyed.

        The reachable maximum is a property of the alphabet; what the pair actually shares is
        a property of the data. Only the first is divided out, so a binary cut of a strongly
        correlated pair still reports well below a sixteen-bin cut of the same values.
        """
        labels, first, second = self._correlated_pair()
        scores = {
            k: mutual_info(
                labels,
                np.column_stack([self._quantile_bins(first, k), self._quantile_bins(second, k)]),
                discrete_features=[False, False],
            )["interfactor"][0, 1]
            for k in (2, 16)
        }
        assert scores[2] < 0.75 < scores[16]

    def test_class_row_is_unmoved_by_the_declaration(self):
        """`discrete_features` reaches the factor block only; the class row divides by H(class).

        The class row's denominator belongs to the class, which is never binned, so the
        declaration has nothing to change there. Keeping the two independent is what lets
        `balance` and `classwise` stay comparable while `factors` is rescaled.
        """
        labels, first, second = self._correlated_pair()
        factors = np.column_stack([self._quantile_bins(first, 16), self._quantile_bins(second, 16)])
        as_own = mutual_info(labels, factors, discrete_features=[True, True])["class_to_factor"]
        as_artifact = mutual_info(labels, factors, discrete_features=[False, False])["class_to_factor"]
        np.testing.assert_array_equal(as_own, as_artifact)

    @pytest.mark.parametrize("declaration", [False, True])
    def test_measured_values_still_reach_the_neighbor_estimator(self, declaration):
        """A column with a fractional part is not tabulable, whatever the caller declares.

        This is the path `Balance` cannot reach, and the one `num_neighbors` exists for.
        """
        rng = np.random.default_rng(2)
        n = 3000
        labels = rng.choice([0, 1, 2], size=n)
        measured = (labels * 2.0 + rng.normal(size=n)).reshape(-1, 1)
        score = mutual_info(labels, measured, discrete_features=[declaration])["class_to_factor"][1]
        # Tabulating a column of 3000 distinct floats would give every row its own cell and
        # score it at zero; the estimator recovers the relationship instead.
        assert score > 0.4

    def test_mixed_pair_does_not_exceed_one(self):
        """A coded factor against a measured near-copy stays on the documented [0, 1] scale.

        This pair takes neither of the branches that clip. It is not tabulable, so the
        chance-corrected path skips it; and only one side offers an entropy, so it is
        divided by that rather than falling through to Linfoot. The neighbor estimator that
        produced the numerator is not bounded by the coded partner's entropy, so a
        near-deterministic pair overshoots -- 1.0011 on this data before the clip.
        """
        rng = np.random.default_rng(0)
        n = 2000
        codes = rng.integers(0, 12, size=n).astype(np.float64)
        jittered = codes + rng.normal(0, 0.01, size=n)
        factors = np.column_stack([codes, jittered])
        score = mutual_info(rng.integers(0, 2, size=n), factors, discrete_features=[True, False])
        assert score["interfactor"][0, 1] <= 1.0

    def test_omitting_discrete_features_raises(self):
        """The declaration is required: the auto-detect that used to cover for it cannot work.

        A factor cut into bins and a factor with that many categories are the same integers,
        so nothing in the array separates them; only the caller knows. v1.1 guessed and
        warned; the guess is gone and the argument is required.
        """
        with pytest.raises(TypeError, match="discrete_features"):
            mutual_info(CLASS_LABELS, FACTOR_DATA)  # type: ignore[reportCallIssue]

    def test_classwise_rejects_discrete_features(self):
        """The argument is gone, not ignored: passing it is a programming error now.

        Every row of this output is divided by the entropy of one class against the rest,
        which belongs to the class label rather than to any factor -- so there was never a
        factor entropy for the declaration to select. v1.1 accepted and warned; the argument
        is removed rather than made required, which is the opposite of `mutual_info`.
        """
        with pytest.raises(TypeError, match="discrete_features"):
            mutual_info_classwise(CLASS_LABELS, FACTOR_DATA, discrete_features=[True, True, True])  # type: ignore[reportCallIssue]

    def test_classwise_calls_without_warnings(self, recwarn):
        """The argument is gone, so a plain call is all there is to check.

        Every row of this output is divided by the entropy of one class against the rest,
        which belongs to the class label rather than to any factor, so the declaration
        that :func:`mutual_info` still takes had no purchase here at all.
        """
        result = mutual_info_classwise(CLASS_LABELS, FACTOR_DATA)
        assert result.shape == (2, FACTOR_DATA.shape[1] + 1)
        assert not recwarn.list

    def test_constant_factor_zero_norm(self):
        """A constant factor has zero entropy, so its normalization factor is 0 and MI is 0.0."""
        # First factor is constant (single value) -> entropy 0 -> norm_factor 0.
        factor_data = np.column_stack([np.full(10, 7), np.arange(10) % 3])
        result = mutual_info(CLASS_LABELS, factor_data, discrete_features=[True, True])

        # The constant factor is index 0 in the interfactor matrix.
        assert result["interfactor"][0, 0] == 0.0
        # class-to-constant-factor MI is also driven to 0.0.
        assert result["class_to_factor"][1] == 0.0
