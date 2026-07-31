"""Validation benchmarks: estimators checked against analytically known values.

The other test modules in this directory are predominantly *regression* tests —
they pin an estimator's output against a value recorded from a previous run. A
regression test fails when behavior changes; it passes just as happily when the
estimator has been wrong from the start.

The tests here are different in kind. Each one constructs a distribution whose
target quantity has a closed-form solution, then asserts that DataEval's
estimator recovers it. They fail when an estimate is *wrong*, not merely when it
moves.

These use synthetic data with an analytic ground truth. That is a stronger
guarantee than a regression test and a weaker one than reproducing a published
benchmark on a real dataset — see the "Known gaps" section of
``docs/source/concepts/ValidationAndTrust.md``.
"""

import numpy as np
import pytest
from scipy.stats import norm

from dataeval.core import ber_knn, ber_mst, divergence_fnn, divergence_mst, nullmodel_metrics

# Samples per class. Large enough that sampling noise is small relative to the
# bounds being tested, small enough that the whole module runs in ~1 second.
N_PER_CLASS = 4000


def _two_gaussians(separation: float, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a binary problem whose Bayes error rate is known in closed form.

    Two isotropic unit-variance Gaussians in 2-D, centered at ``-separation``
    and ``+separation`` along the first axis and identical along the second.
    Only the first axis carries information, so with equal class priors the
    optimal decision boundary is the hyperplane ``x0 = 0`` and the Bayes error
    rate is exactly ``Phi(-separation)``.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, (n, 2))
    a[:, 0] -= separation
    b = rng.normal(0.0, 1.0, (n, 2))
    b[:, 0] += separation
    return np.vstack([a, b]), np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])


def _standard_error(p: float, n: int) -> float:
    """Standard error of an error-rate estimate over ``n`` samples."""
    return float(np.sqrt(p * (1.0 - p) / n))


@pytest.mark.required
class TestBERAgainstAnalyticBayesError:
    """BER estimators must bracket the true Bayes error of a known distribution.

    For two equal-prior isotropic Gaussians separated by ``2 * separation``
    along one axis, the Bayes error rate is exactly ``Phi(-separation)``. Both
    estimators claim to bound that quantity, so the interval they return must
    contain it.

    The comparison allows three standard errors of slack. The bounds are
    estimated from a finite sample, and at wide separations the interval
    narrows to a few thousandths — tight enough that sampling noise alone can
    push an endpoint past the true value. Without this allowance the test is
    seed-dependent; with it, it passes for every seed tried.
    """

    @pytest.mark.parametrize("separation", [0.5, 1.0, 1.5, 2.0])
    def test_mst_bounds_contain_true_ber(self, separation: float) -> None:
        embeddings, labels = _two_gaussians(separation, N_PER_CLASS, seed=0)
        truth = float(norm.cdf(-separation))
        tolerance = 3 * _standard_error(truth, 2 * N_PER_CLASS)

        result = ber_mst(embeddings, labels)

        assert result["lower_bound"] - tolerance <= truth <= result["upper_bound"] + tolerance, (
            f"MST bounds [{result['lower_bound']:.4f}, {result['upper_bound']:.4f}] "
            f"do not contain the analytic BER {truth:.4f} (tolerance {tolerance:.4f})"
        )

    @pytest.mark.parametrize("separation", [0.5, 1.0, 1.5, 2.0])
    def test_knn_bounds_contain_true_ber(self, separation: float) -> None:
        embeddings, labels = _two_gaussians(separation, N_PER_CLASS, seed=0)
        truth = float(norm.cdf(-separation))
        tolerance = 3 * _standard_error(truth, 2 * N_PER_CLASS)

        result = ber_knn(embeddings, labels, k=10)

        assert result["lower_bound"] - tolerance <= truth <= result["upper_bound"] + tolerance, (
            f"KNN bounds [{result['lower_bound']:.4f}, {result['upper_bound']:.4f}] "
            f"do not contain the analytic BER {truth:.4f} (tolerance {tolerance:.4f})"
        )

    def test_bounds_are_ordered(self) -> None:
        """The lower bound never exceeds the upper bound."""
        embeddings, labels = _two_gaussians(1.0, N_PER_CLASS, seed=0)
        for result in (ber_mst(embeddings, labels), ber_knn(embeddings, labels, k=10)):
            assert result["lower_bound"] <= result["upper_bound"]

    @pytest.mark.parametrize("separation", [0.5, 1.0, 1.5, 2.0])
    def test_bounds_are_informative(self, separation: float) -> None:
        """The interval must be tight enough to be worth reporting.

        Containment alone is a one-sided guarantee: an estimator that returned
        ``[0, 1]`` would satisfy it while saying nothing. This pins the bounds
        to within a factor of the true value, so an interval that is merely
        wide fails.

        The factors are set from measured behavior with headroom. Over 20 seeds
        at each separation, ``upper / truth`` stayed within [0.98, 1.76] and
        ``lower / truth`` within [0.65, 1.01] for both estimators; the limits
        below sit outside that range but well inside the roughly 2.5x-3.2x
        ratio a doubled upper bound would produce.
        """
        embeddings, labels = _two_gaussians(separation, N_PER_CLASS, seed=0)
        truth = float(norm.cdf(-separation))

        for name, result in (
            ("MST", ber_mst(embeddings, labels)),
            ("KNN", ber_knn(embeddings, labels, k=10)),
        ):
            assert result["upper_bound"] <= 2.2 * truth, (
                f"{name} upper bound {result['upper_bound']:.4f} is uninformatively "
                f"loose against the analytic BER {truth:.4f}"
            )
            assert result["lower_bound"] >= 0.4 * truth, (
                f"{name} lower bound {result['lower_bound']:.4f} is uninformatively "
                f"slack against the analytic BER {truth:.4f}"
            )

    def test_estimates_decrease_as_classes_separate(self) -> None:
        """More separable classes must yield a lower estimated error.

        The true BER is strictly decreasing in separation, so an estimator that
        tracks it must be too. This catches sign errors and normalization
        mistakes that a single-point check would miss.
        """
        uppers = []
        for separation in (0.5, 1.0, 1.5, 2.0):
            embeddings, labels = _two_gaussians(separation, N_PER_CLASS, seed=0)
            uppers.append(ber_mst(embeddings, labels)["upper_bound"])

        assert uppers == sorted(uppers, reverse=True), f"upper bounds not monotonically decreasing: {uppers}"


@pytest.mark.required
class TestDivergenceEndpoints:
    """Divergence must hit its defined endpoints on constructed distributions.

    The metric is documented as running from 0 (identical) to 1 (completely
    separable). Those two endpoints are exactly reachable with constructed
    inputs, which makes them checkable without reference to a recorded value.
    """

    @pytest.mark.parametrize("estimator", [divergence_mst, divergence_fnn])
    def test_identical_distributions_give_zero(self, estimator) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, (1500, 2))
        b = rng.normal(0.0, 1.0, (1500, 2))

        assert estimator(a, b)["divergence"] == pytest.approx(0.0, abs=0.05)

    @pytest.mark.parametrize("estimator", [divergence_mst, divergence_fnn])
    def test_disjoint_distributions_give_one(self, estimator) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, (1500, 2))
        b = rng.normal(50.0, 1.0, (1500, 2))

        assert estimator(a, b)["divergence"] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("estimator", [divergence_mst, divergence_fnn])
    def test_divergence_increases_with_separation(self, estimator) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, (1500, 2))
        scores = [estimator(a, rng.normal(shift, 1.0, (1500, 2)))["divergence"] for shift in (0.0, 1.0, 2.0, 4.0)]

        assert scores == sorted(scores), f"divergence not monotonically increasing with separation: {scores}"


@pytest.mark.required
class TestNullModelAgainstAnalyticBaselines:
    """Null-model baselines must match their closed-form accuracies.

    For a test set with class proportions ``p`` and a training set with class
    proportions ``q``, the expected accuracy of each dummy classifier is exact:

    - uniform random over ``C`` classes: ``1 / C``
    - proportional random: ``sum(p_i * q_i)``
    - dominant class: ``p_j`` where ``j = argmax(q)``

    Labels are built with exact counts rather than sampled, so the empirical
    proportions equal the intended ones and the comparison is exact rather
    than asymptotic.
    """

    @staticmethod
    def _labels_with_exact_counts(counts: list[int]) -> np.ndarray:
        return np.concatenate([np.full(count, index, dtype=int) for index, count in enumerate(counts)])

    @staticmethod
    def _accuracy(metrics) -> float:
        """Overall accuracy, under whichever key the task type reports it.

        Binary problems report ``accuracy_micro``/``accuracy_macro``; problems
        with three or more classes report ``multiclass_accuracy``. For these
        dummy classifiers the two name the same quantity.
        """
        if "multiclass_accuracy" in metrics:
            return float(metrics["multiclass_accuracy"])
        return float(metrics["accuracy_micro"])

    @pytest.mark.parametrize(
        "counts",
        [
            [500, 500],
            [900, 100],
            [600, 300, 100],
            [250, 250, 250, 250],
        ],
    )
    def test_null_model_accuracies_match_closed_form(self, counts: list[int]) -> None:
        labels = self._labels_with_exact_counts(counts)
        proportions = np.array(counts, dtype=float) / sum(counts)

        result = nullmodel_metrics(labels, labels)

        assert "uniform_random" in result
        assert "proportional_random" in result
        assert "dominant_class" in result

        assert self._accuracy(result["uniform_random"]) == pytest.approx(1.0 / len(counts))
        assert self._accuracy(result["proportional_random"]) == pytest.approx(float((proportions**2).sum()))
        assert self._accuracy(result["dominant_class"]) == pytest.approx(float(proportions.max()))

    def test_baselines_use_training_distribution_scored_on_test(self) -> None:
        """Both train-dependent baselines are fit on train and scored on test.

        Training is majority class 0; test is majority class 1. A classifier
        that always predicts the training-dominant class therefore scores the
        *test* frequency of class 0 (0.2), not the majority frequency of either
        set. The proportional baseline scores ``sum(p_i * q_i)``, here
        ``0.2*0.9 + 0.8*0.1 = 0.26``.

        Using one distribution for both sets would let a train/test mix-up pass
        unnoticed; these differ, so it cannot.
        """
        train = self._labels_with_exact_counts([900, 100])
        test = self._labels_with_exact_counts([200, 800])

        result = nullmodel_metrics(test, train)

        assert "dominant_class" in result
        assert "proportional_random" in result

        assert self._accuracy(result["dominant_class"]) == pytest.approx(0.2)
        assert self._accuracy(result["proportional_random"]) == pytest.approx(0.26)
