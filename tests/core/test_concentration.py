import numpy as np
import pytest
from scipy.stats import beta

from dataeval.core import binomial_endpoint, mean_endpoint, ratio_endpoint, units_to_cross
from dataeval.core._concentration import _projected_bound, _radius

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

# Reference values: spike/feasibility-dfe:spike/feasibility/dfe_reference.py, mean_endpoint
# (torch float64) on x = [1]*80 + [0]*20, support 1.0, delta 0.05.
X = np.r_[np.ones(80), np.zeros(20)]
REF = {
    ("hybrid", "lower"): 0.6641898484259381,
    ("hybrid", "upper"): 0.935810151574062,
    ("hoeffding", "lower"): 0.6776126584659592,
    ("hoeffding", "upper"): 0.9223873415340409,
    ("empirical_bernstein", "lower"): 0.6038612427802877,
    ("empirical_bernstein", "upper"): 0.9961387572197123,
}
REF_RADIUS_HOEFFDING_HYBRID = 0.13581015157406195
REF_RADIUS_BERNSTEIN_HYBRID = 0.22229322484518
REF_VARIANCE = 0.1616161616161617
# Same reference on y with support 2.0, delta 0.1, side="upper".
REF_Y_UPPER = 1.368018483775424


@pytest.mark.required
class TestMeanEndpoint:
    @pytest.mark.parametrize(("method", "side"), list(REF))
    def test_matches_reference_values(self, method, side):
        r = mean_endpoint(X, 1.0, 0.05, side=side, method=method)
        assert r["endpoint"] == pytest.approx(REF[method, side], abs=1e-12)
        assert r["method"] == method
        assert r["side"] == side

    def test_matches_reference_components(self):
        r = mean_endpoint(X, 1.0, 0.05)
        assert r["radius_hoeffding"] == pytest.approx(REF_RADIUS_HOEFFDING_HYBRID, abs=1e-12)
        assert r["radius_bernstein"] == pytest.approx(REF_RADIUS_BERNSTEIN_HYBRID, abs=1e-12)
        assert r["variance"] == pytest.approx(REF_VARIANCE, abs=1e-12)
        assert (r["n"], r["mean"], r["support"], r["delta"]) == (100, 0.8, 1.0, 0.05)

    def test_matches_reference_nonunit_support(self):
        y = np.array([0.1, 0.4, 0.2, 0.9, 0.5, 0.3, 0.7])
        r = mean_endpoint(y, 2.0, 0.1, side="upper")
        assert r["endpoint"] == pytest.approx(REF_Y_UPPER, abs=1e-12)

    def test_perfect_witness_is_not_one(self):
        assert mean_endpoint(np.ones(400), 1.0, 0.05)["endpoint"] == pytest.approx(0.974, abs=5e-4)

    def test_single_observation_uses_hoeffding_only(self):
        r = mean_endpoint(np.array([0.5]), 1.0, 0.05)
        assert r["radius_bernstein"] is None
        assert r["variance"] is None
        assert r["method"] == "hybrid"
        assert r["endpoint"] == 0.0  # radius capped by support, endpoint clipped to [0, B]

    def test_upper_side_mirrors_lower(self):
        x = np.random.default_rng(0).uniform(size=500)
        lo = mean_endpoint(x, 1, 0.05)["endpoint"]
        hi = mean_endpoint(x, 1, 0.05, side="upper")["endpoint"]
        assert lo < x.mean() < hi

    @pytest.mark.parametrize(
        ("values", "support", "delta", "kwargs", "match"),
        [
            ([], 1.0, 0.05, {}, "non-empty"),
            ([0.5, 1.5], 1.0, 0.05, {}, "outside"),
            ([-0.1, 0.5], 1.0, 0.05, {}, "outside"),
            ([np.nan, 0.5], 1.0, 0.05, {}, "finite"),
            ([0.5], 1.0, 0.0, {}, "delta"),
            ([0.5], 1.0, 1.0, {}, "delta"),
            ([0.5], 0.0, 0.05, {}, "support"),
            ([0.5], -1.0, 0.05, {}, "support"),
            ([[0.5, 0.5]], 1.0, 0.05, {}, "1D"),
            ([0.5], 1.0, 0.05, {"side": "both"}, "side"),
            ([0.5], 1.0, 0.05, {"method": "bogus"}, "method"),
        ],
    )
    def test_invalid_inputs_raise(self, values, support, delta, kwargs, match):
        with pytest.raises(ValueError, match=match):
            mean_endpoint(values, support, delta, **kwargs)


@pytest.mark.required
class TestRatioEndpoint:
    def test_ratio_endpoint_is_conservative(self):
        num, den = np.array([3.0, 5.0, 2.0]), np.array([4.0, 6.0, 3.0])
        assert ratio_endpoint(num, den, 10, 10, 0.05) <= num.sum() / den.sum()
        assert ratio_endpoint(num, den, 10, 10, 0.05, side="upper") >= num.sum() / den.sum()

    def test_composes_mean_endpoints_at_half_delta(self):
        rng = np.random.default_rng(1)
        num, den = rng.uniform(0, 1, 300), rng.uniform(0.5, 1, 300)
        lo = mean_endpoint(num, 1, 0.025)["endpoint"] / mean_endpoint(den, 1, 0.025, side="upper")["endpoint"]
        assert ratio_endpoint(num, den, 1, 1, 0.05) == pytest.approx(lo)
        hi = mean_endpoint(num, 1, 0.025, side="upper")["endpoint"] / mean_endpoint(den, 1, 0.025)["endpoint"]
        assert ratio_endpoint(num, den, 1, 1, 0.05, side="upper") == pytest.approx(hi)

    def test_upper_is_infinite_when_denominator_lower_is_zero(self):
        assert ratio_endpoint(np.ones(3), np.zeros(3), 1, 1, 0.05, side="upper") == float("inf")

    @pytest.mark.parametrize("delta", [0.0, 1.0, 1.5, 2.5, float("nan")])
    def test_invalid_delta_raises(self, delta):
        with pytest.raises(ValueError, match=rf"delta must be in \(0, 1\), got {delta}"):
            ratio_endpoint(np.ones(3), np.ones(3), 1, 1, delta)

    def test_invalid_side_and_method_raise(self):
        with pytest.raises(ValueError, match="side"):
            ratio_endpoint(np.ones(3), np.ones(3), 1, 1, 0.05, side="both")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="method"):
            ratio_endpoint(np.ones(3), np.ones(3), 1, 1, 0.05, method="bogus")  # type: ignore[arg-type]

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            ratio_endpoint(np.ones(3), np.ones(4), 1, 1, 0.05)


@pytest.mark.required
class TestUnitsToCross:
    def test_units_to_cross_inverts_the_radius(self):
        n = units_to_cross(0.8, 0.16, 1.0, 0.05, 0.02)
        assert n is not None

        def lower(k):
            ones = int(0.8 * k)
            return mean_endpoint(np.r_[np.ones(ones), np.zeros(k - ones)], 1, 0.05)["endpoint"]

        assert 0.8 - lower(n) <= 0.02 + 1e-3
        assert 0.8 - lower(n - 20) > 0.02 - 1e-3

    @pytest.mark.parametrize("method", ["hoeffding", "empirical_bernstein", "hybrid"])
    def test_smallest_n_below_gap(self, method):
        n = units_to_cross(0.5, 0.25, 1.0, 0.05, 0.05, method=method)
        assert n is not None
        assert n > 2
        assert _radius(n, 0.25, 1.0, 0.05, method) < 0.05 <= _radius(n - 1, 0.25, 1.0, 0.05, method)

    def test_unreachable_returns_none(self):
        assert units_to_cross(0.5, 0.25, 1.0, 0.05, 1e-6) is None

    def test_invalid_gap_raises(self):
        with pytest.raises(ValueError, match="gap"):
            units_to_cross(0.5, 0.25, 1.0, 0.05, 0.0)


@pytest.mark.required
class TestBinomialEndpoint:
    def test_matches_clopper_pearson(self):
        r = binomial_endpoint(37, 50, 0.025, side="lower")
        assert r["endpoint"] == pytest.approx(beta.ppf(0.025, 37, 14))
        assert r["method"] == "exact_binomial"
        assert r["support"] == 1.0
        assert r["mean"] == 37 / 50
        assert r["variance"] == pytest.approx(37 * 13 / (50 * 49))
        assert r["radius_bernstein"] is None
        u = binomial_endpoint(37, 50, 0.025, side="upper")
        assert u["endpoint"] == pytest.approx(beta.ppf(0.975, 38, 13))

    def test_single_trial_has_no_variance(self):
        assert binomial_endpoint(1, 1, 0.05)["variance"] is None

    @pytest.mark.parametrize("n", [1, 7, 400])
    def test_edges(self, n):
        assert binomial_endpoint(0, n, 0.05, side="lower")["endpoint"] == 0.0
        assert binomial_endpoint(0, n, 0.05, side="upper")["endpoint"] == pytest.approx(1 - 0.05 ** (1 / n))
        assert binomial_endpoint(n, n, 0.05, side="lower")["endpoint"] == pytest.approx(0.05 ** (1 / n))
        assert binomial_endpoint(n, n, 0.05, side="upper")["endpoint"] == 1.0

    @pytest.mark.parametrize("bad", [(-1, 5), (6, 5), (1, 0), (1.5, 5), (True, 5), (1, 5.0)])
    def test_rejects_bad_counts(self, bad):
        successes, trials = bad
        with pytest.raises(ValueError, match="integers|trials"):
            binomial_endpoint(successes, trials, 0.05)

    @pytest.mark.parametrize("kwargs", [{"delta": 0.0}, {"delta": 1.0}, {"delta": 0.05, "side": "middle"}])
    def test_rejects_bad_delta_or_side(self, kwargs):
        with pytest.raises(ValueError, match="delta|side"):
            binomial_endpoint(3, 5, **kwargs)

    @pytest.mark.parametrize("p", [0.0, 0.02, 0.5, 0.97, 1.0])
    @pytest.mark.parametrize("delta", [1e-4, 0.05])
    @pytest.mark.parametrize("side", ["lower", "upper"])
    def test_projected_bound_moves_monotonically_toward_the_rate(self, p, delta, side):
        # the real-valued projected count k = p*n gives a bound that never moves away from p as n grows
        bounds = np.array([_projected_bound(p, n, delta, side) for n in range(1, 3000, 7)])
        diffs = np.diff(bounds)
        assert (diffs >= -1e-12).all() if side == "lower" else (diffs <= 1e-12).all()


@pytest.mark.required
class TestUnitsToCrossExactBinomial:
    def test_lower_and_upper(self):
        n = units_to_cross(0.9, 0.0, 1.0, 0.01, 0.05, method="exact_binomial", side="lower")
        assert n is not None
        assert _projected_bound(0.9, n, 0.01, "lower") > 0.85 >= _projected_bound(0.9, n - 1, 0.01, "lower")
        m = units_to_cross(0.02, 0.0, 1.0, 0.01, 0.03, method="exact_binomial", side="upper")
        assert m is not None
        assert _projected_bound(0.02, m, 0.01, "upper") < 0.05 <= _projected_bound(0.02, m - 1, 0.01, "upper")
        assert units_to_cross(0.0, 0.0, 1.0, 0.05, 0.01, method="exact_binomial", side="upper") is not None

    def test_unreachable_returns_none(self):
        assert units_to_cross(0.5, 0.0, 1.0, 0.05, 1e-6, method="exact_binomial") is None

    def test_requires_unit_support(self):
        with pytest.raises(ValueError, match="support"):
            units_to_cross(0.5, 0.0, 2.0, 0.05, 0.05, method="exact_binomial")

    def test_side_is_ignored_by_other_methods(self):
        assert units_to_cross(0.5, 0.25, 1.0, 0.05, 0.05, side="upper") == units_to_cross(0.5, 0.25, 1.0, 0.05, 0.05)
