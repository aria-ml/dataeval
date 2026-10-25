import numpy as np
import pytest

from dataeval.core import ap_lower_staircase, ap_upper_envelope, kink_costs, witness_thresholds
from dataeval.core._ap_bounds import _precision_upper

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")


@pytest.mark.required
class TestEnvelope:
    def test_envelope_reproduces_the_explainer_table(self):
        p = _precision_upper(np.array([0.5, 0.8, 1.0]), np.array([0.0, 0.4]), np.array([0.0, 1.0]), 1.0)
        np.testing.assert_allclose(p, [1.0, 0.8, 1 / 1.4])
        assert np.round(p, 3).tolist() == [1.0, 0.8, 0.714]

    @pytest.mark.parametrize(("w_a", "q_a", "q_b"), [(0.04, 0.9, 0.015), (0.25, 0.9, 0.1), (0.6, 0.9, 0.15)])
    def test_envelope_bounds_an_exact_population_ap(self, w_a, q_a, q_b):
        # Two score types: type A (weight w_a) has P(positive) = q_a, type B the rest with q_b < q_a.
        w, q = np.array([w_a, 1 - w_a]), np.array([q_a, q_b])
        mu = float(w @ q)
        costs = kink_costs(5)
        # Exact cost-weighted Bayes risk: per type, the cheaper of skipping (q) and emitting (cost * (1 - q)).
        risk = (w * np.minimum(q, costs[:, None] * (1 - q))).sum(axis=1)
        # Posterior scorer, ties grouped: emitting A gives (recall w_a q_a / mu, precision q_a); emitting both
        # gives (1, mu). Interpolated AP integrates the best precision at or above each recall.
        r_a = w_a * q_a / mu
        ap = r_a * q_a + (1 - r_a) * mu
        upper = ap_upper_envelope(risk, costs, mu)
        assert ap <= upper < 1

    def test_envelope_is_nonincreasing(self):
        rng = np.random.default_rng(0)
        costs = kink_costs(8)
        risk = np.r_[0.0, rng.uniform(0, 0.5, len(costs) - 1)]
        p = _precision_upper(np.linspace(0, 1, 501), risk, costs, 0.3)
        assert p[0] == 1
        assert np.all(np.diff(p) <= 1e-12)

    def test_absent_class_envelope_is_one(self):
        assert ap_upper_envelope(np.array([0.0, 0.2, 0.4]), np.array([0.0, 1.0, 4.0]), 0.0) == 1.0

    def test_risk_above_prevalence_is_clipped(self):
        costs = np.array([0.0, 1.0])
        assert ap_upper_envelope(np.array([0.0, 0.9]), costs, 0.5) == ap_upper_envelope(
            np.array([0.0, 0.5]), costs, 0.5
        )

    def test_zero_risk_gives_one(self):
        assert ap_upper_envelope(np.zeros(3), np.array([0.0, 1.0, 2.0]), 0.5) == 1.0

    @pytest.mark.parametrize(
        ("risk", "costs", "mu", "match"),
        [
            ([0.1, 0.2], [0.5, 1.0], 0.5, "one zero cost"),
            ([0.0, 0.0, 0.2], [0.0, 0.0, 1.0], 0.5, "distinct"),
            ([0.1, 0.2], [0.0, 1.0], 0.5, "0 at cost 0"),
            ([0.0, -0.1], [0.0, 1.0], 0.5, "non-negative"),
            ([0.0, 0.1], [0.0, 1.0, 2.0], 0.5, "aligned"),
            ([0.0, 0.1], [0.0, 1.0], -0.1, "non-negative"),
            ([0.0, np.nan], [0.0, 1.0], 0.5, "finite"),
        ],
    )
    def test_invalid_inputs_raise(self, risk, costs, mu, match):
        with pytest.raises(ValueError, match=match):
            ap_upper_envelope(np.array(risk), np.array(costs), mu)


@pytest.mark.required
class TestStaircase:
    def test_staircase_is_below_a_perfect_scorer(self):
        tp = np.array([0.1, 0.2, 0.3])
        assert ap_lower_staircase(tp, np.zeros(3), 0.3) == pytest.approx(1.0)
        # Endpoints: TP lower, FP upper and prevalence upper each loosen the witness.
        assert ap_lower_staircase(tp - 0.01, np.full(3, 0.01), 0.32) < 1.0

    def test_staircase_hand_value(self):
        # recall [0.5, 1], precision [1, 0.5] -> 0.5 * 1 + 0.5 * 0.5
        assert ap_lower_staircase(np.array([0.5, 1.0]), np.array([0.0, 1.0]), 1.0) == pytest.approx(0.75)

    @pytest.mark.parametrize(
        ("tp", "fp", "mu", "match"),
        [
            ([0.1], [0.0, 0.1], 0.3, "aligned"),
            ([0.4], [0.0], 0.3, "at most mu_upper"),
            ([0.1], [-0.1], 0.3, "non-negative"),
            ([0.1], [0.0], 0.0, "positive"),
        ],
    )
    def test_invalid_inputs_raise(self, tp, fp, mu, match):
        with pytest.raises(ValueError, match=match):
            ap_lower_staircase(np.array(tp), np.array(fp), mu)


@pytest.mark.required
class TestWitnessThresholds:
    quantiles = np.linspace(0.05, 0.95, 19)

    def test_pos_thresholds_use_only_positives(self):
        rng = np.random.default_rng(0)
        scores = rng.random(200)
        positive = rng.random(200) < 0.05
        moved = np.where(positive, scores, scores / 10)
        a = witness_thresholds(scores, positive, 0.05, pos_below=0.1, quantiles=self.quantiles)
        b = witness_thresholds(moved, positive, 0.05, pos_below=0.1, quantiles=self.quantiles)
        np.testing.assert_array_equal(a, b)
        c = witness_thresholds(moved, positive, 0.05, pos_below=0.01, quantiles=self.quantiles)
        np.testing.assert_array_equal(c, np.quantile(moved, self.quantiles))

    def test_thresholds_are_deduplicated(self):
        scores = np.r_[np.zeros(50), np.ones(50)]
        t = witness_thresholds(scores, scores > 0, 0.5, pos_below=0.1, quantiles=self.quantiles)
        np.testing.assert_allclose(t, [0.0, 0.5, 1.0])

    def test_no_fit_positives_falls_back_to_all_scores(self):
        scores = np.linspace(0, 1, 11)
        t = witness_thresholds(scores, np.zeros(11, bool), 0.0, pos_below=0.1, quantiles=self.quantiles)
        np.testing.assert_array_equal(t, np.quantile(scores, self.quantiles))

    @pytest.mark.parametrize(
        ("scores", "positive", "prevalence", "pos_below", "quantiles", "match"),
        [
            (np.ones((2, 2)), np.ones((2, 2), bool), 0.5, 0.1, [0.5], "1D"),
            (np.ones(0), np.ones(0, bool), 0.5, 0.1, [0.5], "nonempty"),
            (np.ones(3), np.ones(2, bool), 0.5, 0.1, [0.5], "aligned"),
            (np.r_[0.0, np.nan], np.ones(2, bool), 0.5, 0.1, [0.5], "finite"),
            (np.ones(2), np.ones(2, bool), np.inf, 0.1, [0.5], "finite"),
            (np.ones(2), np.ones(2, bool), 0.5, 1.5, [0.5], "pos_below"),
            (np.ones(2), np.ones(2, bool), 0.5, np.nan, [0.5], "pos_below"),
            (np.ones(2), np.ones(2, bool), 0.5, 0.1, [0.5, 1.2], "quantiles"),
            (np.ones(2), np.ones(2, bool), 0.5, 0.1, [], "quantiles"),
        ],
    )
    def test_invalid_inputs_raise(self, scores, positive, prevalence, pos_below, quantiles, match):
        with pytest.raises(ValueError, match=match):
            witness_thresholds(scores, positive, prevalence, pos_below=pos_below, quantiles=np.asarray(quantiles))
