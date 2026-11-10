"""Tests for the classification feasibility compositions: every bound source on synthetic populations."""

from statistics import NormalDist
from typing import Any

import numpy as np
import pytest
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from dataeval.core import binomial_endpoint
from dataeval.performance import _feasibility_classification as fc
from dataeval.performance._feasibility_classification import count_claims, gather_evidence
from dataeval.performance._feasibility_decide import decide
from dataeval.performance._feasibility_evidence import Bound, allocate, check_ledger
from dataeval.performance._feasibility_target import Allowances, Scope, TargetSpec
from dataeval.performance._feasibility_units import split_units
from tests.performance.feasibility_helpers import draw_labels, make_population, make_sequences, make_units

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

WITNESSES = ("knn", "linear_probe")
NAMES = ["a", "b", "Tank"]  # the target class's index (2) is not 1, so a boolean mix-up would show


def _acc(aggregation=None) -> TargetSpec:  # built per test, so collection constructs no experimental class
    return TargetSpec(task="IC", metric="accuracy", threshold=0.8, aggregation=aggregation)


def _ap(cls: str | None = None, aggregation=None) -> TargetSpec:
    return TargetSpec(
        task="IC",
        metric="ap",
        threshold=0.5,
        class_name=cls,
        ap_functional="population_ap_integral",
        aggregation=aggregation,
    )


def _recall(cls: str = "Tank", f0: float = 0.05) -> TargetSpec:
    return TargetSpec(task="IC", metric="recall", threshold=0.9, class_name=cls, max_false_positive_rate=f0)


def _gauss2(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x, y, post = make_population("gauss2").sample(n, rng)
    return x, y, post, rng


def _three_classes(n: int, seed: int):
    """Labels over NAMES, random features, and class decisions that agree with the labels about 80% of the time."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    decisions = np.where(rng.random(n) < 0.8, y, rng.integers(0, 3, size=n))
    return rng.standard_normal((n, 4)), y, decisions


def _counts(decisions, labels, c: int) -> tuple[int, int, int, int]:
    """Decisions for class c among its positives and among its negatives: (k_pos, n_pos, k_neg, n_neg)."""
    hit, positive = decisions == c, labels == c
    return int((hit & positive).sum()), int(positive.sum()), int((hit & ~positive).sum()), int((~positive).sum())


def _gather(specs, units, features, *, alpha=0.05, seed=0, builtin=True, replicas=None, predictions=None, **kw: Any):
    fit, audit = split_units(units, fit_fraction=0.5, seed=seed)
    n_claims = count_claims(
        specs, has_replicas=bool(replicas), has_predictions=predictions is not None, builtin_witnesses=builtin
    )
    options: dict[str, Any] = {"allowances": None, "max_sequence_length": None} | kw
    bounds = gather_evidence(
        specs,
        units=units,
        features=features,
        audit=options.pop("audit", audit),
        fit=fit,
        replicas=replicas,
        predictions=predictions,
        alpha_claim=allocate(alpha, n_claims),
        seed=seed,
        builtin_witnesses=builtin,
        **options,
    )
    return bounds, fit, audit


def _of(bounds: list[Bound], source: str, requirement: int = 0) -> list[Bound]:
    return [b for b in bounds if b.source == source and b.requirement == requirement]


def _one(bounds: list[Bound], source: str, requirement: int = 0, quantity: str | None = None) -> Bound:
    (found,) = [b for b in _of(bounds, source, requirement) if quantity is None or b.quantity == quantity]
    return found


def _value(bounds: list[Bound], source: str, requirement: int = 0) -> float:
    value = _one(bounds, source, requirement).bound
    assert value is not None
    return value


@pytest.mark.required
class TestFeasibilityClassification:
    def test_replica_envelope_covers_bayes_on_gauss2(self):
        x, y, post, rng = _gauss2(2000, 0)
        extra = draw_labels(post, 4, rng)
        bounds, _, _ = _gather([_acc()], make_units(y), x, builtin=False, replicas=dict(enumerate(extra)))
        b = _one(bounds, "replicas")
        bayes = NormalDist().cdf(1.0)
        assert (b.role, b.information_scope, b.unit, b.sample) == ("envelope", "full_input", "image", "replica_items")
        assert (b.bound_type, b.state) == ("upper", "informative")
        assert bayes <= _value(bounds, "replicas") < bayes + 0.08
        assert (b.n, b.method) == (2000, "hybrid")
        assert b.support == pytest.approx(1 - 3 / 5)  # consensus_max(2, 5)

    def test_unequal_replica_panels_use_hoeffding(self):
        x, y, post, rng = _gauss2(2000, 6)
        units = make_units(y, class_names=["a", "b"])
        alternating = {i: draw_labels(post[i : i + 1], 2 + i % 2, rng)[0] for i in range(500)}  # panels of 3 and 4
        equal = {i: draw_labels(post[i : i + 1], 2, rng)[0] for i in range(500)}
        for replicas, method in ((alternating, "hoeffding"), (equal, "hybrid")):
            bounds, _, _ = _gather([_acc(), _ap("a")], units, x, builtin=False, replicas=replicas)
            assert _one(bounds, "replicas", 0).method == method

    def test_local_panel_scope_follows_allowances(self):
        x, y, _, _ = _gauss2(2000, 1)
        units = make_units(y, class_names=["a", "b"])
        specs = [_acc(), _ap("b")]
        bare, _, _ = _gather(specs, units, x, builtin=False)
        with_beta, _, _ = _gather(specs, units, x, builtin=False, allowances=Allowances(beta=0.01))
        both, _, _ = _gather(specs, units, x, builtin=False, allowances=Allowances(beta=0.01, gamma=0.02))
        scopes = lambda bounds: [b.information_scope for b in bounds]  # noqa: E731
        assert scopes(bare) == ["partition_restricted"] * 2
        assert scopes(with_beta) == ["fixed_representation", "partition_restricted"]
        assert scopes(both) == ["full_input", "partition_restricted"]
        # same seed, same panels: beta and gamma each lower the risk bound by their amount; AP ignores allowances
        acc = _value(bare, "local_panels", 0)
        assert 0.5 < acc < 1 - 0.03
        assert _value(with_beta, "local_panels", 0) == pytest.approx(acc + 0.01)
        assert _value(both, "local_panels", 0) == pytest.approx(acc + 0.03)
        ap = _value(bare, "local_panels", 1)
        assert _value(with_beta, "local_panels", 1) == _value(both, "local_panels", 1) == ap
        assert _one(bare, "local_panels", 0).method == "hoeffding"
        assert _one(bare, "local_panels", 1).method is None
        base = ("representation_independent_of_audit_labels",)
        assert [b.assumptions for b in bare] == [base, base]
        assert _one(with_beta, "local_panels", 0).assumptions == (*base, "allowance_beta")
        assert _one(both, "local_panels", 0).assumptions == (*base, "allowance_beta", "allowance_gamma")
        assert _one(both, "local_panels", 1).assumptions == base

    def test_absent_declared_class_keeps_the_macro_denominator(self):
        x, y, _, _ = _gauss2(2000, 2)  # labels 0 and 1 only; class "c" (index 2) is declared but absent
        units = make_units(y, n_classes=3, class_names=["a", "b", "c"])
        bounds, _, _ = _gather([_ap(), _ap("c")], units, x)
        for source in ("local_panels", *WITNESSES):
            absent = _one(bounds, source, 1)
            envelope = absent.role == "envelope"
            assert (absent.bound, absent.state) == (1.0 if envelope else 0.0, "uninformative")
            assert absent.reason == ("bound_at_maximum" if envelope else "bound_at_minimum")
            assert _one(bounds, source, 0).state == "informative"  # the macro average still has the other classes

    def test_sequence_mean_uses_one_item_per_group(self):
        rng = np.random.default_rng(3)
        x, y, _, seq, _ = make_sequences(make_population("gauss2"), 600, max_len=8, rng=rng)
        units = make_units(y, groups=seq, n_classes=2)
        spec = TargetSpec(task="IC", metric="accuracy", threshold=0.8, aggregation="sequence_mean")
        bounds, _, audit = _gather([spec], units, x)
        audit_groups = len(np.unique(units.groups[audit]))
        local = _one(bounds, "local_panels")
        assert (local.unit, local.sample, local.n) == ("sequence", "audit_split_one_per_group", audit_groups)
        for source in WITNESSES:
            assert _one(bounds, source).n == audit_groups

    def test_ungrouped_accuracy_witness_uses_exact_binomial(self):
        x, y, _, _ = _gauss2(2000, 0)
        bounds, _, _ = _gather([_acc()], make_units(y, n_classes=2), x)
        for source in WITNESSES:
            b = _one(bounds, source)
            assert (b.method, b.support, b.radius) == ("exact_binomial", 1.0, None)  # an exact interval has no radius
            assert b.observed is not None
            expected = binomial_endpoint(round(b.observed * b.n), b.n, b.delta)["endpoint"]  # type: ignore[arg-type]
            assert b.bound == pytest.approx(expected)

    def test_grouped_witness_keeps_hybrid_and_a_radius(self):
        rng = np.random.default_rng(3)
        x, y, _, seq, _ = make_sequences(make_population("gauss2"), 600, max_len=8, rng=rng)
        bounds, _, _ = _gather([_acc("sequence_mean")], make_units(y, groups=seq, n_classes=2), x)
        for source in WITNESSES:
            b = _one(bounds, source)
            assert b.method == "hybrid"
            assert b.radius is not None
            assert b.observed is not None
            assert b.bound is not None
            assert b.observed - b.bound == pytest.approx(b.radius)  # unclipped: the bound is the mean less the radius

    def test_frame_pooled_witness_keeps_its_ratio_when_ungrouped(self):
        x, y, _, _ = _gauss2(400, 0)
        bounds, _, _ = _gather([_acc("frame_pooled")], make_units(y, n_classes=2), x, max_sequence_length=4)
        for source in WITNESSES:
            b = _one(bounds, source)
            assert (b.state, b.method, b.radius, b.n) == ("informative", None, None, None)  # a ratio: no closed form

    def test_grouped_cell_count_follows_audit_groups(self, monkeypatch):
        planned = []
        monkeypatch.setattr(fc, "n_cells_for", lambda n: planned.append(n) or 10)
        rng = np.random.default_rng(3)
        x, y, _, seq, _ = make_sequences(make_population("gauss2"), 600, max_len=8, rng=rng)
        units = make_units(y, groups=seq, n_classes=2)
        _, _, audit = _gather([_acc("sequence_mean")], units, x, builtin=False)
        assert planned == [len(np.unique(units.groups[audit]))]  # audit groups, not audit items
        assert planned[0] < len(audit)

    def test_frame_pooled_has_no_local_panel_source(self):
        rng = np.random.default_rng(4)
        x, y, post, seq, lengths = make_sequences(make_population("gauss2"), 300, max_len=8, rng=rng)
        units = make_units(y, groups=seq, n_classes=2, class_names=["a", "b"])
        specs = [
            TargetSpec(task="IC", metric="accuracy", threshold=0.8, aggregation="frame_pooled"),
            _ap("b", "frame_pooled"),
        ]
        replicas = dict(enumerate(draw_labels(post, 2, rng)))
        assert count_claims(specs, has_replicas=True, has_predictions=False, builtin_witnesses=True) == 2 * 3
        undeclared, _, _ = _gather(specs, units, x, replicas=replicas)
        assert {b.source for b in undeclared} == {"replicas", *WITNESSES}
        assert all(b.state == "unavailable" and b.reason == "max_sequence_length_missing" for b in undeclared)
        declared, _, _ = _gather(specs, units, x, replicas=replicas, max_sequence_length=8)
        assert {b.source for b in declared} == {"replicas", *WITNESSES}
        assert len(declared) == 2 * 3
        assert all(b.state == "informative" for b in declared)
        assert all(b.method is None for b in declared)  # ratio and AP: no closed form
        assert "replicated_frames_uniform_sample" in _one(declared, "replicas", 0).assumptions
        with pytest.raises(ValueError, match="max_sequence_length"):
            _gather(specs, units, x, replicas=replicas, max_sequence_length=int(lengths.max()) - 1)

    def test_builtin_witnesses_never_see_audit_items(self, monkeypatch):
        seen = []

        class SpyKNN(KNeighborsClassifier):
            def fit(self, X, y):
                seen.append((np.asarray(X), np.asarray(y)))
                return super().fit(X, y)

        class SpyProbe(LogisticRegression):
            def fit(self, X, y, sample_weight=None):
                seen.append((np.asarray(X), np.asarray(y)))
                return super().fit(X, y, sample_weight)

        monkeypatch.setattr(fc, "KNeighborsClassifier", SpyKNN)
        monkeypatch.setattr(fc, "LogisticRegression", SpyProbe)
        x, y, post, _ = _gauss2(1000, 5)
        units = make_units(y)
        bounds, fit, audit = _gather([_acc()], units, x)
        assert len(seen) == 2
        for features, labels in seen:
            np.testing.assert_array_equal(features, x[fit])
            np.testing.assert_array_equal(labels, y[fit])
        assert {b.information_scope for source in WITNESSES for b in _of(bounds, source)} == {"fixed_representation"}

    def test_user_witness_counts_at_full_input_only(self):
        x, y, post, _ = _gauss2(1000, 5)
        units = make_units(y)
        cases: list[tuple[Scope, str]] = [
            ("full_input", "demonstrated_feasible"),
            ("fixed_representation", "unresolved"),
        ]
        for scope, expected in cases:
            spec = TargetSpec(task="IC", metric="accuracy", threshold=0.5, scope=scope)
            bounds, _, _ = _gather([spec], units, x, builtin=False, predictions=post)
            assert _one(bounds, "user_model").information_scope == "full_input"
            assert decide(spec, 0, bounds).determination == expected

    def test_every_source_ledger_within_alpha_source(self):
        x, y, post, rng = _gauss2(2000, 6)
        units = make_units(y, class_names=["a", "b"])
        replicas = {i: draw_labels(post[i : i + 1], 2 + i % 2, rng)[0] for i in range(500)}  # mixed panel sizes
        specs = [_acc(), TargetSpec(task="IC", metric="accuracy", threshold=0.7), _ap("a"), _ap("b"), _ap()]
        n_claims = count_claims(specs, has_replicas=True, has_predictions=True, builtin_witnesses=True)
        assert n_claims == len(specs) * (1 + 1 + 2 + 1)
        bounds, _, _ = _gather(specs, units, x, replicas=replicas, predictions=post)
        again, _, _ = _gather(specs, units, x, replicas=replicas, predictions=post)
        assert len(bounds) == n_claims
        assert all(b.alpha == allocate(0.05, n_claims) for b in bounds)
        check_ledger(bounds, 0.05)
        assert {b.source for b in bounds} == {"local_panels", "replicas", *WITNESSES, "user_model"}
        assert {b.population for b in bounds} == {"items"}
        assert [b.bound for b in again] == [b.bound for b in bounds]

    def test_the_bounds_come_in_table_order(self):
        x, y, post, rng = _gauss2(1000, 6)
        replicas = dict(enumerate(draw_labels(post[:300], 2, rng)))
        specs = [_acc(), _ap("a")]
        bounds, _, _ = _gather(specs, make_units(y, class_names=["a", "b"]), x, replicas=replicas, predictions=post)
        order = [(b.requirement, b.source) for b in bounds]
        sources = ("local_panels", "replicas", "knn", "linear_probe", "user_model")
        assert order == [(r, s) for r in range(2) for s in sources]

    def test_an_endpoint_split_above_its_claim_raises(self):
        specs = (_acc(),)
        call = fc._Call(make_units([0, 1, 0, 1]), np.zeros((4, 2)), np.arange(4), np.arange(0), 0.01, None, specs)

        def over(_spec):
            return 0.5, None, {"deltas": [0.006, 0.006]}

        with pytest.raises(RuntimeError, match="over its claim"):
            call.item(
                "user_model", "witness", "full_input", specs, over, sample="audit_split", assumptions=lambda _: ()
            )

    @pytest.mark.parametrize(("aggregation", "max_len"), [("sequence_mean", None), ("frame_pooled", 6)])
    def test_group_rounding_stays_within_support(self, aggregation, max_len):
        # One six-frame sequence of 2-1 splits: every frame's consensus is consensus_max(2, 3), and the group's mean
        # (or full-length total) exceeds its support by an ulp. It must be absorbed, not crash the call.
        rng = np.random.default_rng(0)
        groups = np.repeat(np.arange(40), 6)
        y = rng.integers(0, 2, size=len(groups))
        x = rng.standard_normal((len(groups), 4))
        units = make_units(y, groups=groups, n_classes=2)
        replicas = {i: np.array([1 - y[i], y[i]] if groups[i] == 0 else [y[i], y[i]]) for i in range(len(groups))}
        spec = TargetSpec(task="IC", metric="accuracy", threshold=0.8, aggregation=aggregation)
        bounds, _, _ = _gather([spec], units, x, builtin=False, replicas=replicas, max_sequence_length=max_len)
        assert _one(bounds, "replicas").state != "unavailable"  # computed, not crashed
        with pytest.raises(RuntimeError, match="support"):
            fc._mean(np.array([0.5, 1.0 + 1e-6]), 1.0, 0.05, "lower")  # beyond rounding: a real bug

    def test_video_local_panels_stay_partition_restricted(self):
        rng = np.random.default_rng(7)
        x, y, post, seq, _ = make_sequences(make_population("gauss2"), 600, max_len=8, rng=rng)
        units = make_units(y, groups=seq, n_classes=2)
        spec = TargetSpec(task="IC", metric="accuracy", threshold=0.8, aggregation="sequence_mean")
        replicas = dict(enumerate(draw_labels(post, 2, rng)))
        bare, _, _ = _gather([spec], units, x, builtin=False, replicas=replicas)  # same claims, same alpha
        kw = {"allowances": Allowances(beta=0.01, gamma=0.02), "replicas": replicas}
        allowed, _, _ = _gather([spec], units, x, builtin=False, **kw)
        assert {b.source for b in allowed} == {"local_panels", "replicas"}
        local = _one(allowed, "local_panels")
        assert local.information_scope == "partition_restricted"
        assert local.assumptions == ("representation_independent_of_audit_labels",)  # no allowance was applied
        assert _value(allowed, "local_panels") == _value(bare, "local_panels")
        assert set(_one(allowed, "replicas").assumptions) == {
            "replicas_random_sample",
            "replicas_one_protocol",
            "replicas_same_video_context",
            "replicated_sequences_random_sample",
        }

    def test_user_witness_input_is_validated(self):
        rng = np.random.default_rng(8)
        x, y, post, seq, _ = make_sequences(make_population("gauss2"), 300, min_len=2, max_len=6, rng=rng)
        units = make_units(y, groups=seq, n_classes=2)
        nan = post.copy()
        nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            _gather([_acc("sequence_mean")], units, x, builtin=False, predictions=nan)

    def test_recall_witness_has_lower_recall_and_upper_false_positive_bounds(self):
        x, y, decisions = _three_classes(600, 0)
        scores = 0.2 + 0.6 * np.eye(3)[decisions]  # scores whose top class is the decision
        bounds, _, audit = _gather([_recall()], make_units(y, class_names=NAMES), x, predictions=scores)
        half = allocate(0.05, 4) / 2  # local panels, two built-in witnesses and yours; recall and its limit share one
        for source in WITNESSES:  # the built-in witnesses decide by their own top class
            assert [b.quantity for b in _of(bounds, source)] == ["recall", "false_positive_rate"]
        rec, fp_rate = _of(bounds, "user_model")
        assert (rec.quantity, rec.bound_type, fp_rate.quantity, fp_rate.bound_type) == (
            "recall",
            "lower",
            "false_positive_rate",
            "upper",
        )
        assert rec.alpha == fp_rate.alpha == pytest.approx(half)
        k_pos, n_pos, k_neg, n_neg = _counts(decisions[audit], y[audit], 2)
        assert 0 < k_pos < n_pos
        assert 0 < k_neg < n_neg
        assert rec.bound == pytest.approx(beta.ppf(half, k_pos, n_pos - k_pos + 1))
        assert fp_rate.bound == pytest.approx(beta.ppf(1 - half, k_neg + 1, n_neg - k_neg))
        assert (rec.n, rec.method, rec.radius) == (n_pos, "exact_binomial", None)
        assert (fp_rate.n, fp_rate.observed) == (n_neg, pytest.approx(k_neg / n_neg))
        check_ledger(bounds, 0.05)

    def test_class_decisions_give_recall_false_positive_and_accuracy_counts(self):
        x, y, decisions = _three_classes(600, 1)
        bounds, _, audit = _gather(
            [_recall(), _acc()], make_units(y, class_names=NAMES), x, builtin=False, predictions=decisions
        )
        k_pos, n_pos, k_neg, n_neg = _counts(decisions[audit], y[audit], 2)
        rec = _one(bounds, "user_model", 0, "recall")
        assert (rec.observed, rec.n) == (pytest.approx(k_pos / n_pos), n_pos)
        fp_rate = _one(bounds, "user_model", 0, "false_positive_rate")
        assert (fp_rate.observed, fp_rate.n) == (pytest.approx(k_neg / n_neg), n_neg)
        assert _one(bounds, "user_model", 1).observed == pytest.approx((decisions[audit] == y[audit]).mean())

    @pytest.mark.parametrize(
        ("all_tank", "missing", "reason"),
        [
            (False, "recall", "no_positives"),  # Tank only in the fit split: no positives
            (True, "false_positive_rate", "no_negatives"),  # no negatives
        ],
    )
    def test_a_missing_side_of_the_audit_leaves_only_its_bound_unavailable(self, all_tank, missing, reason):
        x, y, decisions = _three_classes(600, 2)
        _, audit = split_units(make_units(y), fit_fraction=0.5, seed=0)  # the split ignores labels
        y[audit] = 2 if all_tank else np.where(y[audit] == 2, 0, y[audit])
        bounds, _, _ = _gather([_recall()], make_units(y, class_names=NAMES), x, builtin=False, predictions=decisions)
        for b in _of(bounds, "user_model"):
            if b.quantity == missing:
                assert (b.state, b.reason, b.bound, b.n) == ("unavailable", reason, None, None)
            else:
                assert b.bound is not None
                assert b.n is not None
        check_ledger(bounds, 0.05)

    def test_a_missing_recall_witness_gives_both_bounds_its_reason(self):
        spec = _recall()
        call = fc._Call(make_units([0, 1, 0, 1]), np.zeros((4, 2)), np.arange(4), np.arange(0), 0.01, None, (spec,))
        missing = fc._Missing("witness_not_fitted")
        bounds = call.item(
            "knn",
            "witness",
            "fixed_representation",
            [spec],
            missing.bound,
            sample="audit_split",
            assumptions=lambda _: (),
        )
        assert [(b.quantity, b.state, b.reason, b.alpha) for b in bounds] == [
            ("recall", "unavailable", "witness_not_fitted", 0.005),
            ("false_positive_rate", "unavailable", "witness_not_fitted", 0.005),
        ]

    def test_recall_envelopes_are_upper_bounds_with_no_closed_form(self):
        x, y, _ = _three_classes(1200, 4)
        rng = np.random.default_rng(4)
        replicas = {i: np.where(rng.random(2) < 0.9, y[i], rng.integers(0, 3, size=2)) for i in range(600)}
        bounds, _, _ = _gather([_recall()], make_units(y, class_names=NAMES), x, builtin=False, replicas=replicas)
        for source in ("local_panels", "replicas"):
            b = _one(bounds, source)
            assert (b.role, b.quantity, b.bound_type) == ("envelope", "recall", "upper")
            assert (b.method, b.mean, b.radius) == (None, None, None)  # no single mean to project: no minimum
            assert b.bound is not None
            assert 0 < b.bound <= 1
        assert _one(bounds, "local_panels").state == "informative"
        check_ledger(bounds, 0.05)

    def test_recall_ceiling_matches_the_exclusion_test(self):
        # U_r < r0 exactly when some cost's l_j exceeds the max over [pi_lo, pi_hi] of pi (1 - r0) + lam (1 - pi) f0
        rng = np.random.default_rng(0)
        outcomes = set()
        for _ in range(2000):
            k = int(rng.integers(1, 5))
            lam, risk = rng.uniform(0.1, 5.0, k), rng.uniform(0.0, 0.6, k)
            lo = 0.0 if rng.random() < 0.2 else rng.uniform(0.0, 0.5)
            pi = (lo, lo + rng.uniform(0.01, 0.5))
            r0, f0 = rng.uniform(0.05, 0.99), rng.uniform(0.01, 0.3)
            excluded = any(
                l_j > max(p * (1 - r0) + c * (1 - p) * f0 for p in pi) for l_j, c in zip(risk, lam, strict=True)
            )
            assert (fc._recall_upper(risk, lam, f0, *pi) < r0) == excluded
            outcomes.add(excluded)
        assert outcomes == {True, False}

    def test_recall_ceiling_uses_the_lower_prevalence_when_the_cost_is_large(self):
        # l < lam f0: g decreases in pi, so its maximum is at pi_lower: 1 - 0.2 - (0.15 - 0.2) / 0.4 (pi_upper: 0.8833)
        assert fc._recall_upper([0.15], [4.0], 0.05, 0.4, 0.6) == pytest.approx(0.925)

    def test_recall_ceiling_uses_the_upper_prevalence_when_the_cost_is_small(self):
        # l > lam f0: g increases in pi, so its maximum is at pi_upper: 1 - 0.0125 - 0.0875 / 0.5 (pi_lower: 0.6958)
        assert fc._recall_upper([0.1], [0.25], 0.05, 0.3, 0.5) == pytest.approx(0.8125)

    def test_equality_case_has_no_division_by_zero(self):
        assert fc._recall_upper([0.2], [4.0], 0.05, 0.0, 0.3) == pytest.approx(0.8)

    def test_zero_lower_prevalence_with_small_risk_is_uninformative(self):
        assert fc._recall_upper([0.01], [4.0], 0.05, 0.0, 0.3) == 1.0
        assert fc._recall_upper([], [], 0.05, 0.1, 0.3) == 1.0  # no positive cost

    def test_no_possible_positives_leave_the_recall_envelope_unavailable(self):
        with pytest.raises(fc._UnavailableError, match="no_positives_possible"):
            fc._recall_upper([0.3], [1.0], 0.05, 0.0, 0.0)

    def test_local_panel_recall_envelope_is_partition_restricted_beside_a_widened_accuracy_one(self):
        x, y, _ = _three_classes(600, 3)
        specs = [_recall(), _acc()]
        bounds, _, _ = _gather(
            specs, make_units(y, class_names=NAMES), x, builtin=False, allowances=Allowances(beta=0.0)
        )
        assert [(b.requirement, b.information_scope) for b in bounds] == [
            (0, "partition_restricted"),
            (1, "fixed_representation"),
        ]
        assert _one(bounds, "local_panels", 1).assumptions[-1] == "allowance_beta"
        check_ledger(bounds, 0.05)
