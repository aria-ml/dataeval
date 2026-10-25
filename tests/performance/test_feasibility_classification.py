"""Tests for the classification feasibility compositions: every evidence source on synthetic populations."""

import dataclasses
import math
from statistics import NormalDist

import numpy as np
import pytest
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from dataeval.core import binomial_endpoint, consensus_max, kink_costs
from dataeval.performance import _feasibility_classification as fc
from dataeval.performance._feasibility_classification import count_claims, gather_evidence
from dataeval.performance._feasibility_evidence import SamplingPlan, allocate, check_ledger, stats_key
from dataeval.performance._feasibility_target import TargetSpec
from dataeval.performance._feasibility_units import split_units
from tests.performance.feasibility_helpers import draw_labels, make_population, make_sequences, make_units

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

LOCAL = "local_panel_ceiling/v1"
REPLICA = "replica_ceiling/v1"
USER = "user_witness/v1"
WITNESSES = ("builtin_witness/v1:knn", "builtin_witness/v1:probe")
NAMES = ["a", "b", "Tank"]  # the target class's index (2) is not 1, so a boolean mix-up would show


def _acc(aggregation=None) -> TargetSpec:  # built per test, so collection constructs no experimental class
    return TargetSpec("IC", "accuracy", 0.8, aggregation=aggregation)


def _ap(cls: str | None = None, aggregation=None) -> TargetSpec:
    return TargetSpec("IC", "ap", 0.5, class_name=cls, ap_functional="population_ap_integral", aggregation=aggregation)


def _recall(cls: str = "Tank", f0: float = 0.05) -> TargetSpec:
    return TargetSpec("IC", "recall", 0.9, class_name=cls, max_false_positive_rate=f0)


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


def _gather(specs, units, features, *, alpha=0.05, seed=0, builtin=True, replicas=None, predictions=None, **kw):
    fit, audit = split_units(units, fit_fraction=0.5, seed=seed)
    n_claims = count_claims(
        specs, has_replicas=bool(replicas), has_predictions=predictions is not None, builtin_witnesses=builtin
    )
    kw = {"provenance": None, "allowances": None, "sampling": None, "look": 1, "max_sequence_length": None} | kw
    evidence = gather_evidence(
        specs,
        units=units,
        features=features,
        audit=kw.pop("audit", audit),
        fit=fit,
        replicas=replicas,
        predictions=predictions,
        alpha_claim=allocate(alpha, n_claims),
        seed=seed,
        builtin_witnesses=builtin,
        **kw,
    )
    return evidence, fit, audit


def _endpoint(item, spec: TargetSpec):
    return next(e for e in item.endpoints if e.target_spec == spec.canonical())


def _value(item, spec: TargetSpec) -> float:
    value = _endpoint(item, spec).value
    assert value is not None
    return value


@pytest.mark.required
class TestFeasibilityClassification:
    def test_replica_ceiling_covers_bayes_on_gauss2(self):
        x, y, post, rng = _gauss2(2000, 0)
        extra = draw_labels(post, 4, rng)
        ev, _, _ = _gather([_acc()], make_units(y), x, builtin=False, replicas=dict(enumerate(extra)))
        item = ev[REPLICA]
        ep = _endpoint(item, _acc())
        bayes = NormalDist().cdf(1.0)
        assert (item.supports, item.information_scope, item.unit) == ("ceiling", "full_input", "item")
        assert (ep.direction, ep.state) == ("upper", "informative")
        assert bayes <= _value(item, _acc()) < bayes + 0.08
        stats = item.details["endpoints"][_acc().canonical()]
        assert (stats["n"], stats["method"]) == (2000, "hybrid")
        assert stats["support"] == pytest.approx(1 - 3 / 5)  # consensus_max(2, 5)
        assert item.details["cost_grid"] == "kinks"

    def test_unequal_replica_panels_use_hoeffding(self):
        x, y, post, rng = _gauss2(2000, 6)
        units = make_units(y, class_names=["a", "b"])
        alternating = {i: draw_labels(post[i : i + 1], 2 + i % 2, rng)[0] for i in range(500)}  # panels of 3 and 4
        equal = {i: draw_labels(post[i : i + 1], 2, rng)[0] for i in range(500)}
        for replicas, method in ((alternating, "hoeffding"), (equal, "hybrid")):
            ev, _, _ = _gather([_acc(), _ap("a")], units, x, builtin=False, replicas=replicas)
            item = ev[REPLICA]
            assert item.details["endpoints"][_acc().canonical()]["method"] == method
            risks = item.details["results"][_ap("a").canonical()]["risk_results"]
            assert {r["method"] for r in risks if r is not None} == {method}

    def test_local_panel_scope_follows_allowances(self):
        x, y, _, _ = _gauss2(2000, 1)
        units = make_units(y, class_names=["a", "b"])
        specs = [_acc(), _ap("b")]
        bare, _, _ = _gather(specs, units, x, builtin=False)
        beta, _, _ = _gather(specs, units, x, builtin=False, allowances={"beta": 0.01})
        both, _, _ = _gather(specs, units, x, builtin=False, allowances={"beta": 0.01, "gamma": 0.02})
        assert list(bare) == [LOCAL]
        assert bare[LOCAL].information_scope == "partition_restricted"
        acc_id, ap_id = f"{LOCAL}:accuracy", f"{LOCAL}:cost_weighted"
        assert list(beta) == list(both) == [acc_id, ap_id]
        assert [beta[i].information_scope for i in (acc_id, ap_id)] == ["fixed_representation", "partition_restricted"]
        assert [both[i].information_scope for i in (acc_id, ap_id)] == ["full_input", "partition_restricted"]
        assert [e.target_spec for e in beta[acc_id].endpoints] == [_acc().canonical()]
        # same seed, same panels: beta and gamma each lower the risk bound by their amount; AP ignores allowances
        acc = _value(bare[LOCAL], _acc())
        assert 0.5 < acc < 1 - 0.03
        assert _value(beta[acc_id], _acc()) == pytest.approx(acc + 0.01)
        assert _value(both[acc_id], _acc()) == pytest.approx(acc + 0.03)
        ap = _value(bare[LOCAL], _ap("b"))
        assert _value(beta[ap_id], _ap("b")) == _value(both[ap_id], _ap("b")) == ap
        assert bare[LOCAL].details["endpoints"][_acc().canonical()]["method"] == "hoeffding"
        assert bare[LOCAL].details["endpoints"][_ap("b").canonical()] is None
        assert bare[LOCAL].details["provisional_defaults"] == ["n_cells", "panel_size"]
        assert bare[LOCAL].details["cost_grid"] == "kinks"
        assert bare[LOCAL].basis.startswith("DataEval's post-stratified pooling (C3) of DFE disjoint within-cell")

    def test_absent_declared_class_keeps_the_macro_denominator(self):
        x, y, _, _ = _gauss2(2000, 2)  # labels 0 and 1 only; class "c" (index 2) is declared but absent
        units = make_units(y, n_classes=3, class_names=["a", "b", "c"])
        macro, absent = _ap(), _ap("c")
        ev, _, _ = _gather([macro, absent], units, x)
        for eid in (LOCAL, *WITNESSES):
            item = ev[eid]
            trivial = 1.0 if item.supports == "ceiling" else 0.0
            assert (_endpoint(item, absent).value, _endpoint(item, absent).state) == (trivial, "uninformative")
            per_class = item.details["results"][macro.canonical()]["per_class"]
            assert len(per_class) == 3
            assert per_class[2]["value"] == trivial
            values = [r["value"] for r in per_class]
            assert _value(item, macro) == pytest.approx(sum(values) / 3)
            assert _endpoint(item, macro).state == "informative"

    def test_sequence_mean_uses_one_item_per_group(self):
        rng = np.random.default_rng(3)
        x, y, _, seq, _ = make_sequences(make_population("gauss2"), 600, max_len=8, rng=rng)
        units = make_units(y, groups=seq, n_classes=2)
        spec = TargetSpec("IC", "accuracy", 0.8, aggregation="sequence_mean")
        ev, fit, audit = _gather([spec], units, x)
        audit_groups = np.unique(units.groups[audit])
        local = ev[LOCAL]
        items = np.asarray(local.details["panel_items"])
        assert np.isin(items, audit).all()
        np.testing.assert_array_equal(np.sort(units.groups[items]), audit_groups)  # each audit group exactly once
        assert local.unit == "group"
        assert local.details["results"][spec.canonical()]["n_units"] == len(audit_groups)
        for eid in WITNESSES:
            assert ev[eid].details["endpoints"][spec.canonical()]["n"] == len(audit_groups)

    def test_ungrouped_accuracy_witness_uses_exact_binomial(self):
        x, y, _, _ = _gauss2(2000, 0)
        ev, _, audit = _gather([_acc()], make_units(y, n_classes=2), x)
        for eid in WITNESSES:
            stats = ev[eid].details["endpoints"][_acc().canonical()]
            assert stats["method"] == "exact_binomial"
            assert stats["support"] == 1.0
            successes = round(stats["mean"] * stats["n"])
            expected = binomial_endpoint(successes, stats["n"], stats["delta"])["endpoint"]
            assert _value(ev[eid], _acc()) == pytest.approx(expected)

    def test_ungrouped_ap_witness_prevalence_is_the_exact_upper_endpoint(self):
        x, y, _, _ = _gauss2(2000, 0)
        units = make_units(y, n_classes=2, class_names=["a", "b"])
        ev, _, audit = _gather([_ap("a")], units, x, builtin=True)
        raw = ev[WITNESSES[0]].details["results"][_ap("a").canonical()]
        positives = int((units.labels[audit] == 0).sum())
        expected = binomial_endpoint(positives, len(audit), raw["deltas"][0], side="upper")["endpoint"]
        assert raw["mu_upper"] == pytest.approx(expected)

    def test_ungrouped_local_panel_prevalence_is_the_exact_upper_endpoint(self):
        x, y, _, _ = _gauss2(2000, 0)
        units = make_units(y, n_classes=2, class_names=["a", "b"])
        ev, _, _ = _gather([_ap("a")], units, x, builtin=False)
        local = ev[LOCAL]
        raw = local.details["results"][_ap("a").canonical()]
        items = np.asarray(local.details["panel_items"])
        expected = binomial_endpoint(
            int((units.labels[items] == 0).sum()), len(items), raw["deltas"][-1], side="upper"
        )["endpoint"]
        assert raw["mu_upper"] == pytest.approx(expected)

    def test_grouped_witness_keeps_hybrid(self):
        rng = np.random.default_rng(3)
        x, y, _, seq, _ = make_sequences(make_population("gauss2"), 600, max_len=8, rng=rng)
        spec = _acc("sequence_mean")
        ev, _, _ = _gather([spec], make_units(y, groups=seq, n_classes=2), x)
        for eid in WITNESSES:
            assert ev[eid].details["endpoints"][spec.canonical()]["method"] == "hybrid"

    def test_frame_pooled_witness_keeps_its_ratio_when_ungrouped(self):
        x, y, _, _ = _gauss2(400, 0)
        spec = _acc("frame_pooled")
        ev, _, _ = _gather([spec], make_units(y, n_classes=2), x, max_sequence_length=4, builtin=True)
        for eid in WITNESSES:
            assert "ratio_endpoint" in ev[eid].details["results"][spec.canonical()]

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
        specs = [TargetSpec("IC", "accuracy", 0.8, aggregation="frame_pooled"), _ap("b", "frame_pooled")]
        replicas = dict(enumerate(draw_labels(post, 2, rng)))
        assert count_claims(specs, has_replicas=True, has_predictions=False, builtin_witnesses=True) == 2 * 3
        undeclared, _, _ = _gather(specs, units, x, replicas=replicas)
        assert set(undeclared) == {REPLICA, *WITNESSES}
        for item in undeclared.values():
            assert all(e.state == "unavailable" and "max_sequence_length" in str(e.reason) for e in item.endpoints)
        declared, _, _ = _gather(specs, units, x, replicas=replicas, max_sequence_length=8)
        assert set(declared) == {REPLICA, *WITNESSES}
        assert sum(len(item.endpoints) for item in declared.values()) == 2 * 3
        for item in declared.values():
            assert all(e.state == "informative" for e in item.endpoints)
            assert all(stats is None for stats in item.details["endpoints"].values())  # ratio and AP: no closed form
        assert "uniform sample of all frames" in " ".join(declared[REPLICA].details["assumptions"])
        ratio = declared[REPLICA].details["results"][specs[0].canonical()]
        assert (ratio["num_support"], ratio["den_support"]) == (8 * consensus_max(2, 3), 8)  # L * vmax over L
        assert _value(declared[REPLICA], specs[0]) == 1 - ratio["ratio_endpoint"]
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
        ev, fit, audit = _gather([_acc()], units, x)
        assert len(seen) == 2
        for features, labels in seen:
            np.testing.assert_array_equal(features, x[fit])
            np.testing.assert_array_equal(labels, y[fit])
        assert all(ev[eid].provenance["reads"] == "representation" for eid in WITNESSES)
        assert all(ev[eid].information_scope == "fixed_representation" for eid in WITNESSES)
        # a user witness that declares an audit item among its training items is rejected
        leaked = {"training_items": [int(units.item_ids[audit[3]])]}
        with pytest.raises(ValueError, match="audit"):
            _gather([_acc()], units, x, builtin=False, predictions=post, provenance=leaked)
        leaked = {"training_items": {int(units.item_ids[audit[3]])}}  # any iterable, a set too
        with pytest.raises(ValueError, match="audit"):
            _gather([_acc()], units, x, builtin=False, predictions=post, provenance=leaked)
        named = dataclasses.replace(units, item_ids=np.array([f"d{i}" for i in range(len(y))]))
        with pytest.raises(ValueError, match="datum ids"):  # ints never match string ids
            _gather([_acc()], named, x, builtin=False, predictions=post, provenance={"training_items": [3]})
        with pytest.raises(ValueError, match="single id"):  # a bare string is one id, never its characters
            _gather([_acc()], named, x, builtin=False, predictions=post, provenance={"training_items": "d3"})
        clean, _, _ = _gather(
            [_acc()], units, x, builtin=False, predictions=post, provenance={"reads": "representation"}
        )
        user = clean["user_witness/v1"]
        assert (user.information_scope, user.provenance["reads"]) == ("fixed_representation", "representation")

    def test_every_source_ledger_within_alpha_source(self):
        x, y, post, rng = _gauss2(2000, 6)
        units = make_units(y, class_names=["a", "b"])
        replicas = {i: draw_labels(post[i : i + 1], 2 + i % 2, rng)[0] for i in range(500)}  # mixed panel sizes
        specs = [_acc(), TargetSpec("IC", "accuracy", 0.7), _ap("a"), _ap("b"), _ap()]
        n_claims = count_claims(specs, has_replicas=True, has_predictions=True, builtin_witnesses=True)
        assert n_claims == len(specs) * (1 + 1 + 2 + 1)
        plan = SamplingPlan((500, 1000), order_seed=0)
        ev, _, _ = _gather(specs, units, x, replicas=replicas, predictions=post, sampling=plan, look=2)
        again, _, _ = _gather(specs, units, x, replicas=replicas, predictions=post, sampling=plan, look=2)
        alpha_claim = allocate(0.05, n_claims)
        endpoints = [e for item in ev.values() for e in item.endpoints]
        assert len(endpoints) == n_claims
        assert all(e.alpha == alpha_claim for e in endpoints)
        check_ledger(ev.values(), 0.05)
        assert set(ev) == {LOCAL, REPLICA, *WITNESSES, "user_witness/v1"}
        for item in ev.values():
            assert item.population == units.population
            assert (item.details["sampling_plan"], item.details["look"]) == ((500, 1000), 2)
            for key, raw in item.details["results"].items():
                assert abs(math.fsum(raw["deltas"]) - alpha_claim) <= 1e-12, (item.kind, key)
        assert ev[LOCAL].details["n_cells"] == 10
        assert [e.value for item in again.values() for e in item.endpoints] == [e.value for e in endpoints]

    def test_an_endpoint_split_above_its_claim_raises(self):
        call = fc._Call(make_units([0, 1, 0, 1]), np.zeros((4, 2)), np.arange(4), np.arange(0), 0.01, None, {})

        def over(_spec):
            return 0.5, None, {"deltas": [0.006, 0.006]}

        with pytest.raises(RuntimeError, match="above its claim"):
            call.item(
                fc._USER, "witness", "full_input", [_acc()], over, sample="audit split", provenance={}, details={}
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
        spec = TargetSpec("IC", "accuracy", 0.8, aggregation=aggregation)
        ev, _, _ = _gather([spec], units, x, builtin=False, replicas=replicas, max_sequence_length=max_len)
        assert _endpoint(ev[REPLICA], spec).state != "unavailable"  # computed, not crashed
        with pytest.raises(RuntimeError, match="support"):
            fc._mean(np.array([0.5, 1.0 + 1e-6]), 1.0, 0.05, "lower")  # beyond rounding: a real bug

    def test_video_local_panels_stay_partition_restricted(self):
        rng = np.random.default_rng(7)
        x, y, post, seq, _ = make_sequences(make_population("gauss2"), 600, max_len=8, rng=rng)
        units = make_units(y, groups=seq, n_classes=2)
        spec = TargetSpec("IC", "accuracy", 0.8, aggregation="sequence_mean")
        replicas = dict(enumerate(draw_labels(post, 2, rng)))
        bare, _, _ = _gather([spec], units, x, builtin=False, replicas=replicas)  # same claims, same alpha
        plan = SamplingPlan((100, 10_000), order_seed=0)
        kw = {"allowances": {"beta": 0.01, "gamma": 0.02}, "replicas": replicas, "sampling": plan}
        allowed, _, _ = _gather([spec], units, x, builtin=False, **kw)
        assert set(allowed) == {LOCAL, REPLICA}
        assert allowed[LOCAL].information_scope == "partition_restricted"
        assert allowed[LOCAL].details["allowances_applied"] == {"beta": None, "gamma": 0.0}
        assert _value(allowed[LOCAL], spec) == _value(bare[LOCAL], spec)
        assumptions = " ".join(allowed[REPLICA].details["assumptions"])
        for phrase in ("video context", "random sample of sequences", "before the first look"):
            assert phrase in assumptions

    def test_user_witness_input_is_validated(self):
        rng = np.random.default_rng(8)
        x, y, post, seq, _ = make_sequences(make_population("gauss2"), 300, min_len=2, max_len=6, rng=rng)
        units = make_units(y, groups=seq, n_classes=2)
        fit, audit = split_units(units, fit_fraction=0.5, seed=0)
        # a training item outside the audit but in an audit item's physical group leaks that group
        held_out = audit[0]
        rest = audit[audit != held_out]
        assert units.groups[held_out] in units.groups[rest]
        leaked = {"training_items": [int(units.item_ids[held_out])]}
        with pytest.raises(ValueError, match="physical groups"):
            _gather([_acc("sequence_mean")], units, x, builtin=False, predictions=post, provenance=leaked, audit=rest)
        nan = post.copy()
        nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            _gather([_acc("sequence_mean")], units, x, builtin=False, predictions=nan)

    def test_recall_witness_has_lower_recall_and_upper_false_positive_endpoints(self):
        x, y, decisions = _three_classes(600, 0)
        scores = 0.2 + 0.6 * np.eye(3)[decisions]  # scores whose top class is the decision
        ev, _, audit = _gather([_recall()], make_units(y, class_names=NAMES), x, predictions=scores)
        half = allocate(0.05, 4) / 2  # local panels, two built-in witnesses and yours; recall and its limit share one
        for eid in WITNESSES:  # the built-in witnesses decide by their own top class
            assert [e.quantity for e in ev[eid].endpoints] == ["metric", "false_positive_rate"]
        rec, fp_rate = ev[USER].endpoints
        assert (rec.quantity, rec.direction, fp_rate.quantity, fp_rate.direction) == (
            "metric",
            "lower",
            "false_positive_rate",
            "upper",
        )
        assert rec.alpha == fp_rate.alpha == pytest.approx(half)
        k_pos, n_pos, k_neg, n_neg = _counts(decisions[audit], y[audit], 2)
        assert 0 < k_pos < n_pos
        assert 0 < k_neg < n_neg
        assert rec.value == pytest.approx(beta.ppf(half, k_pos, n_pos - k_pos + 1))
        assert fp_rate.value == pytest.approx(beta.ppf(1 - half, k_neg + 1, n_neg - k_neg))
        key = _recall().canonical()
        stats = ev[USER].details["endpoints"]
        assert (stats[key]["n"], stats[key]["method"]) == (n_pos, "exact_binomial")
        assert stats[stats_key(key, "false_positive_rate")]["n"] == n_neg
        raw = ev[USER].details["results"][key]
        assert raw["population"] == "audit items labeled 'Tank'"
        assert raw["false_positive_rate"]["population"] == "audit items labeled other than 'Tank'"
        assert raw["deltas"] == pytest.approx([half, half])
        check_ledger(ev.values(), 0.05)

    def test_class_decisions_give_recall_false_positive_and_accuracy_counts(self):
        x, y, decisions = _three_classes(600, 1)
        ev, _, audit = _gather(
            [_recall(), _acc()], make_units(y, class_names=NAMES), x, builtin=False, predictions=decisions
        )
        stats = ev[USER].details["endpoints"]
        key = _recall().canonical()
        k_pos, n_pos, k_neg, n_neg = _counts(decisions[audit], y[audit], 2)
        assert (stats[key]["mean"], stats[key]["n"]) == (pytest.approx(k_pos / n_pos), n_pos)
        fp_rate = stats[stats_key(key, "false_positive_rate")]
        assert (fp_rate["mean"], fp_rate["n"]) == (pytest.approx(k_neg / n_neg), n_neg)
        assert stats[_acc().canonical()]["mean"] == pytest.approx((decisions[audit] == y[audit]).mean())

    @pytest.mark.parametrize(
        ("all_tank", "missing", "reason"),
        [
            (False, "metric", "no audit items labeled 'Tank'"),  # Tank only in the fit split: no positives
            (True, "false_positive_rate", "no audit items labeled other than 'Tank'"),  # no negatives
        ],
    )
    def test_a_missing_side_of_the_audit_leaves_only_its_endpoint_unavailable(self, all_tank, missing, reason):
        x, y, decisions = _three_classes(600, 2)
        _, audit = split_units(make_units(y), fit_fraction=0.5, seed=0)  # the split ignores labels
        y[audit] = 2 if all_tank else np.where(y[audit] == 2, 0, y[audit])
        ev, _, _ = _gather([_recall()], make_units(y, class_names=NAMES), x, builtin=False, predictions=decisions)
        key = _recall().canonical()
        for ep in ev[USER].endpoints:
            stats = ev[USER].details["endpoints"][stats_key(key, ep.quantity)]
            if ep.quantity == missing:
                assert (ep.state, ep.reason, stats) == ("unavailable", reason, None)
            else:
                assert ep.value is not None
                assert stats is not None
        check_ledger(ev.values(), 0.05)

    def test_a_missing_recall_witness_gives_both_endpoints_its_reason(self):
        call = fc._Call(make_units([0, 1, 0, 1]), np.zeros((4, 2)), np.arange(4), np.arange(0), 0.01, None, {})
        spec, missing = _recall(), fc._Missing("no scorer")
        item = call.item(
            fc._BUILTIN, "witness", "fixed_representation", [spec], missing.bound, sample="s", provenance={}, details={}
        )
        assert [(e.quantity, e.state, e.reason, e.alpha) for e in item.endpoints] == [
            ("metric", "unavailable", "no scorer", 0.005),
            ("false_positive_rate", "unavailable", "no scorer", 0.005),
        ]
        assert set(item.details["endpoints"]) == {spec.canonical(), stats_key(spec.canonical(), "false_positive_rate")}

    def test_recall_ceilings_bound_from_cost_weighted_risks_and_two_prevalence_endpoints(self):
        x, y, _ = _three_classes(1200, 4)
        rng = np.random.default_rng(4)
        replicas = {i: np.where(rng.random(2) < 0.9, y[i], rng.integers(0, 3, size=2)) for i in range(600)}
        units = make_units(y, class_names=NAMES)
        ev, _, _ = _gather([_recall()], units, x, builtin=False, replicas=replicas)
        key = _recall().canonical()
        for eid, m in ((LOCAL, ev[LOCAL].details["panel_size"]), (REPLICA, 3)):
            (ep,) = ev[eid].endpoints
            raw = ev[eid].details["results"][key]
            assert (ep.quantity, ep.direction, ep.value) == ("metric", "upper", raw["value"])
            assert ev[eid].details["endpoints"][key] is None  # no single mean to project: no suggested minimum
            assert raw["costs"] == kink_costs(m)[1:].tolist()  # the positive kink costs
            n_endpoints = len(raw["costs"]) + 2  # and the prevalence's lower and upper endpoints
            assert raw["deltas"] == pytest.approx([ep.alpha / n_endpoints] * n_endpoints)
            assert 0 < raw["pi_lower"] < raw["pi_upper"] < 1
            bound = fc._recall_upper(raw["risk_lower"], raw["costs"], 0.05, raw["pi_lower"], raw["pi_upper"])
            assert raw["value"] == bound
        items = np.asarray(ev[LOCAL].details["panel_items"])
        k, n = int((units.labels[items] == 2).sum()), len(items)
        panel = ev[LOCAL].details["results"][key]
        for side in ("lower", "upper"):  # panel units' 0/1 labels: exact binomial
            expected = binomial_endpoint(k, n, panel["deltas"][-1], side=side)["endpoint"]
            assert panel[f"pi_{side}"] == pytest.approx(expected)
        share = np.array([(np.r_[y[i], replicas[i]] == 2).mean() for i in range(600)])
        replica = ev[REPLICA].details["results"][key]
        for side in ("lower", "upper"):  # per-item label shares: the replicas' own method
            assert replica[f"pi_{side}"] == fc._mean(share, 1.0, replica["deltas"][-1], side, "hybrid")["endpoint"]
        assert ev[LOCAL].endpoints[0].state == "informative"
        check_ledger(ev.values(), 0.05)

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

    def test_no_possible_positives_leave_the_recall_ceiling_unavailable(self):
        with pytest.raises(fc._UnavailableError, match="no positives are possible"):
            fc._recall_upper([0.3], [1.0], 0.05, 0.0, 0.0)

    def test_local_panel_recall_ceiling_is_partition_restricted_with_a_cost_weighted_id(self):
        x, y, _ = _three_classes(600, 3)
        specs = [_recall(), _acc()]
        ev, _, _ = _gather(specs, make_units(y, class_names=NAMES), x, builtin=False, allowances={"beta": 0.0})
        assert list(ev) == [f"{LOCAL}:accuracy", f"{LOCAL}:cost_weighted"]
        assert ev[f"{LOCAL}:accuracy"].information_scope == "fixed_representation"
        cost_weighted = ev[f"{LOCAL}:cost_weighted"]
        assert cost_weighted.information_scope == "partition_restricted"
        assert [e.target_spec for e in cost_weighted.endpoints] == [_recall().canonical()]
        check_ledger(ev.values(), 0.05)
