import math
from collections.abc import Sequence
from typing import Any

import pytest

from dataeval._experimental import quiet_experimental
from dataeval.core import units_to_cross
from dataeval.performance._feasibility_decide import _pool, _short_bounds, decide
from dataeval.performance._feasibility_evidence import Endpoint, Evidence, SamplingPlan, stats_key
from dataeval.performance._feasibility_target import Scope, TargetSpec

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

P = "items of d"


def _spec(scope: Scope = "full_input") -> TargetSpec:
    return TargetSpec("IC", "accuracy", 0.8, scope=scope)


def _ev(
    supports: Any,
    value: float | None,
    *,
    scope: Scope = "full_input",
    kind: str | None = None,
    population: str = P,
    reads: str | None = None,
    state: Any = "informative",
    mean: float | None = None,
    reason: str | None = None,
    plan: Any = None,
    spec: TargetSpec | None = None,
    support: float = 1.0,
    variance: float = 0.04,
    method: str = "hybrid",
) -> Evidence:
    """One-endpoint evidence; the recorded mean defaults to the endpoint value (no crossing)."""
    key = (spec or _spec()).canonical()
    stats = None
    if value is not None:
        mean = value if mean is None else mean
        stats = {"mean": mean, "variance": variance, "n": 100, "support": support, "delta": 0.01, "method": method}
    details: dict[str, Any] = {"endpoints": {key: stats}}
    if plan is not None:
        details["sampling_plan"] = plan
    kind = kind or ("replica_ceiling/v1" if supports == "ceiling" else "user_witness/v1")
    return Evidence(
        kind=kind,
        supports=supports,
        basis="test basis",
        population=population,
        sample="100 items with replicas" if kind.startswith("replica") else "audit split",
        information_scope=scope,
        unit="item",
        provenance={} if reads is None else {"reads": reads},
        endpoints=(
            Endpoint(
                key,
                value,
                "upper" if supports == "ceiling" else "lower",
                0.01,
                "unavailable" if value is None else state,
                reason,
            ),
        ),
        details=details,
    )


def _panel(value: float, scope: Scope = "partition_restricted", spec: TargetSpec | None = None) -> Evidence:
    return _ev("ceiling", value, scope=scope, kind="local_panel_ceiling/v1", spec=spec)


def _decide(spec: TargetSpec, evidence: dict[str, Evidence], population: str = P):
    return decide(spec, evidence, population)


with quiet_experimental():  # at import, outside the module's warning filter
    R90_F05 = TargetSpec("IC", "recall", 0.9, class_name="Tank", max_false_positive_rate=0.05)
    R90_F10 = TargetSpec("IC", "recall", 0.9, class_name="Tank", max_false_positive_rate=0.10)
HALF = 0.005  # recall and its false-positive limit share one claim's alpha


def _rate_stats(mean: float, n: int) -> dict[str, Any]:
    return {
        "mean": mean,
        "variance": mean * (1 - mean),
        "n": n,
        "support": 1.0,
        "delta": HALF,
        "method": "exact_binomial",
    }


def _witness_for(
    specs: Sequence[TargetSpec],
    *,
    recall_lb: float,
    fpr_ub: float,
    recall_mean: float | None = None,
    fpr_mean: float | None = None,
    n_pos: int = 200,
    n_neg: int = 800,
    plan: Any = None,
) -> Evidence:
    """A user witness with a recall lower and a false-positive upper endpoint per spec; means default to the bounds."""
    endpoints, stats = [], {}
    for spec in specs:
        key = spec.canonical()
        endpoints += [
            Endpoint(key, recall_lb, "lower", HALF, "informative"),
            Endpoint(key, fpr_ub, "upper", HALF, "informative", quantity="false_positive_rate"),
        ]
        stats[key] = _rate_stats(recall_lb if recall_mean is None else recall_mean, n_pos)
        stats[stats_key(key, "false_positive_rate")] = _rate_stats(fpr_ub if fpr_mean is None else fpr_mean, n_neg)
    details: dict[str, Any] = {"endpoints": stats} | ({} if plan is None else {"sampling_plan": plan})
    return Evidence(
        kind="user_witness/v1",
        supports="witness",
        basis="test basis",
        population=P,
        sample="audit split",
        information_scope="full_input",
        unit="item",
        provenance={},
        endpoints=tuple(endpoints),
        details=details,
    )


def _witness(**kwargs: Any) -> Evidence:
    return _witness_for([R90_F05], **kwargs)


def _ceiling(u_r: float) -> Evidence:
    """A full_input (replica) recall ceiling at U_r."""
    return _ev("ceiling", u_r, spec=R90_F05)


@pytest.mark.required
class TestFeasibilityDecide:
    def test_pool_and_crossing_ignore_false_positive_endpoints(self):
        key = _spec().canonical()
        fpr_key = stats_key(key, "false_positive_rate")
        w = _ev("witness", 0.6)
        stats = w.details["endpoints"][key]
        fp_rate = Endpoint(key, 0.9, "upper", 0.01, "informative", quantity="false_positive_rate")
        w = Evidence(**{
            **{f: getattr(w, f) for f in w.__dataclass_fields__},
            "endpoints": (*w.endpoints, fp_rate),
            "details": {"endpoints": {key: stats, fpr_key: {**stats, "mean": 0.9}}},
        })
        pool = _pool(_spec(), {"w": w}, P)
        assert [i.ep.quantity for i in pool.items] == ["metric"]
        assert pool.witness is not None
        assert pool.witness.value == 0.6

    def test_rule_1_conflict_overrides_feasible(self):
        d = _decide(_spec(), {"c": _ev("ceiling", 0.85), "w": _ev("witness", 0.9)})
        assert d.determination == "evidence_conflict"
        assert (d.witness, d.witness_source, d.ceiling, d.ceiling_scope) == (0.9, "w", 0.85, "full_input")
        assert d.evidence_ids == ("w", "c")
        assert "user_witness/v1" in d.reason
        assert "replica_ceiling/v1" in d.reason

    def test_rule_2_demonstrated_feasible(self):
        d = _decide(_spec(), {"c": _ev("ceiling", 0.95), "w": _ev("witness", 0.8), "w2": _ev("witness", 0.6)})
        assert d.determination == "demonstrated_feasible"
        assert (d.witness, d.witness_source) == (0.8, "w")
        assert d.evidence_ids[0] == "w"
        assert set(d.evidence_ids) == {"c", "w", "w2"}
        assert (d.limiting_factor, d.recommendation, d.suggested_minimum) == (None, None, None)

    def test_rule_3_infeasible(self):
        d = _decide(_spec(), {"w": _ev("witness", 0.6), "c": _ev("ceiling", 0.7, mean=0.65)})
        assert d.determination == "infeasible_under_recorded_assumptions"
        assert (d.ceiling, d.ceiling_scope) == (0.7, "full_input")
        assert d.evidence_ids == ("c", "w")
        assert (d.limiting_factor, d.recommendation, d.suggested_minimum) == (None, None, None)

    def test_rule_4_not_ruled_out(self):
        other = TargetSpec("IC", "accuracy", 0.5, scope="fixed_representation")
        d = _decide(
            _spec(),
            {
                "c": _ev("ceiling", 0.95),
                "c2": _panel(0.9, "full_input"),
                "w": _ev("witness", 0.5),
                "x": _ev("ceiling", 0.1, spec=other),  # another target: ignored
            },
        )
        assert d.determination == "not_ruled_out"
        assert (d.ceiling, d.witness) == (0.9, 0.5)
        assert d.evidence_ids[0] == "c2"
        assert "x" not in d.evidence_ids
        assert d.limiting_factor == "no_witness"  # replicas informative, witness weak
        d = _decide(_spec(), {"c2": _panel(0.9, "full_input"), "w": _ev("witness", 0.5)})
        assert d.limiting_factor == "no_witness"  # an informative full_input local panel: a stronger witness helps

    def test_ceiling_at_threshold_is_not_ruled_out(self):
        assert _decide(_spec(), {"c": _ev("ceiling", 0.8)}).determination == "not_ruled_out"

    def test_full_input_ceiling_decides_fixed_representation(self):
        spec = _spec("fixed_representation")
        d = _decide(spec, {"c": _ev("ceiling", 0.7, spec=spec)})
        assert (d.determination, d.ceiling_scope) == ("infeasible_under_recorded_assumptions", "full_input")

    def test_rule_5_unresolved_when_only_uninformative(self):
        d = _decide(
            _spec(), {"c": _ev("ceiling", 1.0, state="uninformative"), "w": _ev("witness", 0.0, state="uninformative")}
        )
        assert d.determination == "unresolved"
        assert "uninformative" in d.reason
        assert d.limiting_factor == "no_witness"  # replica evidence exists

    def test_no_replicas_when_only_partition_ceilings(self):
        spec = _spec("partition_restricted")
        d = _decide(spec, {"p": _panel(0.9, spec=spec), "w": _ev("witness", 0.5, spec=spec, reads="partition")})
        assert (d.determination, d.limiting_factor) == ("not_ruled_out", "no_replicas")
        assert "repeated labels" in (d.recommendation or "")
        d = _decide(_spec(), {"w": _ev("witness", 0.5)})  # no ceiling at all, as for frame_pooled without replicas
        assert d.limiting_factor == "no_replicas"
        assert (d.recommendation or "").startswith("No ceiling broader than partition_restricted bounds this target")

    def test_witness_equal_to_ceiling_is_not_a_conflict(self):
        assert (
            _decide(_spec(), {"c": _ev("ceiling", 0.85), "w": _ev("witness", 0.85)}).determination
            == "demonstrated_feasible"
        )

    def test_conflict_is_pairwise_by_ceiling_scope(self):
        spec = _spec("fixed_representation")
        user = _ev("witness", 0.9, spec=spec, reads="full_input")
        d = _decide(spec, {"c": _ev("ceiling", 0.7, spec=spec), "w": user})
        assert (d.determination, d.witness_source) == ("evidence_conflict", "w")
        # A full-image model is not in a fixed_representation ceiling's class: no conflict, the ceiling decides.
        d = _decide(spec, {"c": _ev("ceiling", 0.7, spec=spec, scope="fixed_representation"), "w": user})
        assert d.determination == "infeasible_under_recorded_assumptions"

    def test_rules_2_and_3_imply_rule_1(self):
        for w in (0.8, 0.9, 1.0):
            for c in (0.0, 0.5, 0.79):
                assert (
                    _decide(_spec(), {"c": _ev("ceiling", c), "w": _ev("witness", w)}).determination
                    == "evidence_conflict"
                )

    def test_narrower_ceiling_never_decides(self):
        d = _decide(_spec(), {"p": _panel(0.5), "q": _panel(0.6, "fixed_representation")})
        assert d.determination == "unresolved"
        assert d.ceiling is None
        assert set(d.evidence_ids) == {"p", "q"}
        # The same ceiling decides at its own scope, and a broader ceiling decides a narrower target.
        spec = _spec("partition_restricted")
        assert _decide(spec, {"p": _panel(0.5, spec=spec)}).determination.startswith("infeasible")
        assert _decide(spec, {"c": _ev("ceiling", 0.7, spec=spec)}).determination.startswith("infeasible")

    def test_full_image_model_cannot_demonstrate_fixed_representation(self):
        spec = _spec("fixed_representation")
        user = _ev("witness", 0.9, spec=spec)
        d = _decide(spec, {"w": user})
        assert d.determination == "unresolved"
        assert d.witness is None
        assert d.evidence_ids == ("w",)
        assert d.limiting_factor == "no_witness"
        assert _decide(spec, {"w": _ev("witness", 0.9, spec=spec, reads="full_input")}).witness is None
        rep = _ev("witness", 0.9, spec=spec, reads="representation")
        assert _decide(spec, {"w": rep}).determination == "demonstrated_feasible"
        builtin = _ev("witness", 0.9, spec=spec, kind="builtin_witness/v1")
        assert _decide(spec, {"w": builtin}).determination == "demonstrated_feasible"

    def test_partition_restricted_admits_only_partition_witnesses(self):
        spec = _spec("partition_restricted")
        builtin = _ev("witness", 0.9, spec=spec, kind="builtin_witness/v1")
        d = _decide(spec, {"w": builtin})
        assert (d.determination, d.limiting_factor) == ("unresolved", "no_witness")
        assert "Only a ceiling can decide" in (d.recommendation or "")  # evaluate rejects reads="partition"
        assert _decide(spec, {"w": _ev("witness", 0.9, spec=spec)}).determination == "unresolved"
        part = _ev("witness", 0.9, spec=spec, reads="partition")
        assert _decide(spec, {"w": part}).determination == "demonstrated_feasible"
        part = _ev("witness", 0.9, reads="partition")
        assert (
            _decide(_spec(), {"w": part}).determination == "demonstrated_feasible"
        )  # a partition model reads the input
        rep = _spec("fixed_representation")
        part = _ev("witness", 0.9, spec=rep, reads="partition")
        assert _decide(rep, {"w": part}).determination == "demonstrated_feasible"  # it reads only the representation

    def test_population_mismatch_is_not_compared(self):
        evidence = {"c": _ev("ceiling", 0.7, population="items of e"), "w": _ev("witness", 0.9)}
        d = _decide(_spec(), evidence, "items of d")
        assert (d.determination, d.ceiling, d.evidence_ids) == ("demonstrated_feasible", None, ("w",))
        d = _decide(_spec(), evidence, "items of e")
        assert (d.determination, d.witness, d.evidence_ids) == ("infeasible_under_recorded_assumptions", None, ("c",))

    def test_recommendation_names_replicas_when_scope_restricted(self):
        d = _decide(_spec(), {"p": _panel(0.6), "w": _ev("witness", 0.5)})
        assert d.limiting_factor == "scope_restricted"
        assert "repeated labels" in (d.recommendation or "")
        assert "partition_restricted" in (d.recommendation or "")
        assert "(p)." in (d.recommendation or "")  # the evidence id, once
        assert "local_panel_ceiling" not in (d.recommendation or "")

    @pytest.mark.parametrize("scope", ["full_input", "partition_restricted"], ids=["narrower", "judged"])
    def test_sampling_width_when_every_ceiling_lacks_units(self, scope):
        spec = _spec(scope)
        reason = "no partition cell holds 5 audit units, so no panel is complete"
        starved = _ev(
            "ceiling", None, scope="partition_restricted", kind="local_panel_ceiling/v1", reason=reason, spec=spec
        )
        d = _decide(spec, {"p": starved, "w": _ev("witness", 0.5, spec=spec)})
        assert (d.determination, d.limiting_factor) == ("unresolved", "sampling_width")
        assert reason in (d.recommendation or "")
        assert d.suggested_minimum is None

    def test_missing_bound_names_the_bound(self):
        c = _ev("ceiling", None, reason="max_sequence_length is not declared")
        d = _decide(_spec(), {"c": c, "w": _ev("witness", 0.5)})
        assert d.determination == "unresolved"
        assert d.limiting_factor == "missing_bound"
        assert "max_sequence_length" in (d.recommendation or "")

    def test_no_witness_when_none_exists(self):
        d = _decide(_spec(), {"c": _ev("ceiling", 0.9)})
        assert (d.determination, d.limiting_factor) == ("not_ruled_out", "no_witness")

    def test_labeled_items_only_adds_a_note_and_keeps_the_factor(self):
        pop, note = "labeled items of d", "label the excluded items to extend the result to the full dataset."
        weak = _ev("witness", 0.5, population=pop)
        d = _decide(_spec(), {"c": _ev("ceiling", 0.9, population=pop), "w": weak}, pop)
        assert (d.determination, d.limiting_factor) == ("not_ruled_out", "no_witness")
        assert "stronger model" in (d.recommendation or "")
        assert (d.recommendation or "").endswith(note)
        d = _decide(_spec("partition_restricted"), {"w": weak}, pop)
        assert d.limiting_factor == "no_witness"
        assert (d.recommendation or "").endswith(note)

    def test_sampling_width_suggests_units_rounded_to_the_plan(self):
        n = units_to_cross(0.75, 0.04, 1.0, 0.01, 0.05)
        assert n is not None
        assert 100 < n < 1000
        witness = _ev("witness", 0.5)
        d = _decide(_spec(), {"c": _ev("ceiling", 0.85, mean=0.75), "w": witness})
        assert (d.determination, d.limiting_factor, d.suggested_minimum) == ("not_ruled_out", "sampling_width", n)
        assert str(n) in (d.recommendation or "")
        plan = SamplingPlan((100, 1000, 10_000), order_seed=0)
        local = {"kind": "local_panel_ceiling/v1", "mean": 0.75}  # an audit-split sample, at full_input
        d = _decide(_spec(), {"c": _ev("ceiling", 0.85, plan=plan, **local), "w": witness})
        assert d.suggested_minimum == 1000
        assert "alpha/(k(k+1))" in (d.recommendation or "")
        d = _decide(_spec(), {"c": _ev("ceiling", 0.85, plan=(100, 200), **local), "w": witness})
        assert (d.limiting_factor, d.suggested_minimum) == ("sampling_width", None)
        # Units with replicas are not plan units: the replica ceiling's minimum stays unrounded.
        d = _decide(_spec(), {"c": _ev("ceiling", 0.85, mean=0.75, plan=plan), "w": witness})
        assert d.suggested_minimum == n
        assert "alpha/(k(k+1))" not in (d.recommendation or "")

    def test_sampling_width_on_the_witness_side(self):
        d = _decide(_spec(), {"c": _ev("ceiling", 0.95), "w": _ev("witness", 0.75, mean=0.85)})
        assert (d.determination, d.limiting_factor) == ("not_ruled_out", "sampling_width")
        assert d.suggested_minimum == units_to_cross(0.85, 0.04, 1.0, 0.01, 0.05)

    def test_exact_binomial_witness_keeps_its_suggested_minimum(self):
        witness = _ev("witness", 0.75, mean=0.85, method="exact_binomial")
        d = _decide(_spec(), {"c": _ev("ceiling", 0.95), "w": witness})
        n = units_to_cross(0.85, 0.04, 1.0, 0.01, 0.05, method="exact_binomial", side="lower")
        assert n is not None
        assert (d.limiting_factor, d.suggested_minimum) == ("sampling_width", n)

    def test_suggested_minimum_never_raises(self):
        bad = _ev("ceiling", 0.85, mean=0.75, support=0.0)  # a support units_to_cross rejects
        d = _decide(_spec(), {"c": bad, "w": _ev("witness", 0.5)})
        assert (d.limiting_factor, d.suggested_minimum) == ("sampling_width", None)

    def test_replica_ceiling_suggests_a_minimum(self):
        # Replica consensus lives in [0, consensus_max] (1/3 at two classes, three labels), below the accuracy-scale
        # mean; the minimum inverts the radius on the loss scale, 1 - mean.
        t9 = TargetSpec("IC", "accuracy", 0.9)
        # A feasible variance: at most mean * (support - mean) = 0.114 * 0.219 on [0, 1/3].
        ceiling = _ev("ceiling", 0.92, mean=0.886, support=1 / 3, spec=t9, variance=0.01)
        d = _decide(t9, {"c": ceiling, "w": _ev("witness", 0.5, spec=t9)})
        n = units_to_cross(1 - 0.886, 0.01, 1 / 3, 0.01, 0.9 - 0.886)
        assert n is not None
        assert (d.determination, d.limiting_factor, d.suggested_minimum) == ("not_ruled_out", "sampling_width", n)

    def test_conflict_recommends_inspecting_its_causes(self):
        d = _decide(_spec(), {"c": _ev("ceiling", 0.85), "w": _ev("witness", 0.9)})
        assert (d.limiting_factor, d.suggested_minimum) == ("conflict", None)
        for cause in ("leakage", "duplicates", "scope"):
            assert cause in (d.recommendation or "")


@pytest.mark.required
class TestRecallDecide:
    def test_short_bounds_never_name_nothing(self):
        item = _pool(R90_F05, {"w": _witness(recall_lb=0.95, fpr_ub=0.04)}, P).witness  # neither bound is short
        assert item is not None
        assert _short_bounds(item, R90_F05) == "confidence bounds"

    def test_recall_clears_but_false_positives_fail_gives_no_demonstration_or_conflict(self):
        ev = {"w": _witness(recall_lb=0.95, fpr_ub=0.08), "c": _ceiling(0.60)}
        d = _decide(R90_F05, ev)
        assert d.determination not in ("demonstrated_feasible", "evidence_conflict")
        assert d.witness is None
        pool = _pool(R90_F05, ev, P)
        assert [i.eid for i in pool.items if i.ev.supports == "witness"] == ["w"]  # still evidence, never a witness
        assert (pool.witnesses, pool.witness) == ([], None)

    def test_two_witnesses_each_with_one_favorable_bound_do_not_combine(self):
        ev = {"a": _witness(recall_lb=0.95, fpr_ub=0.08), "b": _witness(recall_lb=0.50, fpr_ub=0.01)}
        d = _decide(R90_F05, ev)
        assert d.determination != "demonstrated_feasible"
        assert (d.witness, d.witness_source) == (0.50, "b")

    def test_certified_witness_below_target_still_conflicts_with_a_lower_ceiling(self):
        d = _decide(R90_F05, {"w": _witness(recall_lb=0.70, fpr_ub=0.03), "c": _ceiling(0.60)})
        assert (d.determination, d.witness_source) == ("evidence_conflict", "w")

    def test_two_limits_for_one_class_are_decided_independently(self):
        ev = {"w": _witness_for([R90_F05, R90_F10], recall_lb=0.92, fpr_ub=0.07)}
        assert _decide(R90_F05, ev).determination != "demonstrated_feasible"
        d = _decide(R90_F10, ev)
        assert (d.determination, d.witness_source) == ("demonstrated_feasible", "w")

    def test_never_predicting_the_class_is_no_witness(self):
        w = _witness(recall_mean=0.0, recall_lb=0.0, fpr_mean=0.0, fpr_ub=0.004)
        d = _decide(R90_F05, {"w": w, "c": _ceiling(0.95)})  # a replica ceiling that does not exclude the target
        assert (d.determination, d.witness) == ("not_ruled_out", 0.0)
        assert d.limiting_factor == "no_witness"
        assert d.suggested_minimum is None
        assert "supply a stronger model or a different frozen operating point" in (d.recommendation or "")

    def test_every_witness_over_the_false_positive_limit_is_no_witness(self):
        w = _witness(recall_mean=0.97, recall_lb=0.93, fpr_mean=0.07, fpr_ub=0.09)
        d = _decide(R90_F05, {"w": w, "c": _ceiling(0.95)})
        assert (d.determination, d.limiting_factor, d.suggested_minimum) == ("not_ruled_out", "no_witness", None)
        text = d.recommendation or ""
        assert "false-positive" in text
        assert "supply a stronger model or a different frozen operating point" in text

    def test_false_positive_crossing_counts_images_through_prevalence(self):
        w = _witness(recall_mean=0.97, recall_lb=0.93, n_pos=200, fpr_mean=0.03, fpr_ub=0.06, n_neg=800)
        d = _decide(R90_F05, {"w": w})
        negatives = units_to_cross(0.03, 0.03 * 0.97, 1.0, HALF, 0.02, method="exact_binomial", side="upper")
        assert negatives is not None
        assert (d.determination, d.limiting_factor) == ("unresolved", "sampling_width")
        assert d.suggested_minimum == math.ceil(negatives / 0.8)  # 800 of 1000 audit items are negatives
        assert "estimated prevalence" in (d.recommendation or "")

    def test_both_crossings_take_the_larger_image_count_rounded_to_the_plan(self):
        kwargs = {"recall_mean": 0.95, "recall_lb": 0.88, "fpr_mean": 0.03, "fpr_ub": 0.06}
        positives = units_to_cross(0.95, 0.0475, 1.0, HALF, 0.05, method="exact_binomial", side="lower")
        negatives = units_to_cross(0.03, 0.0291, 1.0, HALF, 0.02, method="exact_binomial", side="upper")
        assert positives is not None
        assert negatives is not None
        images = max(math.ceil(positives / 0.2), math.ceil(negatives / 0.8))
        assert _decide(R90_F05, {"w": _witness(**kwargs)}).suggested_minimum == images
        plan = SamplingPlan((1000, 10 * images, 100 * images), order_seed=0)
        d = _decide(R90_F05, {"w": _witness(**kwargs, plan=plan)})
        assert (d.limiting_factor, d.suggested_minimum) == ("sampling_width", 10 * images)

    def test_observed_recall_at_the_requirement_has_no_attainable_minimum(self):
        d = _decide(R90_F05, {"w": _witness(recall_mean=0.9, recall_lb=0.85, fpr_ub=0.03)})
        assert (d.limiting_factor, d.suggested_minimum) == ("sampling_width", None)
