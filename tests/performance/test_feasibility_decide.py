"""Tests for the ordered decision rules, the limitation of a non-final determination and the suggested minimum."""

import math
from collections.abc import Sequence
from typing import Any, get_args

import pytest

from dataeval._experimental import quiet_experimental
from dataeval.core import units_to_cross
from dataeval.performance._feasibility_decide import Limitation, _pool, decide
from dataeval.performance._feasibility_evidence import Bound, Quantity, Role, Source
from dataeval.performance._feasibility_target import Scope, TargetSpec

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")


def _spec(scope: Scope = "full_input") -> TargetSpec:
    return TargetSpec(task="IC", metric="accuracy", threshold=0.8, scope=scope)


def _b(
    role: Role,
    value: float | None,
    *,
    scope: Scope = "full_input",
    source: Source | None = None,
    requirement: int = 0,
    state: Any = "informative",
    mean: float | None = None,
    reason: Any = None,
    support: float = 1.0,
    variance: float = 0.04,
    method: Any = "hybrid",
    quantity: Quantity = "accuracy",
) -> Bound:
    """One accuracy bound; the recorded mean defaults to the bound's value (no crossing)."""
    source = source or ("replicas" if role == "envelope" else "user_model")
    mean = value if mean is None else mean
    if value is None:
        state, mean, reason = "unavailable", None, reason or "too_few_units"
    elif state != "informative":
        reason = reason or ("bound_at_maximum" if role == "envelope" else "bound_at_minimum")
    has = value is not None
    return Bound(
        requirement=requirement,
        source=source,
        role=role,
        quantity=quantity,
        bound_type="upper" if role == "envelope" else "lower",
        information_scope=scope,
        population="items",
        sample="replica_items" if source == "replicas" else "audit_split",
        unit="image",
        state=state,
        reason=reason,
        bound=value,
        observed=mean if role == "witness" else None,
        n=100 if has else None,
        variance=variance if has else None,
        radius=None,
        method=method if has else None,
        alpha=0.01,
        assumptions=(),
        mean=mean,
        support=support if has else None,
        delta=0.01 if has else None,
    )


def _envelope(value: float | None, **kw: Any) -> Bound:
    return _b("envelope", value, **kw)


def _witness_bound(value: float | None, **kw: Any) -> Bound:
    return _b("witness", value, **kw)


def _panel(value: float | None, scope: Scope = "partition_restricted", **kw: Any) -> Bound:
    return _b("envelope", value, scope=scope, source="local_panels", **kw)


def _decide(spec: TargetSpec, bounds: Sequence[Bound], requirement: int = 0):
    return decide(spec, requirement, bounds)


with quiet_experimental():  # at import, outside the module's warning filter
    R90_F05 = TargetSpec(task="IC", metric="recall", threshold=0.9, class_name="Tank", max_false_positive_rate=0.05)
    R90_F10 = TargetSpec(task="IC", metric="recall", threshold=0.9, class_name="Tank", max_false_positive_rate=0.10)
HALF = 0.005  # recall and its false-positive limit share one claim's alpha


def _rate(source: Source, quantity: Quantity, bound: float, mean: float, n: int, requirement: int) -> Bound:
    return Bound(
        requirement=requirement,
        source=source,
        role="witness",
        quantity=quantity,
        bound_type="lower" if quantity == "recall" else "upper",
        information_scope="full_input",
        population="items",
        sample="audit_split",
        unit="image",
        state="informative",
        reason=None,
        bound=bound,
        observed=mean,
        n=n,
        variance=mean * (1 - mean),
        radius=None,
        method="exact_binomial",
        alpha=HALF,
        assumptions=(),
        mean=mean,
        support=1.0,
        delta=HALF,
    )


def _witness_for(
    specs: Sequence[TargetSpec],
    *,
    recall_lb: float,
    fpr_ub: float,
    recall_mean: float | None = None,
    fpr_mean: float | None = None,
    n_pos: int = 200,
    n_neg: int = 800,
    source: Source = "user_model",
) -> list[Bound]:
    """A witness with a recall lower and a false-positive upper bound per spec; means default to the bounds."""
    out: list[Bound] = []
    for r in range(len(specs)):
        recall = recall_lb if recall_mean is None else recall_mean
        fp = fpr_ub if fpr_mean is None else fpr_mean
        out += [
            _rate(source, "recall", recall_lb, recall, n_pos, r),
            _rate(source, "false_positive_rate", fpr_ub, fp, n_neg, r),
        ]
    return out


def _witness(**kwargs: Any) -> list[Bound]:
    return _witness_for([R90_F05], **kwargs)


def _ceiling(u_r: float, **kw: Any) -> Bound:
    """A full_input (replica) recall envelope at U_r."""
    return _envelope(u_r, quantity="recall", **kw)


# One case per limitation member: (name, spec, bounds, expected determination, expected limitation)
def _limitation_cases() -> list[tuple[str, TargetSpec, list[Bound], str, str]]:
    fixed, part = _spec("fixed_representation"), _spec("partition_restricted")
    return [
        ("conflict", _spec(), [_envelope(0.85), _witness_bound(0.9)], "unresolved", "conflict"),
        ("no_envelope", _spec(), [_witness_bound(0.5)], "unresolved", "no_envelope"),
        ("scope_restricted", _spec(), [_panel(0.6), _witness_bound(0.5)], "unresolved", "scope_restricted"),
        (
            "missing_bound",
            _spec(),
            [_envelope(None, reason="max_sequence_length_missing")],
            "unresolved",
            "missing_bound",
        ),
        (
            "envelope_unavailable",
            part,
            [_panel(None, reason="no_complete_panel"), _witness_bound(0.5, scope="full_input")],
            "unresolved",
            "envelope_unavailable",
        ),
        (
            "no_eligible_witness",
            fixed,
            [_envelope(0.9, scope="full_input"), _witness_bound(0.9)],
            "not_ruled_out",
            "no_eligible_witness",
        ),
        ("within_radius", _spec(), [_envelope(0.85, mean=0.75), _witness_bound(0.5)], "not_ruled_out", "within_radius"),
        ("weak_witness", _spec(), [_envelope(0.9), _witness_bound(0.5)], "not_ruled_out", "weak_witness"),
    ]


@pytest.mark.required
class TestFeasibilityDecide:
    @pytest.mark.parametrize("member", get_args(Limitation))  # the cases are built in the test, not at collection
    def test_each_limitation_member_is_produced(self, member):
        _, spec, bounds, determination, limitation = next(c for c in _limitation_cases() if c[4] == member)
        d = _decide(spec, bounds)
        assert (d.determination, d.limitation) == (determination, limitation)

    def test_every_limitation_member_has_a_case(self):
        assert {case[4] for case in _limitation_cases()} == set(get_args(Limitation))

    def test_pool_and_crossing_ignore_false_positive_bounds(self):
        fp_rate = _rate("user_model", "false_positive_rate", 0.9, 0.9, 100, 0)
        pool = _pool(_spec(), 0, [_witness_bound(0.6), fp_rate])
        assert [b.quantity for b in pool.items] == ["accuracy"]
        assert pool.witness is not None
        assert pool.witness.bound == 0.6

    def test_rule_1_conflict_is_unresolved_with_limitation_conflict(self):
        d = _decide(_spec(), [_envelope(0.85), _witness_bound(0.9)])
        assert (d.determination, d.limitation) == ("unresolved", "conflict")
        assert (d.witness, d.witness_source, d.envelope, d.envelope_scope) == (0.9, "user_model", 0.85, "full_input")
        assert d.determining == {("user_model", "accuracy"), ("replicas", "accuracy")}
        assert (d.suggested_minimum, d.minimum_source, d.minimum_quantity) == (None, None, None)

    def test_rule_2_demonstrated_feasible(self):
        bounds = [_envelope(0.95), _witness_bound(0.8), _witness_bound(0.6, source="knn")]
        d = _decide(_spec(), bounds)
        assert d.determination == "demonstrated_feasible"
        assert (d.witness, d.witness_source) == (0.8, "user_model")
        assert d.determining == {("user_model", "accuracy")}
        assert (d.limitation, d.suggested_minimum) == (None, None)

    def test_rule_3_infeasible(self):
        d = _decide(_spec(), [_witness_bound(0.6), _envelope(0.7, mean=0.65)])
        assert d.determination == "infeasible_under_recorded_assumptions"
        assert (d.envelope, d.envelope_scope) == (0.7, "full_input")
        assert d.determining == {("replicas", "accuracy")}
        assert (d.limitation, d.suggested_minimum) == (None, None)

    def test_rule_4_not_ruled_out(self):
        bounds = [
            _envelope(0.95),
            _panel(0.9, "full_input"),
            _witness_bound(0.5),
            _envelope(0.1, requirement=1),  # another requirement: ignored
        ]
        d = _decide(_spec(), bounds)
        assert d.determination == "not_ruled_out"
        assert (d.envelope, d.witness) == (0.9, 0.5)
        assert d.determining == {("local_panels", "accuracy")}
        assert d.limitation == "weak_witness"  # replicas informative, witness weak
        assert _decide(_spec(), [_panel(0.9, "full_input"), _witness_bound(0.5)]).limitation == "weak_witness"

    def test_envelope_at_threshold_is_not_ruled_out(self):
        assert _decide(_spec(), [_envelope(0.8)]).determination == "not_ruled_out"

    def test_full_input_envelope_decides_fixed_representation(self):
        spec = _spec("fixed_representation")
        d = _decide(spec, [_envelope(0.7)])
        assert (d.determination, d.envelope_scope) == ("infeasible_under_recorded_assumptions", "full_input")

    def test_rule_5_unresolved_when_only_uninformative(self):
        bounds = [_envelope(1.0, state="uninformative"), _witness_bound(0.0, state="uninformative")]
        d = _decide(_spec(), bounds)
        assert d.determination == "unresolved"
        assert d.limitation == "weak_witness"  # replica bounds exist
        assert d.determining == {("replicas", "accuracy"), ("user_model", "accuracy")}

    def test_a_witness_alone_is_demonstrated_or_has_no_envelope(self):
        for value, expected in ((0.9, ("demonstrated_feasible", None)), (0.5, ("unresolved", "no_envelope"))):
            d = _decide(_spec(), [_witness_bound(value)])
            assert (d.determination, d.limitation) == expected
        # only an envelope can bound what is unreachable
        assert _decide(_spec(), [_witness_bound(0.5)]).determination not in (
            "not_ruled_out",
            "infeasible_under_recorded_assumptions",
        )

    def test_witness_equal_to_envelope_is_not_a_conflict(self):
        assert _decide(_spec(), [_envelope(0.85), _witness_bound(0.85)]).determination == "demonstrated_feasible"

    def test_conflict_is_pairwise_by_envelope_scope(self):
        spec = _spec("fixed_representation")
        user = _witness_bound(0.9)
        d = _decide(spec, [_envelope(0.7), user])
        assert (d.determination, d.limitation, d.witness_source) == ("unresolved", "conflict", "user_model")
        # A full-image model is not in a fixed_representation envelope's class: no conflict, the envelope decides.
        d = _decide(spec, [_envelope(0.7, scope="fixed_representation"), user])
        assert d.determination == "infeasible_under_recorded_assumptions"

    def test_rules_2_and_3_imply_rule_1(self):
        for w in (0.8, 0.9, 1.0):
            for c in (0.0, 0.5, 0.79):
                d = _decide(_spec(), [_envelope(c), _witness_bound(w)])
                assert (d.determination, d.limitation) == ("unresolved", "conflict")

    def test_narrower_envelope_never_decides(self):
        d = _decide(_spec(), [_panel(0.5), _panel(0.6, "fixed_representation")])
        assert d.determination == "unresolved"
        assert d.envelope is None
        assert d.determining == frozenset()
        # The same envelope decides at its own scope, and a broader envelope decides a narrower requirement.
        spec = _spec("partition_restricted")
        assert _decide(spec, [_panel(0.5)]).determination.startswith("infeasible")
        assert _decide(spec, [_envelope(0.7)]).determination.startswith("infeasible")

    def test_full_image_model_cannot_demonstrate_fixed_representation(self):
        spec = _spec("fixed_representation")
        d = _decide(spec, [_envelope(0.9), _witness_bound(0.9)])
        assert (d.determination, d.limitation, d.witness) == ("not_ruled_out", "no_eligible_witness", None)
        builtin = _witness_bound(0.9, source="knn", scope="fixed_representation")  # reads only the representation
        assert _decide(spec, [builtin]).determination == "demonstrated_feasible"

    def test_partition_restricted_admits_no_witness(self):
        spec = _spec("partition_restricted")
        for source in ("knn", "user_model"):
            d = _decide(spec, [_panel(0.9), _witness_bound(0.9, source=source)])
            assert (d.determination, d.limitation, d.witness) == ("not_ruled_out", "no_eligible_witness", None)

    @pytest.mark.parametrize("scope", ["full_input", "partition_restricted"], ids=["narrower", "judged"])
    def test_envelope_unavailable_when_every_envelope_lacks_units(self, scope):
        spec = _spec(scope)
        starved = _panel(None, reason="no_complete_panel")
        d = _decide(spec, [starved, _witness_bound(0.5)])
        assert (d.determination, d.limitation, d.suggested_minimum) == ("unresolved", "envelope_unavailable", None)

    def test_missing_bound_names_the_bound(self):
        c = _envelope(None, reason="max_sequence_length_missing")
        d = _decide(_spec(), [c, _witness_bound(0.5)])
        assert (d.determination, d.limitation) == ("unresolved", "missing_bound")

    def test_no_eligible_witness_when_none_exists(self):
        d = _decide(_spec(), [_envelope(0.9)])
        assert (d.determination, d.limitation) == ("not_ruled_out", "no_eligible_witness")

    def test_within_radius_suggests_the_units_to_cross(self):
        n = units_to_cross(0.75, 0.04, 1.0, 0.01, 0.05)
        assert n is not None
        assert 100 < n < 1000
        d = _decide(_spec(), [_envelope(0.85, mean=0.75), _witness_bound(0.5)])
        assert (d.determination, d.limitation, d.suggested_minimum) == ("not_ruled_out", "within_radius", n)
        assert (d.minimum_source, d.minimum_quantity) == ("replicas", "accuracy")

    def test_within_radius_on_the_witness_side(self):
        d = _decide(_spec(), [_envelope(0.95), _witness_bound(0.75, mean=0.85)])
        assert (d.determination, d.limitation) == ("not_ruled_out", "within_radius")
        assert d.suggested_minimum == units_to_cross(0.85, 0.04, 1.0, 0.01, 0.05)
        assert (d.minimum_source, d.minimum_quantity) == ("user_model", "accuracy")

    def test_exact_binomial_witness_keeps_its_suggested_minimum(self):
        witness = _witness_bound(0.75, mean=0.85, method="exact_binomial")
        d = _decide(_spec(), [_envelope(0.95), witness])
        n = units_to_cross(0.85, 0.04, 1.0, 0.01, 0.05, method="exact_binomial", side="lower")
        assert n is not None
        assert (d.limitation, d.suggested_minimum) == ("within_radius", n)

    def test_suggested_minimum_never_raises_and_then_names_no_bound(self):
        bad = _envelope(0.85, mean=0.75, support=0.0)  # a support units_to_cross rejects
        d = _decide(_spec(), [bad, _witness_bound(0.5)])
        assert (d.limitation, d.suggested_minimum) == ("within_radius", None)
        assert (d.minimum_source, d.minimum_quantity) == (None, None)

    def test_replica_envelope_suggests_a_minimum(self):
        # Replica consensus lives in [0, consensus_max] (1/3 at two classes, three labels), below the accuracy-scale
        # mean; the minimum inverts the radius on the loss scale, 1 - mean.
        t9 = TargetSpec(task="IC", metric="accuracy", threshold=0.9)
        # A feasible variance: at most mean * (support - mean) = 0.114 * 0.219 on [0, 1/3].
        ceiling = _envelope(0.92, mean=0.886, support=1 / 3, variance=0.01)
        d = _decide(t9, [ceiling, _witness_bound(0.5)])
        n = units_to_cross(1 - 0.886, 0.01, 1 / 3, 0.01, 0.9 - 0.886)
        assert n is not None
        assert (d.determination, d.limitation, d.suggested_minimum) == ("not_ruled_out", "within_radius", n)

    def test_a_conflict_suggests_no_minimum(self):
        d = _decide(_spec(), [_envelope(0.85), _witness_bound(0.9)])
        assert (d.limitation, d.suggested_minimum) == ("conflict", None)


@pytest.mark.required
class TestRecallDecide:
    def test_recall_clears_but_false_positives_fail_gives_no_demonstration_or_conflict(self):
        bounds = [*_witness(recall_lb=0.95, fpr_ub=0.08), _ceiling(0.60)]
        d = _decide(R90_F05, bounds)
        assert d.determination != "demonstrated_feasible"
        assert d.limitation != "conflict"
        assert d.witness is None
        pool = _pool(R90_F05, 0, bounds)
        assert [b.source for b in pool.items if b.role == "witness"] == ["user_model"]  # still a bound, never a witness
        assert (pool.witnesses, pool.witness) == ([], None)

    def test_two_witnesses_each_with_one_favorable_bound_do_not_combine(self):
        bounds = [*_witness(recall_lb=0.95, fpr_ub=0.08), *_witness(recall_lb=0.50, fpr_ub=0.01, source="knn")]
        d = _decide(R90_F05, bounds)
        assert d.determination != "demonstrated_feasible"
        assert (d.witness, d.witness_source) == (0.50, "knn")

    def test_certified_witness_below_target_still_conflicts_with_a_lower_envelope(self):
        d = _decide(R90_F05, [*_witness(recall_lb=0.70, fpr_ub=0.03), _ceiling(0.60)])
        assert (d.determination, d.limitation, d.witness_source) == ("unresolved", "conflict", "user_model")

    def test_a_deciding_recall_witness_is_certified_by_its_false_positive_bound(self):
        d = _decide(R90_F05, _witness(recall_lb=0.95, fpr_ub=0.03))
        assert d.determination == "demonstrated_feasible"
        assert d.determining == {("user_model", "recall"), ("user_model", "false_positive_rate")}

    def test_two_limits_for_one_class_are_decided_independently(self):
        bounds = _witness_for([R90_F05, R90_F10], recall_lb=0.92, fpr_ub=0.07)
        assert decide(R90_F05, 0, bounds).determination != "demonstrated_feasible"
        d = decide(R90_F10, 1, bounds)
        assert (d.determination, d.witness_source) == ("demonstrated_feasible", "user_model")

    def test_never_predicting_the_class_is_a_weak_witness(self):
        w = _witness(recall_mean=0.0, recall_lb=0.0, fpr_mean=0.0, fpr_ub=0.004)
        d = _decide(R90_F05, [*w, _ceiling(0.95)])  # a replica envelope that does not exclude the requirement
        assert (d.determination, d.witness) == ("not_ruled_out", 0.0)
        assert (d.limitation, d.suggested_minimum) == ("weak_witness", None)

    def test_every_witness_over_the_false_positive_limit_is_a_weak_witness(self):
        w = _witness(recall_mean=0.97, recall_lb=0.93, fpr_mean=0.07, fpr_ub=0.09)
        d = _decide(R90_F05, [*w, _ceiling(0.95)])
        assert (d.determination, d.limitation, d.suggested_minimum) == ("not_ruled_out", "weak_witness", None)

    def test_false_positive_crossing_counts_images_through_prevalence(self):
        w = _witness(recall_mean=0.97, recall_lb=0.93, n_pos=200, fpr_mean=0.03, fpr_ub=0.06, n_neg=800)
        d = _decide(R90_F05, [*w, _ceiling(0.95)])
        negatives = units_to_cross(0.03, 0.03 * 0.97, 1.0, HALF, 0.02, method="exact_binomial", side="upper")
        assert negatives is not None
        assert (d.determination, d.limitation) == ("not_ruled_out", "within_radius")
        assert d.suggested_minimum == math.ceil(negatives / 0.8)  # 800 of 1000 audit items are negatives
        assert (d.minimum_source, d.minimum_quantity) == ("user_model", "false_positive_rate")

    def test_a_witness_alone_keeps_its_minimum_beside_no_envelope(self):
        w = _witness(recall_mean=0.97, recall_lb=0.93, n_pos=200, fpr_mean=0.03, fpr_ub=0.06, n_neg=800)
        d = _decide(R90_F05, w)
        assert (d.determination, d.limitation) == ("unresolved", "no_envelope")
        assert d.suggested_minimum is not None

    def test_both_crossings_take_the_larger_image_count_and_name_it(self):
        kwargs = {"recall_mean": 0.95, "recall_lb": 0.88, "fpr_mean": 0.03, "fpr_ub": 0.06}
        positives = units_to_cross(0.95, 0.0475, 1.0, HALF, 0.05, method="exact_binomial", side="lower")
        negatives = units_to_cross(0.03, 0.0291, 1.0, HALF, 0.02, method="exact_binomial", side="upper")
        assert positives is not None
        assert negatives is not None
        for_recall, for_fp = math.ceil(positives / 0.2), math.ceil(negatives / 0.8)
        d = _decide(R90_F05, [*_witness(**kwargs), _ceiling(0.95)])
        assert (d.limitation, d.suggested_minimum) == ("within_radius", max(for_recall, for_fp))
        assert d.minimum_quantity == ("recall" if for_recall >= for_fp else "false_positive_rate")

    def test_the_minimum_of_an_uncertified_witness_names_a_bounds_row(self):
        bounds = _witness(recall_mean=0.97, recall_lb=0.93, n_pos=200, fpr_mean=0.03, fpr_ub=0.06, n_neg=800)
        d = _decide(R90_F05, [*bounds, _ceiling(0.95)])
        assert d.witness_source is None  # uncertified: its false-positive bound is over the limit
        assert d.minimum_source == "user_model"
        rows = [b for b in bounds if (b.requirement, b.source, b.quantity) == (0, d.minimum_source, d.minimum_quantity)]
        assert len(rows) == 1

    def test_observed_recall_at_the_requirement_has_no_attainable_minimum(self):
        d = _decide(R90_F05, [*_witness(recall_mean=0.9, recall_lb=0.85, fpr_ub=0.03), _ceiling(0.95)])
        assert (d.limitation, d.suggested_minimum, d.minimum_source) == ("within_radius", None, None)
