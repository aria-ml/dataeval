import dataclasses
from typing import Any

import numpy as np
import pytest

from dataeval.performance._feasibility_constants import EVIDENCE_SCHEMA
from dataeval.performance._feasibility_evidence import (
    Endpoint,
    Evidence,
    SamplingPlan,
    allocate,
    check_ledger,
    stats_key,
)

STATS = {"mean": 0.4, "variance": 0.01, "n": 10, "support": 1.0, "delta": 0.01, "method": "m"}


def _endpoint(spec="t1", alpha=0.01, state: Any = "informative", value: float | None = 0.5, **kw):
    return Endpoint(spec, value, "upper", alpha, state, kw.pop("reason", None), **kw)


def _evidence(endpoints=None, basis="local panel", details=None, **kw):
    endpoints = endpoints if endpoints is not None else (_endpoint(),)
    details = details if details is not None else {"endpoints": {e.target_spec: STATS for e in endpoints}}
    args = {
        "kind": "local_panel_ceiling/v1",
        "supports": "ceiling",
        "basis": basis,
        "population": "pop",
        "sample": "audit",
        "information_scope": "full_input",
        "unit": "group",
        "provenance": {"seed": 1},
        "endpoints": tuple(endpoints),
        "details": details,
    }
    return Evidence(**{**args, **kw})


@pytest.mark.required
class TestFeasibilityEvidence:
    def test_endpoint_state_rules(self):
        _endpoint(value=None, state="unavailable", reason="no data")
        with pytest.raises(ValueError, match="value"):
            _endpoint(value=None, state="informative")
        with pytest.raises(ValueError, match="value"):
            _endpoint(value=0.5, state="unavailable")
        for alpha in (0.0, 1.5):
            with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\]"):
                _endpoint(alpha=alpha)
        _endpoint(alpha=1.0)
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match="finite"):
                _endpoint(value=bad)
        with pytest.raises(ValueError, match="state"):
            _endpoint(state="bogus")
        with pytest.raises(ValueError, match="direction"):
            Endpoint("t", 0.5, "sideways", 0.01, "informative", None)  # type: ignore[arg-type]

    def test_empty_basis_rejected(self):
        with pytest.raises(ValueError, match="basis"):
            _evidence(basis="  ")

    def test_scope_and_direction_are_checked(self):
        with pytest.raises(ValueError, match="information_scope"):
            _evidence(information_scope="everywhere")
        with pytest.raises(ValueError, match="must be lower"):
            _evidence(supports="witness")  # the default endpoint is an upper one
        lower = Endpoint("t1", 0.5, "lower", 0.01, "informative")
        with pytest.raises(ValueError, match="must be upper"):
            _evidence(endpoints=(lower,))
        assert _evidence(supports="witness", endpoints=(lower,)).supports == "witness"

    def test_details_must_cover_every_endpoint(self):
        with pytest.raises(ValueError, match="t2"):
            _evidence(endpoints=(_endpoint("t1"), _endpoint("t2")), details={"endpoints": {"t1": None}})
        with pytest.raises(ValueError, match="endpoints"):
            _evidence(details={})

    def test_witness_may_carry_an_upper_false_positive_endpoint(self):
        recall = Endpoint("t1", 0.8, "lower", 0.01, "informative")
        fp_rate = Endpoint("t1", 0.04, "upper", 0.01, "informative", quantity="false_positive_rate")
        stats = {"t1": STATS, stats_key("t1", "false_positive_rate"): STATS}
        ev = _evidence(supports="witness", endpoints=(recall, fp_rate), details={"endpoints": stats})
        assert ev.endpoints[1].quantity == "false_positive_rate"
        assert ev.to_dict()["endpoints"][1]["quantity"] == "false_positive_rate"

    def test_stats_key_is_the_spec_for_the_metric_and_tagged_otherwise(self):
        assert stats_key("t1", "metric") == "t1"
        assert stats_key("t1", "false_positive_rate") == "t1#false_positive_rate"

    def test_witness_metric_endpoint_must_still_be_lower(self):
        with pytest.raises(ValueError, match="lower"):
            _evidence(supports="witness", endpoints=(_endpoint(),))

    def test_witness_false_positive_endpoint_must_be_upper(self):
        fp_rate = Endpoint("t1", 0.04, "lower", 0.01, "informative", quantity="false_positive_rate")
        with pytest.raises(ValueError, match="upper"):
            _evidence(
                supports="witness",
                endpoints=(fp_rate,),
                details={"endpoints": {stats_key("t1", "false_positive_rate"): STATS}},
            )

    def test_ceiling_cannot_carry_a_false_positive_endpoint(self):
        fp_rate = _endpoint(quantity="false_positive_rate")
        with pytest.raises(ValueError, match="false_positive_rate"):
            _evidence(endpoints=(fp_rate,), details={"endpoints": {stats_key("t1", "false_positive_rate"): STATS}})

    def test_false_positive_endpoint_needs_its_own_stats_entry(self):
        recall = Endpoint("t1", 0.8, "lower", 0.01, "informative")
        fp_rate = Endpoint("t1", 0.04, "upper", 0.01, "informative", quantity="false_positive_rate")
        with pytest.raises(ValueError, match="missing"):
            _evidence(supports="witness", endpoints=(recall, fp_rate), details={"endpoints": {"t1": STATS}})

    def test_unknown_quantity_is_rejected(self):
        with pytest.raises(ValueError, match="quantity"):
            _endpoint(quantity="bogus")

    def test_details_entries_need_the_full_stat_keys(self):
        _evidence(details={"endpoints": {"t1": None}})  # unavailable endpoint
        for bad in ({"mean": 0.4}, {**STATS, "extra": 1}, [1, 2]):
            with pytest.raises(ValueError, match="keys"):
                _evidence(details={"endpoints": {"t1": bad}})

    def test_allocate(self):
        assert allocate(0.1, 4) == pytest.approx(0.025)
        with pytest.raises(ValueError, match="n_claims"):
            allocate(0.1, 0)
        with pytest.raises(ValueError, match="alpha"):
            allocate(1.0, 2)

    def test_ledger_over_budget_raises(self):
        ok = [_evidence(endpoints=(_endpoint(alpha=0.02), _endpoint("t2", alpha=0.02))), _evidence()]
        check_ledger(ok, 0.05)
        with pytest.raises(RuntimeError, match="local_panel_ceiling/v1"):
            check_ledger(ok, 0.04)

    def test_look_schedule_sums_below_alpha(self):
        plan = SamplingPlan(sizes=(10, 20), order_seed=0)
        total = sum(plan.look_alpha(0.05, t) for t in range(1, 1001))
        assert total < 0.05
        assert plan.look_alpha(0.05, 1) == pytest.approx(0.025)

    def test_look_outside_plan_rejected(self):
        plan = SamplingPlan(sizes=(10, 20), order_seed=0)
        assert plan.prefix(25, 2) == 20
        with pytest.raises(ValueError, match="look"):
            plan.prefix(25, 3)
        with pytest.raises(ValueError, match="look"):
            plan.prefix(25, 0)
        with pytest.raises(ValueError, match="units"):
            plan.prefix(15, 2)
        with pytest.raises(ValueError, match="look"):
            plan.look_alpha(0.05, 0)
        for sizes in ((), (10, 10), (0, 5), (5, 3)):
            with pytest.raises(ValueError, match="sizes"):
                SamplingPlan(sizes=sizes, order_seed=0)

    def test_evidence_to_dict_round_trips_spine_keys(self):
        raw = np.arange(5)
        e = _evidence(details={"endpoints": {"t1": STATS}, "raw": raw})
        d = e.to_dict()
        assert d["schema"] == EVIDENCE_SCHEMA
        spine = {f.name for f in dataclasses.fields(Evidence)}
        assert spine <= set(d)
        assert d["endpoints"][0]["target_spec"] == "t1"
        assert d["details"]["raw"] is raw  # passed by reference, not copied
        assert d["provenance"] == {"seed": 1}
