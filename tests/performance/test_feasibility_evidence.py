"""Tests for the bound record, the Enums that describe it, and the alpha ledger."""

from typing import Any

import pytest

from dataeval.performance._feasibility_evidence import Bound, allocate, check_ledger


def _bound(**over: Any) -> Bound:
    """A witness's informative lower bound on accuracy; keywords replace fields."""
    fields: dict[str, Any] = {
        "requirement": 0,
        "source": "user_model",
        "role": "witness",
        "quantity": "accuracy",
        "bound_type": "lower",
        "information_scope": "full_input",
        "population": "items",
        "sample": "audit_split",
        "unit": "image",
        "state": "informative",
        "reason": None,
        "bound": 0.8,
        "observed": 0.85,
        "n": 100,
        "variance": 0.1,
        "radius": 0.05,
        "method": "hybrid",
        "alpha": 0.01,
        "assumptions": (),
    }
    return Bound(**{**fields, **over})


def _envelope(**over: Any) -> Bound:
    return _bound(role="envelope", bound_type="upper", bound=0.9, observed=None, source="replicas", **over)


@pytest.mark.required
class TestBound:
    def test_state_rules(self):
        _bound(state="unavailable", reason="too_few_units", bound=None, observed=None, n=None, variance=None)
        with pytest.raises(ValueError, match="iff state is 'unavailable'"):
            _bound(state="informative", bound=None)
        with pytest.raises(ValueError, match="iff state is 'unavailable'"):
            _bound(state="unavailable", reason="too_few_units")  # an unavailable bound has no value
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match="finite"):
                _bound(bound=bad)

    def test_a_reason_comes_with_every_state_but_informative(self):
        with pytest.raises(ValueError, match="reason"):
            _bound(reason="too_few_units")  # informative: nothing to explain
        with pytest.raises(ValueError, match="reason"):
            _bound(state="uninformative")  # uninformative needs its reason
        _bound(state="uninformative", reason="bound_at_minimum", bound=0.0)

    @pytest.mark.parametrize("alpha", [0.0, 1.5, -0.1])
    def test_alpha_is_a_share_in_the_unit_interval(self, alpha):
        with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\]"):
            _bound(alpha=alpha)
        _bound(alpha=1.0)

    def test_unknown_enum_values_are_rejected(self):
        for field, value in (
            ("source", "oracle"),
            ("role", "referee"),
            ("quantity", "f1"),
            ("bound_type", "both"),
            ("information_scope", "everywhere"),
            ("population", "all"),
            ("sample", "everything"),
            ("unit", "frame"),
            ("state", "bogus"),
            ("method", "bayes"),
        ):
            with pytest.raises(ValueError, match=field):
                _bound(**{field: value})
        with pytest.raises(ValueError, match="reason"):
            _bound(state="unavailable", reason="because", bound=None)
        with pytest.raises(ValueError, match="assumption"):
            _bound(assumptions=("trust_me",))

    def test_an_envelope_bounds_the_metric_from_above(self):
        _envelope()
        with pytest.raises(ValueError, match="must be upper"):
            _bound(role="envelope", observed=None)  # the default bound type is lower

    def test_a_witness_bounds_the_metric_from_below_and_false_positives_from_above(self):
        with pytest.raises(ValueError, match="must be lower"):
            _bound(bound_type="upper")
        _bound(quantity="false_positive_rate", bound_type="upper", bound=0.04, observed=0.03)
        with pytest.raises(ValueError, match="must be upper"):
            _bound(quantity="false_positive_rate", bound_type="lower")

    def test_an_envelope_cannot_bound_false_positives(self):
        with pytest.raises(ValueError, match="false_positive_rate"):
            _envelope(quantity="false_positive_rate")

    def test_observed_is_null_for_every_envelope(self):
        with pytest.raises(ValueError, match="observed"):
            _bound(role="envelope", bound_type="upper", observed=0.8)

    def test_assumptions_are_kept_once_in_the_declared_order(self):
        both = _bound(assumptions=("class_order_matches_dataset", "allowance_beta", "allowance_beta"))
        assert both.assumptions == ("allowance_beta", "class_order_matches_dataset")

    def test_the_internal_fields_default_to_none(self):
        b = _bound()
        assert (b.mean, b.support, b.delta) == (None, None, None)

    def test_bounds_are_frozen(self):
        with pytest.raises(AttributeError):
            _bound().bound = 0.5  # type: ignore[misc]


@pytest.mark.required
class TestLedger:
    def test_allocate(self):
        assert allocate(0.1, 4) == pytest.approx(0.025)
        with pytest.raises(ValueError, match="n_claims"):
            allocate(0.1, 0)
        with pytest.raises(ValueError, match="alpha"):
            allocate(1.0, 2)

    def test_ledger_over_budget_raises_and_names_the_sources(self):
        spent = [_bound(alpha=0.02), _bound(alpha=0.02, requirement=1), _envelope(alpha=0.01)]
        check_ledger(spent, 0.05)
        with pytest.raises(RuntimeError, match="user_model"):
            check_ledger(spent, 0.04)
