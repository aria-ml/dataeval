"""Bounds for the feasibility evaluator: one frozen record per bound, the Enums that describe it, the alpha ledger."""

__all__ = []

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, get_args

from dataeval.performance._feasibility_target import Scope

# Every column of the ``bounds`` table that is not a number is one of these closed sets. They are also the Enum dtypes.
Source = Literal["local_panels", "replicas", "knn", "linear_probe", "user_model"]  # the order of the table's rows
Role = Literal["envelope", "witness"]
Quantity = Literal["accuracy", "ap", "recall", "recall_at_fp", "mota", "false_positive_rate"]
BoundType = Literal["lower", "upper"]
Population = Literal["items", "labeled_items"]
Sample = Literal["audit_split", "audit_split_one_per_group", "replica_items"]
Unit = Literal["image", "sequence"]
State = Literal["unavailable", "uninformative", "informative"]
Reason = Literal[
    "too_few_units",
    "max_sequence_length_missing",
    "no_positives_possible",
    "no_positives",
    "no_negatives",
    "needs_scores",
    "fit_split_smaller_than_cells",
    "no_complete_panel",
    "witness_not_fitted",
    "bound_at_maximum",
    "bound_at_minimum",
]
Method = Literal["hoeffding", "hybrid", "exact_binomial"]
Assumption = Literal[
    "representation_independent_of_audit_labels",
    "allowance_beta",
    "allowance_gamma",
    "model_independent_of_audit_items",
    "class_order_matches_dataset",
    "replicas_random_sample",
    "replicas_one_protocol",
    "replicas_same_video_context",
    "replicated_sequences_random_sample",
    "replicated_frames_uniform_sample",
]

_ENUMS: dict[str, tuple[str, ...]] = {
    "source": get_args(Source),
    "role": get_args(Role),
    "quantity": get_args(Quantity),
    "bound_type": get_args(BoundType),
    "information_scope": get_args(Scope),
    "population": get_args(Population),
    "sample": get_args(Sample),
    "unit": get_args(Unit),
    "state": get_args(State),
}
_ASSUMPTIONS = get_args(Assumption)


@dataclass(frozen=True)
class Bound:
    """
    One bound on one requirement, spent at ``alpha`` (its claim's share of the error budget): one ``bounds`` row.

    ``bound`` is None exactly when the state is ``unavailable``; ``reason`` says why a bound is unavailable or
    uninformative. ``observed`` is a witness's point estimate of the quantity and is null for every envelope.
    ``assumptions`` lists what the bound rests on, in the order of :data:`Assumption`. ``mean``, ``support`` and
    ``delta`` are what ``units_to_cross`` needs to project a suggested minimum; they are not table columns. ``mean``
    is the recorded estimate on the metric's scale, which an envelope has although it publishes no ``observed``.
    """

    requirement: int
    source: Source
    role: Role
    quantity: Quantity
    bound_type: BoundType
    information_scope: Scope
    population: Population
    sample: Sample
    unit: Unit
    state: State
    reason: Reason | None
    bound: float | None
    observed: float | None
    n: int | None
    variance: float | None
    radius: float | None
    method: Method | None
    alpha: float
    assumptions: tuple[Assumption, ...]
    mean: float | None = None
    support: float | None = None
    delta: float | None = None

    def __post_init__(self) -> None:
        self._check_enums()
        object.__setattr__(self, "assumptions", tuple(a for a in _ASSUMPTIONS if a in self.assumptions))
        self._check_value()
        self._check_reason()
        self._check_direction()
        self._check_numbers()

    def _check_enums(self) -> None:
        for name, valid in _ENUMS.items():
            if getattr(self, name) not in valid:
                raise ValueError(f"Unknown {name} {getattr(self, name)!r}; valid: {list(valid)}")
        if self.method is not None and self.method not in get_args(Method):
            raise ValueError(f"Unknown method {self.method!r}; valid: {list(get_args(Method))}")
        unknown = [a for a in self.assumptions if a not in _ASSUMPTIONS]
        if unknown:
            raise ValueError(f"Unknown assumption {unknown[0]!r}; valid: {list(_ASSUMPTIONS)}")

    def _check_numbers(self) -> None:
        if self.role == "envelope" and self.observed is not None:
            raise ValueError(f"An envelope has no observed value, got {self.observed!r}")
        if not 0 < self.alpha <= 1:
            raise ValueError(f"Bound alpha must be in (0, 1], got {self.alpha!r}")

    def _check_value(self) -> None:
        if (self.bound is None) != (self.state == "unavailable"):
            raise ValueError(f"Bound must be None iff state is 'unavailable', got {self.bound!r} for {self.state!r}")
        if self.bound is not None and not math.isfinite(self.bound):
            raise ValueError(f"Bound must be finite, got {self.bound!r}")

    def _check_reason(self) -> None:
        if (self.reason is None) != (self.state == "informative"):
            raise ValueError(
                f"A reason comes with every state but 'informative', got {self.reason!r} for {self.state!r}"
            )
        if self.reason is not None and self.reason not in get_args(Reason):
            raise ValueError(f"Unknown reason {self.reason!r}; valid: {list(get_args(Reason))}")

    def _check_direction(self) -> None:
        """Check the direction: an envelope is an upper bound; a witness is lower, except for false positives."""
        if self.quantity == "false_positive_rate":
            if self.role == "envelope":
                raise ValueError("An envelope cannot bound false_positive_rate")
            allowed: BoundType = "upper"
        else:
            allowed = "upper" if self.role == "envelope" else "lower"
        if self.bound_type != allowed:
            raise ValueError(f"A {self.role}'s {self.quantity} bound must be {allowed}, got {self.bound_type!r}")


def allocate(alpha: float, n_claims: int) -> float:
    """Equal split of ``alpha`` across claims, each a (requirement, source) pair."""
    if n_claims < 1:
        raise ValueError(f"n_claims must be >= 1, got {n_claims}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return alpha / n_claims


def check_ledger(bounds: Iterable[Bound], alpha: float) -> None:
    """Raise if the bounds' alphas sum to more than ``alpha``; the error names the contributing sources."""
    spent: dict[str, float] = {}
    for b in bounds:
        spent[b.source] = spent.get(b.source, 0.0) + b.alpha
    total = math.fsum(spent.values())
    if total > alpha + 1e-12:
        raise RuntimeError(f"Internal error: alpha ledger over budget: {total} > {alpha}; spent by source: {spent}")
