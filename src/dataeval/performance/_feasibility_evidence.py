"""Evidence spine for the feasibility evaluator: endpoints, evidence items, the alpha ledger and the sampling plan."""

__all__ = []

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, get_args

from dataeval._experimental import experimental
from dataeval.performance._feasibility_constants import EVIDENCE_SCHEMA
from dataeval.performance._feasibility_target import Scope

Direction = Literal["upper", "lower"]
EndpointState = Literal["unavailable", "uninformative", "informative"]
Supports = Literal["ceiling", "witness"]
Quantity = Literal["metric", "false_positive_rate"]
ENDPOINT_STAT_KEYS = ("mean", "variance", "n", "support", "delta", "method")


def stats_key(target_spec: str, quantity: Quantity) -> str:
    """Return an endpoint's ``details["endpoints"]`` key: the spec for the metric, tagged for any other quantity."""
    return target_spec if quantity == "metric" else f"{target_spec}#{quantity}"


@experimental
@dataclass(frozen=True)
class Endpoint:
    """
    One bound on one target, spent at ``alpha`` (its claim's share of the error budget).

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    ``value`` is None exactly when the endpoint is unavailable; ``reason`` says why it is unavailable or uninformative.
    ``quantity`` names what is bounded: the target's metric (recall for a recall target) or the false-positive rate a
    recall target limits.
    """

    target_spec: str
    value: float | None
    direction: Direction
    alpha: float
    state: EndpointState
    reason: str | None = None
    quantity: Quantity = "metric"

    def __post_init__(self) -> None:
        if self.quantity not in get_args(Quantity):
            raise ValueError(f"Unknown quantity {self.quantity!r}; valid: {list(get_args(Quantity))}")
        if self.direction not in get_args(Direction):
            raise ValueError(f"Unknown direction {self.direction!r}; valid: {list(get_args(Direction))}")
        if self.state not in get_args(EndpointState):
            raise ValueError(f"Unknown state {self.state!r}; valid: {list(get_args(EndpointState))}")
        self._check_value()
        if not 0 < self.alpha <= 1:
            raise ValueError(f"Endpoint alpha must be in (0, 1], got {self.alpha!r}")

    def _check_value(self) -> None:
        if (self.value is None) != (self.state == "unavailable"):
            raise ValueError(
                f"Endpoint value must be None iff state is 'unavailable', got {self.value!r} for {self.state!r}"
            )
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError(f"Endpoint value must be finite, got {self.value!r}")


def _allowed(supports: Supports, quantity: Quantity) -> Direction | None:
    """Return the one direction a ``supports`` may give a ``quantity`` endpoint, or None where it may not."""
    if quantity == "metric":
        return "upper" if supports == "ceiling" else "lower"
    return None if supports == "ceiling" else "upper"


@experimental
@dataclass(frozen=True)
class Evidence:
    """
    One piece of evidence: what it supports, its basis and scope, the endpoints it yields, and raw details.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    ``details["endpoints"]`` maps each endpoint's :func:`stats_key` (its canonical ``target_spec``, tagged with its
    quantity unless that is the metric) to its summary statistics (mean, variance, n, support, delta, method), or None
    where the endpoint is unavailable. A ceiling's endpoints are upper bounds on the metric; a witness's metric
    endpoints are lower bounds, and it may add upper bounds on the false-positive rate.
    """

    kind: str
    supports: Supports
    basis: str
    population: str
    sample: str
    information_scope: Scope
    unit: str
    provenance: Mapping[str, Any]
    endpoints: tuple[Endpoint, ...]
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.basis.strip():
            raise ValueError(f"Evidence {self.kind!r} needs a non-empty basis")
        if self.supports not in get_args(Supports):
            raise ValueError(f"Unknown supports {self.supports!r}; valid: {list(get_args(Supports))}")
        if self.information_scope not in get_args(Scope):
            raise ValueError(f"Unknown information_scope {self.information_scope!r}; valid: {list(get_args(Scope))}")
        self._check_directions()
        self._check_endpoint_details()

    def _check_directions(self) -> None:
        for e in self.endpoints:
            allowed = _allowed(self.supports, e.quantity)
            if e.direction != allowed:
                raise ValueError(
                    f"Evidence {self.kind!r} supports a {self.supports}; its {e.quantity} endpoints "
                    + (f"must be {allowed}" if allowed else "are not allowed")
                    + f": {e.target_spec}"
                )

    def _check_endpoint_details(self) -> None:
        per_endpoint = self.details.get("endpoints")
        if not isinstance(per_endpoint, Mapping):
            raise ValueError(f"Evidence {self.kind!r} details need an 'endpoints' mapping")
        missing = [k for e in self.endpoints if (k := stats_key(e.target_spec, e.quantity)) not in per_endpoint]
        if missing:
            raise ValueError(f"Evidence {self.kind!r} details['endpoints'] is missing entries for {missing}")
        for spec, entry in per_endpoint.items():
            if entry is not None and (not isinstance(entry, Mapping) or set(entry) != set(ENDPOINT_STAT_KEYS)):
                raise ValueError(
                    f"Evidence {self.kind!r} details['endpoints'][{spec!r}] must be None or have exactly the keys "
                    f"{list(ENDPOINT_STAT_KEYS)}, got {entry!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the spine; ``provenance`` and ``details`` pass through by reference, uncopied."""
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["endpoints"] = [asdict(e) for e in self.endpoints]
        out["schema"] = EVIDENCE_SCHEMA
        return out


def allocate(alpha: float, n_claims: int) -> float:
    """Equal split of ``alpha`` across claims, each a (target, evidence source) pair."""
    if n_claims < 1:
        raise ValueError(f"n_claims must be >= 1, got {n_claims}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return alpha / n_claims


def check_ledger(evidence: Iterable[Evidence], alpha: float) -> None:
    """Raise if the endpoint alphas sum to more than ``alpha``; the error names the contributing sources."""
    spent = {}
    for item in evidence:
        spent[item.kind] = spent.get(item.kind, 0.0) + math.fsum(e.alpha for e in item.endpoints)
    total = math.fsum(spent.values())
    if total > alpha + 1e-12:
        raise RuntimeError(f"Internal error: alpha ledger over budget: {total} > {alpha}; spent by source: {spent}")


@experimental
@dataclass(frozen=True)
class SamplingPlan:
    """
    Pre-registered audit prefix sizes (one per look) and the seed of the unit order.

    .. warning::
       This feature is experimental and may change or be removed in future releases.
    """

    sizes: tuple[int, ...]
    order_seed: int

    def __post_init__(self) -> None:
        if not self.sizes or self.sizes[0] < 1 or any(b <= a for a, b in zip(self.sizes, self.sizes[1:], strict=False)):
            raise ValueError(f"sizes must be non-empty, positive and strictly increasing, got {self.sizes}")

    def look_alpha(self, alpha: float, look: int) -> float:
        """Alpha spent at a look: ``alpha / (look * (look + 1))``, which sums to ``alpha`` over all looks."""
        if look < 1:
            raise ValueError(f"look must be >= 1, got {look}")
        return alpha / (look * (look + 1))

    def prefix(self, n_units: int, look: int) -> int:
        """Return the prefix size for ``look`` (1-based), requiring the audit to have at least that many units."""
        if not 1 <= look <= len(self.sizes):
            raise ValueError(f"look {look} is outside the plan of {len(self.sizes)} looks")
        if n_units < self.sizes[look - 1]:
            raise ValueError(f"look {look} needs {self.sizes[look - 1]} units, the audit has {n_units}")
        return self.sizes[look - 1]
