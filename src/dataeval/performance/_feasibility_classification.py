"""
Classification compositions for the feasibility evaluator: every evidence source, built from the core bounds.

Sources: post-stratified local-panel ceilings (C3), replica ceilings, built-in frozen witnesses (k-NN and a linear
probe fitted on the fit split) and a user witness from precomputed class scores or class decisions. Each source is one
``Evidence`` item (local panels give two when accuracy and the cost-weighted AP and recall ceilings differ in scope)
with one endpoint per target it bounds, each spent at the claim's alpha and split inside the endpoint: C3 in halves; an
AP ceiling equally over its positive kink costs and one prevalence upper endpoint; a recall ceiling equally over its
positive kink costs and two prevalence endpoints (lower and upper); an AP witness over a TP lower and an FP upper
endpoint per threshold and one prevalence upper endpoint; a macro AP over its classes. A witness gives a recall target
two endpoints instead, at half the claim's alpha each: recall (lower) and the false-positive rate the target limits
(upper).
"""

__all__ = []

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from dataeval.core._ap_bounds import ap_lower_staircase, ap_upper_envelope, witness_thresholds
from dataeval.core._concentration import EndpointResult, binomial_endpoint, mean_endpoint, ratio_endpoint
from dataeval.core._consensus import (
    consensus_max,
    cost_consensus,
    cost_consensus_max,
    disjoint_panels,
    kink_costs,
    zero_one_consensus,
)
from dataeval.core._local_panels import fit_partition, local_panel_bound
from dataeval.performance._feasibility_constants import (
    AP_INTERVALS,
    COST_GRID,
    PANEL_SIZE,
    WITNESS_POS_PREVALENCE,
    WITNESS_QUANTILES,
    n_cells_for,
)
from dataeval.performance._feasibility_decide import _READS_SCOPE
from dataeval.performance._feasibility_evidence import (
    ENDPOINT_STAT_KEYS,
    Endpoint,
    Evidence,
    Quantity,
    SamplingPlan,
    Supports,
    stats_key,
)
from dataeval.performance._feasibility_target import Aggregation, Scope, TargetSpec
from dataeval.performance._feasibility_units import Units

# value (None: unavailable, its reason in raw), recommend's stats (None: no closed form), raw
_Bound = tuple[float | None, dict[str, Any] | None, dict[str, Any]]
_ClassBound = tuple[float, dict[str, Any]]  # one class's AP or recall endpoint and its raw results
_RiskAt = Callable[[float, float], tuple[float, Mapping[str, Any]]]  # (cost, delta) -> risk lower bound, raw result
_OneVsRest = tuple[NDArray[np.float64], _RiskAt, NDArray[np.float64]]  # a class's cost grid, risk bound, prevalence
_Side = Literal["lower", "upper"]
_Method = Literal["hoeffding", "hybrid", "exact_binomial"]
_TRIVIAL = 1e-12
_LOCAL, _REPLICA, _BUILTIN, _USER = (
    "local_panel_ceiling/v1",
    "replica_ceiling/v1",
    "builtin_witness/v1",
    "user_witness/v1",
)
_ENVELOPE = "AP via the Reid-Williamson (2011) risk-to-AP envelope (DFE operating-point inequalities)"
_RECALL = "recall at a false-positive limit via DFE operating-point inequalities over a two-sided prevalence interval"
_HYBRID = "Hoeffding (1963) and Maurer-Pontil (2009, Theorem 4) hybrid endpoints"
_EXACT = "Clopper-Pearson (1934) exact binomial endpoints on ungrouped Bernoulli trials"
_RATIO = "frame_pooled as a ratio of per-group totals with simultaneous numerator and denominator endpoints"
_WITNESS = (
    f"Frozen witness scored once on the audit; {_EXACT}, else {_HYBRID}; {_RATIO}; AP via the DFE witness staircase"
)
_BASIS = {
    _LOCAL: "DataEval's post-stratified pooling (C3) of DFE disjoint within-cell panels, proved in "
    f"local_panel_bound's Notes with two Hoeffding (1963) steps; DFE consensus inequality; {_ENVELOPE}; {_RECALL}; "
    f"prevalence: {_EXACT}, else hybrid",
    _REPLICA: f"DFE replicated consensus (consensus inequality); {_HYBRID}; {_RATIO}; {_ENVELOPE}; {_RECALL}",
    _BUILTIN: f"{_WITNESS} (Davis-Goadrich 2006)",
    _USER: f"{_WITNESS} (Davis-Goadrich 2006)",
}
_REPRESENTATION = "The representation was fitted and selected independently of the audit units' labels (declared)."
_MODELS: dict[str, Callable[[], Any]] = {
    "knn": lambda: KNeighborsClassifier(n_neighbors=5),
    "probe": lambda: LogisticRegression(max_iter=1000),
}


class _UnavailableError(Exception):
    """An endpoint that cannot be computed; the message is its reason."""


def _need_units(n: int) -> None:
    if n < 2:
        raise _UnavailableError(f"fewer than 2 units ({n})")


def _totals(values: NDArray[np.float64], groups: NDArray[np.intp]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    inverse = np.unique(groups, return_inverse=True)[1].reshape(-1)
    return np.bincount(inverse, weights=values).astype(np.float64), np.bincount(inverse).astype(np.float64)


def _per_unit(
    values: NDArray[np.float64], groups: NDArray[np.intp], aggregation: Aggregation | None
) -> NDArray[np.float64]:
    """Per-unit observations: items as they are, or a group's mean (sequence_mean) or total (frame_pooled)."""
    if aggregation is None:
        return values
    sums, counts = _totals(values, groups)
    return sums / counts if aggregation == "sequence_mean" else sums


def _scale(aggregation: Aggregation | None, max_len: int | None) -> int:
    """Support multiplier of a per-unit total: the declared max_sequence_length for frame_pooled, else 1."""
    if aggregation != "frame_pooled":
        return 1
    if max_len is None:
        raise _UnavailableError("frame_pooled targets need a declared max_sequence_length to bound per-group totals")
    return max_len


def _within(obs: NDArray[np.float64], support: float) -> NDArray[np.float64]:
    """Absorb float rounding over the support (a group mean or total of values at their maximum); more is a bug."""
    if len(obs) and obs.max() > support * (1 + 1e-9):
        raise RuntimeError(f"Internal error: an observation {obs.max()} exceeds its a-priori support {support}.")
    return np.minimum(obs, support)


def _bernoulli(call: "_Call") -> _Method:
    """Exact binomial where trials are independent items; the hybrid bounded-variable endpoints once items group."""
    return "hybrid" if call.units.grouped else "exact_binomial"


def _bounded(method: _Method) -> Literal["hoeffding", "hybrid"]:
    return "hybrid" if method == "exact_binomial" else method


def _mean(
    obs: NDArray[np.float64], support: float, delta: float, side: _Side, method: _Method = "hybrid"
) -> EndpointResult:
    _need_units(len(obs))
    if method == "exact_binomial":
        if support != 1 or not np.isin(obs, (0.0, 1.0)).all():
            raise RuntimeError("Internal error: exact binomial endpoints need 0/1 trials with support 1.")
        return binomial_endpoint(int(obs.sum()), len(obs), delta, side=side)
    return mean_endpoint(_within(obs, support), support, delta, side=side, method=method)


def _flip(x: float, loss: bool) -> float:
    return 1.0 - x if loss else x


def _accuracy_bound(
    values: NDArray[np.float64],
    groups: NDArray[np.intp],
    spec: TargetSpec,
    support: float,
    call: "_Call",
    loss: bool,
    method: _Method = "hybrid",
) -> _Bound:
    """Accuracy from per-item correctness (witness) or consensus loss (``loss``: the ceiling is 1 - the risk lower)."""
    if spec.aggregation == "frame_pooled":
        scale = _scale(spec.aggregation, call.max_len)
        sums, counts = _totals(values, groups)
        _need_units(len(sums))
        ratio = ratio_endpoint(  # a ratio of totals is not a Bernoulli trial
            _within(sums, scale * support), counts, scale * support, scale, call.alpha, method=_bounded(method)
        )
        raw = {"ratio_endpoint": ratio, "num_support": scale * support, "den_support": scale, "deltas": [call.alpha]}
        return _flip(ratio, loss), None, raw
    r = _mean(_per_unit(values, groups, spec.aggregation), support, call.alpha, "lower", method)
    stats = {k: r[k] for k in ("variance", "n", "support", "delta", "method")} | {"mean": _flip(r["mean"], loss)}
    return _flip(r["endpoint"], loss), stats, {**r, "deltas": [call.alpha]}


def _ap_ceiling(
    costs: NDArray[np.float64],
    risk_at: _RiskAt,
    prevalence: NDArray[np.float64],
    scale: int,
    delta: float,
    method: _Method = "hybrid",
) -> _ClassBound:
    """Envelope AP ceiling: ``delta`` split equally over the positive costs and one prevalence upper endpoint.

    ``method`` is the prevalence endpoint's; the risk endpoints come from ``risk_at``.
    """
    n_endpoints = int(np.count_nonzero(costs)) + 1
    share = delta / n_endpoints
    risks = [risk_at(float(c), share) if c > 0 else (0.0, None) for c in costs]  # emitting everything misses nothing
    risk = np.array([r for r, _ in risks])
    mu = _mean(prevalence, scale, share, "upper", method)["endpoint"]
    value = ap_upper_envelope(risk, costs, mu, intervals=AP_INTERVALS)
    raw = {"value": value, "costs": costs.tolist(), "risk_lower": risk.tolist(), "risk_results": [r for _, r in risks]}
    return value, raw | {"mu_upper": mu, "deltas": [share] * n_endpoints}


def _recall_upper(risk: Iterable[float], costs: Iterable[float], f0: float, pi_lower: float, pi_upper: float) -> float:
    """
    ``U_r``: an upper bound on the recall reachable at false-positive rate ``<= f0`` (spec "Ceiling (exclusion)").

    Per positive cost ``lam`` with risk lower bound ``l``, the maximum over positive prevalence in
    ``[pi_lower, pi_upper]`` of ``g(pi) = 1 - lam f0 - (l - lam f0) / pi``, clipped to ``[0, 1]``; the minimum over
    costs (1 with none). ``g`` increases in ``pi`` when ``l > lam f0`` and decreases when ``l < lam f0``.
    """
    terms = []
    for l_j, lam in zip(risk, costs, strict=True):
        slack = l_j - lam * f0
        if slack > 0:
            if pi_upper <= 0:
                raise _UnavailableError("no positives are possible")
            g = 1 - lam * f0 - slack / pi_upper
        elif slack < 0:
            g = math.inf if pi_lower <= 0 else 1 - lam * f0 - slack / pi_lower
        else:
            g = 1 - lam * f0  # the quotient is 0 at every positive prevalence
        terms.append(min(1.0, max(0.0, g)))
    return min(terms, default=1.0)


def _recall_ceiling(
    costs: NDArray[np.float64],
    risk_at: _RiskAt,
    prevalence: NDArray[np.float64],
    delta: float,
    f0: float,
    method: _Method,
) -> _ClassBound:
    """Recall ceiling ``U_r``: ``delta`` split equally over the positive costs and two prevalence endpoints.

    ``method`` is the prevalence endpoints'; the risk endpoints come from ``risk_at``.
    """
    costs = costs[costs > 0]  # at cost 0 the risk is trivially 0 and bounds nothing
    share = delta / (len(costs) + 2)
    risks = [risk_at(float(c), share) for c in costs]
    risk = [r for r, _ in risks]
    pi_lower, pi_upper = (_mean(prevalence, 1, share, side, method)["endpoint"] for side in ("lower", "upper"))
    value = _recall_upper(risk, costs.tolist(), f0, pi_lower, pi_upper)
    raw = {"value": value, "costs": costs.tolist(), "risk_lower": risk, "risk_results": [r for _, r in risks]}
    return value, raw | {"pi_lower": pi_lower, "pi_upper": pi_upper, "deltas": [share] * (len(costs) + 2)}


def _recall_bound(spec: TargetSpec, call: "_Call", one_vs_rest: Callable[[int], _OneVsRest], method: _Method) -> _Bound:
    """Bound the recall of the spec's class at the claim's alpha from that class's cost-weighted inputs."""
    c = _class_index(call.units, cast(str, spec.class_name))  # a recall target always names its class
    f0 = cast(float, spec.max_false_positive_rate)  # and sets its limit
    value, raw = _recall_ceiling(*one_vs_rest(c), call.alpha, f0, method)
    return value, None, raw


def _class_index(units: Units, name: str) -> int:
    names = units.class_names
    if names is not None and len(names) == units.n_classes:
        return names.index(name)
    if names is None and name.isdigit() and int(name) < units.n_classes:
        return int(name)
    raise ValueError(f"Cannot map class {name!r} to a label index: index2label is sparse or does not declare it.")


def _ap_bound(spec: TargetSpec, call: "_Call", per_class: Callable[[int, float], _ClassBound]) -> _Bound:
    """One class at the claim's alpha, or the macro mean over every class index at ``alpha / n_classes`` each."""
    if spec.class_name is not None:
        value, raw = per_class(_class_index(call.units, spec.class_name), call.alpha)
        return value, None, raw
    n = call.units.n_classes
    results = [per_class(c, call.alpha / n) for c in range(n)]  # an unavailable class makes the macro unavailable
    value = math.fsum(v for v, _ in results) / n
    deltas = [d for _, r in results for d in r["deltas"]]
    return value, None, {"value": value, "per_class": [r for _, r in results], "deltas": deltas}


def _endpoint(
    key: str, value: float | None, supports: Supports, alpha: float, reason: str | None, quantity: Quantity = "metric"
) -> Endpoint:
    upper = supports == "ceiling" or quantity == "false_positive_rate"
    direction = "upper" if upper else "lower"
    if value is None:
        return Endpoint(key, None, direction, alpha, "unavailable", reason, quantity)
    if value >= 1 - _TRIVIAL if upper else value <= _TRIVIAL:
        trivial = (
            "a false-positive bound at its maximum (1)"
            if quantity == "false_positive_rate"
            else "a ceiling at the metric's maximum (1)"
            if upper
            else "a witness at the metric's minimum (0)"
        )
        return Endpoint(key, value, direction, alpha, "uninformative", trivial, quantity)
    return Endpoint(key, value, direction, alpha, "informative", quantity=quantity)


def _rate(hits: NDArray[np.bool_], delta: float, side: _Side, population: str) -> dict[str, Any]:
    """Exact binomial endpoint on the share of ``hits`` (decisions for the class) in one audit population."""
    if not len(hits):
        return {"value": None, "stats": None, "reason": f"no {population}", "population": population}
    r = binomial_endpoint(int(hits.sum()), len(hits), delta, side=side)
    stats = {k: r[k] for k in ENDPOINT_STAT_KEYS}
    return {"value": r["endpoint"], "stats": stats, "reason": None, "population": population}


@dataclass(frozen=True)
class _Call:
    """What every source of one call shares: units, features (aligned with labeled items), splits and claim alpha."""

    units: Units
    features: NDArray[Any]
    audit: NDArray[np.intp]
    fit: NDArray[np.intp]
    alpha: float
    max_len: int | None
    details: dict[str, Any]

    def item(
        self,
        kind: str,
        supports: Supports,
        scope: Scope,
        specs: Sequence[TargetSpec],
        bound: Callable[[TargetSpec], _Bound],
        *,
        sample: str,
        provenance: Mapping[str, Any],
        details: Mapping[str, Any],
    ) -> Evidence:
        endpoints, stats, results = [], {}, {}
        for spec in specs:
            key, reason = spec.canonical(), None
            try:
                value, stats[key], results[key] = bound(spec)
            except _UnavailableError as e:
                value, stats[key], results[key], reason = None, None, None, str(e)
            spent = 0.0 if results[key] is None else math.fsum(results[key]["deltas"])
            if spent > self.alpha + 1e-12:
                raise RuntimeError(
                    f"Internal error: {kind} splits {spent} inside {key}, above its claim's {self.alpha}"
                )
            if supports == "ceiling" or spec.metric != "recall":
                endpoints.append(_endpoint(key, value, supports, self.alpha, reason))
                continue
            # A recall witness bounds recall (lower) and the false-positive rate the target limits (upper), half each.
            raw, missing = results[key], {"value": None, "stats": None, "reason": reason}
            rec, fp_rate = (missing, missing) if raw is None else (raw, raw["false_positive_rate"])
            stats[stats_key(key, "false_positive_rate")] = fp_rate["stats"]
            endpoints += [
                _endpoint(key, rec["value"], supports, self.alpha / 2, rec["reason"]),
                _endpoint(key, fp_rate["value"], supports, self.alpha / 2, fp_rate["reason"], "false_positive_rate"),
            ]
        return Evidence(
            kind=kind,
            supports=supports,
            basis=_BASIS[kind],
            population=self.units.population,
            sample=sample,
            information_scope=scope,
            unit="group" if self.units.grouped else "item",
            provenance=dict(provenance),
            endpoints=tuple(endpoints),
            details={"endpoints": stats, "results": results, **self.details, **details},
        )


@dataclass(frozen=True)
class _Missing:
    """A source whose shared setup failed: every endpoint is unavailable for the same reason."""

    reason: str

    def bound(self, _spec: TargetSpec) -> _Bound:
        raise _UnavailableError(self.reason)


# --- local panels ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Panels:
    """One draw of disjoint within-cell panels over the panel units, shared by every local-panel endpoint."""

    call: _Call
    labels: NDArray[np.intp]  # label of each panel unit
    n_by_cell: NDArray[np.intp]
    panels: dict[int, NDArray[np.intp]]  # nonempty cells: (k_h, m) panel-unit indices
    m: int
    beta: Any
    gamma: float  # 0 unless the accuracy scope is full_input

    def bound(self, spec: TargetSpec) -> _Bound:
        if spec.metric == "recall":
            return _recall_bound(spec, self.call, self.one_vs_rest, _bernoulli(self.call))
        return self.accuracy() if spec.metric == "accuracy" else _ap_bound(spec, self.call, self.ap_class)

    def accuracy(self) -> _Bound:
        n_classes, d = self.call.units.n_classes, self.call.alpha / 2
        values = {h: zero_one_consensus(self.labels[ix], n_classes) for h, ix in self.panels.items()}
        vmax = consensus_max(n_classes, self.m)
        r = local_panel_bound(values, self.n_by_cell, value_max=vmax, delta_within=d, delta_between=d, beta=self.beta)
        gbar = np.zeros(len(self.n_by_cell))
        gbar[list(values)] = [v.mean() for v in values.values()]
        beta = np.broadcast_to(np.asarray(0.0 if self.beta is None else self.beta, dtype=np.float64), gbar.shape)
        n, pooled = r["n_units"], np.concatenate(list(values.values()))
        stats = {
            "mean": min(1.0, max(0.0, 1.0 - float(self.n_by_cell @ (gbar - beta)) / n + self.gamma)),
            "variance": float(pooled.var(ddof=1)) if len(pooled) > 1 else 0.0,
            "n": n,
            # Effective Hoeffding constant: units_to_cross(n, delta/2) reproduces r1 + r2 here, and C3's radius scales
            # as 1/sqrt(n) when panels grow with the audit, so the suggested minimum tracks C3, not one Hoeffding term.
            "support": (r["within_radius"] + r["between_radius"]) / math.sqrt(math.log(1 / d) / (2 * n)),
            "delta": d,
            "method": "hoeffding",
        }
        return 1.0 - max(0.0, r["bound"] - self.gamma), stats, {**r, "gamma": self.gamma, "deltas": [d, d]}

    def risk(self, positives: dict[int, NDArray[np.intp]], cost: float, share: float) -> tuple[float, Mapping]:
        values = {h: cost_consensus(k, self.m, cost) for h, k in positives.items()}
        vmax, d = cost_consensus_max(cost, self.m), share / 2
        r = local_panel_bound(values, self.n_by_cell, value_max=vmax, delta_within=d, delta_between=d)
        return r["bound"], r

    def one_vs_rest(self, c: int) -> _OneVsRest:
        """Class ``c``'s cost grid, cost-weighted risk bound and prevalence observations (panel units' 0/1 labels)."""
        positives = {h: (self.labels[ix] == c).sum(axis=1) for h, ix in self.panels.items()}
        return kink_costs(self.m), partial(self.risk, positives), (self.labels == c).astype(np.float64)

    def ap_class(self, c: int, delta: float) -> _ClassBound:
        costs, risk_at, prevalence = self.one_vs_rest(c)
        return _ap_ceiling(costs, risk_at, prevalence, 1, delta, _bernoulli(self.call))


def _one_per_group(items: NDArray[np.intp], groups: NDArray[np.intp], rng: np.random.Generator) -> NDArray[np.intp]:
    """One item per group, uniformly: each group's first item in a uniform random order."""
    order = rng.permutation(len(items))
    return items[order[np.unique(groups[order], return_index=True)[1]]]


def _draw_panels(
    call: _Call, n_cells: int, m: int, partition_seed: int, rng: np.random.Generator, beta_gamma: tuple[Any, float]
) -> tuple[_Panels | _Missing, NDArray[np.intp]]:
    """Fit the label-blind partition on the fit split and draw disjoint panels within its cells on the audit."""
    items = _one_per_group(call.audit, call.units.groups[call.audit], rng) if call.units.grouped else call.audit
    if len(call.fit) < n_cells:
        return _Missing(f"the fit split has {len(call.fit)} items, fewer than the {n_cells} partition cells"), items
    if len(items) < 2:
        return _Missing(f"fewer than 2 audit units ({len(items)})"), items
    cells = fit_partition(call.features[call.fit], n_cells, partition_seed)(call.features[items])
    panels = {h: ix for h, ix in disjoint_panels(cells, m, rng).items() if len(ix)}
    if not panels:
        return _Missing(f"no partition cell holds {m} audit units, so no panel is complete"), items
    n_by_cell = np.bincount(cells, minlength=n_cells)
    return _Panels(call, call.units.labels[items], n_by_cell, panels, m, *beta_gamma), items


def _accuracy_scope(allowances: Mapping[str, Any], grouped: bool) -> Scope:
    if grouped or allowances.get("beta") is None:  # single-frame panels stay partition_restricted for video (phase 1)
        return "partition_restricted"
    return "fixed_representation" if allowances.get("gamma") is None else "full_input"


def _local_panels(
    call: _Call,
    specs: Sequence[TargetSpec],
    allowances: Mapping[str, Any],
    sizes: tuple[int, int, list[str]],
    partition_seed: int,
    rng: np.random.Generator,
) -> dict[str, Evidence]:
    """C3 ceilings; accuracy takes beta (fixed_representation) and gamma (full_input), AP and recall do not."""
    specs = [s for s in specs if s.aggregation != "frame_pooled"]  # frame-weighted panels are not built yet
    if not specs:
        return {}
    n_cells, m, provisional = sizes
    scope = _accuracy_scope(allowances, call.units.grouped)
    beta = None if scope == "partition_restricted" else allowances["beta"]
    gamma = allowances["gamma"] if scope == "full_input" else 0.0
    source, items = _draw_panels(call, n_cells, m, partition_seed, rng, (beta, gamma))
    parts: dict[str, tuple[Scope, list[TargetSpec]]] = {_LOCAL: (scope, specs)}
    if scope != "partition_restricted":
        accuracy = [s for s in specs if s.metric == "accuracy"]
        cost_weighted = [s for s in specs if s.metric != "accuracy"]  # AP and recall: cost-weighted one-vs-rest
        parts = {
            f"{_LOCAL}:accuracy": (scope, accuracy),
            f"{_LOCAL}:cost_weighted": ("partition_restricted", cost_weighted),
        }
    details = {
        "panel_size": m,
        "n_cells": n_cells,
        "provisional_defaults": provisional,
        "cost_grid": COST_GRID,
        "allowances": dict(allowances),
        "allowances_applied": {"beta": beta, "gamma": gamma},
        "panel_items": items,
        "assumptions": [_REPRESENTATION, "beta and gamma allowances are user-supplied and assumed, not estimated"],
    }
    if call.units.grouped:
        details["assumptions"].append("Single-frame local panels stay partition_restricted for video; no allowance.")
    sample = "audit split, one item drawn uniformly per group" if call.units.grouped else "audit split"
    provenance = {"route": "post-stratified local panels (C3)", "partition": "k-means fitted on the fit split"}
    return {
        eid: call.item(_LOCAL, "ceiling", s, part, source.bound, sample=sample, provenance=provenance, details=details)
        for eid, (s, part) in parts.items()
        if part
    }


# --- replicas --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Replicas:
    """Primary label plus replicas per item (m_i = 1 + r_i); consensus is computed per panel size."""

    call: _Call
    rows: list[NDArray[np.intp]]
    sizes: NDArray[np.intp]
    groups: NDArray[np.intp]

    @property
    def method(self) -> _Method:
        """Hoeffding alone when panel sizes differ: the per-item values are then not identically distributed."""
        return "hoeffding" if len(np.unique(self.sizes)) > 1 else "hybrid"

    def per_size(self, fn: Callable[[NDArray[np.intp], int], NDArray[np.float64]]) -> NDArray[np.float64]:
        out = np.empty(len(self.rows))
        for m in np.unique(self.sizes):
            idx = np.flatnonzero(self.sizes == m)
            out[idx] = fn(np.stack([self.rows[i] for i in idx]), int(m))
        return out

    def bound(self, spec: TargetSpec) -> _Bound:
        if spec.metric == "recall":  # recall takes no aggregation
            return _recall_bound(spec, self.call, partial(self.one_vs_rest, aggregation=None), self.method)
        if spec.metric == "ap":
            return _ap_bound(spec, self.call, partial(self.ap_class, spec.aggregation))
        n_classes = self.call.units.n_classes
        loss = self.per_size(lambda labels, _: zero_one_consensus(labels, n_classes))
        vmax = max(consensus_max(n_classes, int(m)) for m in np.unique(self.sizes))
        return _accuracy_bound(loss, self.groups, spec, vmax, self.call, loss=True, method=self.method)

    def risk(self, c: int, aggregation: Aggregation | None, cost: float, share: float) -> tuple[float, Mapping]:
        loss = self.per_size(lambda labels, m: cost_consensus((labels == c).sum(axis=1), m, cost))
        vmax = max(cost_consensus_max(cost, int(m)) for m in np.unique(self.sizes))
        scale = _scale(aggregation, self.call.max_len)
        r = _mean(_per_unit(loss, self.groups, aggregation), scale * vmax, share, "lower", self.method)
        return r["endpoint"], r

    def one_vs_rest(self, c: int, aggregation: Aggregation | None) -> _OneVsRest:
        """Class ``c``'s cost grid (the modal panel size's kinks), risk bound and per-unit label shares (prevalence)."""
        prevalence = _per_unit(self.per_size(lambda labels, _: (labels == c).mean(axis=1)), self.groups, aggregation)
        costs = kink_costs(int(np.bincount(self.sizes).argmax()))
        return costs, partial(self.risk, c, aggregation), prevalence

    def ap_class(self, aggregation: Aggregation | None, c: int, delta: float) -> _ClassBound:
        costs, risk_at, prevalence = self.one_vs_rest(c, aggregation)
        scale = _scale(aggregation, self.call.max_len)
        return _ap_ceiling(costs, risk_at, prevalence, scale, delta, self.method)


def _replica_assumptions(call: _Call, specs: Sequence[TargetSpec]) -> list[str]:
    out = [
        "Items with replicas are a random sample of the population.",
        (
            "Replicas and primary labels follow one annotation protocol, each an independent label of the item "
            "(open item 5)."
        ),
    ]
    if call.units.grouped:
        out.append("Annotators labeled the replicas with the same video context as the primary labels.")
    if any(s.aggregation == "sequence_mean" for s in specs):
        out.append(
            "Replicated groups are a random sample of sequences, with the replicated frames uniform within each."
        )
    if any(s.aggregation == "frame_pooled" for s in specs):
        out.append(
            "Replicated frames are a uniform sample of all frames (whole random sequences, or each frame "
            "independently with one probability)."
        )
    if "sampling_plan" in call.details:
        out.append("The replica set was fixed before the first look.")
    return out


def _replica(call: _Call, specs: Sequence[TargetSpec], replicas: Mapping[int, NDArray[np.intp]]) -> dict[str, Evidence]:
    items = np.array(sorted(replicas), dtype=np.intp)
    rows = [np.r_[call.units.labels[i], np.asarray(replicas[i], dtype=np.intp)] for i in items]
    source = _Replicas(call, rows, np.array([len(r) for r in rows]), call.units.groups[items])
    details = {"replica_items": items, "cost_grid": COST_GRID, "assumptions": _replica_assumptions(call, specs)}
    sample = f"{len(items)} items with replicas"
    provenance = {"route": "replicated consensus", "replicas": "replicas= sidecar plus the primary label"}
    item = call.item(
        _REPLICA, "ceiling", "full_input", specs, source.bound, sample=sample, provenance=provenance, details=details
    )
    return {_REPLICA: item}


# --- witnesses -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Scores:
    """
    A frozen scorer's class scores on the fit split (for thresholds) and on the audit (scored once).

    Its decision is the top class. ``decisions``: the scores are one-hot class decisions, which rank nothing, so no AP.
    """

    call: _Call
    fit: NDArray[np.float64]
    audit: NDArray[np.float64]
    decisions: bool = False

    def bound(self, spec: TargetSpec) -> _Bound:
        if spec.metric == "ap":
            if self.decisions:
                raise _UnavailableError("predictions are class decisions, not scores; AP needs scores")
            return _ap_bound(spec, self.call, partial(self.ap_class, spec.aggregation))
        if spec.metric == "recall":
            return self.recall(spec)
        labels, groups = self.call.units.labels[self.call.audit], self.call.units.groups[self.call.audit]
        correct = (self.audit.argmax(axis=1) == labels).astype(np.float64)
        return _accuracy_bound(correct, groups, spec, 1.0, self.call, loss=False, method=_bernoulli(self.call))

    def recall(self, spec: TargetSpec) -> _Bound:
        """Recall over the class's audit items (lower), false-positive rate over the others (upper), alpha / 2 each."""
        name = cast(str, spec.class_name)  # a recall target always names its class
        c = _class_index(self.call.units, name)
        hit, positive = self.audit.argmax(axis=1) == c, self.call.units.labels[self.call.audit] == c
        d = self.call.alpha / 2  # recall is refused on grouped data, so the audit items are i.i.d. trials
        recall = _rate(hit[positive], d, "lower", f"audit items labeled {name!r}")
        fp_rate = _rate(hit[~positive], d, "upper", f"audit items labeled other than {name!r}")
        return recall["value"], recall["stats"], {**recall, "false_positive_rate": fp_rate, "deltas": [d, d]}

    def ap_class(self, aggregation: Aggregation | None, c: int, delta: float) -> _ClassBound:
        scale = _scale(aggregation, self.call.max_len)
        fit_positive = self.call.units.labels[self.call.fit] == c
        thresholds = witness_thresholds(
            self.fit[:, c],
            fit_positive,
            float(fit_positive.mean()),
            pos_below=WITNESS_POS_PREVALENCE,
            quantiles=WITNESS_QUANTILES,
        )
        positive, groups = self.call.units.labels[self.call.audit] == c, self.call.units.groups[self.call.audit]
        d = delta / (2 * len(thresholds) + 1)

        method: _Method = "hybrid" if aggregation == "frame_pooled" else _bernoulli(self.call)

        def count(indicator: NDArray[np.bool_], side: _Side) -> float:
            obs = _per_unit(indicator.astype(np.float64), groups, aggregation)
            return _mean(obs, scale, d, side, method)["endpoint"]

        emit = self.audit[None, :, c] >= thresholds[:, None]
        tp = np.array([count(e & positive, "lower") for e in emit])
        fp = np.array([count(e & ~positive, "upper") for e in emit])
        mu = count(positive, "upper")
        value = ap_lower_staircase(tp, fp, mu)
        raw = {"value": value, "thresholds": thresholds.tolist(), "tp_lower": tp.tolist(), "fp_upper": fp.tolist()}
        return value, raw | {"mu_upper": mu, "deltas": [d] * (2 * len(thresholds) + 1)}


def _onto_classes(proba: NDArray[np.float64], classes: NDArray[Any], n_classes: int) -> NDArray[np.float64]:
    out = np.zeros((len(proba), n_classes))
    out[:, np.asarray(classes, dtype=np.intp)] = proba  # classes absent from the fit split score 0
    return out


def _fit_scores(call: _Call, model: Any) -> _Scores | _Missing:
    """Fit on the fit split only, then score the fit split (for thresholds) and the audit once."""
    try:
        model.fit(call.features[call.fit], call.units.labels[call.fit])
        p_fit, p_audit = (model.predict_proba(call.features[rows]) for rows in (call.fit, call.audit))
    except ValueError as e:  # too few fit units, a single fit class or an empty audit
        return _Missing(f"{type(model).__name__} could not be fitted on the fit split or score the audit: {e}")
    n = call.units.n_classes
    return _Scores(call, _onto_classes(p_fit, model.classes_, n), _onto_classes(p_audit, model.classes_, n))


def _builtin(call: _Call, specs: Sequence[TargetSpec]) -> dict[str, Evidence]:
    out = {}
    for name, make in _MODELS.items():
        model = make()
        source = _fit_scores(call, model)
        provenance = {"reads": "representation", "model": type(model).__name__, "params": model.get_params()}
        details = {"assumptions": [_REPRESENTATION, "Fitted on the fit split only and frozen before the audit."]}
        out[f"{_BUILTIN}:{name}"] = call.item(
            _BUILTIN,
            "witness",
            "fixed_representation",
            specs,
            source.bound,
            sample="audit split",
            provenance=provenance | {"fit": "fit split"},
            details=details,
        )
    return out


def _check_leak(call: _Call, training_items: Any) -> None:
    if training_items is None:
        return
    if isinstance(training_items, (str, bytes)) or not isinstance(training_items, Iterable):
        raise ValueError("provenance['training_items'] must be a collection of datum ids, not a single id.")
    trained = set(np.asarray(list(training_items), dtype=object).reshape(-1).tolist())  # any iterable, a set too
    ids = call.units.item_ids[call.units.item_indices]
    if {isinstance(t, str) for t in trained} - {isinstance(i, str) for i in ids.tolist()}:  # str vs number never match
        raise ValueError(
            f"provenance['training_items'] holds {sorted({type(t).__name__ for t in trained})} values, but the "
            f"dataset's datum ids are {sorted({type(i).__name__ for i in ids.tolist()})}, so some could never match; "
            "pass datum ids."
        )
    trained_groups = set(call.units.groups[[i in trained for i in ids.tolist()]].tolist())
    leaked = [
        i
        for i, g in zip(ids[call.audit].tolist(), call.units.groups[call.audit].tolist(), strict=True)
        if g in trained_groups
    ]
    if leaked:
        raise ValueError(
            f"provenance['training_items'] shares physical groups with {len(leaked)} audit items (e.g. {leaked[:5]}); "
            "a witness trained on an audit unit cannot bound the population. Exclude those groups from its training "
            "data or from the audit."
        )


def _one_hot_decisions(arr: NDArray[Any], n_items: int, n_classes: int) -> NDArray[np.float64]:
    if not np.issubdtype(arr.dtype, np.integer):  # cast to integers, a boolean would silently mean classes 0 and 1
        raise ValueError(f"1-D predictions are class decisions: pass integer class indices, got dtype {arr.dtype}.")
    if len(arr) != n_items:
        raise ValueError(f"Class decisions need one per dataset item ({n_items}), got {len(arr)}.")
    outside = arr[(arr < 0) | (arr >= n_classes)]
    if len(outside):
        raise ValueError(
            f"Class decisions must be indices of declared classes, in [0, {n_classes}); found "
            f"{outside[:5].tolist()} outside it."
        )
    return np.eye(n_classes)[arr]


def check_predictions(predictions: ArrayLike, n_items: int, n_classes: int) -> tuple[NDArray[np.float64], bool]:
    """
    Return finite class scores of shape ``(n_items, n_classes)`` and whether ``predictions`` were class decisions.

    A 1-D array is class decisions, one per item: integer indices of declared classes, returned as one-hot rows.

    Raises
    ------
    ValueError
        If class decisions are not integers (booleans and floats included), not one per item or outside
        ``[0, n_classes)``, or class scores are misshapen or not finite.
    """
    arr = np.asarray(predictions)
    if arr.ndim == 1:
        return _one_hot_decisions(arr, n_items, n_classes), True
    scores = arr.astype(np.float64)
    expected = (n_items, n_classes)
    if scores.shape != expected:
        raise ValueError(
            f"predictions must be class scores of shape {expected} aligned with the dataset, or 1-D integer class "
            f"indices, got {scores.shape}."
        )
    if not np.isfinite(scores).all():
        raise ValueError("predictions must be finite; found NaN or infinite scores.")
    return scores, False


def _user(
    call: _Call, specs: Sequence[TargetSpec], predictions: Any, provenance: Mapping[str, Any] | None
) -> dict[str, Evidence]:
    declared = {"reads": "full_input", **(provenance or {})}
    _check_leak(call, declared.get("training_items"))
    scores, decisions = check_predictions(predictions, len(call.units.item_ids), call.units.n_classes)
    scores = scores[call.units.item_indices]
    source = _Scores(call, scores[call.fit], scores[call.audit], decisions)
    scope: Scope = _READS_SCOPE.get(declared["reads"], "full_input")  # type: ignore[assignment]
    details = {"assumptions": ["User-declared provenance is recorded, not verified."]}
    if decisions:
        details["assumptions"].append("The class decisions follow a policy frozen before the audit (declared).")
    item = call.item(
        _USER, "witness", scope, specs, source.bound, sample="audit split", provenance=declared, details=details
    )
    return {_USER: item}


# --- entry points ----------------------------------------------------------------------------------------------------


def count_claims(
    specs: Sequence[TargetSpec], *, has_replicas: bool, has_predictions: bool, builtin_witnesses: bool
) -> int:
    """
    Count the (target, evidence source) claims :func:`gather_evidence` will build, for ``allocate``.

    Per target: local panels 1 (0 for ``frame_pooled``, whose frame-weighted panels are not built yet), replicas 1,
    built-in witnesses 2 (k-NN and probe), the user witness 1, each when present.
    """
    per_target = int(has_replicas) + 2 * int(builtin_witnesses) + int(has_predictions)
    return sum(per_target + int(s.aggregation != "frame_pooled") for s in specs)


def gather_evidence(
    specs: Sequence[TargetSpec],
    *,
    units: Units,
    features: NDArray[Any],
    audit: NDArray[np.intp],
    fit: NDArray[np.intp],
    replicas: Mapping[int, NDArray[np.intp]] | None,
    predictions: Any,
    provenance: Mapping[str, Any] | None,
    allowances: Mapping[str, Any] | None,
    alpha_claim: float,
    seed: int,
    sampling: SamplingPlan | None,
    look: int,
    max_sequence_length: int | None,
    n_cells: int | None = None,
    panel_size: int | None = None,
    builtin_witnesses: bool,
) -> dict[str, Evidence]:
    """
    Build every classification evidence source, one endpoint per target it bounds, each at ``alpha_claim``.

    ``features`` are aligned with labeled items (row i is ``units.labels[i]``); ``fit`` and ``audit`` are labeled-item
    positions (``audit`` already the sampling plan's prefix). ``replicas`` maps labeled-item positions to extra labels
    (``resolve_replicas``); ``predictions`` are finite class scores of shape ``(len(dataset), n_classes)`` or 1-D
    integer class decisions (see :func:`check_predictions`). The partition's cell count follows the number of audit
    units (groups when grouped), or the plan's final size in units, so it is frozen across looks. Grouped (video) local
    panels stay ``partition_restricted`` whatever the allowances. ``alpha_claim`` must be
    ``allocate(alpha_call, count_claims(...))`` with the same sources (``builtin_witnesses`` included) for the call's
    ledger to hold.

    Raises
    ------
    ValueError
        If a group is longer than ``max_sequence_length``, ``predictions`` fails :func:`check_predictions`, or its
        declared ``training_items`` share a physical group with an audit unit.
    """
    lengths = np.bincount(units.groups) if len(units.groups) else np.zeros(1, dtype=np.intp)
    if max_sequence_length is not None and lengths.max() > max_sequence_length:
        raise ValueError(
            f"A group has {lengths.max()} items, above the declared max_sequence_length={max_sequence_length}."
        )
    details = {} if sampling is None else {"sampling_plan": sampling.sizes, "look": look}
    call = _Call(units, np.asarray(features), audit, fit, alpha_claim, max_sequence_length, details)
    planned = len(np.unique(units.groups[audit])) if sampling is None else sampling.sizes[-1]  # in units
    provisional = [name for name, given in (("n_cells", n_cells), ("panel_size", panel_size)) if given is None]
    sizes = (n_cells or n_cells_for(planned), panel_size or PANEL_SIZE, provisional)
    partition_ss, panel_ss = np.random.SeedSequence(seed).spawn(2)  # fixed order: deterministic per source
    partition_seed = int(partition_ss.generate_state(1)[0])
    evidence = _local_panels(call, specs, allowances or {}, sizes, partition_seed, np.random.default_rng(panel_ss))
    if replicas:
        evidence |= _replica(call, specs, replicas)
    if builtin_witnesses:
        evidence |= _builtin(call, specs)
    if predictions is not None:
        evidence |= _user(call, specs, predictions, provenance)
    return evidence
