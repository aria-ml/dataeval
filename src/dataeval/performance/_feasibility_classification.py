"""
Classification compositions for the feasibility evaluator: every evidence source, built from the core bounds.

Sources: post-stratified local-panel envelopes (C3), replica envelopes, built-in frozen witnesses (k-NN and a linear
probe fitted on the fit split) and a user witness from class scores or class decisions. Each source gives one ``Bound``
per requirement it bounds (local panels judge accuracy and the cost-weighted AP and recall at different scopes when
allowances widen the first), each spent at the claim's alpha and split inside the bound: C3 in halves; an AP envelope
equally over its positive kink costs and one prevalence upper endpoint; a recall envelope equally over its positive
kink costs and two prevalence endpoints (lower and upper); an AP witness over a TP lower and an FP upper endpoint per
threshold and one prevalence upper endpoint; a macro AP over its classes. A witness gives a recall requirement two
bounds instead, at half the claim's alpha each: recall (lower) and the false-positive rate the requirement limits
(upper).
"""

__all__ = []

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, cast, get_args

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
    PANEL_SIZE,
    WITNESS_POS_PREVALENCE,
    WITNESS_QUANTILES,
    n_cells_for,
)
from dataeval.performance._feasibility_evidence import (
    Assumption,
    Bound,
    Quantity,
    Reason,
    Role,
    Sample,
    Source,
    State,
)
from dataeval.performance._feasibility_target import Aggregation, Allowances, Scope, TargetSpec
from dataeval.performance._feasibility_units import Units

# value (None: unavailable, its reason in raw), the bound's stats (None: no closed form), raw
_Bound = tuple[float | None, dict[str, Any] | None, dict[str, Any]]
_ClassBound = tuple[float, dict[str, Any]]  # one class's AP or recall endpoint and its raw results
_RiskAt = Callable[[float, float], tuple[float, Mapping[str, Any]]]  # (cost, delta) -> risk lower bound, raw result
_OneVsRest = tuple[NDArray[np.float64], _RiskAt, NDArray[np.float64]]  # a class's cost grid, risk bound, prevalence
_Side = Literal["lower", "upper"]
_Method = Literal["hoeffding", "hybrid", "exact_binomial"]
_TRIVIAL = 1e-12
_STAT_KEYS = ("mean", "variance", "n", "support", "delta", "method")  # the exact interval's record; it has no radius
_MODELS: dict[Source, Callable[[], Any]] = {
    "knn": lambda: KNeighborsClassifier(n_neighbors=5),
    "linear_probe": lambda: LogisticRegression(max_iter=1000),
}
_SOURCES = get_args(Source)  # the order of the bounds table's rows


class _UnavailableError(Exception):
    """A bound that cannot be computed; ``reason`` says why."""

    def __init__(self, reason: Reason) -> None:
        super().__init__(reason)
        self.reason: Reason = reason


def _need_units(n: int) -> None:
    if n < 2:
        raise _UnavailableError("too_few_units")


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
        raise _UnavailableError("max_sequence_length_missing")  # a frame_pooled total needs a declared bound
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


def _radius(r: EndpointResult) -> float | None:
    """Return the half-width the endpoint's method applied to the mean; None for an exact interval, which has none."""
    if r["method"] == "exact_binomial":
        return None
    if r["method"] == "hoeffding":
        return r["radius_hoeffding"]
    return min(x for x in (r["radius_hoeffding"], r["radius_bernstein"]) if x is not None)  # hybrid: the smaller one


def _accuracy_bound(
    values: NDArray[np.float64],
    groups: NDArray[np.intp],
    spec: TargetSpec,
    support: float,
    call: "_Call",
    loss: bool,
    method: _Method = "hybrid",
) -> _Bound:
    """Accuracy from per-item correctness (witness) or consensus loss (``loss``: the envelope is 1 - the risk lower)."""
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
    stats = {k: r[k] for k in ("variance", "n", "support", "delta", "method")}
    stats |= {"mean": _flip(r["mean"], loss), "radius": _radius(r)}
    return _flip(r["endpoint"], loss), stats, {**r, "deltas": [call.alpha]}


def _ap_ceiling(
    costs: NDArray[np.float64],
    risk_at: _RiskAt,
    prevalence: NDArray[np.float64],
    scale: int,
    delta: float,
    method: _Method = "hybrid",
) -> _ClassBound:
    """Envelope AP: ``delta`` split equally over the positive costs and one prevalence upper endpoint.

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
                raise _UnavailableError("no_positives_possible")
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
    """Recall envelope ``U_r``: ``delta`` split equally over the positive costs and two prevalence endpoints.

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


def _state(value: float | None, upper: bool, reason: Reason | None) -> tuple[State, Reason | None]:
    """Name a bound's state: unavailable without a value, uninformative at the trivial end, else informative."""
    if value is None:
        return "unavailable", reason
    if value >= 1 - _TRIVIAL if upper else value <= _TRIVIAL:
        return "uninformative", "bound_at_maximum" if upper else "bound_at_minimum"
    return "informative", None


def _plain(value: Any, kind: type) -> Any:
    """Return a numpy scalar as a Python ``kind``; None stays None."""
    return None if value is None else kind(value)


def _make_bound(
    shared: Mapping[str, Any],
    quantity: Quantity,
    value: float | None,
    stats: Mapping[str, Any] | None,
    alpha: float,
    reason: Reason | None,
) -> Bound:
    """One ``Bound`` from a computed value and the stats recorded beside it (None: no closed form)."""
    role: Role = shared["role"]
    upper = role == "envelope" or quantity == "false_positive_rate"
    state, why = _state(value, upper, reason)
    s = stats or {}
    return Bound(
        **shared,
        quantity=quantity,
        bound_type="upper" if upper else "lower",
        state=state,
        reason=why,
        bound=_plain(value, float),
        observed=_plain(s.get("mean"), float) if role == "witness" else None,
        n=_plain(s.get("n"), int),
        variance=_plain(s.get("variance"), float),
        radius=_plain(s.get("radius"), float),
        method=s.get("method"),
        alpha=alpha,
        mean=_plain(s.get("mean"), float),
        support=_plain(s.get("support"), float),
        delta=_plain(s.get("delta"), float),
    )


def _rate(hits: NDArray[np.bool_], delta: float, side: _Side, empty: Reason) -> dict[str, Any]:
    """Exact binomial endpoint on the share of ``hits`` (decisions for the class) in one audit population."""
    if not len(hits):
        return {"value": None, "stats": None, "reason": empty}
    r = binomial_endpoint(int(hits.sum()), len(hits), delta, side=side)
    return {"value": r["endpoint"], "stats": {k: r[k] for k in _STAT_KEYS}, "reason": None}


@dataclass(frozen=True)
class _Call:
    """What every source of one call shares: units, features (aligned with labeled items), splits and claim alpha."""

    units: Units
    features: NDArray[Any]
    audit: NDArray[np.intp]
    fit: NDArray[np.intp]
    alpha: float
    max_len: int | None
    requirements: tuple[TargetSpec, ...]  # the order the caller gave; a bound's ``requirement`` indexes it

    def item(
        self,
        source: Source,
        role: Role,
        scope: Scope,
        specs: Sequence[TargetSpec],
        bound: Callable[[TargetSpec], _Bound],
        *,
        sample: Sample,
        assumptions: Callable[[TargetSpec], Iterable[Assumption]],
    ) -> list[Bound]:
        """Bound each of ``specs`` from one source: one bound each, two for a recall witness."""
        out: list[Bound] = []
        for spec in specs:
            reason: Reason | None = None
            try:
                value, stats, raw = bound(spec)
            except _UnavailableError as e:
                value, stats, raw, reason = None, None, None, e.reason
            spent = 0.0 if raw is None else math.fsum(raw["deltas"])
            if spent > self.alpha + 1e-12:
                raise RuntimeError(
                    f"Internal error: {source} splits {spent} inside a {spec.metric} bound, over its claim {self.alpha}"
                )
            shared = {
                "requirement": self.requirements.index(spec),
                "source": source,
                "role": role,
                "information_scope": scope,
                "population": self.units.population,
                "sample": sample,
                "unit": "sequence" if self.units.grouped else "image",
                "assumptions": tuple(assumptions(spec)),
            }
            if role == "envelope" or spec.metric != "recall":
                out.append(_make_bound(shared, spec.metric, value, stats, self.alpha, reason))
                continue
            # A recall witness bounds recall (lower) and the false-positive rate the target limits (upper), half each.
            missing = {"value": None, "stats": None, "reason": reason}
            rec, fp_rate = (missing, missing) if raw is None else (raw, raw["false_positive_rate"])
            out += [
                _make_bound(shared, "recall", rec["value"], rec["stats"], self.alpha / 2, rec["reason"]),
                _make_bound(
                    shared, "false_positive_rate", fp_rate["value"], fp_rate["stats"], self.alpha / 2, fp_rate["reason"]
                ),
            ]
        return out


@dataclass(frozen=True)
class _Missing:
    """A source whose shared setup failed: every bound is unavailable for the same reason."""

    reason: Reason

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
            "radius": r["within_radius"] + r["between_radius"],
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
        return _Missing("fit_split_smaller_than_cells"), items
    if len(items) < 2:
        return _Missing("too_few_units"), items
    cells = fit_partition(call.features[call.fit], n_cells, partition_seed)(call.features[items])
    panels = {h: ix for h, ix in disjoint_panels(cells, m, rng).items() if len(ix)}
    if not panels:
        return _Missing("no_complete_panel"), items
    n_by_cell = np.bincount(cells, minlength=n_cells)
    return _Panels(call, call.units.labels[items], n_by_cell, panels, m, *beta_gamma), items


def _accuracy_scope(allowances: Allowances | None, grouped: bool) -> Scope:
    if grouped or allowances is None:  # single-frame panels stay partition_restricted for video (phase 1)
        return "partition_restricted"
    return "fixed_representation" if allowances.gamma is None else "full_input"


def _local_panels(
    call: _Call,
    specs: Sequence[TargetSpec],
    allowances: Allowances | None,
    sizes: tuple[int, int],
    partition_seed: int,
    rng: np.random.Generator,
) -> list[Bound]:
    """C3 envelopes; accuracy takes beta (fixed_representation) and gamma (full_input), AP and recall do not."""
    specs = [s for s in specs if s.aggregation != "frame_pooled"]  # frame-weighted panels are not built yet
    if not specs:
        return []
    n_cells, m = sizes
    scope = _accuracy_scope(allowances, call.units.grouped)
    beta = None if allowances is None or scope == "partition_restricted" else allowances.beta
    gamma = (allowances.gamma or 0.0) if allowances is not None and scope == "full_input" else 0.0
    source, _ = _draw_panels(call, n_cells, m, partition_seed, rng, (beta, gamma))
    base: tuple[Assumption, ...] = ("representation_independent_of_audit_labels",)
    # (judged scope, requirements, what they rest on): allowances widen the accuracy envelope's scope and nothing else
    parts: list[tuple[Scope, list[TargetSpec], tuple[Assumption, ...]]] = [(scope, specs, base)]
    if scope != "partition_restricted":
        widened: tuple[Assumption, ...] = ("allowance_beta", *(("allowance_gamma",) if scope == "full_input" else ()))
        parts = [
            (scope, [s for s in specs if s.metric == "accuracy"], (*base, *widened)),
            ("partition_restricted", [s for s in specs if s.metric != "accuracy"], base),  # AP, recall: cost-weighted
        ]
    sample: Sample = "audit_split_one_per_group" if call.units.grouped else "audit_split"
    return [
        b
        for s, part, assumed in parts
        if part
        for b in call.item(
            "local_panels", "envelope", s, part, source.bound, sample=sample, assumptions=lambda _, a=assumed: a
        )
    ]


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


def _replica_assumptions(call: _Call, spec: TargetSpec) -> list[Assumption]:
    out: list[Assumption] = ["replicas_random_sample", "replicas_one_protocol"]
    if call.units.grouped:
        out.append("replicas_same_video_context")
    if spec.aggregation == "sequence_mean":
        out.append("replicated_sequences_random_sample")
    if spec.aggregation == "frame_pooled":
        out.append("replicated_frames_uniform_sample")
    return out


def _replica(call: _Call, specs: Sequence[TargetSpec], replicas: Mapping[int, NDArray[np.intp]]) -> list[Bound]:
    items = np.array(sorted(replicas), dtype=np.intp)
    rows = [np.r_[call.units.labels[i], np.asarray(replicas[i], dtype=np.intp)] for i in items]
    source = _Replicas(call, rows, np.array([len(r) for r in rows]), call.units.groups[items])
    return call.item(
        "replicas",
        "envelope",
        "full_input",
        specs,
        source.bound,
        sample="replica_items",
        assumptions=lambda spec: _replica_assumptions(call, spec),
    )


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
                raise _UnavailableError("needs_scores")  # class decisions rank nothing
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
        recall = _rate(hit[positive], d, "lower", "no_positives")
        fp_rate = _rate(hit[~positive], d, "upper", "no_negatives")
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
    except ValueError:  # too few fit units, a single fit class or an empty audit
        return _Missing("witness_not_fitted")
    n = call.units.n_classes
    return _Scores(call, _onto_classes(p_fit, model.classes_, n), _onto_classes(p_audit, model.classes_, n))


def _builtin(call: _Call, specs: Sequence[TargetSpec]) -> list[Bound]:
    """Build the built-in witnesses, each fitted on the fit split only and frozen before the audit."""
    out: list[Bound] = []
    assumed: tuple[Assumption, ...] = ("representation_independent_of_audit_labels",)
    for name, make in _MODELS.items():
        source = _fit_scores(call, make())
        out += call.item(
            name,
            "witness",
            "fixed_representation",
            specs,
            source.bound,
            sample="audit_split",
            assumptions=lambda _: assumed,
        )
    return out


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
    call: _Call, specs: Sequence[TargetSpec], predictions: Any, class_order_verified: bool = False
) -> list[Bound]:
    scores, decisions = check_predictions(predictions, len(call.units.item_ids), call.units.n_classes)
    scores = scores[call.units.item_indices]
    source = _Scores(call, scores[call.fit], scores[call.audit], decisions)
    # The scores follow the dataset's declared class order: assumed, unless a model's own mapping matched it.
    assumed: tuple[Assumption, ...] = (
        ("model_independent_of_audit_items",)
        if class_order_verified
        else ("model_independent_of_audit_items", "class_order_matches_dataset")
    )
    # A user witness may read the whole input, so it counts at full_input only.
    return call.item(
        "user_model", "witness", "full_input", specs, source.bound, sample="audit_split", assumptions=lambda _: assumed
    )


# --- entry points ----------------------------------------------------------------------------------------------------


def count_claims(
    specs: Sequence[TargetSpec],
    *,
    has_replicas: bool,
    has_predictions: bool,
    builtin_witnesses: bool,
    has_embeddings: bool = True,
) -> int:
    """
    Count the (target, evidence source) claims :func:`gather_evidence` will build, for ``allocate``.

    Per target: local panels 1 (0 for ``frame_pooled``, whose frame-weighted panels are not built yet), replicas 1,
    built-in witnesses 2 (k-NN and probe), the user witness 1, each when present. Local panels and the built-in
    witnesses need embeddings, so a call without them counts neither.
    """
    per_target = int(has_replicas) + 2 * int(builtin_witnesses and has_embeddings) + int(has_predictions)
    return sum(per_target + int(has_embeddings and s.aggregation != "frame_pooled") for s in specs)


def _check_group_lengths(units: Units, max_sequence_length: int | None) -> None:
    """Raise when a group is longer than the declared ``max_sequence_length``; nothing is truncated."""
    lengths = np.bincount(units.groups) if len(units.groups) else np.zeros(1, dtype=np.intp)
    if max_sequence_length is not None and lengths.max() > max_sequence_length:
        raise ValueError(
            f"A group has {lengths.max()} items, above the declared max_sequence_length={max_sequence_length}."
        )


def gather_evidence(
    specs: Sequence[TargetSpec],
    *,
    units: Units,
    features: NDArray[Any] | None,
    audit: NDArray[np.intp],
    fit: NDArray[np.intp],
    replicas: Mapping[int, NDArray[np.intp]] | None,
    predictions: Any,
    allowances: Allowances | None,
    alpha_claim: float,
    seed: int,
    max_sequence_length: int | None,
    n_cells: int | None = None,
    panel_size: int | None = None,
    builtin_witnesses: bool = True,
    class_order_verified: bool = False,
) -> list[Bound]:
    """
    Build every classification evidence source, one bound per requirement it bounds, each at ``alpha_claim``.

    The bounds come back ordered by requirement, then source (local panels, replicas, k-NN, linear probe, user model),
    a witness's recall or metric bound before its false-positive bound. A requirement is an index into ``specs``.

    ``features`` are aligned with labeled items (row i is ``units.labels[i]``); ``fit`` and ``audit`` are labeled-item
    positions. ``replicas`` maps labeled-item positions to extra labels (``resolve_replicas``); ``predictions`` are
    finite class scores of shape ``(len(dataset), n_classes)`` or 1-D integer class decisions (see
    :func:`check_predictions`). Without ``features`` there are no local panels and no built-in witnesses. The
    partition's cell count follows the number of audit units (groups when grouped). Grouped (video) local panels stay
    ``partition_restricted`` whatever the allowances. ``alpha_claim`` must be ``allocate(alpha_call,
    count_claims(...))`` with the same sources (``builtin_witnesses`` and embeddings included) for the call's ledger
    to hold. ``class_order_verified`` says the user model's own class mapping matched the dataset's, which leaves
    ``class_order_matches_dataset`` off its bounds. ``n_cells``, ``panel_size`` and ``builtin_witnesses`` are internal
    knobs, kept so the numerical pin can run at this layer; the evaluator leaves them at their defaults.

    Raises
    ------
    ValueError
        If a group is longer than ``max_sequence_length`` or ``predictions`` fails :func:`check_predictions`.
    """
    _check_group_lengths(units, max_sequence_length)
    # Without embeddings the two sources that read them are skipped, so the empty placeholder is never read.
    placeholder = np.empty((len(units.labels), 0)) if features is None else np.asarray(features)
    call = _Call(units, placeholder, audit, fit, alpha_claim, max_sequence_length, tuple(specs))
    planned = len(np.unique(units.groups[audit]))  # in units
    sizes = (n_cells or n_cells_for(planned), panel_size or PANEL_SIZE)
    partition_ss, panel_ss = np.random.SeedSequence(seed).spawn(2)  # fixed order: deterministic per source
    partition_seed = int(partition_ss.generate_state(1)[0])
    bounds: list[Bound] = []
    if features is not None:
        bounds += _local_panels(call, specs, allowances, sizes, partition_seed, np.random.default_rng(panel_ss))
    if replicas:
        bounds += _replica(call, specs, replicas)
    if builtin_witnesses and features is not None:
        bounds += _builtin(call, specs)
    if predictions is not None:
        bounds += _user(call, specs, predictions, class_order_verified)
    return sorted(bounds, key=lambda b: (b.requirement, _SOURCES.index(b.source), b.quantity == "false_positive_rate"))
