"""Ordered decision procedure for feasibility targets and the recommendation behind non-final determinations."""

__all__ = []

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, NamedTuple, cast

from dataeval.core._concentration import units_to_cross
from dataeval.performance._feasibility_evidence import Endpoint, Evidence, SamplingPlan, stats_key
from dataeval.performance._feasibility_target import Scope, TargetSpec

Determination = Literal[
    "evidence_conflict",
    "demonstrated_feasible",
    "infeasible_under_recorded_assumptions",
    "not_ruled_out",
    "unresolved",
]
LimitingFactor = Literal[
    "sampling_width", "no_replicas", "scope_restricted", "no_witness", "missing_bound", "labels_conditioned", "conflict"
]

_FINAL = ("demonstrated_feasible", "infeasible_under_recorded_assumptions")
# full_input ⊃ fixed_representation ⊃ partition_restricted: a ceiling decides at its scope and every narrower one.
_RANK = {"partition_restricted": 0, "fixed_representation": 1, "full_input": 2}
# A witness's model class from provenance["reads"] (unknown: full_input); it is eligible at that scope and broader.
_READS_SCOPE = {"partition": "partition_restricted", "representation": "fixed_representation"}
_WANTED_WITNESS = {
    "full_input": "a frozen model's predictions",
    "fixed_representation": "predictions from a frozen model reading only the given representation "
    "(provenance reads='representation')",
    "partition_restricted": "predictions from a model giving one answer per cell (provenance reads='partition')",
}
_DECLARED_BOUNDS = ("max_sequence_length", "target_count_bound")  # task assumptions an endpoint can be missing


@dataclass(frozen=True)
class Decision:
    """
    The determination for one target, the endpoints behind it and the recommendation for a non-final one.

    ``ceiling`` is the smallest available ceiling at the judged scope or broader, ``witness`` the largest available
    eligible witness (for recall, certified at the false-positive limit) and ``witness_source`` its evidence id; for a
    conflict, both are the conflicting pair.
    ``evidence_ids`` lists every evidence item with an endpoint for the target on the population, deciding ones first.
    """

    determination: Determination
    ceiling: float | None
    ceiling_scope: Scope | None
    witness: float | None
    witness_source: str | None
    reason: str
    limiting_factor: LimitingFactor | None
    recommendation: str | None
    suggested_minimum: int | None
    evidence_ids: tuple[str, ...]


class _Item(NamedTuple):
    eid: str
    ev: Evidence
    ep: Endpoint

    @property
    def value(self) -> float:
        return math.nan if self.ep.value is None else self.ep.value  # callers read available endpoints only

    @property
    def name(self) -> str:
        return f"{self.ev.kind} {self.eid!r} ({self.ev.information_scope})"


class _Pool(NamedTuple):
    items: list[_Item]  # every endpoint for the target on the population, in evidence order
    ceilings: list[_Item]  # at the judged scope or broader
    narrower: list[_Item]  # ceilings narrower than the judged scope: evidence only, never decide
    witnesses: list[_Item]  # eligible for the judged scope and certified (recall: within the false-positive limit)
    ceiling: _Item | None  # smallest available eligible ceiling
    witness: _Item | None  # largest available eligible, certified witness


def _witness_eligible(ev: Evidence, scope: Scope) -> bool:
    reads = "representation" if ev.kind.startswith("builtin_witness") else ev.provenance.get("reads", "full_input")
    return _RANK[_READS_SCOPE.get(reads, "full_input")] <= _RANK[scope]


def _false_positives(item: _Item) -> Endpoint | None:
    """Return the false-positive endpoint the same evidence gives for the same target, if any."""
    return next(
        (e for e in item.ev.endpoints if e.target_spec == item.ep.target_spec and e.quantity == "false_positive_rate"),
        None,
    )


def _limit(spec: TargetSpec) -> float:
    return cast(float, spec.max_false_positive_rate)  # a recall target always sets its limit


def _certified(item: _Item, spec: TargetSpec) -> bool:
    """Whether a witness counts for ``spec``: for recall, only when its own false-positive bound is within the limit."""
    if spec.metric != "recall":
        return True
    fp = _false_positives(item)
    return fp is not None and fp.value is not None and fp.value <= _limit(spec)


def _tested(p: _Pool, scope: Scope) -> list[_Item]:
    """Every available witness eligible for ``scope``, certified or not."""
    witnesses = (i for i in p.items if i.ev.supports == "witness" and i.ep.value is not None)
    return [i for i in witnesses if _witness_eligible(i.ev, scope)]


def _observed_meets(item: _Item, spec: TargetSpec) -> bool | None:
    """
    Whether a witness's recorded point estimates already meet the target; None when they are not recorded.

    Accuracy and AP must exceed the threshold; recall must reach it with the false-positive rate within the limit.
    """
    entries, key = item.ev.details.get("endpoints", {}), item.ep.target_spec
    pos = entries.get(key)
    if spec.metric != "recall":
        return None if pos is None else pos["mean"] > spec.threshold
    neg = entries.get(stats_key(key, "false_positive_rate"))
    return None if not (pos and neg) else pos["mean"] >= spec.threshold and neg["mean"] <= _limit(spec)


def _pool(spec: TargetSpec, evidence: Mapping[str, Evidence], population: str) -> _Pool:
    key, rank = spec.canonical(), _RANK[spec.scope]
    items = [
        _Item(eid, ev, ep)
        for eid, ev in evidence.items()
        if ev.population == population
        for ep in ev.endpoints
        if ep.target_spec == key and ep.quantity == "metric"
    ]
    ceilings = [i for i in items if i.ev.supports == "ceiling" and _RANK[i.ev.information_scope] >= rank]
    narrower = [i for i in items if i.ev.supports == "ceiling" and _RANK[i.ev.information_scope] < rank]
    witnesses = [
        i for i in items if i.ev.supports == "witness" and _witness_eligible(i.ev, spec.scope) and _certified(i, spec)
    ]
    return _Pool(
        items,
        ceilings,
        narrower,
        witnesses,
        min((i for i in ceilings if i.ep.value is not None), key=lambda i: i.value, default=None),
        max((i for i in witnesses if i.ep.value is not None), key=lambda i: i.value, default=None),
    )


def _unresolved_reason(spec: TargetSpec, p: _Pool) -> str:
    if p.ceilings:
        states = "; ".join(f"{i.name} {i.ep.state}" + (f": {i.ep.reason}" if i.ep.reason else "") for i in p.ceilings)
        ceiling = f"every ceiling is unavailable or uninformative ({states})"
    else:
        ceiling = f"no ceiling at {spec.scope} or broader"
    if p.narrower:
        ceiling += f"; narrower ceilings cannot decide a {spec.scope} target ({', '.join(i.name for i in p.narrower)})"
    if p.witness is None and _tested(p, spec.scope):  # recall only: none is certified
        return f"{ceiling}; no eligible witness has a false-positive upper endpoint within {_limit(spec):.4g}"
    if p.witness is None:
        return f"{ceiling}; no witness eligible for {spec.scope}"
    return f"{ceiling}; best eligible witness {p.witness.name} at {p.witness.value:.4g} is below {spec.threshold:.4g}"


def decide(spec: TargetSpec, evidence: Mapping[str, Evidence], population: str) -> Decision:
    """
    Apply the ordered rules to the endpoints for ``spec`` on ``population`` and attach the recommendation.

    Only evidence whose ``population`` is ``population`` and endpoints whose ``target_spec`` is ``spec.canonical()``
    count; other populations are left out of ``evidence_ids`` too. Ceilings count at the judged scope or broader
    (``full_input`` ⊃ ``fixed_representation`` ⊃ ``partition_restricted``); narrower ones are listed in
    ``evidence_ids`` but never decide. A witness belongs to the model class of what it reads and is eligible at that
    scope and broader: ``provenance["reads"] == "partition"`` (one answer per cell) everywhere, built-in witnesses
    and ``"representation"`` at ``fixed_representation`` and ``full_input``, anything else at ``full_input`` only.
    For a recall target, a witness counts only when certified: the same evidence's false-positive upper endpoint for
    the target is available and at or below ``max_false_positive_rate``. The false-positive endpoint itself never
    enters a comparison. The first matching rule decides:

    1. ``evidence_conflict`` — a (certified) witness eligible for some deciding-scope ceiling's own scope has a lower
       endpoint above that ceiling's upper endpoint, whatever the threshold;
    2. ``demonstrated_feasible`` — the largest witness lower endpoint is at or above the threshold;
    3. ``infeasible_under_recorded_assumptions`` — the smallest ceiling upper endpoint is below the threshold;
    4. ``not_ruled_out`` — an informative ceiling's upper endpoint is at or above the threshold;
    5. ``unresolved`` — otherwise.
    """
    p = _pool(spec, evidence, population)
    t, c, w = spec.threshold, p.ceiling, p.witness
    available_ceilings = [i for i in p.ceilings if i.ep.value is not None]
    conflicts = [
        (wi, ci)
        for ci in available_ceilings
        for wi in p.items
        if wi.ev.supports == "witness"
        and wi.ep.value is not None
        and _witness_eligible(wi.ev, ci.ev.information_scope)
        and _certified(wi, spec)
        and wi.value > ci.value
    ]
    informative = [i for i in available_ceilings if i.ep.state == "informative"]
    determination: Determination
    if conflicts:
        w, c = max(conflicts, key=lambda pair: pair[0].value - pair[1].value)
        determination, deciding = "evidence_conflict", (w, c)
        reason = (
            f"witness {w.name} lower endpoint {w.value:.4g} exceeds ceiling {c.name} upper endpoint {c.value:.4g} "
            f"on {population!r}; the determination is suspended"
        )
    elif w is not None and w.value >= t:
        determination, deciding = "demonstrated_feasible", (w,)
        reason = f"witness {w.name} lower endpoint {w.value:.4g} is at or above the threshold {t:.4g}"
    elif c is not None and c.value < t:
        determination, deciding = "infeasible_under_recorded_assumptions", (c,)
        reason = f"ceiling {c.name} upper endpoint {c.value:.4g} is below the threshold {t:.4g}"
    elif informative:
        c = min(informative, key=lambda i: i.value)
        determination, deciding = "not_ruled_out", (c,)
        reason = (
            f"informative ceiling {c.name} upper endpoint {c.value:.4g} is at or above the threshold {t:.4g} "
            "and no eligible witness reaches it"
        )
    else:
        determination, deciding = "unresolved", tuple(i for i in (c, w) if i is not None)
        reason = _unresolved_reason(spec, p)
    decision = Decision(
        determination=determination,
        ceiling=None if c is None else c.value,
        ceiling_scope=None if c is None else c.ev.information_scope,
        witness=None if w is None else w.value,
        witness_source=None if w is None else w.eid,
        reason=reason,
        limiting_factor=None,
        recommendation=None,
        suggested_minimum=None,
        evidence_ids=tuple(dict.fromkeys(i.eid for i in (*deciding, *p.items))),
    )
    factor, text, minimum = recommend(spec, evidence, decision, population)
    return replace(decision, limiting_factor=factor, recommendation=text, suggested_minimum=minimum)


def _to_cross(stats: Mapping[str, Any], observed: float, gap: float, side: Literal["lower", "upper"]) -> int | None:
    try:
        observed = min(max(observed, 0.0), stats["support"])
        return units_to_cross(
            observed, stats["variance"], stats["support"], stats["delta"], gap, method=stats["method"], side=side
        )
    except (ValueError, TypeError):  # stats it cannot invert (ratio method, an invalid support, no gap): no estimate
        return None


def _on_plan(item: _Item, n: int | None) -> int | None:
    """Round ``n`` up to the next size of the sampling plan that counts the endpoint's units; None beyond the plan."""
    plan = _plan(item)
    if n is None or plan is None:
        return n
    sizes = plan.sizes if isinstance(plan, SamplingPlan) else plan
    return next((s for s in sizes if s >= n), None)


def _units(item: _Item, stats: Mapping[str, Any], gap: float) -> int | None:
    # The recorded mean is on the metric's scale; the support is the observation's. A ceiling observes a loss.
    observed = 1 - stats["mean"] if item.ep.direction == "upper" else stats["mean"]
    return _on_plan(item, _to_cross(stats, observed, gap, "lower"))


def _plan(item: _Item) -> Any:
    """Return the sampling plan whose sizes count this endpoint's units: audit-split samples only, not replicas."""
    return item.ev.details.get("sampling_plan") if item.ev.sample.startswith("audit split") else None


def _recall_crossing(p: _Pool, spec: TargetSpec) -> tuple[_Item, int | None] | None:
    """
    Return the witness whose observed recall and false-positive rate already meet the target, with its images.

    Among scope-eligible witnesses (certified or not) with observed recall at or above the threshold and observed
    false-positive rate at or below the limit, the one with the largest recall lower endpoint. Each bound on the wrong
    side of its limit needs its own units (positives, or negatives through the upper bound); the estimated prevalence
    from the two counts converts both to images, the larger wins, rounded up to the sampling plan. None when a needed
    count is not attainable.
    """
    r0, f0 = spec.threshold, _limit(spec)
    found = []
    for item in _tested(p, spec.scope):
        fp, key, entries = _false_positives(item), item.ep.target_spec, item.ev.details["endpoints"]
        if fp is not None and fp.value is not None and _observed_meets(item, spec):
            found.append((item, fp.value, entries[key], entries[stats_key(key, "false_positive_rate")]))
    if not found:
        return None
    item, fp_bound, pos, neg = max(found, key=lambda f: f[0].value)
    positives = _to_cross(pos, pos["mean"], pos["mean"] - r0, "lower") if item.value < r0 else 0
    negatives = _to_cross(neg, neg["mean"], f0 - neg["mean"], "upper") if fp_bound > f0 else 0
    if positives is None or negatives is None:
        return item, None
    share = pos["n"] / (pos["n"] + neg["n"])  # the estimated prevalence: both endpoints count audit items
    return item, _on_plan(item, max(math.ceil(positives / share), math.ceil(negatives / (1 - share))))


def _crossing(p: _Pool, spec: TargetSpec) -> tuple[_Item, int | None] | None:
    """Return the deciding ceiling or best witness whose recorded mean is already past the threshold, with its units."""
    if spec.metric == "recall":  # a recall ceiling has no single recorded mean to project; the witness side decides
        return _recall_crossing(p, spec)
    threshold = spec.threshold
    for item, sign in ((p.ceiling, -1), (p.witness, 1)):
        stats = (
            None if item is None else item.ev.details["endpoints"].get(stats_key(item.ep.target_spec, item.ep.quantity))
        )
        if item is not None and stats is not None and sign * (stats["mean"] - threshold) > 0:
            return item, _units(item, stats, abs(stats["mean"] - threshold))
    return None


def _names_bound(item: _Item) -> bool:
    return any(b in (item.ep.reason or "") for b in _DECLARED_BOUNDS)


def _factor(spec: TargetSpec, p: _Pool, crossing: bool) -> tuple[LimitingFactor, str]:  # noqa: C901
    missing = next((i for i in p.ceilings if i.ep.value is None and _names_bound(i)), None)
    narrower = [i for i in p.narrower if i.ep.value is not None]
    every = [*p.ceilings, *p.narrower]
    if not p.ceilings and narrower:
        ids = ", ".join(i.eid for i in narrower)
        broadest = max((i.ev.information_scope for i in narrower), key=_RANK.__getitem__)
        return "scope_restricted", (
            f"Only narrower ceilings bound this {spec.scope} target ({ids}). Collect repeated labels on a randomly "
            f"sampled subset of items for a full_input ceiling, declare a justified beta/Gamma allowance, or accept "
            f"the {broadest} claim."
        )
    if missing is not None:
        return "missing_bound", f"{missing.name} is unavailable: {missing.ep.reason}. Declare the bound."
    if every and all(i.ep.value is None and not _names_bound(i) for i in every):
        reasons = "; ".join(f"{i.name}: {i.ep.reason}" for i in every)
        return "sampling_width", (
            f"Every ceiling is unavailable for lack of units ({reasons}). Collect more independent units: enough for "
            "the fit split to cover the partition cells and for each cell's audit units to fill panels."
        )
    tested = _tested(p, spec.scope)
    if not tested and spec.scope == "partition_restricted":
        return "no_witness", (
            "No witness can be eligible for a partition_restricted target in this release: it would give one answer "
            "per cell of the internal partition, which is not exposed. Only a ceiling can decide this target."
        )
    if not tested:
        return "no_witness", (
            f"No witness eligible for {spec.scope} bounds this target; supply {_WANTED_WITNESS[spec.scope]} "
            "on held-out units."
        )
    if crossing:
        return "sampling_width", ""
    replicas = any(i.ev.kind.startswith("replica_ceiling") for i in p.items)
    if not replicas and all(i.ev.information_scope == "partition_restricted" for i in (*p.ceilings, *p.narrower)):
        return "no_replicas", (
            "No ceiling broader than partition_restricted bounds this target and no replica evidence exists; collect "
            "repeated labels on a randomly sampled subset of items."
        )
    return "no_witness", _weak(spec, p)


def _weak(spec: TargetSpec, p: _Pool) -> str:
    """Say why the eligible witnesses fall short and what to supply instead."""
    supply = "supply a stronger model" + (" or a different frozen operating point" if spec.metric == "recall" else "")
    supply += f": {_WANTED_WITNESS[spec.scope]} on held-out units."
    if p.witness is None:  # recall only: every eligible witness misses the false-positive limit
        return f"No eligible witness has a false-positive upper confidence bound within {_limit(spec):.4g}; {supply}"
    w = p.witness
    return f"The best eligible witness {w.name} reaches {w.value:.4g} against {spec.threshold:.4g}; {supply}"


def recommend(
    spec: TargetSpec, evidence: Mapping[str, Evidence], decision: Decision, population: str
) -> tuple[LimitingFactor | None, str | None, int | None]:
    """
    Name the limiting factor, the evidence that addresses it, and the suggested minimum number of units.

    Evidence counts as in :func:`decide`. Final determinations get ``(None, None, None)``; ``evidence_conflict`` gets
    ``"conflict"``, a recommendation to inspect the likely causes, and no minimum. Otherwise the first match wins:

    1. ``scope_restricted`` — no ceiling at the judged scope or broader, but an available narrower one;
    2. ``missing_bound`` — an eligible ceiling is unavailable and its reason names a declared bound
       (``max_sequence_length``, ``target_count_bound``);
    3. ``sampling_width`` — ceilings exist (at any scope) but every one is unavailable for lack of units or panels;
    4. ``no_witness`` — no available witness eligible for the judged scope, certified or not (at
       ``partition_restricted`` none can be in this release);
    5. ``sampling_width`` — the deciding ceiling's recorded mean is below the threshold, or the best witness's above
       it: the endpoint is on the wrong side by less than its radius (also when clipped to the metric's range). For a
       recall target, an eligible witness, certified or not, whose observed recall is at or above the threshold and
       observed false-positive rate at or below the limit, while a bound falls short;
    6. ``no_replicas`` — every ceiling at the judged scope, broader or narrower is ``partition_restricted`` (or there
       is none) and no replica evidence exists;
    7. ``no_witness`` — otherwise: the witness is weak (for recall: its observed recall is below the threshold or its
       observed false-positive rate above the limit; a stronger model or a different frozen operating point).

    When the population covers labeled items only, the recommendation ends with a note saying so; that never
    displaces the factor, so ``labels_conditioned`` is not produced.

    The suggested minimum is set whenever a crossing as in rule 5 exists, whatever the factor: ``units_to_cross`` at
    the endpoint's ``details["endpoints"]`` entry, whose ``mean`` is the point estimate on the metric's scale (inverted
    on the observation's scale: ``1 - mean`` for a ceiling, which observes a loss, clipped into the support), with
    gap ``|mean - threshold|``. For an endpoint whose sample is the audit split it is rounded up to the next size of a
    ``details["sampling_plan"]`` (a ``SamplingPlan`` or its sizes); a replica ceiling's units are not plan units, so
    its minimum is not rounded. It is None when not attainable: beyond ``units_to_cross``'s search, beyond the plan, or
    recorded stats it rejects (a ratio method, an invalid support). A recall target counts evaluation items: the
    positives its recall bound needs and the negatives its false-positive bound needs (the upper exact binomial
    bound), each converted through the estimated prevalence, the larger rounded up to the plan.
    """
    if decision.determination in _FINAL:
        return None, None, None
    if decision.determination == "evidence_conflict":
        return (
            "conflict",
            (
                f"Witness {decision.witness_source!r} exceeds a ceiling it should respect; inspect data leakage (audit "
                "items in its training data), ungrouped duplicates or sequences, and a witness/ceiling scope mismatch "
                "(what the witness reads against the ceiling's information scope)."
            ),
            None,
        )
    p = _pool(spec, evidence, population)
    crossing = _crossing(p, spec)
    factor, text = _factor(spec, p, crossing is not None)
    # Labeled items only is a property of the population, not of the evidence: a note, never the factor.
    conditioned = (
        " The evidence covers labeled items only; label the excluded items to extend the result to the full dataset."
        if "labeled items" in population
        else ""
    )
    if crossing is None:
        return factor, f"{text}{conditioned}".strip(), None
    item, n = crossing
    return factor, f"{text} {_crossing_note(spec, item, n)}{conditioned}".strip(), n


def _short_bounds(item: _Item, spec: TargetSpec) -> str:
    """Name the confidence bounds of a recall witness that fall short of the requirement; never an empty phrase."""
    short = (
        ("lower confidence bound on recall", item.value < spec.threshold),
        ("upper confidence bound on false positives", not _certified(item, spec)),
    )
    return " and ".join(name for name, is_short in short if is_short) or "confidence bounds"


def _crossing_note(spec: TargetSpec, item: _Item, n: int | None) -> str:
    """Say which bound is short of the target and at what size it would cross."""
    if spec.metric == "recall":
        short = _short_bounds(item, spec)
        note = (
            f"{item.name} meets recall {spec.threshold:.4g} at a false-positive rate within {_limit(spec):.4g} on its "
            "observed rates, not yet on its confidence bounds: "
        )
        more = (
            f"its {short} would cross the requirement at about {n} audit-split units, positives and negatives "
            "counted through the estimated prevalence, if the observed rates hold."
            if n is not None
            else f"no attainable sample size lets its {short} cross the requirement under the recorded method and "
            "sampling plan."
        )
    else:
        note = f"{item.name} is within its confidence radius of the threshold {spec.threshold:.4g}: "
        more = (
            f"the endpoint would cross it at about {n} of its independent units (audit-split units for local panels "
            "and witnesses, units with replicas for a replica ceiling) if the observed mean and variance hold."
            if n is not None
            else "no attainable sample size crosses it under the recorded method and sampling plan."
        )
    if n is not None and _plan(item) is not None:
        more += (
            " Later looks spend alpha/(k(k+1)), so the endpoint at that planned size is wider and may still fall short."
        )
    return note + more
