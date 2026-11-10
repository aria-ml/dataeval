"""Ordered decision procedure for feasibility requirements: the determination, its limitation and suggested minimum."""

__all__ = []

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple, cast

from dataeval.core._concentration import units_to_cross
from dataeval.performance._feasibility_evidence import Bound, Method, Quantity, Source
from dataeval.performance._feasibility_target import Scope, TargetSpec

Determination = Literal[
    "demonstrated_feasible",
    "infeasible_under_recorded_assumptions",
    "not_ruled_out",
    "unresolved",
]
Limitation = Literal[
    "conflict",
    "no_envelope",
    "scope_restricted",
    "missing_bound",
    "envelope_unavailable",
    "no_eligible_witness",
    "within_radius",
    "weak_witness",
]

_FINAL = ("demonstrated_feasible", "infeasible_under_recorded_assumptions")
# full_input ⊃ fixed_representation ⊃ partition_restricted: an envelope decides at its scope and every narrower one.
_RANK = {"partition_restricted": 0, "fixed_representation": 1, "full_input": 2}
_BUILTIN: tuple[Source, ...] = ("knn", "linear_probe")


@dataclass(frozen=True)
class Decision:
    """
    The determination for one requirement, what limits a non-final one, and the bounds that made it.

    ``envelope`` is the smallest available envelope at the judged scope or broader, ``witness`` the largest available
    eligible witness (for recall, certified at the false-positive limit) and ``witness_source`` its source; for a
    conflict, both are the conflicting pair. ``suggested_minimum`` is the independent units at which the bound named by
    ``minimum_source`` and ``minimum_quantity`` would cross the requirement if the observed rates held: a projection.
    ``deciding`` names the bounds the determination rests on, as ``(source, quantity)`` of its requirement.
    """

    determination: Determination
    limitation: Limitation | None
    envelope: float | None
    envelope_scope: Scope | None
    witness: float | None
    witness_source: Source | None
    suggested_minimum: int | None
    minimum_source: Source | None
    minimum_quantity: Quantity | None
    deciding: frozenset[tuple[Source, Quantity]]


class _Pool(NamedTuple):
    items: list[Bound]  # every bound on the requirement's metric, in table order
    envelopes: list[Bound]  # at the judged scope or broader
    narrower: list[Bound]  # envelopes narrower than the judged scope: evidence only, never decide
    witnesses: list[Bound]  # eligible for the judged scope and certified (recall: within the false-positive limit)
    false_positives: dict[Source, Bound]  # each witness's false-positive bound, for a recall requirement
    envelope: Bound | None  # smallest available eligible envelope
    witness: Bound | None  # largest available eligible, certified witness


def _eligible(b: Bound, scope: Scope) -> bool:
    """
    Return whether a witness is eligible at ``scope``.

    A built-in witness reads only the representation, so it is eligible at ``fixed_representation`` and broader. A
    user witness may read the whole input, so it is eligible at ``full_input`` only.
    """
    reads: Scope = "fixed_representation" if b.source in _BUILTIN else "full_input"
    return _RANK[reads] <= _RANK[scope]


def _limit(spec: TargetSpec) -> float:
    return cast(float, spec.max_false_positive_rate)  # a recall requirement always sets its limit


def _certified(b: Bound, spec: TargetSpec, fp: dict[Source, Bound]) -> bool:
    """Whether a witness counts for ``spec``: for recall, only when its own false-positive bound is within the limit."""
    if spec.metric != "recall":
        return True
    f = fp.get(b.source)
    return f is not None and f.bound is not None and f.bound <= _limit(spec)


def _tested(p: _Pool, scope: Scope) -> list[Bound]:
    """Every available witness eligible for ``scope``, certified or not."""
    return [b for b in p.items if b.role == "witness" and b.bound is not None and _eligible(b, scope)]


def _observed_meets(b: Bound, fp: Bound | None, spec: TargetSpec) -> bool | None:
    """
    Whether a witness's recorded point estimates already meet the requirement; None when they are not recorded.

    Accuracy and AP must exceed the threshold; recall must reach it with the false-positive rate within the limit.
    """
    if spec.metric != "recall":
        return None if b.observed is None else b.observed > spec.threshold
    if b.observed is None or fp is None or fp.observed is None:
        return None
    return b.observed >= spec.threshold and fp.observed <= _limit(spec)


def _pool(spec: TargetSpec, requirement: int, bounds: Sequence[Bound]) -> _Pool:
    mine = [b for b in bounds if b.requirement == requirement]
    items = [b for b in mine if b.quantity != "false_positive_rate"]
    fp: dict[Source, Bound] = {b.source: b for b in mine if b.quantity == "false_positive_rate"}
    rank = _RANK[spec.scope]
    envelopes = [b for b in items if b.role == "envelope" and _RANK[b.information_scope] >= rank]
    narrower = [b for b in items if b.role == "envelope" and _RANK[b.information_scope] < rank]
    witnesses = [b for b in items if b.role == "witness" and _eligible(b, spec.scope) and _certified(b, spec, fp)]
    return _Pool(
        items,
        envelopes,
        narrower,
        witnesses,
        fp,
        min((b for b in envelopes if b.bound is not None), key=lambda b: cast(float, b.bound), default=None),
        max((b for b in witnesses if b.bound is not None), key=lambda b: cast(float, b.bound), default=None),
    )


def decide(spec: TargetSpec, requirement: int, bounds: Sequence[Bound]) -> Decision:
    """
    Apply the ordered rules to the bounds on requirement ``requirement`` (``spec``) and name what limits the result.

    Envelopes count at the judged scope or broader (``full_input`` ⊃ ``fixed_representation`` ⊃
    ``partition_restricted``); narrower ones are evidence and never decide. A built-in witness is eligible at
    ``fixed_representation`` and ``full_input``, a user witness at ``full_input`` only. For a recall requirement, a
    witness counts only when certified: its false-positive upper bound is available and at or below
    ``max_false_positive_rate``. The false-positive bound itself never enters a comparison. The first matching rule
    decides:

    1. ``unresolved`` with limitation ``conflict`` — a (certified) witness eligible at some deciding-scope envelope's
       own scope has a lower bound above that envelope's upper bound, whatever the threshold: the conclusion is
       suspended;
    2. ``demonstrated_feasible`` — the largest witness lower bound is at or above the threshold;
    3. ``infeasible_under_recorded_assumptions`` — the smallest envelope upper bound is below the threshold;
    4. ``not_ruled_out`` — an informative envelope's upper bound is at or above the threshold, and no eligible witness
       reaches it;
    5. ``unresolved`` — otherwise.

    A non-final determination (and only a non-conflict one) gets a limitation, the first that matches:

    1. ``no_envelope`` — no envelope of any scope exists;
    2. ``scope_restricted`` — no envelope at the judged scope or broader, but an available narrower one. A replica
       envelope is ``full_input``, the broadest scope, so this also means no replica bound exists;
    3. ``missing_bound`` — an eligible envelope is unavailable because ``max_sequence_length`` was not declared;
    4. ``envelope_unavailable`` — envelopes exist (at any scope) but every one is unavailable for lack of units or
       panels;
    5. ``no_eligible_witness`` — no available witness is eligible for the judged scope, certified or not (at
       ``partition_restricted`` none can be in this release);
    6. ``within_radius`` — a crossing exists: the deciding envelope's recorded mean is below the threshold, or the best
       witness's above it, so the bound is on the wrong side by less than its radius. For a recall requirement, an
       eligible witness, certified or not, whose observed recall is at or above the threshold and observed
       false-positive rate at or below the limit, while a bound falls short;
    7. ``weak_witness`` — otherwise: the witness is weak (for recall: its observed recall is below the threshold or its
       observed false-positive rate above the limit).

    The suggested minimum is set whenever a crossing as in rule 6 exists, whatever the limitation: ``units_to_cross``
    at the bound's recorded mean, variance, support and delta (inverted on the observation's scale: ``1 - mean`` for an
    envelope, which observes a loss, clipped into the support), with gap ``|mean - threshold|``. It is None when not
    attainable: beyond ``units_to_cross``'s search, or recorded values it rejects (a ratio method, an invalid support).
    A recall requirement counts evaluation items: the positives its recall bound needs and the negatives its
    false-positive bound needs (the upper exact binomial bound), each converted through the estimated prevalence, the
    larger taken; ``minimum_quantity`` names which.
    """
    p = _pool(spec, requirement, bounds)
    verdict = _verdict(spec, p)
    limitation, minimum = verdict.limitation, None
    if verdict.determination not in _FINAL and limitation is None:
        crossing = _crossing(p, spec)
        limitation = _factor(spec, p, crossing is not None)
        if crossing is not None and crossing[1] is not None:
            minimum = (crossing[1], crossing[0].source, crossing[2])
    c, w = verdict.envelope, verdict.witness
    return Decision(
        determination=verdict.determination,
        limitation=limitation,
        envelope=None if c is None else c.bound,
        envelope_scope=None if c is None else c.information_scope,
        witness=None if w is None else w.bound,
        witness_source=None if w is None else w.source,
        suggested_minimum=None if minimum is None else minimum[0],
        minimum_source=None if minimum is None else minimum[1],
        minimum_quantity=None if minimum is None else minimum[2],
        deciding=_deciding(spec, verdict.deciding),
    )


class _Verdict(NamedTuple):
    determination: Determination
    limitation: Limitation | None  # only a conflict is known at this point
    deciding: tuple[Bound, ...]  # the bounds the determination rests on
    envelope: Bound | None  # what the determination quotes: for a conflict, the conflicting pair
    witness: Bound | None


def _conflicts(spec: TargetSpec, p: _Pool) -> list[tuple[Bound, Bound]]:
    """Return each (witness, envelope) pair where a certified witness, eligible at the envelope's scope, exceeds it."""
    return [
        (wi, ci)
        for ci in p.envelopes
        if ci.bound is not None
        for wi in p.items
        if wi.role == "witness"
        and wi.bound is not None
        and _eligible(wi, ci.information_scope)
        and _certified(wi, spec, p.false_positives)
        and wi.bound > ci.bound
    ]


def _verdict(spec: TargetSpec, p: _Pool) -> _Verdict:
    """Apply the five ordered rules of :func:`decide`; the first that matches decides."""
    t, c, w = spec.threshold, p.envelope, p.witness
    conflicts = _conflicts(spec, p)
    if conflicts:
        w, c = max(conflicts, key=lambda pair: cast(float, pair[0].bound) - cast(float, pair[1].bound))
        return _Verdict("unresolved", "conflict", (w, c), c, w)
    if w is not None and cast(float, w.bound) >= t:
        return _Verdict("demonstrated_feasible", None, (w,), c, w)
    if c is not None and cast(float, c.bound) < t:
        return _Verdict("infeasible_under_recorded_assumptions", None, (c,), c, w)
    informative = [b for b in p.envelopes if b.bound is not None and b.state == "informative"]
    if informative:
        c = min(informative, key=lambda b: cast(float, b.bound))
        return _Verdict("not_ruled_out", None, (c,), c, w)
    return _Verdict("unresolved", None, tuple(b for b in (c, w) if b is not None), c, w)


def _deciding(spec: TargetSpec, deciding: tuple[Bound, ...]) -> frozenset[tuple[Source, Quantity]]:
    """Name the deciding bounds as ``(source, quantity)``; a deciding recall witness brings its false-positive bound."""
    keys: set[tuple[Source, Quantity]] = {(b.source, b.quantity) for b in deciding}
    if spec.metric == "recall":  # the false-positive bound certifies the witness
        keys |= {(b.source, "false_positive_rate") for b in deciding if b.role == "witness"}
    return frozenset(keys)


def _to_cross(b: Bound, observed: float, gap: float, side: Literal["lower", "upper"]) -> int | None:
    try:
        observed = min(max(observed, 0.0), cast(float, b.support))
        return units_to_cross(
            observed,
            cast(float, b.variance),
            cast(float, b.support),
            cast(float, b.delta),
            gap,
            method=cast(Method, b.method),
            side=side,
        )
    except (ValueError, TypeError):  # values it cannot invert (ratio method, an invalid support, no gap): no estimate
        return None


def _units(b: Bound, gap: float) -> int | None:
    # The recorded mean is on the metric's scale; the support is the observation's. An envelope observes a loss.
    mean = cast(float, b.mean)
    return _to_cross(b, 1 - mean if b.bound_type == "upper" else mean, gap, "lower")


def _recall_crossing(p: _Pool, spec: TargetSpec) -> tuple[Bound, int | None, Quantity] | None:
    """
    Return the witness whose observed recall and false-positive rate already meet the requirement, with its images.

    Among scope-eligible witnesses (certified or not) with observed recall at or above the threshold and observed
    false-positive rate at or below the limit, the one with the largest recall lower bound. Each bound on the wrong
    side of its limit needs its own units (positives, or negatives through the upper bound); the estimated prevalence
    from the two counts converts both to images, the larger wins, and the quantity that set it is returned. None when
    a needed count is not attainable.
    """
    r0, f0 = spec.threshold, _limit(spec)
    found = []
    for b in _tested(p, spec.scope):
        fp = p.false_positives.get(b.source)
        if fp is not None and fp.bound is not None and _observed_meets(b, fp, spec):
            found.append((b, fp))
    if not found:
        return None
    b, fp = max(found, key=lambda f: cast(float, f[0].bound))
    pos_mean, neg_mean = cast(float, b.mean), cast(float, fp.mean)
    positives = _to_cross(b, pos_mean, pos_mean - r0, "lower") if cast(float, b.bound) < r0 else 0
    negatives = _to_cross(fp, neg_mean, f0 - neg_mean, "upper") if cast(float, fp.bound) > f0 else 0
    if positives is None or negatives is None:
        return b, None, "recall"
    n_pos, n_neg = cast(int, b.n), cast(int, fp.n)  # both bounds count audit items
    share = n_pos / (n_pos + n_neg)  # the estimated prevalence of the class
    images_for_recall, images_for_fp = math.ceil(positives / share), math.ceil(negatives / (1 - share))
    quantity: Quantity = "recall" if images_for_recall >= images_for_fp else "false_positive_rate"
    return b, max(images_for_recall, images_for_fp), quantity


def _crossing(p: _Pool, spec: TargetSpec) -> tuple[Bound, int | None, Quantity] | None:
    """Return the deciding envelope or best witness whose recorded mean is past the threshold, with its units."""
    if spec.metric == "recall":  # a recall envelope has no single recorded mean to project; the witness side decides
        return _recall_crossing(p, spec)
    threshold = spec.threshold
    for b, sign in ((p.envelope, -1), (p.witness, 1)):
        if b is not None and b.mean is not None and sign * (b.mean - threshold) > 0:
            return b, _units(b, abs(b.mean - threshold)), b.quantity
    return None


def _factor(spec: TargetSpec, p: _Pool, crossing: bool) -> Limitation:  # noqa: C901
    """Name what limits a non-final determination: the first rule of :func:`decide`'s limitation list that matches."""
    every = [*p.envelopes, *p.narrower]
    if not every:
        return "no_envelope"
    unavailable_bound = "max_sequence_length_missing"
    missing = next((b for b in p.envelopes if b.bound is None and b.reason == unavailable_bound), None)
    if not p.envelopes and any(b.bound is not None for b in p.narrower):
        return "scope_restricted"
    if missing is not None:
        return "missing_bound"
    if all(b.bound is None and b.reason != unavailable_bound for b in every):
        return "envelope_unavailable"
    if not _tested(p, spec.scope):
        return "no_eligible_witness"
    return "within_radius" if crossing else "weak_witness"
