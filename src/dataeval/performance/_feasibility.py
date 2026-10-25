__all__ = []

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Annotated, Any, Literal, NamedTuple, get_args

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import Field, PositiveInt

from dataeval._experimental import experimental, quiet_experimental
from dataeval._helpers import reject_filtered_metadata
from dataeval.config import get_seed
from dataeval.exceptions import ShapeMismatchError
from dataeval.performance._feasibility_classification import (
    _class_index,
    check_predictions,
    count_claims,
    gather_evidence,
)
from dataeval.performance._feasibility_constants import ALLOCATION, FIT_FRACTION
from dataeval.performance._feasibility_decide import (
    _FINAL,
    _RANK,
    Decision,
    Determination,
    LimitingFactor,
    _crossing,
    _Item,
    _limit,
    _observed_meets,
    _Pool,
    _pool,
    _short_bounds,
    _tested,
    decide,
)
from dataeval.performance._feasibility_evidence import Evidence, SamplingPlan, allocate, check_ledger, stats_key
from dataeval.performance._feasibility_target import Metric, Scope, TargetSpec, Task, parse_targets
from dataeval.performance._feasibility_units import Units, audit_order, resolve_replicas, resolve_units, split_units
from dataeval.protocols import AnnotatedDataset, ArrayLike, FeatureExtractor
from dataeval.types import DictOutput, Evaluator, EvaluatorConfig, set_metadata
from dataeval.types._config import OpenUnitInterval
from dataeval.types._task import detect_task


def _enum(literal: Any) -> pl.Enum:
    return pl.Enum(list(get_args(literal)))


# The stable API: names and dtypes are frozen in tests. Enums span every task planned, not only phase 1's.
_SCHEMA = {
    "task": _enum(Task),
    "metric": _enum(Metric),
    "threshold": pl.Float64,
    "judged_scope": _enum(Scope),
    "target_spec": pl.String,
    "determination": _enum(Determination),
    "ceiling": pl.Float64,
    "ceiling_scope": _enum(Scope),
    "witness": pl.Float64,
    "witness_source": pl.String,
    "reason": pl.String,
    "limiting_factor": _enum(LimitingFactor),
    "recommendation": pl.String,
    "suggested_minimum": pl.Int64,
    "evidence_ids": pl.List(pl.String),
}
_BOUNDS_SCHEMA = {
    "target_spec": pl.String,
    "metric": _enum(Metric),
    "quantity": pl.String,
    "threshold": pl.Float64,
    "source": pl.String,
    "evidence_id": pl.String,
    "information_scope": _enum(Scope),
    "unit": pl.String,
    "bound_type": pl.Enum(["lower", "upper"]),
    "observed": pl.Float64,
    "bound": pl.Float64,
    "n": pl.Int64,
    "state": pl.String,
}
# The reader's view of _BOUNDS_SCHEMA, most important first; the keys come last.
_EVIDENCE_COLUMNS = (
    "source", "metric", "quantity", "threshold", "bound_type", "bound", "observed", "information_scope", "state", "n",
    "evidence_id", "target_spec",
)  # fmt: skip
_SOURCES = {
    "replica_ceiling/v1": "repeated labels",
    "builtin_witness/v1:knn": "built-in k-NN",
    "builtin_witness/v1:probe": "built-in linear probe",
    "user_witness/v1": "your model",
}
_STATE = ["alpha", "panel_size", "n_cells", "builtin_witnesses", "max_sequence_length", "seed"]
_PROVENANCE = ("reads", "training_items", "frozen_before_audit", "selection", "preprocessing", "decision_policy")
_READS = ("full_input", "representation")  # "partition" needs the internal partition, not exposed in this release


def _source_name(evidence_id: str, kind: str) -> str:
    """Readable name of an evidence source; any unknown kind keeps its evidence id."""
    if kind.startswith("local_panel_ceiling/v1"):
        return "group-based ceiling"
    return _SOURCES.get(evidence_id, evidence_id)


def _unit_noun(unit: str) -> str:
    """Singular noun for an evidence unit: a group is a sequence, anything else an image (image classification)."""
    return "sequence" if unit == "group" else "image"


_SCOPE_WORDS = {
    "full_input": "models that may use the whole {noun}",
    "fixed_representation": "models that may use only the embeddings",
    "partition_restricted": "models that give one answer per group",
}
_MARKS = {
    "demonstrated_feasible": "demonstrated ✓",
    "infeasible_under_recorded_assumptions": "ruled out ✗",
    "not_ruled_out": "not ruled out ?",
    "unresolved": "insufficient evidence ?",
    "evidence_conflict": "evidence conflict !",
}
_HEADER = ("Requirement", "Determination", "Lower confidence bound", "Upper confidence bound")
_RULED_OUT = "* ruled out under the recorded assumptions"


def _pct(value: float, direction: Literal["lower", "upper"] | None = None) -> str:
    """Format to 0.1%; a lower bound rounds down and an upper bound up, so rounding never overstates a bound."""
    if direction is None:
        return f"{value:.1%}"
    tenths = value * 1000  # the 1e-9 absorbs float error only, such as 0.957 * 1000 = 956.9999999999999
    tenths = math.floor(tenths + 1e-9) if direction == "lower" else math.ceil(tenths - 1e-9)
    return f"{tenths / 10:.1f}%"


_AGGREGATION_WORDS = {None: "", "sequence_mean": "per-sequence mean ", "frame_pooled": "frame-pooled "}


def _fp_limit(spec: TargetSpec) -> str:
    return f"{_limit(spec) * 100:g}%"


def _metric_words(spec: TargetSpec) -> str:
    if spec.metric == "recall":  # recall takes no aggregation
        return f"recall for {spec.class_name} with at most {_fp_limit(spec)} false positives"
    if spec.metric != "ap":
        metric = spec.metric
    else:
        metric = "macro AP" if spec.class_name is None else f"AP for class {spec.class_name}"
    return _AGGREGATION_WORDS[spec.aggregation] + metric


def _requirement(spec: TargetSpec, scope: str | None) -> str:
    """Return the requirement cell, with the judged scope when the requirements differ in it."""
    if spec.metric == "recall":
        text = f"{spec.class_name}: ≥{spec.threshold * 100:g}% recall, ≤{_fp_limit(spec)} false positives"
    else:
        text = f"{spec.threshold * 100:g}% {_metric_words(spec)}"
    return text if scope is None else f"{text} ({scope})"


_ACTIONS = {
    "no_replicas": "have several annotators independently label the same {noun}s",
    "no_witness": "test a stronger model",
    "missing_bound": "supply the input named in the recommendation column",
}
# An unresolved row lacks evidence on both sides, so it names both ways forward; allowances stay in the recommendation.
_BOTH_WAYS = "to show it, {show}; to rule it out, collect repeated labels for a whole-{noun} ceiling"
# A recall row's witness is a frozen decision rule, so a different rule can also supply one.
_RECALL_NO_WITNESS = "test a stronger model or a different frozen operating point"
# Led with when every tested model's observed scores fall short: no audit size carries its bound across.
_NO_GAIN = "more evaluation {noun}s would not help at the observed scores; "
_WIDTH = 120  # a Next or Notes line longer than this breaks after its semicolons


def _from_audit(item: _Item) -> bool:
    return item.ev.sample.startswith("audit split")


def _more_units(item: _Item | None, pool: _Pool | None) -> str:
    """Name the units to collect for a sampling-width factor: those the crossing bound counts."""
    ceilings = [] if pool is None else [*pool.ceilings, *pool.narrower]
    if ceilings and all(i.ep.value is None for i in ceilings):
        return "collect more labeled {noun}s"  # no ceiling has enough units, so the fit split is short as well
    if item is not None and not _from_audit(item):
        return "collect more {noun}s with repeated labels"
    return "collect more evaluation {noun}s"


def _crossing_bound(item: _Item | None, noun: str, spec: TargetSpec) -> tuple[str, str]:
    """Name the bound that would cross the requirement and the units it counts."""
    if item is None:
        return "the bound", f"{noun}s"
    source, unit = _source_name(item.eid, item.ev.kind), _unit_noun(item.ev.unit)
    if spec.metric == "recall":  # a witness's: its recall bound, its false-positive bound or both
        bound = f"{source}'s {_short_bounds(item, spec)}"
    elif item.ep.direction == "lower":
        bound = f"{source}'s lower confidence bound"
    else:
        bound = f"the upper confidence bound from the {source}"
    return bound, f"evaluation {unit}s" if _from_audit(item) else f"{unit}s with repeated labels"


def _action(row: dict[str, Any], spec: TargetSpec, item: _Item | None, pool: _Pool | None) -> str | None:
    """Name the plain action for the row's limiting factor, if it has one, led by why more images would not help."""
    factor = row["limiting_factor"]
    show = _RECALL_NO_WITNESS if spec.metric == "recall" else _ACTIONS["no_witness"]
    if row["determination"] == "unresolved" and factor in ("scope_restricted", "no_replicas"):
        action = _BOTH_WAYS.replace("{show}", show)
    elif factor == "no_witness":
        action = show
    elif factor == "sampling_width":
        return _more_units(item, pool)
    else:
        return _ACTIONS.get(factor)
    tested = [] if pool is None else _tested(pool, spec.scope)
    short = bool(tested) and all(_observed_meets(i, spec) is False for i in tested)
    return (_NO_GAIN if short else "") + action


def _next(row: dict[str, Any], spec: TargetSpec, noun: str, pool: _Pool | None) -> str:
    """Name the plain action for the row's limiting factor and, when suggested, the size at which a bound crosses."""
    n = row["suggested_minimum"]
    found = None if pool is None else _crossing(pool, spec)  # names the units even with no size
    item = None if found is None else found[0]
    action = _action(row, spec, item, pool)
    if action is None:
        return ""
    action = action.format(noun=noun)
    if n is None:
        return action
    bound, units = _crossing_bound(item, noun, spec)
    if action == f"collect more {units}":  # the size says what to collect
        return f"about {n:,} {units} in all, for {bound} to cross if the observed rates hold"
    crossing = f"{bound} would cross at about {n:,} {units} if the observed rates hold"
    # what more units would do leads the ways to show or rule out, as "would not help" does
    return f"{action}; {crossing}" if row["limiting_factor"] == "sampling_width" else f"{crossing}; {action}"


def _wrap(line: str, indent: int) -> str:
    """Break a line longer than the printout width after each semicolon, continuing ``indent`` spaces in."""
    return line if len(line) <= _WIDTH else line.replace("; ", ";\n" + " " * indent)


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def _on_units(b_row: dict[str, Any], kind: str) -> str:
    return "" if b_row["n"] is None else f" on {b_row['n']:,} {kind} {b_row['unit']}s"


def _lower(row: dict[str, Any], w_row: dict[str, Any] | None, fp_row: dict[str, Any] | None, spec: TargetSpec) -> str:
    """
    Return the deciding witness's bound with its source; for recall, beside its false-positive bound.

    With no recall witness counted, the tested one's bounds are shown, marked as not counted.
    """
    if row["witness"] is not None:
        source, witness = w_row["source"] if w_row else row["witness_source"], _pct(row["witness"], "lower")
    elif spec.metric == "recall" and w_row is not None:
        source, witness = f"{w_row['source']}, not counted", _pct(w_row["bound"], "lower")
    else:
        return "—"
    if spec.metric != "recall":
        return f"{witness} ({source})"
    fp = "" if fp_row is None else f", FP ≤{_pct(fp_row['bound'], 'upper')}"
    return f"recall {witness}{fp} ({source})"


def _upper(row: dict[str, Any], c_row: dict[str, Any] | None, spec: TargetSpec) -> str:
    """Return the deciding ceiling's bound with its source; a ruled-out row's carries the ``*`` a note explains."""
    if row["ceiling"] is None:
        return "—"
    on = "recall " if spec.metric == "recall" else ""
    star = "*" if row["determination"] == "infeasible_under_recorded_assumptions" else ""
    return f"{on}{_pct(row['ceiling'], 'upper')}{star}" + (f" ({c_row['source']})" if c_row else "")


def _scored(w_row: dict[str, Any] | None, fp_row: dict[str, Any] | None, spec: TargetSpec) -> tuple[str, str] | None:
    """Return a witness's source and what it observed, such as ``96.9% accuracy on 5,000 evaluation images``."""
    if w_row is None or w_row["observed"] is None:
        return None
    if spec.metric != "recall":
        return w_row["source"], f"{_pct(w_row['observed'])} {_metric_words(spec)}{_on_units(w_row, 'evaluation')}"
    text = f"{_pct(w_row['observed'])} recall{_on_units(w_row, str(spec.class_name))}"
    if fp_row is not None and fp_row["observed"] is not None:
        text += f" and {_pct(fp_row['observed'])} false positives{_on_units(fp_row, 'other')}"
    return w_row["source"], text


def _uncertified(label: str, fp_row: dict[str, Any] | None, spec: TargetSpec) -> str:
    """Say why no tested model counts for a recall requirement; the observed rates are in the scored note."""
    if fp_row is None:  # no negatives in the audit
        return f"{label}: no tested model has an available upper confidence bound on false positives."
    return (
        f"{label}: no tested model counts, since none keeps false positives within {_fp_limit(spec)}; "
        f"{fp_row['source']}'s upper confidence bound on false positives is {_pct(fp_row['bound'], 'upper')}."
    )


def _fp_row(
    row: dict[str, Any], mine: list[dict[str, Any]], spec: TargetSpec, pool: _Pool | None
) -> dict[str, Any] | None:
    """
    Return the deciding witness's false-positive bound or, with none certified, a tested witness's.

    The tested witness is the one a crossing names, so Next and Notes name one source; else the one nearest the limit.
    """
    fp_rows = {b["evidence_id"]: b for b in mine if b["quantity"] == "false_positive_rate" and b["bound"] is not None}
    if row["witness_source"] is not None or pool is None:
        return fp_rows.get(row["witness_source"])
    found = _crossing(pool, spec)
    if found is not None and found[0].eid in fp_rows:
        return fp_rows[found[0].eid]
    tested = [fp_rows[i.eid] for i in _tested(pool, spec.scope) if i.eid in fp_rows]
    return min(tested, key=lambda b: b["bound"], default=None)


class _Entry(NamedTuple):
    cells: tuple[str, ...]  # the table row
    next: str  # the plain next step, or empty
    scored: tuple[str, str] | None  # (source, what it observed) for the observed-scores note
    notes: list[tuple[int, str]]  # (section order, text) of the row's other notes


def _entry(
    row: dict[str, Any],
    spec: TargetSpec,
    requirement: str,
    label: str,
    bounds: list[dict[str, Any]],
    evidence: Mapping[str, Evidence],
    noun: str,
) -> _Entry:
    """Lay out one determinations row: its table cells, next step and notes."""
    mine = [b for b in bounds if b["target_spec"] == row["target_spec"]]
    metric = [b for b in mine if b["quantity"] != "false_positive_rate"]  # what the decision rules compared
    ceiling = row["ceiling"]
    w_row = next((b for b in metric if b["evidence_id"] == row["witness_source"] and b["bound_type"] == "lower"), None)
    c_row = next(
        (
            b
            for b in metric
            if b["bound_type"] == "upper"
            and b["bound"] is not None
            and ceiling is not None
            and b["information_scope"] == row["ceiling_scope"]
            and math.isclose(b["bound"], ceiling)
        ),
        None,
    )
    # The pool the decision rules saw, to name the bound behind a suggested minimum; None without the row's evidence.
    ids = [e for e in row["evidence_ids"] or [] if e in evidence]
    pool = _pool(spec, evidence, evidence[ids[0]].population) if ids else None
    fp_row = _fp_row(row, mine, spec, pool)
    if w_row is None and fp_row is not None:  # recall, none counted: the tested witness whose FP bound shows
        tested = (b for b in metric if b["evidence_id"] == fp_row["evidence_id"] and b["bound_type"] == "lower")
        w_row = next(tested, None)
    determination = row["determination"]
    cells = (requirement, _MARKS[determination], _lower(row, w_row, fp_row, spec), _upper(row, c_row, spec))
    narrower = [  # informative ceilings narrower than the judged scope: evidence, never deciding
        b
        for b in metric
        if b["bound_type"] == "upper"
        and b["state"] == "informative"
        and _RANK[b["information_scope"]] < _RANK[spec.scope]
    ]
    notes = [(2, _narrower(b, spec, noun)) for b in narrower]
    tested = pool is not None and bool(_tested(pool, spec.scope))
    if spec.metric == "recall" and row["witness"] is None and determination not in _FINAL and tested:
        notes.append((1, _uncertified(label, fp_row, spec)))
    if determination == "evidence_conflict":
        check = f"check the assumptions, including whether evaluation {noun}s entered training"
        notes.append((3, f"{label}: the evidence conflicts; {check}."))
    return _Entry(cells, _next(row, spec, noun, pool), _scored(w_row, fp_row, spec), notes)


def _narrower(b_row: dict[str, Any], spec: TargetSpec, noun: str) -> str:
    """Say what a ceiling narrower than the judged scope bounds."""
    scope = _SCOPE_WORDS[b_row["information_scope"]].format(noun=noun)
    return f"{_capital(b_row['source'])} ({_pct(b_row['bound'], 'upper')} {_metric_words(spec)}) bounds only {scope}."


def _notes(rows: list[dict[str, Any]], entries: list[_Entry]) -> list[str]:
    """Return the observed scores (one line per source), the rows' notes in section order and the ``*`` note."""
    scored: dict[str, dict[str, None]] = {}
    for e in entries:
        if e.scored is not None:
            scored.setdefault(e.scored[0], {})[e.scored[1]] = None
    notes = [f"{_capital(source)} scored {'; '.join(seen)}." for source, seen in scored.items()]
    notes += [text for _, text in sorted((n for e in entries for n in e.notes), key=lambda n: n[0])]
    if any(r["determination"] == "infeasible_under_recorded_assumptions" for r in rows):
        notes.append(_RULED_OUT)
    return list(dict.fromkeys(notes))


def _printout(rows: list[dict[str, Any]], bounds: list[dict[str, Any]], evidence: Mapping[str, Evidence]) -> str:
    """Lay out the determinations as a header, one table row per requirement, next steps and notes."""
    noun = next((b["unit"] for b in bounds), "image")  # one call, one unit
    several = len({r["judged_scope"] for r in rows}) > 1
    specs = [TargetSpec.from_canonical(r["target_spec"]) for r in rows]
    requirements = [
        _requirement(s, _SCOPE_WORDS[r["judged_scope"]].format(noun=noun) if several else None)
        for s, r in zip(specs, rows, strict=True)
    ]
    labels = [str(s.class_name) if s.metric == "recall" else q for s, q in zip(specs, requirements, strict=True)]
    labels = labels if len(set(labels)) == len(labels) else requirements  # a recall row by its class, if unambiguous
    entries = [_entry(*args, bounds, evidence, noun) for args in zip(rows, specs, requirements, labels, strict=True)]
    judged = "judged at several scopes" if several else f"judged for {_SCOPE_WORDS[rows[0]['judged_scope']]}"
    alpha = next((ev.details["call_alpha"] for ev in evidence.values() if "call_alpha" in ev.details), None)
    # rounded down, so the stated confidence never overstates; a whole percentage drops its ".0"
    confidence = None if alpha is None else _pct(1 - alpha, "lower").replace(".0%", "%")
    held = "Bounds are" if confidence is None else f"All bounds hold together at {confidence} confidence,"
    n = len(rows)
    lines = [
        f"Feasibility of {n} requirement{'' if n == 1 else 's'}, {judged.format(noun=noun)}.",
        f"{held} for new {noun}s from the same population.",
        f"They assume no evaluation {noun} was used to train or choose a model, or to set a requirement.",
        "",
    ]
    table = [_HEADER, *(e.cells for e in entries)]
    widths = [max(len(cells[i]) for cells in table) for i in range(len(_HEADER))]
    lines += ["   ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)).rstrip() for cells in table]
    steps = [(label, e.next) for label, e in zip(labels, entries, strict=True) if e.next]
    if steps:
        width = max(len(label) for label, _ in steps)
        lines += ["", "Next", *(_wrap(f"  {label.ljust(width)}  {step}", width + 4) for label, step in steps)]
    notes = _notes(rows, entries)
    if notes:
        lines += ["", "Notes", *(_wrap(f"  {note}", 4) for note in notes)]
    return "\n".join(lines)


@experimental
@dataclass(frozen=True, repr=False)
class FeasibilityOutput(DictOutput):
    """
    Output class for the :class:`.Feasibility` evaluator.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    Attributes
    ----------
    determinations : pl.DataFrame
        One row per target, in the order given, with columns:

        - task: Enum - ``IC`` (``OD`` and ``MOT`` are planned)
        - metric: Enum - ``accuracy``, ``ap`` or ``recall`` (``recall_at_fp`` and ``mota`` are planned)
        - threshold: float - The value the target asks for (for recall, the minimum recall)
        - judged_scope: Enum - ``full_input``, ``fixed_representation`` or ``partition_restricted``
        - target_spec: str - The canonical, versioned JSON of the full target; equal targets serialize
          identically and ``TargetSpec.from_canonical`` rebuilds it
        - determination: Enum - ``evidence_conflict``, ``demonstrated_feasible``,
          ``infeasible_under_recorded_assumptions``, ``not_ruled_out`` or ``unresolved``
        - ceiling: float - The deciding ceiling's upper endpoint, null when there is none (for recall, on the recall
          any model in its scope reaches at the false-positive limit)
        - ceiling_scope: Enum - That ceiling's information scope
        - witness: float - The deciding witness's lower endpoint, null when there is none (for recall, its recall
          endpoint, from a witness whose false-positive upper endpoint is within the limit)
        - witness_source: str - That witness's evidence id
        - reason: str - Why the first matching rule decided
        - limiting_factor: Enum - For a non-final determination, what limits it: ``sampling_width``,
          ``no_replicas``, ``scope_restricted``, ``no_witness``, ``missing_bound`` or ``conflict``; null for a
          final one. ``labels_conditioned`` is in the enum but not produced: a population of labeled items only
          adds a note to the recommendation instead
        - recommendation: str - The evidence that addresses the limiting factor
        - suggested_minimum: int - Independent units at which the endpoint would cross the threshold if the
          observed mean and variance hold, rounded up to the sampling plan for audit-split endpoints. For recall, a
          witness's: the positives its recall bound needs and the negatives its false-positive bound needs, each
          converted to evaluation images through the estimated prevalence, the larger taken. Null when not
          attainable; a recall ceiling never sets one, having no single mean to project
        - evidence_ids: list[str] - Keys of ``evidence`` behind the row, deciding items first
    evidence : dict[str, Evidence]
        Every evidence item of the call by id: one source run on one sample, with its basis, population,
        information scope, one endpoint per target it bounds (two for a witness on a recall target: recall and
        the false-positive rate; each spent from the call's alpha), and free-form
        ``details`` (raw core results, assumptions, allowances, ``seed``, ``call_alpha``, ``allocation`` and
        ``order_seed``).
    """

    determinations: pl.DataFrame
    evidence: dict[str, Evidence]

    @property
    def plot_type(self) -> Literal["feasibility"]:
        return "feasibility"

    def __repr__(self) -> str:
        """Summarize the evidence by id rather than printing every item's raw details."""
        items = f"<{len(self.evidence)} evidence items: {', '.join(self.evidence)}>"
        return f"{type(self).__name__}(determinations=DataFrame(shape={self.determinations.shape}), evidence={items})"

    def __str__(self) -> str:
        """
        Print the scope, joint confidence and its assumption, a table row per requirement, then next steps and notes.

        Each row gives the determination with its mark (✓ demonstrated, ✗ ruled out, ? insufficient evidence or not
        ruled out, ! evidence conflict), the deciding lower confidence bound and upper confidence bound with their
        sources; a ruled-out row's upper bound carries ``*``. A recall row with no counted witness shows a tested
        one's bounds, marked as not counted. Next names the plain action for each non-final row, separating the ways
        to show and to rule out a requirement, and says when more evaluation images would not help at the observed
        scores; Notes give the observed scores, ceilings narrower than the judged scope and conflicts.
        """
        rows = list(self.determinations.iter_rows(named=True))
        return _printout(rows, list(self._bounds().iter_rows(named=True)), self.evidence)

    def aggregate_by_evidence(self) -> pl.DataFrame:
        """
        Return every bound of the call as one flat table, one row per evidence item, target and bounded quantity.

        Rows follow the order of ``determinations``, then of ``evidence``. Each item's ``basis`` and each
        bound's share of ``alpha`` are on ``evidence``.

        Returns
        -------
        pl.DataFrame
            With columns, most important first:

            - source: str - Readable source: ``built-in k-NN``, ``built-in linear probe``, ``your model``,
              ``group-based ceiling``, ``repeated labels``, or the evidence id for any other kind
            - metric: Enum - The target's metric
            - quantity: str - What the bound is on: the metric, or ``false_positive_rate`` for a witness's second
              bound on a recall target, so a recall target has a recall row and a false-positive row per witness
            - threshold: float - The value the target asks for
            - bound_type: Enum - ``lower`` for a witness's metric, ``upper`` for a ceiling and for a false-positive
              rate
            - bound: float - The confidence bound; null when unavailable
            - observed: float - A witness's observed mean of the quantity; null for a ceiling and where the source
              records no mean (such as AP)
            - information_scope: Enum - What the source can see
            - state: str - The endpoint's state
            - n: int - Independent units behind the bound; null when unavailable and where the source records no
              statistics (such as AP and ``frame_pooled`` bounds)
            - evidence_id: str - Key of ``evidence``
            - target_spec: str - The canonical target the bound is on, the key to ``determinations``; distinguishes
              targets that share a metric and threshold but differ in scope, class or aggregation
        """
        return self._bounds().select(_EVIDENCE_COLUMNS)

    def _bounds(self) -> pl.DataFrame:
        """Every bound with the columns the printout reads, in ``_BOUNDS_SCHEMA``'s order."""
        by_spec = {r["target_spec"]: r for r in self.determinations.iter_rows(named=True)}
        rows = []
        for spec, target in by_spec.items():
            for eid, ev in self.evidence.items():
                for e in ev.endpoints:
                    if e.target_spec != spec:
                        continue
                    stats = ev.details["endpoints"][stats_key(spec, e.quantity)] or {}
                    rows.append({
                        "target_spec": spec,
                        "metric": target["metric"],
                        "quantity": target["metric"] if e.quantity == "metric" else e.quantity,
                        "threshold": target["threshold"],
                        "source": _source_name(eid, ev.kind),
                        "evidence_id": eid,
                        "information_scope": ev.information_scope,
                        "unit": _unit_noun(ev.unit),
                        "bound_type": e.direction,
                        "observed": stats.get("mean") if ev.supports == "witness" else None,
                        "bound": e.value,
                        "n": stats.get("n"),
                        "state": e.state,
                    })
        return pl.DataFrame(rows, schema=_BOUNDS_SCHEMA)


def _allowance(allowances: Mapping[Any, Any], name: str) -> NDArray[np.float64] | None:
    if allowances.get(name) is None:
        return None
    arr = np.asarray(allowances[name], dtype=np.float64)
    if not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError(f"allowances[{name!r}] must be finite and non-negative, got {allowances[name]!r}.")
    return arr


def _check_allowances(allowances: Mapping[Any, Any]) -> dict[str, float]:
    """β and Γ: scalars, Γ only on top of β. A per-cell β would name the internal partition's cells, not exposed."""
    unknown = sorted(set(allowances) - {"beta", "gamma"})
    if unknown:
        raise ValueError(f"allowances accepts 'beta' and 'gamma' only, got {unknown}.")
    beta, gamma = _allowance(allowances, "beta"), _allowance(allowances, "gamma")
    if beta is not None and beta.ndim:
        raise ValueError(
            "allowances['beta'] must be a scalar in this release; a per-cell beta refers to the cells of the internal "
            f"partition, which are not exposed. Got {beta.size} values."
        )
    if gamma is not None and (gamma.ndim or beta is None):
        raise ValueError(
            "allowances['gamma'] must be a scalar, given together with beta. beta must be a justified bound: "
            "beta=0 asserts that no model reading the representation beats one fixed answer per cell."
        )
    return {k: float(v) for k, v in (("beta", beta), ("gamma", gamma)) if v is not None}


def _check_provenance(provenance: Mapping[str, Any] | None, predictions: ArrayLike | None) -> None:
    """Known keys only: a misspelled ``training_items`` would silently skip the leak check."""
    if not provenance:
        return
    if predictions is None:
        raise ValueError("provenance describes the user witness; pass its predictions= too.")
    unknown = sorted(set(provenance) - set(_PROVENANCE))
    if unknown:
        raise ValueError(f"provenance accepts the keys {list(_PROVENANCE)} only, got {unknown}.")
    if provenance.get("reads", "full_input") not in _READS:
        raise ValueError(
            f"provenance['reads'] must be one of {list(_READS)}, got {provenance['reads']!r}; a witness giving one "
            "answer per cell ('partition') needs the internal partition, which this release does not expose."
        )


def _check_class_names(specs: Sequence[TargetSpec], units: Units) -> None:
    """Map every class name to its index now, so an unmappable one raises before the embedding pass."""
    for spec in specs:
        if spec.class_name is not None:
            _class_index(units, spec.class_name)


def _check_predictions(predictions: ArrayLike | None, units: Units, provenance: Mapping[str, Any] | None) -> None:
    """Check the user witness's predictions now, so bad ones raise before the embedding pass."""
    if predictions is None:
        return
    _, decisions = check_predictions(predictions, len(units.item_ids), units.n_classes)
    if not decisions and "decision_policy" in (provenance or {}):
        raise ValueError(
            "provenance['decision_policy'] records the rule that produced the evaluated decisions; pass those class "
            "decisions as a 1-D predictions= array, not scores."
        )


def _audit_prefix(units: Units, audit: NDArray[np.intp], sampling: SamplingPlan | None, look: int) -> NDArray[np.intp]:
    """Return the plan's prefix of the audit in the plan's order, whole units only; without a plan, the audit."""
    if sampling is None:
        if look != 1:
            raise ValueError(f"look={look} needs the sampling plan declared before the first look; pass sampling=.")
        return audit
    ordered = audit_order(units, audit, sampling.order_seed)
    g = units.groups[ordered]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])  # each unit's first item: audit_order keeps units contiguous
    if sampling.sizes[-1] > len(starts):  # at every look: the partition's cell count derives from the final size
        raise ValueError(
            f"The sampling plan's final size {sampling.sizes[-1]} exceeds the audit's {len(starts)} units; "
            "plan only sizes the audit can reach."
        )
    return ordered[: np.r_[starts, len(g)][sampling.prefix(len(starts), look)]]


def _row(spec: TargetSpec, decision: Decision) -> dict[str, Any]:
    target = {
        "task": spec.task,
        "metric": spec.metric,
        "threshold": spec.threshold,
        "judged_scope": spec.scope,
        "target_spec": spec.canonical(),
    }
    return target | asdict(decision) | {"evidence_ids": list(decision.evidence_ids)}


@experimental
class Feasibility(Evaluator):
    """
    Decide whether a classification target is reachable on a dataset, with a proof behind every claim.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    For each target (a metric, its details and a threshold), Feasibility gathers **ceilings** (upper endpoints on
    what any model in a class can reach on the population) and **witnesses** (lower endpoints on what a frozen
    model does reach), all simultaneous at level ``alpha`` over one call, and applies five ordered rules:
    ``evidence_conflict``, ``demonstrated_feasible``, ``infeasible_under_recorded_assumptions``,
    ``not_ruled_out``, ``unresolved``. A non-final determination names its limiting factor and the evidence that
    addresses it. A recall target (a ``TargetSpec`` with ``metric="recall"``, a class and a false-positive limit)
    counts a witness only when its false-positive upper endpoint is within the limit, and its ceilings bound the
    recall any model in their scope reaches at that limit; the printout and
    :meth:`FeasibilityOutput.aggregate_by_evidence` show both of a witness's bounds.

    Sources: local-panel ceilings, DataEval's post-stratified pooling (C3) of the DFE's disjoint label panels [4]_
    within the cells of a label-blind partition fitted on a group-disjoint fit split, proved in
    :func:`~dataeval.core.local_panel_bound`'s Notes with two Hoeffding [1]_ steps; replica ceilings from
    ``replicas=``; built-in frozen k-NN and linear-probe witnesses fitted on the fit split; and a user witness from
    ``predictions=``. Each (target, source) pair gets an equal share of ``alpha``. Endpoints on means use Hoeffding
    [1]_ and empirical Bernstein [2]_ radii, except that ungrouped witness endpoints (accuracy, recall,
    false-positive rate and AP indicators) use exact Clopper-Pearson [5]_ bounds; AP and recall ceilings use
    envelopes built on the cost-weighted risk inequality of [3]_.

    Parameters
    ----------
    extractor : FeatureExtractor or None, default None
        Feature extractor that computes the representation; optional when ``embeddings`` are given to
        :meth:`evaluate`. It must have been fitted and selected independently of the audit items' labels.
    batch_size : int or None, default None
        Batch size for embedding computation.
    alpha : float, default 0.05
        Error budget of one call, shared by every endpoint of every target.
    panel_size : int or None, default None
        Labels per local panel, at least 2; None uses the provisional default (5).
    n_cells : int or None, default None
        Partition cells for local panels; None follows the planned number of audit units (10 below 20,000).
    builtin_witnesses : bool, default True
        Fit and score the built-in k-NN and linear-probe witnesses.
    max_sequence_length : int or None, default None
        Declared largest group size for grouped (video) data, needed by ``frame_pooled`` targets. Data that
        violate it raise rather than being truncated.
    seed : int or None, default None
        Seed of the splits, partition and panels; None uses :func:`dataeval.config.get_seed`, else 0. Keep it
        fixed across the looks of one sampling plan.
    config : Feasibility.Config or None, default None
        Optional configuration object; parameters passed directly to ``__init__`` override it.

    Notes
    -----
    **One call, one ledger.** The guarantee covers one call on one fixed audit. Comparing ``k`` representations or
    models across calls, or keeping the best of ``k`` seeds, needs ``alpha / k`` per call. Growing the audit and
    looking again needs a ``SamplingPlan`` declared before the first look: look ``t`` uses the first ``sizes[t-1]``
    units of one seeded order (``order_seed``) and spends ``alpha / (t (t + 1))``.

    **Scope.** A ceiling decides at its information scope and every narrower one (``full_input`` ⊃
    ``fixed_representation`` ⊃ ``partition_restricted``). Image local-panel accuracy ceilings are
    ``partition_restricted`` without allowances, ``fixed_representation`` with β, and ``full_input`` with β and
    Γ; AP, recall and video local-panel ceilings stay ``partition_restricted``; replica ceilings are ``full_input``.
    Built-in witnesses read the representation, so they are eligible at
    ``fixed_representation`` and broader.

    **Population.** Unlabeled items are excluded and every claim is about the labeled items. A derived view
    (segments, frames, crops) needs ``groups=`` to say which items are independent; a ``View`` straight over a plain
    dataset groups by its source indices.

    References
    ----------
    .. [1] Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables.
       Journal of the American Statistical Association, 58(301), 13-30.
    .. [2] Maurer, A., & Pontil, M. (2009). Empirical Bernstein bounds and sample variance
       penalization. COLT 2009, Theorem 4.
    .. [3] Reid, M. D. and Williamson, R. C. (2011). Information, divergence and risk for
       binary experiments. Journal of Machine Learning Research, 12, 731-817.
    .. [4] DFE manuscript, local-pooling section (disjoint within-cell panels) and consensus inequality.
    .. [5] Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated
       in the case of the binomial. Biometrika, 26(4), 404-413.

    Examples
    --------
    A classification dataset whose given embeddings separate the classes, so the built-in witnesses reach an
    easy accuracy target:

    >>> import numpy as np
    >>> labels = np.array([int(np.argmax(train_ds[i][1])) for i in range(len(train_ds))])
    >>> embeddings = np.eye(4)[labels] + np.random.default_rng(0).normal(scale=0.1, size=(len(labels), 4))
    >>> result = Feasibility(seed=0).evaluate(train_ds, target={"accuracy": 0.5}, embeddings=embeddings)
    >>> result.determinations.select("metric", "threshold", "determination", "witness_source")
    shape: (1, 4)
    ┌──────────┬───────────┬───────────────────────┬────────────────────────┐
    │ metric   ┆ threshold ┆ determination         ┆ witness_source         │
    │ ---      ┆ ---       ┆ ---                   ┆ ---                    │
    │ enum     ┆ f64       ┆ enum                  ┆ str                    │
    ╞══════════╪═══════════╪═══════════════════════╪════════════════════════╡
    │ accuracy ┆ 0.5       ┆ demonstrated_feasible ┆ builtin_witness/v1:knn │
    └──────────┴───────────┴───────────────────────┴────────────────────────┘

    Each row cites the evidence behind it:

    >>> result.evidence["builtin_witness/v1:knn"].information_scope
    'fixed_representation'
    """

    class Config(EvaluatorConfig):
        """
        Configuration for the :class:`Feasibility` evaluator.

        Attributes
        ----------
        extractor : FeatureExtractor or None, default None
            Feature extractor for computing embeddings.
        batch_size : int or None, default None
            Batch size for embedding computation.
        alpha : float, default 0.05
            Error budget of one call.
        panel_size : int or None, default None
            Labels per local panel, at least 2 (None: provisional default).
        n_cells : int or None, default None
            Partition cells for local panels (None: by planned audit size).
        builtin_witnesses : bool, default True
            Fit and score the built-in k-NN and linear-probe witnesses.
        max_sequence_length : int or None, default None
            Declared largest group size, for ``frame_pooled`` targets.
        seed : int or None, default None
            Seed of the splits, partition and panels.
        """

        extractor: FeatureExtractor | None = None
        batch_size: PositiveInt | None = None
        alpha: OpenUnitInterval = 0.05
        panel_size: Annotated[int, Field(ge=2)] | None = None
        n_cells: PositiveInt | None = None
        builtin_witnesses: bool = True
        max_sequence_length: PositiveInt | None = None
        seed: int | None = None

    extractor: FeatureExtractor | None
    batch_size: int | None
    alpha: float
    panel_size: int | None
    n_cells: int | None
    builtin_witnesses: bool
    max_sequence_length: int | None
    seed: int | None
    config: Config

    def __init__(
        self,
        extractor: FeatureExtractor | None = None,
        *,
        batch_size: int | None = None,
        alpha: float | None = None,
        panel_size: int | None = None,
        n_cells: int | None = None,
        builtin_witnesses: bool | None = None,
        max_sequence_length: int | None = None,
        seed: int | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(locals())

    def _features(self, dataset: AnnotatedDataset[Any], embeddings: ArrayLike | None) -> NDArray[np.float64]:
        if embeddings is None:
            if self.extractor is None:
                raise ValueError("Provide pre-computed embeddings, or configure an extractor to compute them.")
            from dataeval._embeddings import Embeddings

            embeddings = Embeddings(dataset, extractor=self.extractor, batch_size=self.batch_size)
        features = np.asarray(embeddings, dtype=np.float64)
        if features.ndim != 2 or len(features) != len(dataset):
            raise ShapeMismatchError(
                f"embeddings must have shape (len(dataset), d) = ({len(dataset)}, d), got {features.shape}."
            )
        return features

    @set_metadata(state=_STATE)
    def evaluate(
        self,
        dataset: AnnotatedDataset[Any],
        *,
        target: Mapping[str, float] | Sequence[TargetSpec],
        embeddings: ArrayLike | None = None,
        groups: ArrayLike | str | None = None,
        replicas: Mapping[Any, ArrayLike] | None = None,
        predictions: ArrayLike | None = None,
        provenance: Mapping[str, Any] | None = None,
        allowances: Mapping[Literal["beta", "gamma"], float | ArrayLike] | None = None,
        sampling: SamplingPlan | None = None,
        look: int = 1,
    ) -> FeasibilityOutput:
        """
        Determine, for each target, whether the evidence demonstrates it, rules it out, or neither.

        Parameters
        ----------
        dataset : AnnotatedDataset
            An image or video classification dataset. Unlabeled items are excluded from the population.
        target : Mapping[str, float] or Sequence[TargetSpec]
            Thresholds by key ``"metric[:class][/scope][|aggregation]"`` (e.g. ``"accuracy"``, ``"ap:cat"``,
            ``"ap"`` for macro AP, ``"accuracy/fixed_representation"``, ``"accuracy|sequence_mean"``), or
            ``TargetSpec`` objects. The judged scope defaults to ``full_input``.
        embeddings : ArrayLike or None, default None
            One representation, shape ``(len(dataset), d)``; computed with the extractor when None.
        groups : ArrayLike or str or None, default None
            The independent unit of each item (array aligned with the dataset, or a metadata factor name).
            Required for derived views, except a ``View`` straight over a plain dataset, which defaults to its
            source indices. When some group holds more than one item, targets need an aggregation.
        replicas : Mapping[int or str, ArrayLike] or None, default None
            Extra independent class labels by datum id, for full_input replica ceilings. The covered items
            are recorded as a random sample of the population.
        predictions : ArrayLike or None, default None
            A frozen model's output on every dataset item, used as a user witness: class scores of shape
            ``(len(dataset), n_classes)``, whose top class is the decision, or a 1-D integer array of class
            decisions, indices of the dataset's declared classes (booleans and floats are rejected). Apply any other
            decision policy, such as a threshold on one class's score, before the call and pass its decisions; one
            call's decisions feed every target, so a different operating point needs its own call, with the error
            budget split across the planned calls. Class decisions rank nothing, so they give no AP witness.
        provenance : Mapping[str, Any] or None, default None
            The user witness's declared provenance, recorded as assumed; only with ``predictions``. Keys:
            ``reads`` — ``"full_input"`` (default; eligible at ``full_input``), ``"representation"`` (reads only
            the given representation; eligible at ``fixed_representation`` and broader); ``"partition"`` (one answer
            per cell of the internal partition) is rejected in this release. ``training_items`` (datum ids, any
            iterable, checked against the audit's groups); ``frozen_before_audit``, ``selection``, ``preprocessing``;
            ``decision_policy`` (the rule that turned scores into the given class decisions, frozen before the
            audit; only with 1-D class decisions). Other keys are rejected.
        allowances : Mapping[str, float or ArrayLike] or None, default None
            ``beta`` (cell heterogeneity: a scalar; a per-cell beta is rejected in this release) and ``gamma``
            (information loss: a scalar, with ``beta``), recorded as assumed. They widen the scope of image
            local-panel accuracy ceilings. Each must be a justified bound: ``beta=0`` asserts that no model reading
            the representation beats one fixed answer per cell.
        sampling : SamplingPlan or None, default None
            A sampling plan declared before the first look; its final size must not exceed the audit's units.
        look : int, default 1
            The 1-based look of ``sampling``; must be 1 without a plan.

        Returns
        -------
        FeasibilityOutput
            The ``determinations`` table and the ``evidence`` behind it.

        Raises
        ------
        NotImplementedError
            If the dataset is object detection or tracking (planned).
        ValueError
            If the dataset has fewer than 2 classes; a target, a class name, allowance, provenance key or
            ``reads``, ``panel_size``, look or sampling plan is invalid; class scores are misshapen or not finite, or
            class decisions are not integer indices of declared classes, one per item; a ``decision_policy`` comes
            without class decisions; a replica key is unknown;
            neither embeddings nor an extractor is given; a derived view lacks ``groups``; or the user witness's
            training items cannot match the datum ids or share groups with the audit.
        ShapeMismatchError
            If the embeddings do not have one row per dataset item, or a ``groups`` array is not aligned with the
            dataset.
        """
        reject_filtered_metadata(dataset, "Feasibility")
        task = detect_task(dataset[0][1])
        if task in ("OD", "MOT"):
            raise NotImplementedError(
                f"Feasibility supports image and video classification in this release; {task} is planned."
            )
        # The evaluator warned on construction; its experimental parts stay quiet without spending their warnings.
        with quiet_experimental():
            units = resolve_units(dataset, groups=groups)
            if units.n_classes < 2:
                raise ValueError(f"Feasibility needs at least 2 classes, the dataset declares {units.n_classes}.")
            specs = parse_targets(target, task="IC", class_names=units.class_names, grouped=units.grouped)
            _check_class_names(specs, units)
            # Cheap argument checks first, before the embedding pass.
            if self.panel_size is not None and self.panel_size < 2:
                raise ValueError(f"panel_size must be at least 2, got {self.panel_size}.")
            checked = _check_allowances(allowances or {})
            _check_provenance(provenance, predictions)
            _check_predictions(predictions, units, provenance)
            by_position = resolve_replicas(replicas, dataset, units) if replicas else {}
            seed = self.seed if self.seed is not None else (get_seed() or 0)
            fit, audit = split_units(units, fit_fraction=FIT_FRACTION, seed=seed)
            audit = _audit_prefix(units, audit, sampling, look)
            alpha = self.alpha if sampling is None else sampling.look_alpha(self.alpha, look)
        features = self._features(dataset, embeddings)[units.item_indices]  # the user's extractor warns as it would
        with quiet_experimental():
            n_claims = count_claims(
                specs,
                has_replicas=bool(by_position),
                has_predictions=predictions is not None,
                builtin_witnesses=self.builtin_witnesses,
            )
            evidence = gather_evidence(
                specs,
                units=units,
                features=features,
                audit=audit,
                fit=fit,
                replicas=by_position,
                predictions=predictions,
                provenance=provenance,
                allowances=checked,
                alpha_claim=allocate(alpha, max(n_claims, 1)),  # no source at all: no endpoint spends anything
                seed=seed,
                sampling=sampling,
                look=look,
                max_sequence_length=self.max_sequence_length,
                n_cells=self.n_cells,
                panel_size=self.panel_size,
                builtin_witnesses=self.builtin_witnesses,
            )
            check_ledger(evidence.values(), alpha)
            recorded = {
                "seed": seed,
                "call_alpha": alpha,
                "allocation": ALLOCATION,
                "order_seed": None if sampling is None else sampling.order_seed,
            }
            evidence = {k: replace(v, details={**v.details, **recorded}) for k, v in evidence.items()}
            rows = [_row(spec, decide(spec, evidence, units.population)) for spec in specs]
            return FeasibilityOutput(pl.DataFrame(rows, schema=_SCHEMA), evidence)
