__all__ = []

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import PositiveInt, field_validator

from dataeval._experimental import experimental, quiet_experimental
from dataeval._helpers import reject_filtered_metadata
from dataeval.config import get_batch_size, get_seed
from dataeval.exceptions import ShapeMismatchError
from dataeval.performance._feasibility_classification import (
    _class_index,
    check_predictions,
    count_claims,
    gather_evidence,
    table_order,
)
from dataeval.performance._feasibility_constants import FIT_FRACTION
from dataeval.performance._feasibility_decide import Decision, Determination, Limitation, decide
from dataeval.performance._feasibility_detection import (
    capacity_evidence,
    check_counts,
    check_detection_call,
    check_detection_layout,
    check_detections,
    gather_detection_evidence,
    read_model_output,
    resolve_class_ids,
    resolve_count_bounds,
)
from dataeval.performance._feasibility_evidence import (
    Assumption,
    Bound,
    BoundType,
    Method,
    Population,
    Quantity,
    Reason,
    Role,
    Sample,
    Source,
    State,
    Unit,
    allocate,
    check_ledger,
)
from dataeval.performance._feasibility_model import check_class_order, model_scores, run_model
from dataeval.performance._feasibility_target import (
    Aggregation,
    Allowances,
    APFunctional,
    Metric,
    Scope,
    TargetSpec,
    Task,
    parse_targets,
    positive_count,
)
from dataeval.performance._feasibility_units import (
    Units,
    _declared_class_names,
    resolve_detection_units,
    resolve_replicas,
    resolve_units,
    split_units,
)
from dataeval.protocols import AnnotatedDataset, ArrayLike, FeatureExtractor, Model, ObjectDetectionTarget
from dataeval.types import DictOutput, Evaluator, EvaluatorConfig, set_metadata
from dataeval.types._config import OpenUnitInterval
from dataeval.types._task import detect_task


def _enum(literal: Any) -> pl.Enum:
    return pl.Enum(list(get_args(literal)))


# The stable API: names and dtypes are frozen in tests. Enums span every task planned, not only phase 1's.
_DETERMINATIONS_SCHEMA = {
    "requirement": pl.Int64,
    "task": _enum(Task),
    "metric": _enum(Metric),
    "threshold": pl.Float64,
    "judged_scope": _enum(Scope),
    "class_name": pl.String,  # a class name from the dataset's own labels: the one free string
    "aggregation": _enum(Aggregation),
    "ap_functional": _enum(APFunctional),
    "max_false_positive_rate": pl.Float64,
    "iou": pl.Float64,
    "max_detections": pl.Int64,
    "determination": _enum(Determination),
    "limitation": _enum(Limitation),
    "suggested_minimum": pl.Int64,
    "minimum_source": _enum(Source),
    "minimum_quantity": _enum(Quantity),
    "envelope": pl.Float64,
    "envelope_scope": _enum(Scope),
    "witness": pl.Float64,
    "witness_source": _enum(Source),
}
_BOUNDS_SCHEMA = {
    "requirement": pl.Int64,
    "source": _enum(Source),
    "role": _enum(Role),
    "quantity": _enum(Quantity),
    "bound_type": _enum(BoundType),
    "information_scope": _enum(Scope),
    "population": _enum(Population),
    "sample": _enum(Sample),
    "unit": _enum(Unit),
    "state": _enum(State),
    "reason": _enum(Reason),
    "bound": pl.Float64,
    "observed": pl.Float64,
    "n": pl.Int64,
    "variance": pl.Float64,
    "radius": pl.Float64,
    "method": _enum(Method),
    "alpha": pl.Float64,
    "assumptions": pl.List(_enum(Assumption)),
    "is_determining": pl.Boolean,
}
_STATE = ["alpha", "max_sequence_length", "max_objects", "resolved_seed"]
_PRIVATE = ("mean", "support", "delta", "project")  # on a ``Bound``, not columns of the ``bounds`` table


def _determinations_frame(specs: Sequence[TargetSpec], determinations: Sequence[Decision]) -> pl.DataFrame:
    """One row per requirement, in the order given."""
    rows = [
        {
            "requirement": i,
            "task": s.task,
            "metric": s.metric,
            "threshold": s.threshold,
            "judged_scope": s.scope,
            "class_name": s.class_name,
            "aggregation": s.aggregation,
            "ap_functional": s.ap_functional,
            "max_false_positive_rate": s.max_false_positive_rate,
            "iou": s.iou,
            "max_detections": s.max_detections,
            "determination": d.determination,
            "limitation": d.limitation,
            "suggested_minimum": d.suggested_minimum,
            "minimum_source": d.minimum_source,
            "minimum_quantity": d.minimum_quantity,
            "envelope": d.envelope,
            "envelope_scope": d.envelope_scope,
            "witness": d.witness,
            "witness_source": d.witness_source,
        }
        for i, (s, d) in enumerate(zip(specs, determinations, strict=True))
    ]
    return pl.DataFrame(rows, schema=_DETERMINATIONS_SCHEMA)


def _bounds_frame(bounds: Sequence[Bound], determinations: Sequence[Decision]) -> pl.DataFrame:
    """One row per bound, in the order ``gather_evidence`` gives, flagged with whether its determination rests on it."""
    rows = [
        {
            **{k: v for k, v in vars(b).items() if k not in _PRIVATE},
            "assumptions": list(b.assumptions),
            "is_determining": (b.source, b.quantity) in determinations[b.requirement].determining,
        }
        for b in bounds
    ]
    return pl.DataFrame(rows, schema=_BOUNDS_SCHEMA)


def _pct(value: float, direction: Literal["lower", "upper"]) -> str:
    """Format to 0.1%; a lower bound rounds down and an upper bound up, so rounding never overstates a bound."""
    tenths = value * 1000  # the 1e-9 absorbs float error only, such as 0.957 * 1000 = 956.9999999999999
    tenths = math.floor(tenths + 1e-9) if direction == "lower" else math.ceil(tenths - 1e-9)
    return f"{tenths / 10:.1f}%"


def _label(spec: TargetSpec) -> str:
    """Return a requirement's name, built from its fields alone."""
    text = f"{spec.metric}:{spec.class_name}" if spec.class_name is not None else spec.metric
    text += f" ≥{spec.threshold:g}"
    if spec.metric == "recall":
        text += f" fp≤{spec.max_false_positive_rate:g}"
    if spec.task == "OD":
        text += f" iou≥{spec.iou:g} k≤{spec.max_detections}"
    if spec.aggregation is not None:
        text += f" |{spec.aggregation}"
    return text if spec.scope == "full_input" else f"{text} @{spec.scope}"


def _max_objects_lines(specs: Sequence[TargetSpec], bound: Any) -> list[str]:
    """Return the ``max_objects`` line: each object detection class once, in requirement order, with its B."""
    classes = dict.fromkeys(s.class_name for s in specs if s.task == "OD")
    if not classes:
        return []
    if not isinstance(bound, int | dict):  # no execution metadata, as on a copy that never went through evaluate
        return ["max_objects: —"]
    return ["max_objects: " + ", ".join(f"{c} {bound[c] if isinstance(bound, dict) else bound}" for c in classes)]


def _determining_source(bounds: list[dict[str, Any]], requirement: int, role: Role) -> str:
    """Return the source of a requirement's determining bound of ``role``; empty when none."""
    return next(
        (b["source"] for b in bounds if b["requirement"] == requirement and b["role"] == role and b["is_determining"]),
        "",
    )


@experimental
@dataclass(frozen=True, repr=False)
class FeasibilityOutput(DictOutput):
    """
    Output class for the :class:`.Feasibility` evaluator.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    Attributes
    ----------
    requirements : tuple[TargetSpec, ...]
        The requirements in the order given; a row's ``requirement`` is an index into it.
    allowances : Allowances or None
        The allowances passed, as given.
    determinations : pl.DataFrame
        One row per requirement, with columns:

        - requirement: int - Index into ``requirements``
        - task, metric, threshold, judged_scope: Enum, Enum, float, Enum - What was asked: ``IC`` or ``OD``, the
          metric, its threshold and the scope it is judged at
        - class_name: str - The class the requirement names, null otherwise
        - aggregation, ap_functional, max_false_positive_rate: Enum, Enum, float - The details that tell requirements
          of one metric apart, null where they do not apply
        - iou, max_detections: float, int - The IoU and the most detections per image of an object detection
          requirement, null for classification
        - determination: Enum - ``demonstrated_feasible``, ``infeasible_under_recorded_assumptions``,
          ``not_ruled_out`` or ``unresolved``
        - limitation: Enum - What limits a non-final determination, null for a final one: ``conflict``, ``no_envelope``,
          ``scope_restricted``, ``missing_bound``, ``envelope_unavailable``, ``no_eligible_witness``,
          ``within_radius`` or ``weak_witness``
        - suggested_minimum: int - Units at which the bound named by ``minimum_source`` and ``minimum_quantity`` would
          cross the requirement if the observed rates held: a projection, not a bound. It counts the units of that
          row's ``n``, drawn as its ``sample`` says: audit units for a witness or local panels, units with replicas for
          a replica envelope, every image for a capacity envelope, and for recall the evaluation images, counted as the
          recall row's ``n`` plus the false-positive row's are. So ``suggested_minimum / n`` (for recall, the sum of the
          two rows' ``n``) is how much that sample must grow. Null when not attainable or when a final determination
          needs none
        - minimum_source, minimum_quantity: Enum - The ``bounds`` row, with ``requirement``, that the minimum projects
          from
        - envelope, envelope_scope: float, Enum - The determining envelope's upper bound and its information scope
        - witness, witness_source: float, Enum - The determining witness's lower bound and its source; for a conflict,
          the conflicting pair sits in these four columns
    bounds : pl.DataFrame
        One row per bound, ordered by ``requirement`` and then source, with columns:

        - requirement: int - Index into ``requirements``
        - source: Enum - ``local_panels``, ``replicas``, ``capacity``, ``knn``, ``linear_probe`` or ``user_model``
        - role: Enum - ``envelope`` or ``witness``
        - quantity: Enum - The metric, or ``false_positive_rate`` for a recall witness's second bound
        - bound_type: Enum - ``lower`` or ``upper``
        - information_scope, population, sample, unit: Enum - What the bound covers: ``full_input``,
          ``fixed_representation`` or ``partition_restricted``; ``items`` or ``labeled_items``; the sample it is
          computed on, ``audit_split``, ``audit_split_one_per_group``, ``replica_items`` or ``all_items`` (every
          image, for a capacity envelope); ``image`` or ``sequence``, always ``image`` for object detection
        - state: Enum - ``unavailable``, ``uninformative`` or ``informative``
        - reason: Enum - Why a bound is unavailable or uninformative, null otherwise
        - bound: float - The confidence bound; null when unavailable
        - observed: float - A witness's point estimate of the quantity. It is null for every envelope, for an
          unavailable witness, for a ``frame_pooled`` accuracy witness, and for an AP witness when its class has no
          target (object detection) or positive (classification) on the audit, or, for macro AP, when any class has
          none. For an AP witness it is the AP staircase of the witness's thresholds at the observed rates, computed at
          those thresholds only: compare it with ``bound``, not with AP from the model's full output
        - n, variance: int, float - Independent units behind the bound and their variance; null where the method does
          not define them
        - radius: float - The half-width the named method applied to the estimate to produce ``bound``; null for exact
          intervals, AP and recall envelopes, AP witnesses and ``frame_pooled`` bounds
        - method: Enum - ``hoeffding``, ``hybrid`` or ``exact_binomial``; null where the bound has no closed form
        - alpha: float - This bound's share of the call's error budget
        - assumptions: list[Enum] - What the bound rests on. Every member is declared and not verified
        - is_determining: bool - Whether the requirement's determination rests on this bound

    The call's ``alpha``, resolved seed and ``max_objects`` are in ``meta().state``.
    """

    requirements: tuple[TargetSpec, ...]
    allowances: Allowances | None
    determinations: pl.DataFrame
    bounds: pl.DataFrame

    @property
    def plot_type(self) -> Literal["feasibility"]:
        return "feasibility"

    def __str__(self) -> str:
        """
        Print one line per requirement, then the joint confidence, the assumptions and what a minimum means.

        A line gives the requirement, its determination, its limitation, the determining envelope and witness with their
        sources, and the suggested minimum. Lower bounds round down and upper bounds up. When a requirement is object
        detection, a ``max_objects`` line after the assumptions gives the count bound of each class the requirements
        name, and the printout ends with a line naming the units of a witness's and a capacity envelope's minimum.
        """
        bounds = list(self.bounds.iter_rows(named=True))
        rows = [("#", "requirement", "determination", "limitation", "envelope", "witness", "minimum")]
        for d, spec in zip(self.determinations.iter_rows(named=True), self.requirements, strict=True):
            i = d["requirement"]
            envelope, witness = "—", "—"
            if d["envelope"] is not None:
                source = _determining_source(bounds, i, "envelope")
                envelope = f"{_pct(d['envelope'], 'upper')} {source or d['envelope_scope']}"
            if d["witness"] is not None:
                witness = f"{_pct(d['witness'], 'lower')} {d['witness_source']}"
            minimum = "—"
            if d["suggested_minimum"] is not None:
                minimum = f"{d['suggested_minimum']:,} {d['minimum_source']} {d['minimum_quantity']}"
            rows.append((str(i), _label(spec), d["determination"], d["limitation"] or "—", envelope, witness, minimum))
        widths = [max(len(row[c]) for row in rows) for c in range(len(rows[0]))]
        lines = ["   ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip() for row in rows]
        alpha = self.meta().state.get("alpha")
        confidence = "—" if alpha is None else _pct(1 - alpha, "lower").replace(".0%", "%")
        assumed = {a for b in bounds for a in b["assumptions"]}
        assumptions = ", ".join(a for a in get_args(Assumption) if a in assumed) or "none"
        units = (
            ["minimum units: a witness counts audit images, half of all images; a capacity envelope counts every image"]
            if any(s.task == "OD" for s in self.requirements)
            else []
        )
        return "\n".join([
            *lines,
            f"joint confidence: {confidence}",
            f"assumes: {assumptions}",
            *_max_objects_lines(self.requirements, self.meta().state.get("max_objects")),
            "minimum: units of the named bound's n to cross the requirement if the observed rates hold",
            *units,
        ])


def _check_allowances(allowances: Allowances | None) -> None:
    if allowances is not None and not isinstance(allowances, Allowances):
        raise ValueError(
            f"allowances must be an Allowances, e.g. Allowances(beta=0.01, gamma=0.02), got {type(allowances).__name__}"
        )


def _check_class_names(specs: Sequence[TargetSpec], units: Units) -> None:
    """Map every class name to its index now, so an unmappable one raises before the embedding pass."""
    for spec in specs:
        if spec.class_name is not None:
            _class_index(units, spec.class_name)


def _check_predictions(predictions: ArrayLike | None, units: Units) -> None:
    """Check the user witness's predictions now, so bad ones raise before the embedding pass."""
    if predictions is not None:
        check_predictions(predictions, len(units.item_ids), units.n_classes)


def _check_sources(
    model: Model | None,
    predictions: ArrayLike | None,
    replicas: Mapping[Any, ArrayLike] | None,
    embeddings: ArrayLike | None,
    extractor: FeatureExtractor | None,
) -> None:
    """Refuse a call with two witnesses or with nothing to bound from."""
    if model is not None and predictions is not None:
        raise ValueError("Pass model or predictions, not both: each is the user's witness, and a call has one.")
    if model is None and predictions is None and not replicas and embeddings is None and extractor is None:
        raise ValueError(
            "Feasibility has nothing to bound with: pass predictions, a model, replicas, embeddings or an extractor."
        )


def _supported_task(dataset: AnnotatedDataset[Any]) -> Literal["IC", "OD"] | None:
    """Return the dataset's task, read from its first target, or None when unknown; tracking raises."""
    task = detect_task(dataset[0][1])
    if task == "MOT":
        raise NotImplementedError(
            "Feasibility supports image and video classification and object detection in this release, not MOT."
        )
    return task


def _model_scores(model: Model, dataset: AnnotatedDataset[Any], units: Units, batch_size: int) -> NDArray[np.float64]:
    """Run the model over the dataset and check it scored every item for every class."""
    scores = model_scores(model, dataset, batch_size)
    expected = (len(dataset), units.n_classes)
    if scores.shape != expected:
        raise ValueError(
            f"The model must score every dataset item for every class: expected shape (len(dataset), n_classes) = "
            f"{expected}, got {scores.shape}."
        )
    return scores


@experimental
class Feasibility(Evaluator):
    """
    Decide whether a classification or object detection AP requirement is reachable, with a proof behind every claim.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    For each requirement (a :class:`.TargetSpec`: a metric, its details and a threshold), Feasibility gathers
    **envelopes** (upper bounds on what any model in a class can reach on the population) and **witnesses** (lower
    bounds on what a frozen model does reach), all simultaneous at level ``alpha`` over one call, and applies the
    DFE's ordered rules, giving one determination per requirement: ``demonstrated_feasible``,
    ``infeasible_under_recorded_assumptions``, ``not_ruled_out`` or ``unresolved``. A conflict, a certified witness
    whose lower bound exceeds an envelope's upper bound, suspends the conclusion: ``unresolved`` with limitation
    ``conflict``. A non-final determination names its limitation. A recall requirement (a ``TargetSpec`` with
    ``metric="recall"``, a class and a false-positive limit) counts a witness only when its false-positive upper bound
    is within the limit, and its envelopes bound the recall any model in their scope reaches at that limit; the
    ``bounds`` table shows both of a witness's bounds. An object detection requirement is the AP of one class.

    Sources: local-panel envelopes, DataEval's post-stratified pooling (C3) of the DFE's disjoint label panels [4]_
    within the cells of a label-blind partition fitted on a group-disjoint fit split, proved in
    :func:`~dataeval.core.local_panel_bound`'s Notes with two Hoeffding [1]_ steps; replica envelopes from
    ``replicas=``; object detection capacity envelopes from the targets of every image; built-in frozen k-NN and
    linear-probe witnesses fitted on the fit split; and a user witness from ``predictions=`` or ``model=``, which
    counts at ``full_input`` scope. Each (requirement, source) pair gets an equal share of ``alpha``. Bounds on means
    use Hoeffding [1]_ and empirical Bernstein [2]_ radii, except that ungrouped witness bounds (accuracy, recall,
    false-positive rate and AP indicators) use exact Clopper-Pearson [5]_ intervals; classification AP and recall
    envelopes are built on the cost-weighted risk inequality of [3]_. An object detection AP witness or capacity
    envelope bounds the means of per-image counts with the Hoeffding and empirical Bernstein radii.

    Parameters
    ----------
    extractor : FeatureExtractor or None, default None
        Feature extractor that computes the representation; optional when ``embeddings`` are given to
        :meth:`evaluate`. It must have been fitted and selected independently of the audit items' labels. Ignored for
        object detection.
    batch_size : int or None, default None
        Batch size for embedding computation and for scoring a ``model``; a ``model`` also needs this or a global
        batch size from :func:`dataeval.config.set_batch_size`.
    alpha : float, default 0.05
        Error budget of one call, shared by every bound of every requirement.
    seed : int or None, default None
        Seed of the splits, partition and panels; None uses :func:`dataeval.config.get_seed`, else 0.
    max_sequence_length : int or None, default None
        Declared largest group size for grouped (video) data, needed by ``frame_pooled`` requirements. Data that
        violate it raise rather than being truncated. Ignored for object detection.
    max_objects : int or Mapping[str, int] or None, default None
        Required for object detection. Declared largest number of targets of a class in any image of the population:
        one int for every class, or a mapping keyed by ``TargetSpec.class_name`` that names every requirement class.
        It is assumed and not verified, and recorded as ``max_objects_holds``. A larger value weakens the bounds, and a
        value below the truth makes them invalid. The largest count in the data is only a floor for it, and data of a
        requirement's class above it raise. A requirement gets a capacity envelope only when its ``max_detections`` is
        below this value. That envelope takes its equal share of ``alpha`` even when it is uninformative, so it
        widens every witness's bound in the call.
    config : Feasibility.Config or None, default None
        Optional configuration object; parameters passed directly to ``__init__`` override it.

    Notes
    -----
    **One call, one audit.** The guarantee covers one call on one fixed audit. Comparing ``k`` representations or
    models across calls, keeping the best of ``k`` seeds, or growing the audit and looking again, needs ``alpha / k``
    per call.

    **Witness only.** A requirement with a witness (``predictions`` or ``model``) and no envelope of any scope is
    witness only. That covers every requirement of a classification call with no embeddings, no extractor and no
    replicas, and every object detection requirement whose ``max_detections`` is at or above its ``max_objects``,
    whatever the other requirements of the call. A witness-only requirement's determination is
    ``demonstrated_feasible``, or ``unresolved`` with limitation ``no_envelope``. It can never be
    ``infeasible_under_recorded_assumptions`` or ``not_ruled_out``, because only an envelope can bound what is
    unreachable.

    **Object detection.** A requirement whose ``max_detections`` K is below its ``max_objects`` gets a capacity
    envelope: an upper bound on the AP that any detector with at most K detections per image can reach, read from
    the targets, K and ``max_objects`` alone. It is ``full_input``, so it decides at every scope, and it is
    informative only when some images hold more than K targets of the class. A capacity envelope below the threshold
    means that, under the recorded assumptions, no detector with at most K detections per image reaches the
    threshold: K is too small for how many targets of the class the images hold, and only a larger K, which is a
    different requirement, could reach it. A capacity envelope's ``suggested_minimum`` is the number of images at which
    it would rule the requirement out at this K, and it counts every image. A witness's is the number of audit images
    at which it would demonstrate the requirement. The audit is half of the images in this release, so supply about
    twice a witness's minimum. Both hold ``max_objects`` and the requirement's ``max_detections`` fixed, along with the
    rates observed in this call. A call without a detector (``predictions`` or ``model``) can never be
    ``demonstrated_feasible``. In such a call, a requirement whose envelope is computed and does not rule it out has
    limitation ``no_eligible_witness``. More images, which narrow the envelope, can also settle it, but only when the
    share of the class's targets that K detections can match is below the threshold.

    **Scope.** An envelope decides at its information scope and every narrower one (``full_input`` ⊃
    ``fixed_representation`` ⊃ ``partition_restricted``). Image local-panel accuracy envelopes are
    ``partition_restricted`` without allowances, ``fixed_representation`` with β, and ``full_input`` with β and
    Γ; AP, recall and video local-panel envelopes stay ``partition_restricted``; replica and object detection capacity
    envelopes are ``full_input``. Built-in witnesses read the representation, so they are eligible at
    ``fixed_representation`` and broader.

    **Population.** For classification, unlabeled items are excluded and every claim is about the labeled items. For
    object detection, every image counts, images with no targets included. A derived view (segments, frames, crops)
    needs ``groups=`` to say which items are independent; a ``View`` straight over a plain dataset groups by its source
    indices.

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
    >>> from dataeval.performance import TargetSpec
    >>> target = [TargetSpec(metric="accuracy", threshold=0.5)]
    >>> result = Feasibility(seed=0).evaluate(train_ds, target=target, embeddings=embeddings)
    >>> result.determinations.select("metric", "threshold", "determination", "witness_source")
    shape: (1, 4)
    ┌──────────┬───────────┬───────────────────────┬────────────────┐
    │ metric   ┆ threshold ┆ determination         ┆ witness_source │
    │ ---      ┆ ---       ┆ ---                   ┆ ---            │
    │ enum     ┆ f64       ┆ enum                  ┆ enum           │
    ╞══════════╪═══════════╪═══════════════════════╪════════════════╡
    │ accuracy ┆ 0.5       ┆ demonstrated_feasible ┆ knn            │
    └──────────┴───────────┴───────────────────────┴────────────────┘

    Printing it gives one line per requirement, then the joint confidence, the assumptions and what a minimum means:

    >>> print(result)
    #   requirement     determination           limitation   envelope   witness     minimum
    0   accuracy ≥0.5   demonstrated_feasible   —            —          81.4% knn   —
    joint confidence: 95%
    assumes: representation_independent_of_audit_labels
    minimum: units of the named bound's n to cross the requirement if the observed rates hold

    The minimal classification call is a dataset, predictions (or a ``model``) and a requirement. With no embeddings,
    no extractor and no replicas there is no envelope, so a requirement that the witness cannot demonstrate stays
    unresolved:

    >>> scores = np.eye(4)[labels]
    >>> hard = [TargetSpec(metric="accuracy", threshold=0.999)]
    >>> bare = Feasibility().evaluate(train_ds, predictions=scores, target=hard)
    >>> bare.determinations.select("determination", "limitation")
    shape: (1, 2)
    ┌───────────────┬─────────────┐
    │ determination ┆ limitation  │
    │ ---           ┆ ---         │
    │ enum          ┆ enum        │
    ╞═══════════════╪═════════════╡
    │ unresolved    ┆ no_envelope │
    └───────────────┴─────────────┘

    Each determination cites the bounds behind it:

    >>> result.bounds.filter("is_determining").select("source", "role", "bound_type", "information_scope")
    shape: (1, 4)
    ┌────────┬─────────┬────────────┬──────────────────────┐
    │ source ┆ role    ┆ bound_type ┆ information_scope    │
    │ ---    ┆ ---     ┆ ---        ┆ ---                  │
    │ enum   ┆ enum    ┆ enum       ┆ enum                 │
    ╞════════╪═════════╪════════════╪══════════════════════╡
    │ knn    ┆ witness ┆ lower      ┆ fixed_representation │
    └────────┴─────────┴────────────┴──────────────────────┘

    An object detection requirement names one class, an IoU and the most detections per image. Here the dataset's own
    targets stand in for a detector's output. ``max_objects=3`` comes from how the dataset was generated, with at most
    three boxes per image:

    >>> detections = [dataset[i][1] for i in range(len(dataset))]
    >>> person = TargetSpec(task="OD", metric="ap", class_name="person", threshold=0.5, iou=0.5, max_detections=10)
    >>> result = Feasibility(max_objects=3).evaluate(dataset, target=[person], predictions=detections)
    >>> print(result)
    #   requirement                   determination   limitation    envelope   witness           minimum
    0   ap:person ≥0.5 iou≥0.5 k≤10   unresolved      no_envelope   —          0.0% user_model   2,154 user_model ap
    joint confidence: 95%
    assumes: model_independent_of_audit_items, class_order_matches_dataset, max_objects_holds
    max_objects: person 3
    minimum: units of the named bound's n to cross the requirement if the observed rates hold
    minimum units: a witness counts audit images, half of all images; a capacity envelope counts every image

    The detections match the targets, and the witness is still 0.0%: this 25-image audit is too small for an informative
    bound, whatever the detector. The minimum projects how many audit images would carry the witness to 0.5 if this
    synthetic audit's rates held. It is not a bound and says nothing about a real detector.
    """

    class Config(EvaluatorConfig):
        """
        Configuration for the :class:`Feasibility` evaluator.

        Attributes
        ----------
        extractor : FeatureExtractor or None, default None
            Feature extractor for computing embeddings.
        batch_size : int or None, default None
            Batch size for embedding computation and for scoring a ``model``; a ``model`` also needs this or a
            global batch size from :func:`dataeval.config.set_batch_size`.
        alpha : float, default 0.05
            Error budget of one call.
        max_sequence_length : int or None, default None
            Declared largest group size, for ``frame_pooled`` requirements.
        max_objects : int or dict[str, int] or None, default None
            Declared count bound for object detection requirements.
        seed : int or None, default None
            Seed of the splits, partition and panels.
        """

        extractor: FeatureExtractor | None = None
        batch_size: PositiveInt | None = None
        alpha: OpenUnitInterval = 0.05
        max_sequence_length: PositiveInt | None = None
        max_objects: int | dict[str, int] | None = None
        seed: int | None = None

        @field_validator("max_objects", mode="before")
        @classmethod
        def _check_max_objects(cls, value: Any) -> Any:
            """Accept integers of at least 1, never a bool or a float, before pydantic's own coercion runs."""
            if value is None:
                return None
            if not isinstance(value, Mapping):
                return positive_count(value, "max_objects")
            for key in value:
                if not isinstance(key, str):
                    raise ValueError(
                        f"max_objects keys are class names (TargetSpec.class_name), e.g. {{'0': 18}}; got key {key!r}"
                    )
            return {key: positive_count(count, f"max_objects[{key!r}]") for key, count in value.items()}

    extractor: FeatureExtractor | None
    batch_size: int | None
    alpha: float
    max_sequence_length: int | None
    max_objects: int | dict[str, int] | None
    seed: int | None
    config: Config

    def __init__(
        self,
        extractor: FeatureExtractor | None = None,
        *,
        batch_size: int | None = None,
        alpha: float | None = None,
        seed: int | None = None,
        max_sequence_length: int | None = None,
        max_objects: int | Mapping[str, int] | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(locals())

    def _model_batch_size(self) -> int:
        """Return the batch size ``model=`` runs in: ``batch_size``, else the global one. There is no default."""
        try:
            return get_batch_size(self.batch_size)
        except ValueError as error:
            raise ValueError(
                "model= scores the dataset in batches, and none is set: pass Feasibility(batch_size=...) or call "
                "dataeval.config.set_batch_size()."
            ) from error

    @property
    def resolved_seed(self) -> int:
        """The seed a call uses: ``seed``, else :func:`dataeval.config.get_seed`, else 0."""
        return self.seed if self.seed is not None else (get_seed() or 0)

    def _features(self, dataset: AnnotatedDataset[Any], embeddings: ArrayLike | None) -> NDArray[np.float64] | None:
        """Return the representation, or None when there is none: no embeddings and no extractor."""
        if embeddings is None:
            if self.extractor is None:
                return None
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
        target: Sequence[TargetSpec],
        predictions: ArrayLike | Sequence[ObjectDetectionTarget] | None = None,
        model: Model | None = None,
        embeddings: ArrayLike | None = None,
        groups: ArrayLike | str | None = None,
        replicas: Mapping[Any, ArrayLike] | None = None,
        allowances: Allowances | None = None,
    ) -> FeasibilityOutput:
        """
        Decide, for each requirement, whether the bounds demonstrate it, rule it out, or neither.

        Parameters
        ----------
        dataset : AnnotatedDataset
            An image or video classification dataset, or an object detection dataset. Unlabeled classification items
            are excluded from the population.
        target : Sequence[TargetSpec]
            The requirements, in the order wanted back. The judged scope of each defaults to ``full_input``.
        predictions : ArrayLike or Sequence[ObjectDetectionTarget] or None, default None
            A frozen model's output on every dataset item, used as a user witness. For classification, class scores of
            shape ``(len(dataset), n_classes)``, whose top class is the decision, or a 1-D integer array of class
            decisions, indices of the dataset's declared classes (booleans and floats are rejected). Apply any other
            decision policy, such as a threshold on one class's score, before the call and pass its decisions; one
            call's decisions feed every requirement, so a different operating point needs its own call, with the
            error budget split across the planned calls. Class decisions rank nothing, so they give no AP witness. The
            model is assumed frozen before the audit and independent of the audit items, which a public pretrained
            model may not be. It counts at ``full_input``. Give ``predictions`` or ``model``, not both. For object
            detection, a detector, ``predictions`` or ``model``, is required unless every requirement's
            ``max_detections`` is below its ``max_objects``. ``predictions`` is one detections object per image, in
            dataset order, with ``boxes`` ``(N, 4)`` as ``(x0, y0, x1, y1)`` in the coordinates of the dataset's
            targets, ``labels`` (indices of the declared classes) and ``scores`` ``(N,)`` or ``(N, n_classes)``, the
            MAITE ``ObjectDetectionTarget`` shape. The labels are assumed to follow the dataset's class order,
            recorded as ``class_order_matches_dataset``.
        model : Model or None, default None
            A frozen MAITE image classifier or object detector, run over the whole dataset in batches of
            ``batch_size`` (or the global batch size) in place of ``predictions``. Each batch is a list of the
            dataset's images, as the dataset returns them. Its output feeds the same witness, so a model gives the
            same determinations and bound values as its own output passed as ``predictions``. A classifier's class
            scores keep the model's own precision, so it can witness AP, which class decisions cannot. A detector
            returns one detections object per image in the ``predictions`` layout, taken as it is: DataEval applies no
            NMS and no score cut, so the witness of a raw-candidate export stays valid but can be weak. Wrap a
            detector whose boxes are in a resized input's pixels or normalized to ``[0, 1]``, or whose labels follow
            another class list. DataEval's own :class:`.OnnxObjectDetector` and :class:`.LiteRtObjectDetector` need
            ``box_coordinates="pixel"``, an export whose class count equals the dataset's, and a batch size the export
            accepts for every batch, the short last one included; they declare no ``index2label``, so the class order
            is assumed. If the model's ``metadata`` and the dataset both declare an ``index2label`` they must be equal,
            else ``ValueError``; a class order is never remapped. If either declares none, the dataset's class order is
            assumed and recorded as ``class_order_matches_dataset``.
        embeddings : ArrayLike or None, default None
            Classification only. One representation, shape ``(len(dataset), d)``; computed with the extractor when
            None. With neither embeddings nor an extractor the call has no local-panel envelope and no built-in
            witness.
        groups : ArrayLike or str or None, default None
            The independent unit of each item (array aligned with the dataset, or a metadata factor name).
            Required for derived views, except a ``View`` straight over a plain dataset, which defaults to its
            source indices. When some group holds more than one item, requirements need an aggregation. Object
            detection images are assumed independent, and grouping them is refused.
        replicas : Mapping[int or str, ArrayLike] or None, default None
            Classification only. Extra independent class labels by datum id, for full_input replica envelopes. The
            covered items are recorded as a random sample of the population.
        allowances : Allowances or None, default None
            Classification only. ``beta`` (cell heterogeneity) and ``gamma`` (information loss), recorded as assumed.
            They widen the scope of image local-panel accuracy envelopes.

        Returns
        -------
        FeasibilityOutput
            The ``determinations`` table and the ``bounds`` behind it.

        Raises
        ------
        NotImplementedError
            If the dataset is tracking.
        ValueError
            If a classification dataset has fewer than 2 classes; a requirement, a class name or the allowances is
            invalid; both ``model`` and ``predictions`` are given; ``model`` is given with no ``batch_size`` and no
            global batch size; a classification call is given none of ``predictions``, ``model``, ``replicas``,
            ``embeddings`` or an extractor; the model's and the dataset's ``index2label`` differ; class scores are
            misshapen or not finite, or class decisions are not integer indices of declared classes, one per item; a
            replica key is unknown; or a derived view lacks ``groups``. For object detection, it is also raised if
            ``max_objects`` gives no bound for a requirement's class, an input or a detector's output is invalid, an
            input is classification only, neither ``predictions`` nor ``model`` is given while a requirement's
            ``max_detections`` is at or above its ``max_objects``, or the data exceed ``max_objects``. Each error
            names the fix.
        ShapeMismatchError
            If the embeddings do not have one row per dataset item, or a ``groups`` array is not aligned with the
            dataset.
        """
        reject_filtered_metadata(dataset, "Feasibility")
        # The evaluator warned on construction; its experimental parts stay quiet without spending their warnings.
        if _supported_task(dataset) == "OD":
            with quiet_experimental():
                return self._evaluate_detection(
                    dataset,
                    target=target,
                    predictions=predictions,
                    model=model,
                    embeddings=embeddings,
                    groups=groups,
                    replicas=replicas,
                    allowances=allowances,
                )
        predictions = cast("ArrayLike | None", predictions)  # classification takes arrays only
        with quiet_experimental():
            units = resolve_units(dataset, groups=groups)
            if units.n_classes < 2:
                raise ValueError(f"Feasibility needs at least 2 classes, the dataset declares {units.n_classes}.")
            specs = parse_targets(target, task="IC", class_names=units.class_names, grouped=units.grouped)
            _check_class_names(specs, units)
            # Cheap argument checks first, before the embedding pass.
            _check_allowances(allowances)
            _check_sources(model, predictions, replicas, embeddings, self.extractor)
            class_order_verified = model is not None and check_class_order(model, dataset)
            _check_predictions(predictions, units)
            by_position = resolve_replicas(replicas, dataset, units) if replicas else {}
            seed = self.resolved_seed
            fit, audit = split_units(units, fit_fraction=FIT_FRACTION, seed=seed)
            if model is not None:
                predictions = _model_scores(model, dataset, units, self._model_batch_size())
                _check_predictions(predictions, units)
        features = self._features(dataset, embeddings)  # the user's extractor warns as it would
        if features is not None:
            features = features[units.item_indices]
        with quiet_experimental():
            n_claims = count_claims(
                specs,
                has_replicas=bool(by_position),
                has_predictions=predictions is not None,
                builtin_witnesses=True,
                has_embeddings=features is not None,
            )
            bounds = gather_evidence(
                specs,
                units=units,
                features=features,
                audit=audit,
                fit=fit,
                replicas=by_position,
                predictions=predictions,
                allowances=allowances,
                alpha_claim=allocate(self.alpha, max(n_claims, 1)),  # no source at all: no bound spends anything
                seed=seed,
                max_sequence_length=self.max_sequence_length,
                class_order_verified=class_order_verified,
            )
            check_ledger(bounds, self.alpha)
            determinations = [decide(spec, i, bounds) for i, spec in enumerate(specs)]
            return FeasibilityOutput(
                tuple(specs),
                allowances,
                _determinations_frame(specs, determinations),
                _bounds_frame(bounds, determinations),
            )

    def _evaluate_detection(
        self,
        dataset: AnnotatedDataset[Any],
        *,
        target: Sequence[TargetSpec],
        predictions: ArrayLike | Sequence[ObjectDetectionTarget] | None,
        model: Model | None,
        embeddings: ArrayLike | None,
        groups: ArrayLike | str | None,
        replicas: Mapping[Any, ArrayLike] | None,
        allowances: Allowances | None,
    ) -> FeasibilityOutput:
        """Judge object detection requirements by capacity and a detector's witness; checks that need no pass first."""
        specs = parse_targets(target, task="OD", class_names=_declared_class_names(dataset), grouped=False)
        max_objects = resolve_count_bounds(self.max_objects, specs)
        check_detection_call(
            model=model,
            predictions=predictions,
            embeddings=embeddings,
            replicas=replicas,
            allowances=allowances,
            specs=specs,
            max_objects=max_objects,
        )
        # Neither reads the dataset's items, so both refuse before the targets' pass.
        batch_size = self._model_batch_size() if model is not None else 0
        class_order_verified = model is not None and check_class_order(model, dataset)
        class_ids = resolve_class_ids(dataset, specs)
        if predictions is not None:
            check_detection_layout(predictions, len(dataset))
        units = resolve_detection_units(dataset, groups=groups)
        if units.grouped:
            raise ValueError(
                "Object detection images must be independent units, but some unit holds more than one image: groups= "
                "pairs images, or a View repeats one. Grouping images, or a video's frames, is not supported in this "
                "release; pass images that are each their own unit."
            )
        counts = check_counts(units, class_ids, max_objects, specs)
        # One claim per requirement whose K is below its B, the capacity envelope, and one per requirement for a
        # detector's witness. check_detection_call refused a call without a detector in which any requirement lacks
        # a capacity claim.
        has_detector = predictions is not None or model is not None
        n_capacity = sum(cast(int, s.max_detections) < b for s, b in zip(specs, max_objects, strict=True))
        alpha_claim = allocate(self.alpha, n_capacity + (len(specs) if has_detector else 0))
        bounds = capacity_evidence(specs, counts=counts, alpha_claim=alpha_claim, max_objects=max_objects)
        if has_detector:
            n_classes = units.n_classes
            detections = (
                run_model(model, dataset, batch_size, lambda i, output: read_model_output(i, output, n_classes))
                if model is not None
                else check_detections(cast(Sequence[Any], predictions), len(dataset), n_classes)
            )
            fit, audit = split_units(units, fit_fraction=FIT_FRACTION, seed=self.resolved_seed)
            bounds += gather_detection_evidence(
                specs,
                units=units,
                detections=detections,
                fit=fit,
                audit=audit,
                alpha_claim=alpha_claim,
                class_ids=class_ids,
                max_objects=max_objects,
                class_order_verified=class_order_verified,
            )
        bounds = sorted(bounds, key=table_order)
        check_ledger(bounds, self.alpha)
        determinations = [decide(spec, i, bounds) for i, spec in enumerate(specs)]
        return FeasibilityOutput(
            tuple(specs),
            None,
            _determinations_frame(specs, determinations),
            _bounds_frame(bounds, determinations),
        )
