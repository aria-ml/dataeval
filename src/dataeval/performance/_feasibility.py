__all__ = []

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, get_args

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import PositiveInt

from dataeval._experimental import experimental, quiet_experimental
from dataeval._helpers import reject_filtered_metadata
from dataeval.config import get_batch_size, get_seed
from dataeval.exceptions import ShapeMismatchError
from dataeval.performance._feasibility_classification import (
    _class_index,
    check_predictions,
    count_claims,
    gather_evidence,
)
from dataeval.performance._feasibility_constants import FIT_FRACTION
from dataeval.performance._feasibility_decide import Decision, Determination, Limitation, decide
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
from dataeval.performance._feasibility_model import check_class_order, model_scores
from dataeval.performance._feasibility_target import (
    Aggregation,
    Allowances,
    APFunctional,
    Metric,
    Scope,
    TargetSpec,
    Task,
    parse_targets,
)
from dataeval.performance._feasibility_units import Units, resolve_replicas, resolve_units, split_units
from dataeval.protocols import AnnotatedDataset, ArrayLike, FeatureExtractor, Model
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
_STATE = ["alpha", "max_sequence_length", "resolved_seed"]
_PRIVATE = ("mean", "support", "delta")  # on a ``Bound``, not columns of the ``bounds`` table


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
    if spec.aggregation is not None:
        text += f" |{spec.aggregation}"
    return text if spec.scope == "full_input" else f"{text} @{spec.scope}"


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
        - task, metric, threshold, judged_scope: Enum, Enum, float, Enum - What was asked: ``IC``, the metric, its
          threshold and the scope it is judged at
        - class_name: str - The class the requirement names, null otherwise
        - aggregation, ap_functional, max_false_positive_rate: Enum, Enum, float - The details that tell requirements
          of one metric apart, null where they do not apply
        - determination: Enum - ``demonstrated_feasible``, ``infeasible_under_recorded_assumptions``,
          ``not_ruled_out`` or ``unresolved``
        - limitation: Enum - What limits a non-final determination, null for a final one: ``conflict``, ``no_envelope``,
          ``scope_restricted``, ``missing_bound``, ``envelope_unavailable``, ``no_eligible_witness``,
          ``within_radius`` or ``weak_witness``
        - suggested_minimum: int - Independent units at which the bound named by ``minimum_source`` and
          ``minimum_quantity`` would cross the requirement if the observed rates held: a projection, not a bound.
          Null when not attainable or when a final determination needs none
        - minimum_source, minimum_quantity: Enum - The ``bounds`` row, with ``requirement``, that the minimum projects
          from
        - envelope, envelope_scope: float, Enum - The determining envelope's upper bound and its information scope
        - witness, witness_source: float, Enum - The determining witness's lower bound and its source; for a conflict,
          the conflicting pair sits in these four columns
    bounds : pl.DataFrame
        One row per bound, ordered by ``requirement`` and then source, with columns:

        - requirement: int - Index into ``requirements``
        - source: Enum - ``local_panels``, ``replicas``, ``knn``, ``linear_probe`` or ``user_model``
        - role: Enum - ``envelope`` or ``witness``
        - quantity: Enum - The metric, or ``false_positive_rate`` for a recall witness's second bound
        - bound_type: Enum - ``lower`` or ``upper``
        - information_scope, population, sample, unit: Enum - What the bound covers: ``full_input``,
          ``fixed_representation`` or ``partition_restricted``; ``items`` or ``labeled_items``; the sample it is
          computed on; ``image`` or ``sequence``
        - state: Enum - ``unavailable``, ``uninformative`` or ``informative``
        - reason: Enum - Why a bound is unavailable or uninformative, null otherwise
        - bound: float - The confidence bound; null when unavailable
        - observed: float - A witness's point estimate of the quantity; null for every envelope and where the source
          records no estimate (such as AP)
        - n, variance: int, float - Independent units behind the bound and their variance; null where the method does
          not define them
        - radius: float - The half-width the named method applied to the estimate to produce ``bound``; null for exact
          intervals, AP and recall envelopes and ``frame_pooled`` bounds
        - method: Enum - ``hoeffding``, ``hybrid`` or ``exact_binomial``; null where the bound has no closed form
        - alpha: float - This bound's share of the call's error budget
        - assumptions: list[Enum] - What the bound rests on. Every member is declared and not verified
        - is_determining: bool - Whether the requirement's determination rests on this bound

    The call's ``alpha`` and resolved seed are in ``meta().state``.
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
        sources, and the suggested minimum. Lower bounds round down and upper bounds up.
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
        return "\n".join([
            *lines,
            f"joint confidence: {confidence}",
            f"assumes: {assumptions}",
            "minimum: independent units to cross the requirement if the observed rates hold",
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
    Decide whether a classification requirement is reachable on a dataset, with a proof behind every claim.

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
    ``bounds`` table shows both of a witness's bounds.

    Sources: local-panel envelopes, DataEval's post-stratified pooling (C3) of the DFE's disjoint label panels [4]_
    within the cells of a label-blind partition fitted on a group-disjoint fit split, proved in
    :func:`~dataeval.core.local_panel_bound`'s Notes with two Hoeffding [1]_ steps; replica envelopes from
    ``replicas=``; built-in frozen k-NN and linear-probe witnesses fitted on the fit split; and a user witness from
    ``predictions=``, which counts at ``full_input`` scope. Each (requirement, source) pair gets an equal share of
    ``alpha``. Bounds on means use Hoeffding [1]_ and empirical Bernstein [2]_ radii, except that ungrouped witness
    bounds (accuracy, recall, false-positive rate and AP indicators) use exact Clopper-Pearson [5]_ intervals; AP and
    recall envelopes use envelopes built on the cost-weighted risk inequality of [3]_.

    Parameters
    ----------
    extractor : FeatureExtractor or None, default None
        Feature extractor that computes the representation; optional when ``embeddings`` are given to
        :meth:`evaluate`. It must have been fitted and selected independently of the audit items' labels.
    batch_size : int or None, default None
        Batch size for embedding computation and for scoring a ``model``; a ``model`` also needs this or a global
        batch size from :func:`dataeval.config.set_batch_size`.
    alpha : float, default 0.05
        Error budget of one call, shared by every bound of every requirement.
    seed : int or None, default None
        Seed of the splits, partition and panels; None uses :func:`dataeval.config.get_seed`, else 0.
    max_sequence_length : int or None, default None
        Declared largest group size for grouped (video) data, needed by ``frame_pooled`` requirements. Data that
        violate it raise rather than being truncated.
    config : Feasibility.Config or None, default None
        Optional configuration object; parameters passed directly to ``__init__`` override it.

    Notes
    -----
    **One call, one audit.** The guarantee covers one call on one fixed audit. Comparing ``k`` representations or
    models across calls, keeping the best of ``k`` seeds, or growing the audit and looking again, needs ``alpha / k``
    per call.

    **Witness only.** A call with a witness (``predictions`` or ``model``) but neither embeddings nor replicas has no
    envelope of any scope. Its determination is ``demonstrated_feasible``, or ``unresolved`` with limitation
    ``no_envelope``. It can never be ``infeasible_under_recorded_assumptions`` or ``not_ruled_out``, because only an
    envelope can bound what is unreachable.

    **Scope.** An envelope decides at its information scope and every narrower one (``full_input`` ⊃
    ``fixed_representation`` ⊃ ``partition_restricted``). Image local-panel accuracy envelopes are
    ``partition_restricted`` without allowances, ``fixed_representation`` with β, and ``full_input`` with β and
    Γ; AP, recall and video local-panel envelopes stay ``partition_restricted``; replica envelopes are ``full_input``.
    Built-in witnesses read the representation, so they are eligible at ``fixed_representation`` and broader.

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
    minimum: independent units to cross the requirement if the observed rates hold

    The minimal call is a dataset, predictions (or a ``model``) and a requirement. With no embeddings and no replicas
    there is no envelope, so a requirement that the witness cannot demonstrate stays unresolved:

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
        seed : int or None, default None
            Seed of the splits, partition and panels.
        """

        extractor: FeatureExtractor | None = None
        batch_size: PositiveInt | None = None
        alpha: OpenUnitInterval = 0.05
        max_sequence_length: PositiveInt | None = None
        seed: int | None = None

    extractor: FeatureExtractor | None
    batch_size: int | None
    alpha: float
    max_sequence_length: int | None
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
        predictions: ArrayLike | None = None,
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
            An image or video classification dataset. Unlabeled items are excluded from the population.
        target : Sequence[TargetSpec]
            The requirements, in the order wanted back. The judged scope of each defaults to ``full_input``.
        predictions : ArrayLike or None, default None
            A frozen model's output on every dataset item, used as a user witness: class scores of shape
            ``(len(dataset), n_classes)``, whose top class is the decision, or a 1-D integer array of class
            decisions, indices of the dataset's declared classes (booleans and floats are rejected). Apply any other
            decision policy, such as a threshold on one class's score, before the call and pass its decisions; one
            call's decisions feed every requirement, so a different operating point needs its own call, with the
            error budget split across the planned calls. Class decisions rank nothing, so they give no AP witness. The
            model is assumed frozen before the audit and independent of the audit items, and counts at ``full_input``.
            Give ``predictions`` or ``model``, not both.
        model : Model or None, default None
            A frozen MAITE model, run over the whole dataset in batches of ``batch_size`` (or the global batch size) in
            place of ``predictions``.
            Its class scores keep the model's own precision and feed the same witness, so a model gives the same
            determinations and bounds as its own scores passed as ``predictions``, and it can witness AP, which class
            decisions cannot. If the model's ``metadata`` and the dataset both declare an ``index2label`` they must be
            equal, else ``ValueError``; a class order is never remapped. If either declares none, the dataset's class
            order is assumed and recorded as ``class_order_matches_dataset``.
        embeddings : ArrayLike or None, default None
            One representation, shape ``(len(dataset), d)``; computed with the extractor when None. With neither
            embeddings nor an extractor the call has no local-panel envelope and no built-in witness.
        groups : ArrayLike or str or None, default None
            The independent unit of each item (array aligned with the dataset, or a metadata factor name).
            Required for derived views, except a ``View`` straight over a plain dataset, which defaults to its
            source indices. When some group holds more than one item, requirements need an aggregation.
        replicas : Mapping[int or str, ArrayLike] or None, default None
            Extra independent class labels by datum id, for full_input replica envelopes. The covered items
            are recorded as a random sample of the population.
        allowances : Allowances or None, default None
            ``beta`` (cell heterogeneity) and ``gamma`` (information loss), recorded as assumed. They widen the scope
            of image local-panel accuracy envelopes.

        Returns
        -------
        FeasibilityOutput
            The ``determinations`` table and the ``bounds`` behind it.

        Raises
        ------
        NotImplementedError
            If the dataset is object detection or tracking (planned).
        ValueError
            If the dataset has fewer than 2 classes; a requirement, a class name or the allowances is invalid; both
            ``model`` and ``predictions`` are given; ``model`` is given with no ``batch_size`` and no global batch
            size; none of ``predictions``, ``model``, ``replicas``, ``embeddings``
            or an extractor is given; the model's and the dataset's ``index2label`` differ; class scores are
            misshapen or not finite, or class decisions are not integer indices of declared classes, one per item; a
            replica key is unknown; or a derived view lacks ``groups``.
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
