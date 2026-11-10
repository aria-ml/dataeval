"""Requirements and allowances for the feasibility evaluator: typed specs and their checks."""

__all__ = []

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, get_args

from dataeval._experimental import experimental

Task = Literal["IC", "OD", "MOT"]
Metric = Literal["accuracy", "ap", "recall", "recall_at_fp", "mota"]
APFunctional = Literal["population_ap_integral", "population_ap_101"]
Scope = Literal["full_input", "fixed_representation", "partition_restricted"]
Aggregation = Literal["frame_pooled", "sequence_mean"]
FalsePositiveScope = Literal["pooled", "each_class"]

_TASK_METRICS: dict[str, tuple[str, ...]] = {
    "IC": ("accuracy", "ap", "recall"),
    "OD": ("ap", "recall_at_fp"),
    "MOT": ("mota",),
}


def _num(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _allowance(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite, non-negative number, got {value!r}")
    return float(value)


@experimental
@dataclass(frozen=True)
class Allowances:
    """
    The allowances a caller declares for a feasibility call, which widen what local-panel accuracy envelopes cover.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    ``beta`` bounds how much the cells of the internal partition differ from one another, and ``gamma`` bounds the
    information the representation loses. Each is a scalar and each must be a justified bound, recorded as assumed
    and not verified: ``beta=0`` asserts that no model reading the representation beats one fixed answer per cell.
    A per-cell ``beta`` refers to the cells of the internal partition, which are not exposed, so it is not supported
    in this release.

    Attributes
    ----------
    beta : float
        Cell heterogeneity, finite and non-negative.
    gamma : float or None, default None
        Information loss, finite and non-negative. It applies on top of ``beta``.
    """

    beta: float
    gamma: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "beta", _allowance(self.beta, "beta"))
        if self.gamma is not None:
            object.__setattr__(self, "gamma", _allowance(self.gamma, "gamma"))


@experimental
@dataclass(frozen=True)
class TargetSpec:
    """
    One feasibility target: the metric, every detail that changes its meaning, and the threshold to reach.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    ``task`` defaults to ``"IC"``, the only supported task. ``class_name=None`` means macro for AP.
    ``ap_functional`` is set for AP only and defaults to ``"population_ap_integral"`` for classification, the only AP
    functional bounded for classification. ``fp_budget`` is set for ``recall_at_fp`` only, ``iou`` for OD and MOT
    only, ``aggregation`` for grouped classification only.

    ``recall`` (classification) is a minimum recall for ``class_name`` at a maximum false-positive rate
    ``max_false_positive_rate`` in (0, 1); it takes neither ``aggregation`` nor ``iou``. The negatives are
    ``false_positive_scope="pooled"``, the mixture of every other declared class (filled in when left ``None``);
    ``"each_class"`` is not supported in this release.

    Examples
    --------
    >>> TargetSpec(metric="accuracy", threshold=0.9).task
    'IC'
    """

    metric: Metric
    threshold: float
    task: Task = "IC"
    class_name: str | None = None
    iou: float | None = None
    ap_functional: APFunctional | None = None
    fp_budget: float | None = None
    aggregation: Aggregation | None = None
    scope: Scope = "full_input"
    max_false_positive_rate: float | None = None
    false_positive_scope: FalsePositiveScope | None = None

    def _check_enums(self) -> None:
        for name, valid in (
            ("task", get_args(Task)),
            ("metric", get_args(Metric)),
            ("scope", get_args(Scope)),
            ("ap_functional", (*get_args(APFunctional), None)),
            ("aggregation", (*get_args(Aggregation), None)),
            ("false_positive_scope", (*get_args(FalsePositiveScope), None)),
        ):
            if getattr(self, name) not in valid:
                raise ValueError(f"Unknown {name} {getattr(self, name)!r}; valid: {list(valid)}")

    def __post_init__(self) -> None:
        self._check_enums()
        if self.metric not in _TASK_METRICS[self.task]:
            raise ValueError(
                f"metric {self.metric!r} is not offered for task {self.task!r}: {_TASK_METRICS[self.task]}"
            )
        if self.metric == "ap" and self.task == "IC" and self.ap_functional is None:
            object.__setattr__(self, "ap_functional", "population_ap_integral")
        # Normalize numerics so equal targets serialize identically (1 vs 1.0).
        for name in ("threshold", "iou", "fp_budget", "max_false_positive_rate"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _num(getattr(self, name), name))
        self._check_threshold()
        self._check_iou()
        self._check_metric_fields()
        self._check_aggregation()

    def _check_threshold(self) -> None:
        if self.metric == "mota":
            if self.threshold > 1:
                raise ValueError(f"threshold for 'mota' must be in (-inf, 1], got {self.threshold}")
        elif not 0 < self.threshold <= 1:
            raise ValueError(f"threshold for {self.metric!r} must be in (0, 1], got {self.threshold}")

    def _check_iou(self) -> None:
        if self.task == "IC":
            if self.iou is not None or self.fp_budget is not None:
                raise ValueError("IC targets take neither iou nor fp_budget")
        elif self.iou is None or not 0 < self.iou <= 1:
            raise ValueError(f"{self.task} targets require iou in (0, 1], got {self.iou!r}")

    def _check_ap_fields(self) -> None:
        if self.metric == "ap" and self.ap_functional is None:
            raise ValueError(f"'ap' targets require ap_functional, one of {list(get_args(APFunctional))}")
        if self.metric != "ap" and self.ap_functional is not None:
            raise ValueError(f"ap_functional applies to 'ap' only, not {self.metric!r}")
        if self.metric not in ("ap", "recall") and self.class_name is not None:
            raise ValueError(f"class_name applies to 'ap' and 'recall' only, not {self.metric!r}")

    def _check_recall_fields(self) -> None:
        if self.metric != "recall":
            if self.max_false_positive_rate is not None or self.false_positive_scope is not None:
                raise ValueError(
                    f"max_false_positive_rate and false_positive_scope apply to 'recall' only, not {self.metric!r}"
                )
            return
        if self.class_name is None or self.aggregation is not None:
            raise ValueError("'recall' targets require class_name and take no aggregation")
        if self.max_false_positive_rate is None or not 0 < self.max_false_positive_rate < 1:
            raise ValueError(
                f"'recall' requires max_false_positive_rate in (0, 1), got {self.max_false_positive_rate!r}"
            )
        self._check_false_positive_scope()

    def _check_false_positive_scope(self) -> None:
        if self.false_positive_scope is None:
            object.__setattr__(self, "false_positive_scope", "pooled")
        elif self.false_positive_scope != "pooled":
            raise ValueError(
                f"false_positive_scope {self.false_positive_scope!r} is not supported in this release; use 'pooled'"
            )

    def _check_metric_fields(self) -> None:
        self._check_ap_fields()
        self._check_recall_fields()
        if self.metric == "recall_at_fp":
            if self.fp_budget is None or self.fp_budget <= 0:
                raise ValueError(f"'recall_at_fp' requires fp_budget > 0, got {self.fp_budget!r}")
        elif self.fp_budget is not None:
            raise ValueError(f"fp_budget applies to 'recall_at_fp' only, not {self.metric!r}")

    def _check_aggregation(self) -> None:
        if self.aggregation is not None and self.task != "IC":
            raise ValueError(f"aggregation applies to classification targets only, not {self.task!r}")
        if self.aggregation == "sequence_mean" and self.metric != "accuracy":
            raise ValueError(f"aggregation 'sequence_mean' is defined for 'accuracy' only, not {self.metric!r}")


def _check_classes(s: TargetSpec, class_names: Sequence[str] | None) -> None:
    if s.metric == "ap" and s.class_name is None and not class_names:
        raise ValueError("Macro 'ap' needs the class list; declare index2label on the dataset metadata")
    if s.class_name is not None and class_names and s.class_name not in class_names:
        raise ValueError(f"Unknown class {s.class_name!r} in target; declared classes: {list(class_names)}")


def _check_context(s: TargetSpec, task: str, class_names: Sequence[str] | None, grouped: bool) -> None:
    if s.task != task:
        raise ValueError(f"A {s.metric!r} target is a {s.task} target but the data is {task}")
    if task == "IC" and s.metric == "ap" and s.ap_functional != "population_ap_integral":
        raise ValueError(
            "Classification AP is bounded as ap_functional 'population_ap_integral' only in this release, "
            f"got {s.ap_functional!r}."
        )
    _check_classes(s, class_names)
    _check_grouping(s, task, grouped)


def _check_grouping(s: TargetSpec, task: str, grouped: bool) -> None:
    if s.metric == "recall" and grouped:
        raise ValueError("recall targets are not supported on grouped data in this release")
    if task == "IC" and grouped and s.aggregation is None:
        raise ValueError(
            f"Grouped data requires an aggregation for {s.metric!r}: set aggregation='frame_pooled' or 'sequence_mean'"
        )
    if not grouped and s.aggregation is not None:
        raise ValueError(f"aggregation {s.aggregation!r} needs grouped data (a group of several items); pass groups=")


def parse_targets(
    target: Sequence[TargetSpec],
    *,
    task: Task,
    class_names: Sequence[str] | None,
    grouped: bool,
) -> list[TargetSpec]:
    """
    Check a sequence of ``TargetSpec`` against the data and return it as a list, in the order given.

    ``class_names`` come from a declared ``index2label``; ``grouped`` is whether some group holds more than one item.
    Classification AP must be ``population_ap_integral``, the only AP functional this release bounds. Equal targets
    are duplicates.
    """
    if (
        isinstance(target, str)
        or not isinstance(target, Sequence)
        or not all(isinstance(s, TargetSpec) for s in target)
    ):
        raise ValueError(
            "target must be a sequence of TargetSpec, e.g. [TargetSpec(metric='accuracy', threshold=0.9)], "
            f"got {type(target).__name__}"
        )
    specs = list(target)
    if not specs:
        raise ValueError("target must name at least one metric")
    for s in specs:
        _check_context(s, task, class_names, grouped)
    if len(set(specs)) != len(specs):
        raise ValueError("target contains duplicate targets")
    return specs
