"""Requirements and allowances for the feasibility evaluator: typed specs and their checks."""

__all__ = []

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Literal, get_args

from dataeval._experimental import experimental
from dataeval.types._task import DatasetTask

Task = DatasetTask
Metric = Literal["accuracy", "ap", "recall", "recall_at_fp", "mota"]
APFunctional = Literal["population_ap_integral", "population_ap_101"]
Scope = Literal["full_input", "fixed_representation", "partition_restricted"]
Aggregation = Literal["frame_pooled", "sequence_mean"]
FalsePositiveScope = Literal["pooled", "each_class"]

_OD_CALL = "TargetSpec(task='OD', metric='ap', class_name=..., threshold=..., iou=..., max_detections=...)"
_OD_HINT = "for an object detection target, set task='OD' with iou=... and max_detections=..."
_AP_BOUNDED = "AP is bounded as ap_functional 'population_ap_integral' only in this release"

_TASK_METRICS: dict[str, tuple[str, ...]] = {
    "IC": ("accuracy", "ap", "recall"),
    "OD": ("ap", "recall_at_fp"),
    "MOT": ("mota",),
}


def _num(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def positive_count(value: object, name: str) -> int:
    """Return ``value`` as an ``int`` when it is a Python or NumPy integer of at least 1, never a bool."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer of at least 1, got {value!r}")
    return int(value)


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

    ``task`` defaults to ``"IC"``. ``"OD"`` (object detection) is supported for ``ap`` on one named class, and
    ``"MOT"`` is not supported in this release. ``class_name=None`` means macro for AP, for classification only.
    ``ap_functional`` is set for AP only and defaults to ``"population_ap_integral"`` for every AP target, the only AP
    functional bounded in this release. ``fp_budget`` is set for ``recall_at_fp`` only, ``iou`` for OD and MOT only,
    ``aggregation`` for grouped classification only. ``max_detections`` (K) is required for OD and refused for the
    other tasks: the most detections per image, for the target's class, that the AP counts.

    An OD ``ap`` target is one class's AP at one IoU, with at most K detections per image matched greedily by score,
    integrated over recall. That is not COCO's AP, which averages over classes and IoU 0.5 to 0.95 and interpolates at
    101 points. A K below the number of the class's targets an image holds caps the true positives that image can show,
    and so caps recall. A larger K also widens the confidence endpoints, so whether raising K raises or lowers the
    witness bound depends on the detector and the data. A K below the class's ``Feasibility(max_objects=...)`` lets the
    capacity envelope bound the AP of any detector with that cap.

    ``recall`` (classification) is a minimum recall for ``class_name`` at a maximum false-positive rate
    ``max_false_positive_rate`` in (0, 1); it takes neither ``aggregation`` nor ``iou``. The negatives are
    ``false_positive_scope="pooled"``, the mixture of every other declared class (filled in when left ``None``);
    ``"each_class"`` is not supported in this release.

    Examples
    --------
    >>> TargetSpec(metric="accuracy", threshold=0.9).task
    'IC'

    An object detection AP target needs no ``ap_functional``:

    >>> person = TargetSpec(task="OD", metric="ap", class_name="person", threshold=0.5, iou=0.5, max_detections=100)
    >>> person.ap_functional
    'population_ap_integral'
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
    max_detections: int | None = None

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
        if self.metric == "ap" and self.ap_functional is None:
            object.__setattr__(self, "ap_functional", "population_ap_integral")
        # Normalize numerics so equal targets serialize identically (1 vs 1.0).
        for name in ("threshold", "iou", "fp_budget", "max_false_positive_rate"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _num(getattr(self, name), name))
        self._check_threshold()
        self._check_iou()
        self._check_metric_fields()
        self._check_aggregation()
        self._check_max_detections()

    def _check_threshold(self) -> None:
        if self.metric == "mota":
            if self.threshold > 1:
                raise ValueError(f"threshold for 'mota' must be in (-inf, 1], got {self.threshold}")
        elif not 0 < self.threshold <= 1:
            raise ValueError(f"threshold for {self.metric!r} must be in (0, 1], got {self.threshold}")

    def _check_iou(self) -> None:
        if self.task == "IC":
            if self.iou is not None or self.fp_budget is not None:
                raise ValueError(f"IC targets take neither iou nor fp_budget; {_OD_HINT}")
        elif self.iou is None or not 0 < self.iou <= 1:
            raise ValueError(f"{self.task} targets require iou in (0, 1], got {self.iou!r}")

    def _check_max_detections(self) -> None:
        if self.task != "OD":
            if self.max_detections is not None:
                raise ValueError(f"max_detections applies to object detection only, not task {self.task!r}; {_OD_HINT}")
        elif self.max_detections is None:
            raise ValueError(
                "OD targets require max_detections, the most detections per image that the AP counts, e.g. 100"
            )
        else:
            object.__setattr__(self, "max_detections", positive_count(self.max_detections, "max_detections"))

    def _check_ap_fields(self) -> None:
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


def _check_od_support(s: TargetSpec) -> None:
    if s.metric == "recall_at_fp":
        raise ValueError(f"'recall_at_fp' is not supported for object detection in this release; use {_OD_CALL}")
    if s.class_name is None:
        raise ValueError(
            f"Macro 'ap' is not supported for object detection in this release; name one class: {_OD_CALL}"
        )
    if s.ap_functional != "population_ap_integral":
        raise ValueError(f"{_AP_BOUNDED}, got {s.ap_functional!r}; use {_OD_CALL}")


def _check_classes(s: TargetSpec, class_names: Sequence[str] | None) -> None:
    if s.metric == "ap" and s.class_name is None and not class_names:
        raise ValueError("Macro 'ap' needs the class list; declare index2label on the dataset metadata")
    if s.class_name is not None and class_names and s.class_name not in class_names:
        raise ValueError(f"Unknown class {s.class_name!r} in target; declared classes: {list(class_names)}")


def _check_context(s: TargetSpec, task: str, class_names: Sequence[str] | None, grouped: bool) -> None:
    if s.task != task:
        hint = f"; use {_OD_CALL}" if task == "OD" else ""
        raise ValueError(f"A {s.metric!r} target is a {s.task} target but the data is {task}{hint}")
    if task == "OD":
        _check_od_support(s)
    elif task == "IC" and s.metric == "ap" and s.ap_functional != "population_ap_integral":
        raise ValueError(f"{_AP_BOUNDED}, got {s.ap_functional!r}.")
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
    AP must be ``population_ap_integral``, the only AP functional this release bounds, for classification and object
    detection. Object detection also refuses macro AP and ``recall_at_fp``. Equal targets are duplicates.
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
