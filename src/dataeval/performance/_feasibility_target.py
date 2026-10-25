"""Target specification for the feasibility evaluator: parsing and canonical (versioned JSON) serialization."""

__all__ = []

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Literal, get_args

from dataeval._experimental import experimental
from dataeval.performance._feasibility_constants import SPEC_VERSION

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
_GRAMMAR = "metric[:class][@iou][/scope][|aggregation]"


def _num(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


@experimental
@dataclass(frozen=True)
class TargetSpec:
    """
    One feasibility target: the metric, every detail that changes its meaning, and the threshold to reach.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    ``class_name=None`` means macro for AP. ``ap_functional`` is set for AP only, ``fp_budget`` for
    ``recall_at_fp`` only, ``iou`` for OD and MOT only, ``aggregation`` for grouped classification only.

    ``recall`` (classification) is a minimum recall for ``class_name`` at a maximum false-positive rate
    ``max_false_positive_rate`` in (0, 1); it takes neither ``aggregation`` nor ``iou``. The negatives are
    ``false_positive_scope="pooled"``, the mixture of every other declared class (filled in when left ``None``);
    ``"each_class"`` is not supported in this release.
    """

    task: Task
    metric: Metric
    threshold: float
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

    def canonical(self) -> str:
        """Sorted-key compact JSON with the spec version; nulls are kept."""
        return json.dumps({**asdict(self), "version": SPEC_VERSION}, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_canonical(cls, text: str) -> "TargetSpec":
        """Rebuild a spec from :meth:`canonical` output."""
        data = json.loads(text)
        version = data.pop("version", None)
        if version != SPEC_VERSION:
            raise ValueError(f"Unsupported target spec version {version!r}; this build reads version {SPEC_VERSION}")
        if set(data) != {f.name for f in fields(cls)}:
            raise ValueError(f"Target spec fields {sorted(data)} do not match {sorted(f.name for f in fields(cls))}")
        return cls(**data)


_KEY = re.compile(
    r"(?P<metric>[^:@/|]*)(?::(?P<cls>[^@/|]+))?(?:@(?P<iou>[^/|]*))?(?:/(?P<scope>[^|]*))?(?:\|(?P<agg>.*))?",
    re.DOTALL,
)


def _parse_key(key: str, threshold: object, task: str) -> TargetSpec:
    m = _KEY.fullmatch(key)
    try:
        if m is None or m["metric"] not in get_args(Metric):
            raise ValueError
        iou = None if m["iou"] is None else float(m["iou"])
        if m["scope"] not in (None, *get_args(Scope)) or m["agg"] not in (None, *get_args(Aggregation)):
            raise ValueError
    except ValueError:
        raise ValueError(
            f"Invalid target key {key!r}; expected {_GRAMMAR} with metric in {list(get_args(Metric))}, "
            f"scope in {list(get_args(Scope))}, aggregation in {list(get_args(Aggregation))}; "
            "e.g. 'ap:cat', 'ap|frame_pooled', 'mota@0.5', 'accuracy/fixed_representation'. "
            "Pass TargetSpec objects for fp_budget."
        ) from None
    if m["metric"] == "recall":
        raise ValueError("recall targets need a false-positive limit; pass a TargetSpec")
    return TargetSpec(
        task=task,  # type: ignore[arg-type]
        metric=m["metric"],  # type: ignore[arg-type]
        threshold=_num(threshold, f"threshold for {key!r}"),
        class_name=m["cls"],
        iou=iou,
        ap_functional="population_ap_integral" if m["metric"] == "ap" else None,
        aggregation=m["agg"],  # type: ignore[arg-type]
        scope=m["scope"] or "full_input",  # type: ignore[arg-type]
    )


def _check_classes(s: TargetSpec, class_names: Sequence[str] | None) -> None:
    if s.metric == "ap" and s.class_name is None and not class_names:
        raise ValueError("Macro 'ap' needs the class list; declare index2label on the dataset metadata")
    if s.class_name is not None and class_names and s.class_name not in class_names:
        raise ValueError(f"Unknown class {s.class_name!r} in target; declared classes: {list(class_names)}")


def _check_context(s: TargetSpec, task: str, class_names: Sequence[str] | None, grouped: bool) -> None:
    if s.task != task:
        raise ValueError(f"Target {s.canonical()} is a {s.task} target but the data is {task}")
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
        raise ValueError(f"Grouped data requires an aggregation for {s.metric!r}: '|frame_pooled' or '|sequence_mean'")
    if not grouped and s.aggregation is not None:
        raise ValueError(f"aggregation {s.aggregation!r} needs grouped data (a group of several items); pass groups=")


def _as_specs(target: Mapping[str, float] | Sequence[TargetSpec], task: str) -> list[TargetSpec]:
    if isinstance(target, Mapping):
        return [_parse_key(k, v, task) for k, v in target.items()]
    specs = list(target)
    if not all(isinstance(s, TargetSpec) for s in specs):
        raise ValueError("target must be a mapping of key to threshold or a sequence of TargetSpec")
    return specs


def parse_targets(
    target: Mapping[str, float] | Sequence[TargetSpec],
    *,
    task: Task,
    class_names: Sequence[str] | None,
    grouped: bool,
) -> list[TargetSpec]:
    """
    Parse a target mapping (``"metric[:class][@iou][/scope][|aggregation]": threshold``) or ``TargetSpec`` list.

    Class names containing ``| / @`` must be passed as ``TargetSpec`` objects. ``class_names`` come from a
    declared ``index2label``; ``grouped`` is whether some group holds more than one item. Classification AP must be
    ``population_ap_integral``, the only AP functional this release bounds.
    """
    specs = _as_specs(target, task)
    if not specs:
        raise ValueError("target must name at least one metric")
    for s in specs:
        _check_context(s, task, class_names, grouped)
    if len({s.canonical() for s in specs}) != len(specs):
        raise ValueError("target contains duplicate targets")
    return specs
