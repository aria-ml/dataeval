import re
from typing import Any

import numpy as np
import pytest

from dataeval.performance._feasibility_constants import n_cells_for
from dataeval.performance._feasibility_target import TargetSpec, Task, parse_targets

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

CLASSES = ["cat", "dog"]


def _parse(target, task: Task = "IC", class_names=CLASSES, grouped=False):
    return parse_targets(target, task=task, class_names=class_names, grouped=grouped)


def _spec(metric: Any = "accuracy", threshold: float = 0.9, **kwargs: Any) -> TargetSpec:
    return TargetSpec(metric=metric, threshold=threshold, **kwargs)


def _mota(threshold: float) -> TargetSpec:
    return TargetSpec(task="MOT", metric="mota", threshold=threshold, iou=0.5)


@pytest.mark.required
class TestFeasibilityTarget:
    def test_equal_targets_are_equal_and_hash_alike(self):
        a = _spec(threshold=1)
        b = TargetSpec(task="IC", metric="accuracy", threshold=1.0)  # 1 and 1.0 are the same threshold
        assert a == b
        assert hash(a) == hash(b)

    def test_a_spec_has_no_serialization_to_keep_in_step(self):
        assert not hasattr(TargetSpec, "canonical")
        assert not hasattr(TargetSpec, "from_canonical")

    def test_mota_accepts_negative_threshold(self):
        assert _parse([_mota(-3.0)], task="MOT")[0].threshold == -3.0
        with pytest.raises(ValueError, match="accuracy"):
            _spec(threshold=-0.1)
        with pytest.raises(ValueError, match="mota"):
            _mota(1.5)

    def test_ap_threshold_above_one_rejected(self):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            _spec("ap", 1.2, class_name="cat")
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            _spec(threshold=0.0)

    def test_macro_ap_requires_declared_classes(self):
        with pytest.raises(ValueError, match="index2label"):
            _parse([_spec("ap", 0.8)], class_names=None)
        spec = _parse([_spec("ap", 0.8)])[0]
        assert spec.class_name is None
        assert spec.ap_functional == "population_ap_integral"
        with pytest.raises(ValueError, match="bird"):
            _parse([_spec("ap", 0.8, class_name="bird")])

    def test_grouped_data_requires_aggregation(self):
        with pytest.raises(ValueError, match="aggregation"):
            _parse([_spec()], grouped=True)
        assert _parse([_spec(aggregation="sequence_mean")], grouped=True)[0].aggregation == "sequence_mean"
        assert _parse([_spec("ap", 0.9, aggregation="frame_pooled")], grouped=True)[0].aggregation == "frame_pooled"
        with pytest.raises(ValueError, match="grouped"):
            _parse([_spec(aggregation="frame_pooled")], grouped=False)

    def test_sequence_mean_ap_rejected(self):
        with pytest.raises(ValueError, match="sequence_mean"):
            _spec("ap", 0.9, aggregation="sequence_mean")

    def test_od_and_mot_targets_parse(self):
        od = _parse(
            [
                TargetSpec(task="OD", metric="ap", threshold=0.7, class_name="cat", iou=0.5, max_detections=20),
                TargetSpec(task="OD", metric="ap", threshold=0.5, class_name="dog", iou=0.75, max_detections=100),
            ],
            task="OD",
        )
        assert [(s.class_name, s.iou, s.max_detections) for s in od] == [("cat", 0.5, 20), ("dog", 0.75, 100)]
        mot = _parse([_mota(0.4)], task="MOT")[0]
        assert (mot.task, mot.metric, mot.iou) == ("MOT", "mota", 0.5)
        # The iou and fp_budget errors come before the max_detections one, which is checked last.
        with pytest.raises(ValueError, match="iou"):
            TargetSpec(task="OD", metric="ap", threshold=0.7, class_name="cat")
        with pytest.raises(ValueError, match="iou"):
            _spec("ap", 0.7, class_name="cat", iou=0.5)  # IC rejects iou
        with pytest.raises(ValueError, match="fp_budget"):
            TargetSpec(task="OD", metric="recall_at_fp", threshold=0.9, iou=0.5, max_detections=20)

    def test_classification_ap_takes_the_integral_functional_only(self):
        ap101 = _spec("ap", 0.8, class_name="cat", ap_functional="population_ap_101")
        with pytest.raises(ValueError, match="'population_ap_integral' only"):
            _parse([ap101])
        video = _spec("ap", 0.8, class_name="cat", ap_functional="population_ap_101", aggregation="frame_pooled")
        with pytest.raises(ValueError, match="'population_ap_integral' only"):
            _parse([video], grouped=True)
        integral = _spec("ap", 0.8, class_name="cat", ap_functional="population_ap_integral")
        assert _parse([integral]) == [integral]

    def test_ap_takes_scope_and_aggregation(self):
        spec = _parse([_spec("ap", 0.8, class_name="cat", scope="fixed_representation")])[0]
        assert (spec.class_name, spec.scope, spec.aggregation) == ("cat", "fixed_representation", None)
        spec = _parse([_spec("ap", 0.8, scope="partition_restricted", aggregation="frame_pooled")], grouped=True)[0]
        assert (spec.class_name, spec.scope, spec.aggregation) == (None, "partition_restricted", "frame_pooled")
        assert spec.ap_functional == "population_ap_integral"
        with pytest.raises(ValueError, match="scope"):
            _spec("ap", 0.8, class_name="cat", scope="everywhere")  # type: ignore[arg-type]

    def test_n_cells_for(self):
        assert [n_cells_for(n) for n in (1, 19_999, 20_000, 49_999, 50_000, 10**7)] == [10, 10, 30, 30, 300, 300]

    def test_recall_spec_fills_in_the_pooled_scope(self):
        s = TargetSpec(task="IC", metric="recall", threshold=0.9, class_name="Tank", max_false_positive_rate=0.05)
        assert s.false_positive_scope == "pooled"
        assert s.max_false_positive_rate == 0.05

    def test_explicit_pooled_scope_is_accepted(self):
        s = TargetSpec(
            task="IC",
            metric="recall",
            threshold=0.9,
            class_name="Tank",
            max_false_positive_rate=0.05,
            false_positive_scope="pooled",
        )
        assert s == TargetSpec(
            task="IC", metric="recall", threshold=0.9, class_name="Tank", max_false_positive_rate=0.05
        )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({}, "max_false_positive_rate"),
            ({"max_false_positive_rate": 0.0}, r"\(0, 1\)"),
            ({"max_false_positive_rate": 1.0}, r"\(0, 1\)"),
            ({"max_false_positive_rate": 1}, r"\(0, 1\)"),
            ({"max_false_positive_rate": 0.05, "false_positive_scope": "each_class"}, "each_class"),
            ({"max_false_positive_rate": 0.05, "aggregation": "frame_pooled"}, "aggregation"),
            ({"max_false_positive_rate": 0.05, "ap_functional": "population_ap_integral"}, "ap_functional"),
            ({"max_false_positive_rate": 0.05, "threshold": 0.0}, "threshold"),
            ({"max_false_positive_rate": 0.05, "threshold": 1.5}, "threshold"),
        ],
    )
    def test_recall_spec_refusals(self, kwargs, match):
        kwargs = {"threshold": 0.9, **kwargs}
        with pytest.raises(ValueError, match=match):
            TargetSpec(task="IC", metric="recall", class_name="Tank", **kwargs)

    def test_recall_needs_a_class(self):
        with pytest.raises(ValueError, match="class_name"):
            TargetSpec(task="IC", metric="recall", threshold=0.9, max_false_positive_rate=0.05)

    @pytest.mark.parametrize("kwargs", [{"max_false_positive_rate": 0.05}, {"false_positive_scope": "pooled"}])
    def test_limit_and_scope_only_apply_to_recall(self, kwargs):
        with pytest.raises(ValueError, match="apply to 'recall' only"):
            TargetSpec(task="IC", metric="accuracy", threshold=0.9, **kwargs)

    def test_recall_on_grouped_data_is_refused(self):
        s = TargetSpec(task="IC", metric="recall", threshold=0.9, class_name="Tank", max_false_positive_rate=0.05)
        with pytest.raises(ValueError, match="grouped data"):
            _parse([s], class_names=["Tank", "BMP"], grouped=True)


def _od(**kwargs: Any) -> TargetSpec:
    return TargetSpec(**{"task": "OD", "metric": "ap", "threshold": 0.5, "class_name": "cat", "iou": 0.5, **kwargs})


@pytest.mark.required
class TestDetectionTargets:
    def test_max_detections_is_required_and_stored_as_int(self):
        k = _od(max_detections=np.int64(20)).max_detections
        assert k == 20
        assert type(k) is int
        with pytest.raises(ValueError, match="max_detections"):
            _od()

    @pytest.mark.parametrize("k", [True, np.True_, 0, -1, 20.0, np.float64(20), 2.5, "20"])
    def test_max_detections_refuses_bool_zero_and_floats(self, k):
        with pytest.raises(ValueError, match="max_detections"):
            _od(max_detections=k)

    def test_max_detections_applies_to_od_only(self):
        with pytest.raises(ValueError, match="object detection only"):
            _spec(max_detections=20)
        with pytest.raises(ValueError, match="object detection only"):
            TargetSpec(task="MOT", metric="mota", threshold=0.4, iou=0.5, max_detections=20)

    def test_ap_functional_defaults_for_od(self):
        assert _od(max_detections=20).ap_functional == "population_ap_integral"

    def test_k_is_part_of_equality(self):
        ten, twenty = _od(max_detections=10), _od(max_detections=20)
        assert ten != twenty
        assert _parse([ten, twenty], task="OD") == [ten, twenty]
        assert _od(max_detections=np.int64(20)) == twenty
        assert hash(_od(max_detections=np.int64(20))) == hash(twenty)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            (
                {"metric": "recall_at_fp", "threshold": 0.9, "class_name": None, "fp_budget": 2.0},
                r"^'recall_at_fp' is not supported for object detection in this release; use TargetSpec\(task='OD'",
            ),
            ({"class_name": None}, r"^Macro 'ap' is not supported for object detection .*name one class: TargetSpec\("),
            (
                {"ap_functional": "population_ap_101"},
                r"'population_ap_integral' only in this release, got 'population_ap_101'; use TargetSpec\(task='OD'",
            ),
        ],
        ids=["recall_at_fp", "macro_ap", "population_ap_101"],
    )
    def test_od_parse_refusals(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _parse([_od(max_detections=20, **kwargs)], task="OD")

    def test_ic_errors_point_to_od(self):
        with pytest.raises(ValueError, match="task='OD'"):
            _spec("ap", 0.6, class_name="tank", iou=0.5, max_detections=20)
        with pytest.raises(ValueError, match="object detection only, not task 'IC'; .*set task='OD'"):
            _spec("ap", 0.6, class_name="tank", max_detections=20)
        with pytest.raises(ValueError, match="task='OD'"):
            _parse([_spec("ap", 0.6, class_name="cat")], task="OD")

    def test_an_ic_recall_target_on_od_data_is_shown_the_supported_od_target(self):
        recall = _spec("recall", 0.9, class_name="cat", max_false_positive_rate=0.05)
        supported = re.escape(
            "TargetSpec(task='OD', metric='ap', class_name=..., threshold=..., iou=..., max_detections=...)"
        )
        with pytest.raises(
            ValueError, match=rf"^A 'recall' target is a IC target but the data is OD; use {supported}$"
        ):
            _parse([recall], task="OD")
