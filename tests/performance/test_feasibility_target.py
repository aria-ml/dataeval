import json

import pytest

from dataeval.performance._feasibility_constants import SPEC_VERSION, n_cells_for
from dataeval.performance._feasibility_target import TargetSpec, Task, parse_targets

CLASSES = ["cat", "dog"]


def _parse(target, task: Task = "IC", class_names=CLASSES, grouped=False):
    return parse_targets(target, task=task, class_names=class_names, grouped=grouped)


@pytest.mark.required
class TestFeasibilityTarget:
    def test_canonical_round_trip(self):
        specs = _parse({"accuracy": 0.9, "ap:cat": 0.8, "accuracy/fixed_representation": 0.7})
        specs += _parse({"mota@0.5": -0.2}, task="MOT")
        for spec in specs:
            text = spec.canonical()
            assert TargetSpec.from_canonical(text) == spec
            assert json.loads(text)["version"] == SPEC_VERSION
            assert list(json.loads(text)) == sorted(json.loads(text))
        assert json.loads(specs[0].canonical())["iou"] is None  # nulls kept

    def test_equal_targets_serialize_identically(self):
        a = _parse({"accuracy": 1})[0]
        b = TargetSpec("IC", "accuracy", 1.0, None, None, None, None, None)
        assert a == b
        assert a.canonical() == b.canonical()

    def test_mota_accepts_negative_threshold(self):
        assert _parse({"mota@0.5": -3.0}, task="MOT")[0].threshold == -3.0
        with pytest.raises(ValueError, match="accuracy"):
            _parse({"accuracy": -0.1})
        with pytest.raises(ValueError, match="mota"):
            _parse({"mota@0.5": 1.5}, task="MOT")

    def test_ap_threshold_above_one_rejected(self):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            _parse({"ap:cat": 1.2})
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            _parse({"accuracy": 0.0})

    def test_macro_ap_requires_declared_classes(self):
        with pytest.raises(ValueError, match="index2label"):
            _parse({"ap": 0.8}, class_names=None)
        spec = _parse({"ap": 0.8})[0]
        assert spec.class_name is None
        assert spec.ap_functional == "population_ap_integral"
        with pytest.raises(ValueError, match="bird"):
            _parse({"ap:bird": 0.8})

    def test_grouped_data_requires_aggregation(self):
        with pytest.raises(ValueError, match="aggregation"):
            _parse({"accuracy": 0.9}, grouped=True)
        assert _parse({"accuracy|sequence_mean": 0.9}, grouped=True)[0].aggregation == "sequence_mean"
        assert _parse({"ap|frame_pooled": 0.9}, grouped=True)[0].aggregation == "frame_pooled"
        with pytest.raises(ValueError, match="grouped"):
            _parse({"accuracy|frame_pooled": 0.9}, grouped=False)

    def test_sequence_mean_ap_rejected(self):
        with pytest.raises(ValueError, match="sequence_mean"):
            _parse({"ap|sequence_mean": 0.9}, grouped=True)

    def test_od_and_mot_targets_parse(self):
        od = _parse({"ap:cat@0.5": 0.7, "ap@0.75": 0.5}, task="OD")
        assert [(s.class_name, s.iou) for s in od] == [("cat", 0.5), (None, 0.75)]
        mot = _parse({"mota@0.5": 0.4}, task="MOT")[0]
        assert (mot.task, mot.metric, mot.iou) == ("MOT", "mota", 0.5)
        rec = TargetSpec("OD", "recall_at_fp", 0.9, None, 0.5, None, 2.0, None)
        assert _parse([rec], task="OD") == [rec]
        with pytest.raises(ValueError, match="iou"):
            _parse({"ap:cat": 0.7}, task="OD")
        with pytest.raises(ValueError, match="iou"):
            _parse({"ap:cat@0.5": 0.7})  # IC rejects iou
        with pytest.raises(ValueError, match="fp_budget"):
            _parse({"recall_at_fp@0.5": 0.9}, task="OD")

    def test_unknown_key_lists_valid_forms(self):
        for key in ("f1", "accuracy/nowhere", "accuracy|mean", "ap@x", ""):
            with pytest.raises(ValueError, match=r"metric\[:class\]\[@iou\]\[/scope\]\[\|aggregation\]") as e:
                _parse({key: 0.5})
            assert repr(key) in str(e.value)
            assert "ap_functional" not in str(e.value)  # a key cannot set it, and classification takes one only

    def test_classification_ap_takes_the_integral_functional_only(self):
        ap101 = TargetSpec("IC", "ap", 0.8, "cat", ap_functional="population_ap_101")
        with pytest.raises(ValueError, match="'population_ap_integral' only"):
            _parse([ap101])
        video = TargetSpec("IC", "ap", 0.8, "cat", ap_functional="population_ap_101", aggregation="frame_pooled")
        with pytest.raises(ValueError, match="'population_ap_integral' only"):
            _parse([video], grouped=True)
        integral = TargetSpec("IC", "ap", 0.8, "cat", ap_functional="population_ap_integral")
        assert _parse([integral]) == [integral]

    def test_ap_keys_take_scope_and_aggregation(self):
        spec = _parse({"ap:cat/fixed_representation": 0.8})[0]
        assert (spec.class_name, spec.scope, spec.aggregation) == ("cat", "fixed_representation", None)
        spec = _parse({"ap/partition_restricted|frame_pooled": 0.8}, grouped=True)[0]
        assert (spec.class_name, spec.scope, spec.aggregation) == (None, "partition_restricted", "frame_pooled")
        assert spec.ap_functional == "population_ap_integral"
        with pytest.raises(ValueError, match="scope"):
            _parse({"ap:cat/everywhere": 0.8})

    def test_duplicate_targets_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            _parse([_parse({"accuracy": 0.9})[0], _parse({"accuracy": 0.9})[0]])

    def test_from_canonical_rejects_bad_version(self):
        text = _parse({"accuracy": 0.9})[0].canonical().replace('"version":1', '"version":99')
        with pytest.raises(ValueError, match="version"):
            TargetSpec.from_canonical(text)

    def test_n_cells_for(self):
        assert [n_cells_for(n) for n in (1, 19_999, 20_000, 49_999, 50_000, 10**7)] == [10, 10, 30, 30, 300, 300]

    def test_recall_spec_round_trips_with_pooled_scope(self):
        s = TargetSpec(task="IC", metric="recall", threshold=0.9, class_name="Tank", max_false_positive_rate=0.05)
        assert s.false_positive_scope == "pooled"
        assert TargetSpec.from_canonical(s.canonical()) == s
        assert '"max_false_positive_rate":0.05' in s.canonical()
        assert '"version":1' in s.canonical()

    def test_explicit_pooled_scope_is_accepted(self):
        s = TargetSpec(
            task="IC",
            metric="recall",
            threshold=0.9,
            class_name="Tank",
            max_false_positive_rate=0.05,
            false_positive_scope="pooled",
        )
        assert TargetSpec.from_canonical(s.canonical()) == s

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

    def test_recall_mapping_key_is_refused(self):
        with pytest.raises(ValueError, match="pass a TargetSpec"):
            _parse({"recall:Tank": 0.9}, class_names=["Tank", "BMP"])

    def test_recall_on_grouped_data_is_refused(self):
        s = TargetSpec(task="IC", metric="recall", threshold=0.9, class_name="Tank", max_false_positive_rate=0.05)
        with pytest.raises(ValueError, match="grouped data"):
            _parse([s], class_names=["Tank", "BMP"], grouped=True)
