"""Feasibility judges an object detection requirement end to end, from a detector's detections."""

import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

import dataeval.performance._feasibility as feasibility_module
from dataeval.data import Indices, View
from dataeval.extractors import FlattenExtractor
from dataeval.performance import Allowances, Feasibility, TargetSpec
from dataeval.performance._feasibility_detection import resolve_class_ids
from dataeval.protocols import DatasetMetadata
from tests.data.conftest import VIDEO_FACTORS, VIDEOS, FactorVideos

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

NAMES = {0: "person", 1: "car"}
N = 60


@dataclass
class _Det:
    """One image's boxes, labels and scores: a dataset target, or a detector double's output for the image."""

    boxes: Any
    labels: Any
    scores: Any


def _box(slot: int, shift: float = 0.0) -> list[float]:
    """The box of slot ``slot``: slots never overlap, so a target matches only a detection in its own slot."""
    return [20.0 * slot + shift, 0.0, 20.0 * slot + 10.0 + shift, 10.0]


class _OD:
    """
    A synthetic detection dataset: per image, a list of target labels, target ``j`` in slot ``j``.

    Item ``i`` has datum id ``im{i}`` unless ``ids`` are given; ``index2label`` is declared unless it is None.
    """

    def __init__(self, images: list[list[int]], index2label: dict[int, str] | None = NAMES, ids=None) -> None:
        self.images = [list(labels) for labels in images]
        self.ids = list(ids) if ids is not None else [f"im{i}" for i in range(len(images))]
        self.metadata = (
            DatasetMetadata(id="toy-od")
            if index2label is None
            else DatasetMetadata(id="toy-od", index2label=index2label)
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        labels = self.images[i]
        boxes = np.array([_box(j) for j in range(len(labels))], dtype=np.float64).reshape(-1, 4)
        target = _Det(boxes, np.array(labels, dtype=np.intp), np.ones(len(labels)))
        return np.zeros((3, 2, 2), dtype=np.float32), target, {"id": self.ids[i]}


def _scenes(n: int = N) -> list[list[int]]:
    """Image ``i`` holds ``i % 3`` persons, then ``i % 2`` cars: image 0 is empty, B is 2 for person, 1 for car."""
    return [[0] * (i % 3) + [1] * (i % 2) for i in range(n)]


def _detector(images: list[list[int]], *, shift: float = 0.0) -> list[_Det]:
    """
    A detector double: target ``j`` of image ``i`` is found at score 0.9, except every fourth ``(i + j)``, which it
    misses; every other image also gets a false positive in slot 9 (no target there) at a score in [0.2, 0.4] and of
    class ``i % 2``. ``shift`` moves every box right, so a large shift leaves no detection matching a target.
    """
    out = []
    for i, labels in enumerate(images):
        found = [(j, c, 0.9) for j, c in enumerate(labels) if (i + j) % 4]
        if i % 2:
            found.append((9, i % 2, 0.2 + 0.05 * (i % 5)))
        boxes = np.array([_box(j, shift) for j, _, _ in found], dtype=np.float64).reshape(-1, 4)
        out.append(_Det(boxes, np.array([c for _, c, _ in found], dtype=np.intp), np.array([s for *_, s in found])))
    return out


def _ap(class_name: str = "person", threshold: float = 0.5, k: int = 5, **kwargs) -> TargetSpec:
    return TargetSpec(
        task="OD", metric="ap", class_name=class_name, threshold=threshold, iou=0.5, max_detections=k, **kwargs
    )


def _run(images=None, target=None, *, max_objects: Any = 3, dataset=None, **kwargs):
    images = _scenes() if images is None else images
    dataset = _OD(images) if dataset is None else dataset
    kwargs.setdefault("predictions", _detector(images))
    return Feasibility(seed=0, max_objects=max_objects).evaluate(dataset, target=target or [_ap()], **kwargs)


@contextmanager
def _no_pass():
    """Fail the test if the Metadata pass over the dataset runs."""
    with patch.object(feasibility_module, "resolve_detection_units", side_effect=AssertionError("the pass ran")):
        yield


@pytest.mark.required
class TestRefusals:
    def test_unusable_inputs_are_named_together_before_the_pass(self):
        unusable = {
            "model": cast(Any, object()),
            "embeddings": np.zeros((N, 2)),
            "replicas": {"im0": [0]},
            "allowances": Allowances(beta=0.0),
        }
        with _no_pass(), pytest.raises(ValueError, match="apply to classification only") as info:
            _run(**unusable)
        message = str(info.value)
        assert all(f"{name}=" in message for name in unusable)
        assert "pass predictions=, one detections object per image" in message
        assert "no envelope" in message
        assert "predictions= is missing" not in message

    def test_missing_predictions_is_named_with_the_unusable_inputs(self):
        with _no_pass(), pytest.raises(ValueError, match="embeddings=") as info:
            _run(predictions=None, embeddings=np.zeros((N, 2)))
        assert "predictions= is missing" in str(info.value)
        with _no_pass(), pytest.raises(ValueError, match="predictions= is missing"):
            _run(predictions=None)

    def test_an_empty_replicas_mapping_is_not_an_input(self):
        assert len(_run(replicas={}).determinations) == 1

    @pytest.mark.parametrize(
        ("make", "match"),
        [
            (lambda d: np.zeros((N, 6)), r"one detections object per image \(60\), got an array of shape \(60, 6\)"),
            (lambda d: pytest.importorskip("torch").zeros((N, 6)), r"got an array of shape \(60, 6\)"),
            (lambda d: d[:-1], r"one detections object per image \(60\), got 59;"),
            (lambda d: [vars(x) for x in d], r"^predictions\[0\] is a mapping; .*types\.SimpleNamespace\(\*\*d\)"),
            (lambda d: [np.c_[x.boxes, x.labels, x.scores] for x in d], r"^predictions\[0\] has no 'boxes'"),
        ],
        ids=["one-array", "one-tensor", "one-short", "a-dict-per-image", "an-array-per-image"],
    )
    def test_misshapen_predictions_are_refused_before_the_pass(self, make, match):
        with _no_pass(), pytest.raises(ValueError, match=match):
            _run(predictions=make(_detector(_scenes())))

    def test_a_classification_target_is_refused_before_the_pass(self):
        with _no_pass(), pytest.raises(ValueError, match=r"use TargetSpec\(task='OD', metric='ap'"):
            _run(target=[TargetSpec(metric="accuracy", threshold=0.5)])

    def test_groups_that_pair_images_are_refused(self):
        groups = np.arange(N) // 2
        with pytest.raises(ValueError, match="images must be independent units"):
            _run(groups=groups)

    def test_tracking_is_not_implemented(self):
        with pytest.raises(NotImplementedError) as info:
            Feasibility(max_objects=3).evaluate(
                cast(Any, FactorVideos(VIDEOS, VIDEO_FACTORS)), target=[TargetSpec(metric="accuracy", threshold=0.5)]
            )
        assert "MOT" in str(info.value)
        assert "planned" not in str(info.value)

    def test_classification_settings_change_nothing(self):
        plain = _run()
        images = _scenes()
        dataset = _OD(images)
        tuned = Feasibility(extractor=FlattenExtractor(), max_sequence_length=2, seed=0, max_objects=3).evaluate(
            dataset, target=[_ap()], predictions=_detector(images)
        )
        assert tuned.determinations.equals(plain.determinations)
        assert tuned.bounds.equals(plain.bounds)


@pytest.mark.required
class TestCountBound:
    @pytest.mark.parametrize("max_objects", [None, {"other": 3}])
    def test_missing_bound_raises_before_any_pass(self, max_objects):
        with _no_pass(), pytest.raises(ValueError, match=r"max_objects=") as info:
            _run(max_objects=max_objects)
        message = str(info.value)
        if max_objects is None:
            assert "largest number of targets of the class in any image of the population" in message
            assert "assumed and not verified" in message
        else:
            assert "['other']" in message
            assert "['person']" in message

    def test_data_above_bound_lists_every_class(self):
        # person: 2 in im1; car: 3 in im2; truck (no requirement) holds 5 and is not checked.
        images = [[0], [0, 0, 1], [1, 1, 1], [], [2] * 5]
        dataset = _OD(images, {0: "person", 1: "car", 2: "truck"})
        with pytest.raises(ValueError, match=r"Feasibility\(max_objects=\.\.\.\) declares") as info:
            _run(images, [_ap("person"), _ap("car")], max_objects=1, dataset=dataset)
        message = str(info.value)
        assert "'person': 2 targets in item 'im1' (dataset index 1), above max_objects 1" in message
        assert "'car': 3 targets in item 'im2' (dataset index 2), above max_objects 1" in message
        assert "truck" not in message
        assert "only a floor" in message

    def test_unused_mapping_key_is_accepted(self):
        out = _run(max_objects={"car": 1, "person": 3})
        assert len(out.determinations) == 1

    def test_mapping_bound_is_recorded_in_state(self):
        assert _run(max_objects={"person": 3}).meta().state["max_objects"] == {"person": 3}


@pytest.mark.required
class TestClassIds:
    def test_sparse_index2label_maps_by_declared_id(self):
        sparse = {1: "a", 3: "b", 7: "c"}
        images = [[7, 1] if i % 2 else [3, 7] for i in range(N)]
        dataset = _OD(images, sparse)
        specs = [_ap("a"), _ap("b"), _ap("c")]
        assert resolve_class_ids(dataset, specs) == [1, 3, 7]
        out = _run(images, [_ap("c")], dataset=dataset, predictions=_detector(images))
        assert out.bounds["state"].to_list() == ["informative"]  # by position, "c" would be label 2, which no box has
        with pytest.raises(ValueError, match="'c': 2 targets in item 'im0'"):
            _run([[7, 7], *images], [_ap("c")], max_objects=1, dataset=_OD([[7, 7], *images], sparse))

    def test_without_index2label_a_digit_string_is_the_label(self):
        images = _scenes()
        dataset = _OD(images, None)
        assert resolve_class_ids(dataset, [_ap("1"), _ap("0")]) == [1, 0]
        with _no_pass(), pytest.raises(ValueError, match="declares no index2label"):
            _run(images, [_ap("person")], dataset=dataset)
        with pytest.raises(ValueError, match=r"Class '5' is label 5.*labels \[0, 1\]"):
            _run(images, [_ap("5")], dataset=dataset)
        assert len(_run(images, [_ap("0")], dataset=dataset).determinations) == 1


@pytest.mark.required
class TestCalls:
    def test_determinism(self):
        first, second = _run(), _run()
        assert first.determinations.equals(second.determinations)
        assert first.bounds.equals(second.bounds)
        images, ids = _scenes(), np.arange(N)
        order = np.random.default_rng(1).permutation(N)
        detections = _detector(images)
        given = _run(images, dataset=_OD(images), predictions=detections, groups=ids)
        reordered = _run(
            [images[i] for i in order],
            dataset=_OD([images[i] for i in order], ids=[f"im{i}" for i in order]),
            predictions=[detections[i] for i in order],
            groups=ids[order],
        )
        assert given.bounds["bound"][0] > 0
        assert reordered.bounds["bound"].to_list() == pytest.approx(given.bounds["bound"].to_list(), rel=1e-12)

    def test_easy_target_is_demonstrated(self):
        # One person per image, found at 0.9 and nothing else: B = K = 1. Every image is alike, so the bound depends
        # only on the audit size (200 of 400 images), not on which images the seed puts there. Every audit count is
        # 1 TP, 0 FP and 1 target, so the witness falls short of 1 only by the endpoints' radii.
        images = [[0]] * 400
        perfect = [_Det([_box(0)], [0], [0.9]) for _ in images]
        out = _run(images, [_ap(threshold=0.3, k=1)], max_objects=1, predictions=perfect)
        assert out.determinations["determination"].to_list() == ["demonstrated_feasible"]
        (row,) = out.bounds.iter_rows(named=True)
        assert (row["source"], row["quantity"], row["is_determining"]) == ("user_model", "ap", True)
        assert row["bound"] == pytest.approx(0.864431942063233, rel=1e-6)  # measured, far above the 0.3 asked

    def test_narrower_scope_is_unresolved(self):
        images = [[0]] * 400
        perfect = [_Det([_box(0)], [0], [0.9]) for _ in images]
        target = [_ap(threshold=0.3, k=1, scope="fixed_representation")]
        row = _run(images, target, max_objects=1, predictions=perfect).determinations.row(0, named=True)
        assert (row["determination"], row["limitation"]) == ("unresolved", "no_envelope")

    def test_one_class_dataset_runs(self):
        images = [[0] * (i % 3) for i in range(N)]
        persons_only = [_Det(d.boxes, np.zeros_like(d.labels), d.scores) for d in _detector(images)]
        out = _run(images, dataset=_OD(images, {0: "person"}), predictions=persons_only)
        assert len(out.determinations) == 1
        assert out.bounds["state"].to_list() == ["informative"]

    def test_torch_detections_match_numpy(self):
        torch = pytest.importorskip("torch")
        images = _scenes()
        tensors = [
            _Det(torch.as_tensor(d.boxes), torch.as_tensor(d.labels), torch.as_tensor(d.scores))
            for d in _detector(images)
        ]
        from_torch, from_numpy = _run(predictions=tensors), _run()
        assert from_torch.determinations.equals(from_numpy.determinations)
        assert from_torch.bounds.equals(from_numpy.bounds)

    def test_requirements_differing_in_k_are_two_claims(self):
        out = _run(target=[_ap(k=1), _ap(k=5)])
        assert len(out.determinations) == 2
        assert out.bounds["requirement"].to_list() == [0, 1]
        assert out.bounds["alpha"].to_list() == [0.05 / 2, 0.05 / 2]

    def test_view_over_od_dataset(self):
        images = _scenes()
        detections = _detector(images)
        view: Any = View(_OD(images), Indices(range(N)))  # no repeats: each image its own unit
        out = Feasibility(seed=0, max_objects=3).evaluate(view, target=[_ap()], predictions=detections)
        assert out.bounds.equals(_run().bounds)
        repeated: Any = View(_OD(images), Indices([0, *range(N)]))  # image 0 twice: its copies are one unit
        with pytest.raises(ValueError, match="images must be independent units"):
            Feasibility(seed=0, max_objects=3).evaluate(
                repeated, target=[_ap()], predictions=[detections[0], *detections]
            )

    def test_no_match_warning_points_at_the_call(self):
        images = _scenes()
        feasibility, dataset, missed = Feasibility(seed=0, max_objects=3), _OD(images), _detector(images, shift=500.0)
        named = r"None of the \d+ kept detections of class 'person' \(label 0\)"
        line = cast(Any, inspect.currentframe()).f_lineno + 2  # the evaluate call's line
        with pytest.warns(UserWarning, match=named) as record:
            feasibility.evaluate(dataset, target=[_ap()], predictions=missed)
        (warning,) = [w for w in record if "None of the" in str(w.message)]
        assert (warning.filename, warning.lineno) == (__file__, line)

    def test_an_od_call_has_one_user_witness_bound_per_requirement(self):
        out = _run(target=[_ap("person"), _ap("car")])
        assert out.bounds.select("requirement", "source", "role", "quantity").rows() == [
            (0, "user_model", "witness", "ap"),
            (1, "user_model", "witness", "ap"),
        ]
        assert out.bounds["alpha"].sum() <= 0.05 + 1e-12
        assert set(out.determinations["determination"]) <= {"demonstrated_feasible", "unresolved"}
        assert set(out.determinations["limitation"].drop_nulls()) <= {"no_envelope"}  # witness only: no envelope
