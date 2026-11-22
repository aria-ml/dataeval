"""Feasibility judges an object detection requirement end to end, from a detector's detections."""

import inspect
import re
from contextlib import contextmanager
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

import dataeval.performance._feasibility as feasibility_module
from dataeval.data import Indices, View
from dataeval.extractors import FlattenExtractor
from dataeval.performance import Allowances, Feasibility, TargetSpec
from dataeval.performance._feasibility_constants import FIT_FRACTION
from dataeval.performance._feasibility_detection import capacity_from_counts, resolve_class_ids
from dataeval.performance._feasibility_units import resolve_detection_units, split_units
from tests.data.conftest import VIDEO_FACTORS, VIDEOS, FactorVideos
from tests.performance.feasibility_helpers import _OD, _box, _Det

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

N = 60
NO_DETECTOR = "No detector is given (predictions= or model=)"
CALL = (
    "Object detection takes the targets, each requirement's max_detections K and Feasibility(max_objects=...) for the "
    "capacity envelope, and a detector for the witness: predictions=, one detections object per image, or model=, a "
    "MAITE object detector."
)


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
        unusable = {"embeddings": np.zeros((N, 2)), "replicas": {"im0": [0]}, "allowances": Allowances(beta=0.0)}
        detector = cast(Any, lambda batch: pytest.fail("the detector ran"))  # with predictions=, two detectors
        with _no_pass(), pytest.raises(ValueError, match="apply to classification only") as info:
            _run(model=detector, **unusable)
        message = str(info.value)
        assert message.startswith("embeddings=, replicas=, allowances= apply to classification only; ")
        assert "model= and predictions= are both given; pass one, since a call has one detector" in message
        assert message.endswith(CALL)
        assert "no envelope" not in message
        assert NO_DETECTOR not in message

    def test_missing_detector_is_named_with_the_unusable_inputs(self):
        with _no_pass(), pytest.raises(ValueError, match="embeddings=") as info:
            _run(predictions=None, embeddings=np.zeros((N, 2)))
        assert NO_DETECTOR in str(info.value)
        assert str(info.value).endswith(CALL)
        with _no_pass(), pytest.raises(ValueError, match=re.escape(NO_DETECTOR)) as info:
            _run(predictions=None)  # K = 5 and B = 3: the cap cannot bind, so only a witness can judge it
        assert "'person' at K = 5, max_objects 3 (requirement 0)" in str(info.value)

    def test_missing_detector_names_every_requirement_whose_cap_cannot_bind(self):
        # A mixed call: person at K = 1 has a capacity envelope, car at K = 5 and 3 do not; B is 3 for both.
        target = [_ap("person", k=1), _ap("car", k=5), _ap("car", k=3)]
        with _no_pass(), pytest.raises(ValueError, match=re.escape(NO_DETECTOR)) as info:
            _run(target=target, predictions=None)
        message = str(info.value)
        assert "'car' at K = 5, max_objects 3 (requirement 1), 'car' at K = 3, max_objects 3 (requirement 2)" in message
        assert "requirement 0" not in message
        assert "every requirement whose max_detections K is at or above max_objects needs one" in message
        assert f"Pass predictions= or model=, or state a max_detections below max_objects. {CALL}" in message

    def test_missing_detector_at_a_bound_of_one_offers_only_a_detector(self):
        # B = 1 leaves no K below it, since TargetSpec refuses max_detections=0, so the only fix is a detector.
        with _no_pass(), pytest.raises(ValueError, match=re.escape(NO_DETECTOR)) as info:
            _run(target=[_ap("car", k=1)], max_objects={"car": 1}, predictions=None)
        message = str(info.value)
        assert f"'car' at K = 1, max_objects 1 (requirement 0). Pass predictions= or model=. {CALL}" in message
        assert "state a max_detections" not in message

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
        # A mixed call with a detector (B = 3): K = 1 gets a capacity claim and K = 5 none, and each of the 2 + 1
        # claims, the K = 5 witness's included, spends alpha / 3.
        out = _run(target=[_ap(k=1), _ap(k=5)])
        assert len(out.determinations) == 2
        assert out.bounds.select("requirement", "source").rows() == [
            (0, "capacity"),
            (0, "user_model"),
            (1, "user_model"),
        ]
        assert out.bounds["alpha"].to_list() == [0.05 / 3] * 3

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

    def test_detection_boxes_with_no_area_are_false_positives(self):
        # The padding of a fixed-size detector export: a box with no area has IoU 0 with every target, so it is a false
        # positive, the same as a box far from every target. Every third image gets one of class 0 at 0.95.
        def padded(box):
            images = _scenes()
            out = []
            for i, d in enumerate(_detector(images)):
                if i % 3 == 0:
                    d = _Det(np.r_[d.boxes, [box]], np.r_[d.labels, 0], np.r_[d.scores, 0.95])
                out.append(d)
            return out

        plain = _run().bounds["bound"][0]
        empty = _run(predictions=padded([0.0, 0.0, 0.0, 0.0])).bounds["bound"][0]
        far = _run(predictions=padded(_box(8))).bounds["bound"][0]
        assert plain > 0  # a visible drop needs a baseline above 0
        assert empty == far
        assert empty < plain

    def test_a_target_box_with_no_area_is_still_refused(self):
        class Flat(_OD):
            def __getitem__(self, i: int):
                image, target, meta = super().__getitem__(i)
                if i == 4:
                    target = _Det(np.array([[5.0, 0.0, 5.0, 10.0]]), np.array([0]), np.ones(1))
                return image, target, meta

        images = _scenes()
        with pytest.raises(ValueError, match=r"dataset item 4 targets: box 0 .*positive width and height"):
            _run(images, dataset=Flat(images))

    def test_an_od_call_has_one_user_witness_bound_per_requirement(self):
        out = _run(target=[_ap("person"), _ap("car")])
        assert out.bounds.select("requirement", "source", "role", "quantity").rows() == [
            (0, "user_model", "witness", "ap"),
            (1, "user_model", "witness", "ap"),
        ]
        assert out.bounds["alpha"].sum() <= 0.05 + 1e-12
        assert set(out.determinations["determination"]) <= {"demonstrated_feasible", "unresolved"}
        assert set(out.determinations["limitation"].drop_nulls()) <= {"no_envelope"}  # witness only: no envelope


def _fit_side(n: int) -> set[int]:
    """The images that ``split_units`` puts on the fit side of ``n`` ungrouped images at seed 0, read from the split."""
    units = resolve_detection_units(_OD([[]] * n), groups=None)  # the split reads only the units and the seed
    return set(split_units(units, fit_fraction=FIT_FRACTION, seed=0)[0].tolist())


def _capacity_row(out) -> dict[str, Any]:
    (row,) = [r for r in out.bounds.iter_rows(named=True) if r["source"] == "capacity"]
    return row


@pytest.mark.required
class TestCapacity:
    """The capacity envelope of a requirement whose K is below its B, with a detector and without one."""

    def test_dense_scene_is_ruled_out_without_a_detector(self):
        # Six persons in every image and K = 2: no detector capped at 2 finds more than a third of them.
        out = _run([[0] * 6] * 200, [_ap(threshold=0.5, k=2)], max_objects=6, predictions=None)
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["limitation"]) == ("infeasible_under_recorded_assumptions", None)
        assert row["envelope"] == pytest.approx(0.3575138752967017, rel=1e-9)  # measured: the ratio 1/3 and its radius
        assert out.bounds.select("source", "role", "alpha", "is_determining").rows() == [
            ("capacity", "envelope", 0.05, True)
        ]

    def test_informative_bound_above_the_threshold_is_not_ruled_out(self):
        # Every other image holds 4 persons and K = 2: the ratio 0.5 is below the 0.6 asked, the bound (0.877) above.
        out = _run([[0] * 4, []] * 50, [_ap(threshold=0.6, k=2)], max_objects=6, predictions=None)
        assert out.bounds["state"].to_list() == ["informative"]
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["limitation"]) == ("not_ruled_out", "no_eligible_witness")
        # Measured (#1652): images at which the capacity envelope, projected at the observed rates, falls below 0.6.
        assert (row["suggested_minimum"], row["minimum_source"], row["minimum_quantity"]) == (1_697, "capacity", "ap")

    def test_bound_is_one_when_no_image_exceeds_k(self):
        # _scenes holds at most 2 persons in an image, so at K = 2 the cap never binds on this data.
        out = _run(target=[_ap(k=2)], predictions=None)
        bound = _capacity_row(out)
        assert (bound["bound"], bound["state"], bound["reason"]) == (1.0, "uninformative", "bound_at_maximum")
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["limitation"]) == ("unresolved", "no_eligible_witness")

    def test_one_image_gives_no_envelope(self):
        out = _run([[0, 0]], [_ap(k=1)], predictions=None)
        bound = _capacity_row(out)
        assert (bound["bound"], bound["state"], bound["reason"]) == (None, "unavailable", "too_few_units")
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["limitation"]) == ("unresolved", "envelope_unavailable")

    def test_every_image_is_used_whatever_the_seed(self):
        images = _scenes(300)
        counts = np.array([labels.count(0) for labels in images])
        expected = capacity_from_counts(counts, max_objects=3, max_detections=1, alpha_claim=0.05 / 2)[0]
        outs = [
            Feasibility(seed=seed, max_objects=3).evaluate(
                _OD(images), target=[_ap(k=1)], predictions=_detector(images)
            )
            for seed in (0, 1)
        ]
        first, second = (_capacity_row(out) for out in outs)
        assert (first["bound"], first["n"], first["sample"]) == (expected, len(images), "all_items")
        assert (second["bound"], second["n"]) == (first["bound"], first["n"])
        assert outs[0].bounds["bound"][1] != outs[1].bounds["bound"][1]  # the seed moved the witness's audit

    def test_a_detector_adds_its_witness_after_the_envelope(self):
        # K = 1 and B = 3 on _scenes(300): the ratio 2/3 is below the 0.7 asked and the bound (0.894) above it.
        out = _run(_scenes(300), [_ap(threshold=0.7, k=1)])
        assert out.bounds.select("source", "role").rows() == [("capacity", "envelope"), ("user_model", "witness")]
        assert out.bounds["alpha"].to_list() == [0.05 / 2, 0.05 / 2]
        assert out.bounds["alpha"].sum() <= 0.05 + 1e-12
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["limitation"]) == ("not_ruled_out", "within_radius")
        # Measured (#1652): only the envelope crosses, since the witness's estimate (0.58) is below the 0.7 asked. Its
        # units are all images, as its row's n is.
        assert (row["suggested_minimum"], row["minimum_source"], row["minimum_quantity"]) == (8_408, "capacity", "ap")

    def test_a_witness_that_would_clear_names_the_audit_images_it_needs(self):
        # K = 1 and B = 3 on _scenes(300): the capacity bound (0.894) is above the 0.5 asked, so the witness decides.
        # Its estimate clears 0.5 and its bound does not. Both numbers are measured (#1652), not derived.
        row = _run(_scenes(300), [_ap(threshold=0.5, k=1)]).determinations.row(0, named=True)
        assert (row["determination"], row["limitation"]) == ("not_ruled_out", "within_radius")
        assert (row["suggested_minimum"], row["minimum_source"], row["minimum_quantity"]) == (2_474, "user_model", "ap")
        # At 0.6 the estimate is below the threshold, so no number of images would carry the bound across.
        row = _run(_scenes(300), [_ap(threshold=0.6, k=1)]).determinations.row(0, named=True)
        assert (row["determination"], row["limitation"]) == ("not_ruled_out", "weak_witness")
        assert row["suggested_minimum"] is None

    def test_a_crossing_envelope_is_reported_before_a_crossing_witness(self):
        # Fit images hold 2 persons and audit images 1, K = 1 and B = 2, a perfect detector, 0.74 asked. The envelope's
        # estimate (0.667, the ratio) is below it and its bound (0.749, measured) above it. The witness's estimate (1.0)
        # is above it and its bound (0.734) below it. Both cross, with no conflict, and the envelope is checked first.
        fit = _fit_side(400)
        images = [[0] * 2 if i in fit else [0] for i in range(400)]
        perfect = [_Det([_box(j) for j in range(len(labels))], labels, [0.9] * len(labels)) for labels in images]
        out = _run(images, [_ap(threshold=0.74, k=1)], max_objects=2, predictions=perfect)
        capacity, witness = out.bounds.iter_rows(named=True)
        assert (capacity["bound"], witness["bound"], witness["observed"]) == pytest.approx(
            (0.7494, 0.7343, 1.0), abs=1e-4
        )
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["limitation"]) == ("not_ruled_out", "within_radius")
        assert (row["suggested_minimum"], row["minimum_source"], row["minimum_quantity"]) == (465, "capacity", "ap")

    def test_a_detector_demonstrates_beside_a_capacity_row(self):
        # One person per image, K = 2 and B = 3: the envelope is built, sits at 1 and spends alpha / 2, and the
        # perfect detector's witness (0.576, measured) still clears the 0.3 asked.
        images = [[0]] * 400
        perfect = [_Det([_box(0)], [0], [0.9]) for _ in images]
        out = _run(images, [_ap(threshold=0.3, k=2)], max_objects=3, predictions=perfect)
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["limitation"]) == ("demonstrated_feasible", None)
        assert row["witness"] == pytest.approx(0.5756907188845602, rel=1e-9)
        assert out.bounds.select("source", "state", "is_determining").rows() == [
            ("capacity", "uninformative", False),
            ("user_model", "informative", True),
        ]

    def test_a_detector_on_a_dense_scene_is_ruled_out(self):
        # Six persons in every image, K = 2 and B = 6: the envelope (0.361, measured) is below the 0.5 asked, and the
        # perfect detector's witness (0.237) is below the envelope, so there is no conflict.
        images = [[0] * 6] * 200
        perfect = [_Det([_box(j) for j in range(6)], [0] * 6, [0.9] * 6) for _ in images]
        out = _run(images, [_ap(threshold=0.5, k=2)], max_objects=6, predictions=perfect)
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["limitation"]) == ("infeasible_under_recorded_assumptions", None)
        assert (row["envelope"], row["witness"]) == pytest.approx((0.3606577015545407, 0.23657412924565846), rel=1e-9)
        assert _capacity_row(out)["is_determining"]

    def test_no_detector_at_a_narrower_scope_is_not_ruled_out(self):
        # The informative bound of test_informative_bound_above_the_threshold_is_not_ruled_out, judged at
        # fixed_representation: the full_input envelope decides there too.
        target = [_ap(threshold=0.6, k=2, scope="fixed_representation")]
        out = _run([[0] * 4, []] * 50, target, max_objects=6, predictions=None)
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["limitation"]) == ("not_ruled_out", "no_eligible_witness")
        assert (row["envelope"], row["envelope_scope"]) == (pytest.approx(0.8772551488514445, rel=1e-9), "full_input")

    def test_every_k_at_or_above_b_is_as_before(self):
        # Recorded by running the route before it had a capacity envelope. Every K here is at or above B, so the
        # call makes no capacity claim and every bound is the same float as before. The minimum is new (#1652), a
        # measured projection of the user_model witness, in audit images.
        perfect = [_Det([_box(0)], [0], [0.9]) for _ in range(400)]
        calls = [
            _run(),
            _run(target=[_ap("person", 0.5, 5), _ap("car", 0.3, 3)], max_objects={"truck": 7, "car": 1, "person": 3}),
            _run([[0]] * 400, [_ap(threshold=0.3, k=1)], max_objects=1, predictions=perfect),
        ]
        assert [out.bounds.select("requirement", "source", "alpha", "bound").rows() for out in calls] == [
            [(0, "user_model", 0.05, 0.0035885242332850415)],
            [(0, "user_model", 0.025, 0.0007832551857172316), (1, "user_model", 0.025, 0.03264955812285865)],
            [(0, "user_model", 0.05, 0.864431942063233)],
        ]
        columns = ("determination", "limitation", "envelope", "witness", "suggested_minimum", "minimum_source")
        assert [out.determinations.select(columns).rows() for out in calls] == [
            [("unresolved", "no_envelope", None, 0.0035885242332850415, 508, "user_model")],
            [
                ("unresolved", "no_envelope", None, 0.0007832551857172316, 567, "user_model"),
                ("unresolved", "no_envelope", None, 0.03264955812285865, 182, "user_model"),
            ],
            [("demonstrated_feasible", None, None, 0.864431942063233, None, None)],
        ]

    def test_narrower_scope_is_decided_by_capacity(self):
        # The detector's witness counts at full_input only; the capacity envelope bounds any predictor, so it decides
        # at every scope.
        out = _run(_scenes(300), [_ap(threshold=0.7, k=1, scope="fixed_representation")])
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["limitation"], row["witness"]) == (
            "not_ruled_out",
            "no_eligible_witness",
            None,
        )
        capacity = _capacity_row(out)
        assert (row["envelope"], capacity["is_determining"]) == (capacity["bound"], True)

    def test_capacity_and_witness_can_conflict(self):
        # Every fit image holds 6 persons and every audit image 1: the witness sees only the audit, where the class
        # is easy, and capacity sees every image. A sampling event, at most alpha likely when both bounds hold.
        fit = _fit_side(400)
        images = [[0] * 6 if i in fit else [0] for i in range(400)]
        perfect = [_Det([_box(j) for j in range(len(labels))], labels, [0.9] * len(labels)) for labels in images]
        out = _run(images, [_ap(threshold=0.3, k=1)], max_objects=6, predictions=perfect)
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["limitation"]) == ("unresolved", "conflict")
        assert row["envelope"] == pytest.approx(0.334, abs=1e-3)
        assert row["witness"] == pytest.approx(0.578, abs=1e-3)

    def test_capacity_counts_by_declared_id(self):
        # Class 'c' is label 7: two in each odd image, one in each even one. By position it would be label 2, which
        # no box has, and its bound would be 1.
        images = [[7, 7, 1] if i % 2 else [3, 7] for i in range(400)]
        out = _run(images, [_ap("c", k=1)], dataset=_OD(images, {1: "a", 3: "b", 7: "c"}), predictions=None)
        counts = np.array([labels.count(7) for labels in images])
        expected = capacity_from_counts(counts, max_objects=3, max_detections=1, alpha_claim=0.05)[0]
        bound = _capacity_row(out)
        assert (bound["bound"], bound["state"]) == (expected, "informative")

    def test_no_audit_targets_with_capacity(self):
        # Three persons in every fit image and none in the audit: the witness has nothing to find and reads 0, while
        # capacity, over every image at K = 1, is informative. Nothing can demonstrate the requirement.
        fit = _fit_side(400)
        images = [[0] * 3 if i in fit else [] for i in range(400)]
        out = _run(images, [_ap(threshold=0.2, k=1)])
        capacity, witness = out.bounds.iter_rows(named=True)
        assert (capacity["source"], capacity["state"]) == ("capacity", "informative")
        assert (witness["source"], witness["bound"]) == ("user_model", 0.0)
        assert (witness["observed"], witness["n"]) == (None, 400 - len(fit))  # no estimate, and still the audit's size
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["limitation"]) == ("not_ruled_out", "weak_witness")
        assert row["suggested_minimum"] is None

    def test_a_class_the_detector_never_predicts_reads_zero_without_a_minimum(self):
        # Persons are in the audit but the detector predicts only cars, so the witness's AP is 0.0 and not None. The
        # capacity envelope does not cross at K = 1 and B = 3 (its ratio 2/3 is above the 0.5 asked), so the witness's
        # estimate decides: it cannot clear, so no number of images would carry the bound across.
        images = _scenes(300)
        cars_only = [
            _Det(d.boxes[d.labels == 1], d.labels[d.labels == 1], d.scores[d.labels == 1]) for d in _detector(images)
        ]
        out = _run(images, [_ap("person", 0.5, k=1)], predictions=cars_only)
        witness = next(r for r in out.bounds.iter_rows(named=True) if r["role"] == "witness")
        assert witness["observed"] == 0.0
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["limitation"]) == ("not_ruled_out", "weak_witness")
        assert row["suggested_minimum"] is None

    def test_capacity_ignores_image_order(self):
        images, ids = _scenes(300), np.arange(300)
        order = np.random.default_rng(1).permutation(300)
        detections = _detector(images)
        given = _run(images, [_ap(k=1)], predictions=detections, groups=ids)
        reordered = _run(
            [images[i] for i in order],
            [_ap(k=1)],
            dataset=_OD([images[i] for i in order], ids=[f"im{i}" for i in order]),
            predictions=[detections[i] for i in order],
            groups=ids[order],
        )
        assert _capacity_row(reordered) == _capacity_row(given)
