"""Feasibility runs a MAITE object detector given as ``model=`` and judges it exactly as its own detections."""

import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
import torch
from polars.testing import assert_frame_equal

from dataeval.config import set_batch_size
from dataeval.models._predictors import _BaseObjectDetector
from dataeval.performance import Feasibility
from dataeval.performance._feasibility_detection import read_model_output
from dataeval.performance._feasibility_model import run_model
from tests.models.test_predictors_od import _meta
from tests.performance.test_feasibility_detection_route import (
    _OD,
    NAMES,
    _ap,
    _box,
    _Det,
    _detector,
    _no_pass,
    _run,
    _scenes,
)

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

THREE = {0: "person", 1: "car", 2: "truck"}
WRAP = " Wrap the model so that its output passes this check."
BOTH = "model= and predictions= are both given; pass one, since a call has one detector"
CALL = (
    "Object detection takes the targets, each requirement's max_detections K and Feasibility(max_objects=...) for the "
    "capacity envelope, and a detector for the witness: predictions=, one detections object per image, or model=, a "
    "MAITE object detector."
)


class _IndexedOD(_OD):
    """The route fixture with image ``i`` filled with ``i``, so a detector double can tell which image it is given."""

    shape = (3, 2, 2)

    def __getitem__(self, i: int):
        _, target, meta = super().__getitem__(i)
        return np.full(self.shape, i, dtype=np.float32), target, meta


class _DetectorModel:
    """A MAITE-shaped detector double: image ``i`` gets ``detections[i]``. It records each batch's image indices."""

    def __init__(self, detections: list[Any], index2label: dict[int, str] | None = None) -> None:
        self.detections, self.batches = detections, []
        self.metadata: dict[str, Any] = {"id": "double"} | ({} if index2label is None else {"index2label": index2label})

    def __call__(self, batch: Any) -> Any:
        indices = [int(image[0, 0, 0]) for image in batch]
        self.batches.append(indices)
        return [self.detections[i] for i in indices]


class _TorchDetector(_DetectorModel):
    """The double over torch parameters: under autograd its boxes and scores require grad. It records whether it is."""

    def __init__(self, detections: list[Any], device: str = "cpu") -> None:
        def param(values: Any) -> torch.nn.Parameter:
            return torch.nn.Parameter(torch.as_tensor(np.asarray(values), device=device))

        super().__init__([_Det(param(d.boxes), d.labels, param(d.scores)) for d in detections])
        self.grad_enabled: list[bool] = []

    def __call__(self, batch: Any) -> Any:
        self.grad_enabled.append(torch.is_grad_enabled())
        return [_Det(d.boxes * 1, d.labels, d.scores * 1) for d in super().__call__(batch)]


class _ShortSecond(_DetectorModel):
    """The double with one output too few for its second batch."""

    def __call__(self, batch: Any) -> Any:
        out = super().__call__(batch)
        return out[:-1] if len(self.batches) == 2 else out


class _ReusingBuffer(_DetectorModel):
    """The double writing every batch's float64 boxes and 1-D float64 scores into one array it keeps, as views."""

    def __init__(self, detections: list[Any]) -> None:
        super().__init__(detections)
        most = max(len(d.labels) for d in detections)
        self.boxes, self.scores = np.zeros((64, most, 4)), np.zeros((64, most))

    def __call__(self, batch: Any) -> Any:
        out = []
        for k, d in enumerate(super().__call__(batch)):
            n = len(d.labels)
            self.boxes[k, :n], self.scores[k, :n] = d.boxes, d.scores
            out.append(_Det(self.boxes[k, :n], d.labels, self.scores[k, :n]))
        return out


class _Refuses:
    """A detector that fails the test if it is called."""

    metadata: dict[str, Any] = {"id": "refuses"}

    def __call__(self, batch: Any) -> Any:
        pytest.fail("the detector ran")


def _raw(images: list[list[int]]) -> list[_Det]:
    """
    ``_detector``'s detections, extended with every case the witness reads; image 0 has none.

    Each image after image 0 also gets a duplicate of each target found, at 0.5, which is a false positive; a ``truck``
    detection at 0.95, a class no requirement names; and, every third image, a person at 0.5 in the empty slot 7, which
    ties the duplicates at the K = 4 boundary where both persons are found. Every fourth image from image 1 gives
    ``(N, 3)`` per-class scores, and every fourth from image 3 float32 torch tensors.
    """
    out = []
    for i, d in enumerate(_detector(images)):
        boxes, labels, scores = d.boxes.tolist(), d.labels.tolist(), d.scores.tolist()
        found = [(box, c) for box, c in zip(boxes, labels, strict=True) if box[0] < 180]  # slot 9 holds no target
        if i:
            boxes += [box for box, _ in found] + [_box(0)]
            labels += [c for _, c in found] + [2]
            scores += [0.5] * len(found) + [0.95]
            if i % 3 == 2:
                boxes, labels, scores = [*boxes, _box(7)], [*labels, 0], [*scores, 0.5]
        b = np.array(boxes, dtype=np.float64).reshape(-1, 4)
        c = np.array(labels, dtype=np.intp)
        s: Any = np.array(scores, dtype=np.float64)
        if i % 4 == 1:
            s = np.full((len(c), 3), 0.01)
            s[np.arange(len(c)), c] = scores
        if i % 4 == 3:
            float32 = torch.float32
            out.append(_Det(torch.tensor(b, dtype=float32), torch.tensor(c), torch.tensor(s, dtype=float32)))
        else:
            out.append(_Det(b, c, s))
    return out


def _three(**kwargs: Any) -> Any:
    """Person at K = 1 (below B = 2) and K = 4, and car at K = 3 (B = 1), over the 60 images of three classes."""
    images = _scenes()
    target = [_ap("person", k=1), _ap("person", k=4), _ap("car", k=3)]
    return _run(images, target, max_objects={"person": 2, "car": 1}, dataset=_IndexedOD(images, THREE), **kwargs)


def _witness(out: Any) -> pl.DataFrame:
    return out.bounds.filter(pl.col("source") == "user_model")


@pytest.mark.required
class TestEquality:
    def test_a_detector_gives_the_determinations_bounds_and_printout_of_its_own_detections(self):
        detections = _raw(_scenes())
        by_model = _three(predictions=None, model=_DetectorModel(detections))
        by_predictions = _three(predictions=detections)
        bounds = _witness(by_model)["bound"].to_list()
        assert len(bounds) == 3
        assert all(bound > 0 for bound in bounds)  # every witness informative, so a wrong detection would show
        assert_frame_equal(by_model.determinations, by_predictions.determinations)
        assert_frame_equal(by_model.bounds, by_predictions.bounds)
        assert str(by_model) == str(by_predictions)

    def test_outputs_shifted_by_one_image_give_other_bounds(self):
        detections = _raw(_scenes())
        with pytest.warns(UserWarning, match="None of the 44 kept detections of class 'car'"):  # each one image off
            shifted = _three(predictions=None, model=_DetectorModel(detections[1:] + detections[:1]))
        assert _witness(shifted)["bound"].to_list() != _witness(_three(predictions=detections))["bound"].to_list()


@pytest.mark.required
class TestBatches:
    def test_batches_of_batch_size_read_every_image_once_in_order(self):
        images = _scenes(10)
        model = cast(Any, _DetectorModel(_detector(images)))
        Feasibility(seed=0, max_objects=3, batch_size=3).evaluate(_IndexedOD(images), target=[_ap()], model=model)
        assert [len(batch) for batch in model.batches] == [3, 3, 3, 1]
        assert [i for batch in model.batches for i in batch] == list(range(10))

    def test_the_global_batch_size_is_used_when_none_is_given(self):
        images = _scenes(10)
        model = cast(Any, _DetectorModel(_detector(images)))
        set_batch_size(4)
        try:
            Feasibility(seed=0, max_objects=3).evaluate(_IndexedOD(images), target=[_ap()], model=model)
        finally:
            set_batch_size(16)  # the suite's batch size (tests/conftest.py)
        assert [len(batch) for batch in model.batches] == [4, 4, 2]

    def test_without_a_batch_size_the_call_is_refused_before_the_pass(self):
        model = _DetectorModel(_detector(_scenes()))
        set_batch_size(None)
        try:
            with _no_pass(), pytest.raises(ValueError, match=r"Feasibility\(batch_size=\.\.\.\) or call"):
                _run(dataset=_IndexedOD(_scenes()), predictions=None, model=model)
        finally:
            set_batch_size(16)
        assert model.batches == []


@pytest.mark.required
class TestAutograd:
    def test_a_torch_detector_runs_without_autograd_and_gives_the_same_bounds(self):
        detections = _raw(_scenes())
        model = _TorchDetector(detections)
        out = _three(predictions=None, model=model)
        assert model.grad_enabled
        assert not any(model.grad_enabled)
        assert_frame_equal(out.bounds, _three(predictions=detections).bounds)

    def test_autograd_is_on_after_a_refused_call(self):
        flipped = [_Det([[5.0, 0.0, 1.0, 1.0]], [0], [0.9])] * len(_scenes())
        with pytest.raises(ValueError, match="nonnegative width") as info:
            _run(dataset=_IndexedOD(_scenes()), predictions=None, model=_TorchDetector(flipped))
        assert info.value is not None  # the exception, and its traceback, are still held
        assert torch.is_grad_enabled()

    @pytest.mark.cuda
    def test_outputs_on_the_gpu_give_the_same_bounds(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        detections = _raw(_scenes())
        out = _three(predictions=None, model=_TorchDetector(detections, device="cuda"))
        assert_frame_equal(out.bounds, _three(predictions=detections).bounds)


@pytest.mark.required
class TestClassOrder:
    def test_an_equal_index2label_leaves_the_assumption_off_the_witness_only(self):
        detections = _raw(_scenes())
        verified = _three(predictions=None, model=_DetectorModel(detections, THREE))
        assumed = _three(predictions=detections)
        for assumptions in _witness(verified)["assumptions"].to_list():
            assert assumptions == ["model_independent_of_audit_items", "max_objects_holds"]
        assert_frame_equal(_witness(verified).drop("assumptions"), _witness(assumed).drop("assumptions"))
        capacity = pl.col("source") == "capacity"
        assert_frame_equal(verified.bounds.filter(capacity), assumed.bounds.filter(capacity))

    def test_a_different_index2label_is_refused_before_the_pass(self):
        swapped = {0: "car", 1: "person", 2: "truck"}
        model = _DetectorModel(_raw(_scenes()), swapped)
        message = (
            f"The model's index2label {swapped} differs from the dataset's {THREE}, and DataEval never remaps classes. "
            "Use a model whose labels follow the dataset's class order, wrap it to map them, or relabel the dataset to "
            "the model's."
        )
        with _no_pass(), pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            _three(predictions=None, model=model)
        assert model.batches == []

    def test_a_detector_without_index2label_keeps_the_assumption(self):
        out = _three(predictions=None, model=_DetectorModel(_raw(_scenes())))
        for assumptions in _witness(out)["assumptions"].to_list():
            assert "class_order_matches_dataset" in assumptions


@pytest.mark.required
class TestRefusals:
    def test_a_model_with_predictions_is_one_error_with_the_other_unusable_inputs(self):
        message = f"embeddings= applies to classification only; {BOTH}. {CALL}"
        with _no_pass(), pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            _run(model=_Refuses(), embeddings=np.zeros((60, 2)))

    def test_a_short_second_batch_is_refused_naming_its_first_image(self):
        images = _scenes()
        message = (
            "The model returned 2 outputs for the batch of 3 images starting at image 3. Return one output per image "
            "of the batch, in order."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            Feasibility(seed=0, max_objects=3, batch_size=3).evaluate(
                _IndexedOD(images), target=[_ap()], model=cast(Any, _ShortSecond(_detector(images)))
            )

    def test_a_mapping_batch_output_is_refused(self):
        class Mapping(_DetectorModel):
            def __call__(self, batch: Any) -> Any:
                return {"detections": super().__call__(batch)}

        images = _scenes()
        message = (
            "The model returned a dict for the batch of 3 images starting at image 0. Return one output per image of "
            "the batch, in order."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            Feasibility(seed=0, max_objects=3, batch_size=3).evaluate(
                _IndexedOD(images), target=[_ap()], model=cast(Any, Mapping(_detector(images)))
            )

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            (
                vars,
                (
                    "model output for image 0 is a mapping; each image's detections need attributes boxes, labels and "
                    "scores (the MAITE ObjectDetectionTarget shape), not keys. Wrap each dict, e.g. "
                    "types.SimpleNamespace(**d)."
                ),
            ),
            (
                lambda d: _Det([[5.0, 0.0, 1.0, 1.0]], [0], [0.9]),
                (
                    "model output for image 0: box 0 is [5.0, 0.0, 1.0, 1.0], with x1 < x0 or y1 < y0. Boxes need "
                    "nonnegative width and height, in (x0, y0, x1, y1) order; fix or drop it."
                ),
            ),
            (
                lambda d: _Det([_box(0)], [3], [0.9]),
                (
                    "model output for image 0: label 3 is outside the classes the dataset declares, [0, 2) (without "
                    "index2label, the labels its targets hold). Map the detector's labels onto the dataset's class "
                    "indices (a detector that reserves label 0 for background, as torchvision's do, can give class i "
                    "the label i + 1), or drop detections of classes the dataset does not declare."
                ),
            ),
            (
                lambda d: _Det([[np.nan, 0.0, 10.0, 10.0]], [0], [0.9]),
                "model output for image 0: box 0 is [nan, 0.0, 10.0, 10.0], not finite. Fix or drop it.",
            ),
        ],
        ids=["a-dict-per-image", "a-negative-width-box", "label-3", "a-nan-box"],
    )
    def test_a_bad_image_output_names_the_image_and_says_to_wrap_the_model(self, change, message):
        images = _scenes()
        model = _DetectorModel([change(d) for d in _detector(images)])
        with pytest.raises(ValueError, match=f"^{re.escape(message + WRAP)}$"):
            _run(images, dataset=_IndexedOD(images), predictions=None, model=model)

    def test_without_a_detector_a_cap_that_cannot_bind_is_refused(self):
        message = (
            "No detector is given (predictions= or model=), and every requirement whose max_detections K is at or "
            "above max_objects needs one, since its cap cannot bind and no capacity envelope is built: 'person' at "
            f"K = 5, max_objects 3 (requirement 0). Pass predictions= or model=, or state a max_detections below "
            f"max_objects. {CALL}"
        )
        with _no_pass(), pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            _run(predictions=None)

    def test_data_above_max_objects_are_refused_before_the_detector_runs(self):
        with pytest.raises(ValueError, match=r"^Some images hold more targets of a class than Feasibility"):
            _run(dataset=_IndexedOD(_scenes()), max_objects=1, predictions=None, model=_Refuses())


@pytest.mark.required
class TestOutputs:
    def test_detection_boxes_with_no_area_count_the_same_through_model(self):
        empty = [[0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 3.0, 3.0], [5.0, 0.0, 5.0, 10.0]]
        padded = [
            _Det(np.r_[d.boxes, empty], np.r_[d.labels, [0] * 3], np.r_[d.scores, [0.95] * 3]) if i % 3 == 0 else d
            for i, d in enumerate(_detector(_scenes()))
        ]
        by_model = _run(dataset=_IndexedOD(_scenes()), predictions=None, model=_DetectorModel(padded))
        by_predictions = _run(dataset=_IndexedOD(_scenes()), predictions=padded)
        assert by_model.bounds["bound"][0] > 0
        assert_frame_equal(by_model.bounds, by_predictions.bounds)

    def test_reordering_the_images_changes_nothing_with_groups(self):
        images, ids = _scenes(), np.arange(60)
        order = np.random.default_rng(1).permutation(60)
        detections = _detector(images)
        given = _run(images, dataset=_IndexedOD(images), predictions=None, model=_DetectorModel(detections), groups=ids)
        reordered = _run(
            [images[i] for i in order],
            dataset=_IndexedOD([images[i] for i in order], ids=[f"im{i}" for i in order]),
            predictions=None,
            model=_DetectorModel([detections[i] for i in order]),  # image j of the reordered dataset is image order[j]
            groups=ids[order],
        )
        assert given.bounds["bound"][0] > 0
        assert reordered.determinations.equals(given.determinations)
        assert reordered.bounds["bound"].to_list() == pytest.approx(given.bounds["bound"].to_list(), rel=1e-12)

    def test_a_reused_output_buffer_gives_the_result_of_fresh_arrays(self):
        detections = _detector(_scenes())  # float64 boxes and 1-D float64 scores, which checking alone never copies
        reused = _run(dataset=_IndexedOD(_scenes()), predictions=None, model=_ReusingBuffer(detections))
        fresh = _run(dataset=_IndexedOD(_scenes()), predictions=None, model=_DetectorModel(detections))
        assert fresh.bounds["bound"][0] > 0
        assert_frame_equal(reused.determinations, fresh.determinations)
        assert_frame_equal(reused.bounds, fresh.bounds)
        # The bounds alone cannot see a lost copy of the scores, so compare each image's detections with its source.
        kept = run_model(
            cast(Any, _ReusingBuffer(detections)),
            cast(Any, _IndexedOD(_scenes())),
            16,
            lambda i, output: read_model_output(i, output, 2),
        )
        for got, want in zip(kept, detections, strict=True):
            np.testing.assert_array_equal(got.boxes, want.boxes)
            np.testing.assert_array_equal(got.scores, want.scores)


class _FramedOD(_IndexedOD):
    """The route fixture with each image as wide and tall as the slots its targets lie in, 200 by 10 pixels."""

    shape = (3, 10, 200)


class _Export:
    """
    A stub export: it reads each image's index from the resized input and emits that image's ``_detector`` boxes,
    normalized to the 200 by 10 frame, with ``(B, nBoxes, n_classes)`` scores, by default for the dataset's two
    classes. Rows past an image's detections are padded with boxes of no area, scored 0.
    """

    input_name = "image"
    output_names = ["boxes", "scores"]

    def __init__(self, detections: list[_Det], n_classes: int = len(NAMES)) -> None:
        self.detections, self.n_boxes, self.n_classes = detections, max(len(d.labels) for d in detections), n_classes

    def run(self, tensor: Any) -> dict[str, Any]:
        boxes = np.zeros((len(tensor), self.n_boxes, 4), dtype=np.float32)
        scores = np.zeros((len(tensor), self.n_boxes, self.n_classes), dtype=np.float32)
        for k, image in enumerate(tensor):
            d = self.detections[round(float(image[0, 0, 0]))]  # a constant image stays constant through the resize
            n = len(d.labels)
            boxes[k, :n] = d.boxes / [200, 10, 200, 10]
            scores[k, np.arange(n), d.labels] = d.scores  # so each row's argmax is its label
        return {"boxes": boxes, "scores": scores}


class _StubDetector(_BaseObjectDetector):
    """DataEval's object detector over the stub export; the scaling and the input building are the shipped code."""

    _backend_label = "stub"

    def __init__(self, detections: list[_Det], metadata_path: Path, n_classes: int = 2, **kwargs: Any) -> None:
        self._export = _Export(detections, n_classes)
        super().__init__("stub.onnx", metadata_path, **kwargs)

    def _make_backend(self, model_path: str | Path) -> Any:
        return self._export


@pytest.mark.required
class TestDataEvalDetector:
    """The metadata's class count is not checked, so ``_meta``'s 4 classes leave the stub's 2 classes as they are."""

    def test_pixel_boxes_give_informative_witnesses_equal_to_its_own_output(self, tmp_path: Path):
        images = _scenes()
        dataset = _FramedOD(images)
        detector = _StubDetector(_detector(images), _meta(tmp_path), box_coordinates="pixel")
        call = {"target": [_ap("person"), _ap("car", k=3)], "max_objects": {"person": 3, "car": 1}, "dataset": dataset}
        by_model = _run(images, predictions=None, model=detector, **call)
        own = detector([dataset[i][0] for i in range(len(dataset))])
        by_predictions = _run(images, predictions=own, **call)
        assert by_model.bounds["state"].to_list() == ["informative", "informative"]
        assert_frame_equal(by_model.bounds, by_predictions.bounds)
        assert_frame_equal(by_model.determinations, by_predictions.determinations)

    def test_normalized_boxes_give_an_empty_witness_and_the_warning_names_the_frame(self, tmp_path: Path):
        images = _scenes()
        detector = _StubDetector(_detector(images), _meta(tmp_path))
        frame = "in the targets' coordinates, not a resized or normalized input's"
        with pytest.warns(UserWarning, match=re.escape(frame)):
            out = _run(images, dataset=_FramedOD(images), predictions=None, model=detector)
        assert out.bounds["bound"].to_list() == [0.0]

    def test_an_export_with_another_class_count_is_refused(self, tmp_path: Path):
        images = _scenes()
        detector = _StubDetector(_detector(images), _meta(tmp_path), n_classes=4, box_coordinates="pixel")
        message = "model output for image 0: scores of shape (4, 4) for 4 boxes; give (4,) or (4, 2)."
        with pytest.raises(ValueError, match=f"^{re.escape(message + WRAP)}$"):
            _run(images, dataset=_FramedOD(images), predictions=None, model=detector)
