"""Tests for the opinionated object-detection predictor."""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from dataeval.models import LiteRtObjectDetector, OnnxObjectDetector
from dataeval.models._predictors import _BaseObjectDetector
from dataeval.protocols import ObjectDetectionTarget


def _meta(tmp_path: Path, interface: str = "IMAGE_OBJECT_DETECTION") -> Path:
    p = tmp_path / "model-metadata.json"
    p.write_text(
        json.dumps({
            "interface": {"name": "JATIC_ONNX", "version": "v1"},
            "io": {
                "batchSize": -1,
                "interface": interface,
                "input": {"channels": "RGB", "height": 8, "width": 8},
                "output": {"nBoxes": 5, "nClasses": 4},
            },
        }),
        encoding="utf-8",
    )
    return p


def test_detector_returns_targets_per_image(onnx_detector: Path, tmp_path: Path):
    model = OnnxObjectDetector(onnx_detector, _meta(tmp_path))
    preds = model([np.zeros((3, 8, 8), dtype=np.uint8), np.full((3, 8, 8), 255, dtype=np.uint8)])
    assert len(preds) == 2
    t = preds[0]
    assert isinstance(t, ObjectDetectionTarget)
    assert np.asarray(t.boxes).shape == (5, 4)
    assert np.asarray(t.labels).shape == (5,)
    assert np.asarray(t.scores).shape == (5, 4)


def test_detector_boxes_are_normalized(onnx_detector: Path, tmp_path: Path):
    model = OnnxObjectDetector(onnx_detector, _meta(tmp_path))
    t = model([np.zeros((3, 8, 8), dtype=np.uint8)])[0]
    boxes = np.asarray(t.boxes)
    assert boxes.min() >= 0.0
    assert boxes.max() <= 1.0


def test_onnx_detector_is_a_base_object_detector(onnx_detector: Path, tmp_path: Path):
    model = OnnxObjectDetector(onnx_detector, _meta(tmp_path))
    assert isinstance(model, _BaseObjectDetector)


def test_base_object_detector_is_abstract(tmp_path: Path):
    with pytest.raises(TypeError):
        _BaseObjectDetector("model.onnx", _meta(tmp_path))  # type: ignore[abstract]


def test_wrong_task_metadata_raises(tmp_path: Path):
    meta = _meta(tmp_path, interface="IMAGE_CLASSIFICATION")
    with pytest.raises(ValueError, match="not IMAGE_OBJECT_DETECTION"):
        OnnxObjectDetector("unused.onnx", meta)


def test_missing_output_raises(stub_model_file, fake_onnxruntime, tmp_path: Path):
    # boxes but no scores -- the pair is checked before either is decoded
    fake_onnxruntime({"boxes": np.zeros((1, 5, 4), dtype=np.float32)})
    model = OnnxObjectDetector(stub_model_file("detector.onnx"), _meta(tmp_path))
    with pytest.raises(ValueError, match="not found"):
        model([np.zeros((3, 8, 8), dtype=np.uint8)])


def test_bad_boxes_shape_raises(stub_model_file, fake_onnxruntime, tmp_path: Path):
    # boxes last dim is 3, not 4
    fake_onnxruntime({
        "boxes": np.zeros((1, 5, 3), dtype=np.float32),
        "scores": np.zeros((1, 5, 4), dtype=np.float32),
    })
    model = OnnxObjectDetector(stub_model_file("detector.onnx"), _meta(tmp_path))
    with pytest.raises(ValueError, match=r"boxes must be \(B, nBoxes, 4\)"):
        model([np.zeros((3, 8, 8), dtype=np.uint8)])


def test_non_3d_scores_raises(stub_model_file, fake_onnxruntime, tmp_path: Path):
    fake_onnxruntime({
        "boxes": np.zeros((1, 5, 4), dtype=np.float32),
        "scores": np.zeros((1, 5), dtype=np.float32),  # 2-D
    })
    model = OnnxObjectDetector(stub_model_file("detector.onnx"), _meta(tmp_path))
    with pytest.raises(ValueError, match=r"scores must be \(B, nBoxes, nClasses\)"):
        model([np.zeros((3, 8, 8), dtype=np.uint8)])


def test_litert_detector_make_backend_loads_tflite(tmp_path: Path):
    # the LiteRT subclass reaches its backend, which rejects the missing model file
    with pytest.raises(FileNotFoundError):
        LiteRtObjectDetector(tmp_path / "missing.tflite", _meta(tmp_path))


def test_detector_runs_with_stubbed_runtime(stub_model_file, fake_onnxruntime, tmp_path: Path):
    """The full construct-preprocess-decode path, without needing a real ONNX runtime."""
    scores = np.tile(np.array([0.1, 0.2, 0.6, 0.1], dtype=np.float32), (5, 1))
    fake_onnxruntime({
        "boxes": lambda t: np.zeros((t.shape[0], 5, 4), dtype=np.float32),
        "scores": lambda t: np.tile(scores, (t.shape[0], 1, 1)),
    })

    model = OnnxObjectDetector(stub_model_file("detector.onnx"), _meta(tmp_path))
    assert model.metadata["id"] == "dataeval-onnx-detector:detector.onnx"

    preds = model([np.zeros((3, 8, 8), dtype=np.uint8), np.full((3, 8, 8), 255, dtype=np.uint8)])
    assert len(preds) == 2
    t = preds[0]
    assert isinstance(t, ObjectDetectionTarget)
    assert np.asarray(t.boxes).shape == (5, 4)
    assert np.asarray(t.scores).shape == (5, 4)
    # labels are the argmax over each detection's class scores
    np.testing.assert_array_equal(np.asarray(t.labels), np.full(5, 2))


def test_detector_honors_image_size_override(stub_model_file, fake_onnxruntime, tmp_path: Path):
    """An explicit image_size resizes the batch instead of the size declared in metadata."""
    session = fake_onnxruntime({
        "boxes": lambda t: np.zeros((t.shape[0], 5, 4), dtype=np.float32),
        "scores": lambda t: np.zeros((t.shape[0], 5, 4), dtype=np.float32),
    })

    model = OnnxObjectDetector(stub_model_file("detector.onnx"), _meta(tmp_path), image_size=(4, 4))
    model([np.zeros((3, 8, 8), dtype=np.uint8)])

    assert session.last_feed is not None
    assert session.last_feed["image"].shape == (1, 3, 4, 4)


# -- box_coordinates (D10) ---------------------------------------------------------------------------------------------

_E10 = "box_coordinates must be 'normalized' or 'pixel', got 'pixels'."


def _normalized_boxes() -> NDArray[np.float32]:
    """Known normalized (x0, y0, x1, y1) boxes for a batch of two images, five boxes each."""
    return np.random.default_rng(7).uniform(0.0, 1.0, size=(2, 5, 4)).astype(np.float32)


def _two_images() -> list[Any]:
    """A (3, 8, 16) image and a (3, 20, 10) image: sizes (H, W) of (8, 16) and (20, 10)."""
    return [np.zeros((3, 8, 16), dtype=np.uint8), np.zeros((3, 20, 10), dtype=np.uint8)]


_SIZES_WH = [(16, 8), (10, 20)]


def _batch_form(images: list[Any], form: str) -> list[Any]:
    if form == "datum":
        return [(image, {}, {}) for image in images]
    if form == "nested":
        return [image.tolist() for image in images]
    return images


@pytest.mark.parametrize("image_size", [None, (4, 4)])
@pytest.mark.parametrize("form", ["array", "datum", "nested"])
def test_pixel_boxes_are_each_images_normalized_boxes_times_its_size(
    stub_model_file, fake_onnxruntime, tmp_path: Path, form: str, image_size: tuple[int, int] | None
):
    normalized = _normalized_boxes()
    fake_onnxruntime({
        "boxes": lambda t: normalized[: t.shape[0]],
        "scores": lambda t: np.zeros((t.shape[0], 5, 4), dtype=np.float32),
    })
    model = OnnxObjectDetector(
        stub_model_file("detector.onnx"), _meta(tmp_path), image_size=image_size, box_coordinates="pixel"
    )

    preds = model(_batch_form(_two_images(), form))

    for i, (w, h) in enumerate(_SIZES_WH):
        boxes = np.asarray(preds[i].boxes)
        assert boxes.dtype == np.float32
        np.testing.assert_allclose(boxes, normalized[i].astype(np.float64) * [w, h, w, h], rtol=1e-6)


def test_pixel_boxes_recover_a_rectangle_through_the_stretch(stub_model_file, fake_onnxruntime, tmp_path: Path):
    """A bright rectangle at a known pixel box comes back at that box. A letterbox would shift y by 5."""
    image = np.zeros((3, 20, 40), dtype=np.float32)
    image[:, 5:15, 10:30] = 1.0  # rows 5..14, columns 10..29 -> pixel box [10, 5, 30, 15]

    def boxes(tensor: NDArray[np.float32]) -> NDArray[np.float32]:
        side = tensor.shape[-1]  # the resized tensor is square, (B, 3, 64, 64)
        out = np.zeros((tensor.shape[0], 1, 4), dtype=np.float32)
        for b in range(tensor.shape[0]):
            rows, cols = np.nonzero(tensor[b, 0] > 0.5)
            out[b, 0] = [cols.min() / side, rows.min() / side, (cols.max() + 1) / side, (rows.max() + 1) / side]
        return out

    fake_onnxruntime({"boxes": boxes, "scores": lambda t: np.zeros((t.shape[0], 1, 4), dtype=np.float32)})
    model = OnnxObjectDetector(
        stub_model_file("detector.onnx"), _meta(tmp_path), image_size=(64, 64), box_coordinates="pixel"
    )

    box = np.asarray(model([image])[0].boxes)[0]

    np.testing.assert_allclose(box, [10, 5, 30, 15], atol=1.0)


@pytest.mark.parametrize("kwargs", [{}, {"box_coordinates": "normalized"}])
def test_default_and_normalized_boxes_are_unchanged(stub_model_file, fake_onnxruntime, tmp_path: Path, kwargs):
    normalized = _normalized_boxes()
    fake_onnxruntime({
        "boxes": lambda t: normalized[: t.shape[0]],
        "scores": lambda t: np.zeros((t.shape[0], 5, 4), dtype=np.float32),
    })
    model = OnnxObjectDetector(stub_model_file("detector.onnx"), _meta(tmp_path), **kwargs)

    preds = model(_two_images())

    assert model.box_coordinates == "normalized"
    for i in range(2):
        assert np.asarray(preds[i].boxes).dtype == np.float32
        np.testing.assert_array_equal(preds[i].boxes, normalized[i])


def test_box_coordinates_must_be_normalized_or_pixel(tmp_path: Path):
    with pytest.raises(ValueError, match=re.escape(_E10)):
        OnnxObjectDetector("unused.onnx", _meta(tmp_path), box_coordinates="pixels")  # type: ignore[arg-type]


def test_outputs_need_one_row_per_image_and_matching_nboxes(stub_model_file, fake_onnxruntime, tmp_path: Path):
    image = [np.zeros((3, 8, 8), dtype=np.uint8)]
    # boxes for two images on a one-image batch
    fake_onnxruntime({
        "boxes": np.zeros((2, 5, 4), dtype=np.float32),
        "scores": np.zeros((2, 5, 4), dtype=np.float32),
    })
    model = OnnxObjectDetector(stub_model_file("detector.onnx"), _meta(tmp_path))
    expected = (
        "detection outputs must have one row per input image (1) and the same nBoxes; "
        "got boxes (2, 5, 4) and scores (2, 5, 4)"
    )
    with pytest.raises(ValueError, match=re.escape(expected)):
        model(image)

    # scores with 6 boxes against boxes with 5 (the stub is installed before the model that loads it)
    fake_onnxruntime({
        "boxes": np.zeros((1, 5, 4), dtype=np.float32),
        "scores": np.zeros((1, 6, 4), dtype=np.float32),
    })
    model = OnnxObjectDetector(stub_model_file("detector.onnx"), _meta(tmp_path))
    expected = (
        "detection outputs must have one row per input image (1) and the same nBoxes; "
        "got boxes (1, 5, 4) and scores (1, 6, 4)"
    )
    with pytest.raises(ValueError, match=re.escape(expected)):
        model(image)


def test_scaling_lives_in_the_base_class(tmp_path: Path):
    """A subclass with its own backend gives the same pixel boxes, which covers LiteRT without a LiteRT runtime."""
    normalized = _normalized_boxes()

    class _Backend:
        input_name = "image"
        output_names = ["boxes", "scores"]

        def run(self, tensor: NDArray[Any]) -> dict[str, NDArray[Any]]:
            return {"boxes": normalized[: tensor.shape[0]], "scores": np.zeros((tensor.shape[0], 5, 4))}

    class _Detector(_BaseObjectDetector):
        _backend_label = "fake"

        def _make_backend(self, model_path: str | Path) -> _Backend:
            return _Backend()

    model = _Detector("unused.model", _meta(tmp_path), box_coordinates="pixel")
    preds = model(_two_images())

    for i, (w, h) in enumerate(_SIZES_WH):
        np.testing.assert_allclose(
            np.asarray(preds[i].boxes), normalized[i].astype(np.float64) * [w, h, w, h], rtol=1e-6
        )
