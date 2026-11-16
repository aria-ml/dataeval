"""Blur reports why a row has no value: `blur_status` for the row, and a reason for directionality alone."""

import warnings
from typing import Any

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.flags import ImageStats

pytestmark = pytest.mark.required


def _checkerboard(size: int = 128, cell: int = 8) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    board = (((y // cell) + (x // cell)) % 2 * 200 + 20).astype(np.uint8)
    return np.stack([board] * 3)


def _stripes(size: int = 128, cell: int = 8) -> np.ndarray:
    """Vertical stripes: every edge runs down the image, so only one axis has edge points."""
    x = np.arange(size)
    row = ((x // cell) % 2 * 200 + 20).astype(np.uint8)
    return np.stack([np.tile(row, (size, 1))] * 3)


def run(images: list[np.ndarray], stats: Any = ImageStats.BLUR, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return image_stats(images, stats=stats, **kwargs)


def details(result: Any, name: str) -> list[Any]:
    return [v.item() if hasattr(v, "item") else v for v in result["details"][name]]


def test_a_measured_image_is_ok_with_values_and_no_reason():
    result = run([_checkerboard()])
    assert details(result, "blur_status") == ["ok"]
    assert np.isfinite(result["stats"]["blur_edge_width"][0])
    assert np.isfinite(result["stats"]["blur_edge_directionality"][0])
    assert details(result, "blur_edge_directionality_reason") == [""]


def test_a_flat_image_has_too_few_edges():
    result = run([np.full((3, 64, 64), 128, np.uint8)])
    assert details(result, "blur_status") == ["insufficient_support"]
    assert np.isnan(result["stats"]["blur_edge_width"][0])


def test_edges_on_one_axis_leave_directionality_alone_nan_with_its_reason():
    result = run([_stripes()])
    assert details(result, "blur_status") == ["ok"]
    assert np.isfinite(result["stats"]["blur_edge_width"][0])
    assert np.isnan(result["stats"]["blur_edge_directionality"][0])
    assert details(result, "blur_edge_directionality_reason") == ["insufficient_axis_support"]


def test_boxes_outside_or_of_zero_area_are_invalid_box_and_one_crossing_the_edge_is_insufficient_support():
    crossing = [-1.0, 0.0, 100.0, 100.0]  # one pixel past the left edge, so the fill outside the image is NaN
    result = run(
        [_checkerboard()],
        stats=ImageStats.BLUR | ImageStats.DIMENSION_INVALID_BOX,
        boxes=[[[300.0, 300.0, 340.0, 340.0], [20.0, 20.0, 20.0, 40.0], crossing]],
        per_image=False,
    )
    assert details(result, "blur_status") == ["invalid_box", "invalid_box", "insufficient_support"]
    assert [bool(v) for v in result["stats"]["invalid_box"]] == [True, True, False]


def test_an_unknown_range_and_a_missing_channel_have_their_own_statuses():
    floats = _checkerboard().astype(np.float32) * 2.0  # values above 255: no float convention applies
    assert details(run([floats]), "blur_status") == ["unknown_range"]
    grouped = run([_checkerboard()], stats={None: ImageStats.PIXEL_MEAN, "far": ImageStats.BLUR}, channels={"far": [7]})
    assert details(grouped, "far_blur_status") == ["missing_channel"]


def test_the_background_has_its_own_status_and_box_rows_read_not_applicable_there():
    result = run(
        [_checkerboard()], stats=ImageStats.VISUAL_BLUR_EDGE_WIDTH, boxes=[[[10, 10, 40, 40]]], per_background=True
    )
    assert details(result, "background_blur_status") == ["ok", "not_applicable"]


def test_ok_always_has_a_value():
    images = [_checkerboard(), _stripes(), np.full((3, 64, 64), 128, np.uint8)]
    result = run(images, boxes=[[[10, 10, 60, 60]], [], [[5, 5, 30, 30]]], per_background=True)
    for prefix in ("", "background_"):
        for status, value in zip(
            details(result, f"{prefix}blur_status"), result["stats"][f"{prefix}blur_edge_width"], strict=True
        ):
            assert (status == "ok") == bool(np.isfinite(value)), (prefix, status, value)


def test_a_request_without_blur_has_no_blur_details():
    assert "blur_status" not in run([_checkerboard()], stats=ImageStats.PIXEL_MEAN).get("details", {})
