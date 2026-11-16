"""Every row carries every name the call produces, even where one image's rows cannot (#1660)."""

import warnings
from typing import Any

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.flags import ImageStats

pytestmark = pytest.mark.required


def _images(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (3, 32, 32), dtype=np.uint8) for _ in range(n)]


def run(boxes: list[list[Any]], **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return image_stats(_images(len(boxes)), boxes=boxes, stats=ImageStats.PIXEL_MEAN, **kwargs)


@pytest.mark.parametrize("order", ["box first", "boxless first"])
def test_an_image_with_no_box_measures_its_background_alone(order):
    boxes = [[[4, 4, 20, 20]], []] if order == "box first" else [[], [[4, 4, 20, 20]]]
    result = run(boxes, per_image=False, per_background=True)
    rows = len(result["source_index"])
    assert rows == 3  # both background rows and the one box row
    for name in ("mean", "background_mean", "background_fraction"):
        assert len(result["stats"][name]) == rows
    by_row = {(s.item, s.key): i for i, s in enumerate(result["source_index"])}
    boxless = 1 if order == "box first" else 0
    # The boxless image's own row is a background row, so it has no box statistic and its whole background.
    assert np.isnan(result["stats"]["mean"][by_row[(boxless, None)]])
    assert result["stats"]["background_fraction"][by_row[(boxless, None)]] == 1.0


def test_the_rows_an_image_can_measure_keep_their_values():
    with_boxless = run([[[4, 4, 20, 20]], []], per_image=False, per_background=True)
    alone = run([[[4, 4, 20, 20]]], per_image=False, per_background=True)
    for name in alone["stats"]:
        np.testing.assert_array_equal(with_boxless["stats"][name][:2], alone["stats"][name])
