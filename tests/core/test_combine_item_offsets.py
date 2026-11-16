"""`combine_stats_results` shifts each result's items past the previous result's, never onto them (#1661)."""

import warnings
from typing import Any

import numpy as np
import pytest

from dataeval.core import combine_stats_results, image_stats
from dataeval.flags import ImageStats
from dataeval.quality._shared import get_dataset_step_from_idx

pytestmark = pytest.mark.required

IMAGES = [np.random.default_rng(i).integers(0, 256, (3, 32, 32), dtype=np.uint8) for i in range(3)]


def stats(boxes: list[list[Any]] | None, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return image_stats(IMAGES, boxes=boxes, stats=ImageStats.PIXEL_MEAN, **kwargs)


CASES = {
    "box rows only, an image without boxes": (
        {"boxes": [[[2, 2, 10, 10]], [], [[4, 4, 12, 12]]], "per_image": False},
        {"boxes": [[[2, 2, 10, 10]], [[1, 1, 9, 9]], []], "per_image": False},
    ),
    "image and box rows": (
        {"boxes": [[[2, 2, 10, 10], [5, 5, 9, 9]], [], [[4, 4, 12, 12]]]},
        {"boxes": [[[2, 2, 10, 10]], [[1, 1, 9, 9]], []]},
    ),
    "image rows only": ({"boxes": None}, {"boxes": None}),
}


@pytest.mark.parametrize("case", CASES)
def test_every_combined_row_maps_back_to_its_own_dataset_and_item(case):
    results = [stats(**kwargs) for kwargs in CASES[case]]
    _, index, steps = combine_stats_results(results)
    expected = [(d, s.item, s.key) for d, r in enumerate(results) for s in r["source_index"]]
    mapped = [(*get_dataset_step_from_idx(s.item, steps), s.key) for s in index]
    assert mapped == expected


def test_no_two_datasets_share_an_item():
    a, b = (stats(**kwargs) for kwargs in CASES["box rows only, an image without boxes"])
    _, index, steps = combine_stats_results([a, b])
    first, second = {s.item for s in index[: len(a["source_index"])]}, {s.item for s in index[len(a["source_index"]) :]}
    assert not first & second
    assert steps == [3, 5]  # the first result's items run to 2, the second's to 1


def test_image_rows_only_are_unchanged():
    a, b = (stats(**kwargs) for kwargs in CASES["image rows only"])
    _, index, steps = combine_stats_results([a, b])
    assert [s.item for s in index] == [0, 1, 2, 3, 4, 5]
    assert steps == [3, 6]


def test_an_empty_result_keeps_its_boundary():
    a = stats(boxes=None)
    empty: Any = {**a, "stats": {k: v[:0] for k, v in a["stats"].items()}, "source_index": []}
    _, index, steps = combine_stats_results([a, empty, a])
    assert steps == [3, 3, 6]
    assert [s.item for s in index] == [0, 1, 2, 3, 4, 5]
