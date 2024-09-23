import numpy as np
import pytest
from numpy.random import randint

from dataeval._internal.metrics.stats import (
    BaseStatsOutput,
    dimensionstats,
    hashstats,
    pixelstats,
    visualstats,
)


def get_dataset(count: int, channels: int):
    return [np.random.random((channels, 64, 64)) for _ in range(count)]


def get_bboxes(count: int, boxes_per_image: int):
    return [np.array([randint(1, 8, (4,)) for _ in range(boxes_per_image)]) for _ in range(count)]


DATA_1 = get_dataset(10, 1)
DATA_3 = get_dataset(10, 3)


class TestStats:
    @pytest.mark.parametrize(
        "stats, data, per_channel, attribute, length",
        [
            [hashstats, DATA_1, None, "xxhash", 10],
            [hashstats, DATA_3, None, "pchash", 10],
            [dimensionstats, DATA_1, None, "width", 10],
            [dimensionstats, DATA_3, None, "height", 10],
            [dimensionstats, DATA_1 + DATA_3, None, "channels", 10 + 10],
            [pixelstats, DATA_1, False, "mean", 10],
            [pixelstats, DATA_1, True, "std", 10],
            [pixelstats, DATA_3, False, "var", 10],
            [pixelstats, DATA_3, True, "skew", 10 * 3],
            [pixelstats, DATA_1 + DATA_3, False, "percentiles", 10 + 10],
            [pixelstats, DATA_1 + DATA_3, True, "histogram", 10 * 3 + 10],
            [visualstats, DATA_1, False, "brightness", 10],
            [visualstats, DATA_1, True, "blurriness", 10],
            [visualstats, DATA_3, False, "contrast", 10],
            [visualstats, DATA_3, True, "darkness", 10 * 3],
            [visualstats, DATA_1 + DATA_3, False, "missing", 10 + 10],
            [visualstats, DATA_1 + DATA_3, True, "zeros", 10 * 3 + 10],
        ],
    )
    def test_stats(self, stats, data, per_channel, attribute, length):
        results = stats(data) if per_channel is None else stats(data, per_channel=per_channel)
        assert len(getattr(results, attribute)) == length
        if hasattr(results, "index_map") and results.index_map is not None:
            assert len(results.index_map) == length

    def test_stats_with_bboxes(self):
        boxes = get_bboxes(10, 4)
        results = pixelstats(DATA_1, boxes, True)
        assert len(results) == 10 * 4

    def test_array_stats_with_bboxes_per_channel_true(self):
        boxes = get_bboxes(10, 4)
        results = pixelstats(DATA_3, boxes, True)
        assert len(results) == 10 * 4 * 3

    def test_array_stats_with_bboxes_per_channel_false(self):
        boxes = get_bboxes(10, 4)
        results = pixelstats(DATA_3, boxes, False)
        assert len(results) == 10 * 4

    def test_stats_no_boxes_channel_mask_max_channels_3(self):
        results = pixelstats(DATA_3 + DATA_1, per_channel=True)
        mask = results.get_channel_mask(0, 3)
        assert len(mask) == len(results)
        assert sum(1 for b in mask if b) == 10

    def test_stats_no_boxes_channel_mask_max_channels_none(self):
        results = pixelstats(DATA_3 + DATA_1, per_channel=True)
        mask = results.get_channel_mask(0)
        assert len(mask) == len(results)
        assert sum(1 for b in mask if b) == 10 + 10

    def test_stats_with_boxes_channel_mask_max_channels_3(self):
        boxes = get_bboxes(20, 4)
        results = pixelstats(DATA_3 + DATA_1, boxes, per_channel=True)
        mask = results.get_channel_mask(0, 3)
        assert len(mask) == len(results)
        assert sum(1 for b in mask if b) == 10 * 4

    def test_stats_with_boxes_channel_mask_max_channels_none(self):
        boxes = get_bboxes(20, 4)
        results = pixelstats(DATA_3 + DATA_1, boxes, per_channel=True)
        mask = results.get_channel_mask(0)
        assert len(mask) == len(results)
        assert sum(1 for b in mask if b) == (10 + 10) * 4

    def test_stats_with_boxes_channel_mask_all_channels_max_channels_3(self):
        boxes = get_bboxes(20, 4)
        results = pixelstats(DATA_3 + DATA_1, boxes, per_channel=True)
        mask = results.get_channel_mask(None, 3)
        assert len(mask) == len(results)
        assert sum(1 for b in mask if b) == 10 * 3 * 4

    def test_stats_len_with_no_annotations(self):
        output = BaseStatsOutput([])
        object.__setattr__(output, "__annotations__", {})
        assert len(output) == 0

    def test_stats_len_with_none_in_annotations(self):
        output = BaseStatsOutput([])
        object.__setattr__(output, "index_map", None)
        assert len(output) == 0
