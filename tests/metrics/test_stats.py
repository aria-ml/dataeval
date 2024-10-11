import numpy as np
import pytest
from numpy.random import randint

from dataeval._internal.metrics.stats.base import SOURCE_INDEX, BaseStatsOutput
from dataeval._internal.metrics.stats.boxratiostats import boxratiostats
from dataeval.metrics.stats import dimensionstats, hashstats, pixelstats, visualstats


def get_dataset(count: int, channels: int):
    return [np.random.random((channels, 64, 64)) for _ in range(count)]


def get_bboxes(count: int, boxes_per_image: int):
    boxes = []
    for _ in range(count):
        box = []
        for _ in range(boxes_per_image):
            x0 = randint(1, 24)
            y0 = randint(1, 24)
            x1 = x0 + randint(1, 24)
            y1 = y0 + randint(1, 24)
            box.append([x0, y0, x1, y1])
        boxes.append(np.asarray(box))
    return boxes


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
            [pixelstats, DATA_1 + DATA_3, False, "entropy", 10 + 10],
            [pixelstats, DATA_1 + DATA_3, True, "histogram", 10 * 3 + 10],
            [visualstats, DATA_1, False, "brightness", 10],
            [visualstats, DATA_1, True, "blurriness", 10],
            [visualstats, DATA_3, False, "contrast", 10],
            [visualstats, DATA_3, True, "darkness", 10 * 3],
            [visualstats, DATA_1 + DATA_3, False, "missing", 10 + 10],
            [visualstats, DATA_1 + DATA_3, True, "percentiles", 10 * 3 + 10],
        ],
    )
    def test_stats(self, stats, data, per_channel, attribute, length):
        results = stats(data) if per_channel is None else stats(data, per_channel=per_channel)
        assert len(getattr(results, attribute)) == length
        if hasattr(results, SOURCE_INDEX) and results.source_index is not None:
            assert len(results.source_index) == length

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

    def test_dimension_stats_with_bboxes(self):
        boxes = get_bboxes(10, 4)
        results = dimensionstats(DATA_3, boxes)
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
        output = BaseStatsOutput([], np.array([]))
        object.__setattr__(output, "__annotations__", {})
        assert len(output) == 0

    def test_boxratio_dimensionstats(self):
        boxes = get_bboxes(10, 4)
        boxstats = dimensionstats(DATA_3, boxes)
        imagestats = dimensionstats(DATA_3, None)
        ratiostats = boxratiostats(boxstats, imagestats)
        assert ratiostats is not None

    def test_boxratio_pixelstats(self):
        boxes = get_bboxes(10, 4)
        boxstats = pixelstats(DATA_3, boxes)
        imagestats = pixelstats(DATA_3, None)
        ratiostats = boxratiostats(boxstats, imagestats)
        assert ratiostats is not None

    def test_boxratio_pixelstats_per_channel(self):
        boxes = get_bboxes(10, 4)
        boxstats = pixelstats(DATA_3, boxes, True)
        imagestats = pixelstats(DATA_3, None, True)
        ratiostats = boxratiostats(boxstats, imagestats)
        assert ratiostats is not None

    def test_boxratio_visualstats(self):
        boxes = get_bboxes(10, 4)
        boxstats = visualstats(DATA_3, boxes)
        imagestats = visualstats(DATA_3, None)
        ratiostats = boxratiostats(boxstats, imagestats)
        assert ratiostats is not None

    def test_boxratio_visualstats_per_channel(self):
        boxes = get_bboxes(10, 4)
        boxstats = pixelstats(DATA_3, boxes, True)
        imagestats = pixelstats(DATA_3, None, True)
        ratiostats = boxratiostats(boxstats, imagestats)
        assert ratiostats is not None

    def test_boxratio_only_boxstats(self):
        boxes = get_bboxes(10, 4)
        boxstats = dimensionstats(DATA_3, boxes)
        with pytest.raises(TypeError):
            boxratiostats(boxstats, boxstats)

    def test_boxratio_only_imagestats(self):
        imagestats = dimensionstats(DATA_3, None)
        with pytest.raises(TypeError):
            boxratiostats(imagestats, imagestats)

    def test_boxratio_mismatch_stats_type(self):
        boxes = get_bboxes(10, 4)
        boxstats = visualstats(DATA_3, boxes)
        imagestats = dimensionstats(DATA_3, None)
        with pytest.raises(TypeError):
            boxratiostats(boxstats, imagestats)

    def test_boxratio_mismatch_stats_source(self):
        boxes = get_bboxes(10, 4)
        boxstats = dimensionstats(DATA_3, boxes)
        imagestats = dimensionstats(DATA_1 + DATA_1, None)
        with pytest.raises(ValueError):
            boxratiostats(boxstats, imagestats)

    def test_stats_box_out_of_range(self):
        boxes = [np.array([-1, -1, 100, 100])]
        with pytest.warns() as warning:
            boxstats = dimensionstats(DATA_1, boxes)
            message = warning[0].message.args[0]  # type: ignore
        assert message == "Bounding box 0: [ -1  -1 100 100] is out of bounds of image 0: (1, 64, 64)."
        assert boxstats is not None

    def test_stats_div_by_zero(self):
        images = [np.zeros((1, 64, 64)) for _ in range(10)]
        boxes = get_bboxes(10, 4)
        vi = visualstats(images)
        vb = visualstats(images, boxes)
        pi = pixelstats(images)
        pb = pixelstats(images, boxes)
        rv = boxratiostats(vb, vi)
        rb = boxratiostats(pb, pi)

        for v in [m.dict().values() for m in [vi, vb, pi, pb, rv, rb]]:
            if isinstance(v, np.ndarray):
                assert not np.isnan(np.sum(v))
                assert not np.isinf(np.sum(v))

    def test_stats_source_index_no_boxes_no_channels(self):
        stats = pixelstats(DATA_3)
        assert all(si.box is None for si in stats.source_index)
        assert all(si.channel is None for si in stats.source_index)

    def test_stats_source_index_with_boxes_no_channels(self):
        boxes = get_bboxes(10, 2)
        stats = pixelstats(DATA_3, boxes)
        assert all(si.box is not None for si in stats.source_index)
        assert all(si.channel is None for si in stats.source_index)

    def test_stats_source_index_no_boxes_with_channels(self):
        stats = pixelstats(DATA_3, per_channel=True)
        assert all(si.box is None for si in stats.source_index)
        assert all(si.channel is not None for si in stats.source_index)

    def test_stats_source_index_with_boxes_with_channels(self):
        boxes = get_bboxes(10, 2)
        stats = pixelstats(DATA_3, boxes, per_channel=True)
        assert all(si.box is not None for si in stats.source_index)
        assert all(si.channel is not None for si in stats.source_index)
