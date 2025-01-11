from functools import partial
from typing import Any

import numpy as np
import pytest
from numpy.random import randint
from numpy.typing import NDArray

import dataeval.metrics.stats.base as stats_base
from dataeval.metrics.stats import dimensionstats, hashstats, labelstats, pixelstats, visualstats
from dataeval.metrics.stats.base import (
    SOURCE_INDEX,
    BaseStatsOutput,
    StatsProcessor,
    normalize_box_shape,
    process_stats_unpack,
)
from dataeval.metrics.stats.boxratiostats import boxratiostats, calculate_ratios
from dataeval.metrics.stats.datasetstats import ChannelStatsOutput, DatasetStatsOutput, channelstats, datasetstats

# do not run stats tests using multiple processing
stats_base.DEFAULT_PROCESSES = 1


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


class LengthStatsOutput(BaseStatsOutput):
    length: int


class LengthProcessor(StatsProcessor[LengthStatsOutput]):
    output_class = LengthStatsOutput
    cache_keys = ["length"]
    image_function_map = {
        "neg_length": lambda x: -(x.get("length")),
        "length": lambda x: len(x.image),
        "shape": lambda x: len(x.shape),
        "scaled": lambda x: len(x.scaled),
        "neg_length2": lambda x: x.get("neg_length"),
        "length2": lambda x: x.get("length"),
        "shape2": lambda x: len(x.shape),
        "scaled2": lambda x: len(x.scaled),
    }


@pytest.mark.parametrize("box", [np.array([0, 0, 16, 16]), None])
@pytest.mark.parametrize("per_channel", [False, True])
class TestBaseStats:
    def test_process_stats_unpack(self, box, per_channel):
        results_list: list[dict[str, NDArray[np.int_]]] = []
        images = DATA_3
        bboxes = [box] * len(DATA_3)
        bboxes[0] = np.array([-1, -1, 1, 1])
        partial_fn = partial(process_stats_unpack, per_channel=per_channel, stats_processor_cls=[LengthProcessor])
        for args in enumerate(zip(images, bboxes)):
            r = partial_fn(args)
            results_list.extend(r.results)
        assert len(results_list) == len(DATA_3)

    def test_stats_processor_properties(self, box, per_channel):
        image = (DATA_3[0] * 255).astype(np.int32)
        processor = LengthProcessor(image, box, per_channel)
        shape = (3, 64, 64) if box is None else (3, 16, 16)
        scaled_shape = (shape[0], shape[1] * shape[2]) if per_channel else shape

        assert processor.shape == shape
        assert processor.image.shape == shape
        assert np.max(processor.image) > 1
        assert processor.scaled.shape == scaled_shape
        assert np.max(processor.scaled) <= 1


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
            [visualstats, DATA_1, True, "sharpness", 10],
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
        with pytest.raises(ValueError):
            boxratiostats(boxstats, boxstats)

    def test_boxratio_only_imagestats(self):
        imagestats = dimensionstats(DATA_3, None)
        with pytest.raises(ValueError):
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
        boxes = [np.array([0, 0, 1, 1]), np.array([-1, -1, 100, 100])]
        with pytest.warns(UserWarning, match=r"Bounding box \[1\]\[0\]"):
            boxstats = dimensionstats(DATA_1, boxes)
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

    def test_labelstats_str_keys(self):
        label_array = np.random.choice(["horse", "cow", "sheep", "pig", "chicken"], 50)
        labels = []
        for i in range(10):
            num_labels = np.random.choice(5) + 1
            selected_labels = list(label_array[5 * i : 5 * i + num_labels])
            labels.append(selected_labels)

        stats = labelstats(labels)
        assert stats is not None

    def test_labelstats_int_keys(self):
        labels = [[0, 0, 0, 0, 0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
        stats = labelstats(labels)

        assert stats.label_counts_per_class == {0: 8, 1: 3, 2: 2, 3: 1}
        assert stats.image_indices_per_label == {0: [0, 1, 2, 3], 1: [1, 2, 3], 2: [2, 3], 3: [3]}
        assert stats.image_counts_per_label == {0: 4, 1: 3, 2: 2, 3: 1}
        assert stats.label_counts_per_image == [5, 2, 3, 4]
        assert stats.image_count == 4
        assert stats.class_count == 4
        assert stats.label_count == 14

    def test_datasetstats(self):
        ds_stats = datasetstats(DATA_3)
        assert ds_stats is not None

    def test_channelstats(self):
        ch_stats = channelstats(DATA_3)
        assert ch_stats is not None

    def test_generator_with_stats(self):
        generator = (np.ones((3, 16, 16)) for _ in range(10))
        stats = datasetstats(generator)
        assert len(stats.dimensionstats) == 10

    def test_boxratiostats_channel_mismatch(self):
        boxes = get_bboxes(10, 4)
        boxstats = pixelstats(DATA_3, boxes, per_channel=False)
        imagestats = pixelstats(DATA_3, None, per_channel=True)
        with pytest.raises(ValueError):
            boxratiostats(boxstats, imagestats)

    def test_calculate_ratios_invalid_key(self):
        with pytest.raises(KeyError):
            calculate_ratios("not_here", MockStatsOutput(10), MockStatsOutput(10))


class MockStatsOutput(BaseStatsOutput):
    def __init__(self, length: int, name: str = "mock"):
        self.length = length
        self.name = name

    def __len__(self) -> int:
        return self.length

    def dict(self) -> dict[str, Any]:
        return {self.name: self.name, f"{self.name}_length": self.length}


class TestStatsOutput:
    def test_datasetstats_post_init_length_mismatch(self):
        with pytest.raises(ValueError):
            DatasetStatsOutput(MockStatsOutput(10), MockStatsOutput(10), MockStatsOutput(10), MockStatsOutput(20))  # type: ignore

    def test_channelstats_post_init_length_mismatch(self):
        with pytest.raises(ValueError):
            ChannelStatsOutput(MockStatsOutput(10), MockStatsOutput(20))  # type: ignore

    def test_channelstats_dict(self):
        c = ChannelStatsOutput(MockStatsOutput(2, "one"), MockStatsOutput(2, "two"))  # type: ignore
        assert c.dict() == {"one": "one", "one_length": 2, "two": "two", "two_length": 2}


class TestNormalizeBoxShape:
    def test_ndim_1(self):
        box = normalize_box_shape(np.array([1]))
        np.testing.assert_array_equal(box, np.array([[1]]))

    def test_ndim_2(self):
        box = normalize_box_shape(np.array([[1, 2, 3, 4]]))
        np.testing.assert_array_equal(box, np.array([[1, 2, 3, 4]]))

    def test_ndim_gt_2_raises(self):
        with pytest.raises(ValueError):
            normalize_box_shape(np.array([[[0]]]))
