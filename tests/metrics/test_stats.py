from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from random import random
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.random import randint
from numpy.typing import NDArray

from dataeval.config import set_max_processes
from dataeval.metrics.stats import dimensionstats, hashstats, labelstats, pixelstats, visualstats
from dataeval.metrics.stats._base import (
    StatsProcessor,
    add_stats,
    combine_stats,
    get_dataset_step_from_idx,
    normalize_box_shape,
    process_stats_unpack,
)
from dataeval.metrics.stats._boxratiostats import boxratiostats, calculate_ratios
from dataeval.metrics.stats._imagestats import (
    imagestats,
)
from dataeval.outputs._stats import SOURCE_INDEX, BaseStatsOutput
from dataeval.utils.data._dataset import _find_max, to_image_classification_dataset, to_object_detection_dataset
from dataeval.utils.data._metadata import Metadata
from dataeval.utils.data._targets import Targets

# do not run stats tests using multiple processing
set_max_processes(1)


def get_images(count: int, channels: int):
    return [np.random.random((channels, 64, 64)) for _ in range(count)]


def get_bboxes(count: int, boxes_per_image: int, as_float: bool):
    boxes = []
    for _ in range(count):
        box = []
        for _ in range(boxes_per_image):
            if as_float:
                x0, y0 = (random() * 24), (random() * 24)
                x1, y1 = x0 + 1 + (random() * 23), y0 + 1 + (random() * 23)
            else:
                x0, y0 = randint(0, 24), randint(0, 24)
                x1, y1 = x0 + randint(1, 24), y0 + randint(1, 24)
            box.append([x0, y0, x1, y1])
        boxes.append(np.asarray(box))
    return boxes


DATA_1 = get_images(10, 1)
DATA_3 = get_images(10, 3)


def get_dataset(
    images: list[np.ndarray],
    targets_per_image: int | None = None,
    as_float: bool = False,
    override: list[np.ndarray] | dict[int, list[np.ndarray]] | None = None,
):
    length = len(images)
    override_dict = dict(enumerate(override)) if isinstance(override, list) else override
    if targets_per_image:
        labels = [[0 for _ in range(targets_per_image)] for _ in range(length)]
        bboxes = get_bboxes(length, targets_per_image, as_float)
        if override_dict is not None:
            for i, boxes in override_dict.items():
                bboxes[i] = boxes
        return to_object_detection_dataset(images, labels, bboxes, None, None)
    else:
        labels = [0 for _ in range(length)]
        return to_image_classification_dataset(images, labels, None, None)


def get_metadata(label_array: list[list[int]]) -> Metadata:
    mock = MagicMock(spec=Metadata)
    mock.class_names = [str(i) for i in range(_find_max(label_array) + 1)]
    mock.targets = [MagicMock(spec=Targets) for _ in range(len(label_array))]
    for i, target in enumerate(mock.targets):
        target.labels = np.asarray(label_array[i])
    return mock


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


@pytest.mark.required
class TestBaseStats:
    @pytest.mark.parametrize("as_float", [False, True])
    @pytest.mark.parametrize("per_box", [False, True])
    @pytest.mark.parametrize("per_channel", [False, True])
    def test_process_stats_unpack(self, as_float, per_box, per_channel):
        results_list: list[dict[str, NDArray[np.int_]]] = []
        dataset = get_dataset(DATA_3, targets_per_image=1, as_float=as_float)
        partial_fn = partial(
            process_stats_unpack,
            dataset=dataset,
            per_box=per_box,
            per_channel=per_channel,
            stats_processor_cls=[LengthProcessor],
        )
        for i in range(len(dataset)):
            r = partial_fn(i)
            results_list.extend(r.results)
        assert len(results_list) == len(DATA_3)

    @pytest.mark.parametrize("box", [np.array([0, 0, 16, 16]), None])
    @pytest.mark.parametrize("per_channel", [False, True])
    def test_stats_processor_properties(self, box, per_channel):
        image = (DATA_3[0] * 255).astype(np.int32)
        processor = LengthProcessor(image, box, per_channel=per_channel)
        shape = (3, 64, 64) if box is None else (3, 16, 16)
        scaled_shape = (shape[0], shape[1] * shape[2]) if per_channel else shape

        assert processor.shape == shape
        assert processor.image.shape == shape
        assert np.max(processor.image) > 1
        assert processor.scaled.shape == scaled_shape
        assert np.max(processor.scaled) <= 1


@pytest.mark.required
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

    def test_stats_len_with_no_annotations(self):
        output = BaseStatsOutput([], np.array([]))
        object.__setattr__(output, "__annotations__", {})
        assert len(output) == 0

    def test_boxratio_only_imagestats(self):
        imagestats = dimensionstats(get_dataset(DATA_3, 4, True), per_box=True)
        with pytest.raises(ValueError):
            boxratiostats(imagestats, imagestats)

    def test_stats_box_out_of_range(self):
        boxes = [np.array([[0, 0, 1, 1], [-1, -1, 100, 100]])]
        with pytest.warns(UserWarning, match=r"Bounding box \[0\]\[1\]"):
            boxstats = dimensionstats(get_dataset(DATA_1, 2, False, boxes), per_box=True)
        assert boxstats is not None

    def test_stats_source_index_no_boxes_no_channels(self):
        stats = pixelstats(DATA_3)
        assert all(si.box is None for si in stats.source_index)
        assert all(si.channel is None for si in stats.source_index)

    def test_stats_source_index_no_boxes_with_channels(self):
        stats = pixelstats(DATA_3, per_channel=True)
        assert all(si.box is None for si in stats.source_index)
        assert all(si.channel is not None for si in stats.source_index)

    def test_imagestats(self):
        ds_stats = imagestats(DATA_3)
        assert ds_stats is not None

    def test_channelstats(self):
        ch_stats = imagestats(DATA_3, per_channel=True)
        assert ch_stats is not None

    def test_sequence_with_stats(self):
        seq = [np.ones((3, 16, 16)) for _ in range(10)]
        stats = imagestats(seq)
        assert len(stats) == 10

    def test_calculate_ratios_invalid_key(self):
        with pytest.raises(KeyError):
            calculate_ratios("not_here", MockStatsOutput(10), MockStatsOutput(10))


@pytest.mark.required
class TestLabelStats:
    @pytest.mark.parametrize("two_d", [True, False])
    def test_labelstats_list_int(self, two_d):
        label_array = [[0, 0, 0, 0, 0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
        labels = label_array if two_d else np.concatenate(label_array).tolist()
        metadata = get_metadata(labels)
        stats = labelstats(metadata)

        assert stats.label_counts_per_class == [8, 3, 2, 1]
        assert stats.class_count == 4
        assert stats.label_count == 14
        if two_d:
            assert stats.image_indices_per_class == [[0, 1, 2, 3], [1, 2, 3], [2, 3], [3]]
            assert stats.image_counts_per_class == [4, 3, 2, 1]
            assert stats.label_counts_per_image == [5, 2, 3, 4]
            assert stats.image_count == 4
        else:
            assert stats.image_indices_per_class == [[0, 1, 2, 3, 4, 5, 7, 10], [6, 8, 11], [9, 12], [13]]
            assert stats.image_counts_per_class == [8, 3, 2, 1]
            assert stats.label_counts_per_image == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            assert stats.image_count == 14

    @pytest.mark.requires_all
    def test_labelstats_to_dataframe(self):
        label_array = [[0, 0, 0, 0, 0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
        metadata = get_metadata(label_array)
        stats = labelstats(metadata)
        stats_df = stats.to_dataframe()
        assert stats_df.shape == (4, 3)


@pytest.mark.required
@pytest.mark.parametrize("as_float", [True, False])
class TestBBoxStats:
    def test_stats_with_bboxes(self, as_float):
        results = pixelstats(get_dataset(DATA_1, 4, as_float), per_box=True, per_channel=True)
        assert len(results) == 10 * 4

    def test_array_stats_with_bboxes_per_channel_true(self, as_float):
        results = pixelstats(get_dataset(DATA_3, 4, as_float), per_box=True, per_channel=True)
        assert len(results) == 10 * 4 * 3

    def test_array_stats_with_bboxes_per_channel_false(self, as_float):
        results = pixelstats(get_dataset(DATA_3, 4, as_float), per_box=True, per_channel=False)
        assert len(results) == 10 * 4

    def test_dimension_stats_with_bboxes(self, as_float):
        results = dimensionstats(get_dataset(DATA_3, 4, as_float), per_box=True)
        assert len(results) == 10 * 4

    def test_stats_with_boxes_channel_mask_max_channels_3(self, as_float):
        results = pixelstats(get_dataset(DATA_3 + DATA_1, 4, as_float), per_box=True, per_channel=True)
        mask = results.get_channel_mask(0, 3)
        assert len(mask) == len(results)
        assert sum(1 for b in mask if b) == 10 * 4

    def test_stats_with_boxes_channel_mask_max_channels_none(self, as_float):
        results = pixelstats(get_dataset(DATA_3 + DATA_1, 4, as_float), per_box=True, per_channel=True)
        mask = results.get_channel_mask(0)
        assert len(mask) == len(results)
        assert sum(1 for b in mask if b) == (10 + 10) * 4

    def test_stats_with_boxes_channel_mask_all_channels_max_channels_3(self, as_float):
        results = pixelstats(get_dataset(DATA_3 + DATA_1, 4, as_float), per_box=True, per_channel=True)
        mask = results.get_channel_mask(None, 3)
        assert len(mask) == len(results)
        assert sum(1 for b in mask if b) == 10 * 3 * 4

    def test_boxratio_dimensionstats_both_datasets(self, as_float):
        boxstats = dimensionstats(get_dataset(DATA_3, 4, as_float), per_box=True)
        imagestats = dimensionstats(get_dataset(DATA_3, 4, as_float), per_box=False)
        ratiostats = boxratiostats(boxstats, imagestats)
        assert ratiostats is not None

    def test_boxratio_dimensionstats(self, as_float):
        boxstats = dimensionstats(get_dataset(DATA_3, 4, as_float), per_box=True)
        imagestats = dimensionstats(DATA_3)
        ratiostats = boxratiostats(boxstats, imagestats)
        assert ratiostats is not None

    def test_boxratio_pixelstats(self, as_float):
        boxstats = pixelstats(get_dataset(DATA_3, 4, as_float), per_box=True)
        imagestats = pixelstats(DATA_3)
        ratiostats = boxratiostats(boxstats, imagestats)
        assert ratiostats is not None

    def test_boxratio_pixelstats_per_channel(self, as_float):
        boxstats = pixelstats(get_dataset(DATA_3, 4, as_float), per_box=True, per_channel=True)
        imagestats = pixelstats(DATA_3, per_channel=True)
        ratiostats = boxratiostats(boxstats, imagestats)
        assert ratiostats is not None

    def test_boxratio_visualstats(self, as_float):
        boxstats = visualstats(get_dataset(DATA_3, 4, as_float), per_box=True)
        imagestats = visualstats(DATA_3)
        ratiostats = boxratiostats(boxstats, imagestats)
        assert ratiostats is not None

    def test_boxratio_visualstats_per_channel(self, as_float):
        boxstats = pixelstats(get_dataset(DATA_3, 4, as_float), per_box=True, per_channel=True)
        imagestats = pixelstats(DATA_3, per_channel=True)
        ratiostats = boxratiostats(boxstats, imagestats)
        assert ratiostats is not None

    def test_boxratio_only_boxstats(self, as_float):
        boxstats = dimensionstats(get_dataset(DATA_3, 4, as_float), per_box=True)
        with pytest.raises(ValueError):
            boxratiostats(boxstats, boxstats)

    def test_boxratio_mismatch_stats_type(self, as_float):
        boxstats = visualstats(get_dataset(DATA_3, 4, as_float), per_box=True)
        imagestats = dimensionstats(DATA_3)
        with pytest.raises(TypeError):
            boxratiostats(boxstats, imagestats)

    def test_boxratio_mismatch_stats_source(self, as_float):
        boxstats = dimensionstats(get_dataset(DATA_3, 4, as_float), per_box=True)
        imagestats = dimensionstats(DATA_1 + DATA_1)
        with pytest.raises(ValueError):
            boxratiostats(boxstats, imagestats)

    def test_stats_source_index_with_boxes_no_channels(self, as_float):
        stats = pixelstats(get_dataset(DATA_3, 2, as_float), per_box=True)
        assert all(si.box is not None for si in stats.source_index)
        assert all(si.channel is None for si in stats.source_index)

    def test_stats_source_index_with_boxes_with_channels(self, as_float):
        stats = pixelstats(get_dataset(DATA_3, 2, as_float), per_box=True, per_channel=True)
        assert all(si.box is not None for si in stats.source_index)
        assert all(si.channel is not None for si in stats.source_index)

    def test_boxratiostats_channel_mismatch(self, as_float):
        boxstats = pixelstats(get_dataset(DATA_3, 4, as_float), per_channel=False)
        imagestats = pixelstats(DATA_3, per_channel=True)
        with pytest.raises(ValueError):
            boxratiostats(boxstats, imagestats)

    def test_stats_div_by_zero(self, as_float):
        images = [np.zeros((1, 64, 64)) for _ in range(10)]
        dataset = get_dataset(images, 4, as_float)
        vi = visualstats(images)
        vb = visualstats(dataset, per_box=True)
        pi = pixelstats(images)
        pb = pixelstats(dataset, per_box=True)
        rv = boxratiostats(vb, vi)
        rb = boxratiostats(pb, pi)

        for v in [m.dict().values() for m in [vi, vb, pi, pb, rv, rb]]:
            if isinstance(v, np.ndarray):
                assert not np.isnan(np.sum(v))
                assert not np.isinf(np.sum(v))


@pytest.mark.required
class MockStatsOutput(BaseStatsOutput):
    def __init__(self, length: int, name: str = "mock"):
        self.length = length
        self.name = name

    def __len__(self) -> int:
        return self.length

    def dict(self) -> dict[str, Any]:
        return {self.name: self.name, f"{self.name}_length": self.length}


@pytest.mark.required
class TestBaseStatsOutput:
    def test_post_init_length_mismatch(self):
        @dataclass(frozen=True)
        class TestStatsOutput(BaseStatsOutput):
            foo: list[int]
            bar: list[str]

        with pytest.raises(ValueError):
            TestStatsOutput(
                foo=[1, 2, 3, 4],
                bar=["a", "b", "c"],
                source_index=[1, 2, 3, 4],  # type: ignore
                box_count=np.array([1, 2, 3, 4]),
            )


@pytest.mark.required
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


@pytest.mark.requires_all
@pytest.mark.required
class TestStatsPlotting:
    @pytest.mark.parametrize(
        "stats, data, log, per_channel, channel",
        [
            [dimensionstats, DATA_3, False, None, None],
            [pixelstats, DATA_3, True, False, None],
            [imagestats, DATA_3, True, False, None],
            [visualstats, DATA_1, False, False, None],
            [imagestats, DATA_3, False, True, None],
            [imagestats, DATA_3, False, True, 2],
            [imagestats, DATA_3, False, True, [0, 2]],
        ],
    )
    def test_stats_plot(self, stats, data, log, per_channel, channel):
        results = stats(data, **({"per_channel": per_channel} if per_channel is not None else {}))
        if channel is None:
            results.plot(log=log)
        elif isinstance(channel, int):
            results.plot(log=log, channel_limit=channel)
        else:
            results.plot(log=log, channel_index=channel)

    def test_labelstats_to_table(self):
        label_array = np.concatenate(
            [
                np.random.choice(5, 45),
                np.random.permutation(5),
            ]
        ).tolist()
        stats = labelstats(get_metadata(label_array))
        assert stats is not None
        table_result = stats.to_table()
        assert isinstance(table_result, str)

    @pytest.mark.parametrize(
        "value, param, expected",
        [
            [2.0, "limit", [3, None]],
            [[0, 1], "limit", [3, None]],
            ["first", "limit", [3, None]],
            [60, "limit", [3, None]],
            [2, "limit", [2, None]],
            [1.0, "index", [3, None]],
            [[1, 2], "index", [2, [False, True, True] * 10]],
            [[2.1, 3.0], "index", [3, None]],
            [0, "index", [1, [True, False, False] * 10]],
            [50, "index", [3, None]],
            ["second", "index", [3, None]],
        ],
    )
    def test_channelstats_plot_params(self, value, param, expected):
        results = imagestats(DATA_3, per_channel=True)
        if param == "limit":
            max_chan, ch_mask = results._get_channels(channel_limit=value)
        else:
            max_chan, ch_mask = results._get_channels(channel_index=value)

        assert max_chan == expected[0]
        assert ch_mask == expected[1]


@pytest.mark.required
class TestCombineStats:
    def test_image_stats_addition(self):
        results_a = pixelstats(DATA_1)
        results_b = pixelstats(DATA_1)
        results_added = add_stats(results_a, results_b)
        results_alldata = pixelstats(DATA_1 + DATA_1)
        assert len(results_added.mean) == 20
        assert len(results_alldata.mean) == 20

    def test_channel_stats_addition(self):
        results_a = pixelstats(DATA_3, per_channel=True)
        results_b = pixelstats(DATA_1, per_channel=True)
        results_c = pixelstats(DATA_3, per_channel=True)
        results_added, dataset_steps = combine_stats((results_a, results_b, results_c))
        results_alldata = pixelstats(DATA_3 + DATA_1 + DATA_3, per_channel=True)
        assert results_added is not None
        np.testing.assert_array_equal(results_added.mean, results_alldata.mean)
        assert dataset_steps == [30, 40, 70]

    def test_combine_different_stats_fail(self):
        stats_a = pixelstats(DATA_1)
        stats_b = hashstats(DATA_1)
        with pytest.raises(TypeError):
            add_stats(stats_a, stats_b)

    def test_combine_different_types(self):
        stats_a = pixelstats(DATA_1)
        random_obj = "hello"
        with pytest.raises(TypeError):
            add_stats(stats_a, random_obj)  # type: ignore

    def test_get_dataset_step_from_idx_no_steps(self):
        results = get_dataset_step_from_idx(0, [])
        assert results == (-1, 0)

    def test_combine_stats_with_invalid_stat_types(self):
        stats = pixelstats(DATA_1)
        with pytest.raises(TypeError):
            combine_stats([])  # type: ignore
        with pytest.raises(TypeError):
            combine_stats([None])  # type: ignore
        with pytest.raises(TypeError):
            combine_stats([1, None])  # type: ignore
        with pytest.raises(TypeError):
            combine_stats([stats, None])  # type: ignore
        with pytest.raises(TypeError):
            combine_stats([stats, 1, None])  # type: ignore
