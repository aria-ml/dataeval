from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray

from dataeval.config import set_max_processes, use_max_processes
from dataeval.metrics.stats import dimensionstats, hashstats, pixelstats, visualstats
from dataeval.metrics.stats._base import (
    BoundingBox,
    StatsProcessor,
    _enumerate,
    add_stats,
    combine_stats,
    get_dataset_step_from_idx,
    process_stats,
    process_stats_unpack,
)
from dataeval.metrics.stats._boxratiostats import boxratiostats, calculate_ratios
from dataeval.metrics.stats._imagestats import (
    imagestats,
)
from dataeval.outputs._stats import BASE_ATTRS, SOURCE_INDEX, BaseStatsOutput
from dataeval.utils.data._dataset import to_object_detection_dataset

# do not run stats tests using multiple processing
set_max_processes(1)

DATA_1 = [np.random.random((1, 64, 64)) for _ in range(10)]
DATA_3 = [np.random.random((3, 64, 64)) for _ in range(10)]

STATS_ATTRS: dict[str, set[str]] = {
    "pixelstats": {"mean", "std", "var", "skew", "kurtosis", "entropy", "histogram"},
    "visualstats": {"brightness", "contrast", "darkness", "missing", "sharpness", "zeros", "percentiles"},
    "dimensionstats": {
        "offset_x",
        "offset_y",
        "width",
        "height",
        "channels",
        "size",
        "aspect_ratio",
        "depth",
        "center",
        "distance_center",
        "distance_edge",
    },
    "hashstats": {"xxhash", "pchash"},
}

STATS_2D_ATTRS: set[str] = {"histogram", "percentiles", "center"}


@pytest.fixture
def stats_attrs(request):
    """Returns set of attributes for a specific StatsOutput class as defined in STATS_ATTRS"""
    p: str = request.param

    if p == "imagestats":
        return set.union(*(STATS_ATTRS[k] for k in ("pixelstats", "visualstats", "dimensionstats")))
    if p == "channelstats":
        return set.union(*(STATS_ATTRS[k] for k in ("pixelstats", "visualstats")))
    return STATS_ATTRS[p]


class LengthStatsOutput(BaseStatsOutput):
    length: int


class LengthProcessor(StatsProcessor[LengthStatsOutput]):
    output_class = LengthStatsOutput
    cache_keys = {"length"}
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
    @pytest.mark.parametrize("per_channel", [False, True])
    def test_process_stats_unpack(self, get_od_dataset, as_float, per_channel):
        results_list: list[dict[str, NDArray[np.int_]]] = []
        dataset = get_od_dataset(DATA_3, targets_per_image=1, as_float=as_float)
        partial_fn = partial(
            process_stats_unpack,
            per_channel=per_channel,
            stats_processor_cls=[LengthProcessor],
        )
        for i in range(len(dataset)):
            r = partial_fn((i, dataset[i][0], None))
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

    @pytest.mark.parametrize("per_box", [False, True])
    def test_enumerate(self, get_od_dataset, per_box):
        dataset = get_od_dataset(DATA_3, targets_per_image=1, as_float=True)
        for i, image, boxes in _enumerate(dataset, per_box):
            assert np.asarray(image).shape == (3, 64, 64)
            assert isinstance(boxes, list) if per_box else boxes is None

    def test_process_stats_out_of_bounds(self):
        invalid_box = BoundingBox(-5.0, -5.0, -1.0, -1.0)
        expected_warning = f"Bounding box [0][0]: {invalid_box} for image shape (3, 16, 16) is invalid."
        output = process_stats(0, np.random.random((3, 16, 16)), [invalid_box], False, [LengthProcessor])
        assert len(output.warnings_list) == 1
        assert output.warnings_list[0] == expected_warning


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
        output = BaseStatsOutput([], np.array([]), 0)
        object.__setattr__(output, "__annotations__", {})
        assert len(output) == 0

    def test_boxratio_only_imagestats(self, get_od_dataset):
        imagestats = dimensionstats(get_od_dataset(DATA_3, 4, True))
        with pytest.raises(ValueError, match="Input for boxstats must contain box information."):
            boxratiostats(imagestats, imagestats)

    def test_boxratio_only_boxstats(self, get_od_dataset):
        boxes = [np.array([[0, 0, 1, 1], [0, 0, 100, 100]])]
        boxstats = pixelstats(get_od_dataset(DATA_1, 2, False, boxes), per_box=True)
        with pytest.raises(ValueError, match="Input for imgstats must not contain box information."):
            boxratiostats(boxstats, boxstats)

    def test_boxratio_inputs_swapped(self, get_od_dataset):
        boxes = [np.array([[0, 0, 1, 1], [0, 0, 100, 100]])]
        imgstats = pixelstats(get_od_dataset(DATA_1, 2, False, boxes), per_box=False, per_channel=True)
        boxstats = pixelstats(get_od_dataset(DATA_1, 2, False, boxes), per_box=True)
        with pytest.raises(ValueError):
            boxratiostats(imgstats, boxstats)

    def test_boxratio_channel_mismatch(self, get_od_dataset):
        boxes = [np.array([[0, 0, 1, 1], [0, 0, 100, 100]])]
        imgstats = pixelstats(get_od_dataset(DATA_1, 2, False, boxes), per_box=False, per_channel=True)
        boxstats = pixelstats(get_od_dataset(DATA_1, 2, False, boxes), per_box=True)
        with pytest.raises(ValueError, match="Input for boxstats and imgstats must have matching channel information."):
            boxratiostats(boxstats, imgstats)

    def test_stats_box_out_of_range(self, get_od_dataset):
        boxes = [np.array([[0, 0, 1, 1], [-10, -10, -1, -1]])]
        with pytest.warns(UserWarning, match=r"Bounding box \[0\]\[1\]"):
            boxstats = dimensionstats(get_od_dataset(DATA_1, 2, False, boxes), per_box=True)
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

    def test_imagestats_multiprocess(self):
        with use_max_processes(2):
            stats = imagestats(DATA_3)
        assert stats is not None


@pytest.mark.required
@pytest.mark.parametrize("as_float", [True, False])
class TestBBoxStats:
    def test_stats_with_bboxes(self, get_od_dataset, as_float):
        results = pixelstats(get_od_dataset(DATA_1, 4, as_float), per_box=True, per_channel=True)
        assert len(results) == 10 * 4

    def test_array_stats_with_bboxes_per_channel_true(self, get_od_dataset, as_float):
        results = pixelstats(get_od_dataset(DATA_3, 4, as_float), per_box=True, per_channel=True)
        assert len(results) == 10 * 4 * 3

    def test_array_stats_with_bboxes_per_channel_false(self, get_od_dataset, as_float):
        results = pixelstats(get_od_dataset(DATA_3, 4, as_float), per_box=True, per_channel=False)
        assert len(results) == 10 * 4

    def test_dimension_stats_with_bboxes(self, get_od_dataset, as_float):
        results = dimensionstats(get_od_dataset(DATA_3, 4, as_float), per_box=True)
        assert len(results) == 10 * 4

    def test_stats_with_boxes_channel_mask_max_channels_3(self, get_od_dataset, as_float):
        results = pixelstats(get_od_dataset(DATA_3 + DATA_1, 4, as_float), per_box=True, per_channel=True)
        mask = results.get_channel_mask(0, 3)
        assert len(mask) == len(results)
        assert sum(1 for b in mask if b) == 10 * 4

    def test_stats_with_boxes_channel_mask_max_channels_none(self, get_od_dataset, as_float):
        results = pixelstats(get_od_dataset(DATA_3 + DATA_1, 4, as_float), per_box=True, per_channel=True)
        mask = results.get_channel_mask(0)
        assert len(mask) == len(results)
        assert sum(1 for b in mask if b) == (10 + 10) * 4

    def test_stats_with_boxes_channel_mask_all_channels_max_channels_3(self, get_od_dataset, as_float):
        results = pixelstats(get_od_dataset(DATA_3 + DATA_1, 4, as_float), per_box=True, per_channel=True)
        mask = results.get_channel_mask(None, 3)
        assert len(mask) == len(results)
        assert sum(1 for b in mask if b) == 10 * 3 * 4

    def test_boxratio_dimensionstats_both_datasets(self, get_od_dataset, as_float):
        boxstats = dimensionstats(get_od_dataset(DATA_3, 4, as_float), per_box=True)
        imagestats = dimensionstats(get_od_dataset(DATA_3, 4, as_float), per_box=False)
        ratiostats = boxratiostats(boxstats, imagestats)
        assert ratiostats is not None

    def test_boxratio_dimensionstats(self, get_od_dataset, as_float):
        boxstats = dimensionstats(get_od_dataset(DATA_3, 4, as_float), per_box=True)
        imagestats = dimensionstats(DATA_3)
        ratiostats = boxratiostats(boxstats, imagestats)
        assert ratiostats is not None

    def test_boxratio_pixelstats(self, get_od_dataset, as_float):
        boxstats = pixelstats(get_od_dataset(DATA_3, 4, as_float), per_box=True)
        imagestats = pixelstats(DATA_3)
        ratiostats = boxratiostats(boxstats, imagestats)
        assert ratiostats is not None

    def test_boxratio_pixelstats_per_channel(self, get_od_dataset, as_float):
        boxstats = pixelstats(get_od_dataset(DATA_3, 4, as_float), per_box=True, per_channel=True)
        imagestats = pixelstats(DATA_3, per_channel=True)
        ratiostats = boxratiostats(boxstats, imagestats)
        assert ratiostats is not None

    def test_boxratio_visualstats(self, get_od_dataset, as_float):
        boxstats = visualstats(get_od_dataset(DATA_3, 4, as_float), per_box=True)
        imagestats = visualstats(DATA_3)
        ratiostats = boxratiostats(boxstats, imagestats)
        assert ratiostats is not None

    def test_boxratio_visualstats_per_channel(self, get_od_dataset, as_float):
        boxstats = pixelstats(get_od_dataset(DATA_3, 4, as_float), per_box=True, per_channel=True)
        imagestats = pixelstats(DATA_3, per_channel=True)
        ratiostats = boxratiostats(boxstats, imagestats)
        assert ratiostats is not None

    def test_boxratio_only_boxstats(self, get_od_dataset, as_float):
        boxstats = dimensionstats(get_od_dataset(DATA_3, 4, as_float), per_box=True)
        with pytest.raises(ValueError):
            boxratiostats(boxstats, boxstats)

    def test_boxratio_mismatch_stats_type(self, get_od_dataset, as_float):
        boxstats = visualstats(get_od_dataset(DATA_3, 4, as_float), per_box=True)
        imagestats = dimensionstats(DATA_3)
        with pytest.raises(TypeError):
            boxratiostats(boxstats, imagestats)

    def test_boxratio_mismatch_stats_source(self, get_od_dataset, as_float):
        boxstats = dimensionstats(get_od_dataset(DATA_3, 4, as_float), per_box=True)
        imagestats = dimensionstats(DATA_1 + DATA_1)
        with pytest.raises(ValueError):
            boxratiostats(boxstats, imagestats)

    def test_stats_source_index_with_boxes_no_channels(self, get_od_dataset, as_float):
        stats = pixelstats(get_od_dataset(DATA_3, 2, as_float), per_box=True)
        assert all(si.box is not None for si in stats.source_index)
        assert all(si.channel is None for si in stats.source_index)

    def test_stats_source_index_with_boxes_with_channels(self, get_od_dataset, as_float):
        stats = pixelstats(get_od_dataset(DATA_3, 2, as_float), per_box=True, per_channel=True)
        assert all(si.box is not None for si in stats.source_index)
        assert all(si.channel is not None for si in stats.source_index)

    def test_boxratiostats_channel_mismatch(self, get_od_dataset, as_float):
        boxstats = pixelstats(get_od_dataset(DATA_3, 4, as_float), per_channel=False)
        imagestats = pixelstats(DATA_3, per_channel=True)
        with pytest.raises(ValueError):
            boxratiostats(boxstats, imagestats)

    def test_stats_div_by_zero(self, get_od_dataset, as_float):
        images = [np.zeros((1, 64, 64)) for _ in range(10)]
        dataset = get_od_dataset(images, 4, as_float)
        vi = visualstats(images)
        vb = visualstats(dataset, per_box=True)
        pi = pixelstats(images)
        pb = pixelstats(dataset, per_box=True)
        rv = boxratiostats(vb, vi)
        rb = boxratiostats(pb, pi)

        for v in [m.data().values() for m in [vi, vb, pi, pb, rv, rb]]:
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

    def data(self) -> dict[str, Any]:
        return {self.name: self.name, f"{self.name}_length": self.length}


@pytest.mark.required
class TestBaseStatsOutput:
    def test_post_init_si_length_mismatch(self):
        @dataclass(frozen=True)
        class TestStatsOutput(BaseStatsOutput):
            foo: list[int]
            bar: list[str]

        with pytest.raises(ValueError, match="All values must have the same length as source_index."):
            TestStatsOutput(
                foo=[1, 2, 3, 4],
                bar=["a", "b", "c"],
                source_index=[1, 2, 3, 4],  # type: ignore
                object_count=np.array([1, 2, 3, 4]),
                image_count=4,
            )

    def test_post_init_oc_length_mismatch(self):
        @dataclass(frozen=True)
        class TestStatsOutput(BaseStatsOutput):
            foo: list[int]
            bar: list[str]

        with pytest.raises(ValueError, match="Total object counts per image does not match image count."):
            TestStatsOutput(
                foo=[1, 2, 3, 4],
                bar=["a", "b", "c", "d"],
                source_index=[1, 2, 3, 4],  # type: ignore
                object_count=np.array([1, 2, 3, 4]),
                image_count=3,
            )


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


IMAGE_COUNT = 10
CLASS_COUNT = 5
IMAGE_DIMS = 32


@pytest.mark.required
class TestOffImageBoxes:
    detections = [np.random.randint(1, 5) for _ in range(IMAGE_COUNT)]
    images = np.random.randint(0, 256, (IMAGE_COUNT, 3, IMAGE_DIMS, IMAGE_DIMS))
    labels = [np.random.randint(0, CLASS_COUNT, (n,)).tolist() for n in detections]
    boxes = [np.sort(np.random.rand(n, 4) * IMAGE_DIMS).tolist() for n in detections]
    metadata = [{"id": i * n, "pose": np.random.randint(0, 5)} for i, n in enumerate(detections)]
    classes = [str(i) for i in range(CLASS_COUNT)]

    @pytest.mark.parametrize("box", [(-10.0, -10.0, 20.0, 20.0), (20.0, 20.0, 50.0, 0.0)])
    def test_off_image_boxes_no_nan(self, box: tuple[float, float, float, float]):
        # set 2 boxes out of bounds
        boxes = self.boxes.copy()
        boxes[0][0] = box

        dataset = to_object_detection_dataset(
            self.images,
            self.labels,
            boxes,
            self.metadata,
            self.classes,
        )

        img_stats = imagestats(dataset)
        box_stats = imagestats(dataset, per_box=True)
        chn_stats = imagestats(dataset, per_box=True, per_channel=True)

        for stat in [img_stats, box_stats, chn_stats]:
            for k, v in stat.factors().items():
                assert not np.any(np.isnan(v)), f"NaN value found in {k}"
                assert not np.any(np.isinf(v)), f"Inf value found in {k}"

    @pytest.mark.parametrize("box", [(10, 9, 8, 7), (5, 9, 8, 7), (10, 5, 8, 7)])
    def test_invalid_bounding_box(self, box: tuple[int, int, int, int]):
        boxes = self.boxes.copy()
        boxes[0][0] = box

        dataset = to_object_detection_dataset(
            self.images,
            self.labels,
            boxes,
            self.metadata,
            self.classes,
        )

        with pytest.warns(UserWarning, match="Invalid bounding box coordinates"):
            imagestats(dataset, per_box=True)

    @pytest.mark.parametrize("box", [(0, 0, 0, 0), (5, 6, 5, 10), (5, 6, 10, 6)])
    def test_zero_area_bounding_box(self, box: tuple[int, int, int, int]):
        boxes = self.boxes.copy()
        boxes[0][0] = box

        dataset = to_object_detection_dataset(
            self.images,
            self.labels,
            boxes,
            self.metadata,
            self.classes,
        )

        output = imagestats(dataset, per_box=True)
        assert np.isnan(output.mean[0])

    def test_empty_bounding_box(self):
        boxes = self.boxes.copy()
        boxes[-1][0] = []

        dataset = to_object_detection_dataset(
            self.images,
            self.labels,
            boxes,
            self.metadata,
            self.classes,
        )

        with pytest.raises(ValueError, match="Invalid bounding box format"):
            imagestats(dataset, per_box=True)

    def test_no_bounding_box(self):
        boxes = self.boxes.copy()
        boxes[-1].clear()

        dataset = to_object_detection_dataset(
            self.images,
            self.labels,
            boxes,
            self.metadata,
            self.classes,
        )

        stats = imagestats(dataset, per_box=True)
        assert stats.source_index[-1].image == 8

    def test_no_bounding_box_boxratio(self):
        boxes = self.boxes.copy()
        boxes[-1].clear()

        dataset = to_object_detection_dataset(
            self.images,
            self.labels,
            boxes,
            self.metadata,
            self.classes,
        )

        imgstats = imagestats(dataset, per_box=False)
        boxstats = imagestats(dataset, per_box=True)
        ratiostats = boxratiostats(boxstats, imgstats)
        assert ratiostats is not None


@pytest.mark.required
@pytest.mark.parametrize(
    "stats_fn, stats_attrs",
    (
        [pixelstats, "pixelstats"],
        [visualstats, "visualstats"],
        [dimensionstats, "dimensionstats"],
        [imagestats, "imagestats"],
        [hashstats, "hashstats"],
    ),
    indirect=["stats_attrs"],  # Only sends the string to fixture
)
class TestStatsDataFormats:
    def test_stats_data(self, get_ic_dataset, stats_fn, stats_attrs: set):
        """
        Test for *StatsOutput class has unique and BaseStatsOutput attributes
        """

        dataset = get_ic_dataset(DATA_1)
        data: dict = stats_fn(dataset).data()

        # Test keys align with BaseStatsOutput and *StatsOutput
        assert isinstance(data, dict)

        # Expects all attributes, including Base and 2-D attributes
        self._check_keys(
            keys=set(data),
            expected_attrs=(stats_attrs | set(BASE_ATTRS)),
            invalid_attrs=set(),
        )  # Empty set has no overlap

    def test_stats_factors(self, get_ic_dataset, stats_fn, stats_attrs: set):
        """
        Test for *StatsOutput class has unique 1-D attributes and no BaseStatsOutput attributes

        Note
        ----
        2-D attributes like histogram are not expected
        """

        dataset = get_ic_dataset(DATA_1)
        factors: dict = stats_fn(dataset).factors()

        # Test keys align only with *StatsOutput
        assert isinstance(factors, dict)
        self._check_keys(
            keys=set(factors),
            expected_attrs=(stats_attrs - STATS_2D_ATTRS),
            invalid_attrs=(set(BASE_ATTRS) | STATS_2D_ATTRS),
        )

    def test_stats_to_df(self, get_ic_dataset, stats_fn, stats_attrs: set):
        """
        Test for *StatsOutput class has unique 1-D attributes and no BaseStatsOutput attributes
        as columns in the dataframe

        Note
        ----
        2-D attributes like histogram are not expected
        """

        dataset = get_ic_dataset(DATA_1)
        dataframe: pl.DataFrame = stats_fn(dataset).to_dataframe()

        # Test dataframe keys align only with *StatsOutput
        assert isinstance(dataframe, pl.DataFrame)

        self._check_keys(
            keys=set(dataframe.columns),
            expected_attrs=(stats_attrs - STATS_2D_ATTRS),
            invalid_attrs=(set(BASE_ATTRS) | STATS_2D_ATTRS),
        )

        # Test final shape follows keys and homogeneous lengths
        rows = len(DATA_1)
        cols = len(stats_attrs - STATS_2D_ATTRS)
        assert dataframe.shape == (rows, cols)

    def _check_keys(self, keys: set, expected_attrs: set, invalid_attrs: set):
        """Checks public attributes equal returned keys"""
        assert "_meta" not in keys

        assert keys.issubset(expected_attrs)
        assert keys.isdisjoint(invalid_attrs)
