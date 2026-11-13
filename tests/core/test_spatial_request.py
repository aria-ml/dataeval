import warnings
from typing import Any, cast

import numpy as np
import pytest

import dataeval._experimental as experimental_module
from dataeval import Metadata, config
from dataeval._experimental import quiet_experimental
from dataeval.core import image_stats
from dataeval.exceptions import ExperimentalWarning
from dataeval.flags import ImageStats
from dataeval.protocols import DatasetMetadata, DatumMetadata
from dataeval.quality import Outliers
from dataeval.utils._annotations import AnnotationContext
from dataeval.utils.thresholds import AdaptiveThreshold, ConstantThreshold
from tests.conftest import _get_mock_od_dataset

pytestmark = [
    pytest.mark.required,
    pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning"),
]

SHAPE = (3, 20, 40)
BOX = (2, 2, 12, 12)
BOXES = [BOX, (20, 2, 30, 12)]
OVERLAPPING = [BOX, (6, 2, 16, 12)]  # IoU 2/3
SPATIAL = ImageStats.SPATIAL
SPATIAL_NAMES = {
    "nn_distance_norm",
    "box_in_image_fraction",
    "covered_fraction",
    "max_iou_same_class",
    "max_iou_other_class",
}


def images(n: int) -> list[np.ndarray]:
    return [np.random.default_rng(i).random(SHAPE) for i in range(n)]


class _Target:
    def __init__(self, boxes: Any, labels: Any) -> None:
        self.boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        self.labels = np.asarray(labels)
        self.scores = np.ones((len(self.labels), 1), dtype=np.float32)


class _Dataset:
    """A small object-detection dataset: one image per item, `images(len)` pixels."""

    metadata = DatasetMetadata(id="spatial_request")

    def __init__(self, items: list[tuple[Any, Any]]) -> None:
        self.targets = [_Target(boxes, labels) for boxes, labels in items]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int) -> Any:
        return images(len(self))[i], self.targets[i], DatumMetadata(id=i)


def details_of(result):
    """The result's details, which a result carries only when a requested statistic reports some."""
    assert "details" in result
    return result["details"]


@pytest.fixture
def fresh_process(monkeypatch):
    """Start the test with no experimental warning given yet, as in a new process."""
    monkeypatch.setattr(experimental_module, "_WARNED", set())


@pytest.mark.parametrize(
    "call",
    [
        lambda: image_stats(images(1), stats=SPATIAL),
        lambda: image_stats(images(1), boxes=[[BOX]], stats=SPATIAL, per_target=False),
        lambda: image_stats(_Dataset([([BOX], [0])]), stats=SPATIAL, per_target=False),
    ],
    ids=["no_boxes", "per_target_false", "dataset_per_target_false"],
)
def test_spatial_alone_without_box_rows_raises(call):
    with pytest.raises(ValueError, match=r"boxes=.*per_target=True"):
        call()


def test_a_mixed_request_warns_once_and_returns_nan_columns():
    with pytest.warns(UserWarning, match="SPATIAL_NN_DISTANCE_NORM") as record, quiet_experimental():
        result = image_stats(images(1), stats=ImageStats.PIXEL_MEAN | SPATIAL)

    assert len([w for w in record if not issubclass(w.category, ExperimentalWarning)]) == 1
    assert np.isnan(result["stats"]["nn_distance_norm"]).all()
    assert np.isfinite(result["stats"]["mean"]).all()


def test_a_request_with_box_rows_warns_nothing():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        image_stats(images(1), boxes=[BOXES], labels=[[0, 1]], stats=ImageStats.PIXEL_MEAN | SPATIAL)


@pytest.mark.parametrize(
    "stats",
    [
        SPATIAL,
        ImageStats.SPATIAL_COVERED_FRACTION,
        ImageStats.SPATIAL_NN_DISTANCE_NORM,
        ImageStats.ALL & ~ImageStats.BLUR,
    ],
)
def test_spatial_is_stable_and_raises_no_experimental_warning(fresh_process, stats):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        image_stats(images(1), boxes=[BOXES], labels=[[0, 1]], stats=stats)


def test_all_still_warns_about_blur_and_never_names_spatial(fresh_process):
    with pytest.warns(ExperimentalWarning, match=r"'ImageStats\.BLUR' is experimental") as record:
        image_stats(images(1), boxes=[BOXES], stats=ImageStats.ALL)

    assert not any("SPATIAL" in str(w.message) for w in record)


@pytest.mark.parametrize("source", ["boxes", "dataset", "no_boxes"])
@pytest.mark.parametrize(
    "group",
    [
        ImageStats.ALL & ~ImageStats.SPATIAL,
        ImageStats.STANDARD,
        ImageStats.BASIC,
        ImageStats.DIMENSION,
        ImageStats.DIMENSION_BOX,
        ImageStats.PIXEL,
        ImageStats.VISUAL,
        ImageStats.HASH,
        ImageStats.BLUR,
    ],
    ids=[
        "all_but_spatial",
        "standard",
        "basic",
        "dimension",
        "dimension_box",
        "pixel",
        "visual",
        "hash",
        "blur",
    ],
)
def test_every_group_without_spatial_is_unchanged(monkeypatch, group, source):
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a context was built for a request that includes no SPATIAL statistic")

    monkeypatch.setattr(AnnotationContext, "__init__", refuse)

    if source == "boxes":
        result = image_stats(images(2), boxes=[BOXES, [BOX]], stats=group)
    elif source == "dataset":
        result = image_stats(_Dataset([(BOXES, [0, 1]), ([BOX], [0])]), stats=group)
    else:
        result = image_stats(images(2), stats=group)

    # `PIXEL` holds saturation, which reports details of its own; no other group reports any.
    assert all(name.startswith("saturation_") for name in result.get("details", {}))
    assert not SPATIAL_NAMES & set(result["stats"])


def test_all_and_every_complement_hold_spatial_unless_it_is_named_out():
    assert ImageStats.ALL & SPATIAL == SPATIAL
    assert (~ImageStats.HASH) & SPATIAL == SPATIAL
    assert (ImageStats.ALL & ~ImageStats.HASH) & SPATIAL == SPATIAL
    assert not (ImageStats.ALL & ~SPATIAL) & SPATIAL


def test_all_computes_the_spatial_statistics_on_box_rows():
    boxes = [BOXES, [BOX]]

    labels = [[0, 1], [0]]
    everything = image_stats(images(2), boxes=boxes, labels=labels, stats=ImageStats.ALL)
    only_spatial = image_stats(images(2), boxes=boxes, labels=labels, stats=SPATIAL)

    assert [si.key for si in everything["source_index"]] == [si.key for si in only_spatial["source_index"]]
    for name in SPATIAL_NAMES:
        np.testing.assert_array_equal(everything["stats"][name], only_spatial["stats"][name], err_msg=name)
    np.testing.assert_array_equal(details_of(everything)["spatial_status"], details_of(only_spatial)["spatial_status"])
    assert np.isfinite(everything["stats"]["mean"]).all()


def test_all_without_box_rows_is_silent_and_returns_nan_columns():
    # `ALL` asks for everything, so a statistic that cannot be measured reads NaN, as on any row it cannot measure.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with quiet_experimental():
            result = image_stats(images(1), stats=ImageStats.ALL)

    for name in SPATIAL_NAMES:
        assert np.isnan(result["stats"][name]).all(), name
    assert np.isfinite(result["stats"]["mean"]).all()


def test_a_flow_shaped_call_warns_nothing():
    """dataeval-flow's default call: `ALL` on a detection dataset, per image, per_target=False."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with quiet_experimental():
            result = image_stats(
                _Dataset([(BOXES, [0, 1]), ([BOX], [0])]), stats=ImageStats.ALL, per_image=True, per_target=False
            )

    assert len(result["source_index"]) == 2
    for name in SPATIAL_NAMES:
        assert np.isnan(result["stats"][name]).all(), name


def test_a_request_short_of_all_still_warns_once_without_naming_all():
    with pytest.warns(UserWarning, match="SPATIAL_NN_DISTANCE_NORM") as record, quiet_experimental():
        image_stats(images(1), stats=ImageStats.ALL & ~ImageStats.BLUR)

    messages = [str(w.message) for w in record if not issubclass(w.category, ExperimentalWarning)]
    assert len(messages) == 1
    assert "ImageStats.ALL" not in messages[0]
    assert messages[0].endswith("or drop them from stats.")


def test_the_whole_group_with_explicit_boxes_and_no_labels_warns_once_naming_the_class_statistics():
    with pytest.warns(UserWarning, match=r"max_iou_same_class|SPATIAL_MAX_IOU_SAME_CLASS") as record:
        result = image_stats(images(1), boxes=[[BOX, (6, 2, 16, 12)]], stats=SPATIAL)

    messages = [str(w.message) for w in record if not issubclass(w.category, ExperimentalWarning)]
    assert len(messages) == 1
    assert "SPATIAL_MAX_IOU_SAME_CLASS, SPATIAL_MAX_IOU_OTHER_CLASS compare a box with the class" in messages[0]
    assert "SPATIAL_NN_DISTANCE_NORM" not in messages[0]
    assert "SPATIAL_COVERED_FRACTION" not in messages[0]
    assert np.isnan(result["stats"]["max_iou_same_class"]).all()  # every label is missing and a peer overlaps each box


def test_all_on_a_dataset_with_float_labels_measures_every_box_without_a_warning_or_an_error():
    """A float label array says "unlabeled" with NaN, and a loader may leave its ignore id, -1, in it."""
    first = [(2, 2, 12, 12), (6, 2, 16, 12), (30, 2, 38, 12)]
    pair = [(2, 2, 12, 12), (4, 2, 14, 12)]
    labeled = [
        (first, np.asarray([0.0, np.nan, 1.0], dtype=np.float32)),  # the overlapping pair has a missing label
        (pair, np.asarray([-1.0, 3.0], dtype=np.float32)),
        (pair, np.asarray([2.0, 3.0], dtype=np.float32)),  # both known, different classes
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with quiet_experimental():
            result = image_stats(_Dataset(labeled), stats=ImageStats.ALL)
            flow = image_stats(_Dataset(labeled), stats=ImageStats.ALL, per_target=False)

    boxes = np.array([si.key is not None for si in result["source_index"]])
    same, other = (result["stats"][f"max_iou_{name}_class"][boxes] for name in ("same", "other"))
    assert boxes.sum() == 7
    # The pair with a missing label cannot say which relationship it has: NaN, and the isolated box reads 0.0.
    assert np.isnan(same[[0, 1, 3, 4]]).all()
    assert np.isnan(other[[0, 1, 3, 4]]).all()
    assert same[[2, 5, 6]].tolist() == [0.0, 0.0, 0.0]
    assert other[[2]].tolist() == [0.0]
    assert other[[5, 6]].tolist() == pytest.approx([2 / 3, 2 / 3], rel=1e-6)
    reasons = details_of(result)["max_iou_other_class_reason"][boxes]
    assert reasons.tolist() == ["missing_labels"] * 2 + [""] + ["missing_labels"] * 2 + ["", ""]
    # The flow-shaped call reads no box rows and no labels, and says nothing.
    assert len(flow["source_index"]) == 3
    assert np.isnan(flow["stats"]["max_iou_other_class"]).all()


def test_all_is_everything_and_its_complement_is_nothing():
    assert ~ImageStats.ALL == ImageStats.NONE


def test_a_dataset_and_explicit_boxes_give_identical_spatial_output():
    items = [(OVERLAPPING, [0, 1]), ([BOX], [0]), ([], [])]
    from_dataset = image_stats(_Dataset(items), stats=SPATIAL)
    explicit = image_stats(images(3), boxes=[OVERLAPPING, [BOX], []], labels=[[0, 1], [0], []], stats=SPATIAL)

    # Not vacuous: the first image's two boxes overlap and are of different classes, so a dataset whose labels were
    # dropped or misread could not match.
    assert (from_dataset["stats"]["covered_fraction"] > 0).any()
    assert (from_dataset["stats"]["max_iou_other_class"] > 0).any()

    for name in from_dataset["stats"]:
        np.testing.assert_array_equal(from_dataset["stats"][name], explicit["stats"][name], err_msg=name)
    for name in details_of(from_dataset):
        np.testing.assert_array_equal(details_of(from_dataset)[name], details_of(explicit)[name], err_msg=name)


def test_a_multi_process_run_matches_a_single_process_run():
    boxes = [OVERLAPPING, [BOX], [BOX, (6, 2, 16, 12), (14, 2, 24, 12)], OVERLAPPING]
    labels = [[0, 1], [0], [0, 0, 1], [1, -1]]
    previous = config.get_max_processes()
    try:
        config.set_max_processes(1)
        single = image_stats(images(4), boxes=boxes, labels=labels, stats=SPATIAL)
        config.set_max_processes(2)
        multi = image_stats(images(4), boxes=boxes, labels=labels, stats=SPATIAL)
    finally:
        config.set_max_processes(previous)

    # Not vacuous: boxes overlap, with a missing label among them, so each new column holds a value to compare.
    for name in ("covered_fraction", "max_iou_same_class", "max_iou_other_class"):
        assert np.nanmax(single["stats"][name]) > 0, name
    assert np.isnan(single["stats"]["max_iou_other_class"]).any()
    for one, other in ((single["stats"], multi["stats"]), (details_of(single), details_of(multi))):
        assert one.keys() == other.keys()
        for name in one:
            np.testing.assert_array_equal(one[name], other[name], err_msg=name)


def _od_dataset_with_one_isolated_pair(overlapped: bool = False):
    """Thirty images of two close boxes, one image whose two boxes are far apart, and one image with a lone box.

    With `overlapped`, that one image instead holds two boxes of different classes that nearly coincide.
    """
    rng = np.random.default_rng(0)
    pictures = [rng.integers(100, 140, (3, 64, 64), dtype=np.uint8) for _ in range(32)]
    labels: list[list[int]] = [[0, 1] for _ in range(30)] + [[0], [0]]
    boxes = [[[2, 2 + (i % 3), 10, 10 + (i % 3)], [12 + (i % 2), 2, 20 + (i % 2), 10]] for i in range(30)]
    boxes[7] = [[2, 2, 10, 10], [3, 2, 11, 10] if overlapped else [52, 2, 60, 10]]
    boxes += [[[2, 2, 10, 10]], [[2, 2, 10, 10]]]
    return _get_mock_od_dataset(pictures, labels, boxes)


def test_metadata_ignores_the_details_and_drops_the_all_null_image_half():
    dataset = _od_dataset_with_one_isolated_pair()
    result = image_stats(dataset, stats=SPATIAL | ImageStats.PIXEL_MEAN)
    metadata = Metadata(cast(Any, dataset))

    metadata.add_factors(result)

    names = set(metadata.factor_names)
    assert {f"instance_{name}" for name in SPATIAL_NAMES} <= names
    assert not {name for name in names if name.startswith("unit_") and any(s in name for s in SPATIAL_NAMES)}
    assert not {name for name in names if "spatial_status" in name or "reason" in name or "target_index" in name}
    for name in SPATIAL_NAMES:
        assert metadata.dropped_factors[f"unit_{name}"] == ["no_values_at_level"], name


@pytest.mark.parametrize("flags", [SPATIAL, ImageStats.ALL], ids=["spatial", "all"])
def test_outliers_judges_an_isolated_target_on_the_per_target_path_and_skips_a_lone_box(flags):
    out = Outliers(flags=flags).evaluate(_od_dataset_with_one_isolated_pair(), per_image=False, per_target=True)

    rows = out.data()
    judged = rows.filter(rows["metric_name"].cast(str) == "nn_distance_norm")
    assert judged["item_index"].to_list() == [7, 7]  # the isolated pair; the lone boxes read NaN and are not judged
    assert sorted(judged["target_index"].to_list()) == [0, 1]


def test_outliers_of_all_judges_the_overlap_columns_on_the_per_target_path():
    out = Outliers(flags=ImageStats.ALL).evaluate(
        _od_dataset_with_one_isolated_pair(overlapped=True), per_image=False, per_target=True
    )

    rows = out.data()
    names = rows["metric_name"].cast(str)
    for name in ("covered_fraction", "max_iou_other_class"):
        judged = rows.filter(names == name)
        assert sorted(judged["item_index"].to_list()) == [7, 7], name  # the two nearly coinciding boxes, nothing else


def test_the_default_outlier_threshold_can_sit_above_the_largest_value_of_a_zero_heavy_column():
    # The v1.3 page says so: a column that runs from 0 to 1 and is mostly 0, as an overlap column can be.
    column = np.concatenate([np.zeros(650), np.linspace(0.01, 0.92, 350)])

    upper = AdaptiveThreshold()(column)[1]

    assert upper is not None
    assert upper > 1.0


@pytest.mark.parametrize(("upper", "flagged"), [(0.5, [7, 7]), (0.99, [])])
def test_a_threshold_for_one_column_decides_what_outliers_flags_in_it(upper, flagged):
    # The v1.3 page's example. The pair's IoU is 7/9, so 0.5 flags it and 0.99 does not, where the default flags it.
    out = Outliers(
        flags=ImageStats.SPATIAL, outlier_threshold={"max_iou_other_class": ConstantThreshold(upper=upper)}
    ).evaluate(_od_dataset_with_one_isolated_pair(overlapped=True), per_image=False, per_target=True)

    rows = out.data()
    judged = rows.filter(rows["metric_name"].cast(str) == "max_iou_other_class")
    assert sorted(judged["item_index"].to_list()) == flagged


def test_outliers_default_path_warns_and_judges_no_spatial_value():
    with pytest.warns(UserWarning, match="SPATIAL_NN_DISTANCE_NORM"):
        out = Outliers(flags=SPATIAL | ImageStats.PIXEL_MEAN).evaluate(_od_dataset_with_one_isolated_pair())

    assert not SPATIAL_NAMES & set(out.data()["metric_name"].cast(str).to_list())


def test_outliers_of_all_on_the_default_path_warns_nothing_and_judges_no_spatial_value():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = Outliers(flags=ImageStats.ALL).evaluate(_od_dataset_with_one_isolated_pair())

    # `ALL` also drops its hash columns from the factors with a warning of its own, which is not what is asked here.
    assert not [w for w in record if "SPATIAL" in str(w.message) and "box rows" in str(w.message)]
    assert not SPATIAL_NAMES & set(out.data()["metric_name"].cast(str).to_list())
