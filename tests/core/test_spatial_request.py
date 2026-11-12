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
from tests.conftest import _get_mock_od_dataset

pytestmark = [
    pytest.mark.required,
    pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning"),
]

SHAPE = (3, 20, 40)
BOX = (2, 2, 12, 12)
BOXES = [BOX, (20, 2, 30, 12)]
SPATIAL = ImageStats.SPATIAL
SPATIAL_NAMES = {"nn_distance_norm", "box_in_image_fraction"}


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
        image_stats(images(1), boxes=[BOXES], stats=ImageStats.PIXEL_MEAN | SPATIAL)


def test_the_experimental_warning_fires_once_on_spatial_and_never_on_the_default(fresh_process):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        image_stats(images(1), boxes=[BOXES], stats=ImageStats.ALL & ~ImageStats.BLUR & ~SPATIAL)
        image_stats(images(1))

    with pytest.warns(ExperimentalWarning, match=r"'ImageStats\.SPATIAL' is experimental"):
        image_stats(images(1), boxes=[BOXES], stats=SPATIAL)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        image_stats(images(1), boxes=[BOXES], stats=SPATIAL)


def test_all_raises_the_experimental_warning_for_spatial_and_says_how_to_leave_it_out(fresh_process):
    with pytest.warns(ExperimentalWarning, match=r"'ImageStats\.SPATIAL' is experimental") as record:
        image_stats(images(1), boxes=[BOXES], stats=ImageStats.ALL)

    spatial_warning = next(w for w in record if "ImageStats.SPATIAL" in str(w.message))
    assert "ImageStats.ALL & ~ImageStats.SPATIAL" in str(spatial_warning.message)


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
    ids=["all_but_spatial", "standard", "basic", "dimension", "dimension_box", "pixel", "visual", "hash", "blur"],
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

    assert "details" not in result
    assert not SPATIAL_NAMES & set(result["stats"])


def test_all_and_every_complement_hold_spatial_unless_it_is_named_out():
    assert ImageStats.ALL & SPATIAL == SPATIAL
    assert (~ImageStats.HASH) & SPATIAL == SPATIAL
    assert (ImageStats.ALL & ~ImageStats.HASH) & SPATIAL == SPATIAL
    assert not (ImageStats.ALL & ~SPATIAL) & SPATIAL


def test_all_computes_the_spatial_statistics_on_box_rows():
    boxes = [BOXES, [BOX]]

    everything = image_stats(images(2), boxes=boxes, stats=ImageStats.ALL)
    only_spatial = image_stats(images(2), boxes=boxes, stats=SPATIAL)

    assert [si.key for si in everything["source_index"]] == [si.key for si in only_spatial["source_index"]]
    for name in SPATIAL_NAMES:
        np.testing.assert_array_equal(everything["stats"][name], only_spatial["stats"][name], err_msg=name)
    np.testing.assert_array_equal(details_of(everything)["spatial_status"], details_of(only_spatial)["spatial_status"])
    assert np.isfinite(everything["stats"]["mean"]).all()


def test_all_without_box_rows_warns_once_and_returns_nan_columns():
    way_out = r"SPATIAL_NN_DISTANCE_NORM.*ImageStats\.ALL & ~ImageStats\.SPATIAL"
    with pytest.warns(UserWarning, match=way_out) as record, quiet_experimental():
        result = image_stats(images(1), stats=ImageStats.ALL)

    assert len([w for w in record if not issubclass(w.category, ExperimentalWarning)]) == 1
    for name in SPATIAL_NAMES:
        assert np.isnan(result["stats"][name]).all(), name
    assert np.isfinite(result["stats"]["mean"]).all()


def test_a_dataset_and_explicit_boxes_give_identical_spatial_output():
    items = [(BOXES, [0, 1]), ([BOX], [0]), ([], [])]
    from_dataset = image_stats(_Dataset(items), stats=SPATIAL)
    explicit = image_stats(images(3), boxes=[BOXES, [BOX], []], stats=SPATIAL)

    for name in from_dataset["stats"]:
        np.testing.assert_array_equal(from_dataset["stats"][name], explicit["stats"][name], err_msg=name)
    for name in details_of(from_dataset):
        np.testing.assert_array_equal(details_of(from_dataset)[name], details_of(explicit)[name], err_msg=name)


def test_a_multi_process_run_matches_a_single_process_run():
    boxes = [BOXES, [BOX], [BOX, (14, 2, 24, 12), (30, 5, 38, 15)], BOXES]
    previous = config.get_max_processes()
    try:
        config.set_max_processes(1)
        single = image_stats(images(4), boxes=boxes, stats=SPATIAL)
        config.set_max_processes(2)
        multi = image_stats(images(4), boxes=boxes, stats=SPATIAL)
    finally:
        config.set_max_processes(previous)

    for one, other in ((single["stats"], multi["stats"]), (details_of(single), details_of(multi))):
        assert one.keys() == other.keys()
        for name in one:
            np.testing.assert_array_equal(one[name], other[name], err_msg=name)


def _od_dataset_with_one_isolated_pair():
    """Thirty images of two close boxes, one image whose two boxes are far apart, and one image with a lone box."""
    rng = np.random.default_rng(0)
    pictures = [rng.integers(100, 140, (3, 64, 64), dtype=np.uint8) for _ in range(32)]
    labels: list[list[int]] = [[0, 1] for _ in range(30)] + [[0], [0]]
    boxes = [[[2, 2 + (i % 3), 10, 10 + (i % 3)], [12 + (i % 2), 2, 20 + (i % 2), 10]] for i in range(30)]
    boxes[7] = [[2, 2, 10, 10], [52, 2, 60, 10]]
    boxes += [[[2, 2, 10, 10]], [[2, 2, 10, 10]]]
    return _get_mock_od_dataset(pictures, labels, boxes)


def test_metadata_ignores_the_details_and_drops_the_all_null_image_half():
    dataset = _od_dataset_with_one_isolated_pair()
    result = image_stats(dataset, stats=SPATIAL | ImageStats.PIXEL_MEAN)
    metadata = Metadata(cast(Any, dataset))

    metadata.add_factors(result)

    names = set(metadata.factor_names)
    assert {"instance_nn_distance_norm", "instance_box_in_image_fraction"} <= names
    assert not {name for name in names if name.startswith("unit_") and any(s in name for s in SPATIAL_NAMES)}
    assert not {name for name in names if "spatial_status" in name or "reason" in name or "target_index" in name}
    assert metadata.dropped_factors["unit_nn_distance_norm"] == ["no_values_at_level"]


@pytest.mark.parametrize("flags", [SPATIAL, ImageStats.ALL], ids=["spatial", "all"])
def test_outliers_judges_an_isolated_target_on_the_per_target_path_and_skips_a_lone_box(flags):
    out = Outliers(flags=flags).evaluate(_od_dataset_with_one_isolated_pair(), per_image=False, per_target=True)

    rows = out.data()
    judged = rows.filter(rows["metric_name"].cast(str) == "nn_distance_norm")
    assert judged["item_index"].to_list() == [7, 7]  # the isolated pair; the lone boxes read NaN and are not judged
    assert sorted(judged["target_index"].to_list()) == [0, 1]


@pytest.mark.parametrize("flags", [SPATIAL | ImageStats.PIXEL_MEAN, ImageStats.ALL], ids=["spatial_and_pixel", "all"])
def test_outliers_default_path_warns_and_judges_no_spatial_value(flags):
    with pytest.warns(UserWarning, match="SPATIAL_NN_DISTANCE_NORM"):
        out = Outliers(flags=flags).evaluate(_od_dataset_with_one_isolated_pair())

    assert not SPATIAL_NAMES & set(out.data()["metric_name"].cast(str).to_list())
