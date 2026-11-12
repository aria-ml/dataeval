import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.core._calculators._spatialstats import SpatialStatCalculator, _nearest_peer
from dataeval.core._image_stats import _collect_calculator_stats
from dataeval.flags import ImageStats
from dataeval.utils.preprocessing import BoundingBox

pytestmark = [
    pytest.mark.required,
    pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning"),
]

SHAPE = (3, 20, 40)  # C, H, W
NAN = float("nan")


def details_of(result):
    """The result's details, which a result carries only when a requested statistic reports some."""
    assert "details" in result
    return result["details"]


def spatial(boxes, shape=SHAPE, stats=ImageStats.SPATIAL, **kw):
    """One image, box rows only (per_image=False), so row i is box i."""
    return image_stats([np.zeros(shape)], boxes=[boxes], stats=stats, per_image=False, **kw)


def test_spatial_flags_are_appended_and_inside_all():
    assert ImageStats.VISUAL_BLUR_DIRECTIONALITY.value < ImageStats.SPATIAL_NN_DISTANCE_NORM.value
    assert ImageStats.ALL & ImageStats.SPATIAL == ImageStats.SPATIAL
    assert ImageStats.SPATIAL == ImageStats.SPATIAL_NN_DISTANCE_NORM | ImageStats.SPATIAL_BOX_IN_IMAGE_FRACTION


def test_box_in_image_fraction_values():
    boxes = [(5, 5, 15, 15), (30, 0, 50, 10), (-20, 0, 10, 10), (1.5, 2.25, 10.75, 9.5)]

    result = spatial(boxes, stats=ImageStats.SPATIAL_BOX_IN_IMAGE_FRACTION)

    fraction = result["stats"]["box_in_image_fraction"]
    assert set(result["stats"]) == {"box_in_image_fraction"}
    assert len(result["source_index"]) == 4
    assert fraction[0] == 1.0
    assert fraction[1] == 0.5
    assert fraction[2] == pytest.approx(1 / 3)
    assert fraction[3] == 1.0
    assert details_of(result)["spatial_status"].tolist() == ["ok"] * 4


@pytest.mark.parametrize(
    ("bad", "valid_status"),
    [
        ((5, 5, 5, 15), "ok"),  # zero area
        ((50, 0, 60, 10), "ok"),  # entirely outside
        ((15, 5, 5, 15), "incomplete_geometry"),  # reversed
        ((NAN, 0, 10, 10), "incomplete_geometry"),  # nonfinite
    ],
    ids=["zero_area", "outside", "reversed", "nonfinite"],
)
def test_box_in_image_fraction_is_nan_for_an_invalid_box(bad, valid_status):
    result = spatial([(1, 1, 9, 9), bad], stats=ImageStats.SPATIAL_BOX_IN_IMAGE_FRACTION)

    fraction = result["stats"]["box_in_image_fraction"]
    assert fraction[0] == 1.0
    assert np.isnan(fraction[1])
    assert details_of(result)["spatial_status"].tolist() == [valid_status, "invalid_box"]


def test_image_rows_are_nan_and_not_applicable():
    result = image_stats([np.zeros(SHAPE)], boxes=[[(1, 1, 9, 9)]], stats=ImageStats.SPATIAL)

    assert [si.key for si in result["source_index"]] == [None, 0]
    fraction = result["stats"]["box_in_image_fraction"]
    assert np.isnan(fraction[0])
    assert fraction[1] == 1.0
    assert details_of(result)["spatial_status"].tolist() == ["not_applicable", "ok"]


def test_images_of_different_sizes_use_their_own_extent():
    result = image_stats(
        [np.zeros((3, 20, 40)), np.zeros((3, 10, 10))],
        boxes=[[(0, 0, 20, 20)], [(0, 0, 20, 20)]],
        stats=ImageStats.SPATIAL_BOX_IN_IMAGE_FRACTION,
        per_image=False,
    )

    assert result["stats"]["box_in_image_fraction"].tolist() == [1.0, 0.25]


def test_no_band_or_background_copy_is_emitted():
    result = spatial(
        [(1, 1, 9, 9)],
        stats=ImageStats.SPATIAL | ImageStats.PIXEL_MEAN,
        per_background=True,
        channels={"a": 0},
    )

    names = set(result["stats"])
    # The band group and the background did run for the pixel statistic, and `background_fraction` is the share of the
    # image left unmasked, not a SPATIAL statistic.
    assert {"a_mean", "background_mean", "background_fraction", "box_in_image_fraction"} <= names
    assert not {"a_box_in_image_fraction", "background_box_in_image_fraction"} & names
    assert not {name for name in names if "nn_distance" in name} - {"nn_distance_norm"}


def test_a_one_dimensional_datum_and_a_missing_context_give_nan():
    stats, details, _, _ = _collect_calculator_stats([(SpatialStatCalculator, ImageStats.SPATIAL)], np.zeros(10), None)

    assert np.isnan(stats[0]["box_in_image_fraction"][0])
    assert details[0]["spatial_status"] == ["not_applicable"]


NN = ImageStats.SPATIAL_NN_DISTANCE_NORM


def nn(boxes, shape=SHAPE):
    """The values, nearest-target indices, reasons and statuses of `boxes`, in box order."""
    result = spatial(boxes, shape=shape, stats=NN)
    details = details_of(result)
    return (
        result["stats"]["nn_distance_norm"],
        details["nn_distance_norm_target_index"],
        details["nn_distance_norm_reason"],
        details["spatial_status"],
    )


@pytest.mark.parametrize("as_boxes", [False, True], ids=["tuples", "bounding_boxes"])
def test_center_distance_and_diagonal(as_boxes):
    boxes = [(0, 0, 10, 10), (30, 0, 40, 10)]
    if as_boxes:
        boxes = [BoundingBox(*b, image_shape=SHAPE) for b in boxes]

    value, index, reason, status = nn(boxes)

    assert value.tolist() == pytest.approx([30 / np.hypot(10, 10)] * 2, rel=1e-6)
    assert index.tolist() == [1, 0]
    assert reason.tolist() == ["", ""]
    assert status.tolist() == ["ok", "ok"]


def test_the_distance_is_directed_and_not_mutual():
    value, _, _, _ = nn([(0, 0, 80, 80), (60, 0, 70, 10)], shape=(3, 100, 100))
    assert value.tolist() == pytest.approx(
        [np.hypot(25, 35) / np.hypot(80, 80), np.hypot(25, 35) / np.hypot(10, 10)], rel=1e-6
    )

    # The first box's nearest is the second, but the second's nearest is the third.
    _, index, _, _ = nn([(0, 0, 10, 10), (20, 0, 30, 10), (30, 0, 40, 10)])
    assert index.tolist() == [1, 2, 1]


def test_original_geometry_is_used_not_the_clipped_box():
    value, _, _, _ = nn([(-20, 0, 10, 10), (30, 0, 40, 10)])

    # Clipped to the image the box would be (0, 0, 10, 10) and read 30 / hypot(10, 10).
    assert value[0] == pytest.approx(40 / np.hypot(30, 10), rel=1e-6)


@pytest.mark.parametrize(
    "boxes",
    [[(1, 1, 9, 9)], [(1, 1, 9, 9), (5, 5, 5, 15), (50, 0, 60, 10)]],
    ids=["alone", "zero_area_and_outside_peers"],
)
def test_no_neighbor(boxes):
    value, index, reason, status = nn(boxes)

    assert np.isnan(value[0])
    assert (index[0], reason[0], status[0]) == (-1, "no_neighbor", "ok")


def test_a_bad_peer_does_not_blank_the_frame():
    value, index, reason, status = nn([(0, 0, 10, 10), (20, 0, 30, 10), (NAN, 0, 5, 5)])

    assert value[:2].tolist() == pytest.approx([20 / np.hypot(10, 10)] * 2, rel=1e-6)
    assert index.tolist() == [1, 0, -1]
    assert status.tolist() == ["incomplete_geometry", "incomplete_geometry", "invalid_box"]
    assert np.isnan(value[2])
    assert reason.tolist() == ["", "", ""]


def test_a_valid_box_whose_only_other_box_is_corrupt_has_no_value_and_no_reason():
    value, index, reason, status = nn([(0, 0, 10, 10), (10, 0, 2, 10)])

    assert np.isnan(value[0])
    assert (index[0], reason[0], status[0]) == (-1, "", "incomplete_geometry")


def test_duplicate_boxes_read_zero_and_the_lowest_index_is_nearest():
    value, index, _, _ = nn([(0, 0, 10, 10)] * 3)
    assert value.tolist() == [0.0, 0.0, 0.0]
    assert index.tolist() == [1, 0, 0]

    # The middle box is 20 from each of the others, so the lower index wins the tie.
    _, index, _, _ = nn([(40, 0, 50, 10), (20, 0, 30, 10), (0, 0, 10, 10)], shape=(3, 20, 60))
    assert index[1] == 0


def test_nearest_peer_does_not_overflow():
    raw = np.array([[0, 0, 1e200, 1e200], [1e200, 1e200, 3e200, 3e200]])

    found = _nearest_peer(raw, np.array([1]), 0)

    assert found is not None
    assert found[0] == 1
    assert found[1] == pytest.approx(1.5)


def test_reordering_permutes_values_and_indices():
    boxes = [(0, 0, 10, 10), (12, 0, 22, 10), (40, 0, 50, 10), (100, 0, 110, 10), (30, 40, 38, 48), (150, 60, 160, 70)]
    perm = [3, 5, 0, 4, 1, 2]
    shape = (3, 100, 200)

    value, index, _, _ = nn(boxes, shape=shape)
    shuffled_value, shuffled_index, _, _ = nn([boxes[p] for p in perm], shape=shape)

    assert index.tolist() == [1, 0, 1, 2, 2, 3]  # no ties, so the permutation is the whole difference
    np.testing.assert_array_equal(shuffled_value, value[perm])
    where = np.argsort(perm)
    assert shuffled_index.tolist() == [where[index[p]] for p in perm]


def test_only_the_selected_statistics_add_details():
    only_fraction = spatial([(1, 1, 9, 9), (20, 5, 30, 15)], stats=ImageStats.SPATIAL_BOX_IN_IMAGE_FRACTION)
    only_nn = spatial([(1, 1, 9, 9), (20, 5, 30, 15)], stats=NN)

    assert set(details_of(only_fraction)) == {"spatial_status"}
    assert set(only_nn["stats"]) == {"nn_distance_norm"}
    assert set(details_of(only_nn)) == {"spatial_status", "nn_distance_norm_target_index", "nn_distance_norm_reason"}
    assert details_of(only_nn)["nn_distance_norm_target_index"].dtype == np.int64
    assert details_of(only_nn)["spatial_status"].dtype.kind == "U"
    assert details_of(only_nn)["nn_distance_norm_reason"].dtype.kind == "U"


def test_a_reversed_bounding_box_reads_ok_because_its_constructor_has_swapped_the_edges():
    # The documented limit: only the coordinates as given can be reversed, and a `BoundingBox` is never given reversed.
    _, _, _, as_box = nn([BoundingBox(10, 0, 0, 10, image_shape=SHAPE), (30, 0, 40, 10)])
    _, _, _, as_tuple = nn([(10, 0, 0, 10), (30, 0, 40, 10)])

    assert as_box.tolist() == ["ok", "ok"]
    assert as_tuple.tolist() == ["invalid_box", "incomplete_geometry"]
