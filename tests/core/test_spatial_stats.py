import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.core._calculators import _spatialstats
from dataeval.core._calculators._spatialstats import (
    SpatialStatCalculator,
    _intersections,
    _nearest_peer,
    _union_fraction,
)
from dataeval.core._image_stats import _collect_calculator_stats
from dataeval.flags import ImageStats
from dataeval.utils.preprocessing import BoundingBox

pytestmark = pytest.mark.required

SHAPE = (3, 20, 40)  # C, H, W
NAN = float("nan")


def details_of(result):
    """The result's details, which a result carries only when a requested statistic reports some."""
    assert "details" in result
    return result["details"]


def spatial(boxes, shape=SHAPE, stats=ImageStats.SPATIAL, **kw):
    """One image, box rows only (per_image=False), so row i is box i. Each box has its own class unless `labels`."""
    kw.setdefault("labels", [list(range(len(boxes)))])
    return image_stats([np.zeros(shape)], boxes=[boxes], stats=stats, per_image=False, **kw)


def test_spatial_flags_are_appended_and_inside_all():
    assert ImageStats.VISUAL_BLUR_DIRECTIONALITY.value < ImageStats.SPATIAL_NN_DISTANCE_NORM.value
    assert ImageStats.SPATIAL_BOX_IN_IMAGE_FRACTION.value < ImageStats.SPATIAL_COVERED_FRACTION.value
    assert ImageStats.ALL & ImageStats.SPATIAL == ImageStats.SPATIAL
    assert ImageStats.SPATIAL_COVERED_FRACTION.value < ImageStats.SPATIAL_MAX_IOU_SAME_CLASS.value
    assert ImageStats.SPATIAL_MAX_IOU_SAME_CLASS.value < ImageStats.SPATIAL_MAX_IOU_OTHER_CLASS.value
    assert (
        ImageStats.SPATIAL
        == ImageStats.SPATIAL_NN_DISTANCE_NORM
        | ImageStats.SPATIAL_BOX_IN_IMAGE_FRACTION
        | ImageStats.SPATIAL_COVERED_FRACTION
        | ImageStats.SPATIAL_MAX_IOU_SAME_CLASS
        | ImageStats.SPATIAL_MAX_IOU_OTHER_CLASS
    )


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
    result = image_stats([np.zeros(SHAPE)], boxes=[[(1, 1, 9, 9)]], labels=[[0]], stats=ImageStats.SPATIAL)

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


COVERED = ImageStats.SPATIAL_COVERED_FRACTION


def covered(boxes, shape=SHAPE):
    """The covered fractions and the details of `boxes`, in box order."""
    result = spatial(boxes, shape=shape, stats=COVERED)
    assert set(result["stats"]) == {"covered_fraction"}
    return result["stats"]["covered_fraction"], details_of(result)


# The first box is the focal one; the second element is its covered fraction, exact where the arithmetic is.
COVERED_CASES = {
    "half_covered": ([(0, 0, 10, 10), (0, 0, 5, 10)], 0.5),  # not read against the hull of its peers
    "two_halves_cover_it": ([(0, 0, 10, 10), (0, 0, 6, 10), (4, 0, 10, 10)], 1.0),
    "alone": ([(0, 0, 10, 10)], 0.0),
    "no_overlap": ([(0, 0, 10, 10), (20, 0, 30, 10)], 0.0),
    "touching_edges": ([(0, 0, 10, 10), (10, 0, 20, 10), (0, 10, 10, 20)], 0.0),
    "duplicate": ([(0, 0, 10, 10), (0, 0, 10, 10)], 1.0),
    "peer_contains_it": ([(2, 2, 8, 8), (0, 0, 10, 10)], 1.0),
    "it_contains_the_peer": ([(0, 0, 10, 10), (2, 2, 8, 8)], 0.36),
    "overlapping_peers_count_once": ([(0, 0, 10, 10), (0, 0, 6, 10), (0, 0, 10, 6)], 0.84),
}


@pytest.mark.parametrize(("boxes", "expected"), COVERED_CASES.values(), ids=COVERED_CASES.keys())
def test_covered_fraction_values(boxes, expected):
    value, details = covered(boxes)

    # A column is stored as float32, so compare as float64 within its resolution; the helpers are tested exactly below.
    assert float(value[0]) == pytest.approx(expected, abs=1e-6)
    if expected in (0.0, 0.5, 1.0):
        assert value[0] == expected  # exact where the arithmetic is, so a fully covered box reads exactly 1.0
    assert details["covered_fraction_reason"][0] == ""
    assert details["spatial_status"][0] == "ok"


def test_covered_fraction_uses_the_boxes_clipped_to_the_image():
    # The focal box reaches x = 50 in an image 40 wide; clipped it is (30, 0, 40, 10) and the peer covers half of that.
    value, _ = covered([(30, 0, 50, 10), (35, 0, 40, 10)])
    assert value[0] == 0.5

    value, _ = covered([(30, 0, 50, 10), (35, 0, 60, 10)])  # the peer's overhang is cut off the same way
    assert value[0] == 0.5


def test_covered_fraction_with_fractional_edges():
    value, _ = covered([(0.5, 0.5, 10.5, 10.5), (5.25, 0.5, 12, 10.5)])

    assert float(value[0]) == pytest.approx(0.525, rel=1e-6)  # float32 storage


def focal_union_fraction(layout):
    """The covered fraction of the first box in float64, before `image_stats` stores a column as float32."""
    clipped = np.asarray(layout, dtype=np.float64)
    hit, rects, _ = _intersections(clipped, np.arange(1, len(clipped)), 0)
    return _union_fraction(clipped[0], rects) if len(hit) else 0.0


@pytest.mark.parametrize("scale", [0.37, 1000.0])
def test_covered_fraction_does_not_depend_on_the_scale(scale):
    layout = [(0, 0, 10, 10), (0, 0, 6, 10), (4, 0, 10, 7), (7, 5, 15, 12), (2, 2, 3, 9)]

    scaled = focal_union_fraction([tuple(c * scale for c in box) for box in layout])

    assert scaled == pytest.approx(focal_union_fraction(layout), rel=1e-12)


def raster_fraction(box, peers, size):
    """The covered fraction by painting each intersection into a boolean array, as a second implementation."""
    grid = np.zeros((size, size), dtype=bool)
    for p in peers:
        x0, y0, x1, y1 = max(p[0], box[0]), max(p[1], box[1]), min(p[2], box[2]), min(p[3], box[3])
        if x1 > x0 and y1 > y0:
            grid[y0:y1, x0:x1] = True
    return grid.sum() / ((box[2] - box[0]) * (box[3] - box[1]))


def random_layout(rng, size=16):
    count = int(rng.integers(1, 13))
    x0, y0 = rng.integers(0, size - 1, count), rng.integers(0, size - 1, count)
    x1 = x0 + 1 + (rng.integers(0, size, count) % (size - x0))
    y1 = y0 + 1 + (rng.integers(0, size, count) % (size - y0))
    layout = np.stack([x0, y0, x1, y1], axis=1)
    if count > 2 and rng.random() < 0.5:
        layout[1] = layout[0]  # duplicates
    return layout


def test_union_fraction_matches_a_raster_reference():
    rng = np.random.default_rng(7)
    for _ in range(300):
        layout = random_layout(rng)
        clipped = layout.astype(np.float64)
        for focal in range(len(layout)):
            peers = np.delete(np.arange(len(layout)), focal)
            hit, rects, _ = _intersections(clipped, peers, focal)
            got = _union_fraction(clipped[focal], rects) if len(hit) else 0.0
            expected = raster_fraction(layout[focal], layout[peers], 16)
            assert got == expected, (layout.tolist(), focal)


def test_covered_fraction_through_image_stats_matches_the_raster_reference():
    rng = np.random.default_rng(11)
    for _ in range(40):
        layout = random_layout(rng)
        value, _ = covered([tuple(int(c) for c in box) for box in layout], shape=(3, 16, 16))
        expected = [raster_fraction(layout[i], np.delete(layout, i, axis=0), 16) for i in range(len(layout))]
        # A column is stored as float32; the float64 comparison is `test_union_fraction_matches_a_raster_reference`.
        np.testing.assert_allclose(value, expected, rtol=1e-6)


def test_intersections_returns_the_overlapping_peers_in_index_order_and_their_iou():
    clipped = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 0, 30, 10], [0, 0, 10, 10], [10, 0, 20, 10]], dtype=float)

    hit, rects, iou = _intersections(clipped, np.array([1, 2, 3, 4]), 0)

    assert hit.tolist() == [1, 3]  # box 2 is apart, box 4 only touches an edge
    assert rects.tolist() == [[5, 0, 10, 10], [0, 0, 10, 10]]
    np.testing.assert_allclose(iou, [50 / 150, 1.0], rtol=1e-12)


def test_covered_fraction_is_unchanged_by_reordering_the_boxes():
    boxes = [
        (0, 0, 10, 10),
        (4, 3, 16, 12),
        (8, 0, 12, 18),
        (0, 8, 30, 12),
        (25, 2, 39, 19),
        (5, 5, 5, 9),
        (60, 0, 70, 5),
    ]
    perm = [4, 0, 6, 2, 5, 1, 3]

    value, details = covered(boxes)
    shuffled_value, shuffled = covered([boxes[p] for p in perm])

    np.testing.assert_array_equal(shuffled_value, value[perm])
    for name, values in details.items():
        np.testing.assert_array_equal(shuffled[name], values[perm], err_msg=name)


def test_covered_fraction_status_and_counts():
    value, details = covered([(0, 0, 10, 10), (5, 0, 15, 10), (5, 5, 5, 15), (50, 0, 60, 10), (NAN, 0, 5, 5)])

    assert value[:2].tolist() == [0.5, 0.5]
    assert np.isnan(value[2:]).all()
    assert details["spatial_status"].tolist() == ["incomplete_geometry"] * 2 + ["invalid_box"] * 3
    # A valid row compares the one other valid box and skips the zero-area and outside boxes; the nonfinite box is in
    # neither count, since it sets the status. A row that is not measured says -1 and not 0.
    assert details["spatial_peer_count"].tolist() == [1, 1, -1, -1, -1]
    assert details["spatial_skipped_count"].tolist() == [2, 2, -1, -1, -1]
    assert details["covered_fraction_reason"].tolist() == [""] * 5
    assert details["spatial_peer_count"].dtype == np.int64
    assert details["covered_fraction_reason"].dtype.kind == "U"


def test_covered_fraction_on_the_image_row_and_in_an_empty_frame():
    result = image_stats([np.zeros(SHAPE)], boxes=[[(1, 1, 9, 9)]], stats=COVERED)
    details = details_of(result)

    assert np.isnan(result["stats"]["covered_fraction"][0])
    assert result["stats"]["covered_fraction"][1] == 0.0
    assert details["spatial_status"].tolist() == ["not_applicable", "ok"]
    assert details["spatial_peer_count"].tolist() == [-1, 0]
    assert details["spatial_skipped_count"].tolist() == [-1, 0]

    empty = image_stats([np.zeros(SHAPE)], boxes=[[]], stats=COVERED, per_target=True)
    assert empty["stats"]["covered_fraction"].shape == (1,)


def test_a_request_for_covered_fraction_alone_needs_no_labels_and_adds_its_own_details():
    only = spatial([(1, 1, 9, 9), (5, 5, 15, 15)], stats=COVERED)

    assert set(details_of(only)) == {
        "spatial_status",
        "spatial_peer_count",
        "spatial_skipped_count",
        "covered_fraction_reason",
    }


def test_covered_fraction_limit_is_loud_and_leaves_the_other_statistics_alone(monkeypatch):
    # A overlaps three boxes that do not overlap each other, so it has three intersecting peers.
    boxes = [(0, 0, 10, 10), (0, 0, 3, 3), (5, 5, 8, 8), (0, 6, 3, 9)]
    stats = COVERED | NN

    monkeypatch.setattr(_spatialstats, "_MAX_OVERLAPS", 3)
    result = spatial(boxes, stats=stats)
    assert result["stats"]["covered_fraction"][0] == pytest.approx(27 / 100)  # exactly at the limit: computed
    assert details_of(result)["covered_fraction_reason"][0] == ""

    monkeypatch.setattr(_spatialstats, "_MAX_OVERLAPS", 2)
    result = spatial(boxes, stats=stats)
    assert np.isnan(result["stats"]["covered_fraction"][0])
    assert details_of(result)["covered_fraction_reason"].tolist() == ["too_many_overlaps", "", "", ""]
    assert details_of(result)["spatial_status"][0] == "ok"
    assert result["stats"]["covered_fraction"][1:].tolist() == [1.0, 1.0, 1.0]  # each has the one peer, A, over it
    assert np.isfinite(result["stats"]["nn_distance_norm"]).all()


SAME = ImageStats.SPATIAL_MAX_IOU_SAME_CLASS
OTHER = ImageStats.SPATIAL_MAX_IOU_OTHER_CLASS
IOU = SAME | OTHER

# The focal box F has IoU 0.6 with A (same class), 0.4 with B (label missing) and 0.3 with C (another class).
F, A, B, C = (0, 0, 10, 10), (0, 0, 10, 6), (0, 0, 10, 4), (0, 0, 10, 3)
APART = (30, 10, 38, 18)  # overlaps none of them


def iou(boxes, labels, stats=IOU, **kw):
    """The class IoUs and the details of `boxes` with `labels`, in box order."""
    result = spatial(boxes, stats=stats, labels=[labels], **kw)
    return result["stats"], details_of(result)


def test_iou_by_class_relationship_with_an_unknown_peer_that_cannot_change_the_answer():
    stats, details = iou([F, A, APART, C], [1, 1, -1, 2])  # APART has IoU 0.0

    assert stats["max_iou_same_class"][0] == pytest.approx(0.6, rel=1e-6)
    assert stats["max_iou_other_class"][0] == pytest.approx(0.3, rel=1e-6)
    assert details["max_iou_same_class_target_index"][0] == 1
    assert details["max_iou_other_class_target_index"][0] == 3
    assert details["max_iou_other_class_label"][0] == 2
    assert details["max_iou_same_class_reason"][0] == details["max_iou_other_class_reason"][0] == ""


def test_an_unknown_peer_that_overlaps_more_than_the_known_maximum_makes_that_maximum_nan():
    stats, details = iou([F, A, B, C], [1, 1, -1, 2])  # B, 0.4 and unknown, beats C's 0.3 but not A's 0.6

    assert stats["max_iou_same_class"][0] == pytest.approx(0.6, rel=1e-6)
    assert np.isnan(stats["max_iou_other_class"][0])
    assert details["max_iou_other_class_reason"][0] == "missing_labels"
    assert details["max_iou_same_class_reason"][0] == ""
    # No counterpart exists for a value that is not known, and a known label is never -1.
    assert details["max_iou_other_class_target_index"][0] == -1
    assert details["max_iou_other_class_label"][0] == -1


def test_an_unknown_focal_label_makes_every_overlap_unknown_and_no_overlap_stays_zero():
    stats, details = iou([F, A, C, APART], [-1, 1, 2, 3])

    assert np.isnan(stats["max_iou_same_class"][0])
    assert np.isnan(stats["max_iou_other_class"][0])
    assert details["max_iou_same_class_reason"][0] == details["max_iou_other_class_reason"][0] == "missing_labels"
    assert stats["max_iou_same_class"][3] == stats["max_iou_other_class"][3] == 0.0  # no peer overlaps APART
    assert details["max_iou_same_class_reason"][3] == details["max_iou_other_class_reason"][3] == ""


@pytest.mark.parametrize("missing", [-1, -7])
def test_any_negative_id_is_missing_and_two_missing_labels_are_not_the_same_class(missing):
    stats, details = iou([F, A], [missing, missing])

    assert np.isnan(stats["max_iou_same_class"]).all()  # unknown, and not "same class" because both read -1
    assert np.isnan(stats["max_iou_other_class"]).all()
    assert details["max_iou_same_class_reason"].tolist() == ["missing_labels"] * 2


def test_a_duplicate_of_another_class_reads_one_and_ties_go_to_the_lowest_index():
    stats, details = iou([F, F, F], [0, 1, 1])

    assert stats["max_iou_other_class"].tolist() == [1.0, 1.0, 1.0]
    assert stats["max_iou_same_class"].tolist() == [0.0, 1.0, 1.0]
    # Box 0 has two other-class duplicates at IoU 1.0 and takes the lower index; box 1's same-class peer is box 2.
    assert details["max_iou_other_class_target_index"].tolist() == [1, 0, 0]
    assert details["max_iou_other_class_label"].tolist() == [1, 0, 0]
    assert details["max_iou_same_class_target_index"].tolist() == [-1, 2, 1]


def test_the_counterpart_is_minus_one_where_the_maximum_is_zero():
    _, details = iou([F, APART], [1, 1])

    assert details["max_iou_same_class_target_index"].tolist() == [-1, -1]
    assert details["max_iou_other_class_target_index"].tolist() == [-1, -1]
    assert details["max_iou_other_class_label"].tolist() == [-1, -1]
    assert details["max_iou_same_class_target_index"].dtype == np.int64
    assert details["max_iou_other_class_reason"].dtype.kind == "U"


def test_class_iou_reads_the_boxes_clipped_to_the_image():
    # In an image 40 wide the focal box (30, 0, 50, 10) is (30, 0, 40, 10); a peer over its left half has IoU 0.5.
    stats, _ = iou([(30, 0, 50, 10), (30, 0, 35, 10)], [1, 1])

    assert stats["max_iou_same_class"].tolist() == [0.5, 0.5]


def test_class_iou_status_and_unmeasured_rows():
    result = image_stats(
        [np.zeros(SHAPE)],
        boxes=[[F, A, (NAN, 0, 5, 5)]],
        labels=[[1, 1, 1]],
        stats=IOU | COVERED,
    )
    stats, details = result["stats"], details_of(result)

    # Rows: the image, the two trusted boxes beside a corrupt one, and the corrupt one.
    assert details["spatial_status"].tolist() == [
        "not_applicable",
        "incomplete_geometry",
        "incomplete_geometry",
        "invalid_box",
    ]
    assert np.isnan(stats["max_iou_same_class"][[0, 3]]).all()
    assert stats["max_iou_same_class"][1:3].tolist() == pytest.approx([0.6, 0.6], rel=1e-6)
    assert details["max_iou_same_class_target_index"].tolist() == [-1, 1, 0, -1]
    assert details["max_iou_other_class_label"].tolist() == [-1, -1, -1, -1]
    assert details["spatial_peer_count"].tolist() == [-1, 1, 1, -1]


def test_reordering_permutes_the_class_statistics_and_their_counterparts():
    boxes = [F, A, (3, 2, 14, 12), (0, 0, 10, 5), (6, 6, 18, 16), APART]
    labels = [1, 1, 2, 1, 2, 3]
    perm = [4, 0, 5, 2, 1, 3]

    stats, details = iou(boxes, labels)
    shuffled_stats, shuffled = iou([boxes[p] for p in perm], [labels[p] for p in perm])

    where = np.argsort(perm)
    for name in stats:
        np.testing.assert_array_equal(shuffled_stats[name], stats[name][perm], err_msg=name)
    for name in ("same_class_target_index", "other_class_target_index"):
        old = details[f"max_iou_{name}"]
        expected = [where[old[p]] if old[p] >= 0 else -1 for p in perm]
        assert shuffled[f"max_iou_{name}"].tolist() == expected, name
    np.testing.assert_array_equal(shuffled["max_iou_other_class_label"], details["max_iou_other_class_label"][perm])


def test_no_labels_leaves_every_label_missing():
    boxes = [F, A, APART]

    with pytest.warns(UserWarning, match="without labels="):
        result = image_stats([np.zeros(SHAPE)], boxes=[boxes], stats=IOU | COVERED, per_image=False)

    stats, details = result["stats"], details_of(result)
    assert np.isnan(stats["max_iou_same_class"][:2]).all()
    assert stats["max_iou_same_class"][2] == stats["max_iou_other_class"][2] == 0.0  # nothing overlaps APART
    assert details["max_iou_other_class_reason"].tolist() == ["missing_labels", "missing_labels", ""]
    assert stats["covered_fraction"].tolist() == pytest.approx([0.6, 1.0, 0.0], rel=1e-6)  # geometry needs no labels


def test_the_class_statistics_alone_without_labels_raise():
    with pytest.raises(ValueError, match=r"pass labels= alongside boxes="):
        image_stats([np.zeros(SHAPE)], boxes=[[F, A]], stats=IOU)


def test_the_class_statistics_add_exactly_their_own_details():
    only_same = spatial([F, A], stats=SAME)
    only_other = spatial([F, A], stats=OTHER)

    assert set(details_of(only_same)) == {
        "spatial_status",
        "spatial_peer_count",
        "spatial_skipped_count",
        "max_iou_same_class_target_index",
        "max_iou_same_class_reason",
    }
    assert set(details_of(only_other)) == {
        "spatial_status",
        "spatial_peer_count",
        "spatial_skipped_count",
        "max_iou_other_class_target_index",
        "max_iou_other_class_label",
        "max_iou_other_class_reason",
    }
    assert set(only_same["stats"]) == {"max_iou_same_class"}


def test_class_statistics_in_an_empty_frame_and_beside_only_invalid_boxes():
    empty = image_stats([np.zeros(SHAPE)], boxes=[[]], labels=[[]], stats=IOU, per_target=True)
    assert empty["stats"]["max_iou_same_class"].shape == (1,)
    assert np.isnan(empty["stats"]["max_iou_other_class"]).all()  # the image row only

    stats, details = iou([(5, 5, 5, 5)], [1])  # one zero-area box: invalid, so NaN and no counterpart
    assert np.isnan(stats["max_iou_same_class"]).all()
    assert details["spatial_status"].tolist() == ["invalid_box"]
    assert details["max_iou_other_class_label"].tolist() == [-1]
    assert details["max_iou_other_class_reason"].tolist() == [""]

    stats, details = iou([(5, 5, 5, 5), F], [1, 1])  # a valid box whose only other box is skipped: nothing overlaps it
    assert stats["max_iou_same_class"][1] == stats["max_iou_other_class"][1] == 0.0
    assert details["spatial_skipped_count"].tolist() == [-1, 1]


def test_the_overlap_limit_is_the_number_the_docs_state():
    assert _spatialstats._MAX_OVERLAPS == 1_000
    assert "more than 1,000 other boxes" in (ImageStats.__doc__ or "")
