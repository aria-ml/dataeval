import numpy as np
import pytest

from dataeval.utils._annotations import AnnotationContext, Kind, build_annotations

pytestmark = pytest.mark.required

EXTENT = (20, 40)  # (H, W)
SHAPE = (3, *EXTENT)
NAN = float("nan")
INF = float("inf")


def context(rows, extent=EXTENT, labels=None):
    return AnnotationContext(build_annotations(rows, (3, *extent), 0, labels=labels), extent)


@pytest.mark.parametrize(
    ("row", "kind"),
    [
        ((5, 5, 15, 15), Kind.VALID),
        ((-5, -5, 10, 10), Kind.VALID),  # partly outside the image
        ((40, 0, 50, 10), Kind.OUTSIDE),  # touches the right edge only
        ((-10, 0, 0, 10), Kind.OUTSIDE),
        ((0, 0, 0, 10), Kind.ZERO_AREA),
        ((50, 0, 50, 10), Kind.ZERO_AREA),  # zero area and outside: zero area wins
        ((10, 0, 2, 10), Kind.REVERSED),
        ((10, 5, 2, 5), Kind.REVERSED),  # reversed and zero height: reversed wins
        ((NAN, 0, 10, 10), Kind.NONFINITE),
        ((0, 0, INF, 10), Kind.NONFINITE),
        ((10, 0, NAN, 10), Kind.NONFINITE),  # nonfinite and reversed: nonfinite wins
    ],
)
def test_kind_table(row, kind):
    assert context([row]).kind.tolist() == [kind]


def test_a_focal_with_a_reversed_peer_is_incomplete_and_a_reversed_focal_is_invalid():
    ctx = context([(5, 5, 15, 15), (10, 0, 2, 10)])

    assert ctx.status(0) == "incomplete_geometry"
    assert ctx.status(1) == "invalid_box"  # its own defect wins over a peer's


@pytest.mark.parametrize("bad", [(NAN, 0, 10, 10), (0, 0, INF, 10)])
def test_a_nonfinite_peer_makes_every_valid_focal_incomplete(bad):
    ctx = context([(5, 5, 15, 15), (20, 5, 30, 15), bad])

    assert [ctx.status(i) for i in range(3)] == ["incomplete_geometry", "incomplete_geometry", "invalid_box"]


def test_zero_area_and_outside_peers_are_skipped_and_counted():
    ctx = context([(5, 5, 15, 15), (0, 0, 0, 10), (40, 0, 50, 10)])

    assert ctx.status(0) == "ok"
    assert ctx.skipped(0) == (1, 1)
    assert ctx.peers(0).size == 0
    assert ctx.skipped(1) == (0, 1)  # the focal itself is not counted


def test_peers_are_the_valid_other_boxes_in_index_order():
    ctx = context([(5, 5, 15, 15), (0, 0, 0, 10), (1, 1, 9, 9), (10, 0, 2, 10), (-5, -5, 10, 10)])

    assert ctx.peers(0).tolist() == [2, 4]
    assert ctx.peers(1).tolist() == [0, 2, 4]
    assert ctx.peers(2).dtype == np.intp


def test_clipped_is_float_and_raw_is_untouched():
    ctx = context([(-5.5, 2.25, 10.5, 30)])

    assert ctx.clipped.dtype == np.float64
    assert ctx.clipped[0].tolist() == [0.0, 2.25, 10.5, 20.0]
    assert ctx.raw[0].tolist() == [-5.5, 2.25, 10.5, 30]


def test_an_unannotated_image_gives_an_empty_context():
    ctx = AnnotationContext(None, EXTENT)

    assert ctx.raw.shape == (0, 4)
    assert ctx.kind.shape == (0,)
    assert ctx.clipped.shape == (0, 4)
    assert ctx.labels is None


def test_labels_are_passed_through():
    labels = context([(1, 1, 5, 5), (2, 2, 6, 6)], labels=[7, 8]).labels

    assert labels is not None
    assert labels.tolist() == [7, 8]


@pytest.mark.parametrize("extent", [(0, 0), (20, 0), (0, 40)])
def test_an_extent_with_no_pixels_makes_every_focal_invalid(extent):
    # The straddling box is the one the edge rules alone let through: it spans the origin, so no edge is outside it.
    ctx = context([(5, 5, 15, 15), (-5, -5, 5, 5)], extent=extent)

    assert ctx.kind.tolist() == [Kind.OUTSIDE, Kind.OUTSIDE]
    assert [ctx.status(0), ctx.status(1)] == ["invalid_box", "invalid_box"]


@pytest.mark.parametrize("focal", [None, -1, 2, "0"])
@pytest.mark.parametrize("method", ["status", "peers", "skipped"])
def test_a_focal_that_is_not_a_box_index_is_refused(method, focal):
    ctx = context([(5, 5, 15, 15), (1, 1, 9, 9)])

    with pytest.raises(ValueError, match=r"focal must be the index of one of this image's 2 boxes"):
        getattr(ctx, method)(focal)


def test_reordering_the_boxes_permutes_kind_and_clipped_and_changes_nothing_else():
    rows = [(5, 5, 15, 15), (-5.5, 2.25, 10.5, 30), (0, 0, 0, 10), (10, 0, 2, 10), (NAN, 0, 10, 10), (40, 0, 50, 10)]
    perm = np.array([3, 5, 0, 4, 1, 2])
    base = context(rows)
    shuffled = context([rows[p] for p in perm])

    np.testing.assert_array_equal(shuffled.kind, base.kind[perm])
    np.testing.assert_array_equal(shuffled.clipped, base.clipped[perm])
    assert [shuffled.status(i) for i in range(len(rows))] == [base.status(p) for p in perm]
    assert [shuffled.skipped(i) for i in range(len(rows))] == [base.skipped(p) for p in perm]


def test_arrays_are_computed_only_when_read():
    ctx = context([(5, 5, 15, 15)])
    assert "kind" not in ctx.__dict__
    assert "clipped" not in ctx.__dict__

    ctx.kind  # noqa: B018
    assert "kind" in ctx.__dict__
    assert "clipped" not in ctx.__dict__
