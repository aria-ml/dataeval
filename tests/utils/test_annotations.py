import pickle

import numpy as np
import pytest

from dataeval.utils._annotations import Annotations, build_annotations
from dataeval.utils.preprocessing import BoundingBox, to_bounding_box

pytestmark = pytest.mark.required

SHAPE = (3, 20, 40)
NAN = float("nan")
INF = float("inf")


def test_raw_keeps_what_conversion_destroys():
    entries = [(0, 0, NAN, 10), (30, 0, 10, 10), [5, 5, 15, 15]]
    ann = build_annotations(entries, SHAPE, 0)

    np.testing.assert_array_equal(ann.raw, np.array(entries, dtype=np.float64))
    assert ann.boxes[0].xyxy == (0, 0, 0, 10)
    assert ann.boxes[1].xyxy == (10, 0, 30, 10)
    assert ann.boxes[2].xyxy == (5, 5, 15, 15)
    for box, entry in zip(ann.boxes[1:], entries[1:], strict=True):
        assert box.xyxy == to_bounding_box(entry, SHAPE).xyxy


def test_one_shot_entries_are_read_once():
    def entries():
        return [iter([1, 2, 3, 4]), (v for v in (5, 6, 7, 8))]

    for source in (entries(), (e for e in entries())):
        ann = build_annotations(source, SHAPE, 0)
        assert ann.raw.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
        assert ann.boxes[0].xyxy == (1, 2, 3, 4)
        assert ann.boxes[1].xyxy == (5, 6, 7, 8)


def test_numpy_rows_and_2d_array_are_accepted():
    rows = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    from_array = build_annotations(rows, SHAPE, 0)
    from_list = build_annotations([rows[0], rows[1]], SHAPE, 0)

    np.testing.assert_array_equal(from_array.raw, from_list.raw)
    assert from_array.raw.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]


@pytest.mark.parametrize("empty", [None, [], np.empty(0), np.empty((0, 4)), iter(())])
def test_every_no_boxes_spelling_is_empty(empty):
    ann = build_annotations(empty, SHAPE, 0)

    assert ann.raw.shape == (0, 4)
    assert ann.boxes == []


def test_none_entry_is_the_full_image_box():
    ann = build_annotations([None], SHAPE, 0)
    assert ann.raw[0].tolist() == [0, 0, 40, 20]
    assert ann.boxes[0].xyxy == (0.0, 0.0, 40.0, 20.0)

    non_spatial = build_annotations([None], (40,), 0)
    assert non_spatial.raw[0].tolist() == [0, 0, 0, 0]


def test_bounding_box_entry_raw_is_its_xyxy():
    box = BoundingBox(10, 20, 30, 40)
    ann = build_annotations([box], SHAPE, 0)

    assert ann.raw[0].tolist() == [10, 20, 30, 40]
    assert ann.boxes[0] is box


@pytest.mark.parametrize(
    ("entry", "where"),
    [((NAN, 0, 10, 10), (0, 0)), ((0, 0, INF, 10), (0, 2)), (BoundingBox(NAN, 0, 10, 10), (0, 0))],
)
def test_nonfinite_converted_box_becomes_the_zero_box_and_raw_keeps_it(entry, where):
    ann = build_annotations([entry], SHAPE, 0)

    assert ann.boxes[0].xyxy == (0, 0, 0, 0)
    assert not np.isfinite(ann.raw[where])


@pytest.mark.parametrize("bad", [(1, 2, 3), (1, 2, 3, 4, 5), 5.0, "invalid", ("a", "b", "c", "d"), (None, 0, 1, 1)])
def test_malformed_rows_raise_naming_image_and_box(bad):
    with pytest.raises(ValueError, match=r"image 3, box 1.*expected 4 values"):
        build_annotations([(0, 0, 1, 1), bad], SHAPE, 3)


@pytest.mark.parametrize("flat", [5.0, 1])
def test_a_single_number_where_a_list_of_boxes_belongs_names_that_fix(flat):
    with pytest.raises(ValueError, match=r"image 2, box 0: expected 4 values.*a single (float|int)") as caught:
        build_annotations([flat], SHAPE, 2)

    assert "one list of boxes per image" in str(caught.value)
    assert "extra column" not in str(caught.value)


def test_a_row_of_the_wrong_length_names_the_extra_column_fix():
    with pytest.raises(ValueError, match=r"drop any extra column.*labels=") as caught:
        build_annotations([(1, 2, 3, 4, 5)], SHAPE, 0)

    assert "one list of boxes per image" not in str(caught.value)


def test_label_errors_say_what_is_wrong_and_the_fix():
    entries = [(0, 0, 5, 5), (1, 1, 9, 9)]

    with pytest.raises(ValueError, match=r"1-D sequence.*shape \(2, 1\)"):
        build_annotations(entries, SHAPE, 0, labels=[[0], [1]])
    with pytest.raises(ValueError, match=r"integer class ids.*astype\(int\)"):
        build_annotations(entries, SHAPE, 0, labels=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match=r"holds 1 class ids for 2 boxes"):
        build_annotations(entries, SHAPE, 0, labels=[0])


def test_labels_align_and_validate():
    entries = [(0, 0, 5, 5), (1, 1, 9, 9)]

    ann = build_annotations(entries, SHAPE, 0, labels=[3, 4])
    assert ann.labels is not None
    assert ann.labels.dtype == np.int64
    assert ann.labels.tolist() == [3, 4]

    from_uint8 = build_annotations(entries, SHAPE, 0, labels=np.array([3, 4], dtype=np.uint8)).labels
    assert from_uint8 is not None
    assert from_uint8.tolist() == [3, 4]
    assert build_annotations(entries, SHAPE, 0).labels is None

    with pytest.raises(ValueError, match=r"image 3.*labels"):
        build_annotations(entries, SHAPE, 3, labels=[3])
    with pytest.raises(ValueError, match="integer class ids"):
        build_annotations(entries, SHAPE, 3, labels=np.array([0.5, 1.5]))

    empty = build_annotations(None, SHAPE, 0, labels=np.empty(0))
    assert empty.labels is not None
    assert empty.labels.dtype == np.int64
    assert empty.labels.size == 0


def test_annotations_pickle_round_trip():
    ann = build_annotations([(0, 0, NAN, 10), (30, 0, 10, 10)], SHAPE, 0, labels=[1, 2])
    back = pickle.loads(pickle.dumps(ann))

    assert isinstance(back, Annotations)
    np.testing.assert_array_equal(back.raw, ann.raw)
    assert [b.xyxy for b in back.boxes] == [b.xyxy for b in ann.boxes]
    assert back.labels is not None
    assert back.labels.tolist() == [1, 2]
