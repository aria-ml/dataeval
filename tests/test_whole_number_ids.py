"""Track ids and labels are read as whole numbers that fit, or refused, never wrapped or truncated (#1662)."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from dataeval.data._locate import _track_ids_of
from dataeval.types._target import instance_arrays, track_ids_of

pytestmark = pytest.mark.required

REFUSED = {
    "uint64 above the signed range": np.array([2**63 + 5, 7, 8], np.uint64),
    "uint64 maximum, which would read as -1": np.array([2**64 - 1, 7, 8], np.uint64),
    "a fraction": np.array([1.0, 1.5, 2.0]),
    "nan": np.array([np.nan, 3.0, 4.0]),
    "infinity": np.array([np.inf, 3.0, 4.0]),
    "a float past the signed range": np.array([1e20, 2.0, 3.0]),
    "strings": np.array(["a", "b", "c"]),
}
ACCEPTED: dict[str, tuple[Any, list[int]]] = {
    "int64": (np.array([3, -1, 5], np.int64), [3, -1, 5]),
    "uint64 in range": (np.array([3, 4, 2**63 - 1], np.uint64), [3, 4, 2**63 - 1]),
    "whole floats": (np.array([3.0, -1.0, 5.0]), [3, -1, 5]),
    "a list": ([3, -1, 5], [3, -1, 5]),
    "uint8": (np.array([3, 4, 5], np.uint8), [3, 4, 5]),
}


def target(**fields: Any) -> Any:
    fields.setdefault("labels", np.array([0, 1, 2]))
    fields.setdefault("boxes", np.zeros((3, 4), np.float32))
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("case", REFUSED)
def test_a_track_id_that_is_not_a_whole_number_that_fits_is_refused(case):
    with pytest.raises(ValueError, match=r"track_ids.*-1"):
        track_ids_of(target(track_ids=REFUSED[case]), 3)


@pytest.mark.parametrize("case", REFUSED)
def test_both_readers_refuse_the_same_ids(case):
    with pytest.raises(ValueError, match="track_ids"):
        _track_ids_of(target(track_ids=REFUSED[case]))


@pytest.mark.parametrize("case", ACCEPTED)
def test_whole_numbers_that_fit_are_read_unchanged_by_both_readers(case):
    ids, expected = ACCEPTED[case]
    assert track_ids_of(target(track_ids=ids), 3).tolist() == expected
    assert _track_ids_of(target(track_ids=ids)) == expected


def test_both_readers_still_pad_and_truncate():
    assert track_ids_of(target(track_ids=[4]), 3).tolist() == [4, -1, -1]
    assert _track_ids_of(target(track_ids=[4])) == [4, -1, -1]
    assert track_ids_of(target(track_ids=[4, 5, 6, 7]), 3).tolist() == [4, 5, 6]
    assert _track_ids_of(target(track_ids=[4, 5, 6, 7])) == [4, 5, 6]


@pytest.mark.parametrize("case", ["a fraction", "nan", "uint64 maximum, which would read as -1", "strings"])
def test_a_label_that_is_not_a_whole_number_that_fits_is_refused(case):
    with pytest.raises(ValueError, match="labels"):
        instance_arrays(target(labels=REFUSED[case]))


def test_whole_float_labels_are_read_as_classes():
    labels, *_ = instance_arrays(target(labels=np.array([0.0, 2.0, 1.0])))
    assert labels.tolist() == [0, 2, 1]
