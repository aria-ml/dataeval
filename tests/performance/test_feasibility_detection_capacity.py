"""
The object detection capacity bound: an AP ceiling for any detector capped at K detections per image, from counts.

Four hybrid endpoints share ``alpha_claim`` equally: the count per image (lower and upper), the shortfall ``(N - K)+``
(lower) and ``min(N, K)`` (upper). The bound is the smaller of the direct ratio and the risk form.
"""

import numpy as np
import pytest

from dataeval.performance._feasibility_classification import _UnavailableError
from dataeval.performance._feasibility_detection import capacity_evidence, capacity_from_counts, check_counts
from dataeval.performance._feasibility_target import TargetSpec
from dataeval.performance._feasibility_units import DetectionUnits

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

ALPHA = 0.05


def _counts(base: list[int], n: int) -> np.ndarray:
    return np.array(base)[np.arange(n) % len(base)]


# Pinned offline by the DFE reference implementation: its capacity_report with the hybrid method, on these counts.
# Endpoints: count lower, count upper, shortfall lower, min(N, K) upper. The first two are decided by the risk form,
# the dense one by the direct form.
PINS = {
    "binding": (
        _counts([0, 1, 2, 3, 4, 5, 6, 2, 3, 4], 60),
        2,
        6,
        0.887246642619408,
        (1.766082602209472, 4.233917397790528, 0.47738840147298145, 2.0),
    ),
    "near_binding": (
        _counts([0, 1, 2, 3, 1, 2, 3, 2, 1, 5], 2000),
        3,
        8,
        0.9439863239125759,
        (1.8442130234390626, 2.1557869765609374, 0.12075355341857168, 1.8946328064791258),
    ),
    "dense": (
        _counts([2, 3, 4, 5, 6, 4, 3, 5], 300),
        1,
        6,
        0.28713571373064595,
        (3.4826737050831382, 4.503992961583529, 2.5334794699867884, 1.0),
    ),
}
_ENDPOINTS = ("count_lower", "count_upper", "shortfall_lower", "matched_upper")


@pytest.mark.parametrize("case", PINS)
def test_matches_dfe_reference(case):
    n, k, b, expected, endpoints = PINS[case]
    value, raw = capacity_from_counts(n, max_objects=b, max_detections=k, alpha_claim=ALPHA)
    assert value == pytest.approx(expected, rel=1e-12)
    assert raw["value"] == value
    assert [raw[key]["endpoint"] for key in _ENDPOINTS] == pytest.approx(endpoints, rel=1e-12)


def test_every_image_at_or_below_k_gives_one():
    value, _ = capacity_from_counts(_counts([0, 1, 2, 3], 400), max_objects=8, max_detections=3, alpha_claim=ALPHA)
    assert value == 1.0


def test_absent_class_gives_one_and_no_mean():
    value, raw = capacity_from_counts(np.zeros(50, dtype=np.intp), max_objects=6, max_detections=2, alpha_claim=ALPHA)
    assert value == 1.0
    assert raw["mean"] is None


def test_mean_is_the_sample_ratio():
    n = _counts([0, 1, 4, 6, 2], 25)
    _, raw = capacity_from_counts(n, max_objects=6, max_detections=2, alpha_claim=ALPHA)
    assert raw["mean"] == (0 + 1 + 2 + 2 + 2) / (0 + 1 + 4 + 6 + 2)
    assert raw["n"] == 25


def test_four_equal_deltas():
    _, raw = capacity_from_counts(_counts([0, 3, 5], 30), max_objects=6, max_detections=2, alpha_claim=ALPHA)
    assert raw["deltas"] == [ALPHA / 4] * 4
    assert [raw[key]["delta"] for key in _ENDPOINTS] == raw["deltas"]


def test_fewer_than_two_images_is_unavailable():
    with pytest.raises(_UnavailableError) as e:
        capacity_from_counts(np.array([4]), max_objects=6, max_detections=2, alpha_claim=ALPHA)
    assert e.value.reason == "too_few_units"


@pytest.mark.parametrize("k", [6, 7])
def test_refuses_k_at_or_above_b(k):
    with pytest.raises(ValueError, match="below max_objects"):
        capacity_from_counts(_counts([0, 3, 5], 30), max_objects=6, max_detections=k, alpha_claim=ALPHA)


# --- the capacity rows ------------------------------------------------------------------------------------------------


def _spec(k: int, class_name: str = "0") -> TargetSpec:
    return TargetSpec(task="OD", metric="ap", threshold=0.5, class_name=class_name, iou=0.5, max_detections=k)


def _units(counts: np.ndarray, other: np.ndarray) -> DetectionUnits:
    """Image ``i`` holds ``counts[i]`` targets of class 0 and ``other[i]`` targets of class 1."""
    labels = tuple(np.array([0] * int(n) + [1] * int(m), dtype=np.intp) for n, m in zip(counts, other, strict=True))
    boxes = tuple(np.zeros((len(x), 4)) for x in labels)
    ids = np.arange(len(counts))
    return DetectionUnits(boxes, labels, ids, 2, ids, None, False)


def test_check_counts_returns_per_image_counts():
    n, m = _counts([0, 1, 4, 6, 2], 25), _counts([3, 0, 1], 25)
    got = check_counts(_units(n, m), [0, 1], [6, 6], [_spec(2), _spec(2, "1")])
    assert [g.tolist() for g in got] == [n.tolist(), m.tolist()]
    assert all(g.dtype == np.intp for g in got)


def test_capacity_row_fields():
    n = PINS["binding"][0]  # K = 2 of B = 6: informative
    (row,) = capacity_evidence([_spec(2)], counts=[n], alpha_claim=ALPHA, max_objects=[6])
    value, raw = capacity_from_counts(n, max_objects=6, max_detections=2, alpha_claim=ALPHA)
    assert (row.requirement, row.source, row.role, row.quantity, row.bound_type) == (
        0,
        "capacity",
        "envelope",
        "ap",
        "upper",
    )
    assert (row.information_scope, row.population, row.sample, row.unit) == (
        "full_input",
        "items",
        "all_items",
        "image",
    )
    assert row.assumptions == ("max_objects_holds",)
    assert row.observed is None
    assert (row.bound, row.mean, row.n, row.alpha) == (value, raw["mean"], 60, ALPHA)
    assert row.mean == n.clip(max=2).sum() / n.sum()
    assert row.variance is None
    assert row.radius is None
    assert row.method is None
    assert row.state == "informative"


def test_no_row_for_k_at_or_above_b():
    n = _counts([0, 1, 4, 6, 2], 25)
    rows = capacity_evidence(
        [_spec(6), _spec(7), _spec(2), _spec(5)], counts=[n] * 4, alpha_claim=ALPHA, max_objects=[6, 6, 6, 6]
    )
    assert [r.requirement for r in rows] == [2, 3]  # in spec order, the index of the spec


def test_two_requirements_on_one_class_get_two_capacity_rows():
    n = PINS["binding"][0]
    rows = capacity_evidence([_spec(2), _spec(3)], counts=[n, n], alpha_claim=ALPHA, max_objects=[6, 6])
    assert [r.requirement for r in rows] == [0, 1]
    assert rows[0].bound != rows[1].bound  # each row is its own K
    for r, k in zip(rows, (2, 3), strict=True):
        assert r.bound == capacity_from_counts(n, max_objects=6, max_detections=k, alpha_claim=ALPHA)[0]


def test_too_few_images_is_an_unavailable_row():
    (row,) = capacity_evidence([_spec(2)], counts=[np.array([4])], alpha_claim=ALPHA, max_objects=[6])
    assert (row.state, row.reason, row.bound) == ("unavailable", "too_few_units", None)


def test_absent_class_is_a_bound_at_maximum_without_a_mean():
    (row,) = capacity_evidence([_spec(2)], counts=[np.zeros(50, dtype=np.intp)], alpha_claim=ALPHA, max_objects=[6])
    assert (row.state, row.reason, row.bound, row.mean) == ("uninformative", "bound_at_maximum", 1.0, None)
