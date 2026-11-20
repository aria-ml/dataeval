"""
The object detection AP witness: a threshold grid fixed on the fit split, count endpoints on the audit, the staircase.

The scene: image ``i`` holds ``COUNTS[i]`` disjoint class-0 targets (10% of images none, at most B = 3) and one class-1
target. Its class-0 detections are each target at 0.9, then a second copy of each at 0.9, then two false positives at
0.1, far from every target. With K = 2 and ties kept in the order given, the top K are the first two of that list. A
low image (``LOW``, one fit image and ten audit images) instead holds one far false positive at 0.1 and its first
target at 0.05. That match is below the fit split's 5% matched quantile, so only the ``-inf`` threshold counts it. The
counts follow from the construction alone (``_derived``), not from the scorer. The fit split is the first half of the
images and the audit the second, each 10% empty.
"""

import math
import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from dataeval.core._ap_bounds import ap_lower_staircase
from dataeval.core._concentration import mean_endpoint
from dataeval.performance._feasibility_constants import WITNESS_QUANTILES
from dataeval.performance._feasibility_detection import (
    ImageDetections,
    ap_from_counts,
    ap_witness,
    detection_grid,
    gather_detection_evidence,
)
from dataeval.performance._feasibility_target import TargetSpec
from dataeval.performance._feasibility_units import DetectionUnits

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

B, K, IOU, ALPHA = 3, 2, 0.5, 0.05
COUNTS = np.tile([0, 1, 2, 3, 1, 2, 3, 2, 1, 3], 10)
FIT, AUDIT = np.arange(50), np.arange(50, 100)
LOW = (1, *range(51, 56), *range(61, 66))  # images with 1 to 3 targets
LOW_AUDIT = np.isin(AUDIT, LOW)
_FAR = [[0.0, 100.0, 10.0, 110.0], [0.0, 200.0, 10.0, 210.0]]  # the two false positives' boxes


def _box(j: int) -> list[float]:
    return [20.0 * j, 0.0, 20.0 * j + 10.0, 10.0]


def _scene(
    counts: NDArray[np.intp], bare: int = 0, low: tuple[int, ...] = LOW
) -> tuple[DetectionUnits, list[ImageDetections]]:
    """The scene of the module docstring, then ``bare`` images with no target and no detection."""
    boxes, labels, dets = [], [], []
    for i, n in enumerate(counts):
        targets = [_box(j) for j in range(n)]
        boxes.append(np.array([*targets, _box(9)], dtype=np.float64))
        labels.append(np.array([0] * n + [1], dtype=np.intp))
        found, scores = (
            ([_FAR[0], targets[0]], [0.1, 0.05])
            if i in low
            else ([*targets, *targets, *_FAR], [0.9] * 2 * n + [0.1, 0.1])
        )
        dets.append(ImageDetections(np.array(found), np.zeros(len(found), dtype=np.intp), np.array(scores)))
    boxes += [np.empty((0, 4))] * bare
    labels += [np.empty(0, dtype=np.intp)] * bare
    dets += [ImageDetections(np.empty((0, 4)), np.empty(0, dtype=np.intp), np.empty(0))] * bare
    n_images = len(counts) + bare
    ids = np.arange(n_images)
    return DetectionUnits(tuple(boxes), tuple(labels), ids, 2, ids, None, False), dets


def _add(dets: list[ImageDetections], boxes: list[list[float]], label: int, score: float) -> list[ImageDetections]:
    """Every image gains these detections."""
    extra = np.array(boxes, dtype=np.float64)
    return [
        ImageDetections(
            np.r_[d.boxes, extra],
            np.r_[d.labels, np.full(len(extra), label)],
            np.r_[d.scores, np.full(len(extra), score)],
        )
        for d in dets
    ]


def _derived(n: NDArray[np.intp], low: NDArray[np.bool_]) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """
    The scene's counts on the grid (-inf, 0.9), from its construction, for images with ``n`` targets, ``low`` or not.

    The originals come first, so the top K hold min(n, K) of them, all matched; then copies, which match nothing.
    At 0.9 the kept detections are the first min(2n, K); at -inf the first min(2n + 2, K). A low image keeps its false
    positive and its 0.05 match, and both count at -inf only.
    """
    tp = np.minimum(n, K)
    tp, fp = np.stack([tp, tp], axis=1), np.stack([np.minimum(2 * n + 2, K) - tp, np.minimum(2 * n, K) - tp], axis=1)
    tp[low], fp[low] = [1, 0], [1, 0]
    return tp, fp


def _formula(tp, fp, n, *, b: int = B, k: int = K, alpha: float = ALPHA) -> tuple[float, dict[str, Any]]:
    """The witness by hand: hybrid endpoints at supports min(B, K), K and B, delta = alpha / (2T + 1); then the AP."""
    t = tp.shape[1]
    d = alpha / (2 * t + 1)
    tp_l = [mean_endpoint(tp[:, j], min(b, k), d, side="lower", method="hybrid")["endpoint"] for j in range(t)]
    fp_u = [mean_endpoint(fp[:, j], k, d, side="upper", method="hybrid")["endpoint"] for j in range(t)]
    mu_u = mean_endpoint(n, b, d, side="upper", method="hybrid")["endpoint"]
    return ap_lower_staircase(np.array(tp_l), np.array(fp_u), mu_u), {
        "tp_lower": tp_l,
        "fp_upper": fp_u,
        "mu_upper": mu_u,
    }


def _endpoints(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: raw[key] for key in ("tp_lower", "fp_upper", "mu_upper")}


def _witness(units, dets, fit=FIT, audit=AUDIT, **kwargs):
    args = {"class_id": 0, "class_name": "0", "iou": IOU, "max_detections": K, "max_objects": B, "alpha_claim": ALPHA}
    args |= kwargs
    return ap_witness(units, dets, fit, audit, **args)


def _spec(iou: float = IOU) -> TargetSpec:
    return TargetSpec(task="OD", metric="ap", threshold=0.5, class_name="0", iou=iou, max_detections=K)


def _evidence(units, dets, specs, fit=FIT, audit=AUDIT):
    return gather_detection_evidence(
        specs,
        units=units,
        detections=dets,
        fit=fit,
        audit=audit,
        alpha_claim=ALPHA,
        class_ids=[0] * len(specs),
        max_objects=[B] * len(specs),
    )


@pytest.mark.required
class TestWitnessValue:
    def test_known_answer_matches_core_functions(self):
        """B = 3 != K = 2, the K cut binds on images with 2 or 3 targets, 10% are empty, ten are found only at -inf."""
        value, raw = _witness(*_scene(COUNTS))
        assert raw["thresholds"] == [-np.inf, 0.9]
        tp, fp = _derived(COUNTS[AUDIT], LOW_AUDIT)
        expected, endpoints = _formula(tp, fp, COUNTS[AUDIT])
        assert value == expected
        assert _endpoints(raw) == endpoints
        assert 0 < value < 1
        # D10's reason for the -inf point: at the same endpoints, the staircase without it is strictly lower.
        tp_l, fp_u = np.array(endpoints["tp_lower"]), np.array(endpoints["fp_upper"])
        assert ap_lower_staircase(tp_l[1:], fp_u[1:], endpoints["mu_upper"]) < value

    def test_matches_dfe_reference(self):
        # Pinned offline by the DFE reference implementation: its hybrid mean_endpoint at supports min(B, K) = 3, K = 4
        # and B = 3, then its witness_ap_lower, on these counts.
        i = np.arange(40)
        n = np.array([3, 2, 3, 1, 2, 3, 2, 3, 0, 1])[i % 10]
        tp = np.stack([n, np.maximum(n - i % 2, 0), np.maximum(n - 1, 0)], axis=1)
        fp0 = np.minimum(i % 3 + 1, 4 - n)
        fp1 = np.minimum(i % 3, fp0)
        fp = np.stack([fp0, fp1, fp1 // 2], axis=1)
        value, raw = ap_from_counts(tp, fp, n, max_objects=3, max_detections=4, alpha_claim=0.05)
        assert value == pytest.approx(0.13855884291077852, rel=1e-12)
        assert raw["tp_lower"] == pytest.approx([1.2038129426092505, 0.7038129426092505, 0.3038129426092506], rel=1e-12)
        assert raw["fp_upper"] == pytest.approx([2.536582743187666, 1.8865827431876658, 1.236582743187666], rel=1e-12)
        assert raw["mu_upper"] == pytest.approx(2.7961870573907497, rel=1e-12)

    def test_detections_below_the_top_k_change_nothing(self):
        """Every image already holds K detections at 0.05 or more; matching boxes at 0.01 are cut."""
        units, dets = _scene(COUNTS)
        assert _witness(units, _add(dets, [_box(0), _box(1), _box(2)], 0, 0.01)) == _witness(units, dets)

    def test_empty_images_count(self):
        """Ten audit images with no target and no detection enter every count as zeros."""
        _, raw = _witness(*_scene(COUNTS))
        audit = np.r_[AUDIT, np.arange(100, 110)]
        value, raw_bare = _witness(*_scene(COUNTS, bare=10), audit=audit)
        n = np.r_[COUNTS[AUDIT], np.zeros(10, dtype=np.intp)]
        tp, fp = (np.r_[counts, np.zeros((10, 2), dtype=np.intp)] for counts in _derived(COUNTS[AUDIT], LOW_AUDIT))
        expected, endpoints = _formula(tp, fp, n)
        assert value == expected
        assert _endpoints(raw_bare) == endpoints
        assert raw_bare["mu_upper"] != raw["mu_upper"]

    def test_other_class_detections_change_nothing(self):
        """Class-1 detections on the class-0 targets, scored above them, would take the K slots if not filtered."""
        units, dets = _scene(COUNTS)
        other = _add(dets, [_box(0), _box(1), _box(2)], 1, 0.99)
        assert _witness(units, other) == _witness(units, dets)
        assert _evidence(units, other, [_spec()]) == _evidence(units, dets, [_spec()])

    def test_negative_and_logit_scores_give_a_valid_grid(self):
        """Per image, scores become 10 (s - 0.5), shifted by 0 to -6: matched scores fall below 0 and above 1."""
        units, dets = _scene(COUNTS)
        logits = [d._replace(scores=10 * (d.scores - 0.5) + i % 7 - 6) for i, d in enumerate(dets)]
        value, raw = _witness(units, logits)
        grid = np.array(raw["thresholds"])
        assert grid[0] == -np.inf
        assert (np.diff(grid) > 0).all()
        assert grid[1] < 0 < 1 < grid[-1]
        assert math.isfinite(value)
        assert 0 < value < 1


@pytest.mark.required
class TestGrid:
    def test_grid_takes_quantiles_of_the_matched_scores_after_minus_inf(self):
        scores = np.r_[np.linspace(0, 1, 21), 5.0, 6.0]  # the unmatched 5 and 6 would move every quantile
        matched = np.r_[np.ones(21, dtype=bool), False, False]
        np.testing.assert_allclose(detection_grid(scores, matched), np.r_[-np.inf, WITNESS_QUANTILES], rtol=1e-12)

    def test_grid_takes_every_score_when_none_matched(self):
        grid = detection_grid(np.array([0.2, 0.4, 0.6]), np.zeros(3, dtype=bool))
        np.testing.assert_allclose(grid, np.r_[-np.inf, 0.2 + 0.4 * WITNESS_QUANTILES], rtol=1e-12)

    def test_grid_without_scores_is_minus_inf(self):
        np.testing.assert_array_equal(detection_grid(np.empty(0), np.empty(0, dtype=bool)), [-np.inf])

    def test_no_fit_detection_gives_the_minus_inf_grid_and_a_computed_bound(self):
        units, dets = _scene(COUNTS)
        empty = ImageDetections(np.empty((0, 4)), np.empty(0, dtype=np.intp), np.empty(0))
        value, raw = _witness(units, [empty] * 50 + dets[50:])
        assert raw["thresholds"] == [-np.inf]
        tp, fp = _derived(COUNTS[AUDIT], LOW_AUDIT)
        assert value == _formula(tp[:, :1], fp[:, :1], COUNTS[AUDIT])[0]

    def test_no_match_on_fit_warns_and_takes_every_kept_score(self):
        """Fit boxes moved off every target: 44 images keep two 0.9s, the 5 empty two 0.1s, the low one 0.1 and 0.05."""
        units, dets = _scene(COUNTS)
        moved = [d._replace(boxes=d.boxes + 1000.0) for d in dets[:50]] + dets[50:]
        with pytest.warns(UserWarning, match=r"of class '0' \(label 0\) .*IoU 0\.5.*\(x0, y0, x1, y1\)"):
            value, raw = _witness(units, moved)
        kept = np.r_[np.full(88, 0.9), np.full(11, 0.1), 0.05]
        assert raw["thresholds"] == np.r_[-np.inf, np.unique(np.quantile(kept, WITNESS_QUANTILES))].tolist()
        assert math.isfinite(value)

    def test_no_fit_target_gives_no_match_warning(self):
        """The fit images hold no class-0 target, only two false positives each: no match, and nothing to warn of."""
        units, dets = _scene(np.r_[np.zeros(50, dtype=np.intp), COUNTS[50:]], low=LOW[1:])
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _, raw = _witness(units, dets)
        assert not [w for w in record if "None of the" in str(w.message)]
        assert raw["thresholds"] == [-np.inf, 0.1]  # every kept score: the false positives at 0.1


@pytest.mark.required
class TestEvidence:
    def test_ledger_shares(self):
        """2T + 1 equal shares that sum to the claim, and each endpoint spent at its share."""
        _, raw = _witness(*_scene(COUNTS))
        t = len(raw["thresholds"])
        assert len(raw["deltas"]) == 2 * t + 1
        assert raw["deltas"] == [ALPHA / (2 * t + 1)] * (2 * t + 1)
        assert math.fsum(raw["deltas"]) == pytest.approx(ALPHA, rel=1e-12)
        spent = [r["delta"] for r in [*raw["tp_results"], *raw["fp_results"], raw["mu_result"]]]
        assert spent == raw["deltas"]

    def test_bounds_row_fields(self):
        units, dets = _scene(COUNTS)
        bounds = _evidence(units, dets, [_spec(), _spec(iou=0.75)])
        assert [b.requirement for b in bounds] == [0, 1]
        assert [b.bound for b in bounds] == [_witness(units, dets, iou=iou)[0] for iou in (IOU, 0.75)]
        for b in bounds:
            row = (b.source, b.role, b.quantity, b.bound_type, b.information_scope, b.population, b.sample, b.unit)
            assert row == ("user_model", "witness", "ap", "lower", "full_input", "items", "audit_split", "image")
            assert (b.observed, b.n, b.variance, b.radius, b.method) == (None,) * 5
            assert b.assumptions == (
                "model_independent_of_audit_items",
                "class_order_matches_dataset",
                "max_objects_holds",
            )
            assert b.alpha == ALPHA
            assert b.state == "informative"

    def test_too_few_audit_images_is_unavailable(self):
        (b,) = _evidence(*_scene(COUNTS), [_spec()], audit=np.array([50]))
        assert (b.state, b.reason, b.bound) == ("unavailable", "too_few_units", None)

    def test_no_audit_targets_is_uninformative(self):
        units, dets = _scene(np.r_[COUNTS[:50], np.zeros(50, dtype=np.intp)], low=(1,))
        (b,) = _evidence(units, dets, [_spec()])
        assert (b.bound, b.state, b.reason) == (0.0, "uninformative", "bound_at_minimum")

    def test_an_overspent_ledger_raises(self, monkeypatch):
        spend = (0.5, {"deltas": [ALPHA, ALPHA]})
        monkeypatch.setattr("dataeval.performance._feasibility_detection.ap_witness", lambda *_, **__: spend)
        with pytest.raises(RuntimeError, match="over its claim"):
            _evidence(*_scene(COUNTS), [_spec()])
