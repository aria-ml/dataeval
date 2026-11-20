"""The detection scorer counts a frozen detector's true and false positives the way the COCO evaluator matches them."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from dataeval.core._detection_counts import check_boxes, detection_counts, kept_matches
from dataeval.utils.preprocessing import compute_iou

# Reference values: DFE_DataEval_Documentation/examples/dfe_reference.py, pairwise_iou (torch float64), run with
# ``python -I`` on these boxes.
_REF_A = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [-4, 2, 6, 12], [0.5, 0.25, 3.75, 9.125]], dtype=np.float64)
_REF_B = np.array(
    [[0, 0, 10, 10], [10, 0, 20, 10], [2, 2, 8, 8], [1.5, 0.5, 4.0, 9.0], [100, 100, 101, 102]], dtype=np.float64
)
_REF_IOU = [
    [1.0, 0.0, 0.36, 0.2125, 0.0],
    [0.14285714285714285, 0.14285714285714285, 0.07086614173228346, 0.0, 0.0],
    [0.3157894736842105, 0.0, 0.21428571428571427, 0.1686746987951807, 0.0],
    [0.2884375, 0.0, 0.1932144910868315, 0.6175580221997982, 0.0],
]

_BOX = [0.0, 0.0, 10.0, 10.0]
_EMPTY = np.zeros((0, 4))


def _counts(targets, boxes, scores, *, iou=0.5, thresholds=(0.5,), k=100):
    result = detection_counts(
        [np.asarray(targets, dtype=np.float64).reshape(-1, 4)],
        [np.asarray(boxes, dtype=np.float64).reshape(-1, 4)],
        [np.asarray(scores, dtype=np.float64)],
        iou=iou,
        thresholds=np.asarray(thresholds, dtype=np.float64),
        max_detections=k,
    )
    return result["true_positives"][0].tolist(), result["false_positives"][0].tolist()


def _coco_counts(targets, boxes, scores, *, iou, threshold, k):
    """
    pycocotools ``COCOeval.evaluateImg`` for one image, one category and one IoU threshold, without crowd or ignore
    regions (cocodataset/cocoapi, PythonAPI/pycocotools/cocoeval.py), run on the detections scoring at least
    ``threshold``. Written loop for loop from that source as the oracle; IoU per pair in plain Python.
    """
    order = np.argsort(-scores, kind="mergesort")[:k]
    kept = [i for i in order if scores[i] >= threshold]

    def pair_iou(a, b):
        w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = w * h
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

    gtm = [0] * len(targets)
    tp = 0
    for d in kept:
        best, m = min(iou, 1 - 1e-10), -1
        for g, target in enumerate(targets):
            if gtm[g] > 0:
                continue
            value = pair_iou(boxes[d], target)
            if value < best:
                continue
            best, m = value, g
        if m == -1:
            continue
        gtm[m] = 1
        tp += 1
    return tp, len(kept) - tp


def _scene(rng, n_targets, n_clutter):
    """Targets, their jittered copies plus clutter as detections, and scores on a coarse grid so ties occur."""
    xy = rng.uniform(0, 60, (n_targets, 2))
    wh = rng.uniform(4, 20, (n_targets, 2))
    targets = np.c_[xy, xy + wh]
    picked = rng.integers(0, n_targets, n_targets + 1) if n_targets else np.zeros(0, dtype=np.intp)
    copies = targets[picked] + rng.normal(0, 2, (len(picked), 4))
    cxy = rng.uniform(0, 60, (n_clutter, 2))
    clutter = np.c_[cxy, cxy + rng.uniform(4, 20, (n_clutter, 2))]
    boxes = np.r_[copies, clutter]
    boxes[:, 2:] = np.maximum(boxes[:, 2:], boxes[:, :2] + 1)  # keep positive width and height after jitter
    scores = np.round(rng.uniform(0, 1, len(boxes)), 1)
    return targets, boxes, scores


@pytest.mark.required
class TestKnownAnswers:
    def test_an_empty_image_counts_nothing(self):
        assert _counts(_EMPTY, _EMPTY, []) == ([0], [0])

    def test_targets_with_no_detections_are_all_missed(self):
        assert _counts([_BOX, [20, 20, 30, 30]], _EMPTY, []) == ([0], [0])

    def test_detections_with_no_targets_are_all_false_positives(self):
        assert _counts(_EMPTY, [_BOX, _BOX], [0.9, 0.8]) == ([0], [2])

    def test_a_duplicate_detection_is_a_false_positive(self):
        assert _counts([_BOX], [_BOX, _BOX], [0.9, 0.9]) == ([1], [1])

    def test_two_identical_targets_are_both_matchable(self):
        assert _counts([_BOX, _BOX], [_BOX, _BOX], [0.9, 0.8]) == ([2], [0])

    def test_a_box_past_the_image_edge_is_not_clipped(self):
        """Clipping [-5, 0, 5, 10] to the image would make it [0, 0, 5, 10], which matches [0, 0, 5, 10] exactly."""
        assert _counts([[0, 0, 5, 10]], [[-5, 0, 5, 10]], [0.9], iou=0.6) == ([0], [1])
        assert _counts([[0, 0, 5, 10]], [[-5, 0, 5, 10]], [0.9], iou=0.5) == ([1], [0])

    def test_an_iou_exactly_at_the_threshold_matches(self):
        assert compute_iou(np.array([[0.0, 0, 10, 10]]), np.array([[0.0, 0, 5, 10]]))[0, 0] == 0.5
        assert _counts([[0, 0, 5, 10]], [_BOX], [0.9], iou=0.5) == ([1], [0])

    def test_a_score_exactly_at_the_threshold_counts(self):
        assert _counts([_BOX], [_BOX], [0.5], thresholds=[0.5, 0.50001]) == ([1, 0], [0, 0])

    def test_a_score_tie_keeps_input_order(self):
        """Two detections of one score compete for one target: the first given takes it."""
        near, far = [0, 0, 10, 9], [0, 0, 10, 6]  # IoU 0.9 and 0.6 with the target
        tp, fp = _counts([_BOX], [far, near], [0.7, 0.7], iou=0.55, thresholds=[0.7], k=1)
        assert (tp, fp) == ([1], [0])  # with K = 1, the first of the tied pair is the one kept
        tp, fp = _counts([_BOX], [far, near], [0.7, 0.7], iou=0.7, thresholds=[0.7], k=1)
        assert (tp, fp) == ([0], [1])  # the far one was kept, and it misses at IoU 0.7

    def test_many_tied_scores_keep_input_order(self):
        """Fifty detections of one score and K = 1: only the first given is kept, and only it matches."""
        boxes = [_BOX] + [[50.0 + i, 50.0, 60.0 + i, 60.0] for i in range(49)]
        assert _counts([_BOX], boxes, [0.5] * 50, k=1) == ([1], [0])

    def test_an_iou_just_below_one_matches_at_iou_one(self):
        """pycocotools caps the threshold at 1 - 1e-10, so rounding cannot make an exact copy miss at iou=1."""
        near = [0.0, 0.0, 1.0 - 3e-11, 1.0]
        overlap = compute_iou(np.array([near]), np.array([[0.0, 0.0, 1.0, 1.0]]))[0, 0]
        assert 1 - 1e-10 <= overlap < 1
        assert _counts([[0, 0, 1, 1]], [near], [0.9], iou=1.0) == ([1], [0])

    def test_an_iou_tie_goes_to_the_later_target(self):
        """pycocotools moves the match on an equal IoU, so the later of two equally good targets is taken."""
        left, right = [0, 0, 10, 10], [10, 0, 20, 10]
        middle = [5, 0, 15, 10]  # IoU 1/3 with each
        second = [10, 0, 20, 10]  # IoU 1 with right, 0 with left
        assert _counts([left, right], [middle, second], [0.9, 0.8], iou=0.3) == ([1], [1])  # right was taken first
        assert _counts([right, left], [middle, second], [0.9, 0.8], iou=0.3) == ([2], [0])  # now left was

    def test_max_detections_cuts_by_score(self):
        assert _counts([_BOX], [[50, 50, 60, 60], _BOX], [0.9, 0.8], k=1) == ([0], [1])
        assert _counts([_BOX], [[50, 50, 60, 60], _BOX], [0.9, 0.8], k=2) == ([1], [1])

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_nan_and_plus_inf_thresholds_are_refused(self, bad):
        with pytest.raises(ValueError, match="thresholds"):
            _counts([_BOX], [_BOX], [0.9], thresholds=[0.5, bad])

    def test_counts_follow_the_thresholds_as_given(self):
        tp, fp = _counts([_BOX], [_BOX, [50, 50, 60, 60]], [0.8, 0.6], thresholds=[0.7, 0.9, 0.5, 0.7])
        assert (tp, fp) == ([1, 0, 1, 1], [0, 0, 1, 0])

    def test_minus_inf_counts_every_kept_detection(self):
        rng = np.random.default_rng(3)
        scenes = [_scene(rng, n, 3) for n in (0, 2, 4, 1)]
        parts = [list(part) for part in zip(*scenes, strict=True)]
        k = 3
        result = detection_counts(*parts, iou=0.5, thresholds=np.array([-np.inf, -1e9]), max_detections=k)
        kept = np.array([min(len(boxes), k) for boxes in parts[1]])
        np.testing.assert_array_equal((result["true_positives"] + result["false_positives"])[:, 0], kept)
        np.testing.assert_array_equal(result["true_positives"][:, 0], result["true_positives"][:, 1])
        np.testing.assert_array_equal(result["false_positives"][:, 0], result["false_positives"][:, 1])


@pytest.mark.required
class TestAgainstTheReferences:
    def test_iou_equals_the_dfe_reference(self):
        np.testing.assert_array_equal(compute_iou(_REF_A, _REF_B), _REF_IOU)

    @pytest.mark.parametrize("seed", range(40))
    def test_matches_the_coco_loop_at_every_threshold(self, seed):
        """The oracle matches afresh at each threshold, so this also checks that one pass and a prefix count suffice."""
        rng = np.random.default_rng(seed)
        targets, boxes, scores = _scene(rng, int(rng.integers(0, 6)), int(rng.integers(0, 5)))
        thresholds = np.array([0.0, 0.3, 0.5, 0.5, 0.8, 1.0])
        iou, k = float(rng.choice([0.3, 0.5, 0.75])), int(rng.integers(1, 8))
        result = detection_counts([targets], [boxes], [scores], iou=iou, thresholds=thresholds, max_detections=k)
        expected = [_coco_counts(targets, boxes, scores, iou=iou, threshold=t, k=k) for t in thresholds]
        assert result["true_positives"][0].tolist() == [tp for tp, _ in expected]
        assert result["false_positives"][0].tolist() == [fp for _, fp in expected]

    @pytest.mark.parametrize("seed", range(40))
    def test_greedy_is_at_most_the_maximum_matching(self, seed):
        """The ordering the DFE relaxation relies on (metrics-and-decisions.md:13): greedy never matches more."""
        rng = np.random.default_rng(seed)
        targets, boxes, scores = _scene(rng, int(rng.integers(1, 8)), int(rng.integers(0, 4)))
        result = detection_counts([targets], [boxes], [scores], iou=0.3, thresholds=np.array([0.0]), max_detections=100)
        edges = csr_matrix(compute_iou(boxes, targets) >= 0.3)
        maximum = int((maximum_bipartite_matching(edges, perm_type="column") >= 0).sum())
        assert result["true_positives"][0, 0] <= maximum

    def test_greedy_equals_the_maximum_matching_when_targets_are_far_apart(self):
        targets = np.array([[0, 0, 10, 10], [100, 0, 110, 10], [0, 100, 10, 110]], dtype=np.float64)
        boxes = targets + 1.0
        result = detection_counts(
            [targets], [boxes], [np.array([0.3, 0.9, 0.6])], iou=0.5, thresholds=np.array([0.0]), max_detections=100
        )
        assert result["true_positives"][0, 0] == 3


@pytest.mark.required
class TestKeptMatches:
    @pytest.mark.parametrize("seed", range(40))
    def test_kept_matches_agree_with_detection_counts(self, seed):
        rng = np.random.default_rng(seed)
        scenes = [_scene(rng, int(rng.integers(0, 6)), int(rng.integers(0, 5))) for _ in range(3)]
        parts = [list(part) for part in zip(*scenes, strict=True)]
        iou, k = float(rng.choice([0.3, 0.5, 0.75])), int(rng.integers(1, 8))
        scores, matched = kept_matches(*parts, iou=iou, max_detections=k)
        assert scores.dtype == np.float64
        assert matched.dtype == np.bool_
        every = detection_counts(*parts, iou=iou, thresholds=np.array([-np.inf]), max_detections=k)
        assert len(scores) == (every["true_positives"] + every["false_positives"]).sum()
        assert matched.sum() == every["true_positives"].sum()
        for t in np.unique(scores):
            counted = detection_counts(*parts, iou=iou, thresholds=np.array([t]), max_detections=k)
            assert (matched & (scores >= t)).sum() == counted["true_positives"].sum()

    def test_kept_matches_cut_to_k(self):
        rng = np.random.default_rng(11)
        scenes = [_scene(rng, 3, 6) for _ in range(4)]
        parts = [list(part) for part in zip(*scenes, strict=True)]
        for k in (1, 2, 5):
            scores, _ = kept_matches(*parts, iou=0.5, max_detections=k)
            assert len(scores) == sum(min(len(s), k) for s in parts[2])

    def test_kept_matches_are_in_image_order_then_kept_order(self):
        far = [50, 50, 60, 60]
        scores, matched = kept_matches(
            [np.array([_BOX]), np.array([_BOX])],
            [np.array([far, _BOX]), np.array([_BOX])],
            [np.array([0.6, 0.9]), np.array([0.3])],
            iou=0.5,
            max_detections=2,
        )
        assert scores.tolist() == [0.9, 0.6, 0.3]
        assert matched.tolist() == [True, False, True]

    def test_kept_matches_with_nothing_kept(self):
        for parts in (([], [], []), ([_EMPTY], [_EMPTY], [np.zeros(0)])):
            scores, matched = kept_matches(*parts, iou=0.5, max_detections=2)
            assert (scores.shape, scores.dtype, matched.shape, matched.dtype) == ((0,), np.float64, (0,), np.bool_)

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"detection_scores": []}, "one entry per image"),
            ({"detection_scores": [np.array([0.9, 0.8])]}, "image 0: 2 scores for 1 detection boxes"),
            ({"detection_scores": [np.array([np.nan])]}, "image 0: scores must be finite"),
            ({"detection_boxes": [np.array([[0, 0, 0, 1]])]}, "image 0 detections: "),
            ({"target_boxes": [np.array([[0, 0, 0, 1]])]}, "image 0 targets: "),
            ({"max_detections": 0}, "max_detections"),
            ({"max_detections": True}, "max_detections"),
            ({"iou": 0.0}, "iou"),
        ],
    )
    def test_kept_matches_refuses(self, overrides, match):
        kwargs = {
            "target_boxes": [np.array([_BOX])],
            "detection_boxes": [np.array([_BOX])],
            "detection_scores": [np.array([0.9])],
            "iou": 0.5,
            "max_detections": 1,
        } | overrides
        with pytest.raises(ValueError, match=match):
            kept_matches(**kwargs)


@pytest.mark.required
class TestImages:
    def test_reordering_images_permutes_the_rows(self):
        rng = np.random.default_rng(7)
        scenes = [_scene(rng, n, 2) for n in (0, 1, 3, 5)]
        kwargs = {"iou": 0.5, "thresholds": np.array([0.2, 0.6]), "max_detections": 4}
        whole = detection_counts(*([list(part) for part in zip(*scenes, strict=True)]), **kwargs)
        order = [2, 0, 3, 1]
        shuffled = detection_counts(*([[s[i] for i in order] for s in zip(*scenes, strict=True)]), **kwargs)
        for name in ("true_positives", "false_positives"):
            np.testing.assert_array_equal(shuffled[name], whole[name][order])

    def test_shapes_and_dtypes(self):
        result = detection_counts(
            [_EMPTY, np.array([_BOX])],
            [_EMPTY, np.array([_BOX])],
            [np.zeros(0), np.array([0.9])],
            iou=0.5,
            thresholds=np.array([0.1, 0.5, 0.95]),
            max_detections=10,
        )
        assert result["true_positives"].shape == result["false_positives"].shape == (2, 3)
        assert result["true_positives"].dtype == np.intp
        assert result["true_positives"].tolist() == [[0, 0, 0], [1, 1, 0]]

    def test_no_images(self):
        result = detection_counts([], [], [], iou=0.5, thresholds=np.array([0.5]), max_detections=1)
        assert result["true_positives"].shape == (0, 1)


@pytest.mark.required
class TestChecks:
    @pytest.mark.parametrize(
        ("boxes", "match"),
        [
            (np.array([[0, 0, 1]]), r"shape \(N, 4\)"),
            (np.array([[0, 0, 1, 1], [0, 0, 1, np.nan]]), "box 1 .*not finite"),
            (np.array([[0, 0, 0, 1]]), "positive width and height"),
            (np.array([[0, 0, 1, 1], [2, 2, 1, 3]]), "box 1"),
        ],
    )
    def test_check_boxes_refuses(self, boxes, match):
        with pytest.raises(ValueError, match=match):
            check_boxes(boxes, "image 3 detections")

    def test_check_boxes_names_where(self):
        with pytest.raises(ValueError, match="^image 3 detections: "):
            check_boxes(np.array([[0, 0, 0, 1]]), "image 3 detections")

    @pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 4, 1)])
    def test_check_boxes_refuses_a_malformed_empty_array(self, shape):
        with pytest.raises(ValueError, match=r"shape \(N, 4\)"):
            check_boxes(np.zeros(shape), "x")

    def test_check_boxes_reads_an_empty_array_as_no_boxes(self):
        assert check_boxes(np.zeros(0), "x").shape == (0, 4)
        assert check_boxes([], "x").dtype == np.float64

    def test_check_boxes_returns_float64(self):
        assert check_boxes(np.array([[0, 0, 1, 1]], dtype=np.float32), "x").dtype == np.float64

    def _call(self, **overrides):
        kwargs = {
            "target_boxes": [np.array([_BOX])],
            "detection_boxes": [np.array([_BOX])],
            "detection_scores": [np.array([0.9])],
            "iou": 0.5,
            "thresholds": np.array([0.5]),
            "max_detections": 1,
        } | overrides
        return detection_counts(**kwargs)

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"detection_scores": []}, "one entry per image"),
            ({"detection_scores": [np.array([0.9, 0.8])]}, "image 0: 2 scores for 1 detection boxes"),
            ({"detection_scores": [np.array([np.nan])]}, "image 0: scores must be finite"),
            ({"detection_boxes": [np.array([[0, 0, 0, 1]])]}, "image 0 detections: "),
            ({"target_boxes": [np.array([[0, 0, 0, 1]])]}, "image 0 targets: "),
            ({"thresholds": np.array([])}, "thresholds"),
            ({"thresholds": np.array([[0.5]])}, "thresholds"),
            ({"max_detections": 0}, "max_detections"),
            ({"max_detections": True}, "max_detections"),
            ({"max_detections": 1.5}, "max_detections"),
            ({"iou": 0.0}, "iou"),
            ({"iou": 1.5}, "iou"),
        ],
    )
    def test_detection_counts_refuses(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            self._call(**overrides)
