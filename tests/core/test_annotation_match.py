import numpy as np
import pytest

from dataeval.core import annotation_divergence, annotation_fingerprint, frame_annotation_hash

BOXES = np.array([[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 50.0, 50.0]])
LABELS = np.array([1, 2])
HW = (100, 100)


@pytest.mark.required
class TestFrameAnnotationHash:
    def test_identical_annotation_agrees(self):
        assert frame_annotation_hash(BOXES, LABELS, image_hw=HW) == frame_annotation_hash(BOXES, LABELS, image_hw=HW)

    def test_box_listing_order_is_irrelevant(self):
        assert frame_annotation_hash(BOXES[::-1], LABELS[::-1], image_hw=HW) == frame_annotation_hash(
            BOXES, LABELS, image_hw=HW
        )

    def test_moved_box_disagrees(self):
        moved = BOXES.copy()
        moved[0, 0] += 5.0
        assert frame_annotation_hash(moved, LABELS, image_hw=HW) != frame_annotation_hash(BOXES, LABELS, image_hw=HW)

    def test_changed_label_disagrees(self):
        assert frame_annotation_hash(BOXES, np.array([1, 3]), image_hw=HW) != frame_annotation_hash(
            BOXES, LABELS, image_hw=HW
        )

    def test_rescaled_copy_agrees_under_normalization(self):
        assert frame_annotation_hash(BOXES * 2.0, LABELS, image_hw=(200, 200)) == frame_annotation_hash(
            BOXES, LABELS, image_hw=HW
        )

    def test_non_square_dimensions_transpose_check(self):
        """Verify width/height are applied to the correct dimensions (x div width, y div height).

        Uses an asymmetric box to catch transposition errors. Box [10, 20, 30, 40] normalized
        by (h=100, w=200) should give [0.05, 0.20, 0.15, 0.40]. If implementation transposed
        width/height, it would divide x by 100 and y by 200, giving [0.10, 0.10, 0.30, 0.20].
        """
        # Asymmetric box: x0=10, y0=20, x1=30, y1=40
        box_absolute = np.array([[10.0, 20.0, 30.0, 40.0]])
        labels = np.array([1])
        # Correct normalization: divide x by width (200), y by height (100)
        # Result: [10/200, 20/100, 30/200, 40/100] = [0.05, 0.20, 0.15, 0.40]
        box_normalized = np.array([[0.05, 0.20, 0.15, 0.40]])

        # Hash the absolute box with explicit dimensions should match
        # the pre-normalized box with no dimensions
        h1 = frame_annotation_hash(box_absolute, labels, image_hw=(100, 200))
        h2 = frame_annotation_hash(box_normalized, labels, image_hw=None)
        assert h1 == h2

    def test_mismatched_box_label_count_raises(self):
        """Boxes and labels must have matching lengths."""
        mismatched_labels = np.array([1])  # 1 label but 2 boxes
        with pytest.raises(ValueError, match="boxes, labels, and track_ids must have matching lengths"):
            frame_annotation_hash(BOXES, mismatched_labels, image_hw=HW)

    def test_mismatched_track_id_count_raises(self):
        """When track_ids is provided, it must match boxes/labels length."""
        mismatched_tracks = np.array([0])  # 1 track id but 2 boxes
        with pytest.raises(ValueError, match="boxes, labels, and track_ids must have matching lengths"):
            frame_annotation_hash(BOXES, LABELS, track_ids=mismatched_tracks, image_hw=HW)

    def test_zero_height_raises(self):
        """Non-positive image height must raise."""
        with pytest.raises(ValueError, match="image_hw dimensions must be positive"):
            frame_annotation_hash(BOXES, LABELS, image_hw=(0, 100))

    def test_zero_width_raises(self):
        """Non-positive image width must raise."""
        with pytest.raises(ValueError, match="image_hw dimensions must be positive"):
            frame_annotation_hash(BOXES, LABELS, image_hw=(100, 0))

    def test_negative_height_raises(self):
        """Negative image height must raise."""
        with pytest.raises(ValueError, match="image_hw dimensions must be positive"):
            frame_annotation_hash(BOXES, LABELS, image_hw=(-100, 100))

    def test_negative_width_raises(self):
        """Negative image width must raise."""
        with pytest.raises(ValueError, match="image_hw dimensions must be positive"):
            frame_annotation_hash(BOXES, LABELS, image_hw=(100, -100))


@pytest.mark.required
class TestAnnotationFingerprint:
    def test_frame_order_matters(self):
        a = [(BOXES, LABELS, None), (BOXES * 0.5, LABELS, None)]
        assert annotation_fingerprint(a, image_hw=HW) != annotation_fingerprint(a[::-1], image_hw=HW)

    def test_frame_count_is_part_of_identity(self):
        one = [(BOXES, LABELS, None)]
        assert annotation_fingerprint(one, image_hw=HW) != annotation_fingerprint(one * 2, image_hw=HW)

    def test_track_renumbering_is_irrelevant(self):
        a = [(BOXES, LABELS, np.array([0, 1])), (BOXES, LABELS, np.array([0, 1]))]
        b = [(BOXES, LABELS, np.array([7, 9])), (BOXES, LABELS, np.array([7, 9]))]
        assert annotation_fingerprint(a, image_hw=HW) == annotation_fingerprint(b, image_hw=HW)

    def test_track_structure_still_matters(self):
        stable = [(BOXES, LABELS, np.array([0, 1])), (BOXES, LABELS, np.array([0, 1]))]
        swapped = [(BOXES, LABELS, np.array([0, 1])), (BOXES, LABELS, np.array([1, 0]))]
        assert annotation_fingerprint(stable, image_hw=HW) != annotation_fingerprint(swapped, image_hw=HW)

    def test_an_image_is_a_one_frame_sequence(self):
        assert annotation_fingerprint([(BOXES, LABELS, None)], image_hw=HW) == annotation_fingerprint(
            [(BOXES, LABELS, None)], image_hw=HW
        )

    def test_mismatched_frame_data_raises(self):
        """Frames with mismatched boxes/labels lengths must raise."""
        mismatched = [(BOXES, np.array([1]), None)]  # 2 boxes, 1 label
        with pytest.raises(ValueError, match="boxes, labels, and track_ids must have matching lengths"):
            annotation_fingerprint(mismatched, image_hw=HW)

    def test_bad_image_hw_in_fingerprint_raises(self):
        """Non-positive image_hw dimensions must raise during fingerprint."""
        with pytest.raises(ValueError, match="image_hw dimensions must be positive"):
            annotation_fingerprint([(BOXES, LABELS, None)], image_hw=(0, 100))


@pytest.mark.required
class TestAnnotationDivergence:
    def test_identical_annotation_has_no_divergence(self):
        a = [(BOXES, LABELS, None)]
        d = annotation_divergence(a, a, image_hw_a=HW, image_hw_b=HW)
        assert d["frames_differing"] == 0
        assert d["boxes_added"] == 0
        assert d["boxes_removed"] == 0
        assert d["mean_iou"] == pytest.approx(1.0)

    def test_a_pure_addition_is_not_a_removal(self):
        a = [(BOXES[:1], LABELS[:1], None)]
        b = [(BOXES, LABELS, None)]
        d = annotation_divergence(a, b, image_hw_a=HW, image_hw_b=HW)
        assert d["boxes_added"] == 1
        assert d["boxes_removed"] == 0

    def test_a_pure_removal_is_not_an_addition(self):
        a = [(BOXES, LABELS, None)]
        b = [(BOXES[:1], LABELS[:1], None)]
        d = annotation_divergence(a, b, image_hw_a=HW, image_hw_b=HW)
        assert d["boxes_removed"] == 1
        assert d["boxes_added"] == 0

    def test_relabeled_box_counts_as_a_label_change(self):
        a = [(BOXES, LABELS, None)]
        b = [(BOXES, np.array([1, 3]), None)]
        d = annotation_divergence(a, b, image_hw_a=HW, image_hw_b=HW)
        assert d["labels_changed"] == 1
        assert d["boxes_added"] == 0
        assert d["boxes_removed"] == 0

    def test_known_iou_is_reported(self):
        a = [(np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([1]), None)]
        b = [(np.array([[0.0, 0.0, 10.0, 20.0]]), np.array([1]), None)]
        d = annotation_divergence(a, b, image_hw_a=HW, image_hw_b=HW)
        assert d["mean_iou"] == pytest.approx(0.5)

    def test_below_threshold_match_is_demoted_to_add_and_remove(self):
        """A candidate pair whose IoU falls below the threshold is not a low-quality match --
        it counts as one box removed from a and one box added in b, and contributes no IoU."""
        a = [(np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([1]), None)]
        b = [(np.array([[5.0, 5.0, 15.0, 15.0]]), np.array([1]), None)]
        d = annotation_divergence(a, b, image_hw_a=HW, image_hw_b=HW)
        assert d["boxes_added"] == 1
        assert d["boxes_removed"] == 1
        assert d["mean_iou"] is None

    def test_mean_iou_is_none_when_nothing_matched(self):
        """An empty mean is not the same claim as a zero IoU: it means there was nothing to
        average, not that the boxes were compared and found not to overlap."""
        a = [(BOXES, LABELS, None)]
        b = [(np.empty((0, 4)), np.empty((0,), dtype=int), None)]
        d = annotation_divergence(a, b, image_hw_a=HW, image_hw_b=HW)
        assert d["mean_iou"] is None

    def test_frames_compared_is_the_shorter_sequence(self):
        a = [(BOXES, LABELS, None)] * 3
        b = [(BOXES, LABELS, None)] * 2
        d = annotation_divergence(a, b, image_hw_a=HW, image_hw_b=HW)
        assert d["frames_compared"] == 2

    def test_tracks_split_is_directional(self):
        """A track in a that fans out to more than one track in b counts as split. The reverse
        direction is a separate call and can, and here does, give a different answer."""
        box = np.array([[10.0, 10.0, 20.0, 20.0]])
        label = np.array([1])
        a = [(box, label, np.array([0])), (box, label, np.array([0]))]
        b = [(box, label, np.array([100])), (box, label, np.array([200]))]

        forward = annotation_divergence(a, b, image_hw_a=HW, image_hw_b=HW)
        backward = annotation_divergence(b, a, image_hw_a=HW, image_hw_b=HW)

        assert forward["tracks_split"] == 1
        assert backward["tracks_split"] == 0

    def test_ragged_frame_in_a_raises(self):
        a = [(BOXES, LABELS[:1], None)]  # 2 boxes, 1 label
        b = [(BOXES, LABELS, None)]
        with pytest.raises(ValueError, match="boxes, labels, and track_ids must have matching lengths"):
            annotation_divergence(a, b, image_hw_a=HW, image_hw_b=HW)

    def test_ragged_frame_in_b_raises(self):
        a = [(BOXES, LABELS, None)]
        b = [(BOXES, LABELS[:1], None)]  # 2 boxes, 1 label
        with pytest.raises(ValueError, match="boxes, labels, and track_ids must have matching lengths"):
            annotation_divergence(a, b, image_hw_a=HW, image_hw_b=HW)

    def test_ragged_track_ids_raises(self):
        a = [(BOXES, LABELS, np.array([0]))]  # 2 boxes, 1 track id
        b = [(BOXES, LABELS, None)]
        with pytest.raises(ValueError, match="boxes, labels, and track_ids must have matching lengths"):
            annotation_divergence(a, b, image_hw_a=HW, image_hw_b=HW)

    def test_tracks_split_detects_a_transposed_pairing(self):
        """A single box per frame can never expose a transposed (row, col) pair: with one
        candidate on each side the assignment is always ``[(0, 0)]``, and swapping the two
        indices of a 1-element pairing changes nothing. This uses three non-overlapping boxes
        per side, arranged as two different 3-cycles across two frames, so the correct pairing
        and a row/col-swapped one genuinely disagree.

        P0, P1, P2 are three boxes that do not overlap each other. Frame 1 puts a's boxes at
        (P0, P1, P2) and b's at (P1, P2, P0) -- a one-step rotation. Frame 2 puts a's boxes at
        (P0, P1, P2) again and b's at (P2, P0, P1) -- a rotation the other way. Under the correct
        (row, col) pairing, every a-track lands on the same b-track in both frames (track 1 always
        matches b-track 12, track 2 always matches 10, track 3 always matches 11), so nothing
        splits. A row/col transpose instead follows the *inverse* of each frame's rotation, which
        is a different rotation in frame 1 than in frame 2 -- every a-track then lands on two
        different b-tracks across the two frames, and all three tracks would count as split.
        """
        p0 = [0.0, 0.0, 10.0, 10.0]
        p1 = [30.0, 30.0, 40.0, 40.0]
        p2 = [60.0, 60.0, 70.0, 70.0]
        labels = np.array([1, 1, 1])

        boxes_a = np.array([p0, p1, p2])
        a = [
            (boxes_a, labels, np.array([1, 2, 3])),
            (boxes_a, labels, np.array([1, 2, 3])),
        ]
        b = [
            (np.array([p1, p2, p0]), labels, np.array([10, 11, 12])),
            (np.array([p2, p0, p1]), labels, np.array([11, 12, 10])),
        ]

        d = annotation_divergence(a, b, image_hw_a=HW, image_hw_b=HW)
        assert d["tracks_split"] == 0
