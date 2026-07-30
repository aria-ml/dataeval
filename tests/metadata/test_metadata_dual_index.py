"""Tests for dual-key (item_index, target_index) indexing in Metadata."""

import logging

import numpy as np
import pytest

from dataeval._metadata import Metadata
from dataeval.exceptions import ShapeMismatchError
from tests.embeddings.test_embeddings import MockDataset


def _od_targets():
    """Targets for a 3-image dataset with 2/1/3 detections."""
    from dataclasses import dataclass

    @dataclass
    class ODTarget:
        boxes: np.ndarray
        labels: np.ndarray
        scores: np.ndarray

    return [
        ODTarget(
            boxes=np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
            labels=np.array([0, 1]),
            scores=np.array([0.9, 0.8]),
        ),
        ODTarget(boxes=np.array([[5, 5, 15, 15]]), labels=np.array([1]), scores=np.array([0.95])),
        ODTarget(
            boxes=np.array([[1, 1, 5, 5], [10, 10, 20, 20], [25, 25, 35, 35]]),
            labels=np.array([0, 0, 2]),
            scores=np.array([0.85, 0.75, 0.92]),
        ),
    ]


@pytest.fixture
def od_dataset_varied_pixels():
    """Same layout as od_dataset_with_metadata, but every image and every box has a distinct mean.

    Uniform images make row-alignment bugs invisible: a misplaced value equals the value it
    displaced. Random pixels give each of the 9 (image, target) stats a unique value.
    """
    rng = np.random.default_rng(0)
    metadata = [
        {"weather": "sunny", "time": "morning"},
        {"weather": "rainy", "time": "afternoon"},
        {"weather": "cloudy", "time": "evening"},
    ]
    return MockDataset(rng.random((3, 3, 32, 32)), _od_targets(), metadata)


@pytest.fixture
def od_dataset_with_metadata():
    """Create a small OD dataset with metadata for testing."""
    from dataclasses import dataclass

    @dataclass
    class ODTarget:
        boxes: np.ndarray
        labels: np.ndarray
        scores: np.ndarray

    # 3 images with varying numbers of detections
    data = np.ones((3, 3, 32, 32))
    targets = [
        ODTarget(
            boxes=np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
            labels=np.array([0, 1]),
            scores=np.array([0.9, 0.8]),
        ),
        ODTarget(boxes=np.array([[5, 5, 15, 15]]), labels=np.array([1]), scores=np.array([0.95])),
        ODTarget(
            boxes=np.array([[1, 1, 5, 5], [10, 10, 20, 20], [25, 25, 35, 35]]),
            labels=np.array([0, 0, 2]),
            scores=np.array([0.85, 0.75, 0.92]),
        ),
    ]
    metadata = [
        {"weather": "sunny", "time": "morning"},
        {"weather": "rainy", "time": "afternoon"},
        {"weather": "cloudy", "time": "evening"},
    ]

    return MockDataset(data, targets, metadata)


class TestDualKeyIndexing:
    """Test dual-key indexing with image_index and target_index."""

    def test_dataframe_structure(self, od_dataset_with_metadata):
        """Test that dataframe has both image-level and target-level rows."""
        md = Metadata(od_dataset_with_metadata)

        # Check that target_index column exists
        assert "target_index" in md.dataframe.columns

        # Total rows should be: 3 image rows + 6 target rows = 9
        assert len(md.dataframe) == 9

        # Check image-level rows (3 images)
        image_rows = md.image_data
        assert len(image_rows) == 3
        assert all(image_rows["target_index"].is_null())
        assert image_rows["item_index"].to_list() == [0, 1, 2]

        # Check target-level rows (2 + 1 + 3 = 6 detections)
        target_rows = md.target_data
        assert len(target_rows) == 6
        assert all(target_rows["target_index"].is_not_null())

    def test_target_index_per_image(self, od_dataset_with_metadata):
        """Test that target_index resets per image (0, 1, 2, ...)."""
        md = Metadata(od_dataset_with_metadata)
        target_rows = md.target_data

        # Image 0 should have targets 0, 1
        img0_targets = target_rows.filter(target_rows["item_index"] == 0)
        assert img0_targets["target_index"].to_list() == [0, 1]

        # Image 1 should have target 0
        img1_targets = target_rows.filter(target_rows["item_index"] == 1)
        assert img1_targets["target_index"].to_list() == [0]

        # Image 2 should have targets 0, 1, 2
        img2_targets = target_rows.filter(target_rows["item_index"] == 2)
        assert img2_targets["target_index"].to_list() == [0, 1, 2]

    def test_image_level_metadata_no_duplication(self, od_dataset_with_metadata):
        """Test that image-level metadata is stored only in image rows."""
        md = Metadata(od_dataset_with_metadata)

        # Image-level rows should have metadata
        image_rows = md.image_data
        assert image_rows["weather"].to_list() == ["sunny", "rainy", "cloudy"]
        assert image_rows["time"].to_list() == ["morning", "afternoon", "evening"]

        # Target-level rows should have None for image-level metadata
        target_rows = md.target_data
        assert all(target_rows["weather"].is_not_null())
        assert all(target_rows["time"].is_not_null())

    def test_target_level_data(self, od_dataset_with_metadata):
        """Test that target-level data (class_label, score, box) is only in target rows."""
        md = Metadata(od_dataset_with_metadata)

        # Image rows should have None for target-level fields
        image_rows = md.image_data
        assert all(image_rows["class_label"].is_null())
        assert all(image_rows["score"].is_null())
        assert all(image_rows["box"].is_null())

        # Target rows should have actual values
        target_rows = md.target_data
        assert target_rows["class_label"].to_list() == [0, 1, 1, 0, 0, 2]
        assert len(target_rows["score"].to_list()) == 6
        assert len(target_rows["box"].to_list()) == 6

    def test_get_image_factors(self, od_dataset_with_metadata):
        """Test retrieving factors for a specific image."""
        md = Metadata(od_dataset_with_metadata)

        # Get factors for image 0
        img0_factors = md.get_image_factors(0)
        assert img0_factors["weather"] == "sunny"
        assert img0_factors["time"] == "morning"
        assert img0_factors["item_index"] == 0

        # Get factors for image 1
        img1_factors = md.get_image_factors(1)
        assert img1_factors["weather"] == "rainy"
        assert img1_factors["time"] == "afternoon"

    def test_get_target_factors(self, od_dataset_with_metadata):
        """Test retrieving factors for a specific target."""
        md = Metadata(od_dataset_with_metadata)

        # Get first target of image 0
        target_factors = md.get_target_factors(0, 0)
        assert target_factors["item_index"] == 0
        assert target_factors["target_index"] == 0
        assert target_factors["class_label"] == 0

        # Get second target of image 0
        target_factors = md.get_target_factors(0, 1)
        assert target_factors["item_index"] == 0
        assert target_factors["target_index"] == 1
        assert target_factors["class_label"] == 1

        # Get only target of image 1
        target_factors = md.get_target_factors(1, 0)
        assert target_factors["item_index"] == 1
        assert target_factors["target_index"] == 0
        assert target_factors["class_label"] == 1

    def test_add_image_level_factors(self, od_dataset_with_metadata):
        """Test adding image-level factors."""
        md = Metadata(od_dataset_with_metadata)

        # Add image-level factors (3 values for 3 images)
        brightness = [0.5, 0.7, 0.3]
        md.add_factors({"brightness": brightness}, level="image")

        # Check that brightness is in image rows
        image_rows = md.image_data
        assert image_rows["brightness"].to_list() == brightness

        # Check that brightness is replicated to target rows via item_index mapping
        target_rows = md.target_data
        target_item_indices = target_rows["item_index"].to_list()
        expected_target_brightness = [brightness[i] for i in target_item_indices]
        assert target_rows["brightness"].to_list() == expected_target_brightness

    def test_add_target_level_factors(self, od_dataset_with_metadata):
        """Test adding target-level factors."""
        md = Metadata(od_dataset_with_metadata)

        # Add target-level factors (6 values for 6 detections)
        iou = [0.9, 0.8, 0.95, 0.85, 0.75, 0.92]
        md.add_factors({"iou": iou}, level="target")

        # Check that iou is None in image rows
        image_rows = md.image_data
        assert all(image_rows["iou"].is_null())

        # Check that iou is in target rows
        target_rows = md.target_data
        assert target_rows["iou"].to_list() == iou

    def test_add_factors_auto_level(self, od_dataset_with_metadata):
        """Test auto-detection of factor level."""
        md = Metadata(od_dataset_with_metadata)

        # Add factors with length matching image count (should auto-detect as image-level)
        brightness = [0.5, 0.7, 0.3]
        md.add_factors({"brightness": brightness})  # level="auto" by default

        image_rows = md.image_data
        assert image_rows["brightness"].to_list() == brightness

        # Add factors with length matching target count (should auto-detect as target-level)
        iou = [0.9, 0.8, 0.95, 0.85, 0.75, 0.92]
        md.add_factors({"iou": iou})

        target_rows = md.target_data
        assert target_rows["iou"].to_list() == iou

    def test_add_factors_wrong_length_raises(self, od_dataset_with_metadata):
        """Test that adding factors with wrong length raises ValueError."""
        md = Metadata(od_dataset_with_metadata)

        # Wrong length for image-level
        with pytest.raises(ValueError, match="different length"):
            md.add_factors({"bad_factor": [1, 2]})  # Only 2 values, need 3 or 6

        # Wrong length with explicit level
        with pytest.raises(ValueError, match="image count"):
            md.add_factors({"bad_factor": [1, 2]}, level="image")

        with pytest.raises(ValueError, match="target count"):
            md.add_factors({"bad_factor": [1, 2]}, level="target")

    def test_factor_info_level_od_dataset(self, od_dataset_with_metadata):
        """Test that factor_info.level distinguishes image vs target factors on OD datasets."""
        md = Metadata(od_dataset_with_metadata)

        # Built-in factors from metadata: weather and time are image-level
        info = md.factor_info
        assert info["weather"].level == "image"
        assert info["time"].level == "image"

    def test_factor_info_level_added_factors(self, od_dataset_with_metadata):
        """Test that added factors get the correct level in factor_info."""
        md = Metadata(od_dataset_with_metadata)

        md.add_factors({"brightness": [0.5, 0.7, 0.3]}, level="image")
        md.add_factors({"iou": [0.9, 0.8, 0.95, 0.85, 0.75, 0.92]}, level="target")

        info = md.factor_info
        assert info["brightness"].level == "image"
        assert info["iou"].level == "target"

    def test_factor_info_level_ic_dataset(self):
        """Test that IC dataset factors default to image level."""
        from tests.conftest import to_metadata

        md = to_metadata({"weather": ["sunny", "rainy"] * 25}, list(range(50)))
        info = md.factor_info
        assert info["weather"].level == "image"

    def test_add_factors_mixed_levels_od_dataset(self, od_dataset_with_metadata):
        """Test that we can add mixed-level factors (image, target, combined) in a single call with level='auto'."""
        md = Metadata(od_dataset_with_metadata)

        # Image-level factor: 3 images
        brightness = [0.5, 0.7, 0.3]
        # Target-level factor: 6 targets
        iou = [0.9, 0.8, 0.95, 0.85, 0.75, 0.92]
        # Combined-level factor: 3 images + 6 targets = 9 elements, in source-index order
        combined_stat = [0.1, 0.4, 0.5, 0.2, 0.6, 0.3, 0.7, 0.8, 0.9]

        md.add_factors(
            {
                "added_brightness": brightness,
                "added_iou": iou,
                "added_combined": combined_stat,
            },
            level="auto",
        )

        info = md.factor_info
        assert info["added_brightness"].level == "image"
        assert info["added_iou"].level == "target"
        assert info["added_combined"].level == "combined"

        # Values must land on the rows they describe, not merely be present.
        assert md.image_data["added_brightness"].to_list() == brightness
        assert md.target_data["added_iou"].to_list() == iou
        assert md.image_data["added_combined"].to_list() == [0.1, 0.2, 0.3]
        assert md.target_data["added_combined"].to_list() == [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def test_add_factors_combined_uses_source_index_order(self, od_dataset_with_metadata):
        """A combined array is ordered by (item, target) with the image row first, not blockwise."""
        md = Metadata(od_dataset_with_metadata)

        # source-index order for 3 images with 2/1/3 detections:
        #   (0,-) (0,0) (0,1) (1,-) (1,0) (2,-) (2,0) (2,1) (2,2)
        md.add_factors({"cs": np.arange(9.0)}, level="combined")

        rows = md.dataframe.select("item_index", "target_index", "cs").rows()
        assert rows == [
            (0, None, 0.0),
            (1, None, 3.0),
            (2, None, 5.0),
            (0, 0, 1.0),
            (0, 1, 2.0),
            (1, 0, 4.0),
            (2, 0, 6.0),
            (2, 1, 7.0),
            (2, 2, 8.0),
        ]

    def test_add_factors_from_compute_stats_and_ratios(self, od_dataset_varied_pixels):
        """Outputs of compute_stats and compute_ratios wire into Metadata with values on the right rows."""
        from dataeval.core import compute_ratios, compute_stats
        from dataeval.flags import ImageStats

        md = Metadata(od_dataset_varied_pixels)

        # 1. Image-level stats only -> one value per image
        img_stats = compute_stats(
            od_dataset_varied_pixels,
            stats=ImageStats.PIXEL_MEAN,
            per_image=True,
            per_target=False,
            per_channel=False,
            normalize_pixel_values=False,
        )
        mean_img = img_stats["stats"]["mean"]

        # 2. Target-level stats only -> one value per detection
        tgt_stats = compute_stats(
            od_dataset_varied_pixels,
            stats=ImageStats.PIXEL_MEAN | ImageStats.DIMENSION,
            per_image=False,
            per_target=True,
            per_channel=False,
            normalize_pixel_values=False,
        )
        width_tgt = tgt_stats["stats"]["width"]
        mean_tgt = tgt_stats["stats"]["mean"]

        # 3. Combined stats -> image and target values interleaved in source-index order
        combined_stats = compute_stats(
            od_dataset_varied_pixels,
            stats=ImageStats.PIXEL_MEAN,
            per_image=True,
            per_target=True,
            per_channel=False,
            normalize_pixel_values=False,
        )
        mean_combined = combined_stats["stats"]["mean"]

        # 4. Ratios -> one value per detection
        ratios = compute_ratios(combined_stats)
        mean_ratio = ratios["stats"]["mean"]

        # Guard the guard: if every mean were equal, a misaligned column would still "pass".
        assert len(set(np.round(mean_combined, 6))) == 9

        md.add_factors(
            {
                "image_mean": mean_img,
                "target_width": width_tgt,
                "combined_mean": mean_combined,
                "target_mean_ratio": mean_ratio,
            },
            level="auto",
        )

        info = md.factor_info
        assert info["image_mean"].level == "image"
        assert info["target_width"].level == "target"
        assert info["combined_mean"].level == "combined"
        assert info["target_mean_ratio"].level == "target"

        # The combined column must decompose back into the independently-computed
        # image-only and target-only stats. This is what catches a bad permutation.
        assert md.image_data["combined_mean"].to_numpy() == pytest.approx(mean_img)
        assert md.target_data["combined_mean"].to_numpy() == pytest.approx(mean_tgt)

        assert md.image_data["image_mean"].to_numpy() == pytest.approx(mean_img)
        assert md.target_data["target_width"].to_numpy() == pytest.approx(width_tgt)
        assert md.target_data["target_mean_ratio"].to_numpy() == pytest.approx(mean_ratio)

    def test_backward_compatibility_image_indices(self, od_dataset_with_metadata):
        """Test that item_indices maps targets back to source items."""
        md = Metadata(od_dataset_with_metadata)

        # item_indices should map to target-level mappings
        assert len(md.item_indices) == 6  # Number of targets
        assert md.item_indices.tolist() == [0, 0, 1, 2, 2, 2]


class TestAddFactorsRobustness:
    """add_factors must not corrupt or silently reshape the Metadata it extends."""

    def test_validation_failure_leaves_metadata_untouched(self, od_dataset_with_metadata):
        """A bad factor anywhere in the mapping must not half-apply the good ones."""
        md = Metadata(od_dataset_with_metadata)
        before = md.factor_names

        with pytest.raises(ShapeMismatchError):
            md.add_factors({"good": [1, 2, 3], "bad": [1, 2]})

        # Registering "good" without writing its column would leave every factor accessor
        # raising ColumnNotFoundError.
        assert md.factor_names == before
        assert "good" not in md.dataframe.columns
        assert md.raw_data.shape[1] == len(before)
        assert md.factor_data.shape[1] == len(before)

    def test_overwrite_drops_stale_binned_columns(self, od_dataset_with_metadata):
        """Overwriting an already-binned factor must re-bin it, not orphan it."""
        md = Metadata(od_dataset_with_metadata)

        md.add_factors({"bright": [0.1, 0.5, 0.9]}, level="image")
        assert md.factor_info["bright"].factor_type is not None  # forces binning

        md.add_factors({"bright": [9.9, 9.8, 9.7]}, level="image", overwrite=True)

        # A leftover binned/digitized companion makes _bin() skip the factor, so it vanishes
        # from factor_info while still being counted in factor_names.
        assert "bright" in md.factor_info
        assert md.factor_data.shape[1] == len(md.factor_names)
        assert md.image_data["bright"].to_list() == [9.9, 9.8, 9.7]

    def test_reserved_column_is_not_clobbered(self, od_dataset_with_metadata):
        """A factor named after a reserved column is stored under a metadata_ prefix."""
        md = Metadata(od_dataset_with_metadata)

        md.add_factors({"target_index": [1, 2, 3]}, level="image")

        # Writing over target_index would collapse the image/target row split entirely.
        assert len(md.image_data) == 3
        assert len(md.target_data) == 6
        assert md.image_data["metadata_target_index"].to_list() == [1, 2, 3]

        # overwrite=True is not an escape hatch onto the reserved column either.
        md.add_factors({"target_index": [4, 5, 6]}, level="image", overwrite=True)
        assert len(md.image_data) == 3
        assert md.image_data["metadata_target_index"].to_list() == [4, 5, 6]

    def test_excluded_factor_is_not_clobbered(self, od_dataset_with_metadata):
        """The collision guard covers filtered-out factors, not just visible ones."""
        md = Metadata(od_dataset_with_metadata, exclude=["weather"])
        assert "weather" not in md.factor_names

        md.add_factors({"weather": ["a", "b", "c"]}, level="image")

        assert md.image_data["weather"].to_list() == ["sunny", "rainy", "cloudy"]
        assert md.image_data["weather_added"].to_list() == ["a", "b", "c"]

    def test_repeated_adds_never_overwrite_earlier_values(self, od_dataset_with_metadata):
        """Each add of a colliding name claims a fresh column, not the previous suffixed one."""
        md = Metadata(od_dataset_with_metadata)

        md.add_factors({"b": [1.0, 2.0, 3.0]}, level="image")
        md.add_factors({"b": [4.0, 5.0, 6.0]}, level="image")
        md.add_factors({"b": [7.0, 8.0, 9.0]}, level="image")

        images = md.image_data
        assert images["b"].to_list() == [1.0, 2.0, 3.0]
        assert images["b_added"].to_list() == [4.0, 5.0, 6.0]
        assert images["b_added_2"].to_list() == [7.0, 8.0, 9.0]

    def test_collision_within_a_single_call(self, od_dataset_with_metadata):
        """Two keys resolving to the same column name must not produce duplicate Series.

        `b_added` is excluded so it is invisible to a factor_names-based collision check,
        which is what lets both keys resolve to the same name.
        """
        md = Metadata(od_dataset_with_metadata, exclude=["b_added"])

        md.add_factors({"b": [1.0, 2.0, 3.0]}, level="image")
        md.add_factors({"b": [4.0, 5.0, 6.0], "b_added": [7.0, 8.0, 9.0]}, level="image")

        images = md.image_data
        assert images["b"].to_list() == [1.0, 2.0, 3.0]
        assert images["b_added"].to_list() == [4.0, 5.0, 6.0]
        assert images["b_added_added"].to_list() == [7.0, 8.0, 9.0]

    def test_overwrite_reuses_the_same_column(self, od_dataset_with_metadata):
        """overwrite=True replaces in place instead of accumulating suffixed columns."""
        md = Metadata(od_dataset_with_metadata)

        md.add_factors({"b": [1.0, 2.0, 3.0]}, level="image")
        md.add_factors({"b": [4.0, 5.0, 6.0]}, level="image", overwrite=True)

        assert "b_added" not in md.dataframe.columns
        assert md.image_data["b"].to_list() == [4.0, 5.0, 6.0]

    def test_target_and_combined_levels_rejected_without_targets(self):
        """An IC dataset has no target rows; asking for one must fail loudly and early."""
        md = Metadata(MockDataset(np.ones((5, 3, 16, 16)), np.eye(3)[[0, 1, 0, 1, 0]], [{"w": i} for i in range(5)]))
        assert not md.has_targets()

        for level in ("target", "combined"):
            with pytest.raises(ValueError, match="no targets"):
                md.add_factors({"foo": np.arange(10.0)}, level=level)

        # A polars ShapeError from deep inside with_columns is not an acceptable substitute,
        # and the rejected factor must not linger in any state.
        assert "foo" not in md.factor_names
        assert "foo" not in md.dataframe.columns

    def test_multidimensional_factors_are_reported_not_silently_dropped(self, od_dataset_with_metadata, caplog):
        """Vector-valued stats have no single-column form; the caller must be told they were skipped."""
        md = Metadata(od_dataset_with_metadata)

        with caplog.at_level(logging.WARNING, logger="dataeval._metadata"):
            md.add_factors({"percentiles": np.random.rand(3, 5), "ok": [1.0, 2.0, 3.0]})

        assert "percentiles" in caplog.text
        assert md.dropped_factors["percentiles"] == ["multi_dimensional"]
        assert "percentiles" not in md.factor_names
        assert "ok" in md.factor_names

    def test_compute_stats_output_with_vector_stats_adds_the_scalar_ones(self, od_dataset_varied_pixels):
        """The h2_add_intrinsic_factors pattern: pipe compute_stats["stats"] straight in."""
        from dataeval.core import compute_stats
        from dataeval.flags import ImageStats

        md = Metadata(od_dataset_varied_pixels)
        results = compute_stats(
            od_dataset_varied_pixels,
            stats=ImageStats.PIXEL | ImageStats.VISUAL,
            per_channel=False,
            normalize_pixel_values=False,
        )

        md.add_factors(results["stats"])

        scalar_stats = {k for k, v in results["stats"].items() if np.asarray(v).ndim == 1}
        assert scalar_stats <= set(md.factor_names)
        # histogram/percentiles are vector-valued and cannot become columns
        assert set(md.dropped_factors) >= {"histogram", "percentiles"} & set(results["stats"])

    def test_combined_factor_survives_target_factors_only(self, od_dataset_with_metadata):
        """Combined factors carry per-target values, so target-only mode must keep them."""
        md = Metadata(od_dataset_with_metadata)
        md.add_factors({"cs": np.arange(9.0)}, level="combined")

        md.target_factors_only = True

        assert "cs" in md.factor_names
        assert md.target_data["cs"].to_list() == [1.0, 2.0, 4.0, 6.0, 7.0, 8.0]
        assert md.factor_data.shape == (6, len(md.factor_names))

    def test_relevelling_a_factor_clears_stale_membership(self, od_dataset_with_metadata):
        """Re-adding a factor at a different level must not leave it registered at both."""
        md = Metadata(od_dataset_with_metadata)

        md.add_factors({"x": np.arange(6.0)}, level="target")
        assert md.factor_info["x"].level == "target"

        md.add_factors({"x": np.arange(3.0)}, level="image", overwrite=True)

        assert md.factor_info["x"].level == "image"
        assert "x" in md._image_factors
        assert "x" not in md._target_factors
        # Stale target membership would silently drop "x" from target-only mode.
        md.target_factors_only = True
        assert "x" not in md.factor_names

    def test_combined_factor_feeds_evaluators_aligned_with_class_labels(self, od_dataset_with_metadata):
        """factor_data rows must stay aligned with class_labels once a combined factor is present."""
        md = Metadata(od_dataset_with_metadata)
        md.add_factors({"cs": np.arange(9.0)}, level="combined")

        assert md.factor_data.shape[0] == len(md.class_labels) == 6
        assert md.factor_data.shape[1] == len(md.factor_names) == len(md.is_discrete)
