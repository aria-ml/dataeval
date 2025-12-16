"""Tests for dual-key (image_index, target_index) indexing in Metadata."""

import numpy as np
import pytest

from dataeval._metadata import Metadata
from tests.embeddings.test_embeddings import MockDataset


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
        assert image_rows["image_index"].to_list() == [0, 1, 2]

        # Check target-level rows (2 + 1 + 3 = 6 detections)
        target_rows = md.target_data
        assert len(target_rows) == 6
        assert all(target_rows["target_index"].is_not_null())

    def test_target_index_per_image(self, od_dataset_with_metadata):
        """Test that target_index resets per image (0, 1, 2, ...)."""
        md = Metadata(od_dataset_with_metadata)
        target_rows = md.target_data

        # Image 0 should have targets 0, 1
        img0_targets = target_rows.filter(target_rows["image_index"] == 0)
        assert img0_targets["target_index"].to_list() == [0, 1]

        # Image 1 should have target 0
        img1_targets = target_rows.filter(target_rows["image_index"] == 1)
        assert img1_targets["target_index"].to_list() == [0]

        # Image 2 should have targets 0, 1, 2
        img2_targets = target_rows.filter(target_rows["image_index"] == 2)
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
        assert img0_factors["image_index"] == 0

        # Get factors for image 1
        img1_factors = md.get_image_factors(1)
        assert img1_factors["weather"] == "rainy"
        assert img1_factors["time"] == "afternoon"

    def test_get_target_factors(self, od_dataset_with_metadata):
        """Test retrieving factors for a specific target."""
        md = Metadata(od_dataset_with_metadata)

        # Get first target of image 0
        target_factors = md.get_target_factors(0, 0)
        assert target_factors["image_index"] == 0
        assert target_factors["target_index"] == 0
        assert target_factors["class_label"] == 0

        # Get second target of image 0
        target_factors = md.get_target_factors(0, 1)
        assert target_factors["image_index"] == 0
        assert target_factors["target_index"] == 1
        assert target_factors["class_label"] == 1

        # Get only target of image 1
        target_factors = md.get_target_factors(1, 0)
        assert target_factors["image_index"] == 1
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

        # Check that brightness is None in target rows
        target_rows = md.target_data
        assert all(target_rows["brightness"].is_null())

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

    def test_backward_compatibility_image_indices(self, od_dataset_with_metadata):
        """Test that item_indices maps targets back to source items."""
        md = Metadata(od_dataset_with_metadata)

        # item_indices should map to target-level mappings
        assert len(md.item_indices) == 6  # Number of targets
        assert md.item_indices.tolist() == [0, 0, 1, 2, 2, 2]
