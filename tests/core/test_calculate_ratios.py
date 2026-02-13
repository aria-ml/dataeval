import numpy as np
import pytest

from dataeval.core import calculate, calculate_ratios
from dataeval.core._calculate import SOURCE_INDEX
from dataeval.flags import ImageStats
from dataeval.types import SourceIndex


class TestCalculateRatios:
    """Test calculate_ratios function with various scenarios."""

    def test_basic_dimension_ratios(self):
        """Test basic dimension ratio calculations."""
        # Create simple test data: 2 images, 2 boxes each
        images = [
            np.random.random((3, 100, 200)),  # Image 0: 100x200
            np.random.random((3, 50, 100)),  # Image 1: 50x100
        ]
        boxes = [
            [[10, 20, 50, 60], [30, 40, 80, 90]],  # Image 0: boxes of size 40x40 and 50x50
            [[5, 10, 25, 30], [15, 20, 45, 80]],  # Image 1: boxes of size 20x20 and 30x60
        ]

        # Calculate both image and box stats
        stats = calculate(
            images,
            boxes,
            stats=ImageStats.DIMENSION,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        # Calculate ratios
        ratios = calculate_ratios(stats)

        # Verify we only have box entries (no image entries)
        assert all(si.target is not None for si in ratios[SOURCE_INDEX])

        # Verify we have 4 box entries (2 boxes per image, 2 images)
        assert len(ratios[SOURCE_INDEX]) == 4

        # Check width ratios
        # Image 0: box 0 width = 40, image width = 200, ratio = 0.2
        # Image 0: box 1 width = 50, image width = 200, ratio = 0.25
        # Image 1: box 0 width = 20, image width = 100, ratio = 0.2
        # Image 1: box 1 width = 30, image width = 100, ratio = 0.3
        assert ratios["stats"]["width"][0] == pytest.approx(40.0 / 200.0, abs=1e-3)
        assert ratios["stats"]["width"][1] == pytest.approx(50.0 / 200.0, abs=1e-3)
        assert ratios["stats"]["width"][2] == pytest.approx(20.0 / 100.0, abs=1e-3)
        assert ratios["stats"]["width"][3] == pytest.approx(30.0 / 100.0, abs=1e-3)

    def test_pixel_ratios(self):
        """Test pixel statistics ratio calculations."""
        images = [
            np.ones((3, 50, 50)) * 0.8,  # Image with high pixel values
        ]
        boxes = [
            [[10, 10, 30, 30]],  # Single box
        ]

        stats = calculate(
            images,
            boxes,
            stats=ImageStats.PIXEL,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        ratios = calculate_ratios(stats)

        # Mean ratio should be close to 1.0 since box and image have similar values
        assert ratios["stats"]["mean"][0] == pytest.approx(1.0, abs=0.1)

        # Verify we have the expected pixel stats
        assert "mean" in ratios["stats"]
        assert "std" in ratios["stats"]
        assert "var" in ratios["stats"]

    def test_per_channel_ratios(self):
        """Test ratio calculations with per-channel statistics."""
        images = [
            np.random.random((3, 50, 50)),
        ]
        boxes = [
            [[10, 10, 30, 30], [15, 15, 35, 35]],  # 2 boxes
        ]

        stats = calculate(
            images,
            boxes,
            stats=ImageStats.PIXEL,
            per_image=True,
            per_target=True,
            per_channel=True,
        )

        ratios = calculate_ratios(stats)

        # With per_channel=True and 3 channels, we should have 2 boxes * 3 channels = 6 entries
        assert len(ratios[SOURCE_INDEX]) == 6

        # Verify channel indices are preserved
        channels_seen = {si.channel for si in ratios[SOURCE_INDEX]}
        assert channels_seen == {0, 1, 2}

        # Verify each box has entries for all 3 channels
        box_0_channels = [si.channel for si in ratios[SOURCE_INDEX] if si.target == 0]
        box_1_channels = [si.channel for si in ratios[SOURCE_INDEX] if si.target == 1]
        assert set(box_0_channels) == {0, 1, 2}
        assert set(box_1_channels) == {0, 1, 2}

    def test_visual_ratios(self):
        """Test visual statistics ratio calculations."""
        images = [
            np.random.random((3, 50, 50)),
        ]
        boxes = [
            [[10, 10, 30, 30]],
        ]

        stats = calculate(
            images,
            boxes,
            stats=ImageStats.VISUAL,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        ratios = calculate_ratios(stats)

        # Verify we have visual stats
        assert "brightness" in ratios["stats"]
        assert "contrast" in ratios["stats"]
        assert "darkness" in ratios["stats"]
        assert "sharpness" in ratios["stats"]

    def test_offset_override(self):
        """Test that offset_x and offset_y use custom override calculations."""
        images = [
            np.random.random((3, 100, 200)),  # 100 tall, 200 wide
        ]
        boxes = [
            [[50, 25, 100, 75]],  # Box at x=50, y=25, width=50, height=50
        ]

        stats = calculate(
            images,
            boxes,
            stats=ImageStats.DIMENSION,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        ratios = calculate_ratios(stats)

        # offset_x should be normalized by image width (200)
        # offset_x = 50, image width = 200, ratio = 0.25
        assert ratios["stats"]["offset_x"][0] == pytest.approx(50.0 / 200.0, abs=1e-3)

        # offset_y should be normalized by image height (100)
        # offset_y = 25, image height = 100, ratio = 0.25
        assert ratios["stats"]["offset_y"][0] == pytest.approx(25.0 / 100.0, abs=1e-3)

    def test_channels_depth_override(self):
        """Test that channels and depth are preserved from box stats (not divided)."""
        images = [
            np.random.random((3, 50, 50)),
        ]
        boxes = [
            [[10, 10, 30, 30]],
        ]

        stats = calculate(
            images,
            boxes,
            stats=ImageStats.DIMENSION,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        ratios = calculate_ratios(stats)

        # Channels should be preserved from box stats (= 3)
        assert ratios["stats"]["channels"][0] == 3

        # Depth should be preserved from box stats
        assert "depth" in ratios["stats"]

    def test_custom_override_map(self):
        """Test using a custom override map."""
        images = [
            np.ones((3, 50, 50)) * 2.0,
        ]
        boxes = [
            [[10, 10, 30, 30]],
        ]

        stats = calculate(
            images,
            boxes,
            stats=ImageStats.PIXEL,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        # Custom override that returns a constant
        custom_map = {
            "mean": lambda box, img: 42.0,
        }

        ratios = calculate_ratios(stats, override_map=custom_map)  # type: ignore

        # Mean should use custom calculation
        assert ratios["stats"]["mean"][0] == 42.0

        # Other stats should use default ratio calculation
        assert "std" in ratios["stats"]

    def test_missing_image_stats_raises_error(self):
        """Test that error is raised if no image-level stats present."""
        images = [np.random.random((3, 50, 50))]
        boxes = [[[10, 10, 30, 30]]]

        # Only calculate box stats (per_image=False)
        stats = calculate(
            images,
            boxes,
            stats=ImageStats.DIMENSION,
            per_image=False,
            per_target=True,
            per_channel=False,
        )

        with pytest.raises(ValueError, match="must contain image-level statistics"):
            calculate_ratios(stats)

    def test_missing_tgt_stats_raises_error(self):
        """Test that error is raised if no box-level stats present."""
        images = [np.random.random((3, 50, 50))]

        # Only calculate image stats (no boxes)
        stats = calculate(
            images,
            None,
            stats=ImageStats.DIMENSION,
            per_image=True,
            per_target=False,
            per_channel=False,
        )

        with pytest.raises(ValueError, match="must contain box-level statistics"):
            calculate_ratios(stats)

    def test_missing_source_index_raises_error(self):
        """Test that error is raised if source_index is missing."""
        bad_stats = {
            "width": [100, 200],
            "height": [50, 75],
        }

        with pytest.raises(KeyError, match="source_index"):
            calculate_ratios(bad_stats)  # type: ignore

    def test_preserves_base_attributes(self):
        """Test that base attributes are preserved in output."""
        images = [
            np.random.random((3, 50, 50)),
            np.random.random((3, 50, 50)),
        ]
        boxes = [
            [[10, 10, 30, 30]],
            [[15, 15, 35, 35], [5, 5, 25, 25]],
        ]

        stats = calculate(
            images,
            boxes,
            stats=ImageStats.DIMENSION,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        ratios = calculate_ratios(stats)

        # Verify base attributes are present
        assert SOURCE_INDEX in ratios
        assert "object_count" in ratios
        assert "invalid_box_count" in ratios
        assert "image_count" in ratios

        # Verify values match original
        assert ratios["object_count"] == stats["object_count"]
        assert ratios["invalid_box_count"] == stats["invalid_box_count"]
        assert ratios["image_count"] == stats["image_count"]

    def test_multiple_images_multiple_boxes(self):
        """Test with multiple images and varying box counts."""
        images = [
            np.random.random((3, 100, 100)),
            np.random.random((3, 200, 200)),
            np.random.random((3, 150, 150)),
        ]
        boxes = [
            [[10, 10, 50, 50]],  # 1 box
            [[20, 20, 80, 80], [30, 30, 90, 90]],  # 2 boxes
            [[15, 15, 45, 45], [25, 25, 55, 55], [35, 35, 65, 65]],  # 3 boxes
        ]

        stats = calculate(
            images,
            boxes,
            stats=ImageStats.DIMENSION,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        ratios = calculate_ratios(stats)

        # Should have 1 + 2 + 3 = 6 box entries
        assert len(ratios[SOURCE_INDEX]) == 6

        # Verify image indices are correct
        image_indices = [si.item for si in ratios[SOURCE_INDEX]]
        assert image_indices == [0, 1, 1, 2, 2, 2]

        # Verify box indices are correct
        box_indices = [si.target for si in ratios[SOURCE_INDEX]]
        assert box_indices == [0, 0, 1, 0, 1, 2]

    def test_divide_by_zero_handling(self):
        """Test that division by zero is handled gracefully."""
        # Create image with zero values
        images = [
            np.zeros((3, 50, 50)),
        ]
        boxes = [
            [[10, 10, 30, 30]],
        ]

        stats = calculate(
            images,
            boxes,
            stats=ImageStats.PIXEL,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        # Should not raise error
        ratios = calculate_ratios(stats)

        # Check that results don't contain Inf
        # (NaN is acceptable for some stats like skew/kurtosis with constant values)
        for key, values in ratios["stats"].items():
            if isinstance(values, list | np.ndarray):
                values_arr = np.asarray(values)
                # Don't check for NaN in skew/kurtosis - they can legitimately be NaN for constant distributions
                if key not in ["skew", "kurtosis"]:
                    assert not np.any(np.isnan(values_arr)), f"{key} contains NaN"
                assert not np.any(np.isinf(values_arr)), f"{key} contains Inf"

    def test_source_index_structure(self):
        """Test that SourceIndex objects are properly structured."""
        images = [np.random.random((3, 50, 50))]
        boxes = [[[10, 10, 30, 30], [15, 15, 35, 35]]]

        stats = calculate(
            images,
            boxes,
            stats=ImageStats.DIMENSION,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        ratios = calculate_ratios(stats)

        # Check SourceIndex structure
        for si in ratios[SOURCE_INDEX]:
            assert isinstance(si, SourceIndex)
            assert isinstance(si.item, int)
            assert isinstance(si.target, int)
            assert si.target is not None  # All entries should be boxes
            assert si.channel is None or isinstance(si.channel, int)

    def test_hash_stats_ratios(self):
        """Test that hash statistics work with ratios (even though ratios don't make sense for hashes)."""
        images = [np.random.random((3, 50, 50))]
        boxes = [[[10, 10, 30, 30]]]

        stats = calculate(
            images,
            boxes,
            stats=ImageStats.HASH,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        # Should not raise error even though hash ratios don't make semantic sense
        ratios = calculate_ratios(stats)

        assert "xxhash" in ratios["stats"]
        assert "phash" in ratios["stats"]

    def test_all_stats_combined(self):
        """Test with all stat types combined."""
        images = [np.random.random((3, 50, 50))]
        boxes = [[[10, 10, 30, 30]]]

        stats = calculate(
            images,
            boxes,
            stats=ImageStats.ALL,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        ratios = calculate_ratios(stats)

        # Should have stats from all categories
        assert "width" in ratios["stats"]  # DIMENSION
        assert "mean" in ratios["stats"]  # PIXEL
        assert "brightness" in ratios["stats"]  # VISUAL
        assert "xxhash" in ratios["stats"]  # HASH

    def test_empty_override_map(self):
        """Test with explicitly empty override map."""
        images = [np.random.random((3, 50, 50))]
        boxes = [[[10, 10, 30, 30]]]

        stats = calculate(
            images,
            boxes,
            stats=ImageStats.DIMENSION,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        # Use empty override map - all stats should use default division
        ratios = calculate_ratios(stats, override_map={})

        # offset_x should now use default division instead of custom calculation
        # This is different from the default override behavior
        assert "offset_x" in ratios["stats"]
        assert len(ratios[SOURCE_INDEX]) == 1


class TestCalculateRatiosSeparateInputs:
    """Test calculate_ratios with separate image and box stats inputs (Pattern 2)."""

    def test_separate_inputs_basic(self):
        """Test basic usage with separate image and box stats."""
        images = [
            np.random.random((3, 100, 200)),  # Image 0: 100x200
        ]
        boxes = [
            [[10, 20, 50, 60]],  # Box of size 40x40
        ]

        # Calculate image and box stats separately
        img_stats = calculate(
            images,
            boxes,
            stats=ImageStats.DIMENSION,
            per_image=True,
            per_target=False,
        )

        tgt_stats = calculate(
            images,
            boxes,
            stats=ImageStats.DIMENSION,
            per_image=False,
            per_target=True,
        )

        # Calculate ratios using separate inputs
        ratios = calculate_ratios(img_stats, target_stats_output=tgt_stats)

        # Should have 1 box entry
        assert len(ratios[SOURCE_INDEX]) == 1

        # Check width ratio: 40 / 200 = 0.2
        assert ratios["stats"]["width"][0] == pytest.approx(40.0 / 200.0, abs=1e-3)

    def test_separate_inputs_multiple_images(self):
        """Test separate inputs with multiple images."""
        images = [
            np.random.random((3, 100, 100)),
            np.random.random((3, 200, 200)),
        ]
        boxes = [
            [[10, 10, 50, 50]],  # 1 box in image 0
            [[20, 20, 80, 80], [30, 30, 90, 90]],  # 2 boxes in image 1
        ]

        img_stats = calculate(images, boxes, stats=ImageStats.DIMENSION, per_image=True, per_target=False)
        tgt_stats = calculate(images, boxes, stats=ImageStats.DIMENSION, per_image=False, per_target=True)

        ratios = calculate_ratios(img_stats, target_stats_output=tgt_stats)

        # Should have 3 box entries total
        assert len(ratios[SOURCE_INDEX]) == 3

    def test_separate_inputs_per_channel(self):
        """Test separate inputs with per-channel stats."""
        images = [np.random.random((3, 50, 50))]
        boxes = [[[10, 10, 30, 30]]]

        img_stats = calculate(images, boxes, stats=ImageStats.PIXEL, per_image=True, per_target=False, per_channel=True)
        tgt_stats = calculate(images, boxes, stats=ImageStats.PIXEL, per_image=False, per_target=True, per_channel=True)

        ratios = calculate_ratios(img_stats, target_stats_output=tgt_stats)

        # Should have 3 entries (1 box * 3 channels)
        assert len(ratios[SOURCE_INDEX]) == 3

    def test_separate_inputs_image_count_mismatch(self):
        """Test that error is raised when image counts don't match."""
        images1 = [np.random.random((3, 50, 50))]
        images2 = [np.random.random((3, 50, 50)), np.random.random((3, 50, 50))]
        boxes = [[[10, 10, 30, 30]]]

        img_stats = calculate(images1, None, stats=ImageStats.DIMENSION, per_image=True, per_target=False)
        tgt_stats = calculate(
            images2,
            [boxes[0], boxes[0]],
            stats=ImageStats.DIMENSION,
            per_image=False,
            per_target=True,
        )

        with pytest.raises(ValueError, match="Image count mismatch"):
            calculate_ratios(img_stats, target_stats_output=tgt_stats)

    def test_separate_inputs_channel_mismatch(self):
        """Test that error is raised when per_channel settings don't match."""
        images = [np.random.random((3, 50, 50))]
        boxes = [[[10, 10, 30, 30]]]

        img_stats = calculate(images, boxes, stats=ImageStats.PIXEL, per_image=True, per_target=False, per_channel=True)
        tgt_stats = calculate(
            images,
            boxes,
            stats=ImageStats.PIXEL,
            per_image=False,
            per_target=True,
            per_channel=False,
        )

        with pytest.raises(ValueError, match="Channel mismatch"):
            calculate_ratios(img_stats, target_stats_output=tgt_stats)

    def test_separate_inputs_with_box_in_img_stats(self):
        """Test error when stats_output contains box entries."""
        images = [np.random.random((3, 50, 50))]
        boxes = [[[10, 10, 30, 30]]]

        # This has both image and box stats
        mixed_stats = calculate(images, boxes, stats=ImageStats.DIMENSION, per_image=True, per_target=True)
        tgt_stats = calculate(images, boxes, stats=ImageStats.DIMENSION, per_image=False, per_target=True)

        with pytest.raises(ValueError, match="should contain only image-level statistics"):
            calculate_ratios(mixed_stats, target_stats_output=tgt_stats)

    def test_separate_inputs_with_img_in_tgt_stats(self):
        """Test error when tgt_stats_output contains image entries."""
        images = [np.random.random((3, 50, 50))]
        boxes = [[[10, 10, 30, 30]]]

        img_stats = calculate(images, boxes, stats=ImageStats.DIMENSION, per_image=True, per_target=False)
        # This has both image and box stats
        mixed_stats = calculate(images, boxes, stats=ImageStats.DIMENSION, per_image=True, per_target=True)

        with pytest.raises(ValueError, match="should contain only box-level statistics"):
            calculate_ratios(img_stats, target_stats_output=mixed_stats)


class TestCalculateRatiosEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_pixel_box(self):
        """Test with a single-pixel bounding box."""
        images = [np.random.random((3, 50, 50))]
        boxes = [[[10, 10, 11, 11]]]  # 1x1 box

        stats = calculate(
            images,
            boxes,
            stats=ImageStats.DIMENSION,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        ratios = calculate_ratios(stats)

        assert len(ratios[SOURCE_INDEX]) == 1
        assert ratios["stats"]["width"][0] == pytest.approx(1.0 / 50.0, abs=1e-3)
        assert ratios["stats"]["height"][0] == pytest.approx(1.0 / 50.0, abs=1e-3)

    def test_box_equals_image(self):
        """Test when box covers entire image."""
        images = [np.random.random((3, 50, 50))]
        boxes = [[[0, 0, 50, 50]]]  # Full image box

        stats = calculate(
            images,
            boxes,
            stats=ImageStats.DIMENSION,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        ratios = calculate_ratios(stats)

        # Width and height ratios should be 1.0
        assert ratios["stats"]["width"][0] == pytest.approx(1.0, abs=1e-3)
        assert ratios["stats"]["height"][0] == pytest.approx(1.0, abs=1e-3)

    def test_single_channel_image(self):
        """Test with single-channel (grayscale) images."""
        images = [np.random.random((1, 50, 50))]
        boxes = [[[10, 10, 30, 30]]]

        stats = calculate(
            images,
            boxes,
            stats=ImageStats.DIMENSION,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        ratios = calculate_ratios(stats)

        assert ratios["stats"]["channels"][0] == 1
