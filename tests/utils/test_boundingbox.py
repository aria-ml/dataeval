from unittest.mock import patch

import pytest

from dataeval.utils._boundingbox import (
    BoundingBox,
    BoundingBoxFormat,
    clip_box,
    is_valid_box,
    to_bounding_box,
    to_int_box,
)


@pytest.mark.required
class TestBoundingBoxFormat:
    def test_enum_values(self):
        assert BoundingBoxFormat.XYXY.value == "xyxy"
        assert BoundingBoxFormat.XYWH.value == "xywh"
        assert BoundingBoxFormat.CXCYWH.value == "cxcywh"
        assert BoundingBoxFormat.YOLO.value == "yolo"


@pytest.mark.required
class TestBoundingBox:
    def test_init_xyxy_valid(self):
        bbox = BoundingBox(10, 20, 30, 40)
        assert bbox.xyxy == (10, 20, 30, 40)

    def test_init_xyxy_invalid_coordinates_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            bbox = BoundingBox(30, 40, 10, 20)
        assert len([rec for rec in caplog.records if "Invalid bounding box coordinates" in rec.message]) == 1
        assert bbox.xyxy == (10, 20, 30, 40)  # Should swap coordinates

    def test_init_xywh(self):
        bbox = BoundingBox(10, 20, 15, 25, bbox_format=BoundingBoxFormat.XYWH)
        assert bbox.xyxy == (10, 20, 25, 45)

    def test_init_cxcywh(self):
        bbox = BoundingBox(20, 30, 10, 20, bbox_format=BoundingBoxFormat.CXCYWH)
        assert bbox.xyxy == (15, 20, 25, 40)

    def test_init_yolo(self):
        bbox = BoundingBox(0.5, 0.6, 0.2, 0.4, bbox_format=BoundingBoxFormat.YOLO, image_shape=(3, 100, 200))
        # center_x=100, center_y=60, w=40, h=40
        assert bbox.xyxy == (80, 40, 120, 80)

    def test_init_unknown_format(self):
        with pytest.raises(ValueError, match="Unknown format"):
            BoundingBox(10, 20, 30, 40, bbox_format="unknown")  # type: ignore

    def test_properties_coordinates(self):
        bbox = BoundingBox(10.5, 20.5, 30.5, 40.5)
        assert bbox.x0 == 10.5
        assert bbox.y0 == 20.5
        assert bbox.x1 == 30.5
        assert bbox.y1 == 40.5

    def test_xyxy_int(self):
        bbox = BoundingBox(10.7, 20.3, 30.2, 40.8)
        assert bbox.xyxy_int == (10, 20, 31, 41)  # floor, floor, ceil, ceil

    def test_xywh_property(self):
        bbox = BoundingBox(10, 20, 30, 40)
        assert bbox.xywh == (10, 20, 20, 20)

    def test_cxcywh_property(self):
        bbox = BoundingBox(10, 20, 30, 40)
        assert bbox.cxcywh == (20, 30, 20, 20)

    def test_yolo_property(self):
        bbox = BoundingBox(10, 20, 30, 40, image_shape=(3, 100, 200))
        expected = (20 / 200, 30 / 100, 20 / 200, 20 / 100)  # normalized
        assert bbox.yolo == expected

    def test_width_height(self):
        bbox = BoundingBox(10, 20, 30, 40)
        assert bbox.width == 20
        assert bbox.height == 20

    def test_image_hw_with_shape(self):
        bbox = BoundingBox(10, 20, 30, 40, image_shape=(3, 100, 200))
        assert bbox.image_hw == (100, 200)

    def test_image_hw_without_shape(self):
        bbox = BoundingBox(10, 20, 30, 40)
        with pytest.raises(ValueError, match="Image shape is required"):
            bbox.image_hw

    def test_area(self):
        bbox = BoundingBox(10, 20, 30, 40)
        assert bbox.area() == 400

    def test_center(self):
        bbox = BoundingBox(10, 20, 30, 40)
        assert bbox.center() == (20, 30)

    def test_is_inside_true(self):
        bbox = BoundingBox(10, 20, 30, 40, image_shape=(3, 100, 200))
        assert bbox.is_inside() is True

    def test_is_inside_false(self):
        bbox = BoundingBox(-5, 20, 30, 40, image_shape=(3, 100, 200))
        assert bbox.is_inside() is False

    def test_is_outside_true(self):
        bbox = BoundingBox(250, 20, 300, 40, image_shape=(3, 100, 200))
        assert bbox.is_outside() is True

    def test_is_outside_false(self):
        bbox = BoundingBox(10, 20, 30, 40, image_shape=(3, 100, 200))
        assert bbox.is_outside() is False

    def test_is_partial_true(self):
        bbox = BoundingBox(-5, 20, 30, 40, image_shape=(3, 100, 200))
        assert bbox.is_partial() is True

    def test_is_partial_false_inside(self):
        bbox = BoundingBox(10, 20, 30, 40, image_shape=(3, 100, 200))
        assert bbox.is_partial() is False

    def test_is_partial_false_outside(self):
        bbox = BoundingBox(250, 20, 300, 40, image_shape=(3, 100, 200))
        assert bbox.is_partial() is False

    def test_is_valid_true(self):
        bbox = BoundingBox(10, 20, 30, 40, image_shape=(3, 100, 200))
        assert bbox.is_valid() is True

    def test_is_valid_false_zero_width(self):
        bbox = BoundingBox(10, 20, 10, 40, image_shape=(3, 100, 200))
        assert bbox.is_valid() is False

    def test_is_valid_false_outside(self):
        bbox = BoundingBox(250, 20, 300, 40, image_shape=(3, 100, 200))
        assert bbox.is_valid() is False

    @patch("dataeval.utils._boundingbox.is_valid_box")
    @patch("dataeval.utils._boundingbox.clip_box")
    def test_is_clippable_true(self, mock_clip, mock_valid):
        bbox = BoundingBox(10, 20, 30, 40, image_shape=(3, 100, 200))
        mock_clip.return_value = (10, 20, 30, 40)
        mock_valid.return_value = True
        assert bbox.is_clippable() is True
        mock_clip.assert_called_once_with((100, 200), (10, 20, 30, 40))
        mock_valid.assert_called_once_with((10, 20, 30, 40))

    @patch("dataeval.utils._boundingbox.is_valid_box")
    @patch("dataeval.utils._boundingbox.clip_box")
    def test_is_clippable_false(self, mock_clip, mock_valid):
        bbox = BoundingBox(10, 20, 30, 40, image_shape=(3, 100, 200))
        mock_clip.return_value = (10, 20, 10, 40)
        mock_valid.return_value = False
        assert bbox.is_clippable() is False

    def test_to_bounding_box_bounding_box(self):
        original = BoundingBox(10, 20, 30, 40)
        result = to_bounding_box(original)
        assert result is original

    def test_to_bounding_box_tuple(self):
        result = to_bounding_box((10, 20, 30, 40))
        assert result.xyxy == (10, 20, 30, 40)

    def test_to_bounding_box_list(self):
        result = to_bounding_box([10, 20, 30, 40])
        assert result.xyxy == (10, 20, 30, 40)

    def test_to_bounding_box_iterable(self):
        result = to_bounding_box(iter([10, 20, 30, 40]))
        assert result.xyxy == (10, 20, 30, 40)

    def test_to_bounding_box_image_shape_fallback(self):
        result = to_bounding_box(None, image_shape=(3, 100, 200))
        assert result.xyxy == (0, 0, 100, 200)

    def test_to_bounding_box_invalid_with_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            result = to_bounding_box("invalid")  # type: ignore
        assert len([rec for rec in caplog.records if "Invalid bounding box format" in rec.message]) == 1
        assert result.xyxy == (0, 0, 0, 0)


@pytest.mark.required
class TestBoundingBoxUtilityFunctions:
    def test_to_int_box(self):
        result = to_int_box((10.7, 20.3, 30.2, 40.8))
        assert result == (10, 20, 31, 41)

    def test_clip_box_no_clipping_needed(self):
        result = clip_box((100, 200), (10, 20, 30, 40))
        assert result == (10, 20, 30, 40)

    def test_clip_box_all_sides_clipped(self):
        result = clip_box((100, 200), (-5, -10, 250, 150))
        assert result == (0, 0, 200, 100)

    def test_is_valid_box_true(self):
        assert is_valid_box((10, 20, 30, 40)) is True

    def test_is_valid_box_false_zero_width(self):
        assert is_valid_box((10, 20, 10, 40)) is False

    def test_is_valid_box_false_zero_height(self):
        assert is_valid_box((10, 20, 30, 20)) is False

    def test_is_valid_box_false_negative_dimensions(self):
        assert is_valid_box((30, 40, 10, 20)) is False
