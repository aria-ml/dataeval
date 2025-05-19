import types
from typing import NamedTuple
from unittest.mock import mock_open, patch

import numpy as np
import pytest

# Import the MILCO class
from dataeval.utils.datasets import MILCO

# Extract the _read_annotations method
read_annotations_func = MILCO._read_annotations


class Annotation(NamedTuple):
    """Helper class to create string annotations using different separators and a conversion function to xxyy."""

    class_id: int
    xcenter: float
    ycenter: float
    width: float
    height: float
    separator: list[str]

    def __str__(self) -> str:
        """Convert the annotation content to a string format."""
        separator = self.separator or [" "]
        output = ""
        for i, value in enumerate((self.class_id, self.xcenter, self.ycenter, self.width, self.height)):
            output += f"{value}" if i == 0 else f"{separator[(i - 1) % len(separator)]}{value}"
        return output

    def to_xxyy(self) -> tuple[float, float, float, float]:
        """Convert to x0, y0, x1, y1 format."""
        x0 = self.xcenter - self.width / 2
        x1 = x0 + self.width
        y0 = self.ycenter - self.height / 2
        y1 = y0 + self.height
        return x0, y0, x1, y1


class TestMILCOReadAnnotations:
    """Tests for MILCO._read_annotations method."""

    @pytest.mark.parametrize(
        "annotation",
        [
            Annotation(0, 0.5, 0.5, 0.2, 0.2, [" "]),  # Single annotation with normal spaces
            Annotation(1, 0.5, 0.5, 0.2, 0.2, ["\t"]),  # Single annotation with tabs
            Annotation(2, 0.5, 0.5, 0.2, 0.2, ["  "]),  # Single annotation with multiple spaces
            Annotation(3, 0.5, 0.5, 0.2, 0.2, [" ", "\t"]),  # Single annotation with mixed spaces and tabs
        ],
    )
    def test_read_annotations_single_line(self, annotation: Annotation):
        """Test if MILCO._read_annotations correctly handles various whitespace formats for a single line."""
        # Create a mock instance
        mock_self = types.SimpleNamespace()

        # Mock the open function
        with patch("builtins.open", mock_open(read_data=str(annotation))):
            # Call the method
            boxes, labels, _ = read_annotations_func(mock_self, "/dummy/path/to/annotation.txt")  # type: ignore

        # Get expected values
        expected_class_id = annotation.class_id
        expected_box = annotation.to_xxyy()

        # Check results
        assert len(boxes) == 1, "Expected 1 box"
        assert len(labels) == 1, "Expected 1 label"
        assert labels[0] == expected_class_id, f"Expected label {expected_class_id}, got {labels[0]}"
        assert np.allclose(boxes[0], expected_box, rtol=1e-5), f"Box mismatch: {boxes[0]} vs {expected_box}"

    def test_read_annotations_multiple_lines(self):
        """Test if MILCO._read_annotations can handle various separation characters on multiple lines."""
        annotations = [
            Annotation(0, 0.5, 0.5, 0.2, 0.2, [" "]),
            Annotation(1, 0.8, 0.8, 0.3, 0.3, ["\t"]),
            Annotation(2, 0.2, 0.2, 0.1, 0.1, ["  "]),
            Annotation(3, 0.5, 0.5, 0.2, 0.2, [" ", "\t"]),  # Single annotation with mixed spaces and tabs
        ]

        # Create a mock instance
        mock_self = types.SimpleNamespace()

        # Mock the open function
        with patch("builtins.open", mock_open(read_data="\n".join(str(annotation) for annotation in annotations))):
            # Call the method
            boxes, labels, _ = read_annotations_func(mock_self, "/dummy/path/to/annotation.txt")  # type: ignore

        # Check results
        assert len(boxes) == len(annotations), f"Expected {len(annotations)} boxes, got {len(boxes)}"
        assert len(labels) == len(annotations), f"Expected {len(annotations)} labels, got {len(labels)}"

        for i, (box, annotation) in enumerate(zip(boxes, annotations)):
            assert np.allclose(box, annotation.to_xxyy(), rtol=1e-5), (
                f"Box {i} mismatch: {box} vs {annotation.to_xxyy()}"
            )

        for i, (label, annotation) in enumerate(zip(labels, annotations)):
            assert label == annotation.class_id, f"Label {i} mismatch: {label} vs {annotation.class_id}"

    @pytest.mark.parametrize(
        "invalid_content",
        [
            "0 0.5 0.5\n",  # Missing values
            "class x y w h\n",  # Non-numeric values
        ],
    )
    def test_read_annotations_with_invalid_formats_that_raise(self, invalid_content):
        """Test if MILCO._read_annotations raises exceptions for invalid formats."""
        # Create a mock instance
        mock_self = types.SimpleNamespace()

        # Mock the open function and expect an exception
        with patch("builtins.open", mock_open(read_data=invalid_content)), pytest.raises((ValueError, IndexError)):
            read_annotations_func(mock_self, "/dummy/path/to/annotation.txt")  # type: ignore

    def test_read_annotations_with_empty_file(self):
        """Test if MILCO._read_annotations handles empty files correctly."""
        # Create a mock instance
        mock_self = types.SimpleNamespace()

        # Mock the open function
        with patch("builtins.open", mock_open(read_data="")):
            boxes, labels, _ = read_annotations_func(mock_self, "/dummy/path/to/annotation.txt")  # type: ignore

        assert len(boxes) == 0, "Expected empty boxes list"
        assert len(labels) == 0, "Expected empty labels list"
