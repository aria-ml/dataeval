"""Fixtures for extractors tests."""

import sys
from pathlib import Path

# Add models conftest fixtures to this scope by directly importing
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
from conftest import onnx_classifier, onnx_detector  # noqa: F401  # type: ignore
