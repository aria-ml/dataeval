"""Tests for the shared CHW resize helper."""

import numpy as np
import pytest

from dataeval.extractors._resize import resize_chw


def test_resize_chw_changes_spatial_dims_only():
    img = np.random.rand(3, 16, 20).astype(np.float32)
    out = resize_chw(img, (8, 10))
    assert out.shape == (3, 8, 10)


def test_resize_chw_rejects_non_chw():
    with pytest.raises(ValueError, match="CHW"):
        resize_chw(np.zeros((8, 10), dtype=np.float32), (4, 5))
