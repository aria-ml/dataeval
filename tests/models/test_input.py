"""Tests for the opinionated model input builder (IR-3.1-S-1 / S-4)."""

import numpy as np
import pytest

from dataeval.models import ModelIOSpec, build_model_input

RGB = ModelIOSpec("IMAGE_CLASSIFICATION", "RGB", 8, 8, -1, 3)
GRAY = ModelIOSpec("IMAGE_CLASSIFICATION", "GRAYSCALE", 8, 8, -1, 3)


def test_builds_nchw_float01_tensor():
    imgs = [np.full((3, 16, 16), 255, dtype=np.uint8)]
    out = build_model_input(imgs, RGB)
    assert out.shape == (1, 3, 8, 8)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert np.allclose(out, 1.0)


def test_user_override_wins_over_spec():
    imgs = [np.zeros((3, 16, 16), dtype=np.uint8)]
    out = build_model_input(imgs, RGB, height=4, width=5)
    assert out.shape == (1, 3, 4, 5)


def test_rgb_to_grayscale_channel_reduction():
    imgs = [np.zeros((3, 8, 8), dtype=np.uint8)]
    out = build_model_input(imgs, GRAY)
    assert out.shape == (1, 1, 8, 8)


def test_grayscale_to_rgb_expand():
    imgs = [np.zeros((1, 8, 8), dtype=np.uint8)]
    out = build_model_input(imgs, RGB)
    assert out.shape == (1, 3, 8, 8)


def test_variable_dim_without_override_raises():
    spec = ModelIOSpec("IMAGE_CLASSIFICATION", "RGB", -1, -1, -1, 3)
    with pytest.raises(ValueError, match="height"):
        build_model_input([np.zeros((3, 8, 8), dtype=np.uint8)], spec)


def test_float_image_already_normalized_passes_through():
    # Float inputs are assumed to be in [0, 1] and must NOT be divided by 255.
    out = build_model_input([np.full((3, 8, 8), 0.5, dtype=np.float32)], RGB)
    assert np.allclose(out, 0.5)


def test_float_image_overshooting_one_is_not_blacked_out():
    # A normalized float that slightly exceeds 1.0 must not be scaled by 1/255.
    out = build_model_input([np.full((3, 8, 8), 1.002, dtype=np.float32)], RGB)
    assert out.max() > 0.9
