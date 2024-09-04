from unittest.mock import MagicMock

import numpy as np
import pytest

from dataeval._internal.metrics.utils import (
    compute_neighbors,
    edge_filter,
    flatten,
    get_bitdepth,
    get_classes_counts,
    minimum_spanning_tree,
    normalize_image_shape,
    rescale,
)


def test_class_min():
    with pytest.raises(ValueError):
        get_classes_counts(np.ones(1))


def test_get_bitdepth_negatives():
    image = np.random.random((3, 28, 28)) - 0.5
    bitdepth = get_bitdepth(image)
    assert bitdepth == (0, np.min(image), np.max(image))


def test_get_bitdepth_float():
    image = np.random.random((3, 28, 28))
    bitdepth = get_bitdepth(image)
    assert bitdepth == (1, 0, 1)


def test_get_bitdepth_8bit():
    image = (np.random.random((3, 28, 28)) * (2**8 - 1)).astype(np.uint8)
    bitdepth = get_bitdepth(image)
    assert bitdepth == (8, 0, (2**8 - 1))


def test_get_bitdepth_16bit():
    image = (np.random.random((3, 28, 28)) * (2**16 - 1)).astype(np.uint16)
    bitdepth = get_bitdepth(image)
    assert bitdepth == (16, 0, (2**16 - 1))


def test_get_bitdepth_64bit():
    image = (np.random.random((3, 28, 28)) * (2**64 - 1)).astype(np.uint64)
    bitdepth = get_bitdepth(image)
    assert bitdepth == (32, 0, (2**32 - 1))


def test_rescale_noop():
    image = np.random.random((3, 28, 28))
    scaled = rescale(image)
    assert np.min(scaled) == np.min(image)
    assert np.max(scaled) == np.max(image)


def test_rescale_8bit():
    image = np.random.random((3, 28, 28))
    scaled = rescale(image, 8)
    assert np.min(scaled) == np.min(image) * (2**8 - 1)
    assert np.max(scaled) == np.max(image) * (2**8 - 1)


def test_rescale_float():
    image = (np.random.random((3, 28, 28)) * 255).astype(np.uint8)
    scaled = rescale(image)
    assert np.min(scaled) == np.min(image) / (2**8 - 1)
    assert np.max(scaled) == np.max(image) / (2**8 - 1)


def test_normalize_image_shape_expand():
    image = np.zeros((28, 28))
    normalized = normalize_image_shape(image)
    assert normalized.shape == (1, 28, 28)


def test_normalize_image_shape_noop():
    image = np.zeros((1, 28, 28))
    normalized = normalize_image_shape(image)
    assert normalized.shape == (1, 28, 28)


def test_normalize_image_shape_slice():
    image = np.zeros((10, 3, 28, 28))
    normalized = normalize_image_shape(image)
    assert normalized.shape == (3, 28, 28)


def test_normalize_image_valueerror():
    image = np.zeros(10)
    with pytest.raises(ValueError):
        normalize_image_shape(image)


def test_edge_filter():
    image = np.zeros((28, 28))
    edge = edge_filter(image, 0.5)
    np.testing.assert_array_equal(image + 0.5, edge)


def test_flatten_noop():
    images = np.ones(shape=(10, 3))
    assert flatten(images).shape == (10, 3)


def test_flatten_0dim():
    images = np.ones(shape=(10))
    assert flatten(images).shape == (10, 1)


def test_flatten_1dim():
    images = np.ones(shape=(10, 1))
    assert flatten(images).shape == (10, 1)


def test_flatten_3dim():
    images = np.ones(shape=(10, 3, 3))
    assert flatten(images).shape == (10, 9)


def test_mst():
    images = np.ones((10, 3, 3))
    assert minimum_spanning_tree(images).shape == (10, 10)


def test_compute_neighbors():
    images_0 = np.zeros((10, 3, 3))
    images_1 = np.ones((10, 3, 3))

    assert compute_neighbors(images_0, images_1).shape == (10,)


def test_compute_neighbors_k0():
    images_0 = np.zeros((10, 3, 3))
    images_1 = np.ones((10, 3, 3))

    with pytest.raises(ValueError):
        compute_neighbors(images_0, images_1, k=0).shape


def test_compute_neighbors_k2():
    images_0 = np.zeros((10, 3, 3))
    images_1 = np.ones((10, 3, 3))

    assert compute_neighbors(images_0, images_1, k=2).shape == (10, 2)


def test_compute_neighbors_kdtree():
    images_0 = np.zeros((10, 3, 3))
    images_1 = np.ones((10, 3, 3))
    assert compute_neighbors(images_0, images_1, algorithm="kd_tree").shape == (10,)


def test_compute_neighbors_balltree():
    images_0 = np.zeros((10, 3, 3))
    images_1 = np.ones((10, 3, 3))
    assert compute_neighbors(images_0, images_1, algorithm="ball_tree").shape == (10,)


def test_compute_neighbors_invalid_alg():
    """Brute algorithm is valid for sklearn.NearestNeighbor, but is invalid for DataEval"""

    with pytest.raises(ValueError):
        compute_neighbors(MagicMock(), MagicMock(), algorithm="brute")  # type: ignore #
