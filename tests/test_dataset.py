import copy
from typing import List, Tuple

import numpy as np
import numpy.testing as npt
import pytest

from daml._internal.datasets import DamlDataset


class TestDamlImageDataset:
    """Tests for a dataset containing only images"""

    shape = (10, 28, 28, 1)  # N, H, W, C
    images = np.ones(shape=shape)
    dds = DamlDataset(images)

    def test_init(self):
        assert self.dds._images is not None
        npt.assert_array_equal(self.dds._images, self.images)

        assert self.dds._labels is None
        assert self.dds._boxes is None

    def test_len(self):
        assert len(self.dds) == len(self.images)

    def test_getitem(self):
        output = self.dds[0]
        image_0 = self.images[0]

        assert isinstance(output, np.ndarray)

        npt.assert_array_equal(output, image_0)

    def test_property(self):
        npt.assert_array_equal(self.dds.images, self.images)  # Images
        npt.assert_array_equal(self.dds.labels, np.array([]))  # Labels
        npt.assert_array_equal(self.dds.boxes, np.array([]))  # Boxes


class TestDamlClassificationDataset:
    """Tests for a dataset containing only images and labels"""

    shape = (10, 28, 28, 1)
    images = np.ones(shape=shape)
    labels = np.ones(shape[0])  # [num labels]
    dds = DamlDataset(images, labels)

    def test_init(self):
        assert self.dds._images is not None
        npt.assert_array_equal(self.dds._images, self.images)

        assert self.dds._labels is not None
        npt.assert_array_equal(self.dds._labels, self.labels)

        assert self.dds._boxes is None

    def test_len(self):
        assert len(self.dds) == len(self.images)
        assert len(self.dds) == len(self.images)

    def test_getitem(self):
        output = self.dds[0]
        image_0 = self.images[0]
        label_0 = self.labels[0]

        assert isinstance(output, Tuple)
        assert len(output) == 2

        assert isinstance(output[0], np.ndarray)
        assert isinstance(output[1], float)

        npt.assert_array_equal(output[0], image_0)
        npt.assert_array_equal(output[1], label_0)

    def test_setter(self):
        dds2 = copy.deepcopy(self.dds)
        dds2.images = np.zeros(shape=self.shape)
        dds2.labels = np.zeros(self.shape[0])
        dds2.boxes = np.zeros(self.shape[0])

        assert not np.array_equal(dds2.images, self.dds.images)
        assert not np.array_equal(dds2.labels, self.dds.labels)
        assert dds2.boxes is not None

    def test_property(self):
        npt.assert_array_equal(self.dds.images, self.images)  # Images
        npt.assert_array_equal(self.dds.labels, self.labels)  # Labels
        npt.assert_array_equal(self.dds.boxes, np.array([]))  # Boxes


class TestDamlObjectDetectionDataset:
    """Tests for a dataset containing images, labels, and bounding boxes"""

    shape = (10, 28, 28, 1)
    images = np.ones(shape=shape)
    labels = np.ones(shape=(shape[0], 1))  # [images, labels per image]
    boxes = np.ones(shape=(shape[0], 1, 4))  # [images, boxes per image, box coords]
    dds = DamlDataset(images, labels, boxes)

    def test_init(self):
        assert self.dds._images is not None
        npt.assert_array_equal(self.dds._images, self.images)

        assert self.dds._labels is not None
        npt.assert_array_equal(self.dds._labels, self.labels)

        assert self.dds._boxes is not None
        npt.assert_array_equal(self.dds._boxes, self.boxes)

    def test_len(self):
        assert len(self.dds) == len(self.images)
        assert len(self.dds) == len(self.images)
        assert len(self.dds) == len(self.boxes)

    def test_getitem(self):
        output = self.dds[0]
        image_0 = self.images[0]
        label_0 = self.labels[0]
        box_0 = self.boxes[0]

        assert isinstance(output, Tuple)
        assert len(output) == 3

        assert isinstance(output[0], np.ndarray)
        # Each image contains a list of labels and boxes
        assert isinstance(output[1], (np.ndarray, List))
        assert isinstance(output[2], (np.ndarray, List))

        # Each label in the list of labels is a float
        assert isinstance(output[1][0], float)
        # Each bounding box coords in the list of boxes is a list of (x, y, w, h)
        assert isinstance(output[2][0], (np.ndarray, List))
        assert len(output[2][0]) == 4
        # Each bounding box coord (x|y|w|h) in coords is a float
        assert isinstance(output[2][0][0], float)

        # Confirm the inputs are not changed (value, order, shape, etc)
        npt.assert_array_equal(output[0], image_0)
        npt.assert_array_equal(output[1], label_0)
        npt.assert_array_equal(output[2], box_0)

    def test_setitem(self):
        """DamlDataset properties can be modified correctly"""
        dds = DamlDataset(np.array([]), np.array([]), np.array([]))

        # Confirm they are initialized correctly
        assert not len(dds.images)
        assert not len(dds.labels)
        assert not len(dds.boxes)

        # Set them to new values
        dds.images = self.images
        dds.labels = self.labels
        dds.boxes = self.boxes

        # Confirm they were set
        npt.assert_array_equal(dds.images, self.images)
        npt.assert_array_equal(dds.images, self.images)
        npt.assert_array_equal(dds.images, self.images)

    def test_property(self):
        """DamlDataset properly sets values during init"""
        npt.assert_array_equal(self.dds.images, self.images)  # Images
        npt.assert_array_equal(self.dds.labels, self.labels)  # Labels
        npt.assert_array_equal(self.dds.boxes, self.boxes)  # Boxes


@pytest.mark.skip(reason="Future improvements")
class TestDamlInvalidDataset:
    """
    Tests for incorrect types and
    non-CV dataset inputs (time series, text, etc)
    """

    def test_inputs(self):
        pass


@pytest.mark.skip(reason="Future functionality")
class TestConvertToDamlDataset:
    """Tests for compatibility for DAML dataset from popular data backends"""

    def test_pytorch(self):
        pytest.importorskip("torch")

    def test_tensorflow(self):
        pytest.importorskip("tensorflow")

    def test_numpy(self):
        pytest.importorskip("numpy")

    def test_pandas(self):
        pytest.importorskip("pandas")

    def test_files(self):
        pass
