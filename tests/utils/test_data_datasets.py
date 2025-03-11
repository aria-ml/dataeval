from collections import Counter

import numpy as np
import pytest

from dataeval.utils.data.datasets._ic import MNIST, ShipDataset


@pytest.mark.optional
class TestMNIST:
    def test_mnist_initialization(self, mnist_npy):
        """Test MNIST dataset initialization."""
        dataset = MNIST(root=str(mnist_npy), size=-1, balance=False)
        assert isinstance(dataset._labels, list)
        assert isinstance(dataset._labels[0], int)
        assert len(dataset) == 50000
        img, *_ = dataset[0]
        assert img.shape == (1, 28, 28)

    def test_mnist_wrong_initialization(self, wrong_mnist):
        """Test MNIST dataset initialization if asked for normal but downloaded corrupt."""
        dataset = MNIST(root=wrong_mnist, image_set="base", size=-1, balance=False)
        assert isinstance(dataset._labels, list)
        assert isinstance(dataset._labels[0], int)
        assert len(dataset) == 10000
        img, *_ = dataset[0]
        assert img.shape == (1, 28, 28)

    def test_mnist_test_data(self, mnist_npy):
        """Test loading the test set."""
        dataset = MNIST(root=mnist_npy, image_set="base", size=-1, balance=False)
        assert isinstance(dataset._labels, list)
        assert isinstance(dataset._labels[0], int)
        assert len(dataset) == 60000
        img, *_ = dataset[0]
        assert img.shape == (1, 28, 28)

    @pytest.mark.parametrize(
        "classes, subset, expected",
        [
            (["zero", "one", "two", "five", "nine"], {0, 1, 2, 5, 9}, 200),
            ([4, 7, 8, 9, 15], {4, 7, 8, 9}, 250),
            ("six", {6}, 1000),
            (3, {3}, 1000),
            (np.array([3, 5]), {3, 5}, 500),
            (("2", "8"), {2, 8}, 500),
        ],
    )
    def test_mnist_class_selection(self, mnist_npy, classes, subset, expected):
        """Test class selection and equalize functionality."""
        dataset = MNIST(root=mnist_npy, size=1000, classes=classes, balance=True)
        label_array = np.array(dataset._labels, dtype=np.uintp)
        labels = np.unique(label_array[dataset._reorder])
        counts = Counter(label_array[dataset._reorder])
        assert set(labels).issubset(subset)
        assert counts[labels[0]] == expected  # type: ignore

    def test_mnist_corruption(self, capsys, mnist_npy):
        """Test the loading of all the corruptions."""
        dataset = MNIST(root=mnist_npy, size=-1, balance=False, corruption="identity")
        assert isinstance(dataset._labels, list)
        assert isinstance(dataset._labels[0], int)
        assert len(dataset) == 50000
        img, *_ = dataset[0]
        assert img.shape == (1, 28, 28)
        dataset = MNIST(root=mnist_npy, corruption="identity", verbose=True)
        captured = capsys.readouterr()
        msg = "Identity is not a corrupted dataset but the original MNIST dataset."
        assert msg in captured.out


@pytest.mark.optional
class TestShipDataset:
    def test_ship_initialization(self, ship_fake):
        """Test Ship dataset initialization."""
        dataset = ShipDataset(root=str(ship_fake))
        assert isinstance(dataset._labels, list)
        assert isinstance(dataset._labels[0], int)
        assert len(dataset) == 4000
        img, *_ = dataset[0]
        assert img.shape == (3, 10, 10)
        scene = dataset.get_scene(0)
        assert scene.shape == (3, 1500, 1250)
        assert dataset._datum_metadata
