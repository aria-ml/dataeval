import hashlib

import numpy as np
import pytest

from dataeval.utils.data.datasets._base import DataLocation
from dataeval.utils.data.datasets._cifar10 import CIFAR10
from dataeval.utils.data.datasets._milco import MILCO
from dataeval.utils.data.datasets._mnist import MNIST
from dataeval.utils.data.datasets._ships import Ships
from dataeval.utils.data.datasets._types import ObjectDetectionTarget
from dataeval.utils.data.datasets._voc import VOCDetection


def get_tmp_hash(fpath, chunk_size=65535):
    hasher = hashlib.md5()
    with open(fpath, "rb") as fpath_file:
        while chunk := fpath_file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


@pytest.mark.optional
class TestMNIST:
    def test_mnist_initialization(self, mnist_npy):
        """Test MNIST dataset initialization."""
        dataset = MNIST(root=str(mnist_npy))
        assert isinstance(dataset._targets, list)
        assert isinstance(dataset._targets[0], int)
        assert len(dataset) == 50000
        img, *_ = dataset[0]
        assert img.shape == (1, 28, 28)

    def test_mnist_wrong_initialization(self, wrong_mnist):
        """Test MNIST dataset initialization if asked for normal but downloaded corrupt."""
        dataset = MNIST(root=wrong_mnist, image_set="base")
        assert isinstance(dataset._targets, list)
        assert isinstance(dataset._targets[0], int)
        assert len(dataset) == 10000
        img, *_ = dataset[0]
        assert img.shape == (1, 28, 28)

    def test_mnist_test_data(self, mnist_npy):
        """Test loading the test set."""
        dataset = MNIST(root=mnist_npy, image_set="base")
        assert isinstance(dataset._targets, list)
        assert isinstance(dataset._targets[0], int)
        assert len(dataset) == 60000
        img, *_ = dataset[0]
        assert img.shape == (1, 28, 28)

    # @pytest.mark.parametrize(
    #     "classes, subset, expected",
    #     [
    #         (["zero", "one", "two", "five", "nine"], {0, 1, 2, 5, 9}, 200),
    #         ([4, 7, 8, 9, 15], {4, 7, 8, 9}, 250),
    #         ("six", {6}, 1000),
    #         (3, {3}, 1000),
    #         (np.array([3, 5]), {3, 5}, 500),
    #         (("2", "8"), {2, 8}, 500),
    #     ],
    # )
    # def test_mnist_class_selection(self, mnist_npy, classes, subset, expected):
    #     """Test class selection and equalize functionality."""
    #     dataset = MNIST(root=mnist_npy, size=1000, classes=classes, balance=True)
    #     label_array = np.array(dataset._annotations, dtype=np.uintp)
    #     labels = np.unique(label_array[dataset._indices])
    #     counts = Counter(label_array[dataset._indices])
    #     assert set(labels).issubset(subset)
    #     assert counts[labels[0]] == expected  # type: ignore

    def test_mnist_corruption(self, capsys, mnist_npy):
        """Test the loading of all the corruptions."""
        dataset = MNIST(root=mnist_npy, corruption="identity", verbose=False)
        assert isinstance(dataset._targets, list)
        assert isinstance(dataset._targets[0], int)
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
        dataset = Ships(root=str(ship_fake))
        assert isinstance(dataset._targets, list)
        assert isinstance(dataset._targets[0], int)
        assert len(dataset) == 4000
        img, *_ = dataset[0]
        assert img.shape == (3, 10, 10)
        assert dataset._datum_metadata != {}
        scene = dataset.get_scene(0)
        assert scene.shape == (3, 1500, 1250)


@pytest.mark.optional
class TestCIFAR10:
    def test_cifar10_initialization(self, cifar_fake):
        """Test CIFAR10 dataset initialization."""
        dataset = CIFAR10(root=cifar_fake, download=True)
        assert isinstance(dataset._targets, list)
        assert isinstance(dataset._targets[0], int)
        assert len(dataset) == 50000
        img, *_ = dataset[0]
        assert img.shape == (3, 32, 32)
        assert dataset._datum_metadata != {}

    def test_cifar10_base(self, cifar_fake):
        """Test CIFAR10 dataset with base"""
        dataset = CIFAR10(root=cifar_fake, image_set="base")
        assert isinstance(dataset._targets, list)
        assert isinstance(dataset._targets[0], int)
        assert len(dataset) == 60000
        img, *_ = dataset[0]
        assert img.shape == (3, 32, 32)
        assert dataset._datum_metadata != {}


@pytest.mark.optional
class TestVOC:
    def mock_resources(self, voc_fake):
        resources = [
            DataLocation(
                url="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
                filename="VOCtrainval_11-May-2012.tar",
                md5=True,
                checksum=get_tmp_hash(voc_fake / "VOCtrainval_11-May-2012.tar"),
            ),
        ]
        return resources

    def test_voc_detection(self, voc_fake, monkeypatch):
        """Test VOC detection dataset initialization"""
        monkeypatch.setattr(VOCDetection, "_resources", self.mock_resources(voc_fake))
        dataset = VOCDetection(root=voc_fake)
        img, target, datum_meta = dataset[0]
        assert img.shape == (3, 10, 10)
        assert np.all(target.labels == [11, 8])
        assert "pose" in datum_meta

        dataset = VOCDetection(root=voc_fake / "VOC2012", image_set="val")
        img, target, datum_meta = dataset[0]
        assert img.shape == (3, 10, 10)
        assert np.all(target.labels == [11, 8])
        assert "pose" in datum_meta


@pytest.mark.optional
class TestMILCO:
    def test_od_dataset(self, milco_fake):
        "Test to make sure the BaseODDataset has all the required parts"
        dataset = MILCO(root=milco_fake)
        if isinstance(dataset, MILCO):
            assert dataset._resources is not None
            assert dataset.index2label != {}
            assert dataset.label2index != {}
            assert "id" in dataset.metadata
            assert len(dataset) == 12
            for i in range(6):
                img, target, datum_meta = dataset[i]
                assert img.shape == (3, 10, 10)
                assert isinstance(target, ObjectDetectionTarget)
                assert "year" in datum_meta
