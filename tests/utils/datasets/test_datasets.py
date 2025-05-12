import hashlib

import numpy as np
import pytest

from dataeval.utils.datasets._antiuav import AntiUAVDetection
from dataeval.utils.datasets._base import DataLocation
from dataeval.utils.datasets._cifar10 import CIFAR10
from dataeval.utils.datasets._milco import MILCO
from dataeval.utils.datasets._mnist import MNIST
from dataeval.utils.datasets._ships import Ships
from dataeval.utils.datasets._types import ObjectDetectionTarget
from dataeval.utils.datasets._voc import VOCDetection


def get_tmp_hash(fpath, chunk_size=65535):
    hasher = hashlib.sha256()
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
        _ = CIFAR10(root=cifar_fake)
        dataset = CIFAR10(root=cifar_fake)
        assert isinstance(dataset._targets, list)
        assert isinstance(dataset._targets[0], int)
        assert len(dataset) == 50000
        img, *_ = dataset[0]
        assert img.shape == (3, 32, 32)
        assert dataset._datum_metadata != {}

    def test_cifar10_base(self, cifar_fake):
        """Test CIFAR10 dataset with base"""
        _ = CIFAR10(root=cifar_fake, image_set="base")
        dataset = CIFAR10(root=cifar_fake, image_set="base")
        assert isinstance(dataset._targets, list)
        assert isinstance(dataset._targets[0], int)
        assert len(dataset) == 60000
        img, *_ = dataset[0]
        assert img.shape == (3, 32, 32)
        assert dataset._datum_metadata != {}

    def test_cifar10_test(self, cifar_fake):
        """Test CIFAR10 dataset with base"""
        _ = CIFAR10(root=cifar_fake, image_set="test")
        dataset = CIFAR10(root=cifar_fake, image_set="test")
        assert isinstance(dataset._targets, list)
        assert isinstance(dataset._targets[0], int)
        assert len(dataset) == 10000
        img, *_ = dataset[0]
        assert img.shape == (3, 32, 32)
        assert dataset._datum_metadata != {}


@pytest.mark.optional
class TestVOC:
    def mock_resources(self, base, year=2012, test=False):
        tmp_checksum = get_tmp_hash(base / f"VOCtrainval-{year}.tar")
        resources = [
            DataLocation(
                url="https://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar",
                filename="VOCtrainval-2012.tar",
                md5=False,
                checksum=tmp_checksum if year == 2012 else "abcdefg",
            ),
            DataLocation(
                url="http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar",
                filename="VOCtrainval-2011.tar",
                md5=False,
                checksum=tmp_checksum if year == 2011 else "abcdefg",
            ),
            DataLocation(
                url="http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar",
                filename="VOCtrainval-2010.tar",
                md5=False,
                checksum=tmp_checksum if year == 2010 else "abcdefg",
            ),
            DataLocation(
                url="http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar",
                filename="VOCtrainval-2009.tar",
                md5=False,
                checksum=tmp_checksum if year == 2009 else "abcdefg",
            ),
            DataLocation(
                url="http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar",
                filename="VOCtrainval-2008.tar",
                md5=False,
                checksum="7f0ca53c1b5a838fbe946965fc106c6e86832183240af5c88e3f6c306318d42e",
            ),
            DataLocation(
                url="https://data.brainchip.com/dataset-mirror/voc/VOCtrainval_06-Nov-2007.tar",
                filename="VOCtrainval-2007.tar",
                md5=False,
                checksum="7d8cd951101b0957ddfd7a530bdc8a94f06121cfc1e511bb5937e973020c7508",
            ),
            DataLocation(
                url="https://data.brainchip.com/dataset-mirror/voc/VOC2012test.tar",
                filename="VOC2012test.tar",
                md5=False,
                checksum=get_tmp_hash(base / "VOC2012test.tar") if test else "abcdefg",
            ),
            DataLocation(
                url="https://data.brainchip.com/dataset-mirror/voc/VOCtest_06-Nov-2007.tar",
                filename="VOCtest_06-Nov-2007.tar",
                md5=False,
                checksum="6836888e2e01dca84577a849d339fa4f73e1e4f135d312430c4856b5609b4892",
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

        dir_path = voc_fake / "vocdataset" / "VOCdevkit" / "VOC2012"
        dataset = VOCDetection(root=dir_path, image_set="val")
        img, target, datum_meta = dataset[0]
        assert img.shape == (3, 10, 10)
        assert np.all(target.labels == [11, 8])
        assert "pose" in datum_meta

        dir_path = voc_fake / "vocdataset" / "VOCdevkit"
        dataset = VOCDetection(root=dir_path, verbose=True)
        img, target, datum_meta = dataset[0]
        assert img.shape == (3, 10, 10)
        assert np.all(target.labels == [11, 8])
        assert "pose" in datum_meta

    @pytest.mark.year(2011)
    def test_voc_2011(self, voc_fake, monkeypatch):
        monkeypatch.setattr(VOCDetection, "_resources", self.mock_resources(voc_fake, year=2011))
        dataset = VOCDetection(root=voc_fake, year="2011")
        img, target, datum_meta = dataset[0]
        assert img.shape == (3, 10, 10)
        assert np.all(target.labels == [11, 8])
        assert "pose" in datum_meta

    @pytest.mark.parametrize("first, second", [("train", "base"), ("test", "base")])
    def test_voc_half_downloaded_base(self, voc_fake_test, monkeypatch, first, second):
        monkeypatch.setattr(VOCDetection, "_resources", self.mock_resources(voc_fake_test, test=True))
        _ = VOCDetection(root=voc_fake_test, image_set=first)
        dataset = VOCDetection(root=voc_fake_test, image_set=second)
        img, target, datum_meta = dataset[0]
        assert img.shape == (3, 10, 10)
        assert np.all(target.labels == [11, 8])
        assert "pose" in datum_meta

    def test_voc_half_downloaded_test(self, voc_fake_test, monkeypatch):
        monkeypatch.setattr(VOCDetection, "_resources", self.mock_resources(voc_fake_test, test=True))
        _ = VOCDetection(root=voc_fake_test, image_set="train")
        dataset = VOCDetection(root=voc_fake_test, image_set="test")
        img, target, datum_meta = dataset[0]
        assert img.shape == (3, 10, 10)
        assert np.all(target.labels == [11, 8])
        assert "pose" in datum_meta

    def test_voc_path_errors(self, voc_fake):
        dir_path = voc_fake / "vocdataset" / "VOCdevkit" / "VOC2012"
        with pytest.raises(NotADirectoryError, match="Directory VOC2012 specified but doesn't exist."):
            VOCDetection(root=dir_path)
        dir_path = voc_fake / "vocdataset" / "VOCdevkit"
        with pytest.raises(NotADirectoryError, match="Directory VOCdevkit/VOC2012 subdirectory doesn't exist."):
            VOCDetection(root=dir_path)


@pytest.mark.optional
class TestMILCO:
    def test_milco_dataset(self, milco_fake):
        "Test MILCO dataset initialization"
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


@pytest.mark.optional
class TestAntiUAVDetection:
    def test_antiuav_dataset(self, antiuav_fake):
        "Test AntiUAVDetection dataset initialization"
        dataset = AntiUAVDetection(root=antiuav_fake)
        if isinstance(dataset, AntiUAVDetection):
            assert dataset._resources is not None
            assert dataset.index2label != {}
            assert dataset.label2index != {}
            assert "id" in dataset.metadata
            assert len(dataset) == 12
            for i in range(12):
                img, target, datum_meta = dataset[i]
                assert img.shape == (3, 10, 10)
                assert isinstance(target, ObjectDetectionTarget)
