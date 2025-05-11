import hashlib

import pytest

from dataeval.utils.datasets._base import DataLocation
from dataeval.utils.datasets._mnist import MNIST
from dataeval.utils.datasets._types import SegmentationTarget
from dataeval.utils.datasets._voc import VOCSegmentation

TEMP_MD5 = "d149274109b50d5147c09d6fc7e80c71"
TEMP_SHA256 = "2b749913055289cb3a5c602a17196b5437dc59bba50e986ea449012a303f7201"


def get_tmp_hash(fpath, chunk_size=65535):
    hasher = hashlib.sha256()
    with open(fpath, "rb") as fpath_file:
        while chunk := fpath_file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


@pytest.mark.optional
class TestBaseDataset:
    @pytest.mark.parametrize("verbose", [True, False])
    def test_get_resource(self, capsys, dataset_nested_folder, mnist_npy, verbose, monkeypatch):
        def mock_resources(dataset_nested_folder, mnist_npy):
            resources = [
                DataLocation(
                    url="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
                    filename="mnist.npz",
                    md5=False,
                    checksum=get_tmp_hash(mnist_npy / "mnist.npz"),
                ),
                DataLocation(
                    url="https://zenodo.org/record/3239543/files/mnist_c.zip",
                    filename="mnist_c.zip",
                    md5=False,
                    checksum=get_tmp_hash(dataset_nested_folder),
                ),
            ]
            return resources

        monkeypatch.setattr(MNIST, "_resources", mock_resources(dataset_nested_folder, mnist_npy))
        datasetA = MNIST(root=dataset_nested_folder.parent, download=False, corruption="translate", verbose=verbose)
        assert len(datasetA) == 5000
        img, *_ = datasetA[0]
        assert img.shape == (1, 28, 28)
        if verbose:
            captured = capsys.readouterr()
            assert "Determining if mnist_c.zip needs to be downloaded." in captured.out
            assert "mnist_c.zip already exists, skipping download." in captured.out
        datasetB = MNIST(root=mnist_npy, download=False, verbose=verbose)
        assert len(datasetB) == 50000
        img, *_ = datasetA[0]
        assert img.shape == (1, 28, 28)
        if verbose:
            captured = capsys.readouterr()
            assert "Determining if mnist.npz needs to be downloaded." in captured.out
            assert "No download needed, loaded data successfully." in captured.out
        print(datasetA)
        captured = capsys.readouterr()
        assert "Dataset" in captured.out


@pytest.mark.optional
class TestBaseVOCDataset:
    def mock_resources(self, base):
        resources = [
            DataLocation(
                url="https://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar",
                filename="VOCtrainval-2012.tar",
                md5=False,
                checksum=get_tmp_hash(base / "VOCtrainval-2012.tar"),
            ),
            DataLocation(
                url="http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar",
                filename="VOCtrainval_25-May-2011.tar",
                md5=False,
                checksum="0a7f5f5d154f7290ec65ec3f78b72ef72c6d93ff6d79acd40dc222a9ee5248ba",
            ),
            DataLocation(
                url="http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar",
                filename="VOCtrainval_03-May-2010.tar",
                md5=False,
                checksum="1af4189cbe44323ab212bff7afbc7d0f55a267cc191eb3aac911037887e5c7d4",
            ),
            DataLocation(
                url="http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar",
                filename="VOCtrainval_11-May-2009.tar",
                md5=False,
                checksum="11cbe1741fb5bdadbbca3c08e9ec62cd95c14884845527d50847bc2cf57e7fd6",
            ),
            DataLocation(
                url="http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar",
                filename="VOCtrainval_14-Jul-2008.tar",
                md5=False,
                checksum="7f0ca53c1b5a838fbe946965fc106c6e86832183240af5c88e3f6c306318d42e",
            ),
            DataLocation(
                url="https://data.brainchip.com/dataset-mirror/voc/VOCtrainval_06-Nov-2007.tar",
                filename="VOCtrainval_06-Nov-2007.tar",
                md5=False,
                checksum="7d8cd951101b0957ddfd7a530bdc8a94f06121cfc1e511bb5937e973020c7508",
            ),
            DataLocation(
                url="https://data.brainchip.com/dataset-mirror/voc/VOC2012test.tar",
                filename="VOC2012test.tar",
                md5=False,
                checksum=get_tmp_hash(base / "VOC2012test.tar"),
            ),
            DataLocation(
                url="https://data.brainchip.com/dataset-mirror/voc/VOCtest_06-Nov-2007.tar",
                filename="VOCtest_06-Nov-2007.tar",
                md5=False,
                checksum="6836888e2e01dca84577a849d339fa4f73e1e4f135d312430c4856b5609b4892",
            ),
        ]
        return resources

    def test_seg_dataset(self, voc_fake_test, monkeypatch):
        "Test to make sure the BaseSegDataset has all the required parts"
        monkeypatch.setattr(VOCSegmentation, "_resources", self.mock_resources(voc_fake_test))
        dataset = VOCSegmentation(root=voc_fake_test)
        if isinstance(dataset, VOCSegmentation):
            assert dataset._resources is not None
            assert dataset.index2label != {}
            assert dataset.label2index != {}
            assert "id" in dataset.metadata
            assert len(dataset) == 3
            img, target, datum_meta = dataset[1]
            assert img.shape == (3, 10, 10)
            assert isinstance(target, SegmentationTarget)
            assert "pose" in datum_meta

    def test_voc_wrong_year(self, voc_fake):
        """Test ask for test set with wrong year"""
        err_msg = "The only test sets available are for the years 2007 and 2012, not 2010."
        with pytest.raises(ValueError) as e:
            VOCSegmentation(root=voc_fake, image_set="test", year="2010")
        assert err_msg in str(e.value)

    def test_voc_2012_test(self, voc_fake_test, monkeypatch):
        """Test correctly ask for test set"""
        monkeypatch.setattr(VOCSegmentation, "_resources", self.mock_resources(voc_fake_test))
        dataset = VOCSegmentation(root=voc_fake_test, image_set="test", year="2012")
        assert dataset.path.stem == "VOC2012"
        assert (dataset.path / "ImageSets" / "Main" / "test.txt").exists()

    def test_voc_base(self, voc_fake_test, monkeypatch):
        """Test asking for base set with a test set"""
        monkeypatch.setattr(VOCSegmentation, "_resources", self.mock_resources(voc_fake_test))
        dataset = VOCSegmentation(root=voc_fake_test, image_set="base", year="2012")
        assert dataset.path.stem == "VOC2012"
        assert (dataset.path / "ImageSets" / "Main" / "test.txt").exists()
        assert (dataset.path / "ImageSets" / "Main" / "trainval.txt").exists()
