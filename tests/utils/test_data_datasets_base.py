import hashlib

import pytest

from dataeval.utils.data._types import SegmentationTarget
from dataeval.utils.data.datasets._base import DataLocation
from dataeval.utils.data.datasets._mnist import MNIST
from dataeval.utils.data.datasets._voc import VOCSegmentation

TEMP_MD5 = "d149274109b50d5147c09d6fc7e80c71"
TEMP_SHA256 = "2b749913055289cb3a5c602a17196b5437dc59bba50e986ea449012a303f7201"


def get_tmp_hash(fpath, chunk_size=65535):
    hasher = hashlib.md5()
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
                    md5=True,
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
                url="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
                filename="VOCtrainval_11-May-2012.tar",
                md5=True,
                checksum=get_tmp_hash(base / "VOCtrainval_11-May-2012.tar"),
            ),
        ]
        return resources

    def test_seg_dataset(self, voc_fake, monkeypatch):
        "Test to make sure the BaseSegDataset has all the required parts"
        monkeypatch.setattr(VOCSegmentation, "_resources", self.mock_resources(voc_fake))
        dataset = VOCSegmentation(root=voc_fake)
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
        err_msg = "The only test set available is for the year 2007, not 2012."
        with pytest.raises(ValueError) as e:
            VOCSegmentation(root=voc_fake, image_set="test")
        assert err_msg in str(e.value)

    def test_voc_2007_test(self, voc_fake, monkeypatch):
        """Test correctly ask for test set"""
        monkeypatch.setattr(VOCSegmentation, "_resources", self.mock_resources(voc_fake))
        dataset = VOCSegmentation(root=voc_fake, year="2007", image_set="test")
        assert dataset.path.stem == "VOC2007"
