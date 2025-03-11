import hashlib
from pathlib import Path

import numpy as np
import pytest
import requests
from requests import RequestException, Response

from dataeval.utils.data.datasets._base import (
    DataLocation,
    _archive_extraction,
    _download_dataset,
    _ensure_exists,
    _extract_tar_archive,
    _extract_zip_archive,
    _flatten_extraction,
    _ic_data_subselection,
    _validate_file,
)
from dataeval.utils.data.datasets._milco import MILCO
from dataeval.utils.data.datasets._mnist import MNIST
from dataeval.utils.data.datasets._ships import Ships
from dataeval.utils.data.datasets._types import ObjectDetectionTarget, SegmentationTarget
from dataeval.utils.data.datasets._voc import VOCSegmentation

TEMP_MD5 = "d149274109b50d5147c09d6fc7e80c71"
TEMP_SHA256 = "2b749913055289cb3a5c602a17196b5437dc59bba50e986ea449012a303f7201"


def get_tmp_hash(fpath, chunk_size=65535):
    hasher = hashlib.md5()
    with open(fpath, "rb") as fpath_file:
        while chunk := fpath_file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


class MockHTTPError(Response):
    def __init__(self):
        super().__init__()
        self.reason = "MockError"
        self.status_code = 404


class MockRequestException(Response):
    def __init__(self):
        self.reason = "MockError"
        self.status_code = 404

    def raise_for_status(self):
        raise RequestException


@pytest.mark.optional
class TestHelperFunctionsBaseDataset:
    @pytest.mark.parametrize("verbose", [True, False])
    def test_ensure_exists_no_zip(self, capsys, dataset_no_zip, verbose):
        resource = DataLocation(url="fakeurl", filename="stuff.txt", md5=True, checksum=TEMP_MD5)
        _ensure_exists(resource, dataset_no_zip.parent, dataset_no_zip.parent, True, verbose)
        if verbose:
            captured = capsys.readouterr()
            assert captured.out == "stuff.txt already exists, skipping download.\n"

    @pytest.mark.parametrize("verbose", [True, False])
    def test_ensure_exists_single_zip(self, capsys, dataset_single_zip, verbose):
        checksum = get_tmp_hash(dataset_single_zip)
        resource = DataLocation(url="fakeurl", filename="testing.zip", md5=True, checksum=checksum)
        _ensure_exists(resource, dataset_single_zip.parent, dataset_single_zip.parent, True, verbose)
        if verbose:
            captured = capsys.readouterr()
            assert "Extracting testing.zip..." in captured.out

    def test_ensure_exists_file_exists_bad_checksum(self, dataset_no_zip):
        resource = DataLocation(url="fakeurl", filename="stuff.txt", md5=True, checksum=TEMP_SHA256)
        err_msg = "File checksum mismatch. Remove current file and retry download."
        with pytest.raises(Exception) as e:
            _ensure_exists(resource, dataset_no_zip.parent, dataset_no_zip.parent, False)
        assert err_msg in str(e.value)

    def test_ensure_exists_download_non_zip(self, capsys, mnist_folder):
        resource = DataLocation(
            url="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            filename="mnist.npz",
            md5=False,
            checksum="731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1",
        )
        _ensure_exists(resource, mnist_folder, mnist_folder.parent, True, True)
        captured = capsys.readouterr()
        assert (
            "Downloading mnist.npz from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
            in captured.out
        )

    def test_ensure_exists_download_bad_checksum(self, mnist_folder):
        resource = DataLocation(
            url="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            filename="mnist.npz",
            md5=False,
            checksum="abc",
        )
        err_msg = "File checksum mismatch. Remove current file and retry download."
        with pytest.raises(Exception) as e:
            _ensure_exists(resource, mnist_folder, mnist_folder.parent, True, False)
        assert err_msg in str(e.value)

    @pytest.mark.parametrize("verbose", [True, False])
    def test_ensure_exists_download_zip(self, capsys, mnist_folder, verbose):
        resource = DataLocation(
            url="https://figshare.com/ndownloader/files/43168999",
            filename="2021.zip",
            md5=True,
            checksum="b84749b21fa95a4a4c7de3741db78bc7",
        )
        _ensure_exists(resource, mnist_folder, mnist_folder.parent, True, verbose)
        if verbose:
            captured = capsys.readouterr()
            assert f"Extracting {resource.filename}..." in captured.out

    def test_ensure_exists_error(self, dataset_no_zip):
        resource = DataLocation(url="fakeurl", filename="something.zip", md5=True, checksum="")
        err_msg = "Data could not be loaded with the provided root directory,"
        with pytest.raises(FileNotFoundError) as e:
            _ensure_exists(resource, dataset_no_zip.parent, dataset_no_zip.parent, False)
        assert err_msg in str(e.value)

    def test_download_dataset_http_error(self, monkeypatch):
        def mock_get(*args, **kwargs):
            return MockHTTPError()

        monkeypatch.setattr(requests, "get", mock_get)
        with pytest.raises(RuntimeError):
            _download_dataset(url="http://mock/", file_path=Path("fake/path"))

    def test_download_dataset_request_error(self, monkeypatch):
        def mock_get(*args, **kwargs):
            return MockRequestException()

        monkeypatch.setattr(requests, "get", mock_get)
        with pytest.raises(ValueError):
            _download_dataset(url="http://mock/", file_path=Path("fake/path"))

    @pytest.mark.parametrize("use_md5, hash_value", [(True, TEMP_MD5), (False, TEMP_SHA256)])
    def test_validate_file(self, dataset_no_zip, use_md5, hash_value):
        assert _validate_file(dataset_no_zip, hash_value, use_md5)

    @pytest.mark.parametrize("verbose", [True, False])
    def test_zip_extraction_nested_zip(self, capsys, dataset_nested_zip, verbose):
        _archive_extraction(dataset_nested_zip.suffix, dataset_nested_zip, dataset_nested_zip.parent, False, verbose)
        if verbose:
            captured = capsys.readouterr()
            assert "Extracting nested zip" in captured.out

    def test_extract_archive_bad_zip(self, dataset_no_zip):
        err_msg = f"{dataset_no_zip.name} is not a valid zip file, skipping extraction."
        with pytest.raises(FileNotFoundError) as e:
            _extract_zip_archive(dataset_no_zip, dataset_no_zip.parent)
        assert err_msg in str(e.value)

    @pytest.mark.parametrize("verbose", [True, False])
    def test_flatten_extraction(self, capsys, dataset_nested_folder, verbose):
        base = dataset_nested_folder.parent
        _extract_zip_archive(dataset_nested_folder, base)
        _flatten_extraction(
            base,
            verbose,
        )
        if verbose:
            captured = capsys.readouterr()
            assert "Moving translate to /tmp" in captured.out
            assert "Removing empty folder mnist_c" in captured.out
        assert (base / "translate").exists()

    @pytest.mark.parametrize(
        "size, from_back, balance, trunc, expected",
        [
            (-1, False, False, False, np.array([0, 1, 2, 5, 4, 3, 7, 6, 8, 9])),
            (3, True, True, False, np.array([9, 7, 6])),
            (1, False, True, False, np.array([0, 1, 2])),
            (15, False, True, False, np.array([0, 1, 2, 5, 4, 3, 7, 6, 8])),
            (15, False, False, False, np.array([0, 1, 2, 5, 4, 3, 7, 6, 8, 9])),
            (5, True, False, False, np.array([9, 7, 6, 8, 5])),
            (-1, False, False, True, np.array([0, 1, 2, 5, 4, 3, 7, 6, 8])),
        ],
    )
    def test_ic_data_subselection(self, size, from_back, balance, trunc, expected):
        labels = ["0", "1", "2", "2", "1", "0", "1", "0", "2", "2"]
        if trunc:
            labels = labels[:-1]
        out = _ic_data_subselection(labels, {0, 1, 2}, size, from_back, balance, False)
        assert np.all(out == expected)

    @pytest.mark.parametrize(
        "size, from_back, balance, expected",
        [
            (15, False, True, np.array([0, 1, 2, 5, 4, 3, 7, 6, 8])),
            (15, True, True, np.array([9, 7, 6, 8, 5, 4, 3, 0, 1])),
            (15, False, False, np.array([0, 1, 2, 5, 4, 3, 7, 6, 8, 9])),
        ],
    )
    def test_ic_data_subselection_warning(self, size, from_back, balance, expected):
        labels = ["0", "1", "2", "2", "1", "0", "1", "0", "2", "2"]
        if balance:
            warn_msg = (
                f"Because of dataset limitations, only {9} samples will be returned, instead of the desired {size}."
            )
        else:
            sent2 = "Adjusting down to raw dataset size."
            warn_msg = f"Asked for more samples, {size}, than the raw dataset contains, {10}. " + sent2
        with pytest.warns(UserWarning, match=warn_msg):
            out = _ic_data_subselection(labels, {0, 1, 2}, 15, from_back, balance, True)
        assert np.all(out == expected)

    def test_tarfile_error(self, dataset_single_zip):
        err_msg = f"{dataset_single_zip.name} is not a valid tar file"
        with pytest.raises(FileNotFoundError) as e:
            _extract_tar_archive(file_path=dataset_single_zip, extract_to=dataset_single_zip.parent)
        assert err_msg in str(e.value)


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
        assert datasetA._reorder.size == 5000
        img, *_ = datasetA[0]
        assert img.shape == (1, 28, 28)
        if verbose:
            captured = capsys.readouterr()
            assert "Determining if mnist_c.zip needs to be downloaded." in captured.out
            assert "mnist_c.zip already exists, skipping download." in captured.out
        datasetB = MNIST(root=mnist_npy, download=False, balance=False, verbose=verbose)
        assert datasetB._reorder.size == 50000
        img, *_ = datasetA[0]
        assert img.shape == (1, 28, 28)
        if verbose:
            captured = capsys.readouterr()
            assert "Determining if mnist.npz needs to be downloaded." in captured.out
            assert "No download needed, loaded data successfully." in captured.out

    def test_dataset_unit_interval(self, mnist_npy):
        """Test unit_interval functionality"""
        dataset = MNIST(root=mnist_npy, size=1000, unit_interval=True)
        images = np.vstack([img for img, _, _ in dataset])
        assert np.all((images >= 0) & (images <= 1))

    def test_dataset_normalize(self, mnist_npy):
        """Test normalization functionality."""
        dataset = MNIST(root=mnist_npy, size=1000, unit_interval=True, normalize=(0.5, 0.5), dtype=np.float32)
        images = np.vstack([img for img, _, _ in dataset])
        assert np.all((images >= -1) & (images <= 1))
        assert np.min(images) == -1

    def test_dataset_flatten(self, mnist_npy):
        """Test flattening functionality."""
        dataset = MNIST(root=mnist_npy, size=1000, flatten=True)
        images = np.vstack([img for img, _, _ in dataset])
        assert images.shape == (1000, 784)

    @pytest.mark.parametrize(
        "channels, expected_img, expected_scene",
        [
            ("channels_first", (3, 10, 10), (3, 1500, 1250)),
            ("channels_last", (10, 10, 3), (1500, 1250, 3)),
        ],
    )
    def test_dataset_channels(self, ship_fake, channels, expected_img, expected_scene):
        """Test channels_first functionality."""
        dataset = Ships(root=str(ship_fake), size=1000, channels=channels)
        img, _, _ = dataset[0]
        assert img.shape == expected_img
        scene = dataset.get_scene(0)
        assert scene.shape == expected_scene


@pytest.mark.optional
class TestBaseICDataset:
    @pytest.mark.parametrize("verbose", [True, False])
    def test_dataset_preprocess(self, capsys, wrong_mnist, verbose):
        """Test selecting different sized datasets."""
        if not verbose:
            dataset = MNIST(root=wrong_mnist, size=5000, verbose=verbose)
            assert len(dataset) == 5000
        else:
            dataset = MNIST(root=wrong_mnist, size=5000, slice_back=False, verbose=verbose)
            captured = capsys.readouterr()
            assert "Running data preprocessing steps" in captured.out
            assert len(dataset) == 5000

    @pytest.mark.parametrize("balance, from_back", [(True, False), (False, True), (False, False)])
    def test_dataset_preprocess_metadata(self, ship_fake, wrong_mnist, balance, from_back):
        """Test selecting different sized datasets with and without metadata."""
        dataset = Ships(root=ship_fake, balance=balance, slice_back=from_back)
        assert dataset._datum_metadata != {}
        dataset = MNIST(root=wrong_mnist, balance=balance, slice_back=from_back)
        assert dataset._datum_metadata == {}

    @pytest.mark.parametrize("ship, expected", [(True, (4000, 500)), (False, (49680, 100))])
    def test_dataset_slice_back(self, ship_fake, mnist_npy, ship, expected):
        """Test the functionality of slicing from the back."""
        if ship:
            datasetA = Ships(root=ship_fake, size=-1, slice_back=True)
            datasetB = Ships(root=ship_fake, size=1000, slice_back=True, balance=False)
        else:
            datasetA = MNIST(root=mnist_npy, size=-1, slice_back=True)
            datasetB = MNIST(root=mnist_npy, size=1000, slice_back=True, balance=False)
            label_arrayA = np.array(datasetB._labels, dtype=np.uintp)
            _, countsA = np.unique_counts(label_arrayA[datasetA._reorder[:1000]])
            label_arrayB = np.array(datasetB._labels, dtype=np.uintp)
            _, countsB = np.unique_counts(label_arrayB[datasetB._reorder])
            assert np.all(countsA == countsB)

        assert datasetA._reorder.size == expected[0]
        label_arrayB = np.array(datasetB._labels, dtype=np.uintp)
        _, counts = np.unique_counts(label_arrayB[datasetB._reorder])
        assert np.all(counts == expected[1])

    def test_ic_dataset(self, mnist_npy):
        """Test dataset properties."""
        dataset = MNIST(root=mnist_npy, size=1000)
        if isinstance(dataset, MNIST):
            assert dataset._resources is not None
            assert dataset.index2label != {}
            assert dataset.label2index != {}
            assert "id" in dataset.metadata
            assert dataset.class_set != {}
            assert dataset.num_classes is not None
            assert len(dataset) == 1000
            img, *_ = dataset[0]
            assert img.shape == (1, 28, 28)
            this = dataset.info()
            assert "Train\n-----\n" in this


@pytest.mark.optional
class TestBaseODDataset:
    def test_od_dataset(self, milco_fake):
        "Test to make sure the BaseODDataset has all the required parts"
        dataset = MILCO(root=milco_fake)  # type: ignore
        if isinstance(dataset, MILCO):
            assert dataset._resources is not None
            assert dataset.index2label != {}
            assert dataset.label2index != {}
            assert "id" in dataset.metadata
            assert dataset.class_set != {}
            assert dataset.num_classes is not None
            assert len(dataset) == 12
            for i in range(6):
                img, target, datum_meta = dataset[i]
                assert img.shape == (3, 10, 10)
                assert isinstance(target, ObjectDetectionTarget)
                assert "year" in datum_meta


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
            assert dataset.class_set != {}
            assert dataset.num_classes is not None
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
        dataset = VOCSegmentation(
            root=voc_fake,
            image_set="test",
            year="2007",
            size=1,
            unit_interval=True,
            dtype=np.float32,
            channels="channels_last",
            normalize=(2.0, 1.0),
            slice_back=True,
        )
        assert dataset.dataset_dir.stem == "VOC2007"
