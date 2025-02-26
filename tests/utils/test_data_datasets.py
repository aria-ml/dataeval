import hashlib
from collections import Counter
from unittest.mock import patch

import numpy as np
import pytest
from requests import HTTPError, RequestException, Response

from dataeval.utils.data.datasets._ic import (
    MNIST,
    DataLocation,
    ShipDataset,
    _download_dataset,
    _ensure_exists,
    _extract_archive,
    _validate_file,
    _zip_extraction,
)

TEMP_MD5 = "d149274109b50d5147c09d6fc7e80c71"
TEMP_SHA256 = "2b749913055289cb3a5c602a17196b5437dc59bba50e986ea449012a303f7201"


def get_tmp_hash(fpath, chunk_size=65535):
    hasher = hashlib.md5()
    with open(fpath, "rb") as fpath_file:
        while chunk := fpath_file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


class MockHTTPError(HTTPError):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.response = Response()
        self.response.reason = "MockError"
        self.response.status_code = 404


@pytest.mark.optional
class TestBaseICDatasetFile:
    @pytest.mark.parametrize("verbose", [True, False])
    def test_ensure_exists_no_zip(self, capsys, dataset_no_zip, verbose):
        resource = DataLocation(url="fakeurl", filename="stuff.txt", md5=True, checksum=TEMP_MD5)
        _ensure_exists(resource, dataset_no_zip.parent, True, verbose)
        if verbose:
            captured = capsys.readouterr()
            assert captured.out == "stuff.txt already exists, skipping download.\n"

    @pytest.mark.parametrize("verbose", [True, False])
    def test_ensure_exists_single_zip(self, capsys, dataset_single_zip, verbose):
        checksum = get_tmp_hash(dataset_single_zip)
        resource = DataLocation(url="fakeurl", filename="testing.zip", md5=True, checksum=checksum)
        _ensure_exists(resource, dataset_single_zip.parent, True, verbose)
        if verbose:
            captured = capsys.readouterr()
            assert "Extracting testing.zip..." in captured.out

    def test_ensure_exists_file_exists_bad_checksum(self, dataset_no_zip):
        resource = DataLocation(url="fakeurl", filename="stuff.txt", md5=True, checksum=TEMP_SHA256)
        err_msg = "File checksum mismatch. Remove current file and retry download."
        with pytest.raises(Exception) as e:
            _ensure_exists(resource, dataset_no_zip.parent, False)
        assert err_msg in str(e.value)

    def test_ensure_exists_download(self, capsys, mnist_folder):
        resource = DataLocation(
            url="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            filename="mnist.npz",
            md5=False,
            checksum="731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1",
        )
        _ensure_exists(resource, mnist_folder, True, True)
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
            _ensure_exists(resource, mnist_folder, True, False)
        assert err_msg in str(e.value)

    @pytest.mark.parametrize("use_md5, hash_value", [(True, TEMP_MD5), (False, TEMP_SHA256)])
    def test_validate_file_md5(self, dataset_no_zip, use_md5, hash_value):
        assert _validate_file(dataset_no_zip, hash_value, use_md5)

    @pytest.mark.parametrize("verbose", [True, False])
    def test_extract_archive_nested_zip(self, capsys, dataset_nested_zip, verbose):
        _extract_archive(dataset_nested_zip, dataset_nested_zip.parent, verbose)
        if verbose:
            captured = capsys.readouterr()
            assert "Extracting nested zip" in captured.out

    def test_extract_archive_bad_zip(self, dataset_no_zip):
        err_msg = f"{dataset_no_zip.name} is not a valid zip file, skipping extraction."
        with pytest.raises(FileNotFoundError) as e:
            _extract_archive(dataset_no_zip, dataset_no_zip.parent, False)
        assert err_msg in str(e.value)

    @pytest.mark.parametrize("verbose", [True, False])
    def test_flatten_extraction(self, capsys, dataset_nested_folder, verbose):
        base = dataset_nested_folder.parent
        _zip_extraction(
            dataset_nested_folder,
            base,
            verbose,
        )
        if verbose:
            captured = capsys.readouterr()
            assert "Moving translate to /tmp" in captured.out
            assert "Removing empty folder mnist_c" in captured.out
        assert (base / "translate").exists()

    @patch("dataeval.utils.data.datasets._ic.requests.get", side_effect=MockHTTPError())
    def test_download_dataset_http_error(self, mock_get, dataset_no_zip):
        with pytest.raises(RuntimeError):
            _download_dataset(url="http://mock/", file_path=dataset_no_zip)

    @patch("dataeval.utils.data.datasets._ic.requests.get", side_effect=RequestException())
    def test_get_file_request_error(self, mock_get, dataset_no_zip):
        with pytest.raises(ValueError):
            _download_dataset(url="http://mock/", file_path=dataset_no_zip)


@pytest.mark.optional
class TestBaseICDataset:
    @pytest.mark.parametrize(
        "size, expected, balanced",
        [
            (5, (5,), False),
            (100, (100,), False),
            (1000, (1000,), False),
            (5, (10,), True),
            (55, (50,), True),
            (1000, (1000,), True),
        ],
    )
    def test_dataset_size(self, mnist_npy, size, expected, balanced):
        """Test selecting different sized datasets."""
        dataset = MNIST(root=mnist_npy, size=size, balance=balanced)
        assert dataset._targets.shape == expected

    def test_dataset_oversized(self, ship_fake):
        """Test asking for more data than is available."""
        warn_msg = (
            f"Asked for more samples, {15000}, than the raw dataset contains, {4000}. "
            "Adjusting down to raw dataset size."
        )
        with pytest.warns(UserWarning, match=warn_msg):
            dataset = ShipDataset(root=ship_fake, size=15000, balance=False)
        assert dataset._targets.shape == (4000,)
        dataset = ShipDataset(root=ship_fake, size=5000, balance=True, verbose=False)
        assert dataset._targets.shape == (2000,)

    def test_dataset_overbalanced(self, mnist_npy):
        warn_msg = (
            f"Because of dataset limitations, only {9680} samples will be returned, instead of the desired {9800}."
        )
        with pytest.warns(UserWarning, match=warn_msg):
            dataset = MNIST(root=mnist_npy, train=False, size=9800, balance=True)
        assert dataset._targets.shape == (9680,)
        dataset = MNIST(root=mnist_npy, train=False, size=9800, balance=False, verbose=False)
        assert dataset._targets.shape == (9800,)

    @pytest.mark.parametrize("ship, expected", [(True, (4000, 500)), (False, (59680, 100))])
    def test_dataset_slice_back(self, ship_fake, mnist_npy, ship, expected):
        """Test the functionality of slicing from the back."""
        if ship:
            datasetA = ShipDataset(root=ship_fake, size=-1, slice_back=True, randomize=False)
            datasetB = ShipDataset(root=ship_fake, size=1000, slice_back=True, balance=False, randomize=True)
        else:
            datasetA = MNIST(root=mnist_npy, size=-1, slice_back=True, randomize=False)
            datasetB = MNIST(root=mnist_npy, size=1000, slice_back=True, balance=False, randomize=True)
            _, countsA = np.unique_counts(datasetA._targets[:1000])
            _, countsB = np.unique_counts(datasetB._targets)
            assert np.all(countsA == countsB)

        assert datasetA._targets.size == expected[0]
        _, counts = np.unique_counts(datasetB._targets)
        assert np.all(counts == expected[1])

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
        dataset = ShipDataset(root=str(ship_fake), size=1000, channels=channels)
        img, _, _ = dataset[0]
        assert img.shape == expected_img
        assert dataset.scenes[0].shape == expected_scene

    def test_dataset(self, mnist_npy):
        """Test dataset properties."""
        dataset = MNIST(root=mnist_npy, size=1000)
        if isinstance(dataset, MNIST):
            assert len(dataset) == 1000
            img, *_ = dataset[0]
            assert img.shape == (1, 28, 28)
            assert "id" in dataset.metadata


@pytest.mark.optional
class TestMNIST:
    def test_mnist_initialization(self, mnist_npy):
        """Test MNIST dataset initialization."""
        dataset = MNIST(root=str(mnist_npy), size=-1, balance=False)
        assert dataset._targets.ndim == 1
        assert dataset._data.shape == (60000, 1, 28, 28)

    def test_mnist_wrong_initialization(self, wrong_mnist):
        """Test MNIST dataset initialization if asked for normal but downloaded corrupt."""
        dataset = MNIST(root=wrong_mnist, train=True, size=-1, balance=False)
        assert dataset._targets.ndim == 1
        assert dataset._data.shape == (5000, 1, 28, 28)

    def test_mnist_test_data(self, mnist_npy):
        """Test loading the test set."""
        dataset = MNIST(root=mnist_npy, train=False, size=-1, balance=False)
        assert dataset._targets.ndim == 1
        assert dataset._data.shape == (10000, 1, 28, 28)

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
        labels = np.unique(dataset._targets)
        counts = Counter(dataset._targets)
        assert set(labels).issubset(subset)
        assert counts[labels[0]] == expected  # type: ignore

    def test_mnist_corruption(self, capsys, mnist_npy):
        """Test the loading of all the corruptions."""
        dataset = MNIST(root=mnist_npy, size=-1, balance=False, corruption="identity", verbose=False)
        assert dataset._targets.ndim == 1
        assert dataset._data.shape == (60000, 1, 28, 28)
        dataset = MNIST(root=mnist_npy, corruption="identity")
        captured = capsys.readouterr()
        msg = "Identity is not a corrupted dataset but the original MNIST dataset."
        assert msg in captured.out


@pytest.mark.optional
class TestShipDataset:
    def test_ship_initialization(self, ship_fake):
        """Test Ship dataset initialization."""
        dataset = ShipDataset(root=str(ship_fake))
        assert dataset._targets.ndim == 1
        assert dataset._data.shape == (4000, 3, 10, 10)
        assert dataset.scenes[0].shape == (3, 1500, 1250)
        assert dataset._datum_metadata
