from collections import Counter
from unittest.mock import patch

import numpy as np
import pytest
from requests import HTTPError, RequestException, Response

from dataeval.utils.data.datasets._ic import (
    MNIST,
    _check_exists,
    _download_dataset,
    _extract_archive,
    _get_file,
    _validate_file,
)

TEMP_MD5 = "d149274109b50d5147c09d6fc7e80c71"
TEMP_SHA256 = "2b749913055289cb3a5c602a17196b5437dc59bba50e986ea449012a303f7201"


class MockHTTPError(HTTPError):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.response = Response()
        self.response.reason = "MockError"
        self.response.status_code = 404


@pytest.mark.optional
class TestMNISTFile:
    def test_check_exists_path_exists(self, capsys, mnist_file):
        _check_exists(mnist_file, "fakeurl", "root", mnist_file.name, "file_hash", False, True)
        captured = capsys.readouterr()
        assert captured.out == "Files already downloaded and verified\n"
        location = _check_exists(mnist_file, "fakeurl", "root", mnist_file.name, "file_hash", False, False)
        assert str(mnist_file) == location

    @pytest.mark.parametrize("use_md5, hash_value", [(True, TEMP_MD5), (False, TEMP_SHA256)])
    def test_validate_file_md5(self, mnist_file, use_md5, hash_value):
        assert _validate_file(mnist_file, hash_value, use_md5)

    @pytest.mark.parametrize("use_md5, hash_value", [(True, TEMP_MD5), (False, TEMP_SHA256)])
    def test_get_file_exists_md5(self, mnist_file, use_md5, hash_value):
        _get_file(
            root=mnist_file.parent, fname=mnist_file.name, origin="http://mock", file_hash=hash_value, md5=use_md5
        )

    def test_check_exists_no_path(self):
        with pytest.raises(RuntimeError):
            _check_exists("folder_path", "fakeurl", "root", "name", "file_hash", False)

    def test_check_exists_download(self, capsys, mnist_download):
        parent, name = mnist_download
        _check_exists(folder="folder_path", url="http://mock", root=parent, fname=name, file_hash=TEMP_SHA256)
        captured = capsys.readouterr()
        assert captured.out == "File already downloaded and verified.\n"
        location = _check_exists(
            folder="folder_path", url="http://mock", root=parent, fname=name, file_hash=TEMP_SHA256, verbose=False
        )
        assert str(parent / "mnist") == location

    @patch("dataeval.utils.data.datasets._ic.requests.get", side_effect=MockHTTPError())
    def test_get_file_http_error(self, mock_get, mnist_download):
        parent, name = mnist_download
        with pytest.raises(RuntimeError):
            _get_file(root=parent, fname=name, origin="http://mock", file_hash=TEMP_SHA256, md5=True)

    @patch("dataeval.utils.data.datasets._ic.requests.get", side_effect=RequestException())
    def test_get_file_request_error(self, mock_get, mnist_download):
        _, name = mnist_download
        with pytest.raises(ValueError):
            _get_file(root="wrong_path", fname=name, origin="http://mock", file_hash=TEMP_SHA256, md5=True)

    def test_extract_archive(self, zip_file):
        location = _extract_archive(zip_file)
        assert str(zip_file.parent) == location
        location = _extract_archive(zip_file, zip_file.parent, remove_finished=True)
        assert str(zip_file.parent) == location

    @patch("dataeval.utils.data.datasets._ic._get_file")
    @patch("dataeval.utils.data.datasets._ic._extract_archive")
    @pytest.mark.parametrize("md5", [True, False])
    def test_download_dataset_extract_on_mnist_zip(self, mock_extract_archive, mock_get_file, md5, tmp_path):
        _download_dataset("mock", tmp_path, "mock.zip", "abc", md5=md5)
        assert mock_get_file.called
        assert mock_extract_archive.called == md5


@pytest.mark.optional
class TestMNIST:
    def test_mnist_initialization(self, mnist_npy):
        """Test MNIST dataset initialization."""
        dataset = MNIST(root=str(mnist_npy), size=-1, balance=False)
        assert dataset.targets.ndim == 1
        assert dataset.data.shape == (60000, 28, 28)

    def test_mnist_test_data(self, mnist_npy):
        """Test loading the test set."""
        dataset = MNIST(root=mnist_npy, train=False, size=-1, balance=False)
        assert dataset.targets.ndim == 1
        assert dataset.data.shape == (10000, 28, 28)

    @pytest.mark.parametrize(
        "size, expected",
        [
            (5, (5,)),
            (100, (100,)),
            (1000, (1000,)),
        ],
    )
    def test_mnist_size_data(self, mnist_npy, size, expected):
        """Test selecting different sized datasets."""
        dataset = MNIST(root=mnist_npy, size=size, balance=False)
        assert dataset.targets.shape == expected

    def test_mnist_oversized(self, mnist_npy):
        """Test asking for more data than is available."""
        warn_msg = (
            f"Asked for more samples, {15000}, than the raw dataset contains, {10000}. "
            "Adjusting down to raw dataset size."
        )
        with pytest.warns(UserWarning, match=warn_msg):
            dataset = MNIST(root=mnist_npy, train=False, size=15000, balance=False)
        assert dataset.targets.shape == (10000,)
        dataset = MNIST(root=mnist_npy, train=False, size=15000, balance=True, verbose=False)
        assert dataset.targets.shape == (9680,)
        warn_msg = (
            f"Because of dataset limitations, only {9680} samples will be returned, instead of the desired {9800}."
        )
        with pytest.warns(UserWarning, match=warn_msg):
            dataset = MNIST(root=mnist_npy, train=False, size=9800, balance=False)
        assert dataset.targets.shape == (9680,)
        dataset = MNIST(root=mnist_npy, train=False, size=9000, balance=False, verbose=False)
        assert dataset.targets.shape == (9000,)

    def test_mnist_normalize(self, mnist_npy):
        """Test if unit_interval, normalization, and dtype works properly."""
        dataset = MNIST(root=mnist_npy, unit_interval=True, normalize=(0.5, 0.5), dtype=np.float32)
        assert np.all((dataset.data >= -1) & (dataset.data <= 1))
        assert np.min(dataset.data) == -1
        assert str(dataset.data.dtype) == "float32"

    def test_mnist_flatten(self, mnist_npy):
        """Test flattening functionality."""
        dataset = MNIST(root=mnist_npy, size=1000, flatten=True)
        assert dataset.data.shape == (1000, 784)

    @pytest.mark.parametrize(
        "channels, expected",
        [
            ("channels_first", (1000, 1, 28, 28)),
            ("channels_last", (1000, 28, 28, 1)),
        ],
    )
    def test_mnist_channels(self, mnist_npy, channels, expected):
        """Test channels_first functionality."""
        dataset = MNIST(root=mnist_npy, size=1000, channels=channels)
        assert dataset.data.shape == expected

    @pytest.mark.parametrize(
        "classes, subset, expected",
        [
            (["zero", "one", "two", "five", "nine"], {0, 1, 2, 5, 9}, 200),
            ([4, 7, 8, 9, 15], {4, 7, 8, 9}, 250),
            ("six", {6}, 1000),
            (3, {3}, 1000),
        ],
    )
    def test_mnist_class_selection(self, mnist_npy, classes, subset, expected):
        """Test class selection and equalize functionality."""
        dataset = MNIST(root=mnist_npy, size=1000, classes=classes, balance=True)
        labels = np.unique(dataset.targets)
        counts = Counter(dataset.targets)
        assert set(labels).issubset(subset)
        assert counts[labels[0]] == expected  # type: ignore

    def test_mnist_corruption(self, capsys, mnist_npy):
        """Test the loading of all the corruptions."""
        dataset = MNIST(root=mnist_npy, size=-1, balance=False, corruption="identity", verbose=False)
        assert dataset.targets.ndim == 1
        assert dataset.data.shape == (60000, 28, 28)
        dataset = MNIST(root=mnist_npy, corruption="identity")
        captured = capsys.readouterr()
        assert captured.out == (
            "Identity is not a corrupted dataset but the original MNIST dataset.\n"
            "Files already downloaded and verified\n"
        )

    def test_mnist_slice_back(self, mnist_npy):
        """Test the functionality of slicing from the back."""
        datasetA = MNIST(root=mnist_npy, size=1000, slice_back=True, randomize=True)
        datasetB = MNIST(root=mnist_npy, size=1000, randomize=False)
        assert np.all(datasetA.targets == datasetB.targets)
        assert not np.all(datasetA.data == datasetB.data)

    def test_mnist_dataset(self, mnist_npy):
        """Test dataset properties."""
        dataset = MNIST(root=mnist_npy, size=1000)
        if isinstance(dataset, MNIST):
            assert len(dataset) == 1000
            img, *_ = dataset[0]
            assert img.shape == (28, 28)
