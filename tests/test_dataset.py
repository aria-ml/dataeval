from collections import Counter

import numpy as np
import pytest

from dataeval._internal.datasets import MNIST, _get_file, _validate_file, check_exists, extract_archive

TEMP_MD5 = "d149274109b50d5147c09d6fc7e80c71"
TEMP_SHA256 = "2b749913055289cb3a5c602a17196b5437dc59bba50e986ea449012a303f7201"


@pytest.mark.xdist_group(name="mnist_file")
def test_check_exists_path_exists(capsys, mnist_file):
    check_exists(mnist_file, "fakeurl", "root", mnist_file.name, "file_hash", False, True)
    captured = capsys.readouterr()
    assert captured.out == "Files already downloaded and verified\n"
    location = check_exists(mnist_file, "fakeurl", "root", mnist_file.name, "file_hash", False, False)
    assert str(mnist_file) == location


def test_check_exists_no_path():
    with pytest.raises(RuntimeError):
        check_exists("folder_path", "fakeurl", "root", "name", "file_hash", False)


@pytest.mark.xdist_group(name="mnist_download")
def test_check_exists_download(capsys, mnist_download):
    parent, name = mnist_download
    check_exists(folder="folder_path", url="http://mock", root=parent, fname=name, file_hash=TEMP_SHA256)
    captured = capsys.readouterr()
    assert captured.out == "File already downloaded and verified.\n"
    location = check_exists(
        folder="folder_path", url="http://mock", root=parent, fname=name, file_hash=TEMP_SHA256, verbose=False
    )
    assert str(parent / "mnist") == location


@pytest.mark.xdist_group(name="mnist_file")
@pytest.mark.parametrize("use_md5, hash_value", [(True, TEMP_MD5), (False, TEMP_SHA256)])
def test_validate_file_md5(mnist_file, use_md5, hash_value):
    assert _validate_file(mnist_file, hash_value, use_md5)


@pytest.mark.xdist_group(name="mnist_file")
@pytest.mark.parametrize("use_md5, hash_value", [(True, TEMP_MD5), (False, TEMP_SHA256)])
def test_get_file_exists_md5(mnist_file, use_md5, hash_value):
    _get_file(root=mnist_file.parent, fname=mnist_file.name, origin="http://mock", file_hash=hash_value, md5=use_md5)


@pytest.mark.xdist_group(name="mnist_download")
def test_get_file_error(mnist_download):
    parent, name = mnist_download
    with pytest.raises(Exception):
        _get_file(root=parent, fname=name, origin="http://mock", file_hash=TEMP_SHA256, md5=True)
    with pytest.raises(Exception):
        _get_file(root="wrong_path", fname=name, origin="http://mock", file_hash=TEMP_SHA256, md5=True)


@pytest.mark.xdist_group(name="mnist_zip")
def test_extract_archive(zip_file):
    location = extract_archive(zip_file)
    assert str(zip_file.parent) == location
    location = extract_archive(zip_file, zip_file.parent, remove_finished=True)
    assert str(zip_file.parent) == location


class TestMNIST:
    @pytest.mark.xdist_group(name="mnist_npy")
    def test_mnist_initialization(self, mnist_npy):
        """Test MNIST dataset initialization."""
        dataset = MNIST(root=str(mnist_npy), size=-1, balance=False)
        assert dataset.targets.ndim == 1
        assert dataset.data.shape == (60000, 28, 28)

    @pytest.mark.xdist_group(name="mnist_npy")
    def test_mnist_test_data(self, mnist_npy):
        """Test loading the test set."""
        dataset = MNIST(root=mnist_npy, train=False, size=-1, balance=False)
        assert dataset.targets.ndim == 1
        assert dataset.data.shape == (10000, 28, 28)

    @pytest.mark.xdist_group(name="mnist_npy")
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

    @pytest.mark.xdist_group(name="mnist_npy")
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

    @pytest.mark.xdist_group(name="mnist_npy")
    def test_mnist_normalize(self, mnist_npy):
        """Test if unit_interval, normalization, and dtype works properly."""
        dataset = MNIST(root=mnist_npy, unit_interval=True, normalize=(0.5, 0.5), dtype=np.float32)
        assert np.all((dataset.data >= -1) & (dataset.data <= 1))
        assert np.min(dataset.data) == -1
        assert str(dataset.data.dtype) == "float32"

    @pytest.mark.xdist_group(name="mnist_npy")
    def test_mnist_flatten(self, mnist_npy):
        """Test flattening functionality."""
        dataset = MNIST(root=mnist_npy, size=1000, flatten=True)
        assert dataset.data.shape == (1000, 784)

    @pytest.mark.xdist_group(name="mnist_npy")
    @pytest.mark.parametrize(
        "channels, expected",
        [
            ("channels_first", (1000, 1, 28, 28)),
            ("channels_last", (1000, 28, 28, 1)),
        ],
    )
    @pytest.mark.xdist_group(name="mnist_npy")
    def test_mnist_channels(self, mnist_npy, channels, expected):
        """Test channels_first functionality."""
        dataset = MNIST(root=mnist_npy, size=1000, channels=channels)
        assert dataset.data.shape == expected

    @pytest.mark.xdist_group(name="mnist_npy")
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

    @pytest.mark.xdist_group(name="mnist_npy")
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

    @pytest.mark.xdist_group(name="mnist_npy")
    def test_mnist_slice_back(self, mnist_npy):
        """Test the functionality of slicing from the back."""
        datasetA = MNIST(root=mnist_npy, size=1000, slice_back=True, randomize=True)
        datasetB = MNIST(root=mnist_npy, size=1000)
        assert np.all(datasetA.targets == datasetB.targets)
        assert not np.all(datasetA.data == datasetB.data)

    @pytest.mark.xdist_group(name="mnist_npy")
    def test_mnist_dataset(self, mnist_npy):
        """Test dataset properties."""
        dataset = MNIST(root=mnist_npy, size=1000)
        if isinstance(dataset, MNIST):
            assert len(dataset) == 1000
            img, _ = dataset[0]
            assert img.shape == (28, 28)
