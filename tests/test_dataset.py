import shutil
import tempfile
import warnings
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from dataeval._internal.datasets import MNIST, _get_file, check_exists, extract_archive
from tests.conftest import mnist, skip_mnist


@contextmanager
def wait_lock(path: Path, timeout: int = 30):
    try:
        from filelock import FileLock
    except ImportError:
        warnings.warn("FileLock dependency not found, read/write collisions possible when running in parallel.")
        yield
        return

    lock = FileLock(str(path), timeout=timeout)
    with lock:
        yield


@skip_mnist
class TestDownloadingFunctions:
    mirror = "https://zenodo.org/record/3239543/files/"
    resources = ("mnist_c.zip", "4b34b33045869ee6d424616cd3a65da3")

    def test_folder_error(self):
        with pytest.raises(RuntimeError):
            check_exists("will", "this", "work", "yes", "no", False)

    def test_download_corruption(self):
        path = Path("./data2")
        if not path.is_absolute():
            path = path.resolve()

        lock_file = Path(path, "mnist.lock")

        folder = Path(path) / "mnist_c" / "identity"
        with wait_lock(lock_file, timeout=60):
            check_exists(folder, self.mirror, path, self.resources[0], self.resources[1], md5=True)
        check_exists(folder, self.mirror, path, self.resources[0], self.resources[1], verbose=False, md5=True)

        folder = Path(path) / "mnist_c"
        with wait_lock(lock_file, timeout=60):
            tmp_path = _get_file(
                folder, self.resources[0], self.mirror + self.resources[0], self.resources[1], False, md5=True
            )
            extract_archive(str(tmp_path))
            _get_file(folder, self.resources[0], self.mirror + self.resources[0], self.resources[1], True, md5=True)
            with pytest.raises(ValueError):
                _get_file(folder, self.resources[0], self.mirror + self.resources[0], "wrongNumber", md5=True)

        shutil.rmtree(path)

    def test_load_corruption(self):
        path = Path("./data3")
        if not path.is_absolute():
            path = path.resolve()

        data, targets = mnist(root=path, size=-1, corruption="identity", verbose=False)
        assert targets.ndim == 1
        assert data.shape == (54210, 28, 28)

        shutil.rmtree(path)

    def test_url_error(self):
        temp_dir = tempfile.gettempdir()
        lock_file = Path(temp_dir, "mnist.lock")

        with wait_lock(lock_file), pytest.raises(Exception):
            _get_file(temp_dir, self.resources[0], "badurl", "wrongNumber")


@skip_mnist
class TestMNIST:
    def test_mnist_initialization(self):
        """Test MNIST dataset initialization."""
        data, targets = mnist(size=-1, balance=False)
        assert targets.ndim == 1
        assert data.shape == (60000, 28, 28)

    def test_mnist_test_data(self):
        """Test loading the test set."""
        data, targets = mnist(train=False, size=-1, balance=False)
        assert targets.ndim == 1
        assert data.shape == (10000, 28, 28)

    @pytest.mark.parametrize(
        "size, expected",
        [
            (5, (5,)),
            (100, (100,)),
            (1000, (1000,)),
        ],
    )
    def test_mnist_size_data(self, size, expected):
        """Test selecting different sized datasets."""
        _, targets = mnist(size=size, balance=False)
        assert targets.shape == expected

    def test_mnist_oversized(self):
        """Test asking for more data than is available."""
        with pytest.warns(UserWarning):
            _, targets = mnist(train=False, size=15000, balance=False)
        assert targets.shape == (10000,)
        _, targets = mnist(train=False, size=15000, balance=True, verbose=False)
        assert targets.shape == (8920,)
        with pytest.warns(UserWarning):
            _, targets = mnist(train=False, size=9000, balance=False)
        assert targets.shape == (8920,)
        _, targets = mnist(train=False, size=9000, balance=False, verbose=False)
        assert targets.shape == (8920,)

    def test_mnist_normalize(self):
        """Test if unit_interval, normalization, and dtype works properly."""
        data, _ = mnist(unit_normalize=True, normalize=(0.5, 0.5), dtype=np.float32)
        assert np.all((data >= -1) & (data <= 1))
        assert np.min(data) == -1
        assert str(data.dtype) == "float32"

    def test_mnist_flatten(self):
        """Test flattening functionality."""
        data, _ = mnist(flatten=True)
        assert data.shape == (1000, 784)

    @pytest.mark.parametrize(
        "channels, expected",
        [
            ("channels_first", (1000, 1, 28, 28)),
            ("channels_last", (1000, 28, 28, 1)),
        ],
    )
    def test_mnist_channels(self, channels, expected):
        """Test channels_first functionality."""
        data, _ = mnist(channels=channels)
        assert data.shape == expected

    @pytest.mark.parametrize(
        "classes, subset, expected",
        [
            (["zero", "one", "two", "five", "nine"], {0, 1, 2, 5, 9}, 200),
            ([4, 7, 8, 9, 15], {4, 7, 8, 9}, 250),
            ("six", {6}, 1000),
            (3, {3}, 1000),
        ],
    )
    def test_mnist_class_selection(self, classes, subset, expected):
        """Test class selection and equalize functionality."""
        _, targets = mnist(classes=classes, balance=True)
        labels = np.unique(targets)
        counts = Counter(targets)
        assert set(labels).issubset(subset)
        assert counts[labels[0]] == expected  # type: ignore

    # @pytest.mark.parametrize(
    #     "corruption",
    #     [
    #         "identity",
    #         "shot_noise",
    #         "impulse_noise",
    #         "glass_blur",
    #         "motion_blur",
    #         "shear",
    #         "scale",
    #         "rotate",
    #         "brightness",
    #         "translate",
    #         "stripe",
    #         "fog",
    #         "spatter",
    #         "dotted_line",
    #         "zigzag",
    #         "canny_edges",
    #     ],
    # )
    # def test_mnist_corruption(self, corruption):
    #     """Test the loading of all the corruptions."""
    #     data, targets = mnist(size=-1, balance=False, corruption=corruption)
    #     assert targets.ndim == 1
    #     assert data.shape == (60000, 28, 28)

    def test_mnist_slice_back(self):
        """Test the functionality of slicing from the back."""
        dataA, targetsA = mnist(slice_back=True, randomize=True)
        dataB, targetsB = mnist()
        assert np.all(targetsA == targetsB)
        assert ~np.all(dataA == dataB)

    def test_mnist_dataset(self):
        """Test dataset properties."""
        dataset = mnist(return_dataset=True)
        if isinstance(dataset, MNIST):
            assert len(dataset) == 1000
            img, _ = dataset[0]
            assert img.shape == (28, 28)
