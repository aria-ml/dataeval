"""Provides access to common Computer Vision datasets."""

from __future__ import annotations

__all__ = ["MNIST", "CIFAR10", "VOCDetection"]

import hashlib
import os
import zipfile
from pathlib import Path
from typing import Any, Callable, Literal, Sized, TypeVar
from warnings import warn

import numpy as np
import requests
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10 as _CIFAR10
from torchvision.datasets import VOCDetection as _VOCDetection

TDatum = TypeVar("TDatum")


class DatasetWrapperMixin(Dataset[TDatum]):
    _data: Dataset[TDatum]

    def __getitem__(self, index: int) -> TDatum:
        return self._data.__getitem__(index)

    def __len__(self) -> int:
        if isinstance(self._data, Sized):
            return len(self._data)

        raise NotImplementedError("Dataset does not have a length function.")


class InfoMixin:
    _image_set: str

    def info(self) -> str:
        """Pretty prints dataset name and info"""

        return f"{self._image_set.capitalize()}\n{'-' * len(self._image_set)}\n{self}\n"


class CIFAR10(DatasetWrapperMixin[tuple[Tensor, int]], InfoMixin):
    """
    `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory of the CIFAR10 Dataset.
    train : bool, default True
        If True, creates dataset from training set, otherwise creates from test set.
    download : bool, default False
        If true, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again.
    transform : Callable or None, default None:
        A function/transform that takes in a PIL image and returns a transformed version.
        E.g, ``torchvision.transforms.RandomCrop``
    target_transform : Callable or None, default None:
        A function/transform that takes in the target and transforms it.
    """

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        download: bool = False,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        self._data = _CIFAR10(root, train, transform, target_transform, download)
        self._image_set = "train" if train else "test"


class VOCDetection(DatasetWrapperMixin[tuple[Tensor, dict[str, Any]]], InfoMixin):
    """
    `Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory of the VOC Dataset.
    year : "2007", "2008", "2009", "2010", "2011" or "2012", default "2012"
        The dataset year.
    image_set : "train", "trainval", "val", or "test", default "train"
        "test" is only valid for the year "2007"
    download : bool, default False
        If true, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again.
    transform : Callable or None, default None:
        A function/transform that takes in a PIL image and returns a transformed version.
        E.g, ``torchvision.transforms.RandomCrop``
    target_transform : Callable or None, default None:
        A function/transform that takes in the target and transforms it.
    transforms : Callable or None, default None
        A function/transform that takes input sample and its target as entry and returns a transformed version.
    """

    def __init__(
        self,
        root: str | Path,
        year: Literal["2007", "2008", "2009", "2010", "2011", "2012"] = "2012",
        image_set: Literal["train", "trainval", "val", "test"] = "train",
        download: bool = False,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
    ) -> None:
        self._data = _VOCDetection(root, year, image_set, download, transform, target_transform, transforms)
        self._image_set = image_set


ClassStringMap = Literal["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TClassMap = TypeVar("TClassMap", ClassStringMap, int, list[ClassStringMap], list[int])
CorruptionStringMap = Literal[
    "identity",
    "shot_noise",
    "impulse_noise",
    "glass_blur",
    "motion_blur",
    "shear",
    "scale",
    "rotate",
    "brightness",
    "translate",
    "stripe",
    "fog",
    "spatter",
    "dotted_line",
    "zigzag",
    "canny_edges",
]


def _validate_file(fpath, file_md5, md5=False, chunk_size=65535):
    hasher = hashlib.md5() if md5 else hashlib.sha256()
    with open(fpath, "rb") as fpath_file:
        while chunk := fpath_file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest() == file_md5


def _get_file(
    root: str | Path,
    fname: str,
    origin: str,
    file_hash: str | None = None,
    verbose: bool = True,
    md5: bool = False,
    timeout: int = 60,
):
    fpath = os.path.join(root, fname)
    download = True
    if os.path.exists(fpath) and file_hash is not None and _validate_file(fpath, file_hash, md5):
        download = False
        if verbose:
            print("File already downloaded and verified.")
            if md5:
                print("Extracting zip file...")

    if download:
        try:
            error_msg = "URL fetch failure on {}: {} -- {}"
            try:
                with requests.get(origin, stream=True, timeout=timeout) as r:
                    r.raise_for_status()
                    with open(fpath, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
            except requests.exceptions.HTTPError as e:
                raise RuntimeError(f"{error_msg.format(origin, e.response.status_code, e.response.reason)}") from e
            except requests.exceptions.RequestException as e:
                raise ValueError(f"{error_msg.format(origin, 'Unknown error', str(e))}") from e
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

        if os.path.exists(fpath) and file_hash is not None and not _validate_file(fpath, file_hash, md5):
            raise ValueError(
                "Incomplete or corrupted file detected. "
                f"The file hash does not match the provided value "
                f"of {file_hash}.",
            )

    return fpath


def _check_exists(
    folder: str | Path,
    url: str,
    root: str | Path,
    fname: str,
    file_hash: str,
    download: bool = True,
    verbose: bool = True,
    md5: bool = False,
):
    """Determine if the dataset has already been downloaded."""
    location = str(folder)
    if not os.path.exists(folder):
        if download:
            location = _download_dataset(url, root, fname, file_hash, verbose, md5)
        else:
            raise RuntimeError("Dataset not found. You can use download=True to download it")
    else:
        if verbose:
            print("Files already downloaded and verified")
    return location


def _download_dataset(
    url: str, root: str | Path, fname: str, file_hash: str, verbose: bool = True, md5: bool = False
) -> str:
    """Code to download mnist and corruptions, originates from tensorflow_datasets (tfds):
    https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/mnist_corrupted.py
    """
    name, _ = os.path.splitext(fname)
    folder = os.path.join(root, name)
    os.makedirs(folder, exist_ok=True)

    fpath = _get_file(
        folder,
        fname,
        origin=url + fname,
        file_hash=file_hash,
        verbose=verbose,
        md5=md5,
    )
    if md5:
        folder = _extract_archive(fpath, root, remove_finished=True)
    return folder


def _extract_archive(
    from_path: str | Path,
    to_path: str | Path | None = None,
    remove_finished: bool = False,
) -> str:
    """Extract an archive.

    The archive type and a possible compression is automatically detected from the file name.
    """
    from_path = Path(from_path)
    if not from_path.is_absolute():
        from_path = from_path.resolve()

    if to_path is None or not os.path.exists(to_path):
        to_path = os.path.dirname(from_path)
    to_path = Path(to_path)
    if not to_path.is_absolute():
        to_path = to_path.resolve()

    # Extracting zip
    with zipfile.ZipFile(from_path, "r", compression=zipfile.ZIP_STORED) as zzip:
        zzip.extractall(to_path)

    if remove_finished:
        os.remove(from_path)
    return str(to_path)


def _subselect(arr: NDArray, count: int, from_back: bool = False):
    if from_back:
        return arr[-count:]
    return arr[:count]


class MNIST(Dataset[tuple[NDArray[np.float64], int]], InfoMixin):
    """MNIST Dataset and Corruptions.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory of dataset where the ``mnist_c/`` folder exists.
    train : bool, default True
        If True, creates dataset from ``train_images.npy`` and ``train_labels.npy``.
    download : bool, default False
        If True, downloads the dataset from the internet and puts it in root
        directory. If dataset is already downloaded, it is not downloaded again.
    size : int, default -1
        Limit the dataset size, must be a value greater than 0.
    unit_interval : bool, default False
        Shift the data values to the unit interval [0-1].
    dtype : type | None, default None
        Change the :term:`NumPy` dtype - data is loaded as np.uint8
    channels : "channels_first", "channels_last" or None, default None
        Location of channel axis if desired, default has no channels (N, 28, 28)
    flatten : bool, default False
        Flatten data into single dimension (N, 784) - cannot use both channels and flatten,
        channels takes priority over flatten.
    normalize : tuple[mean, std] or None, default None
        Normalize images acorrding to provided mean and standard deviation
    corruption : "identity", "shot_noise", "impulse_noise", "glass_blur", "motion_blur", \
        "shear", "scale", "rotate", "brightness", "translate", "stripe" "fog", "spatter", \
        "dotted_line", "zigzag", "canny_edges" or None, default None
        The desired corruption style or None.
    classes : "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", \
        int, list, or None, default None
        Option to select specific classes from dataset.
    balance : bool, default True
        If True, returns equal number of samples for each class.
    randomize : bool, default True
        If True, shuffles the data prior to selection - uses a set seed for reproducibility.
    slice_back : bool, default False
        If True and size has a value greater than 0, then grabs selection starting at the last image.
    verbose : bool, default True
        If True, outputs print statements.
    """

    _mirrors: tuple[str, ...] = (
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/",
        "https://zenodo.org/record/3239543/files/",
    )

    _resources: tuple[tuple[str, str], ...] = (
        ("mnist.npz", "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"),
        ("mnist_c.zip", "4b34b33045869ee6d424616cd3a65da3"),
    )

    class_dict: dict[str, int] = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
    }

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        download: bool = False,
        size: int = -1,
        unit_interval: bool = False,
        dtype: type | None = None,
        channels: Literal["channels_first", "channels_last"] | None = None,
        flatten: bool = False,
        normalize: tuple[float, float] | None = None,
        corruption: CorruptionStringMap | None = None,
        classes: TClassMap | None = None,
        balance: bool = True,
        randomize: bool = True,
        slice_back: bool = False,
        verbose: bool = True,
    ) -> None:
        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root  # location of stored dataset
        self.train = train  # training set or test set
        self.size = size
        self.unit_interval = unit_interval
        self.dtype = dtype
        self.channels = channels
        self.flatten = flatten
        self.normalize = normalize
        self.corruption = corruption
        self.balance = balance
        self.randomize = randomize
        self.from_back = slice_back
        self.verbose = verbose
        self.data: NDArray[np.float64]
        self.targets: NDArray[np.int_]
        self.size: int
        self._image_set = "train" if train else "test"

        self._class_set = []
        if classes is not None:
            if not isinstance(classes, list):
                classes = [classes]  # type: ignore

            for val in classes:  # type: ignore
                if isinstance(val, int) and 0 <= val < 10:
                    self._class_set.append(val)
                elif isinstance(val, str):
                    self._class_set.append(self.class_dict[val])
            self._class_set = set(self._class_set)

        if not self._class_set:
            self._class_set = set(self.class_dict.values())

        self._num_classes = len(self._class_set)

        if self.corruption is None:
            file_resource = self._resources[0]
            mirror = self._mirrors[0]
            md5 = False
        else:
            if self.corruption == "identity" and verbose:
                print("Identity is not a corrupted dataset but the original MNIST dataset.")
            file_resource = self._resources[1]
            mirror = self._mirrors[1]
            md5 = True
        _check_exists(self.mnist_folder, mirror, self.root, file_resource[0], file_resource[1], download, verbose, md5)

        self.data, self.targets = self._load_data()

        self._augmentations()

    def _load_data(self) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        if self.corruption is None:
            image_file = self._resources[0][0]
            data, targets = self._read_normal_file(os.path.join(self.mnist_folder, image_file))
        else:
            image_file = f"{'train' if self.train else 'test'}_images.npy"
            data = self._read_corrupt_file(os.path.join(self.mnist_folder, image_file))
            data = data.squeeze()

            label_file = f"{'train' if self.train else 'test'}_labels.npy"
            targets = self._read_corrupt_file(os.path.join(self.mnist_folder, label_file))

        return data, targets

    def _augmentations(self):
        if self.size > self.targets.shape[0] and self.verbose:
            warn(
                f"Asked for more samples, {self.size}, than the raw dataset contains, {self.targets.shape[0]}. "
                "Adjusting down to raw dataset size."
            )
            self.size = -1

        if self.randomize:
            rdm_seed = np.random.default_rng(2023)
            shuffled_indices = rdm_seed.permutation(self.data.shape[0])
            self.data = self.data[shuffled_indices]
            self.targets = self.targets[shuffled_indices]

        if not self.balance and self._num_classes > self.size:
            if self.size > 0:
                self.data = _subselect(self.data, self.size, self.from_back)
                self.targets = _subselect(self.targets, self.size, self.from_back)
        else:
            label_dict = {label: np.where(self.targets == label)[0] for label in self._class_set}
            min_label_count = min(len(indices) for indices in label_dict.values())

            self._per_class_count = int(np.ceil(self.size / self._num_classes)) if self.size > 0 else min_label_count

            if self._per_class_count > min_label_count:
                self._per_class_count = min_label_count
                if not self.balance and self.verbose:
                    warn(
                        f"Because of dataset limitations, only {min_label_count*self._num_classes} samples "
                        f"will be returned, instead of the desired {self.size}."
                    )

            all_indices: NDArray[np.int_] = np.empty(shape=(self._num_classes, self._per_class_count), dtype=np.int_)
            for i, label in enumerate(self._class_set):
                all_indices[i] = _subselect(label_dict[label], self._per_class_count, self.from_back)
            self.data = np.vstack(self.data[all_indices.T])  # type: ignore
            self.targets = np.hstack(self.targets[all_indices.T])  # type: ignore

        if self.unit_interval:
            self.data = self.data / 255

        if self.normalize:
            self.data = (self.data - self.normalize[0]) / self.normalize[1]

        if self.dtype:
            self.data = self.data.astype(self.dtype)

        if self.channels == "channels_first":
            self.data = self.data[:, np.newaxis, :, :]
        elif self.channels == "channels_last":
            self.data = self.data[:, :, :, np.newaxis]

        if self.flatten and self.channels is None:
            self.data = self.data.reshape(self.data.shape[0], -1)

    def __getitem__(self, index: int) -> tuple[NDArray[np.float64], int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def mnist_folder(self) -> str:
        if self.corruption is None:
            return os.path.join(self.root, "mnist")
        return os.path.join(self.root, "mnist_c", self.corruption)

    def _read_normal_file(self, path: str) -> tuple[NDArray, NDArray]:
        with np.load(path, allow_pickle=True) as f:
            if self.train:
                x, y = f["x_train"], f["y_train"]
            else:
                x, y = f["x_test"], f["y_test"]
        return x, y

    def _read_corrupt_file(self, path: str) -> NDArray:
        x = np.load(path, allow_pickle=False)
        return x
