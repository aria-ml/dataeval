from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, VOCDetection  # noqa: F401


def _validate_file(fpath, file_md5, chunk_size=65535):
    hasher = hashlib.md5()
    with open(fpath, "rb") as fpath_file:
        while chunk := fpath_file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest() == file_md5


def _get_file(
    root: str | Path,
    fname: str,
    origin: str,
    file_md5: str | None = None,
    verbose: bool = True,
):
    fname = os.fspath(fname) if isinstance(fname, os.PathLike) else fname
    fpath = os.path.join(root, fname)

    download = False
    if os.path.exists(fpath):
        if file_md5 is not None and not _validate_file(fpath, file_md5):
            download = True
        else:
            if verbose:
                print("Zip already downloaded and verified.")
                print("Extracting...")
    else:
        download = True

    if download:
        try:
            error_msg = "URL fetch failure on {}: {} -- {}"
            try:
                urlretrieve(origin, fpath)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg)) from e
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason)) from e
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

        if os.path.exists(fpath) and file_md5 is not None and not _validate_file(fpath, file_md5):
            raise ValueError(
                "Incomplete or corrupted file detected. "
                f"The md5 file hash does not match the provided value "
                f"of {file_md5}.",
            )
    return fpath


def check_exists(
    folder: str | Path, url: str, root: str | Path, fname: str, md5: str, download: bool = True, verbose: bool = True
):
    if not os.path.exists(folder):
        if download:
            download_dataset(url, root, fname, md5, verbose)
        else:
            raise RuntimeError("Dataset not found. You can use download=True to download it")
    else:
        if verbose:
            print("Files already downloaded and verified")


def download_dataset(url: str, root: str | Path, fname: str, md5: str, verbose: bool = True) -> str:
    """Code to download mnist and corruptions, originates from tensorflow_datasets (tfds):
    https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/mnist_corrupted.py
    """
    name, _ = os.path.splitext(fname)
    folder = os.path.join(root, name)
    os.makedirs(folder, exist_ok=True)

    path = _get_file(
        root,
        fname,
        origin=url + fname,
        file_md5=md5,
        verbose=verbose,
    )
    extract_archive(path, remove_finished=True)
    return path


def extract_archive(
    from_path: str | Path,
    to_path: str | Path | None = None,
    remove_finished: bool = False,
):
    """Extract an archive.

    The archive type and a possible compression is automatically detected from the file name.
    """
    from_path = Path(from_path)
    if not from_path.is_absolute():
        from_path = from_path.resolve()

    if to_path is None:
        to_path = os.path.dirname(from_path)

    # Extracting zip
    with zipfile.ZipFile(from_path, "r", compression=zipfile.ZIP_STORED) as zzip:
        zzip.extractall(to_path)

    if remove_finished:
        os.remove(from_path)


def subselect(arr: NDArray, count: int, from_back: bool = False):
    if from_back:
        return arr[-count:]
    return arr[:count]


class MNIST(Dataset):
    """MNIST Dataset and Corruptions.

    Args:
        root : str | ``pathlib.Path``
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
            Change the numpy dtype - data is loaded as np.uint8
        channels : Literal['channels_first' | 'channels_last'] | None, default None
            Location of channel axis if desired, default has no channels (N, 28, 28)
        flatten : bool, default False
            Flatten data into single dimension (N, 784) - cannot use both channels and flatten,
            channels takes priority over flatten.
        normalize : tuple[mean, std] | None, default None
            Normalize images acorrding to provided mean and standard deviation
        corruption : Literal['identity' | 'shot_noise' | 'impulse_noise' | 'glass_blur' |
            'motion_blur' | 'shear' | 'scale' | 'rotate' | 'brightness' | 'translate' | 'stripe' |
            'fog' | 'spatter' | 'dotted_line' | 'zigzag' | 'canny_edges'] | None, default None
            The desired corruption style or None.
        classes : Literal["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
            | int | list[int] | list[Literal["zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine"]] | None, default None
            Option to select specific classes from dataset.
        balance : bool, default True
            If True, returns equal number of samples for each class.
        randomize : bool, default False
            If True, shuffles the data prior to selection - uses a set seed for reproducibility.
        slice_back : bool, default False
            If True and size has a value greater than 0, then grabs selection starting at the last image.
        verbose : bool, default True
            If True, outputs print statements.
    """

    mirror = "https://zenodo.org/record/3239543/files/"

    resources = ("mnist_c.zip", "4b34b33045869ee6d424616cd3a65da3")

    class_dict = {
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
        corruption: Literal[
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
        | None = None,
        classes: Literal[
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]
        | int
        | list[int]
        | list[
            Literal[
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
            ]
        ]
        | None = None,
        balance: bool = True,
        randomize: bool = False,
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
        self.balance = balance
        self.randomize = randomize
        self.from_back = slice_back

        if corruption is None:
            corruption = "identity"
        elif corruption == "identity" and verbose:
            print("Identity is not a corrupted dataset but the original MNIST dataset")
        self.corruption = corruption

        self.class_set = []
        if classes is not None:
            if not isinstance(classes, list):
                classes = [classes]  # type: ignore

            for val in classes:  # type: ignore
                if isinstance(val, int) and 0 <= val < 10:
                    self.class_set.append(val)
                elif isinstance(val, str):
                    self.class_set.append(self.class_dict[val])
            self.class_set = set(self.class_set)

        if not self.class_set:
            self.class_set = set(self.class_dict.values())

        self.num_classes = len(self.class_set)

        check_exists(self.mnist_folder, self.mirror, self.root, self.resources[0], self.resources[1], download, verbose)

        self.data, self.targets = self._load_data()

        self._augmentations()

    def _load_data(self):
        image_file = f"{'train' if self.train else 'test'}_images.npy"
        data = self._read_file(os.path.join(self.mnist_folder, image_file))

        label_file = f"{'train' if self.train else 'test'}_labels.npy"
        targets = self._read_file(os.path.join(self.mnist_folder, label_file))

        return data, targets

    def _augmentations(self):
        if self.size > self.targets.shape[0]:
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

        if not self.balance and self.num_classes > self.size:
            if self.size > 0:
                self.data = subselect(self.data, self.size, self.from_back)
                self.targets = subselect(self.targets, self.size, self.from_back)
        else:
            label_dict = {label: np.where(self.targets == label)[0] for label in self.class_set}
            min_label_count = min(len(indices) for indices in label_dict.values())

            self.per_class_count = int(np.ceil(self.size / self.num_classes)) if self.size > 0 else min_label_count

            if self.per_class_count > min_label_count:
                self.per_class_count = min_label_count
                if not self.balance:
                    warn(
                        f"Because of dataset limitations, only {min_label_count*self.num_classes} samples "
                        f"will be returned, instead of the desired {self.size}."
                    )

            all_indices = np.empty(shape=(self.num_classes, self.per_class_count), dtype=int)
            for i, label in enumerate(self.class_set):
                all_indices[i] = subselect(label_dict[label], self.per_class_count, self.from_back)
            self.data = np.vstack(self.data[all_indices.T])  # type: ignore
            self.targets = np.hstack(self.targets[all_indices.T])  # type: ignore

        if self.unit_interval:
            self.data = self.data / 255

        if self.normalize:
            self.data = (self.data - self.normalize[0]) / self.normalize[1]

        if self.dtype:
            self.data = self.data.astype(self.dtype)

        if self.channels == "channels_first":
            self.data = np.moveaxis(self.data, -1, 1)
        elif self.channels is None:
            self.data = self.data[:, :, :, 0]

        if self.flatten and self.channels is None:
            self.data = self.data.reshape(self.data.shape[0], -1)

    def __getitem__(self, index: int) -> tuple[NDArray, int]:
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
        return os.path.join(self.root, "mnist_c", self.corruption)

    def _read_file(self, path: str) -> NDArray:
        x = np.load(path, allow_pickle=False)
        return x
