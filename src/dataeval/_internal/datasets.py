from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

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
):
    fname = os.fspath(fname) if isinstance(fname, os.PathLike) else fname
    fpath = os.path.join(root, fname)

    download = False
    if os.path.exists(fpath):
        if file_md5 is not None and not _validate_file(fpath, file_md5):
            download = True
        else:
            print("Files already downloaded and verified")
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


def download_dataset(url: str, root: str | Path, fname: str, md5: str) -> str:
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
    """

    mirror = "https://zenodo.org/record/3239543/files/"

    resources = ("mnist_c.zip", "4b34b33045869ee6d424616cd3a65da3")

    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    @property
    def train_labels(self):
        return self.targets

    @property
    def test_labels(self):
        return self.targets

    @property
    def train_data(self):
        return self.data

    @property
    def test_data(self):
        return self.data

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

        if corruption is None:
            corruption = "identity"
        elif corruption == "identity":
            print("Identity is not a corrupted dataset but the original MNIST dataset")
        self.corruption = corruption

        if os.path.exists(self.mnist_folder):
            print("Files already downloaded and verified")
        elif download:
            download_dataset(self.mirror, self.root, self.resources[0], self.resources[1])
        else:
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()

    def _load_data(self):
        image_file = f"{'train' if self.train else 'test'}_images.npy"
        data = self._read_image_file(os.path.join(self.mnist_folder, image_file))

        label_file = f"{'train' if self.train else 'test'}_labels.npy"
        targets = self._read_label_file(os.path.join(self.mnist_folder, label_file))

        if self.size >= 1 and self.size >= len(self.classes):
            final_data = []
            final_targets = []
            for label in range(len(self.classes)):
                indices = np.where(targets == label)[0]
                selected_indices = indices[: int(self.size / len(self.classes))]
                final_data.append(data[selected_indices])
                final_targets.append(targets[selected_indices])
            data = np.concatenate(final_data)
            targets = np.concatenate(final_targets)
            shuffled_indices = np.random.permutation(data.shape[0])
            data = data[shuffled_indices]
            targets = targets[shuffled_indices]
        elif self.size >= 1:
            data = data[: self.size]
            targets = targets[: self.size]

        if self.unit_interval:
            data = data / 255

        if self.normalize:
            data = (data - self.normalize[0]) / self.normalize[1]

        if self.dtype:
            data = data.astype(self.dtype)

        if self.channels == "channels_first":
            data = np.moveaxis(data, -1, 1)
        elif self.channels is None:
            data = data[:, :, :, 0]

        if self.flatten and self.channels is None:
            data = data.reshape(data.shape[0], -1)

        return data, targets

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

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _read_label_file(self, path: str) -> NDArray:
        x = np.load(path, allow_pickle=False)
        return x

    def _read_image_file(self, path: str) -> NDArray:
        x = np.load(path, allow_pickle=False)
        return x
