from __future__ import annotations

__all__ = []

from pathlib import Path
from typing import Any, Literal, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray

from dataeval.utils.data._types import Transform
from dataeval.utils.data.datasets._base import BaseICDataset, DataLocation
from dataeval.utils.data.datasets._mixin import BaseDatasetNumpyMixin

MNISTClassStringMap = Literal["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TMNISTClassMap = TypeVar("TMNISTClassMap", MNISTClassStringMap, int, list[MNISTClassStringMap], list[int])
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


class MNIST(BaseICDataset[NDArray[Any]], BaseDatasetNumpyMixin):
    """`MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_ Dataset and `Corruptions <https://arxiv.org/abs/1906.02337>`_.

    There are 15 different styles of corruptions. This class downloads differently depending on if you
    need just the original dataset or if you need corruptions. If you need both a corrupt version and the
    original version then choose `corruption="identity"` as this downloads all of the corrupt datasets and
    provides the original as `identity`. If you just need the original, then using `corruption=None` will
    download only the original dataset to save time and space.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory of dataset where the ``mnist`` folder exists.
    download : bool, default False
        If True, downloads the dataset from the internet and puts it in root directory.
        Class checks to see if data is already downloaded to ensure it does not create a duplicate download.
    image_set : "train", "test" or "base", default "train"
        If "base", returns all of the data to allow the user to create their own splits.
    verbose : bool, default False
        If True, outputs print statements.

    Attributes
    ----------
    index2label : dict
        Dictionary which translates from class integers to the associated class strings.
    label2index : dict
        Dictionary which translates from class strings to the associated class integers.
    path : Path
        Location of the folder containing the data.
    metadata : dict
        Dictionary containing Dataset metadata, such as `id` which returns the dataset class name.
    """

    _resources = [
        DataLocation(
            url="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            filename="mnist.npz",
            md5=False,
            checksum="731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1",
        ),
        DataLocation(
            url="https://zenodo.org/record/3239543/files/mnist_c.zip",
            filename="mnist_c.zip",
            md5=True,
            checksum="4b34b33045869ee6d424616cd3a65da3",
        ),
    ]

    index2label: dict[int, str] = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
    }

    def __init__(
        self,
        root: str | Path,
        download: bool = False,
        image_set: Literal["train", "test", "base"] = "train",
        corruption: CorruptionStringMap | None = None,
        transforms: Transform[NDArray[Any]] | Sequence[Transform[NDArray[Any]]] | None = None,
        verbose: bool = False,
    ) -> None:
        self.corruption = corruption
        if self.corruption == "identity" and verbose:
            print("Identity is not a corrupted dataset but the original MNIST dataset.")
        self._resource_index = 0 if self.corruption is None else 1

        super().__init__(
            root,
            download,
            image_set,
            transforms,
            verbose,
        )

    def _load_data_inner(self) -> tuple[list[str], list[int], dict[str, Any]]:
        """Function to load in the file paths for the data and labels from the correct data format"""
        if self.corruption is None:
            try:
                file_path = self.path / self._resource.filename
                self._loaded_data, labels = self._grab_data(file_path)
            except FileNotFoundError:
                self._loaded_data, labels = self._load_corruption()
        else:
            self._loaded_data, labels = self._load_corruption()

        index_strings = np.arange(self._loaded_data.shape[0]).astype(str).tolist()
        return index_strings, labels.tolist(), {}

    def _load_corruption(self) -> tuple[NDArray[Any], NDArray[np.uintp]]:
        """Function to load in the file paths for the data and labels for the different corrupt data formats"""
        corruption = self.corruption if self.corruption is not None else "identity"
        base_path = self.path / corruption
        if self.image_set == "base":
            raw_data = []
            raw_labels = []
            for group in ["train", "test"]:
                file_path = base_path / f"{group}_images.npy"
                raw_data.append(self._grab_corruption_data(file_path))

                label_path = base_path / f"{group}_labels.npy"
                raw_labels.append(self._grab_corruption_data(label_path))

            data = np.concatenate(raw_data, axis=0).transpose(0, 3, 1, 2)
            labels = np.concatenate(raw_labels).astype(np.uintp)
        else:
            file_path = base_path / f"{self.image_set}_images.npy"
            data = self._grab_corruption_data(file_path)
            data = data.astype(np.float64).transpose(0, 3, 1, 2)

            label_path = base_path / f"{self.image_set}_labels.npy"
            labels = self._grab_corruption_data(label_path)
            labels = labels.astype(np.uintp)

        return data, labels

    def _grab_data(self, path: Path) -> tuple[NDArray[Any], NDArray[np.uintp]]:
        """Function to load in the data numpy array"""
        with np.load(path, allow_pickle=True) as data_array:
            if self.image_set == "base":
                data = np.concatenate([data_array["x_train"], data_array["x_test"]], axis=0)
                labels = np.concatenate([data_array["y_train"], data_array["y_test"]], axis=0).astype(np.uintp)
            else:
                data, labels = data_array[f"x_{self.image_set}"], data_array[f"y_{self.image_set}"].astype(np.uintp)
            data = np.expand_dims(data, axis=1)
        return data, labels

    def _grab_corruption_data(self, path: Path) -> NDArray[Any]:
        """Function to load in the data numpy array for the previously chosen corrupt format"""
        x = np.load(path, allow_pickle=False)
        return x

    def _read_file(self, path: str) -> NDArray[Any]:
        """
        Function to grab the correct image from the loaded data.
        Overwrite of the base `_read_file` because data is an all or nothing load.
        """
        index = int(path)
        return self._loaded_data[index]
