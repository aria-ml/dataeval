from __future__ import annotations

__all__ = []

from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

from dataeval.utils.data.datasets._base import (
    BaseClassificationDataset,
    DataLocation,
)

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


class MNIST(BaseClassificationDataset):
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
    size : int, default -1
        Limit the dataset size, must be a value greater than 0.
    classes : "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", \
        int, list, or None, default None
        Option to select specific classes from dataset. Classes are 0-9, any other number is ignored.
    unit_interval : bool, default False
        Shift the data values to the unit interval [0-1].
    dtype : type | None, default None
        Change the :term:`NumPy` dtype - data is loaded as np.uint8
    channels : "channels_first" or "channels_last", default "channels_first"
        Location of channel axis, default is channels first (N, 1, 28, 28)
    flatten : bool, default False
        Flatten data into single dimension (N, 784) - cannot use both channels and flatten.
        If True, channels parameter is ignored.
    normalize : tuple[mean, std] or None, default None
        Normalize images acorrding to provided mean and standard deviation
    corruption : "identity", "shot_noise", "impulse_noise", "glass_blur", "motion_blur", \
        "shear", "scale", "rotate", "brightness", "translate", "stripe" "fog", "spatter", \
        "dotted_line", "zigzag", "canny_edges" or None, default None
        The desired corruption style or None.
    balance : bool, default True
        If True, returns equal number of samples for each class.
    slice_back : bool, default False
        If True and size has a value greater than 0, then grabs selection starting at the last image.
    verbose : bool, default False
        If True, outputs print statements.
    
    Attributes
    ----------
    index2label : dict
        Dictionary which translates from class integers to the associated class strings.
    label2index : dict
        Dictionary which translates from class strings to the associated class integers.
    dataset_dir : Path
        Location of the folder containing the data. Different from `root` if downloading data.
    metadata : dict
        Dictionary containing Dataset metadata, such as `id` which returns the dataset class name.
    class_set : set
        The chosen set of labels to use.
        Default is all 10 classes (0-9) but can be down selected using the `classes` parameter.
    num_classes : int
        The number of classes in `class_set`.
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
    label2index: dict[str, int] = {v: k for k, v in index2label.items()}

    def __init__(
        self,
        root: str | Path,
        download: bool = False,
        image_set: Literal["train", "test", "base"] = "train",
        size: int = -1,
        classes: TMNISTClassMap | None = None,
        unit_interval: bool = False,
        dtype: type | None = None,
        channels: Literal["channels_first", "channels_last"] = "channels_first",
        flatten: bool = False,
        crop: int | None = None,
        normalize: tuple[float, float] | None = None,
        corruption: CorruptionStringMap | None = None,
        balance: bool = True,
        slice_back: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            root,
            download,
            image_set,
            size,
            unit_interval,
            dtype,
            channels,
            flatten,
            crop,
            normalize,
            balance,
            slice_back,
            verbose,
        )

        self.corruption = corruption
        self._filepaths: list[str]
        self._loaded_data: NDArray[np.float64]

        self.class_set: set[int] = self._reduce_classes(classes)
        self.num_classes: int = len(self.class_set)

        if self.corruption == "identity" and verbose:
            print("Identity is not a corrupted dataset but the original MNIST dataset.")

        self._resource: DataLocation = self._resources[0] if self.corruption is None else self._resources[1]

        # Load the data
        self._filepaths, self._labels, self._datum_metadata = self._load_data()
        # Adjust the data as desired
        self._reorder = self._preprocess()

    def _load_data_inner(self) -> tuple[list[str], list[int], dict[str, Any]]:
        """Function to load in the file paths for the data and labels from the correct data format"""
        if self.corruption is None:
            try:
                file_path = self.dataset_dir / self._resource.filename
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
        base_path = self.dataset_dir / corruption
        if self._image_set == "base":
            raw_data = []
            raw_labels = []
            for group in ["train", "test"]:
                file_path = base_path / f"{group}_images.npy"
                raw_data.append(self._grab_corruption_data(file_path))

                label_path = base_path / f"{group}_labels.npy"
                raw_labels.append(self._grab_corruption_data(label_path))

            data = np.concatenate(raw_data, axis=0)
            labels = np.concatenate(raw_labels).astype(np.uintp)
        else:
            file_path = base_path / f"{self._image_set}_images.npy"
            data = self._grab_corruption_data(file_path)
            data = data.astype(np.float64)

            label_path = base_path / f"{self._image_set}_labels.npy"
            labels = self._grab_corruption_data(label_path)
            labels = labels.astype(np.uintp)

        return data, labels

    def _grab_data(self, path: Path) -> tuple[NDArray[Any], NDArray[np.uintp]]:
        """Function to load in the data numpy array"""
        with np.load(path, allow_pickle=True) as data_array:
            if self._image_set == "base":
                data = np.concatenate([data_array["x_train"], data_array["x_test"]], axis=0)
                labels = np.concatenate([data_array["y_train"], data_array["y_test"]], axis=0).astype(np.uintp)
            else:
                data, labels = data_array[f"x_{self._image_set}"], data_array[f"y_{self._image_set}"].astype(np.uintp)
            data = np.expand_dims(data, -1)
        return data, labels

    def _grab_corruption_data(self, path: Path) -> NDArray[Any]:
        """Function to load in the data numpy array for the previously chosen corrupt format"""
        x = np.load(path, allow_pickle=False)
        return x

    def _read_file(self, path: str) -> NDArray[np.float64]:
        """
        Function to grab the correct image from the loaded data.
        Overwrite of the base `_read_file` because data is an all or nothing load.
        """
        index = int(path)
        return self._loaded_data[index]
