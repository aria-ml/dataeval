from __future__ import annotations

__all__ = []

from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn.functional import one_hot
from torchvision.datasets import CIFAR10 as _CIFAR10
from torchvision.transforms import v2

from dataeval.utils.data.datasets._base import (
    BaseClassificationDataset,
    CorruptionStringMap,
    DataLocation,
    TMNISTClassMap,
)
from dataeval.utils.data.datasets._types import DatasetMetadata, ImageClassificationDataset, InfoMixin


class CIFAR10(ImageClassificationDataset[torch.Tensor, DatasetMetadata], InfoMixin):
    """
    `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset as Torch tensors.

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
        ToImage() and ToDtype(torch.float32, scale=True) are applied by default.
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
        if transform is None:
            transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        self._data = _CIFAR10(root, train, transform, target_transform, download)
        self._image_set = "train" if train else "test"

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        item: tuple[torch.Tensor, int] = self._data[index]
        target = one_hot(torch.tensor(item[1]), len(self._data.class_to_idx))
        return item[0], target, {}

    def __len__(self) -> int:
        return len(self._data)


class MNIST(BaseClassificationDataset, InfoMixin):
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


class ShipDataset(BaseClassificationDataset):
    """
    A dataset that focuses on identifying ships from satellite images.

    The dataset comes from kaggle,
    `Ships in Satellite Imagery <https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery>`_.
    The images come from Planet satellite imagery when they gave
    `open-access to a portion of their data <https://www.planet.com/pulse/open-california-rapideye-data/>`_.

    There are 4000 80x80x3 (HWC) images of ships, sea, and land.
    There are also 8 larger scene images similar to what would be operationally provided.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory of dataset where the ``shipdataset`` folder exists.
    download : bool, default False
        If True, downloads the dataset from the internet and puts it in root directory.
        Class checks to see if data is already downloaded to ensure it does not create a duplicate download.
    size : int, default -1
        Limit the dataset size, must be a value greater than 0.
    unit_interval : bool, default False
        Shift the data values to the unit interval [0-1].
    dtype : type | None, default None
        If None, data is loaded as np.uint8.
        Otherwise specify the desired :term:`NumPy` dtype.
    channels : "channels_first" or "channels_last", default channels_first
        Location of channel axis if desired, default is downloaded image which contains channels last
    flatten : bool, default False
        Flatten data into single dimension (N, 19200) - cannot use both channels and flatten.
        If True, channels parameter is ignored.
    normalize : tuple[mean, std] or None, default None
        Normalize images acorrding to provided mean and standard deviation
    balance : bool, default False
        If True, limits the data to equal number of samples for each class (1000 samples per class).
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
        The chosen set of labels to use. This is a binary dataset so there is only 0 ("no ship") and 1 ("ship").
    num_classes : int
        The number of classes in `class_set`.
    scenes : list[str]
        Path to extra data samples that are large satellite images encompassing an entire scene.
        Useful for testing models and pipelines on "real data".
    """

    _resources = [
        DataLocation(
            url="https://zenodo.org/record/3611230/files/ships-in-satellite-imagery.zip",
            filename="ships-in-satellite-imagery.zip",
            md5=True,
            checksum="b2e8a41ed029592b373bd72ee4b89f32",
        ),
    ]

    index2label: dict[int, str] = {
        0: "no ship",
        1: "ship",
    }
    label2index: dict[str, int] = {v: k for k, v in index2label.items()}

    def __init__(
        self,
        root: str | Path,
        download: bool = False,
        size: int = -1,
        unit_interval: bool = False,
        dtype: type | None = None,
        channels: Literal["channels_first", "channels_last"] = "channels_first",
        flatten: bool = False,
        normalize: tuple[float, float] | None = None,
        balance: bool = False,
        slice_back: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            root,
            download,
            "base",
            size,
            unit_interval,
            dtype,
            channels,
            flatten,
            normalize,
            balance,
            slice_back,
            verbose,
        )

        self.class_set: set[int] = set(self.index2label)
        self.num_classes: int = len(self.class_set)
        self._resource: DataLocation = self._resources[0]
        self._filepaths: list[str]
        self.scenes: list[str]

        # Load the data
        self._filepaths, self._labels, self._datum_metadata = self._load_data()
        self.scenes = self._load_scenes()
        # Adjust the data as desired
        self._reorder = self._preprocess()

    def _load_data_inner(self) -> tuple[list[str], list[int], dict[str, Any]]:
        """Function to load in the file paths for the data and labels"""
        file_data = {"label": [], "scene_id": [], "longitude": [], "latitude": [], "path": []}
        data_folder = self.dataset_dir / "shipsnet"
        for entry in data_folder.iterdir():
            # Remove file extension and split by "_"
            parts = entry.stem.split("__")  # Removes ".png" and splits the string
            file_data["label"].append(int(parts[0]))
            file_data["scene_id"].append(parts[1])
            lat_lon = parts[2].split("_")
            file_data["longitude"].append(float(lat_lon[0]))
            file_data["latitude"].append(float(lat_lon[1]))
            file_data["path"].append(entry)
        data = file_data.pop("path")
        labels = file_data.pop("label")
        return data, labels, file_data

    def _load_scenes(self) -> list[str]:
        """Function to load in the file paths for the scene images"""
        data_folder = self.dataset_dir / "scenes"
        scene = [str(entry) for entry in data_folder.iterdir()]
        return scene

    def get_scene(self, index: int) -> NDArray[np.uintp]:
        """
        Get the desired satellite image (scene) by passing in the index of the desired file.

        Args
        ----
        index : int
            Value of the desired data point

        Returns
        -------
        NDArray[np.uintp]
            Scene image

        Note
        ----
        The scene will be returned with the channel axis in the position specified
        by the class channels parameter (default is channels first).
        """
        scene = self._read_file(self.scenes[index])
        return scene if self.channels == "channels_last" else np.moveaxis(scene, -1, 0)
