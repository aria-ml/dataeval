from __future__ import annotations

__all__ = []

from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from dataeval.utils.data.datasets._base import (
    BaseClassificationDataset,
    DataLocation,
)


class Ships(BaseClassificationDataset):
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
        crop: int | None = None,
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
            crop,
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
