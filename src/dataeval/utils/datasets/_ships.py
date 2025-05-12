from __future__ import annotations

__all__ = []

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from numpy.typing import NDArray

from dataeval.utils.datasets._base import BaseICDataset, DataLocation
from dataeval.utils.datasets._mixin import BaseDatasetNumpyMixin

if TYPE_CHECKING:
    from dataeval.typing import Transform


class Ships(BaseICDataset[NDArray[Any]], BaseDatasetNumpyMixin):
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
        Root directory where the data should be downloaded to or the ``ships`` folder of the already downloaded data.
    transforms : Transform, Sequence[Transform] or None, default None
        Transform(s) to apply to the data.
    download : bool, default False
        If True, downloads the dataset from the internet and puts it in root directory.
        Class checks to see if data is already downloaded to ensure it does not create a duplicate download.
    verbose : bool, default False
        If True, outputs print statements.

    Attributes
    ----------
    path : pathlib.Path
        Location of the folder containing the data.
    image_set : "base"
        The base image set is the only available image set for the Ships dataset.
    index2label : dict[int, str]
        Dictionary which translates from class integers to the associated class strings.
    label2index : dict[str, int]
        Dictionary which translates from class strings to the associated class integers.
    metadata : DatasetMetadata
        Typed dictionary containing dataset metadata, such as `id` which returns the dataset class name.
    transforms : Sequence[Transform]
        The transforms to be applied to the data.
    size : int
        The size of the dataset.

    Note
    ----
    Data License: `CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0/>`_
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

    def __init__(
        self,
        root: str | Path,
        transforms: Transform[NDArray[Any]] | Sequence[Transform[NDArray[Any]]] | None = None,
        download: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            root,
            "base",
            transforms,
            download,
            verbose,
        )
        self._scenes: list[str] = self._load_scenes()
        self._remove_extraneous_json_file()

    def _remove_extraneous_json_file(self) -> None:
        json_path = self.path / "shipsnet.json"
        if json_path.exists():
            json_path.unlink()

    def _load_data_inner(self) -> tuple[list[str], list[int], dict[str, Any]]:
        """Function to load in the file paths for the data and labels"""
        file_data = {"label": [], "scene_id": [], "longitude": [], "latitude": [], "path": []}
        data_folder = sorted((self.path / "shipsnet").glob("*.png"))
        if not data_folder:
            raise FileNotFoundError

        for entry in data_folder:
            # Remove file extension and split by "_"
            parts = entry.stem.split("__")
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
        return sorted(str(entry) for entry in (self.path / "scenes").glob("*.png"))

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
        The scene will be returned with the channel axis first.
        """
        scene = self._read_file(self._scenes[index])
        np.moveaxis(scene, -1, 0)
        return scene
