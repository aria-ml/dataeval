from __future__ import annotations

__all__ = []

from pathlib import Path
from typing import Any, Callable, Literal
from xml.etree.ElementTree import parse

import numpy as np
import torch
from numpy.typing import NDArray
from torchvision.datasets import VOCDetection as _VOCDetection
from torchvision.transforms import v2

from dataeval.utils.data.datasets._base import BaseODDataset, DataLocation
from dataeval.utils.data.datasets._types import (
    DatasetMetadata,
    InfoMixin,
    ObjectDetectionDataset,
    ObjectDetectionTarget,
)


class VOCDetection(ObjectDetectionDataset[torch.Tensor, DatasetMetadata], InfoMixin):
    """
    `Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory of the VOC Dataset.
    year : "2007", "2008", "2009", "2010", "2011" or "2012", default "2012"
        The dataset year.
    image_set : "train", "val", "test", or "base", default "train"
        If "test", then dataset year must be "2007".
        If "base", then the combined dataset of "train" and "val" is returned.
    download : bool, default False
        If true, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again.
    transform : Callable or None, default None:
        A function/transform that takes in a PIL image and returns a transformed version.
        ToImage() and ToDtype(torch.float32, scale=True) are applied by default.
    target_transform : Callable or None, default None:
        A function/transform that takes in the target and transforms it.
    transforms : Callable or None, default None
        A function/transform that takes input sample and its target as entry and returns a transformed version.
    """

    _data: _VOCDetection
    metadata: DatasetMetadata

    def __init__(
        self,
        root: str | Path,
        year: Literal["2007", "2008", "2009", "2010", "2011", "2012"] = "2012",
        image_set: Literal["train", "val", "test", "base"] = "train",
        download: bool = False,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
    ) -> None:
        if transform is None:
            transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self._data = _VOCDetection(root, year, image_set, download, transform, target_transform, transforms)
        self._image_set = image_set
        self.data: list[str]

        # pull out an alphabetized list of labels
        labels: set[str] = set()
        for i in range(len(self._data)):
            objects = self._data.parse_voc_xml(parse(self._data.annotations[i]).getroot())["annotation"]["object"]
            labels.update([o["name"] for o in objects])
            if len(objects) == 20:
                break
        self.classes: list[str] = sorted(labels)

        self.metadata: DatasetMetadata = {
            "id": f"{self.__class__.__name__}_{year}_{image_set}",
            "index2label": {i: self.classes[i] for i in range(len(self.classes))},
            "split": image_set,
        }

    def __str__(self) -> str:
        return str(self._data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ObjectDetectionTarget[torch.Tensor], dict[str, Any]]:
        datum = self._data[index]

        boxes: list[torch.Tensor] = []
        labels: list[int] = []

        for o in datum[1]["annotation"]["object"]:
            bndbox = o["bndbox"]
            box = [float(bndbox["xmin"]), float(bndbox["ymin"]), float(bndbox["xmax"]), float(bndbox["ymax"])]
            boxes.append(torch.tensor(box))
            labels.append(self.classes.index(o["name"]))

        return (
            datum[0],
            ObjectDetectionTarget(torch.stack(boxes), torch.tensor(labels), torch.zeros((len(labels),))),
            datum[1],
        )

    def __len__(self) -> int:
        return len(self._data)


class MILCO(BaseODDataset):
    """
    A side-scan sonar dataset focused on mine (object) detection.


    The dataset comes from the paper
    `Side-scan sonar imaging data of underwater vehicles for mine detection <https://doi.org/10.1016/j.dib.2024.110132>`_
    by N.P. Santos et. al. (2024).

    This class only accesses a portion of the above dataset due to size constraints.
    The full dataset contains 1170 side-scan sonar images collected using a 900-1800 kHz Marine Sonic
    dual frequency side-scan sonar of a Teledyne Marine Gavia Autonomous Underwater Vehicle.
    All the images were carefully analyzed and annotated, including the image coordinates of the
    Bounding Box (BB) of the detected objects divided into NOn-Mine-like BOttom Objects (NOMBO)
    and MIne-Like COntacts (MILCO) classes.

    This dataset is consists of 261 images (120 images from 2015, 93 images from 2017, and 48 images from 2021).
    In these 261 images, there are 315 MILCO objects, and 175 NOMBO objects.
    The class “0” corresponds to a MILCO object and the class “1” corresponds to a NOMBO object.
    The raw BB coordinates provided in the downloaded text files are (x, y, w, h),
    given as percentages of the image (x_BB = x/img_width, y_BB = y/img_height, etc.).
    The images come in 2 sizes, 416 x 416 or 1024 x 1024.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory of dataset where the ``milco`` folder exists.
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
    normalize : tuple[mean, std] or None, default None
        Normalize images acorrding to provided mean and standard deviation
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
        The chosen set of labels to use. This is a binary dataset so there is only 0 ("MILCO") and 1 ("NOMBO").
    num_classes : int
        The number of classes in `class_set`.
    data : list[str]
        Paths to the images
    """

    _resources = [
        DataLocation(
            url="https://figshare.com/ndownloader/files/43169002",
            filename="2015.zip",
            md5=True,
            checksum="93dfbb4fb7987734152c372496b4884c",
        ),
        DataLocation(
            url="https://figshare.com/ndownloader/files/43169005",
            filename="2017.zip",
            md5=True,
            checksum="9c2de230a2bbf654921416bea6fc0f42",
        ),
        DataLocation(
            url="https://figshare.com/ndownloader/files/43168999",
            filename="2021.zip",
            md5=True,
            checksum="b84749b21fa95a4a4c7de3741db78bc7",
        ),
    ]

    index2label: dict[int, str] = {
        0: "MILCO",
        1: "NOMBO",
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
        normalize: tuple[float, float] | None = None,
        # balance: bool = False,
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
            normalize,
            # balance,
            slice_back,
            verbose,
        )

        self.class_set: set[int] = set(self.index2label)
        self.num_classes: int = len(self.class_set)
        self._filepaths: list[str] = []
        self._annotations: list[str] = []
        self._datum_metadata: dict[str, list[Any]] = {}

        # Load the data
        metadata_list = []
        for resource in self._resources:
            self._resource = resource
            result = self._load_data()
            self._filepaths.extend(result[0])
            self._annotations.extend(result[1])
            metadata_list.append(result[2])

        # Adjust datum metadata to correct format
        for data_dict in metadata_list:
            for key, val in data_dict.items():
                if key not in self._datum_metadata:
                    self._datum_metadata[str(key)] = []
                self._datum_metadata[str(key)].extend(val)

    def _load_data_inner(self) -> tuple[list[str], list[str], dict[str, Any]]:
        file_data = {"year": [], "image_id": [], "data_path": [], "label_path": []}
        data_folder = self.dataset_dir / self._resource.filename[:-4]
        for entry in data_folder.iterdir():
            if entry.is_file() and entry.suffix == ".jpg":
                # Remove file extension and split by "_"
                parts = entry.stem.split("_")
                file_data["image_id"].append(parts[0])
                file_data["year"].append(parts[1])
                file_data["data_path"].append(str(entry))
                file_data["label_path"].append(str(entry.parent / entry.stem) + ".txt")
        data = file_data.pop("data_path")
        annotations = file_data.pop("label_path")

        return data, annotations, file_data

    def _read_annotations(self, annotation: str) -> tuple[NDArray[np.float64], NDArray[np.uintp], dict[str, Any]]:
        """Function for extracting the info out of the text files"""
        labels: list[int] = []
        boxes: list[list[float]] = []
        with open(annotation) as f:
            for line in f.readlines():
                out = line.strip().split(" ")
                labels.append(int(out[0]))
                boxes.append([float(out[1]), float(out[2]), float(out[3]), float(out[4])])

        return np.array(boxes, dtype=np.float64), np.array(labels, dtype=np.uintp), {}
