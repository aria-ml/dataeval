from __future__ import annotations

__all__ = []

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

from defusedxml.ElementTree import parse
from numpy.typing import NDArray

from dataeval.utils.datasets._base import BaseODDataset, DataLocation
from dataeval.utils.datasets._mixin import BaseDatasetNumpyMixin

if TYPE_CHECKING:
    from dataeval.typing import Transform


class AntiUAVDetection(BaseODDataset[NDArray[Any]], BaseDatasetNumpyMixin):
    """
    A UAV detection dataset focused on detecting UAVs in natural images against large variation in backgrounds.

    The dataset comes from the paper
    `Vision-based Anti-UAV Detection and Tracking <https://ieeexplore.ieee.org/document/9785379>`_
    by Jie Zhao et. al. (2022).

    The dataset is approximately 1.3 GB and can be found `here <https://github.com/wangdongdut/DUT-Anti-UAV>`_.
    Images are collected against a variety of different backgrounds with a variety in the number and type of UAV.
    Ground truth labels are provided for the train, validation and test set.
    There are 35 different types of drones along with a variety in lighting conditions and weather conditions.

    There are 10,000 images: 5200 images in the training set, 2200 images in the validation set,
    and 2600 images in the test set.
    The dataset only has a single UAV class with the focus being on identifying object location in the image.
    Ground-truth bounding boxes are provided in (x0, y0, x1, y1) format.
    The images come in a variety of sizes from 3744 x 5616 to 160 x 240.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory where the data should be downloaded to or
        the ``antiuavdetection`` folder of the already downloaded data.
    image_set: "train", "val", "test", or "base", default "train"
        If "base", then the full dataset is selected (train, val and test).
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
    image_set : "train", "val", "test", or "base"
        The selected image set from the dataset.
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
    Data License: `Apache 2.0 <https://www.apache.org/licenses/LICENSE-2.0.txt>`_
    """

    # Need to run the sha256 on the files and then store that
    _resources = [
        DataLocation(
            url="https://drive.usercontent.google.com/download?id=1RVsSGPUKTdmoyoPTBTWwroyulLek1eTj&export=download&authuser=0&confirm=t&uuid=6bca4f94-a242-4bc2-9663-fb03cd94ef2c&at=APcmpox0--NroQ_3bqeTFaJxP7Pw%3A1746552902927",
            filename="train.zip",
            md5=False,
            checksum="14f927290556df60e23cedfa80dffc10dc21e4a3b6843e150cfc49644376eece",
        ),
        DataLocation(
            url="https://drive.usercontent.google.com/download?id=1333uEQfGuqTKslRkkeLSCxylh6AQ0X6n&export=download&authuser=0&confirm=t&uuid=c2ad2f01-aca8-4a85-96bb-b8ef6e40feea&at=APcmpozY-8bhk3nZSFaYbE8rq1Fi%3A1746551543297",
            filename="val.zip",
            md5=False,
            checksum="238be0ceb3e7c5be6711ee3247e49df2750d52f91f54f5366c68bebac112ebf8",
        ),
        DataLocation(
            url="https://drive.usercontent.google.com/download?id=1L1zeW1EMDLlXHClSDcCjl3rs_A6sVai0&export=download&authuser=0&confirm=t&uuid=5a1d7650-d8cd-4461-8354-7daf7292f06c&at=APcmpozLQC1CuP-n5_UX2JnP53Zo%3A1746551676177",
            filename="test.zip",
            md5=False,
            checksum="a671989a01cff98c684aeb084e59b86f4152c50499d86152eb970a9fc7fb1cbe",
        ),
    ]

    index2label: dict[int, str] = {
        0: "unknown",
        1: "UAV",
    }

    def __init__(
        self,
        root: str | Path,
        image_set: Literal["train", "val", "test", "base"] = "train",
        transforms: Transform[NDArray[Any]] | Sequence[Transform[NDArray[Any]]] | None = None,
        download: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            root,
            image_set,
            transforms,
            download,
            verbose,
        )

    def _load_data(self) -> tuple[list[str], list[str], dict[str, list[Any]]]:
        filepaths: list[str] = []
        targets: list[str] = []
        datum_metadata: dict[str, list[Any]] = {}

        # If base, load all resources
        if self.image_set == "base":
            metadata_list: list[dict[str, Any]] = []

            for resource in self._resources:
                self._resource = resource
                resource_filepaths, resource_targets, resource_metadata = super()._load_data()
                filepaths.extend(resource_filepaths)
                targets.extend(resource_targets)
                metadata_list.append(resource_metadata)

            # Combine metadata
            for data_dict in metadata_list:
                for key, val in data_dict.items():
                    str_key = str(key)  # Ensure key is string
                    if str_key not in datum_metadata:
                        datum_metadata[str_key] = []
                    datum_metadata[str_key].extend(val)

        else:
            # Grab only the desired data
            for resource in self._resources:
                if self.image_set in resource.filename:
                    self._resource = resource
                    resource_filepaths, resource_targets, resource_metadata = super()._load_data()
                    filepaths.extend(resource_filepaths)
                    targets.extend(resource_targets)
                    datum_metadata.update(resource_metadata)

        return filepaths, targets, datum_metadata

    def _load_data_inner(self) -> tuple[list[str], list[str], dict[str, Any]]:
        resource_name = self._resource.filename[:-4]
        base_dir = self.path / resource_name
        data_folder = sorted((base_dir / "img").glob("*.jpg"))
        if not data_folder:
            raise FileNotFoundError

        file_data = {"image_id": [f"{resource_name}_{entry.name}" for entry in data_folder]}
        data = [str(entry) for entry in data_folder]
        annotations = sorted(str(entry) for entry in (base_dir / "xml").glob("*.xml"))

        return data, annotations, file_data

    def _read_annotations(self, annotation: str) -> tuple[list[list[float]], list[int], dict[str, Any]]:
        """Function for extracting the info for the label and boxes"""
        boxes: list[list[float]] = []
        labels = []
        root = parse(annotation).getroot()
        if root is None:
            raise ValueError(f"Unable to parse {annotation}")
        additional_meta: dict[str, Any] = {
            "image_width": int(root.findtext("size/width", default="-1")),
            "image_height": int(root.findtext("size/height", default="-1")),
            "image_depth": int(root.findtext("size/depth", default="-1")),
        }
        for obj in root.findall("object"):
            labels.append(1 if obj.findtext("name", default="") == "UAV" else 0)
            boxes.append(
                [
                    float(obj.findtext("bndbox/xmin", default="0")),
                    float(obj.findtext("bndbox/ymin", default="0")),
                    float(obj.findtext("bndbox/xmax", default="0")),
                    float(obj.findtext("bndbox/ymax", default="0")),
                ]
            )

        return boxes, labels, additional_meta
