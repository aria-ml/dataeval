from __future__ import annotations

__all__ = []

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence, TypeVar

import torch
from defusedxml.ElementTree import parse
from numpy.typing import NDArray

from dataeval.utils.data.datasets._base import (
    BaseDataset,
    BaseODDataset,
    BaseSegDataset,
    DataLocation,
)
from dataeval.utils.data.datasets._mixin import BaseDatasetNumpyMixin, BaseDatasetTorchMixin
from dataeval.utils.data.datasets._types import ObjectDetectionTarget, SegmentationTarget

if TYPE_CHECKING:
    from dataeval.typing import Transform

_TArray = TypeVar("_TArray")
_TTarget = TypeVar("_TTarget")

VOCClassStringMap = Literal[
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
TVOCClassMap = TypeVar("TVOCClassMap", VOCClassStringMap, int, list[VOCClassStringMap], list[int])


class BaseVOCDataset(BaseDataset[_TArray, _TTarget, list[str]]):
    _resources = [
        DataLocation(
            url="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
            filename="VOCtrainval_11-May-2012.tar",
            md5=True,
            checksum="6cd6e144f989b92b3379bac3b3de84fd",
        ),
        DataLocation(
            url="http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar",
            filename="VOCtrainval_25-May-2011.tar",
            md5=True,
            checksum="6c3384ef61512963050cb5d687e5bf1e",
        ),
        DataLocation(
            url="http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar",
            filename="VOCtrainval_03-May-2010.tar",
            md5=True,
            checksum="da459979d0c395079b5c75ee67908abb",
        ),
        DataLocation(
            url="http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar",
            filename="VOCtrainval_11-May-2009.tar",
            md5=True,
            checksum="da459979d0c395079b5c75ee67908abb",
        ),
        DataLocation(
            url="http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar",
            filename="VOCtrainval_14-Jul-2008.tar",
            md5=True,
            checksum="2629fa636546599198acfcfbfcf1904a",
        ),
        DataLocation(
            url="http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
            filename="VOCtrainval_06-Nov-2007.tar",
            md5=True,
            checksum="c52e279531787c972589f7e41ab4ae64",
        ),
        DataLocation(
            url="http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
            filename="VOCtest_06-Nov-2007.tar",
            md5=True,
            checksum="b6e924de25625d8de591ea690078ad9f",
        ),
    ]

    index2label: dict[int, str] = {
        0: "aeroplane",
        1: "bicycle",
        2: "bird",
        3: "boat",
        4: "bottle",
        5: "bus",
        6: "car",
        7: "cat",
        8: "chair",
        9: "cow",
        10: "diningtable",
        11: "dog",
        12: "horse",
        13: "motorbike",
        14: "person",
        15: "pottedplant",
        16: "sheep",
        17: "sofa",
        18: "train",
        19: "tvmonitor",
    }

    def __init__(
        self,
        root: str | Path,
        year: Literal["2007", "2008", "2009", "2010", "2011", "2012"] = "2012",
        image_set: Literal["train", "val", "test", "base"] = "train",
        download: bool = False,
        transforms: Transform[_TArray] | Sequence[Transform[_TArray]] | None = None,
        verbose: bool = False,
    ) -> None:
        self.year = year
        self._resource_index = self._get_year_image_set_index(year, image_set)
        super().__init__(
            root,
            download,
            image_set,
            transforms,
            verbose,
        )

    def _get_dataset_dir(self) -> Path:
        """Function to reassign the dataset directory for common use with the VOC detection and segmentation classes"""
        if self._root.stem == f"VOC{self.year}":
            dataset_dir: Path = self._root
        else:
            dataset_dir: Path = self._root / f"VOC{self.year}"
        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def _get_year_image_set_index(self, year, image_set) -> int:
        """Function to ensure that the correct resource file is accessed"""
        if year == "2007" and image_set == "test":
            return -1
        elif year != "2007" and image_set == "test":
            raise ValueError(
                f"The only test set available is for the year 2007, not {year}. "
                "Either select the year 2007 or use a different image_set."
            )
        else:
            return 2012 - int(year)

    def _get_image_sets(self) -> dict[str, list[str]]:
        """Function to create the list of images in each image set"""
        image_folder = self.path / "JPEGImages"
        image_set_list = ["train", "val", "trainval"] if self.image_set != "test" else ["test"]
        image_sets = {}
        for image_set in image_set_list:
            text_file = self.path / "ImageSets" / "Main" / (image_set + ".txt")
            selected_images: list[str] = []
            with open(text_file) as f:
                for line in f.readlines():
                    out = line.strip()
                    selected_images.append(str(image_folder / (out + ".jpg")))

            name = "base" if image_set == "trainval" else image_set
            image_sets[name] = selected_images
        return image_sets

    def _load_data_inner(self) -> tuple[list[str], list[str], dict[str, Any]]:
        """Function to load in the file paths for the data, annotations and segmentation masks"""
        file_meta = {"year": [], "image_id": [], "mask_path": []}
        ann_folder = self.path / "Annotations"
        seg_folder = self.path / "SegmentationClass"

        # Load in the image sets
        image_sets = self._get_image_sets()

        # Get the data, annotations and metadata
        annotations = []
        data = image_sets[self.image_set]
        for entry in data:
            file_name = Path(entry).name
            file_stem = Path(entry).stem
            # Remove file extension and split by "_"
            parts = file_stem.split("_")
            file_meta["year"].append(parts[0])
            file_meta["image_id"].append(parts[1])
            file_meta["mask_path"].append(str(seg_folder / file_name))
            annotations.append(str(ann_folder / file_stem) + ".xml")

        return data, annotations, file_meta

    def _read_annotations(self, annotation: str) -> tuple[list[list[float]], list[int], dict[str, Any]]:
        boxes: list[list[float]] = []
        label_str = []
        root = parse(annotation).getroot()
        if root is None:
            raise ValueError(f"Unable to parse {annotation}")
        num_objects = len(root.findall("object"))
        additional_meta: dict[str, Any] = {
            "folder": [root.findtext("folder", default="") for _ in range(num_objects)],
            "filename": [root.findtext("filename", default="") for _ in range(num_objects)],
            "database": [root.findtext("source/database", default="") for _ in range(num_objects)],
            "annotation_source": [root.findtext("source/annotation", default="") for _ in range(num_objects)],
            "image_source": [root.findtext("source/image", default="") for _ in range(num_objects)],
            "image_width": [int(root.findtext("size/width", default="-1")) for _ in range(num_objects)],
            "image_height": [int(root.findtext("size/height", default="-1")) for _ in range(num_objects)],
            "image_depth": [int(root.findtext("size/depth", default="-1")) for _ in range(num_objects)],
            "segmented": [int(root.findtext("segmented", default="-1")) for _ in range(num_objects)],
            "pose": [],
            "truncated": [],
            "difficult": [],
        }
        for obj in root.findall("object"):
            label_str.append(obj.findtext("name", default=""))
            additional_meta["pose"].append(obj.findtext("pose", default=""))
            additional_meta["truncated"].append(int(obj.findtext("truncated", default="-1")))
            additional_meta["difficult"].append(int(obj.findtext("difficult", default="-1")))
            boxes.append(
                [
                    float(obj.findtext("bndbox/xmin", default="0")),
                    float(obj.findtext("bndbox/ymin", default="0")),
                    float(obj.findtext("bndbox/xmax", default="0")),
                    float(obj.findtext("bndbox/ymax", default="0")),
                ]
            )
        labels = [self.label2index[label] for label in label_str]
        return boxes, labels, additional_meta


class VOCDetection(
    BaseVOCDataset[NDArray[Any], ObjectDetectionTarget[NDArray[Any]]],
    BaseODDataset[NDArray[Any]],
    BaseDatasetNumpyMixin,
):
    """
    `Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory of dataset where the ``vocdataset`` folder exists.
    download : bool, default False
        If True, downloads the dataset from the internet and puts it in root directory.
        Class checks to see if data is already downloaded to ensure it does not create a duplicate download.
    image_set : "train", "val", "test", or "base", default "train"
        If "test", then dataset year must be "2007".
        If "base", then the combined dataset of "train" and "val" is returned.
    year : "2007", "2008", "2009", "2010", "2011" or "2012", default "2012"
        The dataset year.
    transforms : Transform, Sequence[Transform] or None, default None
        Transform(s) to apply to the data.
    verbose : bool, default False
        If True, outputs print statements.

    Attributes
    ----------
    path : pathlib.Path
        Location of the folder containing the data.
    image_set : "train", "val", "test" or "base"
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
    """


class VOCDetectionTorch(
    BaseVOCDataset[torch.Tensor, ObjectDetectionTarget[torch.Tensor]],
    BaseODDataset[torch.Tensor],
    BaseDatasetTorchMixin,
):
    """
    `Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset as PyTorch tensors.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory of dataset where the ``vocdataset`` folder exists.
    download : bool, default False
        If True, downloads the dataset from the internet and puts it in root directory.
        Class checks to see if data is already downloaded to ensure it does not create a duplicate download.
    image_set : "train", "val", "test", or "base", default "train"
        If "test", then dataset year must be "2007".
        If "base", then the combined dataset of "train" and "val" is returned.
    year : "2007", "2008", "2009", "2010", "2011" or "2012", default "2012"
        The dataset year.
    transforms : Transform, Sequence[Transform] or None, default None
        Transform(s) to apply to the data.
    verbose : bool, default False
        If True, outputs print statements.

    Attributes
    ----------
    path : pathlib.Path
        Location of the folder containing the data.
    image_set : "train", "val", "test" or "base"
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
    """


class VOCSegmentation(
    BaseVOCDataset[NDArray[Any], SegmentationTarget[NDArray[Any]]],
    BaseSegDataset[NDArray[Any]],
    BaseDatasetNumpyMixin,
):
    """
    `Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory of dataset where the ``vocdataset`` folder exists.
    download : bool, default False
        If True, downloads the dataset from the internet and puts it in root directory.
        Class checks to see if data is already downloaded to ensure it does not create a duplicate download.
    image_set : "train", "val", "test", or "base", default "train"
        If "test", then dataset year must be "2007".
        If "base", then the combined dataset of "train" and "val" is returned.
    year : "2007", "2008", "2009", "2010", "2011" or "2012", default "2012"
        The dataset year.
    transforms : Transform, Sequence[Transform] or None, default None
        Transform(s) to apply to the data.
    verbose : bool, default False
        If True, outputs print statements.

    Attributes
    ----------
    path : pathlib.Path
        Location of the folder containing the data.
    image_set : "train", "val", "test" or "base"
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
    """

    def _load_data(self) -> tuple[list[str], list[str], dict[str, list[Any]]]:
        filepaths, targets, datum_metadata = super()._load_data()
        self._masks = datum_metadata.pop("mask_path")
        return filepaths, targets, datum_metadata
