from __future__ import annotations

__all__ = []

from pathlib import Path
from typing import Any, Literal, TypeVar
from xml.etree.ElementTree import parse

import numpy as np
from numpy.typing import NDArray

from dataeval.utils.data.datasets._base import BaseDetectionDataset, BaseODDataset, BaseSegDataset, DataLocation

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


class BaseVOCDataset(BaseDetectionDataset):
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
    label2index: dict[str, int] = {v: k for k, v in index2label.items()}

    def _unique_id(self) -> str:
        unique_id = f"{self.__class__.__name__}_{self._image_set}_{self._year}"
        if self.size > 0:
            unique_id += f"_{self.size}"
        if self.unit_interval:
            unique_id += "_on-unit-interval"
        if self.dtype is not None:
            unique_id += f"_{self.dtype}"
        if self.channels == "channels_last":
            unique_id += "_channels-last"
        if self.normalize:
            unique_id += "_normalized"
        # if self.balance:
        #     unique_id += "_balanced"
        if self.from_back:
            unique_id += "_sliced-from-back"

        return unique_id

    def _get_directory(self, year) -> None:
        """Function to reassign the dataset directory for common use with the VOC detection and segmentation classes"""
        self.dataset_dir.rmdir()
        if self.root.stem == f"VOC{year}":
            self.dataset_dir: Path = self.root
        else:
            self.dataset_dir: Path = self.root / f"VOC{year}"
        if not self.dataset_dir.exists():
            self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def _check_year_image_set(self, year, image_set) -> None:
        """Function to ensure that the correct resource file is accessed"""
        if year == "2007" and image_set == "test":
            self._resource = self._resources[-1]
        elif year != "2007" and image_set == "test":
            raise ValueError(
                f"The only test set available is for the year 2007, not {year}. "
                "Either select the year 2007 or use a different image_set."
            )
        else:
            self._resource = self._resources[2012 - int(year)]

    def _get_image_set_dict(self) -> None:
        """Function to create the list of images in each image set"""
        image_folder = self.dataset_dir / "JPEGImages"
        image_set_list = ["train", "val", "trainval"] if self._image_set != "test" else ["test"]
        for image_set in image_set_list:
            text_file = self.dataset_dir / "ImageSets" / "Main" / (image_set + ".txt")
            selected_images: list[str] = []
            with open(text_file) as f:
                for line in f.readlines():
                    out = line.strip()
                    selected_images.append(str(image_folder / (out + ".jpg")))

            name = "base" if image_set == "trainval" else image_set
            self._image_set_dict[name] = selected_images

    def _load_data_inner(self) -> tuple[list[str], list[str], dict[str, Any]]:
        """Function to load in the file paths for the data, annotations and segmentation masks"""
        file_meta = {"year": [], "image_id": [], "mask_path": []}
        ann_folder = self.dataset_dir / "Annotations"
        seg_folder = self.dataset_dir / "SegmentationClass"

        # Load in the image sets
        self._get_image_set_dict()

        # Get the data, annotations and metadata
        annotations = []
        data = self._image_set_dict[self._image_set]
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

    def _read_annotations(self, annotation: str) -> tuple[NDArray[np.float64], NDArray[np.uintp], dict[str, Any]]:
        """Function for extracting the info out of the text files"""
        boxes = []
        label_str = []
        root = parse(annotation).getroot()
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
                    int(obj.findtext("bndbox/xmin", default="0")),
                    int(obj.findtext("bndbox/ymin", default="0")),
                    int(obj.findtext("bndbox/xmax", default="0")),
                    int(obj.findtext("bndbox/ymax", default="0")),
                ]
            )

        labels = [self.label2index[label] for label in label_str]

        return np.array(boxes, dtype=np.float64), np.array(labels, dtype=np.uintp), additional_meta


class VOCDetection(BaseVOCDataset, BaseODDataset):
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
    size : int, default -1
        Limit the dataset size, must be a value greater than 0.
    classes : "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", \
        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", \
        int, list, or None, default None
        Option to select specific classes from dataset. Classes are 0-9, any other number is ignored.
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
    """

    def __init__(
        self,
        root: str | Path,
        download: bool = False,
        image_set: Literal["train", "val", "test", "base"] = "train",
        year: Literal["2007", "2008", "2009", "2010", "2011", "2012"] = "2012",
        size: int = -1,
        classes: TVOCClassMap | None = None,
        unit_interval: bool = False,
        dtype: type | None = None,
        channels: Literal["channels_first", "channels_last"] = "channels_first",
        crop: int | None = None,
        normalize: tuple[float, float] | None = None,
        # balance: bool = False,
        slice_back: bool = False,
        verbose: bool = False,
    ) -> None:
        self._year = year
        super().__init__(
            root,
            download,
            image_set,
            size,
            unit_interval,
            dtype,
            channels,
            crop,
            normalize,
            # balance,
            slice_back,
            verbose,
        )

        self.class_set: set[int] = self._reduce_classes(classes)
        self.num_classes: int = len(self.class_set)
        self._image_set = image_set
        self._filepaths: list[str]

        # Adjust the directory and make sure image_set and year are compatible
        self._get_directory(year)
        self._check_year_image_set(year, image_set)

        # Get the image files
        self._filepaths, self._annotations, self._datum_metadata = self._load_data()


class VOCSegmentation(BaseVOCDataset, BaseSegDataset):
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
    size : int, default -1
        Limit the dataset size, must be a value greater than 0.
    classes : "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", \
        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", \
        int, list, or None, default None
        Option to select specific classes from dataset. Classes are 0-9, any other number is ignored.
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
    """

    def __init__(
        self,
        root: str | Path,
        download: bool = False,
        image_set: Literal["train", "val", "test", "base"] = "train",
        year: Literal["2007", "2008", "2009", "2010", "2011", "2012"] = "2012",
        size: int = -1,
        classes: TVOCClassMap | None = None,
        unit_interval: bool = False,
        dtype: type | None = None,
        channels: Literal["channels_first", "channels_last"] = "channels_first",
        crop: int | None = None,
        normalize: tuple[float, float] | None = None,
        # balance: bool = False,
        slice_back: bool = False,
        verbose: bool = False,
    ) -> None:
        self._year = year
        super().__init__(
            root,
            download,
            image_set,
            size,
            unit_interval,
            dtype,
            channels,
            crop,
            normalize,
            # balance,
            slice_back,
            verbose,
        )

        self.class_set: set[int] = self._reduce_classes(classes)
        self.num_classes: int = len(self.class_set)
        self._image_set = image_set
        self._filepaths: list[str]

        # Adjust the directory and make sure image_set and year are compatible
        self._get_directory(year)
        self._check_year_image_set(year, image_set)

        # Get the image files
        self._filepaths, self._annotations, self._datum_metadata = self._load_data()
        self._masks = self._datum_metadata.pop("mask_path")
