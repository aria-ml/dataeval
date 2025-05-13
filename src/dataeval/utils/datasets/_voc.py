from __future__ import annotations

__all__ = []

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence, TypeVar

import torch
from defusedxml.ElementTree import parse
from numpy.typing import NDArray

from dataeval.utils.datasets._base import (
    BaseDataset,
    BaseODDataset,
    BaseSegDataset,
    DataLocation,
    _ensure_exists,
    _TArray,
    _TTarget,
)
from dataeval.utils.datasets._mixin import BaseDatasetNumpyMixin, BaseDatasetTorchMixin
from dataeval.utils.datasets._types import ObjectDetectionTarget, SegmentationTarget

if TYPE_CHECKING:
    from dataeval.typing import Transform

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
            url="https://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar",
            filename="VOCtrainval_11-May-2012.tar",
            md5=False,
            checksum="e14f763270cf193d0b5f74b169f44157a4b0c6efa708f4dd0ff78ee691763bcb",
        ),
        DataLocation(
            url="http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar",
            filename="VOCtrainval_25-May-2011.tar",
            md5=False,
            checksum="0a7f5f5d154f7290ec65ec3f78b72ef72c6d93ff6d79acd40dc222a9ee5248ba",
        ),
        DataLocation(
            url="http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar",
            filename="VOCtrainval_03-May-2010.tar",
            md5=False,
            checksum="1af4189cbe44323ab212bff7afbc7d0f55a267cc191eb3aac911037887e5c7d4",
        ),
        DataLocation(
            url="http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar",
            filename="VOCtrainval_11-May-2009.tar",
            md5=False,
            checksum="11cbe1741fb5bdadbbca3c08e9ec62cd95c14884845527d50847bc2cf57e7fd6",
        ),
        DataLocation(
            url="http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar",
            filename="VOCtrainval_14-Jul-2008.tar",
            md5=False,
            checksum="7f0ca53c1b5a838fbe946965fc106c6e86832183240af5c88e3f6c306318d42e",
        ),
        DataLocation(
            url="https://data.brainchip.com/dataset-mirror/voc/VOCtrainval_06-Nov-2007.tar",
            filename="VOCtrainval_06-Nov-2007.tar",
            md5=False,
            checksum="7d8cd951101b0957ddfd7a530bdc8a94f06121cfc1e511bb5937e973020c7508",
        ),
        DataLocation(
            url="https://data.brainchip.com/dataset-mirror/voc/VOC2012test.tar",
            filename="VOC2012test.tar",
            md5=False,
            checksum="f08582b1935816c5eab3bbb1eb6d06201a789eaa173cdf1cf400c26f0cac2fb3",
        ),
        DataLocation(
            url="https://data.brainchip.com/dataset-mirror/voc/VOCtest_06-Nov-2007.tar",
            filename="VOCtest_06-Nov-2007.tar",
            md5=False,
            checksum="6836888e2e01dca84577a849d339fa4f73e1e4f135d312430c4856b5609b4892",
        ),
    ]
    _base2007: tuple[int, int] = (5, 7)
    _base2012: tuple[int, int] = (0, 6)

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
        image_set: Literal["train", "val", "test", "base"] = "train",
        year: Literal["2007", "2008", "2009", "2010", "2011", "2012"] = "2012",
        transforms: Transform[_TArray] | Sequence[Transform[_TArray]] | None = None,
        download: bool = False,
        verbose: bool = False,
    ) -> None:
        self.year = year
        self._resource_index = self._get_year_image_set_index(year, image_set)
        super().__init__(
            root,
            image_set,
            transforms,
            download,
            verbose,
        )

    def _get_dataset_dir(self) -> Path:
        """Overrides the base function to determine correct dataset directory for VOC class"""
        return self._find_main_VOC_dir(self._root)

    def _find_main_VOC_dir(self, base: Path) -> Path:
        """
        Determine the correct dataset directory for VOC detection and segmentation classes.
        Handles various directory structure possibilities and validates existence.
        """

        # VOCdataset directory possibilities
        dataset_dir = base if base.stem.lower() == "vocdataset" else base / "vocdataset"

        # Define possible directory structures based on patterns
        # 1. Root is already the specific VOC year directory
        # 2. Root is the VOCdevkit directory
        # 3. Standard structure
        # 4. Special case for year 2011
        # 5. Within VOCdataset directory
        # 6. Special case for year 2011 within VOCdataset
        possible_paths = [
            base if base.stem == f"VOC{self.year}" else None,
            base / f"VOC{self.year}" if base.stem == "VOCdevkit" else None,
            base / "VOCdevkit" / f"VOC{self.year}",
            base / "TrainVal" / "VOCdevkit" / f"VOC{self.year}" if self.year == "2011" else None,
            dataset_dir / "VOCdevkit" / f"VOC{self.year}",
            dataset_dir / "TrainVal" / "VOCdevkit" / f"VOC{self.year}" if self.year == "2011" else None,
        ]

        # Filter out None values and check each path
        for path in filter(None, possible_paths):
            if path.exists():
                return path

        # If no existing path is found, create and return the dataset directory
        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True, exist_ok=True)

        return dataset_dir

    def _get_year_image_set_index(self, year: str, image_set: str) -> int:
        """Function to ensure that the correct resource file is accessed"""
        if year == "2007" and image_set == "test":
            return -1
        if year == "2012" and image_set == "test":
            return -2
        if year != "2007" and image_set == "test":
            raise ValueError(
                f"The only test sets available are for the years 2007 and 2012, not {year}. "
                "Either select the year 2007 or 2012, or use a different image_set."
            )
        return 2012 - int(year)

    def _update_path(self) -> None:
        """Update the path to the new folder structure"""
        if self.year == "2011" and self.path.stem.lower() == "vocdataset":
            self.path: Path = self.path / "TrainVal" / "VOCdevkit" / f"VOC{self.year}"
        elif self.path.stem.lower() == "vocdataset":
            self.path: Path = self.path / "VOCdevkit" / f"VOC{self.year}"

    def _load_data_exception(self) -> tuple[list[str], list[str], dict[str, Any]]:
        """Adjust how the directory is created for the 2007 and 2012 test set"""
        filepaths: list[str] = []
        targets: list[str] = []
        datum_metadata: dict[str, list[Any]] = {}
        tmp_path: Path = self._root / "tmp_directory_for_download"
        tmp_path.mkdir(exist_ok=True)
        resource_idx = self._base2007 if self.year == "2007" else self._base2012

        # Determine if text files exist
        train_file = self.path / "ImageSets" / "Main" / "trainval.txt"
        test_file = self.path / "ImageSets" / "Main" / "test.txt"
        train_exists = train_file.exists()
        test_exists = test_file.exists()

        if self.image_set == "base":
            if not train_exists and not test_exists:
                _ensure_exists(*self._resources[resource_idx[0]], self.path, self._root, self._download, self._verbose)
                self._update_path()
                _ensure_exists(*self._resources[resource_idx[1]], tmp_path, self._root, self._download, self._verbose)
                self._merge_voc_directories(tmp_path)

            elif train_exists and not test_exists:
                _ensure_exists(*self._resources[resource_idx[1]], tmp_path, self._root, self._download, self._verbose)
                self._merge_voc_directories(tmp_path)

            elif not train_exists and test_exists:
                _ensure_exists(*self._resources[resource_idx[0]], tmp_path, self._root, self._download, self._verbose)
                self._merge_voc_directories(tmp_path)

            # Code to determine what is needed in each category
            metadata_list: list[dict[str, Any]] = []

            for img_set in ["test", "base"]:
                self.image_set = img_set
                resource_filepaths, resource_targets, resource_metadata = self._load_data_inner()
                filepaths.extend(resource_filepaths)
                targets.extend(resource_targets)
                metadata_list.append(resource_metadata)

            # Combine metadata from all resources
            for data_dict in metadata_list:
                for key, val in data_dict.items():
                    str_key = str(key)  # Ensure key is string
                    if str_key not in datum_metadata:
                        datum_metadata[str_key] = []
                    datum_metadata[str_key].extend(val)

        else:
            self._resource = self._resources[resource_idx[1]]

            if train_exists and not test_exists:
                _ensure_exists(*self._resource, tmp_path, self._root, self._download, self._verbose)
                self._merge_voc_directories(tmp_path)

            resource_filepaths, resource_targets, resource_metadata = self._load_try_and_update()
            filepaths.extend(resource_filepaths)
            targets.extend(resource_targets)
            datum_metadata.update(resource_metadata)

        return filepaths, targets, datum_metadata

    def _merge_voc_directories(self, source_dir: Path) -> None:
        """Merge two VOC directories, handling file conflicts intelligently."""
        base: Path = self._find_main_VOC_dir(source_dir)
        # Create all subdirectories in target if they don't exist
        for dirpath, dirnames, filenames in os.walk(base):
            # Convert to Path objects
            source_path = Path(dirpath)

            # Get the relative path from source_dir
            rel_path = source_path.relative_to(base)

            # Create the corresponding target path
            target_path = self.path / rel_path
            target_path.mkdir(parents=True, exist_ok=True)

            # Copy all files
            for filename in filenames:
                source_file = source_path / filename
                target_file = target_path / filename

                # File doesn't exist in target, just move it
                if not target_file.exists():
                    shutil.move(source_file, target_file)
                else:
                    # File exists in both assume they're identical and skip
                    pass

        shutil.rmtree(source_dir)

    def _load_try_and_update(self) -> tuple[list[str], list[str], dict[str, Any]]:
        """Test if data needs to be downloaded and update path if it does"""
        if self._verbose:
            print(f"Determining if {self._resource.filename} needs to be downloaded.")

        try:
            result = self._load_data_inner()
            if self._verbose:
                print("No download needed, loaded data successfully.")
        except FileNotFoundError:
            _ensure_exists(*self._resource, self.path, self._root, self._download, self._verbose)
            self._update_path()
            result = self._load_data_inner()
        return result

    def _load_data(self) -> tuple[list[str], list[str], dict[str, Any]]:
        """
        Function to determine if data can be accessed or if it needs to be downloaded and/or extracted.
        """
        # Exception - test sets
        year_set_bool = (self.image_set == "test" or self.image_set == "base") and (
            self.year == "2012" or self.year == "2007"
        )
        if year_set_bool:
            return self._load_data_exception()

        return self._load_try_and_update()

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
            if self.year != "2007":
                # Remove file extension and split by "_"
                parts = file_stem.split("_")
                file_meta["year"].append(parts[0])
                file_meta["image_id"].append(parts[1])
            else:
                file_meta["year"].append(self.year)
                file_meta["image_id"].append(file_stem)
            file_meta["mask_path"].append(str(seg_folder / file_name))
            annotations.append(str(ann_folder / file_stem) + ".xml")

        return data, annotations, file_meta

    def _read_annotations(self, annotation: str) -> tuple[list[list[float]], list[int], dict[str, Any]]:
        boxes: list[list[float]] = []
        label_str = []
        if not Path(annotation).exists():
            return boxes, label_str, {}
        root = parse(annotation).getroot()
        if root is None:
            raise ValueError(f"Unable to parse {annotation}")
        additional_meta: dict[str, Any] = {
            "folder": root.findtext("folder", default=""),
            "filename": root.findtext("filename", default=""),
            "database": root.findtext("source/database", default=""),
            "annotation_source": root.findtext("source/annotation", default=""),
            "image_source": root.findtext("source/image", default=""),
            "image_width": int(root.findtext("size/width", default="-1")),
            "image_height": int(root.findtext("size/height", default="-1")),
            "image_depth": int(root.findtext("size/depth", default="-1")),
            "segmented": int(root.findtext("segmented", default="-1")),
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
        Because of the structure of the PASCAL VOC datasets, the root needs to be one of 4 folders.
        1) Directory containing the year of the **already downloaded** dataset (i.e. .../VOCdevkit/VOC2012 <-)
        2) Directory to the VOCdevkit folder of the **already downloaded** dataset (i.e. .../VOCdevkit <- /VOC2012)
        3) Directory to the folder one level up from the VOCdevkit folder,
        data **may** or **may not** be already downloaded (i.e. ... <- /VOCdevkit/VOC2012)
        4) Directory to where you would like the dataset to be downloaded
    image_set : "train", "val", "test", or "base", default "train"
        If "test", then dataset year must be "2007" or "2012". Note that the 2012 test set does not contain annotations.
        If "base", then the combined dataset of "train" and "val" is returned.
    year : "2007", "2008", "2009", "2010", "2011" or "2012", default "2012"
        The dataset year.
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
    year : "2007", "2008", "2009", "2010", "2011" or "2012"
        The selected dataset year.
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

    Note
    ----
    Data License: `Flickr Terms of Use <http://www.flickr.com/terms.gne?legacy=1>`_
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
        Because of the structure of the PASCAL VOC datasets, the root needs to be one of 4 folders.
        1) Directory containing the year of the **already downloaded** dataset (i.e. .../VOCdevkit/VOC2012 <-)
        2) Directory to the VOCdevkit folder of the **already downloaded** dataset (i.e. .../VOCdevkit <- /VOC2012)
        3) Directory to the folder one level up from the VOCdevkit folder,
        data **may** or **may not** be already downloaded (i.e. ... <- /VOCdevkit/VOC2012)
        4) Directory to where you would like the dataset to be downloaded
    image_set : "train", "val", "test", or "base", default "train"
        If "test", then dataset year must be "2007" or "2012". Note that the 2012 test set does not contain annotations.
        If "base", then the combined dataset of "train" and "val" is returned.
    year : "2007", "2008", "2009", "2010", "2011" or "2012", default "2012"
        The dataset year.
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
    year : "2007", "2008", "2009", "2010", "2011" or "2012"
        The selected dataset year.
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

    Note
    ----
    Data License: `Flickr Terms of Use <http://www.flickr.com/terms.gne?legacy=1>`_
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
    image_set : "train", "val", "test", or "base", default "train"
        If "test", then dataset year must be "2007".
        If "base", then the combined dataset of "train" and "val" is returned.
    year : "2007", "2008", "2009", "2010", "2011" or "2012", default "2012"
        The dataset year.
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
    year : "2007", "2008", "2009", "2010", "2011" or "2012"
        The selected dataset year.
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

    Note
    ----
    Data License: `Flickr Terms of Use <http://www.flickr.com/terms.gne?legacy=1>`_
    """

    def _load_data(self) -> tuple[list[str], list[str], dict[str, list[Any]]]:
        """Overload base load data to split out masks for segmentation."""
        # Exception - test sets
        year_set_bool = (self.image_set == "test" or self.image_set == "base") and (
            self.year == "2012" or self.year == "2007"
        )
        if year_set_bool:
            filepaths, targets, datum_metadata = self._load_data_exception()
        else:
            filepaths, targets, datum_metadata = self._load_try_and_update()
        self._masks = datum_metadata.pop("mask_path")
        return filepaths, targets, datum_metadata
