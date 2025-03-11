from __future__ import annotations

__all__ = []

import hashlib
import shutil
import tarfile
import zipfile
from abc import abstractmethod
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypeVar
from warnings import warn

import numpy as np
import requests
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

from dataeval.utils.data.datasets._types import (
    DatasetMetadata,
    ImageClassificationDataset,
    InfoMixin,
    ObjectDetectionDataset,
    ObjectDetectionTarget,
)

ARCHIVE_ENDINGS = [".zip", ".tar", ".tgz"]
COMPRESS_ENDINGS = [".gz", ".bz2"]
MNISTClassStringMap = Literal["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TMNISTClassMap = TypeVar("TMNISTClassMap", MNISTClassStringMap, int, list[MNISTClassStringMap], list[int])
CIFARClassStringMap = Literal["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
TCIFARClassMap = TypeVar("TCIFARClassMap", CIFARClassStringMap, int, list[CIFARClassStringMap], list[int])
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


class DataLocation(NamedTuple):
    url: str
    filename: str
    md5: bool
    checksum: str


def _validate_file(fpath, file_md5, md5: bool = False, chunk_size=65535) -> bool:
    hasher = hashlib.md5(usedforsecurity=False) if md5 else hashlib.sha256()
    with open(fpath, "rb") as fpath_file:
        while chunk := fpath_file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest() == file_md5


def _ensure_exists(
    resource: DataLocation, directory: Path, root: Path, download: bool = True, verbose: bool = False
) -> None:
    """
    For each resource, download it if it doesn't exist in the dataset_dir.
    If the resource is a zip file, extract it (including recursively extracting nested zips).
    """
    file_path = directory / str(resource.filename)
    alternate_path = root / str(resource.filename)
    _, file_ext = file_path.stem, file_path.suffix
    compression = False
    if file_ext in COMPRESS_ENDINGS:
        file_ext = file_path.suffixes[0]
        compression = True

    check_path = alternate_path if alternate_path.exists() and not file_path.exists() else file_path

    # Download file if it doesn't exist.
    if not check_path.exists() and download:
        if verbose:
            print(f"Downloading {resource.filename} from {resource.url}")
        _download_dataset(resource.url, check_path)

        if not _validate_file(check_path, resource.checksum, resource.md5):
            raise Exception("File checksum mismatch. Remove current file and retry download.")

        # If the file is a zip, tar or tgz extract it into the designated folder.
        if file_ext in ARCHIVE_ENDINGS:
            if verbose:
                print(f"Extracting {resource.filename}...")
            _archive_extraction(file_ext, check_path, directory, compression, verbose)

    elif not check_path.exists() and not download:
        raise FileNotFoundError(
            "Data could not be loaded with the provided root directory, ",
            f"the file path to the file {resource.filename} does not exist, ",
            "and the download parameter is set to False.",
        )
    else:
        if not _validate_file(check_path, resource.checksum, resource.md5):
            raise Exception("File checksum mismatch. Remove current file and retry download.")
        if verbose:
            print(f"{resource.filename} already exists, skipping download.")

        if file_ext in ARCHIVE_ENDINGS:
            if verbose:
                print(f"Extracting {resource.filename}...")
            _archive_extraction(file_ext, check_path, directory, compression, verbose)


def _download_dataset(url: str, file_path: Path, timeout: int = 60) -> None:
    """Download a single resource from its URL to the `data_folder`."""
    error_msg = "URL fetch failure on {}: {} -- {}"
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"{error_msg.format(url, e.response.status_code, e.response.reason)}") from e
    except requests.exceptions.RequestException as e:
        raise ValueError(f"{error_msg.format(url, 'Unknown error', str(e))}") from e

    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192  # 8 KB
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(block_size):
            f.write(chunk)
            progress_bar.update(len(chunk))
    progress_bar.close()


def _archive_extraction(file_ext, file_path, directory, compression: bool = False, verbose: bool = False):
    """
    Single function to extract and then flatten if necessary.
    Recursively extracts nested zip files as well.
    Extracts and flattens all folders to the base directory.
    """
    if file_ext != ".zip" or compression:
        _extract_tar_archive(file_path, directory)
    else:
        _extract_zip_archive(file_path, directory)
    # Look for nested zip files in the extraction directory and extract them recursively.
    # Does NOT extract in place - extracts everything to directory
    for child in directory.iterdir():
        if child.suffix == ".zip":
            if verbose:
                print(f"Extracting nested zip: {child} to {directory}")
            _extract_zip_archive(child, directory)

    # Determine if there are nested folders and remove them
    # Helps ensure there that data is at most one folder below main directory
    _flatten_extraction(directory, verbose)


def _extract_zip_archive(file_path: Path, extract_to: Path) -> None:
    """Extracts the zip file to the given directory."""
    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
            file_path.unlink()
    except zipfile.BadZipFile:
        raise FileNotFoundError(f"{file_path.name} is not a valid zip file, skipping extraction.")


def _extract_tar_archive(file_path: Path, extract_to: Path) -> None:
    """Extracts a tar file (or compressed tar) to the specified directory."""
    try:
        with tarfile.open(file_path, "r:*") as tar_ref:
            tar_ref.extractall(extract_to)
            file_path.unlink()
    except tarfile.TarError:
        raise FileNotFoundError(f"{file_path.name} is not a valid tar file, skipping extraction.")


def _flatten_extraction(base_directory: Path, verbose: bool = False) -> None:
    """
    If the extracted folder contains only directories (and no files),
    move all its subfolders to the dataset_dir and remove the now-empty folder.
    """
    for child in base_directory.iterdir():
        if child.is_dir():
            inner_list = list(child.iterdir())
            if all(subchild.is_dir() for subchild in inner_list):
                for subchild in child.iterdir():
                    if verbose:
                        print(f"Moving {subchild.stem} to {base_directory}")
                    shutil.move(subchild, base_directory)

                if verbose:
                    print(f"Removing empty folder {child.stem}")
                child.rmdir()

                # Checking for additional placeholder folders
                if len(inner_list) == 1:
                    _flatten_extraction(base_directory, verbose)


def _ic_data_subselection(
    labels: list[str],
    class_set: set[int],
    desired_samples: int,
    slice_backwards: bool = False,
    balance: bool = True,
    verbose: bool = True,
) -> NDArray[np.intp]:
    """Function to limit the data to the desired size based on parameters."""
    if verbose:
        print("Running data preprocessing steps - randomization and/or class balancing")

    # Use all samples if desired_samples is not positive
    num_samples = desired_samples if desired_samples > 0 else len(labels)

    # Retrieve which samples belong to which class
    label_array = np.array(labels, dtype=np.uintp)
    indices = sorted(
        [
            np.nonzero(label_array == i)[0][::-1] if slice_backwards else np.nonzero(label_array == i)[0]
            for i in class_set
        ],
        key=len,
        reverse=slice_backwards,
    )
    counts = np.array([class_labels.size for class_labels in indices])
    num_indices = len(indices)
    max_size = counts.min()

    # Determine initial samples per class
    if num_samples >= num_indices and num_samples < max_size * num_indices:
        per_class = num_samples // num_indices
    elif num_samples > 0 and num_samples < num_indices:
        per_class = 1
    else:
        per_class = max_size

    # Initial selection from each class
    selection = np.dstack([row[:per_class] for row in indices]).reshape(-1)

    if balance or selection.size >= num_samples:
        if verbose and selection.size < num_samples:
            warn(
                f"Because of dataset limitations, only {selection.size} samples "
                f"will be returned, instead of the desired {num_samples}."
            )
        return selection

    # Adjust if desired samples exceed total available samples
    if num_samples > label_array.size:
        if verbose:
            warn(
                f"Asked for more samples, {num_samples}, than the raw dataset contains, {label_array.size}. "
                "Adjusting down to raw dataset size."
            )
        num_samples = label_array.size

    # Round-robin selection from each group for additional samples.
    additional_indices = []
    current_ptr = per_class  # pointer for the next element in each group
    while len(additional_indices) < num_samples - selection.size:
        for i, grp in enumerate(indices):
            if current_ptr < grp.size:
                additional_indices.append(grp[current_ptr])
            if len(additional_indices) + selection.size >= num_samples:
                break
        current_ptr += 1
        if current_ptr >= counts.max():
            break

    selection = np.concatenate([selection, np.array(additional_indices)])
    return selection


class BaseDataset(InfoMixin):
    """
    Base class for internet downloaded datasets.
    """

    # Each subclass should override the attributes below.
    # Each resource tuple must contain:
    #    'url': str, the URL to download from
    #    'filename': str, the name of the file once downloaded
    #    'md5': boolean, True if it's the checksum value is md5
    #    'checksum': str, the associated checksum for the downloaded file
    _resources: list[DataLocation]
    index2label: dict[int, str]
    label2index: dict[str, int]

    def __init__(
        self,
        root: str | Path,
        download: bool = False,
        image_set: Literal["train", "val", "test", "base"] = "train",
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
        # User Parameters
        self.root: Path = root.absolute() if isinstance(root, Path) else Path(root).absolute()
        self.download = download
        self._image_set = image_set
        self.size = size
        self.unit_interval = unit_interval
        self.dtype = dtype
        self.channels = channels
        self.flatten = flatten
        self.normalize = normalize
        self.balance = balance
        self.from_back = slice_back
        self.verbose = verbose

        # Class Attributes
        self.metadata: DatasetMetadata = {
            "id": self._unique_id(),
            "index2label": self.index2label,
            "split": self._image_set,
        }
        self.class_set: set[int]
        self.num_classes: int
        self._filepaths: list[str]

        # Internal Parameters
        self._image_set = image_set
        self._one_hot_encoder = np.eye(len(self.index2label), dtype=np.uintp)
        self._year: str
        self._labels: list[Any]
        self._annotations: list[str]
        self._datum_metadata: dict[str, list[Any]]
        self._resource: DataLocation
        self._reorder: NDArray[np.intp]
        self._image_set_dict: dict[str, list[str]] = {}

        # Create a designated folder for this dataset (named after the class)
        if self.root.stem in [
            self.__class__.__name__.lower(),
            self.__class__.__name__.upper(),
            self.__class__.__name__,
        ]:
            self.dataset_dir: Path = self.root
        else:
            self.dataset_dir: Path = self.root / self.__class__.__name__.lower()
        if not self.dataset_dir.exists():
            self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def _unique_id(self) -> str:
        unique_id = f"{self.__class__.__name__}_{self._image_set}"
        if self.size > 0:
            unique_id += f"_{self.size}"
        if self.unit_interval:
            unique_id += "_on-unit-interval"
        if self.dtype is not None:
            unique_id += f"_{self.dtype}"
        if self.channels == "channels_last":
            unique_id += "_channels-last"
        if self.flatten:
            unique_id += "_flattened"
        if self.normalize:
            unique_id += "_normalized"
        if self.balance:
            unique_id += "_balanced"
        if self.from_back:
            unique_id += "_sliced-from-back"

        return unique_id

    @abstractmethod
    def __getitem__(
        self, index: int
    ) -> (
        tuple[NDArray[Any], NDArray[np.uintp], dict[str, Any]]
        | tuple[NDArray[Any], ObjectDetectionTarget[NDArray[Any]], dict[str, Any]]
    ): ...

    @abstractmethod
    def _load_data_inner(
        self,
    ) -> tuple[list[str], list[str], dict[str, Any]] | tuple[list[str], list[int], dict[str, Any]]: ...

    def __len__(self) -> int:
        return len(self._filepaths)

    def _reduce_classes(self, classes: TMNISTClassMap | TCIFARClassMap | TVOCClassMap | None) -> set[int]:
        temp_set = []
        if classes is not None:
            if not isinstance(classes, (list, tuple, np.ndarray)):
                classes = [classes]  # type: ignore
            elif isinstance(classes, np.ndarray):
                classes = classes.tolist()

            for val in classes:  # type: ignore
                if isinstance(val, int) and 0 <= val < 10:
                    temp_set.append(val)
                elif isinstance(val, str):
                    try:
                        temp_set.append(int(val))
                    except ValueError:
                        temp_set.append(self.label2index[val])

        result = set(temp_set) if temp_set else set(self.index2label)

        return result

    def _read_file(self, path: str) -> NDArray:
        """Function to read in the data"""
        x = np.array(Image.open(path))
        return x

    def _load_data(self) -> tuple[list[str], list[Any], dict[str, Any]]:
        """
        Function to determine if data can be accessed or if it needs to be downloaded and/or extracted.
        """
        if self.verbose:
            print(f"Determining if {self._resource.filename} needs to be downloaded.")

        try:
            result = self._load_data_inner()
            if self.verbose:
                print("No download needed, loaded data successfully.")
        except FileNotFoundError:
            _ensure_exists(self._resource, self.dataset_dir, self.root, self.download, self.verbose)
            result = self._load_data_inner()
        return result

    def _transforms(self, image: NDArray[np.float64]) -> NDArray[Any]:
        """Function to transform the image prior to returning based on parameters passed in."""
        if self.verbose:
            print("Running data transformations steps.")

        if self.flatten:
            image = image.reshape(image.size)
        elif self.channels == "channels_first":
            image = np.moveaxis(image, -1, 0)

        if self.unit_interval:
            image = image / 255

        if self.normalize:
            image = (image - self.normalize[0]) / self.normalize[1]  # type: ignore

        if self.dtype:
            image = image.astype(self.dtype)

        return image


class BaseClassificationDataset(BaseDataset, ImageClassificationDataset[NDArray[Any], DatasetMetadata]):
    """
    Base class for image classification datasets.
    """

    def __init__(
        self,
        root: str | Path,
        download: bool = False,
        image_set: Literal["train", "val", "test", "base"] = "train",
        size: int = -1,
        unit_interval: bool = False,
        dtype: type | None = None,
        channels: Literal["channels_first", "channels_last"] = "channels_first",
        flatten: bool = False,
        normalize: tuple[float, float] | None = None,
        balance: bool = False,
        slice_back: bool = False,
        verbose: bool = True,
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

    def __len__(self) -> int:
        return self.size if self.size > 0 else len(self._reorder)

    def _preprocess(self) -> NDArray[np.intp]:
        """Function to adjust what data is selected for use based on parameters passed in."""
        selection = _ic_data_subselection(
            self._labels, self.class_set, self.size, self.from_back, self.balance, self.verbose
        )

        return selection

    def __getitem__(self, index: int) -> tuple[NDArray[Any], NDArray[np.uintp], dict[str, Any]]:
        """
        Args
        ----
        index : int
            Value of the desired data point

        Returns
        -------
        tuple[NDArray[Any], NDArray[np.uintp], dict[str, Any]]
            Image, target, datum_metadata - where target is one-hot encoding of class.
        """
        # Get the adjusted data index
        selection = self._reorder[index]
        # Get the associated label and score
        label = self._labels[selection]
        score = self._one_hot_encoder[label].copy()
        # Get the image
        img = self._read_file(self._filepaths[selection])
        img = self._transforms(img)

        img_metadata = {key: val[index] for key, val in self._datum_metadata.items()}

        return img, score, img_metadata


class BaseDetectionDataset(BaseDataset):
    """
    Base class for object detection or segmentation datasets.
    """

    def __init__(
        self,
        root: str | Path,
        download: bool = False,
        image_set: Literal["train", "val", "test", "base"] = "train",
        size: int = -1,
        unit_interval: bool = False,
        dtype: type | None = None,
        channels: Literal["channels_first", "channels_last"] = "channels_first",
        normalize: tuple[float, float] | None = None,
        # balance: bool = False,
        slice_back: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            root,
            download,
            image_set,
            size,
            unit_interval,
            dtype,
            channels,
            False,
            normalize,
            False,
            slice_back,
            verbose,
        )

    @abstractmethod
    def _read_annotations(self, annotation: str) -> tuple[NDArray[np.float64], NDArray[np.uintp], dict[str, Any]]: ...


class BaseODDataset(BaseDetectionDataset, ObjectDetectionDataset[NDArray[np.float64], DatasetMetadata]):
    """
    Base class for object detection datasets.
    """

    def __getitem__(self, index: int) -> tuple[NDArray[Any], ObjectDetectionTarget[NDArray[Any]], dict[str, Any]]:
        """
        Args
        ----
        index : int
            Value of the desired data point

        Returns
        -------
        tuple[NDArray[Any], ObjectDetectionTarget[NDArray[Any]], dict[str, Any]]
            Image, target, datum_metadata - target.boxes returns boxes in x0, y0, x1, y1 format
        """
        # Grab the bounding boxes and labels from the annotations
        boxes, labels, additional_metadata = self._read_annotations(self._annotations[index])
        scores = np.ones(len(labels))
        # Get the image
        img = self._read_file(self._filepaths[index])
        img = self._transforms(img)

        target = ObjectDetectionTarget(boxes, labels, scores)

        img_metadata = {key: val[index] for key, val in self._datum_metadata.items()}
        img_metadata = img_metadata | additional_metadata

        return img, target, img_metadata
