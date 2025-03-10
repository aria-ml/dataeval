from __future__ import annotations

__all__ = []

import hashlib
import shutil
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple, TypeVar
from warnings import warn

import numpy as np
import requests
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.nn.functional import one_hot
from torchvision.datasets import CIFAR10 as _CIFAR10
from torchvision.transforms import v2
from tqdm import tqdm

from dataeval.utils.data.datasets._types import DatasetMetadata, ImageClassificationDataset, InfoMixin


class DataLocation(NamedTuple):
    url: str
    filename: str
    md5: bool
    checksum: str


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


ClassStringMap = Literal["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TClassMap = TypeVar("TClassMap", ClassStringMap, int, list[ClassStringMap], list[int])
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


def _validate_file(fpath, file_md5, md5: bool = False, chunk_size=65535) -> bool:
    hasher = hashlib.md5() if md5 else hashlib.sha256()
    with open(fpath, "rb") as fpath_file:
        while chunk := fpath_file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest() == file_md5


def _ensure_exists(resource: DataLocation, directory: Path, download: bool = True, verbose: bool = False) -> None:
    """
    For each resource, download it if it doesn't exist in the dataset_dir.
    If the resource is a zip file, extract it (including recursively extracting nested zips).
    """
    file_path = directory / str(resource.filename)
    _, file_ext = file_path.stem, file_path.suffix

    # Download file if it doesn't exist.
    if not file_path.exists() and download:
        if verbose:
            print(f"Downloading {resource.filename} from {resource.url}")
        _download_dataset(resource.url, file_path)

        if not _validate_file(file_path, resource.checksum, resource.md5):
            raise Exception("File checksum mismatch. Remove current file and retry download.")

        # If the file is a zip, extract it into the designated folder.
        if file_ext == ".zip":
            if verbose:
                print(f"Extracting {resource.filename}...")
            _zip_extraction(file_path, directory, verbose)

    elif not file_path.exists() and not download:
        raise FileNotFoundError(
            "Data could not be loaded with the provided root directory, ",
            f"the file path to the file {resource.filename} does not exist, ",
            "and the download parameter is set to False.",
        )
    else:
        if not _validate_file(file_path, resource.checksum, resource.md5):
            raise Exception("File checksum mismatch. Remove current file and retry download.")
        if verbose:
            print(f"{resource.filename} already exists, skipping download.")

        if file_ext == ".zip":
            if verbose:
                print(f"Extracting {resource.filename}...")
            _zip_extraction(file_path, directory, verbose)


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
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                progress_bar.update(len(chunk))
    progress_bar.close()


def _zip_extraction(file_path, directory, verbose: bool = False):
    """
    Single function to extract and then flatten if necessary.
    Recursively extracts nested zip files as well.
    Extracts and flattens all folders to the base directory.
    """
    _extract_archive(file_path, directory, verbose)
    # Look for nested zip files in the extraction directory and extract them recursively.
    # Does NOT extract in place - extracts everything to directory
    for child in directory.iterdir():
        if child.suffix == ".zip":
            if verbose:
                print(f"Extracting nested zip: {child} to {directory}")
            _extract_archive(child, directory, verbose)
    # Determine if there are nested folders and remove them
    # Helps ensure there that data is at most one folder below main directory
    _flatten_extraction(directory, verbose)


def _extract_archive(file_path: Path, extract_to: Path, verbose: bool = False) -> None:
    """
    Extracts the zip file to the given directory.
    """
    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
            file_path.unlink()
    except zipfile.BadZipFile:
        raise FileNotFoundError(f"{file_path.name} is not a valid zip file, skipping extraction.")


def _flatten_extraction(base_directory: Path, verbose: bool = False) -> None:
    """
    If the extracted folder contains only directories (and no files),
    move all its subfolders to the dataset_dir and remove the now-empty folder.
    """
    for child in base_directory.iterdir():
        if child.is_dir() and all(subchild.is_dir() for subchild in child.iterdir()):
            for subchild in child.iterdir():
                if verbose:
                    print(f"Moving {subchild.stem} to {base_directory}")
                shutil.move(subchild, base_directory)

            if verbose:
                print(f"Removing empty folder {child.stem}")
            child.rmdir()


def _randomize_data(
    data_arr: NDArray, target_arr: NDArray, datum_metadata: dict[str, list[Any]]
) -> tuple[NDArray, NDArray, dict[str, list[Any]]]:
    """Function to randomly shuffle the data prior to sampling"""
    rdm_seed = np.random.default_rng(2023)
    shuffled_indices = rdm_seed.permutation(data_arr.shape[0])
    rdata = data_arr[shuffled_indices]
    tdata = target_arr[shuffled_indices]
    if datum_metadata:
        for key in datum_metadata:
            datum_metadata[key] = np.array(datum_metadata[key])[shuffled_indices].tolist()

    return rdata, tdata, datum_metadata


def _data_subselection(
    labels: NDArray[np.intp],
    class_set: set[int],
    num_samples: int,
    slice_backwards: bool = False,
    balance: bool = True,
    verbose: bool = True,
) -> NDArray[np.intp]:
    """Function to limit the data to the desired size based on parameters."""
    indices = [np.nonzero(labels == i)[0] for i in class_set]
    counts = np.array([class_labels.size for class_labels in indices])
    num_indices = len(indices)
    max_size = counts.min()

    if slice_backwards:
        selection = np.dstack([row[-max_size:] for row in indices])
        selection = selection.reshape(selection.size)
    else:
        selection = np.dstack([row[:max_size] for row in indices])
        selection = selection.reshape(selection.size)

    if not balance:
        counts -= counts.min()
        if counts.sum() > 0:
            if slice_backwards:
                additional = np.concatenate([indices[i][:-max_size] for i in range(num_indices) if counts[i] > 0])
            else:
                additional = np.concatenate([indices[i][max_size:] for i in range(num_indices) if counts[i] > 0])
            selection = np.concatenate([selection, additional])
        selection = selection[:num_samples] if num_samples > 0 else selection
    else:
        if num_samples > num_indices and num_samples < max_size * num_indices:
            size = (num_samples // num_indices) * num_indices
        elif num_samples > 0 and num_samples < num_indices:
            size = num_indices
        else:
            size = max_size * num_indices
            if verbose and num_samples > max_size * num_indices:
                warn(
                    f"Because of dataset limitations, only {max_size * num_indices} samples "
                    f"will be returned, instead of the desired {num_samples}."
                )

        selection = selection[:size]
    return selection


class BaseICDataset(ImageClassificationDataset[NDArray[np.float64], DatasetMetadata], ABC):
    """
    Base class for internet downloaded datasets.
    """

    # Each subclass should override this with a list of resource tuples.
    # Each resource tuple must contain:
    #    'url': str, the URL to download from
    #    'filename': str, the name of the file once downloaded
    #    'md5': boolean, True if it's the checksum value is md5
    #    'checksum': str, the associated checksum for the downloaded file
    _resources: list[DataLocation]
    metadata: DatasetMetadata
    index2label: dict[int, str]
    label2index: dict[str, int]

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
        randomize: bool = True,
        slice_back: bool = False,
        verbose: bool = True,
    ) -> None:
        self.root: Path = root.absolute() if isinstance(root, Path) else Path(root).absolute()
        self.download = download
        self.size = size
        self.unit_interval = unit_interval
        self.dtype = dtype
        self.channels = channels
        self.flatten = flatten
        self.normalize = normalize
        self.balance = balance
        self.randomize = randomize
        self.from_back = slice_back
        self.verbose = verbose
        self._one_hot_encoder = np.eye(len(self.index2label))
        self.metadata = {"id": self.__class__.__name__, "index2label": self.index2label}
        self.class_set: set[int]
        self.num_classes: int
        self._data: NDArray[Any]
        self._targets: NDArray[np.int_]
        self._datum_metadata: dict[str, list[Any]]
        self._resource: DataLocation

        # Create a designated folder for this dataset (named after the class)
        if self.root.stem in [
            self.__class__.__name__.lower(),
            self.__class__.__name__.upper(),
            self.__class__.__name__,
        ]:
            self.dataset_dir: Path = self.root
        else:
            self.dataset_dir: Path = self.root / self.__class__.__name__.lower()
            self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def _get_resource(self) -> tuple[NDArray[Any], NDArray[np.int_], dict[str, Any]]:
        if self.verbose:
            print("Determining if data needs to be downloaded")

        try:
            result = self._load_data()
            if self.verbose:
                print("Loaded data successfully")
        except FileNotFoundError:
            _ensure_exists(self._resource, self.dataset_dir, self.download, self.verbose)
            result = self._load_data()
        return result

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[NDArray[Any], NDArray[np.float64], dict[str, Any]]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, dict) where target is one-hot encoding of class.
        """
        img, label = self._data[index], int(self._targets[index])
        img = self._transforms(img)
        target = self._one_hot_encoder[label].copy()

        img_metadata = {key: val[index] for key, val in self._datum_metadata.items()}

        return img, target, img_metadata

    def _transforms(self, image: NDArray[np.float64]) -> NDArray[Any]:
        if self.flatten:
            image = image.reshape(image.size)
        elif self.channels == "channels_last":
            image = np.moveaxis(image, 0, -1)

        if self.unit_interval:
            image = image / 255

        if self.normalize:
            image = (image - self.normalize[0]) / self.normalize[1]

        if self.dtype:
            image = image.astype(self.dtype)

        return image

    @abstractmethod
    def _load_data(self) -> tuple[NDArray[Any], NDArray[np.int_], dict[str, Any]]: ...

    @abstractmethod
    def _read_file(self, path: Path) -> NDArray[Any]: ...

    def _preprocess(self) -> None:
        """Function to adjust the data prior to selection for use based on parameters passed in."""
        if self.verbose:
            print("Running data preprocessing steps")

        if self.randomize:
            self._data, self._targets, self._datum_metadata = _randomize_data(
                self._data, self._targets, self._datum_metadata
            )

        if self.size > self._targets.size:
            if self.verbose:
                warn(
                    f"Asked for more samples, {self.size}, than the raw dataset contains, {self._targets.shape[0]}. "
                    "Adjusting down to raw dataset size."
                )
            self.size: int = -1

        if self.size > 0 or self.balance:
            selection = _data_subselection(
                self._targets, self.class_set, self.size, self.from_back, self.balance, self.verbose
            )
            self._data = self._data[selection]
            self._targets = self._targets[selection]
            if self._datum_metadata:
                for key in self._datum_metadata:
                    self._datum_metadata[key] = np.array(self._datum_metadata[key])[selection].tolist()
        elif self.size < 1 and not self.balance and self.from_back:
            self._data = self._data[::-1]
            self._targets = self._targets[::-1]
            if self._datum_metadata:
                for key in self._datum_metadata:
                    self._datum_metadata[key] = np.array(self._datum_metadata[key])[::-1].tolist()


class MNIST(BaseICDataset, InfoMixin):
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
    train : bool, default True
        If True, creates dataset from ``train_images.npy`` and ``train_labels.npy``.
    size : int, default -1
        Limit the dataset size, must be a value greater than 0.
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
    classes : "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", \
        int, list, or None, default None
        Option to select specific classes from dataset. Classes are 0-9, any other number is ignored.
    balance : bool, default True
        If True, returns equal number of samples for each class.
    randomize : bool, default True
        If True, shuffles the data prior to selection - uses a set seed for reproducibility.
    slice_back : bool, default False
        If True and size has a value greater than 0, then grabs selection starting at the last image.
    verbose : bool, default True
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
        train: bool = True,
        size: int = -1,
        unit_interval: bool = False,
        dtype: type | None = None,
        channels: Literal["channels_first", "channels_last"] = "channels_first",
        flatten: bool = False,
        normalize: tuple[float, float] | None = None,
        corruption: CorruptionStringMap | None = None,
        classes: TClassMap | None = None,
        balance: bool = True,
        randomize: bool = True,
        slice_back: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            root,
            download,
            size,
            unit_interval,
            dtype,
            channels,
            flatten,
            normalize,
            balance,
            randomize,
            slice_back,
            verbose,
        )

        self.train = train  # training set or test set
        self.corruption = corruption
        self._image_set = "train" if train else "test"
        self._data: NDArray[np.float64]
        self._targets: NDArray[np.int_]
        self._datum_metadata: dict[str, list[Any]]

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

        self.class_set: set[int] = set(temp_set) if temp_set else set(self.index2label)
        self.num_classes: int = len(self.class_set)

        if self.corruption == "identity" and verbose:
            print("Identity is not a corrupted dataset but the original MNIST dataset.")

        self._resource: DataLocation = self._resources[0] if self.corruption is None else self._resources[1]

        # Load the data
        self._data, self._targets, self._datum_metadata = self._get_resource()
        # Adjust the data as desired
        self._preprocess()

    def _load_data(self) -> tuple[NDArray[np.float64], NDArray[np.int_], dict[str, Any]]:
        if self.corruption is None:
            try:
                file_path = self.dataset_dir / self._resource.filename
                data, targets = self._read_normal_file(file_path)
            except FileNotFoundError:
                data, targets = self._load_corruption()
        else:
            data, targets = self._load_corruption()
        return data, targets, {}

    def _load_corruption(self) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
        corruption = self.corruption if self.corruption is not None else "identity"
        base_path = self.dataset_dir / corruption
        file_path = base_path / f"{'train' if self.train else 'test'}_images.npy"
        data = self._read_file(file_path)
        data = data.astype(np.float64)
        data = np.moveaxis(data, -1, 1)

        label_path = base_path / f"{'train' if self.train else 'test'}_labels.npy"
        targets = self._read_file(label_path)
        targets = targets.astype(np.int_)

        return data, targets

    def _read_normal_file(self, path: Path) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
        with np.load(path, allow_pickle=True) as f:
            if self.train:
                x, y = f["x_train"], f["y_train"]
            else:
                x, y = f["x_test"], f["y_test"]
            x = x[:, np.newaxis, ...]
        return x, y

    def _read_file(self, path: Path) -> NDArray[Any]:
        x = np.load(path, allow_pickle=False)
        return x


class ShipDataset(BaseICDataset, InfoMixin):
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
        Root directory of dataset where the ``ships-in-satellite-imagery`` folder exists.
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
    randomize : bool, default True
        If True, shuffles the data prior to selection - uses a set seed for reproducibility.
    slice_back : bool, default False
        If True and size has a value greater than 0, then grabs selection starting at the last image.
    verbose : bool, default True
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
    scenes : list[NDArray]
        These are extra data samples that are large satellite images encompassing an entire scene.
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
        randomize: bool = True,
        slice_back: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            root,
            download,
            size,
            unit_interval,
            dtype,
            channels,
            flatten,
            normalize,
            balance,
            randomize,
            slice_back,
            verbose,
        )

        self.class_set: set[int] = set(self.index2label)
        self.num_classes: int = len(self.class_set)
        self._resource: DataLocation = self._resources[0]
        self._data: NDArray[np.uint8]
        self._targets: NDArray[np.int_]
        self._datum_metadata: dict[str, list[Any]]
        self.scenes: list[NDArray[np.uint8]]

        # Load the data
        self._data, self._targets, self._datum_metadata = self._get_resource()
        self.scenes = self._load_scenes()
        # Adjust the data as desired
        self._preprocess()

    def _load_data(self) -> tuple[NDArray[np.uint8], NDArray[np.int_], dict[str, Any]]:
        file_data = {"label": [], "scene_id": [], "longitude": [], "latitude": [], "path": []}
        data_folder = self.dataset_dir / "shipsnet"
        for entry in data_folder.iterdir():
            if entry.is_file() and entry.suffix == ".png":
                # Remove file extension and split by "_"
                parts = entry.stem.split("__")  # Removes ".png" and splits the string
                if len(parts) == 3:  # Ensure correct structure
                    file_data["label"].append(parts[0])
                    file_data["scene_id"].append(parts[1])
                    lat_lon = parts[2].split("_")
                    file_data["longitude"].append(float(lat_lon[0]))
                    file_data["latitude"].append(float(lat_lon[1]))
                file_data["path"].append(entry)
        data = np.stack([self._read_file(entry) for entry in file_data["path"]], axis=0, dtype=np.uint8)
        targets = np.array(file_data["label"], dtype=np.int_)
        del file_data["path"]
        del file_data["label"]
        data = np.moveaxis(data, -1, 1)
        return data, targets, file_data

    def _load_scenes(self) -> list[NDArray[np.uint8]]:
        data_folder = self.dataset_dir / "scenes"
        scene = []
        for entry in data_folder.iterdir():
            if entry.is_file() and entry.suffix == ".png":
                if self.channels == "channels_last":
                    scene.append(self._read_file(entry))
                else:
                    scene.append(np.moveaxis(self._read_file(entry), -1, 0))
        return scene

    def _read_file(self, path: Path) -> NDArray:
        x = np.array(Image.open(path))
        return x
