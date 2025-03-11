from __future__ import annotations

__all__ = []

from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from dataeval.utils.data.datasets._base import (
    BaseClassificationDataset,
    DataLocation,
)

CIFARClassStringMap = Literal["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
TCIFARClassMap = TypeVar("TCIFARClassMap", CIFARClassStringMap, int, list[CIFARClassStringMap], list[int])


class CIFAR10(BaseClassificationDataset):
    """
    `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset as Torch tensors.

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
    classes : "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck", \
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
            url="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
            filename="cifar-10-binary.tar.gz",
            md5=True,
            checksum="c32a1d4ab5d03f1284b67883e8d87530",
        ),
    ]

    index2label: dict[int, str] = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }
    label2index: dict[str, int] = {v: k for k, v in index2label.items()}

    def __init__(
        self,
        root: str | Path,
        download: bool = False,
        image_set: Literal["train", "test", "base"] = "train",
        size: int = -1,
        classes: TCIFARClassMap | None = None,
        unit_interval: bool = False,
        dtype: type | None = None,
        channels: Literal["channels_first", "channels_last"] = "channels_first",
        flatten: bool = False,
        crop: int | None = None,
        normalize: tuple[float, float] | None = None,
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
            crop,
            normalize,
            balance,
            slice_back,
            verbose,
        )

        self._resource: DataLocation = self._resources[0]
        self._filepaths: list[str]

        # Load the classes
        self.class_set: set[int] = self._reduce_classes(classes)
        self.num_classes: int = len(self.class_set)

        # Load the data
        self._filepaths, self._labels, self._datum_metadata = self._load_data()

        # Adjust the data as desired
        self._reorder = self._preprocess()

    def _load_data_inner(self) -> tuple[list[str], list[int], dict[str, Any]]:
        """Function to load in the file paths for the data and labels and retrieve metadata"""
        file_meta = {"batch_num": []}
        raw_data = []
        file_paths = []
        labels = []
        train_paths = []
        test_paths = []
        data_folder = self.dataset_dir / "cifar-10-batches-bin"
        save_folder = self.dataset_dir / "images"

        # Process each batch file, skipping .meta and .html files
        for entry in data_folder.iterdir():
            if entry.suffix == ".bin":
                batch_data, batch_labels = self._unpack_batch_files(entry)
                raw_data.append(batch_data)
                group = "train" if "test" not in entry.stem else "test"
                name_split = entry.stem.split("_")
                batch_num = int(name_split[-1]) - 1 if group == "train" else 5
                file_names = [
                    str(save_folder / f"{i + 10000 * batch_num:05d}_{self.index2label[label]}.png")
                    for i, label in enumerate(batch_labels)
                ]
                file_paths.extend(file_names)
                if group == "train":
                    train_paths.extend(file_names)
                else:
                    test_paths.extend(file_names)

                if group == self._image_set or self._image_set == "base":
                    labels.extend(batch_labels)
                    file_meta["batch_num"].extend([batch_num] * len(labels))

        # Stack and reshape images
        images = np.vstack(raw_data).reshape(-1, 3, 32, 32)

        # Create image_set_dict from batch type
        self._image_set_dict["train"] = train_paths
        self._image_set_dict["test"] = test_paths
        self._image_set_dict["base"] = file_paths

        # Save the raw data into images if not already there
        if not save_folder.exists():
            save_folder.mkdir(exist_ok=True)
            for i, file in enumerate(file_paths):
                Image.fromarray(images[i].transpose(1, 2, 0).astype(np.uint8)).save(file)

        # Returning just the desired image set
        if self._image_set == "base":
            data_paths = file_paths
        elif self._image_set == "train":
            data_paths = train_paths
        else:
            data_paths = test_paths

        return data_paths, labels, file_meta

    def _unpack_batch_files(self, file_path: Path) -> tuple[NDArray[Any], list[int]]:
        # Load pickle data with latin1 encoding
        with file_path.open("rb") as f:
            buffer = np.frombuffer(f.read(), "B")
            labels = buffer[::3073]
            pixels = np.delete(buffer, np.arange(0, buffer.size, 3073))
            images = pixels.reshape(-1, 3072)
        return images, labels.tolist()
