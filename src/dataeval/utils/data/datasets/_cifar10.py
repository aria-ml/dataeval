from __future__ import annotations

__all__ = []

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from dataeval.utils.data.datasets._base import BaseICDataset, DataLocation
from dataeval.utils.data.datasets._mixin import BaseDatasetNumpyMixin

if TYPE_CHECKING:
    from dataeval.typing import Transform

CIFARClassStringMap = Literal["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
TCIFARClassMap = TypeVar("TCIFARClassMap", CIFARClassStringMap, int, list[CIFARClassStringMap], list[int])


class CIFAR10(BaseICDataset[NDArray[Any]], BaseDatasetNumpyMixin):
    """
    `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset as NumPy arrays.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory of dataset where the ``mnist`` folder exists.
    download : bool, default False
        If True, downloads the dataset from the internet and puts it in root directory.
        Class checks to see if data is already downloaded to ensure it does not create a duplicate download.
    image_set : "train", "test" or "base", default "train"
        If "base", returns all of the data to allow the user to create their own splits.
    transforms : Transform, Sequence[Transform] or None, default None
        Transform(s) to apply to the data.
    verbose : bool, default False
        If True, outputs print statements.

    Attributes
    ----------
    path : pathlib.Path
        Location of the folder containing the data.
    image_set : "train", "test" or "base"
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

    def __init__(
        self,
        root: str | Path,
        download: bool = False,
        image_set: Literal["train", "test", "base"] = "train",
        transforms: Transform[NDArray[Any]] | Sequence[Transform[NDArray[Any]]] | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            root,
            download,
            image_set,
            transforms,
            verbose,
        )

    def _load_data_inner(self) -> tuple[list[str], list[int], dict[str, Any]]:
        """Function to load in the file paths for the data and labels and retrieve metadata"""
        file_meta = {"batch_num": []}
        raw_data = []
        labels = []
        data_folder = self.path / "cifar-10-batches-bin"
        save_folder = self.path / "images"
        image_sets: dict[str, list[str]] = {"base": [], "train": [], "test": []}

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
                image_sets["base"].extend(file_names)
                image_sets[group].extend(file_names)

                if self.image_set in (group, "base"):
                    labels.extend(batch_labels)
                    file_meta["batch_num"].extend([batch_num] * len(labels))

        # Stack and reshape images
        images = np.vstack(raw_data).reshape(-1, 3, 32, 32)

        # Save the raw data into images if not already there
        if not save_folder.exists():
            save_folder.mkdir(exist_ok=True)
            for i, file in enumerate(image_sets["base"]):
                Image.fromarray(images[i].transpose(1, 2, 0).astype(np.uint8)).save(file)

        return image_sets[self.image_set], labels, file_meta

    def _unpack_batch_files(self, file_path: Path) -> tuple[NDArray[Any], list[int]]:
        # Load pickle data with latin1 encoding
        with file_path.open("rb") as f:
            buffer = np.frombuffer(f.read(), "B")
            labels = buffer[::3073]
            pixels = np.delete(buffer, np.arange(0, buffer.size, 3073))
            images = pixels.reshape(-1, 3072)
        return images, labels.tolist()
