from __future__ import annotations

__all__ = []

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

from numpy.typing import NDArray

from dataeval.utils.datasets._base import BaseODDataset, DataLocation
from dataeval.utils.datasets._mixin import BaseDatasetNumpyMixin

if TYPE_CHECKING:
    from dataeval.typing import Transform


class MILCO(BaseODDataset[NDArray[Any]], BaseDatasetNumpyMixin):
    """
    A side-scan sonar dataset focused on mine-like object detection.

    The dataset comes from the paper
    `Side-scan sonar imaging data of underwater vehicles for mine detection <https://doi.org/10.1016/j.dib.2024.110132>`_
    by N.P. Santos et. al. (2024).

    The full dataset contains 1170 side-scan sonar images collected using a 900-1800 kHz Marine Sonic
    dual frequency side-scan sonar of a Teledyne Marine Gavia Autonomous Underwater Vehicle.
    All the images were carefully analyzed and annotated, including the image coordinates of the
    Bounding Box (BB) of the detected objects divided into NOn-Mine-like BOttom Objects (NOMBO)
    and MIne-Like COntacts (MILCO) classes.

    This dataset is consists of 345 images from 2010, 120 images from 2015, 93 images from 2017, 564 images from 2018,
    and 48 images from 2021). In these 1170 images, there are 432 MILCO objects, and 235 NOMBO objects.
    The class “0” corresponds to a MILCO object and the class “1” corresponds to a NOMBO object.
    The raw BB coordinates provided in the downloaded text files are (x, y, w, h),
    given as percentages of the image (x_BB = x/img_width, y_BB = y/img_height, etc.).
    The images come in 2 sizes, 416 x 416 or 1024 x 1024.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory where the data should be downloaded to or the ``milco`` folder of the already downloaded data.
    image_set: "train", "operational", or "base", default "train"
        If "train", then the images from 2015, 2017 and 2021 are selected,
        resulting in 315 MILCO objects and 177 NOMBO objects.
        If "operational", then the images from 2010 and 2018 are selected,
        resulting in 117 MILCO objects and 58 NOMBO objects.
        If "base", then the full dataset is selected.
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
    image_set : "train", "operational" or "base"
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
    Data License: `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_
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
        DataLocation(
            url="https://figshare.com/ndownloader/files/43169008",
            filename="2010.zip",
            md5=True,
            checksum="43347a0cc383c0d3dbe0d24ae56f328d",
        ),
        DataLocation(
            url="https://figshare.com/ndownloader/files/43169011",
            filename="2018.zip",
            md5=True,
            checksum="25d091044a10c78674fedad655023e3b",
        ),
    ]

    index2label: dict[int, str] = {
        0: "MILCO",
        1: "NOMBO",
    }

    def __init__(
        self,
        root: str | Path,
        image_set: Literal["train", "operational", "base"] = "train",
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
        self._bboxes_per_size = True

    def _load_data(self) -> tuple[list[str], list[str], dict[str, list[Any]]]:
        filepaths: list[str] = []
        targets: list[str] = []
        datum_metadata: dict[str, list[Any]] = {}
        metadata_list: list[dict[str, Any]] = []
        image_sets: dict[str, list[int]] = {
            "base": list(range(len(self._resources))),
            "train": list(range(3)),
            "operational": list(range(3, len(self._resources))),
        }

        # Load the data
        resource_indices = image_sets[self.image_set]
        for idx in resource_indices:
            self._resource = self._resources[idx]
            filepath, target, metadata = super()._load_data()
            filepaths.extend(filepath)
            targets.extend(target)
            metadata_list.append(metadata)

        # Adjust datum metadata to correct format
        for data_dict in metadata_list:
            for key, val in data_dict.items():
                if key not in datum_metadata:
                    datum_metadata[str(key)] = []
                datum_metadata[str(key)].extend(val)

        return filepaths, targets, datum_metadata

    def _load_data_inner(self) -> tuple[list[str], list[str], dict[str, Any]]:
        file_data = {"year": [], "image_id": [], "data_path": [], "label_path": []}
        data_folder = sorted((self.path / self._resource.filename[:-4]).glob("*.jpg"))
        if not data_folder:
            raise FileNotFoundError

        for entry in data_folder:
            # Remove file extension and split by "_"
            parts = entry.stem.split("_")
            file_data["image_id"].append(parts[0])
            file_data["year"].append(parts[1])
            file_data["data_path"].append(str(entry))
            file_data["label_path"].append(str(entry.parent / entry.stem) + ".txt")
        data = file_data.pop("data_path")
        annotations = file_data.pop("label_path")

        return data, annotations, file_data

    def _read_annotations(self, annotation: str) -> tuple[list[list[float]], list[int], dict[str, Any]]:
        """Function for extracting the info out of the text files"""
        labels: list[int] = []
        boxes: list[list[float]] = []
        with open(annotation) as f:
            for line in f.readlines():
                out = line.strip().split()
                labels.append(int(out[0]))

                xcenter, ycenter, width, height = [float(out[1]), float(out[2]), float(out[3]), float(out[4])]

                x0 = xcenter - width / 2
                x1 = x0 + width
                y0 = ycenter - height / 2
                y1 = y0 + height
                boxes.append([x0, y0, x1, y1])

        return boxes, labels, {}
