from __future__ import annotations

__all__ = []

from pathlib import Path
from typing import Any, Callable, Literal
from xml.etree.ElementTree import parse

import torch
from torchvision.datasets import VOCDetection as _VOCDetection
from torchvision.transforms import v2

from dataeval.utils.data.datasets._types import InfoMixin, ObjectDetectionDataset, ObjectDetectionTarget


class VOCDetection(ObjectDetectionDataset[torch.Tensor], InfoMixin):
    """
    `Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory of the VOC Dataset.
    year : "2007", "2008", "2009", "2010", "2011" or "2012", default "2012"
        The dataset year.
    image_set : "train", "trainval", "val", or "test", default "train"
        "test" is only valid for the year "2007"
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

    def __init__(
        self,
        root: str | Path,
        year: Literal["2007", "2008", "2009", "2010", "2011", "2012"] = "2012",
        image_set: Literal["train", "trainval", "val", "test"] = "train",
        download: bool = False,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
    ) -> None:
        if transform is None:
            transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self._data = _VOCDetection(root, year, image_set, download, transform, target_transform, transforms)
        self._image_set = image_set

        # pull out an alphabetized list of labels
        labels: set[str] = set()
        for i in range(len(self._data)):
            objects = self._data.parse_voc_xml(parse(self._data.annotations[i]).getroot())["annotation"]["object"]
            labels.update([o["name"] for o in objects])
            if len(objects) == 20:
                break
        self.classes: list[str] = sorted(labels)

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
