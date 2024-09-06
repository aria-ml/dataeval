from typing import Any, Tuple, Union

import numpy as np
from torch.utils.data import Dataset


class DataEvalDataset(Dataset):
    """Holds the arrays of images and labels"""

    def __init__(
        self,
        images: Any,
        labels: Any = None,
        boxes: Any = None,
    ) -> None:
        self._images: Any = images
        self._labels = labels
        self._boxes = boxes

        self._validate()

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index) -> Union[Any, Tuple]:
        image: Any = self._images[index]

        # Return image if no other attributes
        if self._labels is None:
            return image

        labels: Any = self._labels[index]
        # Return image and label for image classification
        if self._boxes is None:
            return image, labels

        # Return images, labels, boxes for object detection
        boxes: Any = self.boxes[index]
        return image, labels, boxes

    @property
    def images(self) -> Any:
        return self._images

    @property
    def labels(self) -> Any:
        if self._labels is None:
            return np.array([])
        else:
            return self._labels

    @property
    def boxes(self) -> Any:
        if self._boxes is None:
            return np.array([])
        else:
            return self._boxes

    def _validate(self):
        if self._labels is not None:
            if self._boxes is None:
                assert len(self._images) == len(self._labels)
            else:
                assert len(self._boxes) == len(self._labels)
        else:
            assert self._boxes is None
