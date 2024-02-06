from typing import Optional, Tuple, Union

import numpy as np
from torch.utils.data import Dataset


class DamlDataset(Dataset):
    """Holds the arrays of images and labels"""

    def __init__(
        self,
        images: np.ndarray,
        labels: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
    ) -> None:
        self._images: np.ndarray = images
        self._labels = labels
        self._boxes = boxes

        self._validate()

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index) -> Union[np.ndarray, Tuple]:
        image: np.ndarray = self._images[index]

        # Return image if no other attributes
        if self._labels is None:
            return image

        labels: np.ndarray = self._labels[index]
        # Return image and label for image classification
        if self._boxes is None:
            return image, labels

        # Return images, labels, boxes for object detection
        boxes: np.ndarray = self.boxes[index]
        return image, labels, boxes

    @property
    def images(self) -> np.ndarray:
        return self._images

    @property
    def labels(self) -> np.ndarray:
        if self._labels is None:
            return np.array([])
        else:
            return self._labels

    @property
    def boxes(self) -> np.ndarray:
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
