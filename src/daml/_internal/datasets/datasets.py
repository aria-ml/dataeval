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

    @images.setter
    def images(self, value: np.ndarray) -> None:
        self._images = value

    @property
    def labels(self) -> np.ndarray:
        if self._labels is None:
            return np.array([])
        else:
            return self._labels

    @labels.setter
    def labels(self, value: Optional[np.ndarray]) -> None:
        self._labels = value

    @property
    def boxes(self) -> np.ndarray:
        if self._boxes is None:
            return np.array([])
        else:
            return self._boxes

    @boxes.setter
    def boxes(self, value: Optional[np.ndarray]) -> None:
        self._boxes = value
