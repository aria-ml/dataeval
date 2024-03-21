from typing import Dict

import numpy as np


class MockImageClassificationDataset:
    """
    A class representing a mock image classification dataset

    :param images: An array of multi-dimensional arrays mocking an image
     format of Height, Width, Channels
    :type images: :class:`np.ndarray`
    :param labels: An array of ints representing the image labels
    :type labels :class:`np.ndarray`
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        """Constructor method"""
        self.images = images
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Returns an image-label pair specified by idx

        :param idx: Index of requested image and label
        :type idx: int

        :return: An image, label pair
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        # return (self.images[idx], self.labels[idx])
        return {"image": self.images[idx], "label": self.labels[idx]}

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self):
        yield from self.images


class MockObjectDetectionDataset:
    def __init__(self) -> None:
        pass

    def __getitem__(self, idx: int) -> None:
        pass

    def __len__(self) -> None:
        pass
