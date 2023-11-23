import numpy as np


class DamlDataset:
    """Holds the arrays of images and labels"""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray = np.array([]),
    ):
        self._set_data(X, y)

    def _set_data(self, X, y):
        self._images = X
        self._labels = y

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index):
        item = (self._images[index],)

        if len(self._labels):
            item += self._labels[index]

        return item

    @property
    def images(self) -> np.ndarray:
        return self._images

    @property
    def labels(self) -> np.ndarray:
        return self._labels
