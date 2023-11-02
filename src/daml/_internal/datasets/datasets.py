import numpy as np


class DamlDataset:
    """Holds the arrays of images and labels"""

    def __init__(self, X: np.ndarray, y: np.ndarray = np.array([])):
        self._set_data(X, y)

    def _set_data(self, X, y):
        self._images = X
        self._labels = y

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index):
        # If there are labels, return them
        if len(self._labels):
            return self._images[index], self._labels[index]
        # Else just return images
        return self._images[index]

    @property
    def images(self) -> np.ndarray:
        return self._images

    @property
    def labels(self) -> np.ndarray:
        return self._labels
