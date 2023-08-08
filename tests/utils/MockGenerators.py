
from typing import Union, Sequence, Tuple, Optional, List
import numpy as np

from MockDatasets import MockImageClassificationDataset

class MockImageClassificationGenerator:
    """A class that creates a mock dataset as a :class:`MockImageClassificationDataset`

    This class creates arrays of multi-dimensional arrays representing a list of images and 
    their corresponding labels. They are then stored as :class:`MockImageClassificationDataset` 
    and exposed with the attribute `dataset`
    
    :param limit: The total number of images and labels in the dataset
    :type limit: int
    :param labels: The values of labels for the dataset
    :type labels: int, Tuple, Sequence, np.ndarray
    :param img_dims: The height, width (optional) of a single image
    :type img_dims: int, Tuple, Sequence, np.ndarray
    :param channels: The number of channels for the image, default is 1
    :type channels: int, optional
    """
    def __init__(self, 
                 limit: int, 
                 labels: Union[int, Sequence[int]], 
                 img_dims: Union[int, Sequence[int]],
                 channels: int = 1) -> None:
        super()

        self._limit = limit
        if isinstance(labels, int):
            labels = [labels]
        self._labels = np.array(labels, dtype=int)
        self._num_labels = 1 if isinstance(labels, int) else len(self._labels)

        if isinstance(img_dims, int):
            img_dims = [img_dims]
        self._img_dims = self._set_dims(img_dims, channels)
        self._create_dataset()

    @property
    def dataset(self) -> MockImageClassificationDataset:
        """Returns the created dataset
        
        :return: A dataset containing images and labels based on given parameters
        :rtype: :class:`MockImageClassificationDataset`
        """
        return self._dataset    
    
    def _create_dataset(self) -> None:
        if self._num_labels > 1:    
            images, labels = self._create_data()
        else:
            images = np.ones(shape=(self._limit, *self._img_dims))
            labels = np.ones(shape=(self._limit, 1)) * self._labels[0]

        self._dataset = MockImageClassificationDataset(images, labels)
    
    def _create_data(self) -> Tuple[np.ndarray, np.ndarray]:
        # Create an index for each label
        mock_data = np.ones(shape=(self._limit, 32, 32, 3))
        mock_labels = np.ones(shape=(self._limit, 1), dtype=int)

        mock_data = np.array_split(mock_data, self._num_labels)
        mock_labels = np.array_split(mock_labels, self._num_labels)

        assert(len(mock_data) == self._num_labels)
        assert(len(mock_labels) == self._num_labels)

        mock_data = np.concatenate([x*label for x, label in zip(mock_data, self._labels)])
        mock_labels = np.concatenate([x*label for x, label in zip(mock_labels, self._labels)])

        return (mock_data, mock_labels)

    def _set_dims(self, 
                  dims: Union[int, Sequence[int]], 
                  channels: int = 1
                  ) -> Tuple[int]:

        if isinstance(dims, int):
            dims = [dims]

        dim_size = len(dims)
        if dim_size == 3:
            return tuple(dims)
        
        assert(0 < dim_size < 3)

        new_dims = [0, 0, 0]
        new_dims[0] = dims[0]
        new_dims[2] = channels
        if dim_size < 2:   # If 1 dimension, make image square
            new_dims[1] = dims[0]
        else:
            new_dims[1] = dims[1]

        return tuple(new_dims)

if __name__ == "__main__":
    # TODO Add simple usage case as an example
    gen = MockImageClassificationGenerator(50000, 1, 32)
    ds = gen.dataset
