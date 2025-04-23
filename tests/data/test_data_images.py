from unittest.mock import MagicMock

import numpy as np
import pytest

from dataeval.data import Images


def get_dataset(size: int = 10):
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = size
    mock_dataset.__getitem__.side_effect = lambda _: (np.zeros((3, 16, 16)), [], {})
    return mock_dataset


@pytest.mark.required
class TestImages:
    def test_images(self):
        images = Images(get_dataset())
        assert isinstance(images.to_list(), list)
        assert len(images.to_list()) == len(images)
        assert len(images[0:3]) == 3
        for image in images:
            assert np.array_equal(image, np.zeros((3, 16, 16)))

        with pytest.raises(TypeError):
            images["string"]  # type: ignore

    def test_images_on_image_only_dataset(self):
        mock_dataset = get_dataset()
        mock_dataset.__getitem__.side_effect = lambda _: np.zeros((3, 16, 16))
        images = Images(mock_dataset)
        assert isinstance(images.to_list(), list)
        assert len(images.to_list()) == len(images)
        assert len(images[0:3]) == 3
        for image in images:
            assert np.array_equal(image, np.zeros((3, 16, 16)))

    @pytest.mark.requires_all
    def test_images_plot(self):
        images = Images(get_dataset())
        assert images.plot([0]) is not None
