from typing import Dict, Tuple

import maite.protocols
import maite.protocols.image_classification as ic
import numpy as np
import numpy.testing as npt
import pytest
import torch

from dataeval._internal.maite.utils import arraylike_to_numpy, extract_to_numpy


# TODO: Generalize types for all ArrayLike checks
# TODO: Make fixture for all ArrayLikes
class TestArrayLikeConversions:
    @pytest.mark.parametrize(
        "xp",
        [
            np.arange(10),
            torch.arange(10),
        ],
    )
    def test_valid_to_numpy(self, xp):
        "Test common arraylike datatypes that can be converted to numpy"
        assert isinstance(xp, maite.protocols.ArrayLike)
        result = arraylike_to_numpy(xp)
        assert isinstance(result, np.ndarray)

    @pytest.mark.parametrize(
        "xp_invalid, expected_error",
        [
            ([[], [1], [2, 2]], ValueError),
            ([np.arange(i) for i in range(5)], ValueError),
            ([torch.arange(i) for i in range(5)], ValueError),
        ],
    )
    def test_invalid_to_numpy(self, xp_invalid, expected_error):
        """Test known failing datatypes fail to be converted to numpy"""

        assert not isinstance(xp_invalid, maite.protocols.ArrayLike)

        with pytest.raises(expected_error):
            arraylike_to_numpy(xp_invalid)


class MockDataset:
    def __init__(self, images: maite.protocols.ArrayLike, labels: maite.protocols.ArrayLike):
        self.images = images
        self.labels = labels

    def __getitem__(self, idx) -> Tuple[maite.protocols.ArrayLike, maite.protocols.ArrayLike, Dict]:
        return self.images[idx], self.labels[idx], {}  # type: ignore

    def __len__(self) -> int:
        return len(self.images)  # type: ignore


class MockInvalidDataset:
    def __init__(self, images: maite.protocols.ArrayLike):
        self.images = images

    def __getitem__(self, idx) -> Tuple[maite.protocols.ArrayLike]:
        return self.images[idx]  # type: ignore

    def __len__(self) -> int:
        return len(self.images)  # type: ignore


@pytest.mark.parametrize(
    "images, labels",
    [
        (np.ones(shape=(3, 5, 5)), np.ones(shape=(3,))),
        (torch.ones(size=(3, 5, 5)), torch.ones(size=(3,))),
    ],
)
class TestExtractFromDataset:
    def test_extract_equals_dataset(self, images, labels):
        """Tests that splitting images and labels does not modify the data"""
        dataset: ic.Dataset = MockDataset(images, labels)  # Should be no static typecheck error

        extracted_images, extracted_labels = extract_to_numpy(dataset)

        assert len(extracted_images) == len(images)
        assert len(extracted_labels) == len(labels)

        for img1, img2 in zip(extracted_images, images):
            npt.assert_array_equal(img1, img2)

        for lbl1, lbl2 in zip(extracted_labels, labels):
            npt.assert_array_equal(lbl1, lbl2)

    def test_extract_correct_types(self, images, labels):
        """Tests that extracting correctly casts to type"""
        dataset: ic.Dataset = MockDataset(images, labels)

        extracted_images, extracted_labels = extract_to_numpy(dataset)

        assert isinstance(extracted_images, np.ndarray)
        assert isinstance(extracted_labels, np.ndarray)

    def test_not_ic_dataset(self, images, labels):
        dataset: ic.Dataset = MockInvalidDataset(images)  # type: ignore

        with pytest.raises(ValueError):
            extract_to_numpy(dataset)

    def test_non_iterable_type(self, images, labels):
        """Non iterable types fail"""
        with pytest.raises(TypeError):
            extract_to_numpy(1.0)  # type: ignore

    def test_unpackable_types(self, images, labels):
        with pytest.raises(TypeError):
            extract_to_numpy(range(10))  # type: ignore
