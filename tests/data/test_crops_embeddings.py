"""Crops flow through ``Embeddings``: crops of different sizes batch, and a second read extracts nothing."""

from typing import Any

import numpy as np
import pytest
import torch

from dataeval import Embeddings
from dataeval.data import DetectionCrops, SequenceFrames
from dataeval.extractors import TorchExtractor
from tests.data.test_crops import _tracking_dataset

# The frame a crop was cut from, as `_tracking_dataset` lays them out. Every pixel of fake frame `f` is `f`, so a
# crop's mean says which frame its pixels came from, whatever its size and padding.
FRAME_OF_EACH_CROP = [0, 1, 1, 2, 2, 0, 0, 1, 0, 0]


def _resize_to_8(crop: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.interpolate(crop[None].float(), size=(8, 8), mode="nearest")[0]


class _CountsImages:
    """An extractor that reports how many images it was asked to embed, and returns each image's mean."""

    def __init__(self) -> None:
        self.images = 0

    def __call__(self, data: Any) -> np.ndarray:
        self.images += len(data)
        return np.stack([np.full(2, np.asarray(image, dtype=np.float32).mean()) for image in data])


@pytest.mark.required
class TestCropsThroughEmbeddings:
    def test_crops_of_different_sizes_batch_through_a_resizing_extractor(self):
        crops = DetectionCrops(SequenceFrames(_tracking_dataset()), min_size=4)
        sizes = {crops[i][0].shape[1:] for i in range(len(crops))}
        assert len(sizes) > 1  # the crops differ in size, so a batch needs the resize

        model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        extractor = TorchExtractor(model, transforms=[_resize_to_8], device="cpu")
        rows = np.asarray(Embeddings(crops, extractor=extractor, batch_size=4))

        assert rows.shape == (10, 3)
        assert rows[:, 0].tolist() == FRAME_OF_EACH_CROP

    def test_a_second_read_of_one_embeddings_makes_no_extractor_calls(self):
        crops = DetectionCrops(SequenceFrames(_tracking_dataset()), min_size=4)
        extractor = _CountsImages()
        embeddings = Embeddings(crops, extractor=extractor, batch_size=4)

        first = np.asarray(embeddings)
        assert extractor.images == len(crops) == 10
        second = np.asarray(embeddings)

        assert extractor.images == 10
        assert np.array_equal(first, second)
