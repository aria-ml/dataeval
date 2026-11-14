"""Images that are tensors reach the statistics through the shared converter, so a CUDA item no longer raises.

A tensor that requires grad stands in for a CUDA tensor on a CPU-only runner: `np.asarray` refuses both, because each
needs `.detach().cpu()` first. The real CUDA case is the last test, which skips where there is no GPU.
"""

import numpy as np
import pytest
import torch

from dataeval.core import image_stats
from dataeval.data import unzip_dataset
from dataeval.flags import ImageStats
from dataeval.protocols import DatasetMetadata, DatumMetadata
from dataeval.quality import Duplicates, Outliers

pytestmark = pytest.mark.required

STATS = ImageStats.DIMENSION | ImageStats.PIXEL_MEAN
BOX = [[2, 2, 8, 8]]


def _arrays(n: int = 6) -> list[np.ndarray]:
    return [np.random.default_rng(i).random((3, 16, 16), dtype=np.float32) for i in range(n)]


def _needs_detach(arrays: list[np.ndarray]) -> list[torch.Tensor]:
    return [torch.from_numpy(a).requires_grad_(True) for a in arrays]


class _Images:
    """An image-classification dataset that returns whatever image objects it is given."""

    metadata = DatasetMetadata(id="tensor_items")

    def __init__(self, images) -> None:
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], np.array([1.0, 0.0]), DatumMetadata(id=i)


def _same_stats(left, right) -> None:
    assert left["image_count"] == right["image_count"]
    assert left["stats"].keys() == right["stats"].keys()
    for name, values in left["stats"].items():
        np.testing.assert_array_equal(values, right["stats"][name], err_msg=name)


def test_image_stats_takes_tensors_that_need_detach():
    arrays = _arrays()
    _same_stats(image_stats(_needs_detach(arrays), stats=STATS), image_stats(arrays, stats=STATS))


def test_image_stats_takes_tensors_that_need_detach_with_boxes():
    arrays = _arrays()
    boxes = [BOX] * len(arrays)
    _same_stats(
        image_stats(_needs_detach(arrays), boxes=boxes, stats=STATS),
        image_stats(arrays, boxes=boxes, stats=STATS),
    )


def test_unzip_dataset_takes_tensors_that_need_detach():
    arrays = _arrays()
    images, _ = unzip_dataset(_Images(_needs_detach(arrays)), per_target=False)
    for got, expected in zip(images, arrays, strict=True):
        np.testing.assert_array_equal(got, expected)


def test_outliers_and_duplicates_take_tensors_that_need_detach():
    arrays = _arrays(12)
    tensors = _Images(_needs_detach(arrays))
    host = _Images(arrays)
    assert Outliers().evaluate(tensors).data().equals(Outliers().evaluate(host).data())
    assert Duplicates().evaluate(tensors).data().equals(Duplicates().evaluate(host).data())


@pytest.mark.cuda
def test_cuda_items_give_the_same_stats_as_their_host_twin():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    arrays = _arrays(12)
    on_gpu = _Images([torch.from_numpy(a).cuda() for a in arrays])
    host = _Images(arrays)
    assert Outliers().evaluate(on_gpu).data().equals(Outliers().evaluate(host).data())
    assert Duplicates().evaluate(on_gpu).data().equals(Duplicates().evaluate(host).data())
