import numpy as np
import pytest
import torch

from dataeval.utils.data._collate import collate
from dataeval.utils.data.datasets._types import ObjectDetectionTarget


@pytest.mark.required
class TestCollateDataset:
    """
    Test collate aggregates MAITE style data into separate collections from tuple return
    """

    @pytest.mark.parametrize(
        "data, labels",
        [
            # # Returns two lists
            [[0, 1, 2], [3, 4, 5]],  # List of scalar pairs e.g. preds & scores
            [np.ones((10, 3, 3)), np.ones((10,))],  # Array of images and labels
            [np.ones((10, 3, 3)), np.ones((10, 3, 3))],  # Array of images and images (AE)
        ],
    )
    def test_double_input(self, data, labels):
        """Tests common (input, target) dataset output"""

        class ICDataset:
            """Basic form of Image Classification Dataset"""

            def __init__(self, data, labels):
                self.data = data
                self.labels = labels

            def __getitem__(self, idx):
                return self.data[idx], [0.0, 0.0, 1.0], {"meta": 0}

            def __len__(self) -> int:
                return len(self.data)

        ds = ICDataset(data, labels)

        result = collate(ds)  # type: ignore -> dont need to subclass from torch.utils.data.Dataset

        assert isinstance(result, tuple)
        assert len(result) == 3

    @pytest.mark.parametrize(
        "data, labels, metadata",
        [
            # Images, labels, metadata
            [np.ones((10, 3, 3)), np.ones((10, 3)), [{i: i} for i in range(10)]],
            # Images, ObjectDetectionTarget, Metadata
            [
                np.ones((10, 3, 3)),
                [ObjectDetectionTarget([[0, 1, 2], [3, 4, 5]], [0, 1, 2], []) for _ in range(10)],
                [{i: i} for i in range(10)],
            ],
        ],
    )
    def test_triple_input(self, data, labels, metadata):
        """Tests common (input, target, metadata) dataset output"""

        class ODDataset:
            """Basic form of Object Detection Dataset"""

            def __init__(self, data, labels, metadata):
                self.data = data
                self.labels = labels
                self.metadata = metadata

            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx], self.metadata[idx]

            def __len__(self) -> int:
                return len(self.data)

        ds = ODDataset(data, labels, metadata)

        result = collate(ds)  # type: ignore -> dont need to subclass from torch.utils.data.Dataset

        assert isinstance(result, tuple)
        assert len(result) == 3

    @pytest.mark.parametrize(
        "data, targets",
        [
            [
                torch.ones((10, 3, 3)),
                torch.nn.functional.one_hot(torch.arange(10)),
            ],
            [
                torch.ones((10, 3, 3)),
                [ObjectDetectionTarget(torch.ones((10, 4)), torch.arange(10), torch.arange(10)) for _ in range(10)],
            ],
        ],
    )
    def test_with_model_encode(self, data, targets):
        """Tests with basic identity model"""

        class TorchDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.data = data
                self.targets = targets

            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx], {"meta": idx}

            def __len__(self):
                return len(self.data)

        class IdentityModel(torch.nn.Module):
            def forward(self, x):
                return x

            def encode(self, x):
                return x

        result = collate(TorchDataset(), model=IdentityModel())  # type: ignore

        assert len(result) == 3
