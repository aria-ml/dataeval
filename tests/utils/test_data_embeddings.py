from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from dataeval.utils.data import Embeddings, Images, Metadata, Targets
from dataeval.utils.data.datasets._types import ObjectDetectionTarget


@pytest.mark.required
class TestEmbeddings:
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

        em = Embeddings(ds, batch_size=64)  # type: ignore -> dont need to subclass from torch.utils.data.Dataset

        assert len(ds) == len(em)

    @pytest.mark.parametrize(
        "data, labels, metadata",
        [
            # Images, labels, metadata
            [np.ones((10, 3, 3)), np.ones((10, 3)), [{i: i} for i in range(10)]],
            # Images, ObjectDetectionTarget, Metadata
            [
                np.ones((10, 3, 3)),
                [ObjectDetectionTarget([[0, 1, 2, 3], [4, 5, 6, 7]], [0, 1], [1, 0]) for _ in range(10)],
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

        em = Embeddings(ds, batch_size=64)  # type: ignore -> dont need to subclass from torch.utils.data.Dataset

        assert len(ds) == len(em)

    @pytest.mark.parametrize(
        "data, targets",
        [
            [
                torch.ones((10, 1, 3, 3)),
                torch.nn.functional.one_hot(torch.arange(10)),
            ],
            [
                torch.ones((10, 1, 3, 3)),
                [ObjectDetectionTarget(torch.ones(10, 4), torch.zeros(10), torch.zeros(10)) for _ in range(10)],
            ],
        ],
    )
    def test_with_model_encode(self, data, targets):
        """Tests with basic identity model"""

        class TorchDataset(torch.utils.data.Dataset):
            metadata = {"id": 0, "index2label": {k: str(k) for k in range(10)}}

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

        ds = TorchDataset()

        em = Embeddings(ds, batch_size=64, model=IdentityModel(), device="cpu")  # type: ignore
        md = Metadata(ds)  # type: ignore

        assert len(ds) == len(em)
        assert len(em) == len(ds)

        for i in range(len(ds)):
            assert torch.allclose(em[i], data[i])

        for idx, e in enumerate(em):
            torch.allclose(ds[idx][0], e)

        assert isinstance(md.targets, Targets)

    def test_embeddings(self):
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.__getitem__.side_effect = lambda _: (np.zeros((3, 16, 16)), [], {})

        embs = Embeddings(mock_dataset, 10, model=torch.nn.Identity())
        assert isinstance(embs.to_tensor(), torch.Tensor)
        assert len(embs.to_tensor()) == len(embs)

        assert len(embs[0:3]) == 3
        assert np.array_equal(embs[0].cpu().numpy(), np.zeros((3, 16, 16)))

        for emb in embs:
            assert np.array_equal(emb.cpu().numpy(), np.zeros((3, 16, 16)))

        with pytest.raises(TypeError):
            embs["string"]  # type: ignore

    def test_images(self):
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.__getitem__.side_effect = lambda _: (np.zeros((3, 16, 16)), [], {})

        images = Images(mock_dataset)
        assert isinstance(images.to_list(), list)
        assert len(images.to_list()) == len(images)

        assert len(images[0:3]) == 3
        assert np.array_equal(images[0], np.zeros((3, 16, 16)))

        for image in images:
            assert np.array_equal(image, np.zeros((3, 16, 16)))

        with pytest.raises(TypeError):
            images["string"]  # type: ignore
