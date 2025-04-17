from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from dataeval.utils.data import Embeddings, Metadata, Targets
from dataeval.utils.data.datasets._types import ObjectDetectionTarget


def get_dataset(size: int = 10):
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = size
    mock_dataset.__getitem__.side_effect = lambda _: (np.zeros((3, 16, 16)), [], {})
    return mock_dataset


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
        embs = Embeddings(get_dataset(), 10, model=torch.nn.Identity(), transforms=lambda x: x + 1)
        assert len(embs[0:3]) == 3

        embs_tt = embs.to_tensor()
        assert isinstance(embs_tt, torch.Tensor)
        assert len(embs_tt) == len(embs)

        embs_np = embs.to_numpy()
        assert isinstance(embs_np, np.ndarray)
        assert len(embs_np) == len(embs)

        for emb in embs:
            assert np.array_equal(emb.cpu().numpy(), np.ones((3, 16, 16)))

        with pytest.raises(TypeError):
            embs["string"]  # type: ignore

    def test_embeddings_cache(self):
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.__getitem__.side_effect = lambda _: (np.zeros((3, 16, 16)), [], {})

        embs = Embeddings(mock_dataset, 10, model=torch.nn.Identity(), transforms=lambda x: x + 1, cache=True)
        assert not embs._embeddings.shape

        # instantiate mixed embeddings
        part1 = embs[0:4]
        assert isinstance(part1, torch.Tensor)
        assert part1.shape == (4, 3, 16, 16)

        assert isinstance(embs._embeddings, torch.Tensor)
        assert len(embs._embeddings) == 10
        assert embs._cached_idx == {0, 1, 2, 3}
        assert np.array_equal(embs._embeddings[0:4], np.ones((4, 3, 16, 16)))

        # zero out remaining uninitialized embeddings
        embs._embeddings[4:10] = 0

        part2 = embs[2:7]
        assert isinstance(part2, torch.Tensor)
        assert part2.shape == (5, 3, 16, 16)
        assert np.array_equal(embs._embeddings[2:7], np.ones((5, 3, 16, 16)))
        assert embs._cached_idx == {0, 1, 2, 3, 4, 5, 6}

        part3 = embs[9]
        assert isinstance(part3, torch.Tensor)
        assert part3.shape == (3, 16, 16)
        assert np.array_equal(embs._embeddings[9], np.ones((3, 16, 16)))
        assert embs._cached_idx == {0, 1, 2, 3, 4, 5, 6, 9}

        assert np.array_equal(embs._embeddings[7:9], np.zeros((2, 3, 16, 16)))

    def test_embeddings_cache_hit(self):
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.__getitem__.side_effect = lambda _: (np.zeros((3, 16, 16)), [], {})

        embs = Embeddings(mock_dataset, 10, model=torch.nn.Identity(), transforms=lambda x: x + 1, cache=True)
        t1 = embs.to_tensor()
        assert isinstance(t1, torch.Tensor)
        assert len(t1) == 10
        assert np.array_equal(t1, np.ones((10, 3, 16, 16)))

        embs._embeddings[0:10] = 0
        t2 = embs.to_tensor()
        assert np.array_equal(t2, np.zeros((10, 3, 16, 16)))

    def test_embeddings_from_array(self):
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        embs = Embeddings.from_array(arr)
        assert isinstance(embs, Embeddings)
        assert len(embs) == arr.shape[0]
        assert np.array_equal(embs.to_tensor().numpy(), arr)

    def test_embeddings_shallow_no_embeddings(self):
        embs = Embeddings([], 1)
        embs._shallow = True
        with pytest.raises(ValueError):
            embs[0]
