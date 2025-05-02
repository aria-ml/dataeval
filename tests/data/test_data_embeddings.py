from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from dataeval.data import Embeddings, Metadata, Targets
from dataeval.typing import DatasetMetadata
from dataeval.utils.datasets._types import ObjectDetectionTarget


class MockDataset:
    metadata = DatasetMetadata({"id": "mock_dataset"})

    def __init__(self, data, targets, metadata=None):
        self.data = data
        self.targets = targets
        self.datum_metadata = metadata

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], self.datum_metadata[idx] if self.datum_metadata else {"id": idx}

    def __len__(self) -> int:
        return len(self.data)


class TorchDataset(torch.utils.data.Dataset):
    metadata = DatasetMetadata({"id": "torch_dataset", "index2label": {k: str(k) for k in range(10)}})

    def __init__(self, data, targets):
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


def get_dataset(size: int = 10):
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = size
    mock_dataset.__getitem__.side_effect = lambda _: (np.zeros((3, 16, 16)), [], {})
    return mock_dataset


@pytest.fixture(scope="module")
def torch_ic_ds():
    return TorchDataset(torch.ones((10, 1, 3, 3)), torch.nn.functional.one_hot(torch.arange(10)))


@pytest.mark.required
class TestEmbeddings:
    """
    Test collate aggregates MAITE style data into separate collections from tuple return
    """

    @pytest.mark.parametrize(
        "data, labels, metadata",
        [
            [[0, 1, 2], [3, 4, 5], None],
            [np.ones((10, 3, 3)), np.ones((10,)), None],
            [np.ones((10, 3, 3)), np.ones((10, 3, 3)), None],
            [np.ones((10, 3, 3)), np.ones((10, 3)), [{i: i} for i in range(10)]],
            [
                np.ones((10, 3, 3)),
                [ObjectDetectionTarget([[0, 1, 2, 3], [4, 5, 6, 7]], [0, 1], [1, 0]) for _ in range(10)],
                [{i: i} for i in range(10)],
            ],
        ],
    )
    def test_mock_inputs(self, data, labels, metadata):
        """Tests common (input, target, metadata) dataset output"""
        ds = MockDataset(data, labels, metadata)
        em = Embeddings(ds, batch_size=64)

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
        ds = TorchDataset(data, targets)
        em = Embeddings(ds, batch_size=64, model=IdentityModel(), device="cpu")
        md = Metadata(ds)

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

    def test_embeddings_embeddings_only_no_embeddings(self):
        embs = Embeddings([], 1)
        embs._embeddings_only = True
        with pytest.raises(ValueError):
            embs[0]

    def test_embeddings_set_cache_bool(self, torch_ic_ds):
        embs = Embeddings(torch_ic_ds, batch_size=64, model=IdentityModel(), device="cpu", cache=True)
        embs[:2]
        assert len(embs._cached_idx) == 2
        embs.cache = False
        embs[:2]
        assert len(embs._cached_idx) == 0
        embs.cache = True
        embs[:2]
        assert len(embs._cached_idx) == 2

    def test_embeddings_set_cache_path(self, torch_ic_ds, tmp_path):
        embs = Embeddings(torch_ic_ds, batch_size=64, model=IdentityModel(), device="cpu")
        embs[:2]
        assert len(embs._cached_idx) == 0
        file = tmp_path / "test.pt"
        embs.cache = file
        embs[:2]
        assert file.exists()
        assert len(embs._cached_idx) == 2
        path = tmp_path
        embs.cache = str(path)
        embs[:2]
        assert (path / f"emb-{hash(embs)}.pt").exists()
        assert len(embs._cached_idx) == 2

    def test_embeddings_cache_embeddings_only_to_disk(self, tmp_path):
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        embs = Embeddings.from_array(arr)
        embs.cache = tmp_path
        assert hash(embs)
        assert (tmp_path / f"emb-{hash(embs)}.pt").exists()

    def test_embeddings_cache_to_disk(self, torch_ic_ds, tmp_path):
        original = Embeddings(torch_ic_ds, batch_size=64, model=IdentityModel(), device="cpu", cache=tmp_path)
        original_values = original.to_numpy()
        digest = hash(original)
        file = tmp_path / f"emb-{digest}.pt"
        assert file.exists()

        from_file = Embeddings.load(file)
        assert np.array_equal(original_values, from_file.to_numpy())

    def test_embeddings_partial_cache_to_disk(self, torch_ic_ds, tmp_path):
        original = Embeddings(torch_ic_ds, batch_size=64, model=IdentityModel(), device="cpu", cache=tmp_path)
        original_values = original[:5].numpy()
        digest = hash(original)
        file = tmp_path / f"emb-{digest}.pt"
        assert file.exists()

        from_file = Embeddings.load(file)
        assert np.array_equal(original_values, from_file[:5].numpy())
        with pytest.raises(ValueError):
            from_file[5:]
        with pytest.raises(ValueError):
            from_file.to_tensor()

    def test_embeddings_save_and_load(self, torch_ic_ds, tmp_path):
        original = Embeddings(torch_ic_ds, batch_size=64, model=IdentityModel(), device="cpu")
        file = tmp_path / "file.pt"
        original.save(file)
        assert file.exists()
        original_values = original.to_numpy()
        from_file = Embeddings.load(file)
        assert np.array_equal(original_values, from_file.to_numpy())

    def test_embeddings_load_file_not_found(self, tmp_path):
        bad_file = tmp_path / "non_existant.pt"
        with pytest.raises(FileNotFoundError):
            Embeddings.load(bad_file)
        with pytest.raises(FileNotFoundError):
            Embeddings.load(str(bad_file))

    def test_embeddings_new(self, torch_ic_ds):
        embs = Embeddings(torch_ic_ds, batch_size=64, model=IdentityModel(), device="cpu", transforms=lambda x: x + 1)
        mini_ds = TorchDataset(torch.ones((5, 1, 3, 3)), torch.nn.functional.one_hot(torch.arange(5)))
        mini_embs = embs.new(mini_ds)
        assert mini_embs.batch_size == embs.batch_size
        assert mini_embs.device == embs.device
        assert len(mini_embs) == 5
        assert mini_embs._dataset != embs._dataset
        assert mini_embs._transforms == embs._transforms
        assert mini_embs._model == embs._model

    @patch("dataeval.data._embeddings.torch.load", side_effect=OSError())
    def test_embeddings_load_failure(self, tmp_path):
        test_path = tmp_path / "mock.pt"
        test_path.touch()
        with pytest.raises(OSError):
            Embeddings.load(tmp_path)

    def test_embeddings_new_embeddings_only_raises(self):
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        embs = Embeddings.from_array(arr)
        with pytest.raises(ValueError):
            embs.new([])
