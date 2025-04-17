import logging
from pathlib import PosixPath

import numpy as np
import pytest
import torch

from dataeval.utils._array import channels_first_to_last, ensure_embeddings, flatten, to_numpy, to_numpy_iter


@pytest.mark.optional
class TestInterop:
    def test_torch_to_numpy(self):
        t = torch.tensor([1, 2, 3, 4, 5])
        n = to_numpy(t)
        assert list(n) == list(t)

    def test_torch_non_tensor_to_numpy(self):
        t = torch.int
        n = to_numpy(t)  # type: ignore
        assert n.shape == ()

    def test_to_numpy_iter(self):
        t = [torch.tensor([1]), torch.tensor([2, 3]), torch.tensor([4, 5, 6])]
        count = 0
        for n in to_numpy_iter(t):
            count += 1
            assert len(n) == count
            assert isinstance(n, np.ndarray)
        assert count == 3


@pytest.mark.required
class TestInteropLogging:
    def test_logging(self, tmp_path: PosixPath):
        log = logging.getLogger("dataeval")
        log.setLevel(logging.DEBUG)
        handler = logging.FileHandler(filename=tmp_path / "test.log", mode="w")
        formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
        handler.setFormatter(formatter)
        log.addHandler(handler)
        t = torch.tensor([1, 2, 3, 4, 5])
        to_numpy(t)
        assert (tmp_path / "test.log").exists()


array_native = [[0, 1], [2, 3]]
array_expected = np.asarray(array_native)


@pytest.mark.optional
class TestInteropArrayLike:
    @pytest.mark.parametrize(
        "param, expected",
        (
            (array_native, array_expected),
            (np.array(array_native), array_expected),
            (torch.Tensor(array_native), array_expected),
            (None, np.array([])),
        ),
    )
    def test_to_numpy(self, param, expected):
        np.testing.assert_equal(to_numpy(param), expected)


@pytest.mark.required
class TestEnsureEmbeddings:
    tt = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    na = np.array([[5, 6], [7, 8]], dtype=np.int64)

    def test_ensure_embeddings_torch(self):
        assert type(ensure_embeddings(self.tt) is torch.Tensor)

    def test_ensure_embeddings_torch_dtype(self):
        emb = ensure_embeddings(self.tt, dtype=torch.int32)
        assert emb.dtype == torch.int32
        assert type(emb) is torch.Tensor

    def test_ensure_embeddings_torch_unit_interval(self):
        with pytest.raises(ValueError):
            ensure_embeddings(self.tt, dtype=torch.int64, unit_interval=True)

    def test_ensure_embeddings_torch_force_unit_interval(self):
        emb = ensure_embeddings(self.tt, dtype=torch.float32, unit_interval="force")
        assert emb.dtype == torch.float32
        assert emb.min() >= 0.0
        assert emb.max() <= 1.0
        assert type(emb) is torch.Tensor

    def test_ensure_embeddings_numpy(self):
        assert type(ensure_embeddings(self.na) is np.ndarray)

    def test_ensure_embeddings_numpy_dtype(self):
        emb = ensure_embeddings(self.na, dtype=np.int32)
        assert emb.dtype == np.int32
        assert type(emb) is np.ndarray

    def test_ensure_embeddings_numpy_unit_interval(self):
        with pytest.raises(ValueError):
            ensure_embeddings(self.na, dtype=np.int64, unit_interval=True)

    def test_ensure_embeddings_numpy_force_unit_interval(self):
        emb = ensure_embeddings(self.na, dtype=np.float32, unit_interval="force")
        assert emb.dtype == np.float32
        assert emb.min() >= 0.0
        assert emb.max() <= 1.0
        assert type(emb) is np.ndarray


@pytest.mark.required
class TestFlatten:
    tt = torch.rand((4, 1, 16, 16))
    na = np.random.random((4, 1, 16, 16))

    def test_flatten_torch(self):
        flat = flatten(self.tt)
        assert flat.shape == (4, 256)
        assert type(flat) is torch.Tensor

    def test_flatten_numpy(self):
        flat = flatten(self.na)
        assert flat.shape == (4, 256)
        assert type(flat) is np.ndarray

    def test_invalid_input(self):
        with pytest.raises(TypeError):
            flatten("invalid input")  # type: ignore


@pytest.mark.required
class TestChannelsFirstToLast:
    tt = torch.rand((1, 4, 8))
    na = np.random.random((1, 4, 8))

    def test_channels_first_to_last_torch(self):
        flipped = channels_first_to_last(self.tt)
        assert flipped.shape == (4, 8, 1)

    def test_channels_first_to_last_numpy(self):
        flipped = channels_first_to_last(self.na)
        assert flipped.shape == (4, 8, 1)

    def test_invalid_input(self):
        with pytest.raises(TypeError):
            channels_first_to_last("invalid input")  # type: ignore
