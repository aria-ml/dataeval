import numpy as np
import torch

from dataeval.utils.data.collate import list_collate_fn, numpy_collate_fn, torch_collate_fn


class TestCollateFn:
    def test_list_collate_fn(self):
        assert list_collate_fn([("a", 1, 2), ("b", 2, 3), ("c", 3, 4)]) == (["a", "b", "c"], [1, 2, 3], [2, 3, 4])

    def test_list_collate_fn_empty(self):
        assert list_collate_fn([]) == ([], [], [])

    def test_numpy_collate_fn(self):
        collated = numpy_collate_fn([([1, 2], 1, {"id": 1}), ([3, 4], 2, {"id": 2}), ([5, 6], 3, {"id": 3})])
        assert np.array_equal(collated[0], np.array([[1, 2], [3, 4], [5, 6]]))
        assert collated[1] == [1, 2, 3]
        assert collated[2] == [{"id": 1}, {"id": 2}, {"id": 3}]

    def test_numpy_collate_fn_empty(self):
        collated = numpy_collate_fn([])
        assert np.array_equal(collated[0], np.array([]))
        assert collated[1] == []
        assert collated[2] == []

    def test_torch_collate_fn(self):
        collated = torch_collate_fn([([1, 2], 1, {"id": 1}), ([3, 4], 2, {"id": 2}), ([5, 6], 3, {"id": 3})])
        assert torch.equal(collated[0], torch.tensor([[1, 2], [3, 4], [5, 6]]))
        assert collated[1] == [1, 2, 3]
        assert collated[2] == [{"id": 1}, {"id": 2}, {"id": 3}]

    def test_torch_collate_fn_empty(self):
        collated = torch_collate_fn([])
        assert torch.equal(collated[0], torch.tensor([]))
        assert collated[1] == []
        assert collated[2] == []
