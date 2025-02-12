import logging
from pathlib import PosixPath

import numpy as np
import pytest
import torch

from dataeval._interop import _simplify_type, _try_cast, to_numpy, to_numpy_iter


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


@pytest.mark.required
class TestCastSimplify:
    @pytest.mark.parametrize(
        "value, target, output",
        (
            ("123", int, 123),
            ("123", float, 123.0),
            ("123", str, "123"),
            ("12.3", int, None),
            ("12.3", float, 12.3),
            ("12.3", str, "12.3"),
            ("foo", int, None),
            ("foo", float, None),
            ("foo", str, "foo"),
        ),
    )
    def test_try_cast_(self, value, target, output):
        assert output == _try_cast(value, target)

    @pytest.mark.parametrize(
        "value, output",
        (
            ("123", 123),
            ("12.3", 12.3),
            ("foo", "foo"),
            ([123, "12.3"], [123.0, 12.3]),
            ([123, "foo"], ["123", "foo"]),
            (["123", "456"], [123, 456]),
        ),
    )
    def test_convert_type(self, value, output):
        assert output == _simplify_type(value)
