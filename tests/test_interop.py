import logging
from pathlib import PosixPath

import torch

from dataeval.interop import to_numpy


class TestInterop:
    def test_torch_to_numpy(self):
        t = torch.tensor([1, 2, 3, 4, 5])
        n = to_numpy(t)
        assert list(n) == list(t)


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
