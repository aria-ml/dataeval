from __future__ import annotations

__all__ = []

from typing import Any, Generic, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

_TArray = TypeVar("_TArray")


class BaseDatasetMixin(Generic[_TArray]):
    index2label: dict[int, str]

    def _as_array(self, raw: list[Any]) -> _TArray: ...
    def _one_hot_encode(self, value: int | list[int]) -> _TArray: ...
    def _read_file(self, path: str) -> _TArray: ...


class BaseDatasetNumpyMixin(BaseDatasetMixin[NDArray[Any]]):
    def _as_array(self, raw: list[Any]) -> NDArray[Any]:
        return np.asarray(raw)

    def _one_hot_encode(self, value: int | list[int]) -> NDArray[Any]:
        if isinstance(value, int):
            encoded = np.zeros(len(self.index2label))
            encoded[value] = 1
        else:
            encoded = np.zeros((len(value), len(self.index2label)))
            encoded[np.arange(len(value)), value] = 1
        return encoded

    def _read_file(self, path: str) -> NDArray[Any]:
        return np.array(Image.open(path)).transpose(2, 0, 1)


class BaseDatasetTorchMixin(BaseDatasetMixin[torch.Tensor]):
    def _as_array(self, raw: list[Any]) -> torch.Tensor:
        return torch.as_tensor(raw)

    def _one_hot_encode(self, value: int | list[int]) -> torch.Tensor:
        if isinstance(value, int):
            encoded = torch.zeros(len(self.index2label))
            encoded[value] = 1
        else:
            encoded = torch.zeros((len(value), len(self.index2label)))
            encoded[torch.arange(len(value)), value] = 1
        return encoded

    def _read_file(self, path: str) -> torch.Tensor:
        return torch.as_tensor(np.array(Image.open(path)).transpose(2, 0, 1))
