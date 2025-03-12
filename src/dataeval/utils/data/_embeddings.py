from __future__ import annotations

__all__ = []

import math
from typing import Any, Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataeval.config import get_device
from dataeval.typing import TArray
from dataeval.utils.data.datasets._types import SizedDataset
from dataeval.utils.torch.models import SupportsEncode


class Embeddings:
    """
    Collection of image embeddings from a dataset.

    Embeddings are accessed by index or slice and are only loaded on-demand.

    Parameters
    ----------
    dataset : ImageClassificationDataset or ObjectDetectionDataset
        Dataset to access original images from.
    batch_size : int, optional
        Batch size to use when encoding images.
    model : torch.nn.Module, optional
        Model to use for encoding images.
    device : torch.device, optional
        Device to use for encoding images.
    verbose : bool, optional
        Whether to print progress bar when encoding images.
    """

    device: torch.device
    batch_size: int
    verbose: bool

    def __init__(
        self,
        dataset: SizedDataset[TArray, Any],
        batch_size: int,
        indices: Sequence[int] | None = None,
        model: torch.nn.Module | None = None,
        device: torch.device | str | None = None,
        verbose: bool = False,
    ) -> None:
        self.device = get_device(device)
        self.batch_size = batch_size
        self.verbose = verbose

        self._dataset = dataset
        self._indices = indices if indices is not None else range(len(dataset))
        model = torch.nn.Flatten() if model is None else model
        self._model = model.to(self.device).eval()
        self._encoder = model.encode if isinstance(model, SupportsEncode) else model
        self._collate_fn = lambda datum: [torch.as_tensor(i) for i, _, _ in datum]

    def to_tensor(self) -> torch.Tensor:
        """
        Converts entire dataset to embeddings.

        Warning
        -------
        Will process the entire dataset in batches and return
        embeddings as a single Tensor in memory.

        Returns
        -------
        torch.Tensor
        """
        return self[:]

    # Reduce overhead cost by not tracking tensor gradients
    @torch.no_grad
    def _batch(self, indices: Sequence[int]) -> Iterator[torch.Tensor]:
        # manual batching
        dataloader = DataLoader(Subset(self._dataset, indices), batch_size=self.batch_size, collate_fn=self._collate_fn)
        for i, images in (
            tqdm(enumerate(dataloader), total=math.ceil(len(indices) / self.batch_size), desc="Batch processing")
            if self.verbose
            else enumerate(dataloader)
        ):
            embeddings = self._encoder(torch.stack(images).to(self.device))
            yield embeddings

    def __getitem__(self, key: int | slice | list[int]) -> torch.Tensor:
        if isinstance(key, list):
            return torch.vstack(list(self._batch(key))).to(self.device)
        if isinstance(key, slice):
            return torch.vstack(list(self._batch(range(len(self._dataset))[key]))).to(self.device)
        elif isinstance(key, int):
            return self._encoder(torch.as_tensor(self._dataset[key][0]).to(self.device))
        raise TypeError("Invalid argument type.")

    def __iter__(self) -> Iterator[torch.Tensor]:
        # process in batches while yielding individual embeddings
        for batch in self._batch(range(len(self._dataset))):
            yield from batch

    def __len__(self) -> int:
        return len(self._dataset)
