from __future__ import annotations

__all__ = []

import math
from typing import Any, Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataeval.config import DeviceLike, get_device
from dataeval.typing import Array, Dataset
from dataeval.utils.torch.models import SupportsEncode


class Embeddings:
    """
    Collection of image embeddings from a dataset.

    Embeddings are accessed by index or slice and are only loaded on-demand.

    Parameters
    ----------
    dataset : ImageClassificationDataset or ObjectDetectionDataset
        Dataset to access original images from.
    batch_size : int
        Batch size to use when encoding images.
    model : torch.nn.Module or None, default None
        Model to use for encoding images.
    device : DeviceLike or None, default None
        The hardware device to use if specified, otherwise uses the DataEval
        default or torch default.
    verbose : bool, default False
        Whether to print progress bar when encoding images.
    """

    device: torch.device
    batch_size: int
    verbose: bool

    def __init__(
        self,
        dataset: Dataset[tuple[Array, Any, Any]],
        batch_size: int,
        model: torch.nn.Module | None = None,
        device: DeviceLike | None = None,
        verbose: bool = False,
    ) -> None:
        self.device = get_device(device)
        self.batch_size = batch_size
        self.verbose = verbose

        self._dataset = dataset
        model = torch.nn.Flatten() if model is None else model
        self._model = model.to(self.device).eval()
        self._encoder = model.encode if isinstance(model, SupportsEncode) else model
        self._collate_fn = lambda datum: [torch.as_tensor(i) for i, _, _ in datum]

    def to_tensor(self, indices: Sequence[int] | None = None) -> torch.Tensor:
        """
        Converts dataset to embeddings.

        Parameters
        ----------
        indices : Sequence[int] or None, default None
            The indices to convert to embeddings

        Returns
        -------
        torch.Tensor

        Warning
        -------
        Processing large quantities of data can be resource intensive.
        """
        if indices is not None:
            return torch.vstack(list(self._batch(indices))).to(self.device)
        else:
            return self[:]

    # Reduce overhead cost by not tracking tensor gradients
    @torch.no_grad
    def _batch(self, indices: Sequence[int]) -> Iterator[torch.Tensor]:
        # manual batching
        dataloader = DataLoader(Subset(self._dataset, indices), batch_size=self.batch_size, collate_fn=self._collate_fn)  # type: ignore
        for i, images in (
            tqdm(enumerate(dataloader), total=math.ceil(len(indices) / self.batch_size), desc="Batch processing")
            if self.verbose
            else enumerate(dataloader)
        ):
            embeddings = self._encoder(torch.stack(images).to(self.device))
            yield embeddings

    def __getitem__(self, key: int | slice, /) -> torch.Tensor:
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
