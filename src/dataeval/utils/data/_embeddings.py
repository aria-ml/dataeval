from __future__ import annotations

__all__ = []

import math
from typing import Any, Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataeval.config import DeviceLike, get_device
from dataeval.typing import Array, Dataset, Transform
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
    transforms : Transform or Sequence[Transform] or None, default None
        Transforms to apply to images before encoding.
    model : torch.nn.Module or None, default None
        Model to use for encoding images.
    device : DeviceLike or None, default None
        The hardware device to use if specified, otherwise uses the DataEval
        default or torch default.
    cache : bool, default False
        Whether to cache the embeddings in memory.
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
        transforms: Transform[torch.Tensor] | Sequence[Transform[torch.Tensor]] | None = None,
        model: torch.nn.Module | None = None,
        device: DeviceLike | None = None,
        cache: bool = False,
        verbose: bool = False,
    ) -> None:
        self.device = get_device(device)
        self.cache = cache
        self.batch_size = batch_size
        self.verbose = verbose

        self._dataset = dataset
        model = torch.nn.Flatten() if model is None else model
        self._transforms = [transforms] if isinstance(transforms, Transform) else transforms
        self._model = model.to(self.device).eval()
        self._encoder = model.encode if isinstance(model, SupportsEncode) else model
        self._collate_fn = lambda datum: [torch.as_tensor(i) for i, _, _ in datum]
        self._cached_idx = set()
        self._embeddings: torch.Tensor | None = None

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
            if self._transforms:
                images = [transform(image) for transform in self._transforms for image in images]
            images = torch.stack(images).to(self.device)
            embeddings = self._encoder(images)
            yield embeddings

    def __getitem__(self, key: int | slice, /) -> torch.Tensor:
        if not isinstance(key, slice) and not hasattr(key, "__int__"):
            raise TypeError("Invalid argument type.")

        indices = list(range(len(self._dataset))[key]) if isinstance(key, slice) else [int(key)]
        if self.cache:
            uncached = [i for i in indices if i not in self._cached_idx]
            for i, embeddings in enumerate(self._batch(uncached)):
                batch = uncached[i * self.batch_size : (i + 1) * self.batch_size]
                if self._embeddings is None:
                    self._embeddings = torch.empty(
                        (len(self._dataset), *embeddings.shape[1:]), dtype=embeddings.dtype, device=self.device
                    )
                self._embeddings[batch] = embeddings
                self._cached_idx.update(batch)
        if self.cache and self._embeddings is not None:
            embeddings = self._embeddings[indices].to(self.device)
        else:
            embeddings = torch.vstack(list(self._batch(indices))).to(self.device)
        return embeddings.squeeze(0) if len(indices) == 1 else embeddings

    def __iter__(self) -> Iterator[torch.Tensor]:
        # process in batches while yielding individual embeddings
        for batch in self._batch(range(len(self._dataset))):
            yield from batch

    def __len__(self) -> int:
        return len(self._dataset)
