from __future__ import annotations

__all__ = []

import math
from typing import Any, Generic, Iterator, Sequence, overload

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataeval.config import get_device
from dataeval.typing import TArray
from dataeval.utils import Metadata, Targets
from dataeval.utils._array import as_numpy
from dataeval.utils.data.datasets._types import (
    ImageClassificationDataset,
    ObjectDetectionDataset,
    ObjectDetectionTarget,
)
from dataeval.utils.metadata import merge, preprocess
from dataeval.utils.torch.models import SupportsEncode


class Images(Generic[TArray]):
    """
    Collection of image data from a dataset.

    Images are accessed by index or slice and are only loaded on-demand.

    Parameters
    ----------
    dataset : ImageClassificationDataset or ObjectDetectionDataset
        Dataset to access images from.
    """

    def __init__(
        self,
        dataset: ImageClassificationDataset[TArray] | ObjectDetectionDataset[TArray],
    ) -> None:
        self._dataset = dataset

    def to_list(self) -> Sequence[TArray]:
        """
        Converts entire dataset to a sequence of images.

        Warning
        -------
        Will load the entire dataset and return the images as a
        single sequence of images in memory.

        Returns
        -------
        list[TArray]
        """
        return self[:]

    @overload
    def __getitem__(self, key: slice) -> Sequence[TArray]: ...

    @overload
    def __getitem__(self, key: int) -> TArray: ...

    def __getitem__(self, key: int | slice) -> Sequence[TArray] | TArray:
        if isinstance(key, slice):
            indices = list(range(len(self._dataset))[key])
            return [self._dataset[i][0] for i in indices]
        elif isinstance(key, int):
            return self._dataset[key][0]
        raise TypeError("Invalid argument type.")

    def __iter__(self) -> Iterator[TArray]:
        for i in range(len(self._dataset)):
            yield self._dataset[i][0]

    def __len__(self) -> int:
        return len(self._dataset)


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
        dataset: ImageClassificationDataset[TArray] | ObjectDetectionDataset[TArray],
        batch_size: int,
        model: torch.nn.Module | None = None,
        device: torch.device | str | None = None,
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

    def __getitem__(self, key: slice | int) -> torch.Tensor:
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


class DataProcessor(Generic[TArray]):
    """
    Collection of image embeddings and metadata from a dataset.

    Parameters
    ----------
    dataset : ImageClassificationDataset or ObjectDetectionDataset
        Dataset to access original images from.
    batch_size : int
        Batch size to use when encoding images.
    model : torch.nn.Module, optional
        Model to use for encoding images.
    device : torch.device, optional
        Device to use for encoding images.

    Warnings
    --------
    Embeddings expect the raw dataset images to be preprocessed for
    dimensionality and normalization.
    """

    images: Images[TArray]
    embeddings: Embeddings

    def __init__(
        self,
        dataset: ImageClassificationDataset[TArray] | ObjectDetectionDataset[TArray],
        batch_size: int,
        model: torch.nn.Module | None = None,
        device: torch.device | str | None = None,
    ):
        # image embeddings are processed on-demand
        self.images: Images[TArray] = Images(dataset)
        self.embeddings: Embeddings = Embeddings(dataset, batch_size, model, device)

        self._dataset: ImageClassificationDataset[TArray] | ObjectDetectionDataset[TArray] = dataset
        self._collated = False
        self._metadata = None

    def _collate_metadata(self):
        raw_metadata: list[dict[str, Any]] = []

        labels = []
        bboxes = []
        scores = []
        srcidx = []
        is_od = None
        for i in range(len(self._dataset)):
            _, target, metadata = self._dataset[i]

            raw_metadata.append(metadata)

            if is_od_target := isinstance(target, ObjectDetectionTarget):
                target_len = len(target.labels)
                labels.extend(as_numpy(target.labels).tolist())
                bboxes.extend(as_numpy(target.boxes).tolist())
                scores.extend(as_numpy(target.scores).tolist())
                srcidx.extend([i] * target_len)
            else:
                target_len = 1
                labels.append(int(np.argmax(as_numpy(target))))
                scores.append(target)

            is_od = is_od_target if is_od is None else is_od
            if is_od != is_od_target:
                raise ValueError("Encountered unexpected target type in dataset")

        labels = as_numpy(labels).astype(np.intp)
        scores = as_numpy(scores).astype(np.float32)
        bboxes = as_numpy(bboxes).astype(np.float32) if is_od else None
        srcidx = as_numpy(srcidx).astype(np.intp) if is_od else None

        self._targets = Targets(labels, scores, bboxes, srcidx)
        self._raw_metadata = raw_metadata

    @property
    def targets(self) -> Targets:
        """
        Access the targets of the dataset.

        Returns
        -------
        Targets
        """
        if not self._collated:
            self._collate_metadata()
        return self._targets

    @property
    def raw_metadata(self) -> list[dict[str, Any]]:
        """
        Access the raw metadata of the dataset.

        Returns
        -------
        list[dict[str, Any]]
        """
        if not self._collated:
            self._collate_metadata()
        return self._raw_metadata

    @property
    def metadata(self) -> Metadata:
        """
        Access the metadata of the dataset.

        Returns
        -------
        Metadata
        """
        if self._metadata is None:
            targets_per_image = (
                None if self.targets.source is None else np.unique(self.targets.source, return_counts=True)[1].tolist()
            )
            merged = merge(self.raw_metadata, targets_per_image=targets_per_image)
            self._metadata = preprocess(merged, self._targets.labels)
        return self._metadata

    def __len__(self) -> int:
        return len(self._dataset)
