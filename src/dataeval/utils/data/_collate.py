from __future__ import annotations

__all__ = []

from typing import Any, Iterable, Sequence, TypeVar

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataeval.config import get_device
from dataeval.utils._targets import Targets
from dataeval.utils.data.datasets._types import ImageClassificationDataset, ObjectDetectionDataset, TArrayLike
from dataeval.utils.torch.models import SupportsEncode


# Reduce overhead cost by not tracking tensor gradients
@torch.no_grad
def collate(
    dataset: ImageClassificationDataset[TArrayLike] | ObjectDetectionDataset[TArrayLike],
    model: torch.nn.Module | None = None,
    device: torch.device | str | None = None,
    batch_size: int = 64,
) -> tuple[torch.Tensor, Targets, list[dict[str, Any]]] | tuple[list[TArrayLike], Targets, list[dict[str, Any]]]:
    """
    Collates a dataset to images/embeddings, targets and metadata.

    Parameters
    ----------
    dataset : ImageClassificationDataset or ObjectDetectionDataset
        A dataset conforming to MAITE dataset protocols.
    model : torch.nn.Module or None, default None
        A torch model to use for encoding. If an `encode()` function
        is present on the model it will be called, otherwise it will use
        the `__call__()` function.
    device : torch.device, str or None, default None
        Device to use when encoding with the provided model.
    batch_size : int, default 64
        Batch sizes to use when encoding with the provided model.

    Returns
    -------
    tuple[torch.Tensor | list[Any], Targets, list[dict[str, Any]]]
        - Images as a list of original source data or embeddings as a torch.Tensor if encoded.
        - Targets including labels, scores as well as boxes and source indices for objects.
        - Metadata aggregated as a list of individual datum metadata dictionaries.

    Note
    ----
    For more on supported image classification and object detection dataset
    protocols, see protocol documentations for `MAITE <https://mit-ll-ai-technology.github.io/maite/explanation/protocol_overview.html>`_.
    """
    device = get_device(device)
    encoder = None

    if model is not None:
        model.to(device).eval()
        encoder = model.encode if isinstance(model, SupportsEncode) else model

    embeddings: list[torch.Tensor] | torch.Tensor = []
    images, targets, metadata, source = [], [], [], []

    is_target_dict = False

    collate_fn = collate_as_tensor_fn if model is not None else default_collate_fn
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    total_batches = len(dataset) // batch_size + int(len(dataset) % batch_size != 0)

    for i, (image, target, metadatum) in tqdm(enumerate(dataloader), total=total_batches, desc="Batch processing"):
        # process image
        if encoder is None:
            images.extend(image)
            source.append(i)
        else:
            outputs = encoder(torch.stack(image).to(device))
            embeddings.extend(outputs)

        # process target
        if is_target_dict := is_target_dict or isinstance(target[0], dict):
            target = [{k: torch.as_tensor(v).detach().cpu() for k, v in t.items()} for t in target]
            source.extend([i] * len(target))
        else:
            target = [torch.as_tensor(t).detach().cpu() for t in target]
        targets.extend(target)

        # process metadata
        metadata.extend(metadatum)

    if is_target_dict:
        labels = np.asarray([int(label) for t in targets for label in t.get("labels", [])], dtype=np.intp)
        scores = np.asarray([float(score) for t in targets for score in t.get("scores", [])], dtype=np.float32)
        bboxes = np.asarray([box.numpy(force=True) for t in targets for box in t.get("boxes", [])], dtype=np.float32)
        source = np.asarray(source, dtype=np.intp)
    else:
        labels = np.argmax(np.asarray(targets), axis=1).astype(np.intp)
        scores = np.asarray(targets, dtype=np.float32)
        bboxes = None
        source = None

    targets = Targets(labels, scores, bboxes, source)

    if encoder is None:
        return images, targets, metadata
    else:
        return torch.stack(embeddings).to(device), targets, metadata


# modified from maite._internals.workflows.generic

T_in = TypeVar("T_in")
T_tgt = TypeVar("T_tgt")
T_md = TypeVar("T_md")


def default_collate_fn(
    batch_data_as_singles: Iterable[tuple[T_in, T_tgt, T_md]],
) -> tuple[Sequence[T_in], Sequence[T_tgt], Sequence[T_md]]:
    input_batch: list[T_in] = []
    target_batch: list[T_tgt] = []
    metadata_batch: list[T_md] = []
    for input_datum, target_datum, metadata_datum in batch_data_as_singles:
        input_batch.append(input_datum)
        target_batch.append(target_datum)
        metadata_batch.append(metadata_datum)

    return input_batch, target_batch, metadata_batch


def collate_as_tensor_fn(
    batch_data_as_singles: Iterable[tuple[T_in, T_tgt, T_md]],
) -> tuple[Sequence[torch.Tensor], Sequence[T_tgt], Sequence[T_md]]:
    input_batch: list[torch.Tensor] = []
    target_batch: list[T_tgt] = []
    metadata_batch: list[T_md] = []
    for input_datum, target_datum, metadata_datum in batch_data_as_singles:
        input_batch.append(torch.as_tensor(input_datum))
        target_batch.append(target_datum)
        metadata_batch.append(metadata_datum)

    return input_batch, target_batch, metadata_batch
