"""Run a MAITE model over a dataset for the feasibility evaluator, keeping the model's own output precision."""

__all__ = []

from collections.abc import Callable, Mapping
from contextlib import closing
from itertools import islice
from operator import itemgetter
from typing import Any, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray

from dataeval.protocols import AnnotatedDataset, Model
from dataeval.utils._array import to_numpy
from dataeval.utils._internal import read_dataset

T = TypeVar("T")


def run_model(model: Model, dataset: AnnotatedDataset[Any], batch_size: int, read: Callable[[int, Any], T]) -> list[T]:
    """
    Run the model over every dataset item in batches of ``batch_size`` images and return ``read(i, output)`` for each.

    ``i`` is the item's index in the dataset and ``output`` is the model's output for that image alone. Each batch must
    return one output per image, in order: a mapping, anything that cannot be iterated, or the wrong number of outputs
    is refused with an error that names the batch's first image. A torch model runs without autograd, so ``read`` does
    too, and its outputs may live on any device. ``read`` must copy what it keeps from ``output``, since a model may
    reuse one buffer for every batch.
    """
    results: list[T] = []
    # One read of the whole dataset, in `load_threads` threads when that is set, so reads run on into the next batch
    # while the model scores this one. Closed if the model raises, so no read thread outlives the error.
    images = read_dataset(dataset, range(len(dataset)), itemgetter(0))
    with torch.no_grad(), closing(images):  # without no_grad a torch model's outputs carry a graph and refuse numpy()
        start = 0
        while batch := list(islice(images, batch_size)):
            output = model(batch)
            refusal = (
                f"The model returned a {type(output).__name__} for the batch of {len(batch)} images starting at image "
                f"{start}. Return one output per image of the batch, in order."
            )
            if isinstance(output, Mapping):
                raise ValueError(refusal)
            try:
                outputs = list(output)  # not an Iterable check: a sequence that iterates by __getitem__ works too
            except TypeError as error:
                raise ValueError(refusal) from error
            if len(outputs) != len(batch):
                raise ValueError(
                    f"The model returned {len(outputs)} outputs for the batch of {len(batch)} images starting at "
                    f"image {start}. Return one output per image of the batch, in order."
                )
            results.extend(read(start + k, one) for k, one in enumerate(outputs))
            start += len(batch)
    return results


def model_scores(model: Model, dataset: AnnotatedDataset[Any], batch_size: int) -> NDArray[np.float64]:
    """
    Return the model's class scores for every dataset item, shape ``(len(dataset), n_classes)``.

    The model runs over the whole dataset in batches of ``batch_size`` images, and each item is scored once. Scores
    are widened to float64, never narrowed: a float32 cast would turn the float64 scores ``[0.5, 0.5000000001]`` into a
    tie and could change which class wins. A torch model runs without autograd, and its outputs may live on any
    device.
    """
    return np.stack(run_model(model, dataset, batch_size, lambda _, scores: to_numpy(scores, copy=True))).astype(
        np.float64
    )


def _declared(metadata: Any) -> dict[int, str] | None:
    """Return the ``index2label`` a metadata mapping declares, as ``{index: name}``; None when it declares none."""
    try:
        declared = metadata.get("index2label")
    except AttributeError:
        return None
    return {int(k): str(v) for k, v in declared.items()} if declared else None


def check_class_order(model: Model, dataset: AnnotatedDataset[Any]) -> bool:
    """
    Check that a model and a dataset declare the same classes, and say whether that could be verified.

    Returns True when both declare an ``index2label`` and the two are equal, and False when either declares none
    (the dataset's class order is then assumed).

    Raises
    ------
    ValueError
        If both declare an ``index2label`` and they differ. Opposite class orders pass shape validation and put a
        classifier's scores or a detector's labels on the wrong classes, and DataEval never remaps one to the other,
        because that would decide for the caller.
    """
    theirs, ours = _declared(getattr(model, "metadata", None)), _declared(getattr(dataset, "metadata", None))
    if theirs is None or ours is None:
        return False
    if theirs != ours:
        raise ValueError(
            f"The model's index2label {theirs} differs from the dataset's {ours}, and DataEval never remaps classes. "
            "Use a model whose labels follow the dataset's class order, wrap it to map them, or relabel the dataset to "
            "the model's."
        )
    return True
