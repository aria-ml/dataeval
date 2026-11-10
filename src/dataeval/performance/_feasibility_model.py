"""Run a MAITE model over a dataset for the feasibility evaluator, keeping the model's own score precision."""

__all__ = []

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from dataeval.protocols import AnnotatedDataset, Model
from dataeval.utils._array import to_numpy


def model_scores(model: Model, dataset: AnnotatedDataset[Any], batch_size: int) -> NDArray[np.float64]:
    """
    Return the model's class scores for every dataset item, shape ``(len(dataset), n_classes)``.

    The model runs over the whole dataset in batches of ``batch_size`` images, and each item is scored once. Scores
    are widened to float64, never narrowed: a float32 cast would turn the float64 scores ``[0.5, 0.5000000001]`` into a
    tie and could change which class wins. A torch model runs without autograd, and its outputs may live on any
    device.
    """
    rows: list[NDArray[Any]] = []
    with torch.no_grad():  # a torch model's outputs would otherwise carry an autograd graph and refuse numpy()
        for start in range(0, len(dataset), batch_size):
            batch = [dataset[i][0] for i in range(start, min(start + batch_size, len(dataset)))]
            rows.extend(to_numpy(scores, copy=False) for scores in model(batch))
    return np.stack(rows).astype(np.float64)


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
        If both declare an ``index2label`` and they differ. Opposite class orders pass shape validation and score
        the wrong classes, and DataEval never remaps one to the other, because that would decide for the caller.
    """
    theirs, ours = _declared(getattr(model, "metadata", None)), _declared(getattr(dataset, "metadata", None))
    if theirs is None or ours is None:
        return False
    if theirs != ours:
        raise ValueError(
            f"The model's index2label {theirs} differs from the dataset's {ours}, and DataEval never remaps classes. "
            "Score the dataset with a model that uses its class order, or relabel the dataset to the model's."
        )
    return True
