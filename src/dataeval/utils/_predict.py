from __future__ import annotations

__all__ = []

from collections.abc import Callable

import numpy as np
import torch

from dataeval.config import DeviceLike, get_device
from dataeval.protocols import Array


def predict(
    x: Array,
    model: torch.nn.Module,
    device: DeviceLike | None = None,
    batch_size: int = int(1e10),
    preprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Make batch predictions on a model.

    Parameters
    ----------
    x : np.ndarray | torch.Tensor
        Batch of instances.
    model : nn.Module
        PyTorch model.
    device : DeviceLike or None, default None
        The hardware device to use if specified, otherwise uses the DataEval
        default or torch default.
    batch_size : int, default 1e10
        Batch size used during prediction.
    preprocess_fn : Callable | None, default None
        Optional preprocessing function for each batch.

    Returns
    -------
    torch.Tensor
        PyTorch tensor with model outputs.
    """
    device = get_device(device)
    if isinstance(model, torch.nn.Module):
        model = model.to(device).eval()
    x = torch.tensor(x, device=device)
    n = len(x)
    n_minibatch = int(np.ceil(n / batch_size))
    preds_array = []
    with torch.no_grad():
        for i in range(n_minibatch):
            istart, istop = i * batch_size, min((i + 1) * batch_size, n)
            x_batch = x[istart:istop]
            if isinstance(preprocess_fn, Callable):
                x_batch = preprocess_fn(x_batch)
            preds_array.append(model(x_batch.to(dtype=torch.float32)).cpu())

    return torch.cat(preds_array, dim=0)
