__all__ = []

from collections.abc import Callable

import numpy as np
import torch

from dataeval.config import DeviceLike, get_batch_size, get_device
from dataeval.protocols import Array


def predict(
    x: Array,
    model: torch.nn.Module,
    device: DeviceLike | None = None,
    batch_size: int | None = None,
    preprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
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
    torch.Tensor | tuple[torch.Tensor, ...]
        PyTorch tensor with model outputs, or tuple of tensors if model returns tuple
        (e.g., VAE models return (reconstruction, mu, logvar)).
    """
    device = get_device(device)
    if isinstance(model, torch.nn.Module):
        model = model.to(device).eval()
    x = torch.tensor(x, device=device)
    n = len(x)
    batch_size = get_batch_size(batch_size)
    n_minibatch = int(np.ceil(n / batch_size))
    preds_array = []
    with torch.no_grad():
        for i in range(n_minibatch):
            istart, istop = i * batch_size, min((i + 1) * batch_size, n)
            x_batch = x[istart:istop]
            if isinstance(preprocess_fn, Callable):
                x_batch = preprocess_fn(x_batch)
            output = model(x_batch.to(dtype=torch.float32))
            output = tuple(o.cpu() for o in output) if isinstance(output, tuple) else output.cpu()
            preds_array.append(output)

    # Concatenate predictions
    if preds_array and isinstance(preds_array[0], tuple):
        # If model returns tuples, concatenate each element separately
        num_outputs = len(preds_array[0])
        return tuple(torch.cat([batch[i] for batch in preds_array], dim=0) for i in range(num_outputs))
    return torch.cat(preds_array, dim=0)
