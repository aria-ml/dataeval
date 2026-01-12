"""
Utility functions for training and inference with PyTorch models.
"""

__all__ = ["train", "predict"]

import logging
from collections.abc import Callable, Sized
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from dataeval.config import DeviceLike, get_batch_size, get_device
from dataeval.protocols import Array, ProgressCallback

_logger = logging.getLogger(__name__)


def train(
    model: torch.nn.Module,
    x_train: NDArray[Any],
    y_train: NDArray[Any] | None,
    loss_fn: Callable[..., torch.Tensor] | None,
    optimizer: torch.optim.Optimizer | None,
    preprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    epochs: int,
    batch_size: int | None,
    device: DeviceLike | None = None,
    *,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """
    Train PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    x_train : NDArray
        Training data.
    y_train : NDArray or None
        Training labels. If None, assumes autoencoder-style training where x is the target.
    loss_fn : Callable or None
        Loss function used for training. If None, uses MSELoss.
    optimizer : torch.optim.Optimizer or None
        Optimizer used for training. If None, uses Adam with lr=0.001.
    preprocess_fn : Callable or None
        Preprocessing function applied to each training batch.
    epochs : int
        Number of training epochs.
    batch_size : int or None
        Batch size used for training.
    device : DeviceLike or None, default None
        The hardware device to use if specified, otherwise uses the DataEval
        default or torch default.
    progress_callback : ProgressCallback or None, default None
        Optional progress callback function.
    """
    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if y_train is None:
        dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32))
    else:
        dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))

    loader = DataLoader(dataset=dataset, batch_size=batch_size)
    device = get_device(device)
    model = model.to(device)

    # iterate over epochs
    loss = torch.scalar_tensor(torch.nan)

    # Calculate total steps if a callback is provided.
    total_steps = epochs * len(loader) if progress_callback and isinstance(loader, Sized) else None

    for epoch in range(epochs):
        epoch_loss = loss
        for step, data in enumerate(loader):
            x, y = [d.to(device) for d in data] if len(data) > 1 else (data[0].to(device), None)

            if isinstance(preprocess_fn, Callable):
                x = preprocess_fn(x)

            y_hat = model(x)
            y = x if y is None else y

            loss = loss_fn(y, *y_hat) if isinstance(y_hat, tuple) else loss_fn(y, y_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if progress_callback:
                current_step = epoch * len(loader) + step + 1
                extra_info = {
                    "loss": float(loss.item()),
                    "epoch_loss": float(epoch_loss.item()),  # Loss at start of epoch
                    "epoch": epoch,
                }
                progress_callback(step=current_step, total=total_steps, extra_info=extra_info)


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
    x : np.ndarray or torch.Tensor
        Batch of instances.
    model : torch.nn.Module
        PyTorch model.
    device : DeviceLike or None, default None
        The hardware device to use if specified, otherwise uses the DataEval
        default or torch default.
    batch_size : int or None, default None
        Batch size used during prediction. If None, uses DataEval default (1e10).
    preprocess_fn : Callable or None, default None
        Optional preprocessing function for each batch.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, ...]
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
