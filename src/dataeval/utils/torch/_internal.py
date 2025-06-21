from __future__ import annotations

__all__ = []

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from dataeval.config import DeviceLike, get_device
from dataeval.typing import Array


def predict_batch(
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


def trainer(
    model: torch.nn.Module,
    x_train: NDArray[Any],
    y_train: NDArray[Any] | None,
    loss_fn: Callable[..., torch.Tensor] | None,
    optimizer: torch.optim.Optimizer | None,
    preprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    epochs: int,
    batch_size: int,
    device: torch.device,
    verbose: bool,
) -> None:
    """
    Train Pytorch model.

    Parameters
    ----------
    model
        Model to train.
    loss_fn
        Loss function used for training.
    x_train
        Training data.
    y_train
        Training labels.
    optimizer
        Optimizer used for training.
    preprocess_fn
        Preprocessing function applied to each training batch.
    epochs
        Number of training epochs.
    reg_loss_fn
        Allows an additional regularisation term to be defined as reg_loss_fn(model)
    batch_size
        Batch size used for training.
    buffer_size
        Maximum number of elements that will be buffered when prefetching.
    verbose
        Whether to print training progress.
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

    model = model.to(device)

    # iterate over epochs
    loss = torch.scalar_tensor(torch.nan)
    disable_tqdm = not verbose
    for epoch in (pbar := tqdm(range(epochs), disable=disable_tqdm)):
        epoch_loss = loss
        for step, data in enumerate(loader):
            if step % 250 == 0:
                pbar.set_description(f"Epoch: {epoch} ({epoch_loss:.3f}), loss: {loss:.3f}")

            x, y = [d.to(device) for d in data] if len(data) > 1 else (data[0].to(device), None)

            if isinstance(preprocess_fn, Callable):
                x = preprocess_fn(x)

            y_hat = model(x)
            y = x if y is None else y

            loss = loss_fn(y, *y_hat) if isinstance(y_hat, tuple) else loss_fn(y, y_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
