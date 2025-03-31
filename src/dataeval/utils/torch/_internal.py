from __future__ import annotations

__all__ = []

from functools import partial
from typing import Any, Callable

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dataeval.config import DeviceLike, get_device


def predict_batch(
    x: NDArray[Any] | torch.Tensor,
    model: Callable | torch.nn.Module | torch.nn.Sequential,
    device: DeviceLike | None = None,
    batch_size: int = int(1e10),
    preprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    dtype: type[np.generic] | torch.dtype = np.float32,
) -> NDArray[Any] | torch.Tensor | tuple[Any, ...]:
    """
    Make batch predictions on a model.

    Parameters
    ----------
    x : np.ndarray | torch.Tensor
        Batch of instances.
    model : Callable | nn.Module | nn.Sequential
        PyTorch model.
    device : DeviceLike or None, default None
        The hardware device to use if specified, otherwise uses the DataEval
        default or torch default.
    batch_size : int, default 1e10
        Batch size used during prediction.
    preprocess_fn : Callable | None, default None
        Optional preprocessing function for each batch.
    dtype : np.dtype | torch.dtype, default np.float32
        Model output type, either a :term:`NumPy` or torch dtype, e.g. np.float32 or torch.float32.

    Returns
    -------
    NDArray | torch.Tensor | tuple
        Numpy array, torch tensor or tuples of those with model outputs.
    """
    device = get_device(device)
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, device=device)
    n = len(x)
    n_minibatch = int(np.ceil(n / batch_size))
    return_np = not isinstance(dtype, torch.dtype)
    preds_tuple = None
    preds_array = []
    with torch.no_grad():
        for i in range(n_minibatch):
            istart, istop = i * batch_size, min((i + 1) * batch_size, n)
            x_batch = x[istart:istop]
            if isinstance(preprocess_fn, Callable):
                x_batch = preprocess_fn(x_batch)

            preds_tmp = model(x_batch.to(dtype=torch.float32))
            if isinstance(preds_tmp, (list, tuple)):
                if preds_tuple is None:  # init tuple with lists to store predictions
                    preds_tuple = tuple([] for _ in range(len(preds_tmp)))
                for j, p in enumerate(preds_tmp):
                    p = p.cpu() if isinstance(p, torch.Tensor) else p
                    preds_tuple[j].append(p if not return_np or isinstance(p, np.ndarray) else p.numpy())
            elif isinstance(preds_tmp, (np.ndarray, torch.Tensor)):
                preds_tmp = preds_tmp.cpu() if isinstance(preds_tmp, torch.Tensor) else preds_tmp
                preds_array.append(
                    preds_tmp if not return_np or isinstance(preds_tmp, np.ndarray) else preds_tmp.numpy()
                )
            else:
                raise TypeError(
                    f"Model output type {type(preds_tmp)} not supported. The model \
                    output type needs to be one of list, tuple, NDArray or \
                    torch.Tensor."
                )
    concat = partial(np.concatenate, axis=0) if return_np else partial(torch.cat, dim=0)
    out = tuple(concat(p) for p in preds_tuple) if preds_tuple is not None else concat(preds_array)
    return out


def trainer(
    model: torch.nn.Module,
    x_train: NDArray[Any],
    y_train: NDArray[Any] | None,
    loss_fn: Callable[..., torch.Tensor | torch.nn.Module] | None,
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
    loss = torch.nan
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

            loss = loss_fn(y, *y_hat) if isinstance(y_hat, tuple) else loss_fn(y, y_hat)  # type: ignore

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
