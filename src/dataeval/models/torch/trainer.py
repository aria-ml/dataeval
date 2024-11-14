"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

from typing import Any, Callable, cast

import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

# from dataeval._internal.detectors.ood.base_torch import GenericOptimizer


def trainer(
    model: torch.nn.Module,
    x_train: NDArray[Any],
    y_train: NDArray[Any] | None = None,
    loss_fn: Callable[..., torch.Tensor] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    preprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    epochs: int = 20,
    reg_loss_fn: Callable[[torch.nn.Module], torch.Tensor] = (
        lambda _: cast(torch.Tensor, torch.tensor(0, dtype=torch.float32))
    ),
    batch_size: int = 64,
    buffer_size: int = 1024,
    verbose: bool = False,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if y_train is None:
        dataset = TensorDataset(torch.from_numpy(x_train).to(torch.float32))
    else:
        dataset = TensorDataset(
            torch.from_numpy(x_train).to(torch.float32), torch.from_numpy(y_train).to(torch.float32)
        )

    loader = DataLoader(dataset=dataset)
    # iterate over epochs
    for epoch in range(epochs):
        if verbose:
            print(f"Epoch {epoch}...")
        first_time_in_epoch = True

        for step, data in enumerate(loader):
            x, y = [d.to(torch.float32) for d in data] if len(data) > 1 else (data[0].to(torch.float32), None)

            if isinstance(preprocess_fn, Callable):
                x = preprocess_fn(x)

            y_hat = model(x)
            y = x if y is None else y

            loss = loss_fn(y, y_hat)  # type: ignore

            optimizer.zero_grad()
            loss.backward()

            total_norm = 0
            parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)  # type: ignore
                total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5

            psave = list(model.parameters())[0].clone().detach().numpy()
            optimizer.step()
            pnew = list(model.parameters())[0].clone().detach().numpy()

            if (psave == pnew).all() and first_time_in_epoch:  # type: ignore
                first_time_in_epoch = False
                print(f"epoch: {epoch}, step: {step}, parameters not changing")
                pass
            else:
                pass

            if step % 500 == 0 and verbose:
                print(f"loss: {loss:.3f}, |grad|: {total_norm:.3f}")
        pass
