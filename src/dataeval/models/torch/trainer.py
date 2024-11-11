"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

from typing import Callable, cast

import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

# from dataeval._internal.detectors.ood.base_torch import GenericOptimizer


def trainer(
    model: torch.nn.Module,
    x_train: NDArray,
    y_train: NDArray | None = None,
    loss_fn: Callable[..., torch.Tensor] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    preprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    epochs: int = 20,
    reg_loss_fn: Callable[[torch.nn.Module], torch.Tensor] = (
        lambda _: cast(torch.Tensor, torch.tensor(0, dtype=torch.float32))
    ),
    batch_size: int = 64,
    buffer_size: int = 1024,
    verbose: bool = True,
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
    #
    # THIS WILL NEED MORE THAN JUST TYPO CHANGES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #
    # if optimizer is None:
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    # loss_fn = loss_fn() if isinstance(loss_fn, type) else loss_fn
    # optimizer = optimizer() if isinstance(optimizer, type) else optimizer

    # train_data = x_train if y_train is None else (x_train, y_train)
    # torch.tensor(train_data)  # make a Dataset from this!
    if y_train is None:
        dataset = TensorDataset(torch.from_numpy(x_train).to(torch.float32))
    else:
        dataset = TensorDataset(
            torch.from_numpy(x_train).to(torch.float32), torch.from_numpy(y_train).to(torch.float32)
        )

    # dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
    # n_minibatch = len(dataset)

    # def crapnorm(xraw):
    #     return (xraw - torch.tensor(0.5)) * torch.tensor(2.0)

    # preprocess_fn = crapnorm

    loader = DataLoader(dataset=dataset)
    # iterate over epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch}...")
        first_time_in_epoch = True

        for step, data in enumerate(loader):
            x, y = [d.to(torch.float32) for d in data] if len(data) > 1 else (data[0].to(torch.float32), None)

            if isinstance(preprocess_fn, Callable):
                x = preprocess_fn(x)

            y_hat = model(x)  # .clone().detach().requires_grad_(True)
            y = x if y is None else y
            # y = y.clone().detach().requires_grad_(True)

            loss = loss_fn(y, y_hat)  # type: ignore

            # if isinstance(loss_fn, Callable):
            #     args = [y] + list(y_hat) if isinstance(y_hat, tuple) else [y, y_hat]
            #     loss = loss_fn(*args)
            # else:
            #     loss = cast(torch.Tensor, torch.tensor(0.0, dtype=torch.float32))
            # if model.losses:  # additional model losses
            #     loss = cast(torch.Tensor, torch.add(sum(model.losses), loss))

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
                # print(total_norm)
                pass

            if step % 500 == 0:
                print(f"loss: {loss}, |grad|: {total_norm}")
        pass
