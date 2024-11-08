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
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_fn = loss_fn() if isinstance(loss_fn, type) else loss_fn
    # optimizer = optimizer() if isinstance(optimizer, type) else optimizer

    # train_data = x_train if y_train is None else (x_train, y_train)
    # torch.tensor(train_data)  # make a Dataset from this!
    if y_train is None:
        dataset = TensorDataset(torch.from_numpy(x_train))
    else:
        dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))

    # dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
    # n_minibatch = len(dataset)

    loader = DataLoader(dataset=dataset)
    # iterate over epochs
    for epoch in range(epochs):
        for step, data in enumerate(loader):
            x, y = [d.to(torch.float32) for d in data] if len(data) > 1 else (data[0].to(torch.float32), None)

            if isinstance(preprocess_fn, Callable):
                x = preprocess_fn(x)

            y_hat = model(x)
            y = x if y is None else y
            if isinstance(loss_fn, Callable):
                args = [y] + list(y_hat) if isinstance(y_hat, tuple) else [y, y_hat]
                loss = loss_fn(*args)
            else:
                loss = cast(torch.Tensor, torch.tensor(0.0, dtype=torch.float32))
            # if model.losses:  # additional model losses
            #     loss = cast(torch.Tensor, torch.add(sum(model.losses), loss))

            optimizer.zero_grad()
            # loss = cast(torch.Tensor, torch.add(reg_loss_fn(model), loss))  # alternative way they might be specified
            loss.backward()
            optimizer.step()
            if step % 500 == 0:
                print(loss)
        pass
