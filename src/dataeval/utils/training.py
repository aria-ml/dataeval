"""Utility functions for training and inference with PyTorch models."""

__all__ = ["train", "predict"]

import logging
from collections.abc import Callable, Iterable, Sized
from typing import Any, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from dataeval.config import DeviceLike, get_batch_size, get_device
from dataeval.protocols import Array, ProgressCallback

_logger = logging.getLogger(__name__)


PreprocessFn: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
"""
Per-instance preprocessing applied by :func:`~dataeval.utils.training.predict`.

Receives a single image tensor and returns the transformed tensor, which is then
stacked with its batch-mates before inference. This is where variable-size inputs
are normalized (e.g. resized) to a common, stackable shape.
"""

PostprocessFn: TypeAlias = Callable[[Any], torch.Tensor | tuple[torch.Tensor, ...]]
"""
Batch-level decoding of raw model output in :func:`~dataeval.utils.training.predict`.

Receives the model's raw output for a batch (which may be model-specific, hence the
``Any`` input) and returns the per-instance score tensor -- or tuple of tensors -- that
downstream code consumes. The return must be a tensor or tuple of tensors so that
``predict`` can move it to the CPU and concatenate across batches.
"""


def train(  # noqa: C901
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

    _logger.info(
        "Training %s on %d samples (%s) for %d epochs, batch_size=%s, device=%s",
        model.__class__.__name__,
        len(x_train),
        "supervised" if y_train is not None else "autoencoder",
        epochs,
        batch_size,
        device,
    )

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

        _logger.debug("Epoch %d/%d complete: loss=%.6f", epoch + 1, epochs, float(loss.item()))

    _logger.info("Training complete: final loss=%.6f", float(loss.item()))


def predict(
    x: Iterable[Array],
    model: torch.nn.Module,
    device: DeviceLike | None = None,
    batch_size: int | None = None,
    preprocess_fn: PreprocessFn | None = None,
    postprocess_fn: PostprocessFn | None = None,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """
    Make batch predictions on a model.

    Parameters
    ----------
    x : Iterable[Array]
        An iterable of per-instance images (e.g. a numpy array batched along
        axis 0, a list of arrays, or any iterable yielding one image at a time).
        Each instance is passed through ``preprocess_fn`` individually and only
        then stacked into batches, so instances need not share a common shape on
        input -- ``preprocess_fn`` is responsible for normalizing them (e.g.
        resizing) to a stackable shape. This is what allows variable-size images
        (such as detection inputs) to be supplied without pre-batching.
    model : torch.nn.Module
        PyTorch model.
    device : DeviceLike or None, default None
        The hardware device to use if specified, otherwise uses the DataEval
        default or torch default.
    batch_size : int or None, default None
        Batch size used during prediction. If None, uses DataEval default (1e10).
    preprocess_fn : PreprocessFn or None, default None
        Optional per-instance preprocessing applied to each image before it is
        stacked into a batch. Receives a single image tensor and returns the
        transformed tensor; use it to normalize variable-size inputs (e.g. resize)
        to a common, stackable shape.
    postprocess_fn : PostprocessFn or None, default None
        Optional batch-level decoding applied to each batch's raw model output
        before it is moved to the CPU. Receives the model output and must return a
        tensor or tuple of tensors (e.g. to decode raw detection outputs into
        per-detection class scores).

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, ...]
        PyTorch tensor with model outputs, or tuple of tensors if model returns tuple
        (e.g., VAE models return (reconstruction, mu, logvar)).
    """
    device = get_device(device)
    if isinstance(model, torch.nn.Module):
        model = model.to(device).eval()
    x = _to_tensor_list(x, device)
    n = len(x)
    batch_size = get_batch_size(batch_size)
    n_minibatch = int(np.ceil(n / batch_size))
    _logger.info(
        "Predicting with %s on %d instances in %d minibatch(es), batch_size=%d, device=%s",
        model.__class__.__name__,
        n,
        n_minibatch,
        batch_size,
        device,
    )
    preds_array = []
    with torch.no_grad():
        for i in range(n_minibatch):
            istart, istop = i * batch_size, min((i + 1) * batch_size, n)
            x_batch_raw = x[istart:istop]
            if isinstance(preprocess_fn, Callable):
                x_batch = torch.stack([preprocess_fn(img) for img in x_batch_raw])
            else:
                x_batch = torch.stack(x_batch_raw)
            output = model(x_batch.to(dtype=torch.float32))
            if postprocess_fn is not None:
                output = postprocess_fn(output)
            output = tuple(o.cpu() for o in output) if isinstance(output, tuple) else output.cpu()
            _logger.debug("Minibatch %d/%d: %d instances", i + 1, n_minibatch, istop - istart)
            preds_array.append(output)

    predictions = _concat_predictions(preds_array)
    prediction_shape_log = (
        f"tuple of {len(predictions)} tensors"
        if isinstance(predictions, tuple)
        else f"output shape {tuple(predictions.shape)}"
    )
    _logger.debug("Prediction complete: %s", prediction_shape_log)
    return predictions


def _to_tensor_list(x: Iterable[Array], device: torch.device) -> list[torch.Tensor]:
    """Materialize an iterable of per-instance images into device tensors.

    Iterating handles every input shape uniformly -- a numpy array (yielding rows),
    a list, or any other iterable. Per-instance conversion (rather than a single
    ``torch.as_tensor(x)``) is required because instances may be ragged (e.g.
    variable-size images), which cannot be stacked into one tensor until
    ``preprocess_fn`` normalizes them. ``as_tensor`` returns existing tensors
    without copying and converts arrays in place where possible.
    """
    return [torch.as_tensor(item, device=device) for item in x]


def _concat_predictions(preds_array: list) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Concatenate per-batch predictions, handling tuple-valued model outputs."""
    first = preds_array[0]
    if isinstance(first, tuple):
        return tuple(torch.cat([batch[i] for batch in preds_array], dim=0) for i in range(len(first)))
    return torch.cat(preds_array, dim=0)
