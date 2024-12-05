from __future__ import annotations

__all__ = ["read_dataset"]

from collections import defaultdict
from functools import partial
from typing import Any, Callable

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


def read_dataset(dataset: Dataset[Any]) -> list[list[Any]]:
    """
    Extract information from a dataset at each index into individual lists of each information position

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Input dataset

    Returns
    -------
    List[List[Any]]
        All objects in individual lists based on return position from dataset

    Warning
    -------
    No type checking is done between lists or data inside lists

    See Also
    --------
    torch.utils.data.Dataset

    Examples
    --------
    >>> import numpy as np
    >>> data = np.ones((10, 1, 3, 3))
    >>> labels = np.ones((10,))
    >>> class ICDataset:
    ...     def __init__(self, data, labels):
    ...         self.data = data
    ...         self.labels = labels
    ...
    ...     def __getitem__(self, idx):
    ...         return self.data[idx], self.labels[idx]

    >>> ds = ICDataset(data, labels)

    >>> result = read_dataset(ds)
    >>> len(result)  # images and labels
    2
    >>> np.asarray(result[0]).shape  # images
    (10, 1, 3, 3)
    >>> np.asarray(result[1]).shape  # labels
    (10,)
    """

    ddict: dict[int, list[Any]] = defaultdict(list[Any])

    for data in dataset:
        for i, d in enumerate(data if isinstance(data, tuple) else (data,)):
            ddict[i].append(d)

    return list(ddict.values())


def get_device(device: str | torch.device | None = None) -> torch.device:
    """
    Instantiates a PyTorch device object.

    Parameters
    ----------
    device : str | torch.device | None, default None
        Either ``None``, a str ('gpu' or 'cpu') indicating the device to choose, or an
        already instantiated device object. If ``None``, the GPU is selected if it is
        detected, otherwise the CPU is used as a fallback.

    Returns
    -------
    The instantiated device object.
    """
    if isinstance(device, torch.device):  # Already a torch device
        return device
    else:  # Instantiate device
        if device is None or device.lower() in ["gpu", "cuda"]:
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            torch_device = torch.device("cpu")
    return torch_device


def predict_batch(
    x: NDArray[Any] | torch.Tensor,
    model: Callable | torch.nn.Module | torch.nn.Sequential,
    device: torch.device | None = None,
    batch_size: int = int(1e10),
    preprocess_fn: Callable | None = None,
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
    device : torch.device | None, default None
        Device type used. The default None tries to use the GPU and falls back on CPU.
        Can be specified by passing either torch.device('cuda') or torch.device('cpu').
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
        x = torch.from_numpy(x).to(device)
    n = len(x)
    n_minibatch = int(np.ceil(n / batch_size))
    return_np = not isinstance(dtype, torch.dtype)
    preds = []
    with torch.no_grad():
        for i in range(n_minibatch):
            istart, istop = i * batch_size, min((i + 1) * batch_size, n)
            x_batch = x[istart:istop]
            if isinstance(preprocess_fn, Callable):
                x_batch = preprocess_fn(x_batch)

            preds_tmp = model(x_batch.to(torch.float32).to(device))
            if isinstance(preds_tmp, (list, tuple)):
                if len(preds) == 0:  # init tuple with lists to store predictions
                    preds = tuple([] for _ in range(len(preds_tmp)))
                for j, p in enumerate(preds_tmp):
                    if isinstance(p, torch.Tensor):
                        p = p.cpu()
                    preds[j].append(p if not return_np or isinstance(p, np.ndarray) else p.numpy())
            elif isinstance(preds_tmp, (np.ndarray, torch.Tensor)):
                if isinstance(preds_tmp, torch.Tensor):
                    preds_tmp = preds_tmp.cpu()
                if isinstance(preds, tuple):
                    preds = list(preds)
                preds.append(
                    preds_tmp
                    if not return_np or isinstance(preds_tmp, np.ndarray)  # type: ignore
                    else preds_tmp.numpy()
                )
            else:
                raise TypeError(
                    f"Model output type {type(preds_tmp)} not supported. The model \
                    output type needs to be one of list, tuple, NDArray or \
                    torch.Tensor."
                )
    concat = partial(np.concatenate, axis=0) if return_np else partial(torch.cat, dim=0)
    out: tuple | np.ndarray | torch.Tensor = (
        tuple(concat(p) for p in preds) if isinstance(preds, tuple) else concat(preds)  # type: ignore
    )
    return out
