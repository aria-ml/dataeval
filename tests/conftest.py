from __future__ import annotations

import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

skip_mnist = pytest.mark.skip(reason="Skip MNIST tests for unit testing")

try:
    from torch.utils.data import Dataset

    from dataeval._internal.datasets import MNIST
except ImportError:
    MNIST = None


@pytest.fixture
def np_rand_1000_28x28():
    return np.random.random((1000, 28 * 28)).astype(np.float32)


@contextmanager
def wait_lock(path: Path, timeout: int = 30):
    try:
        from filelock import FileLock
    except ImportError:
        warnings.warn("FileLock dependency not found, read/write collisions possible when running in parallel.")
        yield
        return

    lock = FileLock(str(path), timeout=timeout)
    with lock:
        yield


def mnist(
    root: str | Path = "./data",
    train: bool = True,
    download: bool = True,
    size: int = 1000,
    unit_normalize: bool = False,
    dtype: type | None = None,
    channels: Literal["channels_first", "channels_last"] | None = None,
    flatten: bool = False,
    normalize: tuple[float, float] | None = None,
    corruption: Literal[
        "identity",
        "shot_noise",
        "impulse_noise",
        "glass_blur",
        "motion_blur",
        "shear",
        "scale",
        "rotate",
        "brightness",
        "translate",
        "stripe",
        "fog",
        "spatter",
        "dotted_line",
        "zigzag",
        "canny_edges",
    ]
    | None = None,
    classes: Literal[
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    ]
    | int
    | list[int]
    | list[
        Literal[
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]
    ]
    | None = None,
    balance: bool = True,
    randomize: bool = False,
    slice_back: bool = False,
    verbose: bool = True,
    return_dataset: bool = False,
) -> tuple[NDArray, NDArray] | Dataset:
    if MNIST is None:
        raise ImportError("MNIST dataset requires torch and torchvision.")

    path = Path(root)
    if not path.is_absolute():
        path = path.resolve()

    temp_dir = tempfile.gettempdir()
    lock_file = Path(temp_dir, "mnist.lock")

    with wait_lock(lock_file, timeout=60):
        dataset = MNIST(
            root=root,
            train=train,
            download=download,
            size=size,
            unit_interval=unit_normalize,
            dtype=dtype,
            channels=channels,
            flatten=flatten,
            normalize=normalize,
            corruption=corruption,
            classes=classes,
            balance=balance,
            randomize=randomize,
            slice_back=slice_back,
            verbose=verbose,
        )

    assert False

    if return_dataset:
        return dataset
    return dataset.data, dataset.targets
