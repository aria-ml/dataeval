from __future__ import annotations

import warnings
from contextlib import contextmanager
from os import makedirs
from pathlib import Path
from typing import Literal

import pytest
from numpy.typing import NDArray

try:
    from dataeval._internal.datasets import MNIST
except ImportError:
    MNIST = None


@contextmanager
def wait_lock(name: str, timeout: int = 120):
    try:
        from filelock import FileLock
    except ImportError:
        warnings.warn("FileLock dependency not found, read/write collisions possible when running in parallel.")
        yield
        return

    path = Path(name)

    if not path.is_absolute():
        path = path.resolve()

    # If we are writing to a new temp folder, create any parent paths
    makedirs(path.parent, exist_ok=True)

    # https://stackoverflow.com/a/60281933/315168
    lock_file = path.parent / (path.name + ".lock")

    lock = FileLock(lock_file, timeout=timeout)
    try:
        with lock:
            yield
    finally:
        lock.release()


@pytest.fixture(scope="session")
def mnist():
    def _method(
        train: bool = True,
        size: int = 1000,
        unit_normalize: bool = False,
        dtype: type | None = None,
        channels: Literal["channels_first", "channels_last"] | None = None,
        flatten: bool = False,
    ) -> tuple[NDArray, NDArray]:
        if MNIST is None:
            raise ImportError("MNIST dataset requires torch and torchvision.")

        with wait_lock("./data/mnist"):
            dataset = MNIST(
                root="./data/",
                train=train,
                download=True,
                size=size,
                unit_interval=unit_normalize,
                dtype=dtype,
                channels=channels,
                flatten=flatten,
            )

        return dataset.data, dataset.targets

    return _method
