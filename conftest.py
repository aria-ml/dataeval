from __future__ import annotations

from zipfile import ZipFile

import numpy as np
import pytest

import dataeval.metrics.stats.base as base
from tests.doctest import *  # noqa: F403 - configure doctest

TEMP_CONTENTS = "ABCDEF1234567890"

# Temporarily set DEFAULT_PROCESSES to 1
base.DEFAULT_PROCESSES = 1


@pytest.fixture(scope="session")
def RNG():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def mnist_file(tmp_path_factory):
    temp = tmp_path_factory.mktemp("data")
    mnist_file = temp / "mnist.txt"

    with mnist_file.open(mode="w") as f:
        f.write(TEMP_CONTENTS)
    yield mnist_file


@pytest.fixture(scope="session")
def mnist_download(tmp_path_factory):
    temp = tmp_path_factory.mktemp("data")
    mnist_temp = temp / "mnist" / "mnist.txt"
    mnist_temp.parent.mkdir(exist_ok=True)
    with mnist_temp.open(mode="w") as f:
        f.write(TEMP_CONTENTS)
    yield mnist_temp.parents[1], mnist_temp.name


@pytest.fixture(scope="session")
def zip_file(tmp_path_factory):
    temp = tmp_path_factory.mktemp("data")
    zip_temp = temp / "mnist.zip"
    file_temp = temp / "mnist_stuff.txt"
    with file_temp.open(mode="w") as f:
        f.write(TEMP_CONTENTS)
    with ZipFile(zip_temp, "w") as myzip:
        myzip.write(file_temp)
    yield zip_temp


@pytest.fixture(scope="session")
def mnist_npy(tmp_path_factory):
    temp = tmp_path_factory.mktemp("data")
    mnist_temp = temp / "mnist"
    mnist_temp.mkdir(exist_ok=True)
    mnistc_temp = temp / "mnist_c" / "identity"
    mnistc_temp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    labels = np.concatenate([rng.choice(10, 10000), np.arange(10).repeat(5000)])
    train = np.ones((60000, 28, 28)) * labels[:, np.newaxis, np.newaxis]
    train[:, 13:16, 13:16] += 1
    train[-5000:, 13:16, 13:16] += 1

    np.savez(mnist_temp / "mnist.npz", x_train=train, x_test=train[:10000], y_train=labels, y_test=labels[:10000])
    np.save(mnistc_temp / "train_images.npy", train, allow_pickle=False)
    np.save(mnistc_temp / "train_labels.npy", labels, allow_pickle=False)
    yield temp
