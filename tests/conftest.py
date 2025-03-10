from __future__ import annotations

import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from dataeval.utils.data._metadata import Metadata
from dataeval.utils.data._targets import Targets

TEMP_CONTENTS = "ABCDEF1234567890"


def preprocess(factors, class_labels, continuous_factor_bins=None, exclude=None, include=None):
    if isinstance(class_labels[0], str):
        class_names = sorted(set(class_labels))
        index2label = {i: class_names[i] for i in range(len(class_names))}
        class_labels = [class_names.index(i) for i in class_labels]
    elif isinstance(class_labels[0], int):
        index2label = {i: str(i) for i in range(max(class_labels) + 1)}
    else:
        index2label = {}
    targets = Targets(labels=np.asarray(class_labels), scores=np.ndarray([]), bboxes=None, source=None)
    metadata = Metadata(
        None,  # type: ignore
        continuous_factor_bins=continuous_factor_bins,
        exclude=exclude,
        include=include,
    )

    # set internal attributes
    metadata._raw = [{} for _ in range(len(class_labels))]
    metadata._targets = targets
    metadata._class_labels = targets.labels
    metadata._class_names = [index2label.get(i, str(i)) for i in np.unique(targets.labels)]
    metadata._collated = True
    metadata._merged = factors, {}
    return metadata


@pytest.fixture(scope="session")
def RNG():
    return np.random.default_rng(0)


@pytest.fixture
def mnist_folder(tmp_path):
    temp = tmp_path / "data"
    temp.mkdir()
    mnist_folder = temp / "mnist"
    mnist_folder.mkdir(exist_ok=True)
    yield mnist_folder


@pytest.fixture
def dataset_no_zip(tmp_path):
    temp = tmp_path / "data"
    temp.mkdir()
    file_temp = temp / "stuff.txt"
    with file_temp.open(mode="w") as f:
        f.write(TEMP_CONTENTS)
    yield file_temp


@pytest.fixture
def dataset_single_zip(tmp_path):
    temp = tmp_path / "data"
    temp.mkdir()
    zip_temp = temp / "testing.zip"
    file_temp = "stuff.txt"
    with open(file_temp, mode="w") as f:
        f.write(TEMP_CONTENTS)
    with ZipFile(zip_temp, "w") as myzip:
        myzip.write(file_temp)
    yield zip_temp


@pytest.fixture
def dataset_nested_zip(tmp_path):
    temp = tmp_path / "data"
    temp.mkdir()
    zip_temp = temp / "testing.zip"
    nested_temp = "nested.zip"
    file_temp = "stuff.txt"
    with open(file_temp, mode="w") as f:
        f.write(TEMP_CONTENTS)
    with ZipFile(nested_temp, "w") as myzip:
        myzip.write(file_temp)
    with ZipFile(zip_temp, "w") as myzip2:
        myzip2.write(nested_temp)
    Path(nested_temp).unlink()
    Path(file_temp).unlink()
    yield zip_temp


@pytest.fixture
def dataset_nested_folder(mnist_folder):
    zip_temp = mnist_folder / "mnist_c.zip"
    nested_temp = Path("mnist_c") / "translate"
    nested_temp.mkdir(parents=True)
    file_temp = nested_temp / "data.txt"
    with open(file_temp, mode="w") as f:
        f.write(TEMP_CONTENTS)
    shutil.make_archive(str(mnist_folder / "mnist_c"), "zip", base_dir="./mnist_c")
    file_temp.unlink()
    nested_temp.rmdir()
    nested_temp.parent.rmdir()
    yield zip_temp


@pytest.fixture
def wrong_mnist(mnist_folder):
    ident_temp = mnist_folder / "identity"
    ident_temp.mkdir(exist_ok=True)
    labels = np.arange(10).repeat(500)
    train = np.ones((5000, 28, 28, 1)) * labels[:, None, None, None]

    np.save(ident_temp / "train_images.npy", train, allow_pickle=False)
    np.save(ident_temp / "train_labels.npy", labels, allow_pickle=False)
    yield mnist_folder.parent


@pytest.fixture(scope="session")
def mnist_npy(tmp_path_factory):
    temp = tmp_path_factory.mktemp("data")
    mnist_temp = temp / "mnist"
    mnist_temp.mkdir(exist_ok=True)
    mnistc_temp = temp / "mnist" / "identity"
    mnistc_temp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    labels = np.concatenate([rng.choice(10, 10000), np.arange(10).repeat(5000)])
    train = np.ones((60000, 28, 28)) * labels[:, np.newaxis, np.newaxis]
    train[:, 13:16, 13:16] += 1
    train[-5000:, 13:16, 13:16] += 1

    np.savez(mnist_temp / "mnist.npz", x_train=train, x_test=train[:10000], y_train=labels, y_test=labels[:10000])
    np.save(mnistc_temp / "train_images.npy", train[..., None], allow_pickle=False)
    np.save(mnistc_temp / "train_labels.npy", labels, allow_pickle=False)
    yield temp


@pytest.fixture(scope="session")
def ship_fake(tmp_path_factory):
    temp = tmp_path_factory.mktemp("data")
    ship_temp = temp / "shipdataset" / "shipsnet"
    ship_temp.mkdir(parents=True, exist_ok=True)
    scene_temp = temp / "shipdataset" / "scenes"
    scene_temp.mkdir(parents=True, exist_ok=True)
    labels = np.concatenate([np.ones(1000, dtype=np.uint8), np.zeros(3000, dtype=np.uint8)])
    data = np.ones((4000, 10, 10, 3), dtype=np.uint8) * labels[:, np.newaxis, np.newaxis, np.newaxis]
    for i in range(labels.size):
        image = Image.fromarray(data[i])
        image.save(ship_temp / f"{labels[i]}__abc__105_{i}.png")
    scene = Image.fromarray(np.ones((1500, 1250, 3), dtype=np.uint8))
    scene.save(scene_temp / "img_1.png")
    yield temp
