from __future__ import annotations

import json
import shutil
from pathlib import Path
from random import choice
from typing import Literal, Sequence, Union, overload
from zipfile import ZipFile

import numpy as np
import pytest
from numpy.random import randint, random
from numpy.typing import NDArray
from PIL import Image

from dataeval.config import set_seed
from dataeval.data import Metadata
from dataeval.typing import ImageClassificationDataset, ObjectDetectionDataset
from dataeval.utils.data._dataset import to_image_classification_dataset, to_object_detection_dataset

BoxLike = Union[
    NDArray[np.number],
    Sequence[int],
    Sequence[float],
    tuple[int, int, int, int],
    tuple[float, float, float, float],
]

TEMP_CONTENTS = "ABCDEF1234567890"

pytest_plugins = ["tests.fixtures.metadata"]

set_seed(0, all_generators=True)


def to_metadata(factors, class_labels, continuous_factor_bins=None, exclude=None, include=None):
    if isinstance(class_labels[0], str):
        class_names = sorted(set(class_labels))
        class_labels = [class_names.index(i) for i in class_labels]
    else:
        class_names = None
    dataset = to_image_classification_dataset(
        np.zeros((len(class_labels), 1, 16, 16)), labels=class_labels, metadata=factors, classes=class_names
    )
    metadata = Metadata(
        dataset,  # type: ignore
        continuous_factor_bins=continuous_factor_bins,
        exclude=exclude,
        include=include,
    )
    return metadata


def get_images(count: int, channels: int | None, dims: int | None):
    channels = channels or 3
    dims = dims or 64
    return [np.random.random((channels, dims, dims)) for _ in range(count)]


def get_bboxes(count: int, boxes_per_image: int, as_float: bool):
    boxes = []
    for _ in range(count):
        box = []
        for _ in range(boxes_per_image):
            if as_float:
                x0, y0 = (random() * 24), (random() * 24)
                x1, y1 = x0 + 1 + (random() * 23), y0 + 1 + (random() * 23)
            else:
                x0, y0 = randint(0, 24), randint(0, 24)
                x1, y1 = x0 + randint(1, 24), y0 + randint(1, 24)
            box.append([x0, y0, x1, y1])
        boxes.append(np.asarray(box))
    return boxes


@pytest.fixture(scope="session")
def DATA_1():
    return get_images(10, 1, 64)


@pytest.fixture(scope="session")
def DATA_3():
    return get_images(10, 3, 64)


@overload
def _get_dataset(
    images: list[np.ndarray] | tuple[int, int | None, int | None] | None,
    targets_per_image: int,
    as_float: bool = False,
    override: list[BoxLike] | dict[int, list[BoxLike]] | None = None,
) -> ObjectDetectionDataset: ...


@overload
def _get_dataset(
    images: list[np.ndarray] | tuple[int, int | None, int | None] | None,
    targets_per_image: Literal[None] = None,
    as_float: bool = False,
    override: list[BoxLike] | dict[int, list[BoxLike]] | None = None,
) -> ImageClassificationDataset: ...


def _get_dataset(
    images: list[np.ndarray] | tuple[int, int | None, int | None] | None,
    targets_per_image: int | None = None,
    as_float: bool = False,
    override: list[BoxLike] | dict[int, list[BoxLike]] | None = None,
):
    if images is None:
        images = (10, 3, 64)
    if isinstance(images, tuple):
        images = get_images(images[0], images[1], images[2])
    length = len(images)
    override_dict = dict(enumerate(override)) if isinstance(override, list) else override
    if targets_per_image:
        labels = [[0 for _ in range(targets_per_image)] for _ in range(length)]
        bboxes = get_bboxes(length, targets_per_image, as_float)
        if override_dict is not None:
            for i, boxes in override_dict.items():
                bboxes[i] = boxes
        return to_object_detection_dataset(images, labels, bboxes, None, None)
    else:
        labels = [0 for _ in range(length)]
        return to_image_classification_dataset(images, labels, None, None)


def _get_ic_dataset(
    images: list[np.ndarray] | tuple[int, int | None, int | None] | None,
    as_float: bool = False,
    override: list[BoxLike] | dict[int, list[BoxLike]] | None = None,
) -> ImageClassificationDataset:
    return _get_dataset(images, None, as_float, override)


def _get_od_dataset(
    images: list[np.ndarray] | tuple[int, int | None, int | None] | None,
    targets_per_image: int,
    as_float: bool = False,
    override: list[BoxLike] | dict[int, list[BoxLike]] | None = None,
) -> ObjectDetectionDataset:
    return _get_dataset(images, targets_per_image, as_float, override)


@pytest.fixture
def get_ic_dataset():
    return _get_ic_dataset


@pytest.fixture
def get_od_dataset():
    return _get_od_dataset


@pytest.fixture(scope="session")
def RNG() -> np.random.Generator:
    return np.random.default_rng(0)


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
    random_temp = tmp_path / "random"
    random_temp.mkdir()
    zip_temp = temp / "testing.zip"
    file_temp = random_temp / "stuff.txt"
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
def mnist_folder(tmp_path):
    temp = tmp_path / "data"
    temp.mkdir()
    mnist_folder = temp / "mnist"
    mnist_folder.mkdir(exist_ok=True)
    yield mnist_folder


@pytest.fixture
def dataset_nested_folder(mnist_folder):
    random_temp = mnist_folder.parent.parent / "random"
    random_temp.mkdir()
    zip_temp = mnist_folder / "mnist_c.zip"
    nested_temp = random_temp / "mnist_c" / "translate"
    nested_temp.mkdir(parents=True)
    temp_labels = nested_temp / "train_labels.npy"
    temp_images = nested_temp / "train_images.npy"
    labels = np.arange(10).repeat(500)
    train = np.ones((5000, 28, 28, 1)) * labels[:, None, None, None]
    np.save(temp_images, train, allow_pickle=False)
    np.save(temp_labels, labels, allow_pickle=False)

    shutil.make_archive(str(mnist_folder / "mnist_c"), "zip", root_dir=random_temp)
    yield zip_temp


@pytest.fixture
def wrong_mnist(mnist_folder):
    ident_temp = mnist_folder / "mnist_c" / "identity"
    ident_temp.mkdir(parents=True, exist_ok=True)
    labels = np.arange(10).repeat(500)
    train = np.ones((5000, 28, 28, 1)) * labels[:, None, None, None]

    np.save(ident_temp / "train_images.npy", train, allow_pickle=False)
    np.save(ident_temp / "train_labels.npy", labels, allow_pickle=False)
    np.save(ident_temp / "test_images.npy", train, allow_pickle=False)
    np.save(ident_temp / "test_labels.npy", labels, allow_pickle=False)
    yield mnist_folder.parent


@pytest.fixture(scope="session")
def mnist_npy(tmp_path_factory):
    temp = tmp_path_factory.mktemp("data")
    mnist_temp = temp / "mnist"
    mnist_temp.mkdir(exist_ok=True)
    mnistc_temp = mnist_temp / "mnist_c" / "identity"
    mnistc_temp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    labels = np.concatenate([rng.choice(10, 10000), np.arange(10).repeat(4000)])
    train = np.ones((50000, 28, 28)) * labels[:, np.newaxis, np.newaxis]
    train[:, 13:16, 13:16] += 1
    train[-5000:, 13:16, 13:16] += 1

    np.savez(mnist_temp / "mnist.npz", x_train=train, x_test=train[:10000], y_train=labels, y_test=labels[:10000])
    np.save(mnistc_temp / "train_images.npy", train[..., None], allow_pickle=False)
    np.save(mnistc_temp / "train_labels.npy", labels, allow_pickle=False)
    yield mnist_temp


@pytest.fixture(scope="session")
def ship_fake(tmp_path_factory):
    temp = tmp_path_factory.mktemp("data")
    ship_temp = temp / "ships" / "shipsnet"
    ship_temp.mkdir(parents=True, exist_ok=True)
    scene_temp = temp / "ships" / "scenes"
    scene_temp.mkdir(parents=True, exist_ok=True)
    labels = np.concatenate([np.ones(1000, dtype=np.uint8), np.zeros(3000, dtype=np.uint8)])
    data = np.ones((4000, 10, 10, 3), dtype=np.uint8) * labels[:, np.newaxis, np.newaxis, np.newaxis]
    for i in range(labels.size):
        image = Image.fromarray(data[i])
        image.save(ship_temp / f"{labels[i]}__abc__105_{i}.png")
    scene = Image.fromarray(np.ones((1500, 1250, 3), dtype=np.uint8))
    scene.save(scene_temp / "img_1.png")
    with open(temp / "ships" / "shipsnet.json", "w") as f:
        json.dump({"saving": "as_a_json"}, f)
    yield temp


@pytest.fixture
def cifar_fake(tmp_path):
    temp = tmp_path / "data"
    temp.mkdir()
    cifar_temp = temp / "cifar10" / "cifar-10-batches-bin"
    cifar_temp.mkdir(parents=True, exist_ok=True)

    for filename in [
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin",
        "test_batch.bin",
    ]:
        with open(cifar_temp / filename, "wb") as file:
            # Write 10000 images for each batch
            for _ in range(10000):
                # Write label
                file.write(choice(range(10)).to_bytes(1, byteorder="big"))
                # Write 3072 zeros
                file.write(bytes(3072))
    yield temp


@pytest.fixture(scope="session")
def milco_fake(tmp_path_factory):
    temp = tmp_path_factory.mktemp("data")
    a_temp = temp / "milco" / "2015"
    a_temp.mkdir(parents=True, exist_ok=True)
    b_temp = temp / "milco" / "2017"
    b_temp.mkdir(parents=True, exist_ok=True)
    c_temp = temp / "milco" / "2021"
    c_temp.mkdir(parents=True, exist_ok=True)
    data = (np.random.random((12, 10, 10, 3)) * 255).astype(np.uint8)
    for i in range(6):
        image = Image.fromarray(data[i])
        image.save(a_temp / f"{i}_2015.jpg")
        with open(a_temp / f"{i}_2015.txt", mode="w") as f:
            f.write(f"{int(np.random.choice([0, 1]))} {300 / 1024} {753 / 1024} {56 / 1024} {43 / 1024}")
    for i in range(2):
        image = Image.fromarray(data[i + 6])
        image.save(b_temp / f"{i}_2017.jpg")
        with open(b_temp / f"{i}_2017.txt", mode="w") as f:
            f.write("")
    for i in range(4):
        image = Image.fromarray(data[i + 8])
        image.save(c_temp / f"{i}_2021.jpg")
        object1 = f"{int(np.random.choice([0, 1]))} {300 / 1024} {753 / 1024} {56 / 1024} {43 / 1024}"
        object2 = f"{int(np.random.choice([0, 1]))} {829 / 1024} {115 / 1024} {56 / 1024} {43 / 1024}"
        with open(c_temp / f"{i}_2015.txt", mode="w") as f:
            f.write(f"{object1}\n{object2}")
    yield temp


@pytest.fixture
def voc_fake(tmp_path, request):
    marker = request.node.get_closest_marker("year")
    year = 2012 if marker is None else marker.args[0]

    temp = tmp_path / "data"
    temp.mkdir()
    random_temp = tmp_path / "vocdataset"
    random_temp.mkdir()
    if year != 2011:
        base_nested = random_temp / "VOCdevkit" / f"VOC{year}"
    else:
        base_nested = random_temp / "TrainVal" / "VOCdevkit" / f"VOC{year}"
    base_nested.mkdir(parents=True)
    img_temp = base_nested / "JPEGImages"
    img_temp.mkdir(exist_ok=True)
    label_temp = base_nested / "Annotations"
    label_temp.mkdir(exist_ok=True)
    sets_temp = base_nested / "ImageSets" / "Main"
    sets_temp.mkdir(parents=True, exist_ok=True)
    seg_temp = base_nested / "SegmentationClass"
    seg_temp.mkdir(exist_ok=True)

    file_list = [f"2009_00{i}573" for i in range(5)]
    img = Image.fromarray(np.ones((10, 10, 3), dtype=np.uint8))
    complicate = np.zeros((10, 10, 3), dtype=np.uint8)
    complicate[3:7, 3:7] = 4
    seg = Image.fromarray(complicate)
    annotation_str = """
    <annotation>
        <folder>VOC2012</folder>
        <filename>2009_001573.jpg</filename>
        <source>
            <database>The VOC2009 Database</database>
            <annotation>PASCAL VOC2009</annotation>
            <image>flickr</image>
        </source>
        <size>
            <width>500</width>
            <height>375</height>
            <depth>3</depth>
        </size>
        <segmented>1</segmented>
        <object>
            <name>dog</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>123</xmin>
                <ymin>115</ymin>
                <xmax>379</xmax>
                <ymax>275</ymax>
            </bndbox>
        </object>
        <object>
            <name>chair</name>
            <pose>Frontal</pose>
            <truncated>1</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>75</xmin>
                <ymin>1</ymin>
                <xmax>428</xmax>
                <ymax>375</ymax>
            </bndbox>
        </object>
    </annotation>
    """
    # Creating the sample files
    for file in file_list:
        img_save = img_temp / (file + ".jpg")
        img.save(img_save)
        seg_save = seg_temp / (file + ".jpg")
        seg.save(seg_save)
        label_save = label_temp / (file + ".xml")
        with open(label_save, "w") as f:
            f.write(annotation_str)
    with open(sets_temp / "train.txt", "w") as f:
        f.write("\n".join(file_list[1:4]))
    with open(sets_temp / "val.txt", "w") as f:
        f.write("\n".join([file_list[0], file_list[-1]]))
    with open(sets_temp / "trainval.txt", "w") as f:
        f.write("\n".join(file_list))

    # Making the tar file
    shutil.make_archive(str(temp / f"VOCtrainval-{year}"), "tar", root_dir=random_temp)

    # Remove all of the files
    shutil.rmtree(random_temp)

    yield temp


@pytest.fixture
def voc_fake_test(voc_fake):
    temp = voc_fake
    random_temp = temp / "vocdataset"
    random_temp.mkdir(exist_ok=True)
    dev_temp = random_temp / "VOCdevkit"
    dev_temp.mkdir(exist_ok=True)
    base_nested = dev_temp / "VOC2012"
    base_nested.mkdir(exist_ok=True)
    img_temp = base_nested / "JPEGImages"
    img_temp.mkdir(exist_ok=True)
    label_temp = base_nested / "Annotations"
    label_temp.mkdir(exist_ok=True)
    sets_temp = base_nested / "ImageSets" / "Main"
    sets_temp.mkdir(parents=True, exist_ok=True)
    seg_temp = base_nested / "SegmentationClass"
    seg_temp.mkdir(exist_ok=True)

    file_list = [f"2009_0015{5 + i}3" for i in range(5)]
    img = Image.fromarray(np.ones((10, 10, 3), dtype=np.uint8))
    complicate = np.zeros((10, 10, 3), dtype=np.uint8)
    complicate[3:7, 3:7] = 4
    seg = Image.fromarray(complicate)
    annotation_str = """
    <annotation>
        <folder>VOC2012</folder>
        <filename>2009_001563.jpg</filename>
        <source>
            <database>The VOC2009 Database</database>
            <annotation>PASCAL VOC2009</annotation>
            <image>flickr</image>
        </source>
        <size>
            <width>500</width>
            <height>375</height>
            <depth>3</depth>
        </size>
        <segmented>1</segmented>
        <object>
            <name>dog</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>123</xmin>
                <ymin>115</ymin>
                <xmax>379</xmax>
                <ymax>275</ymax>
            </bndbox>
        </object>
        <object>
            <name>chair</name>
            <pose>Frontal</pose>
            <truncated>1</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>75</xmin>
                <ymin>1</ymin>
                <xmax>428</xmax>
                <ymax>375</ymax>
            </bndbox>
        </object>
    </annotation>
    """
    # Creating the sample files
    for i, file in enumerate(file_list):
        img_save = img_temp / (file + ".jpg")
        img.save(img_save)
        seg_save = seg_temp / (file + ".jpg")
        seg.save(seg_save)
        if i < 3:
            label_save = label_temp / (file + ".xml")
            with open(label_save, "w") as f:
                f.write(annotation_str)
    with open(sets_temp / "test.txt", "w") as f:
        f.write("\n".join(file_list))

    # Making the tar file
    shutil.make_archive(str(temp / "VOC2012test"), "tar", root_dir=random_temp)

    # Removing all the folders
    shutil.rmtree(random_temp)
    yield temp


@pytest.fixture(scope="session")
def antiuav_fake(tmp_path_factory):
    temp = tmp_path_factory.mktemp("data")
    img_temp = temp / "antiuavdetection" / "train" / "img"
    img_temp.mkdir(parents=True, exist_ok=True)
    ann_temp = temp / "antiuavdetection" / "train" / "xml"
    ann_temp.mkdir(exist_ok=True)
    data = (np.random.random((12, 10, 10, 3)) * 255).astype(np.uint8)
    annotation_str = """
    <annotation>
        <folder>train</folder>
        <filename>00100.jpg</filename>
        <path>./train/00100.jpg</path>
        <source>
            <database>DUT Anti-UAV Detection</database>
        </source>
        <size>
            <width>550</width>
            <height>412</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>
        <object>
            <name>UAV</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>228</xmin>
                <ymin>155</ymin>
                <xmax>353</xmax>
                <ymax>245</ymax>
            </bndbox>
        </object>
        <object>
            <name>UAV</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>36</xmin>
                <ymin>312</ymin>
                <xmax>49</xmax>
                <ymax>337</ymax>
            </bndbox>
        </object>
    </annotation>
    """

    for i in range(len(data)):
        image = Image.fromarray(data[i])
        image.save(img_temp / f"{i:05}.jpg")
        with open(ann_temp / f"{i:05}.xml", mode="w") as f:
            f.write(annotation_str)
    yield temp
