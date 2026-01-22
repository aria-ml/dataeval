"""doctest fixtures"""

import pathlib
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest
import sklearn.datasets as dsets
import torch
from numpy.typing import NDArray
from typing_extensions import Self

from dataeval._metadata import FactorInfo, Metadata
from dataeval.config import set_batch_size, set_seed
from dataeval.core import calculate
from dataeval.flags import ImageStats
from dataeval.shift._ood._base import OODOutput

# Manually add the import path for test_drift_uncertainty
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute() / "tests" / "shift"))
from test_drift_uncertainty import PtModel

# Set global batch_size for doctests
set_batch_size(32)

# Set numpy print option to legacy 1.25 so native numpy types
# are not printed with dtype information.
if np.__version__[0] == "2":
    # WITHOUT LEGACY=1.25
    # >>> np.int32(16)
    # np.int32(16)

    # WITH LEGACY=1.25
    # >>> np.int32(16)
    # 16
    np.set_printoptions(legacy="1.25", precision=3)  # type: ignore
else:
    np.set_printoptions(precision=3)


@dataclass
class MockMetadata:
    """Simple Metadata implementation for doctests."""

    class_labels: NDArray[np.intp]
    factor_data: NDArray[np.int64]
    factor_names: Sequence[str]
    is_discrete: Sequence[bool]
    index2label: Mapping[int, str]


def generate_random_metadata(
    labels: Sequence[str], factors: Mapping[str, Sequence[str | int]], length: int, random_seed: int
) -> MockMetadata:
    rng = np.random.default_rng(random_seed)
    labels_arr = rng.choice(range(len(labels)), (length))

    # Create artificially biased metadata where certain classes correlate with specific factors
    metadata_dict: dict[str, list[Any]] = {}
    factor_names = list(factors.keys())

    # If we have the expected doctor/artist/teacher example with age/income/gender factors
    if set(labels) == {"doctor", "artist", "teacher"} and set(factor_names) == {"age", "income", "gender"}:
        # Create biased distributions
        metadata_dict["age"] = []
        metadata_dict["income"] = []
        metadata_dict["gender"] = []

        for label_idx in labels_arr:
            if label_idx == 0:  # doctor
                # Doctors: tend to be older (35, 45), higher income, more male
                metadata_dict["age"].append(rng.choice([35, 45, 30, 25], p=[0.5, 0.35, 0.1, 0.05]))
                metadata_dict["income"].append(rng.choice([80000, 65000, 50000], p=[0.7, 0.25, 0.05]))
                metadata_dict["gender"].append(rng.choice(["M", "F"], p=[0.8, 0.2]))
            elif label_idx == 1:  # artist
                # Artists: tend to be younger (25, 30), lower income, more female
                metadata_dict["age"].append(rng.choice([25, 30, 35, 45], p=[0.5, 0.35, 0.1, 0.05]))
                metadata_dict["income"].append(rng.choice([50000, 65000, 80000], p=[0.6, 0.3, 0.1]))
                metadata_dict["gender"].append(rng.choice(["F", "M"], p=[0.65, 0.35]))
            else:  # teacher (label_idx == 2)
                # Teachers: middle-aged (30, 35), middle income, balanced gender
                metadata_dict["age"].append(rng.choice([30, 35, 25, 45], p=[0.4, 0.4, 0.1, 0.1]))
                metadata_dict["income"].append(rng.choice([65000, 50000, 80000], p=[0.5, 0.35, 0.15]))
                metadata_dict["gender"].append(rng.choice(["F", "M"], p=[0.55, 0.45]))
    else:
        # Default: random generation for other factor combinations
        metadata_dict = {k: list(rng.choice(v, (length))) for k, v in factors.items()}

    # Sort factor names to ensure consistent ordering
    sorted_factor_names = sorted(factors.keys())

    # Convert factors to binned_data (digitize each factor) in sorted order
    binned_columns = []
    for name in sorted_factor_names:
        factor_values = metadata_dict[name]
        _, inverse = np.unique(factor_values, return_inverse=True)
        binned_columns.append(inverse)
    binned_data = np.column_stack(binned_columns).astype(np.int64) if binned_columns else np.array([], dtype=np.int64)

    return MockMetadata(
        class_labels=labels_arr,
        factor_data=binned_data,
        factor_names=sorted_factor_names,
        is_discrete=[True] * len(factors),
        index2label=dict(enumerate(labels)),
    )


def get_one_hot(class_count: int, sub_labels: Sequence[int]) -> list[list[float]]:
    one_hot = [[0.0] * class_count for _ in range(len(sub_labels))]
    for i, label in enumerate(sub_labels):
        one_hot[i][label] = 1.0
    return one_hot


def get_object_detection_target(idx: int, det_data_mm: MagicMock) -> MagicMock:
    obj_det_mm = MagicMock(
        _labels=det_data_mm._labels[idx],
        _bboxes=det_data_mm._bboxes[idx],
        _scores=get_one_hot(len(det_data_mm._classes), det_data_mm._labels[idx]),
        labels=det_data_mm._labels[idx],
        boxes=det_data_mm._bboxes[idx],
    )
    obj_det_mm.scores = obj_det_mm._scores
    return obj_det_mm


class ClassificationModel(PtModel):
    def __init__(self) -> None:
        super().__init__(16, 3, softmax=True, dropout=False)


@pytest.fixture(autouse=True, scope="function")
def reset_random_seed() -> Any:
    """
    Automatically reset the random seed before each test.
    This ensures deterministic behavior for tests involving randomness.
    """
    # Set manual seeds
    set_seed(0, all_generators=True)
    yield


@pytest.fixture(autouse=True, scope="session")
def add_tmp_path(doctest_namespace: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_path = tmp_path_factory.mktemp("dataeval_doctest")
    doctest_namespace["tmp_path"] = tmp_path


@pytest.fixture(autouse=True, scope="session")
def add_all(doctest_namespace: dict[str, Any]) -> None:
    doctest_namespace["np"] = np
    doctest_namespace["generate_random_metadata"] = generate_random_metadata
    doctest_namespace["OODOutput"] = OODOutput
    doctest_namespace["ExampleDataset"] = ExampleDataset


@pytest.fixture(autouse=True, scope="session")
def doctest_metadata_explanatory_funcs(doctest_namespace: dict[str, Any]) -> None:
    md1 = MagicMock(spec=Metadata)
    md2 = MagicMock(spec=Metadata)

    factor_names = ["time", "altitude"]
    factor_data1 = np.array([[1.2, 235], [3.4, 6789], [5.6, 101112]])
    factor_data2 = np.array([[7.8, 532], [9.10, 9876], [11.12, 211101]])
    factor_info = dict.fromkeys(factor_names, FactorInfo("continuous"))

    md1.factor_names = factor_names
    md1.factor_data = factor_data1
    md1.factor_info = factor_info
    md1.dataframe = pl.DataFrame(factor_data1, schema=factor_names)
    md1.filter_by_factor = lambda _: factor_data1
    md1.calculate_distance = lambda x: Metadata.calculate_distance(md1, x)

    md2.factor_names = factor_names
    md2.factor_data = factor_data2
    md2.factor_types = factor_info
    md2.dataframe = pl.DataFrame(factor_data2, schema=factor_names)
    md2.filter_by_factor = lambda _: factor_data2
    md2.calculate_distance = lambda x: Metadata.calculate_distance(md2, x)

    doctest_namespace["metadata1"] = md1
    doctest_namespace["metadata2"] = md2


@pytest.fixture(autouse=True, scope="session")
def doctest_detectors_linters_duplicates(doctest_namespace: dict[str, Any]) -> None:
    rng = np.random.default_rng(42)
    base = np.concatenate([np.ones((32, 64)), np.zeros((32, 64))])
    images = np.stack([rng.permutation(base) * i for i in range(50)], axis=0)
    images = (images * 255).astype(np.uint8)
    images[16] = images[37]
    images[3] = images[20]
    images[5] = np.ones((64, 64), dtype=np.uint8) * 150

    """dataeval.quality.Duplicates"""

    doctest_namespace["duplicate_images"] = images
    doctest_namespace["hashes1"] = calculate(images[:24], None, ImageStats.HASH)
    doctest_namespace["hashes2"] = calculate(images[25:], None, ImageStats.HASH)


@pytest.fixture(autouse=True, scope="session")
def doctest_detectors_linters_outliers(doctest_namespace: dict[str, Any]) -> None:
    images = np.ones((30, 1, 128, 128), dtype=np.int32) * 2
    images = images + np.repeat(np.arange(10), 3 * 128 * 128).reshape(30, -1, 128, 128)
    images[10:13, :, 50:80, 50:80] = 0
    images[[7, 11, 18, 25]] = 512

    """dataeval.quality.Outliers"""

    doctest_namespace["outlier_images"] = images
    doctest_namespace["stats1"] = calculate(images[:14], None, ImageStats.PIXEL)
    doctest_namespace["stats2"] = calculate(images[15:], None, ImageStats.PIXEL)


@pytest.fixture(autouse=True, scope="session")
def doctest_detectors_drift_uncertainty(doctest_namespace: dict[str, Any]) -> None:
    x_ref = np.random.random((500, 16)).astype(np.float32)
    x_test = np.ones_like(x_ref)
    doctest_namespace["ClassificationModel"] = ClassificationModel
    doctest_namespace["x_ref"] = x_ref
    doctest_namespace["x_test"] = x_test


@pytest.fixture(autouse=True, scope="session")
def doctest_metrics_bias_coverage(doctest_namespace: dict[str, Any]) -> None:
    blobs = dsets.make_blobs(n_samples=500, centers=np.array([(1, 1), (3, 3)]), cluster_std=0.5, random_state=498)
    blobs = np.asarray(blobs[0], dtype=np.float64)
    blobs = blobs - np.min(blobs)
    blobs = blobs / np.max(blobs)

    """dataeval.bias.coverage.coverage"""
    doctest_namespace["embeddings"] = blobs


@pytest.fixture(autouse=True, scope="session")
def doctest_metrics_stats(doctest_namespace: dict[str, Any]) -> None:
    # Create 8 images with consistent pixel values (to avoid image-level outliers)
    # but with varying bbox content to create target-level outliers
    rng = np.random.default_rng(42)
    images = []

    for i in range(8):
        # Create base image with normal pixel values (mean ~128, std ~30)
        img = rng.normal(128, 30, (3, 128, 128)).astype(np.int32)
        img = np.clip(img, 0, 255)
        images.append(img)

    # Make images 2 and 4 have different dimensions for dimension outliers
    images[2] = np.resize(images[2].mean(axis=0), (1, 96, 128))
    images[4] = np.resize(images[4].mean(axis=0), (1, 96, 64))

    bboxes = [
        [[5, 21, 24, 43], [7, 4, 17, 21]],  # Image 0: 2 normal bboxes
        [[12, 23, 28, 24]],  # Image 1: 1 normal bbox
        [[13, 9, 29, 23], [17, 7, 39, 20], [2, 14, 9, 26]],  # Image 2: 3 bboxes
        [[18, 14, 28, 29]],  # Image 3: 1 bbox
        [[21, 18, 44, 27], [15, 13, 28, 23]],  # Image 4: 2 bboxes
        [[13, 2, 23, 14]],  # Image 5: 1 bbox
        [[4, 16, 8, 20], [16, 14, 25, 29]],  # Image 6: 2 bboxes
        [[1, 22, 13, 45], [12, 20, 27, 21], [16, 22, 39, 28]],  # Image 7: 3 bboxes
    ]

    # Create target-level outliers by modifying bbox regions
    # Make some bboxes very bright (outliers), some very dark (outliers), rest normal
    for img_idx, bbox_list in enumerate(bboxes):
        for bbox_idx, bbox in enumerate(bbox_list):
            y1, x1, y2, x2 = bbox
            if img_idx == 0 and bbox_idx == 0:
                # Very bright bbox - target outlier
                images[img_idx][:, x1:x2, y1:y2] = 250
            elif img_idx == 2 and bbox_idx == 1:
                # Very dark bbox - target outlier
                images[img_idx][:, x1:x2, y1:y2] = 5
            elif img_idx == 5 and bbox_idx == 0:
                # Very bright bbox - target outlier
                images[img_idx][:, x1:x2, y1:y2] = 245
            elif img_idx == 7 and bbox_idx == 2:
                # Very dark bbox - target outlier
                images[img_idx][:, x1:x2, y1:y2] = 10
            # Other bboxes remain with normal pixel values from base image

    images = list(images)

    rng = np.random.default_rng(4)
    label_array = rng.choice(5, 50)
    labels = []
    for i, boxes in enumerate(bboxes):
        num_labels = len(boxes)
        selected_labels = label_array[5 * i : 5 * i + num_labels].tolist()
        labels.append(selected_labels)

    classes = ["horse", "cow", "sheep", "pig", "chicken"]

    """dataeval.core.calculate"""

    index2label = dict(enumerate(classes))
    obj_det_dataset_mm = MagicMock(
        _labels=labels,
        _bboxes=bboxes,
        _images=images,
        _classes=classes,
        _index2label=index2label,
        _metadata=None,
        _id=f"{len(images)}_image_{len(index2label)}_class_od_dataset",
    )
    obj_det_dataset_mm.__len__.return_value = len(obj_det_dataset_mm._images)
    obj_det_dataset_mm.metadata = {"id": obj_det_dataset_mm._id, "index2label": obj_det_dataset_mm._index2label}

    obj_det_dataset_getitem_side_effect = [
        (
            obj_det_dataset_mm._images[idx],
            get_object_detection_target(idx, obj_det_dataset_mm),
            {"id": idx},
        )
        for idx in range(len(obj_det_dataset_mm._images))
    ]

    def _mock_getitem(key: int) -> tuple[Any, MagicMock, dict[str, Any]] | None:
        return obj_det_dataset_getitem_side_effect[key]

    obj_det_dataset_mm.__getitem__.side_effect = _mock_getitem
    doctest_namespace["dataset"] = obj_det_dataset_mm
    doctest_namespace["images"] = images
    doctest_namespace["boxes"] = bboxes


@pytest.fixture(autouse=True, scope="session")
def doctest_workflows_sufficiency(doctest_namespace: dict[str, Any]) -> None:
    model = MagicMock()
    train_ds = MagicMock()
    train_ds.__len__.return_value = 100
    test_ds = MagicMock()
    test_ds.__len__.return_value = 10

    class EStrategy:
        def __init__(self, batch_size: int = 16) -> None:
            self.batch_size = batch_size

        def evaluate(self, model, dataset) -> dict:  # noqa
            return {"test": 1.0}

    """dataeval.performance.Sufficiency"""

    doctest_namespace["model"] = model
    doctest_namespace["train_ds"] = train_ds
    doctest_namespace["test_ds"] = test_ds
    doctest_namespace["CustomTrainingStrategy"] = MagicMock()
    doctest_namespace["CustomEvaluationStrategy"] = EStrategy


@pytest.fixture(autouse=True, scope="session")
def doctest_sampledataset(doctest_namespace: dict[str, Any]) -> None:
    class SampleDataset(torch.utils.data.Dataset):
        def __init__(self, size: int, class_count: int) -> None:
            self._size = size
            self._class_count = class_count

        def __getitem__(self, index: int) -> tuple[str, np.ndarray, dict[str, int]]:
            one_hot_label = np.zeros(10)
            one_hot_label[index % self._class_count] = 1
            return f"data_{index}", one_hot_label, {"id": index}

        def __len__(self) -> int:
            return self._size

    doctest_namespace["SampleDataset"] = SampleDataset


class AnnotatedDatasetMock:
    """Mock that properly implements AnnotatedDataset protocol."""

    def __init__(
        self,
        start: int,
        end: int,
        images: list[np.ndarray],
        labels_data: list[int],
        dataset_metadata: dict[str, Any],
        num_classes: int = 3,
    ) -> None:
        self._start = start
        self._end = end
        self._images = images
        self._labels_data = labels_data
        self._metadata = dataset_metadata
        self._num_classes = num_classes

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def __len__(self) -> int:
        return self._end - self._start

    def __getitem__(self, key: int) -> tuple[Any, np.ndarray, dict[str, Any]]:
        if isinstance(key, int):
            actual_idx = self._start + key
            label = self._labels_data[actual_idx]

            # Return one-hot encoded labels for Metadata compatibility
            one_hot = np.zeros(self._num_classes, dtype=np.float32)
            one_hot[label] = 1.0
            return (torch.from_numpy(self._images[actual_idx]), one_hot, {"id": actual_idx})
        raise TypeError(f"indices must be integers, not {type(key).__name__}")


class SliceableDataset:
    """Dataset with slicing support for Prioritize doctests."""

    def __init__(
        self,
        images: list[np.ndarray],
        labels_data: list[int],
        dataset_metadata: dict[str, Any],
        num_classes: int = 3,
    ) -> None:
        self._images = images
        self._labels_data = labels_data
        self._dataset_metadata = dataset_metadata
        self._num_classes = num_classes
        self._data = AnnotatedDatasetMock(0, len(images), images, labels_data, dataset_metadata, num_classes)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._data.metadata

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: int | slice) -> Any:
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop or len(self._images)
            return AnnotatedDatasetMock(
                start,
                stop,
                self._images,
                self._labels_data,
                self._dataset_metadata,
                self._num_classes,
            )
        return self._data[key]


@pytest.fixture(autouse=True, scope="session")
def doctest_quality_prioritize(doctest_namespace: dict[str, Any]) -> None:
    """Create dataset fixture for Prioritize doctests."""
    # Create a simple classification dataset with consistent image dimensions
    rng = np.random.default_rng(42)
    images = [rng.random((3, 32, 32)).astype(np.float32) for _ in range(100)]
    labels_data = list(rng.choice([0, 1, 2], size=100))
    dataset_metadata = {"id": "prioritize_doctest_dataset", "index2label": {0: "class_0", 1: "class_1", 2: "class_2"}}
    num_classes = 3

    # Create datasets with appropriate parameters
    unlabeled_data_mm = SliceableDataset(images, labels_data, dataset_metadata, num_classes)
    labeled_data_mm = AnnotatedDatasetMock(0, 50, images, labels_data, dataset_metadata, num_classes)
    reference_data_mm = AnnotatedDatasetMock(50, 100, images, labels_data, dataset_metadata, num_classes)

    # Create labels array for class_balanced examples
    class_labels = np.array(labels_data[:100], dtype=np.intp)

    doctest_namespace["unlabeled_data"] = unlabeled_data_mm
    doctest_namespace["labeled_data"] = labeled_data_mm
    doctest_namespace["reference_data"] = reference_data_mm
    doctest_namespace["class_labels"] = class_labels
    doctest_namespace["prioritizer"] = None  # Will be set in examples
    doctest_namespace["model"] = torch.nn.Flatten()  # Simple model for examples


class ExampleDataset:
    """Example annotated dataset for doctests and examples.

    This class provides a simple, reusable dataset that implements the
    AnnotatedDataset protocol. Always returns images in CHW format (channels, height, width).

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    image_shape : tuple[int, int, int], default (3, 32, 32)
        Shape of images as (channels, height, width).
    n_classes : int, default 10
        Number of classes for classification.
    seed : int, default 42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int,
        image_shape: tuple[int, int, int] = (3, 32, 32),
        n_classes: int = 10,
        seed: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.image_shape = image_shape
        self.n_classes = n_classes
        self.metadata = {"id": f"example_dataset_{n_samples}"}

        # Pre-generate all data for reproducibility (always 3D CHW format)
        np.random.seed(seed)
        self.images = [np.random.randn(*image_shape).astype(np.float32) for _ in range(n_samples)]
        self.targets = [np.eye(n_classes)[i % n_classes] for i in range(n_samples)]
        self.metadatas = [{"id": i, "brightness": 0.5 + 0.01 * i, "contrast": 0.5 + 0.01 * i} for i in range(n_samples)]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int | slice) -> tuple[Any, Any, dict[str, Any]] | Self:
        if isinstance(idx, slice):
            # Return a new dataset with sliced data
            start, stop, step = idx.indices(self.n_samples)
            sliced = self.__class__.__new__(self.__class__)
            sliced.n_samples = len(range(start, stop, step))
            sliced.image_shape = self.image_shape
            sliced.n_classes = self.n_classes
            sliced.metadata = self.metadata
            sliced.images = self.images[idx]
            sliced.targets = self.targets[idx]
            sliced.metadatas = self.metadatas[idx]
            return sliced
        return self.images[idx], self.targets[idx], self.metadatas[idx]


@pytest.fixture(autouse=True, scope="session")
def doctest_metadata_object(doctest_namespace: dict[str, Any]) -> None:
    """Create metadata object and od_dataset for Metadata doctests."""
    # Create a simple OD dataset with metadata for testing
    # 3 images with varying numbers of detections
    images = [
        np.ones((64, 64), dtype=np.uint8) * 100,  # Image 0
        np.ones((64, 64), dtype=np.uint8) * 150,  # Image 1
        np.ones((64, 64), dtype=np.uint8) * 150,  # Image 2
    ]

    # Labels for each image (per detection)
    labels = [
        [0, 1],  # Image 0: 2 detections
        [1, 2, 0],  # Image 1: 3 detections
        [],  # Image 2: no detections
    ]

    # Bounding boxes for each detection
    bboxes = [
        [[10, 10, 20, 20], [30, 30, 40, 40]],  # Image 0
        [[5, 5, 15, 15], [25, 25, 35, 35], [45, 45, 55, 55]],  # Image 1
        [],  # Image 2
    ]

    # Image-level metadata (varies per image)
    image_metadata = [
        {"temp": 72.5, "time": "morning", "loc": "urban"},
        {"temp": 65.3, "time": "afternoon", "loc": "rural"},
        {"temp": 68.1, "time": "evening", "loc": "suburban"},
    ]

    classes = ["car", "person", "bike"]

    # Create mock OD dataset
    od_dataset_mm = MagicMock()
    od_dataset_mm._labels = labels
    od_dataset_mm._bboxes = bboxes
    od_dataset_mm._images = images
    od_dataset_mm._classes = classes
    od_dataset_mm._metadata = image_metadata

    index2label = dict(enumerate(classes))
    od_dataset_mm.__len__.return_value = len(images)
    od_dataset_mm.metadata = {"index2label": index2label}

    def get_od_target(idx: int) -> MagicMock:
        target_mm = MagicMock()
        target_labels = labels[idx]
        target_bboxes = bboxes[idx]

        # Create one-hot scores
        scores = [[0.0] * len(classes) for _ in target_labels]
        for i, label in enumerate(target_labels):
            scores[i][label] = 1.0

        target_mm.labels = np.array(target_labels, dtype=np.intp)
        target_mm.boxes = np.array(target_bboxes, dtype=np.float32)
        target_mm.scores = np.array(scores, dtype=np.float32)
        return target_mm

    def getitem_side_effect(idx: int) -> tuple[Any, MagicMock, dict[str, Any]]:
        return (images[idx], get_od_target(idx), image_metadata[idx])

    od_dataset_mm.__getitem__.side_effect = getitem_side_effect

    # Create Metadata object
    metadata_obj = Metadata(od_dataset_mm)

    # Add to doctest namespace
    doctest_namespace["od_dataset"] = od_dataset_mm
    doctest_namespace["metadata"] = metadata_obj


@pytest.fixture(autouse=True, scope="session")
def doctest_embeddings_model(doctest_namespace: dict[str, Any]) -> None:
    """Create a simple model and dataset for Embeddings doctests."""
    # Simple model that takes image input and outputs embeddings
    my_model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((4, 4)),  # Reduce to fixed 4x4
        torch.nn.Flatten(),
        torch.nn.LazyLinear(64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
    )
    # Initialize lazy modules with a dummy forward pass
    dummy_input = torch.randn(1, 3, 32, 32)
    my_model(dummy_input)

    # Create a dataset with consistent image sizes for embeddings examples
    embeddings_dataset = ExampleDataset(n_samples=20, image_shape=(3, 32, 32), n_classes=5, seed=42)

    doctest_namespace["my_model"] = my_model
    doctest_namespace["embeddings_dataset"] = embeddings_dataset
    doctest_namespace["train_dataset"] = embeddings_dataset
    doctest_namespace["test_dataset"] = embeddings_dataset
