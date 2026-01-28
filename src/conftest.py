"""Doctest fixtures for dataeval.

This module provides unified, reusable fixtures for all doctests.
The fixtures are designed around a consistent computer vision dataset
with object detection annotations and rich metadata.

Classes: person, car, van, boat, plane
Metadata: time_of_day, weather, angle, location
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from typing_extensions import Self

from dataeval.config import _config, set_batch_size, set_device, set_seed

# Set numpy print option to legacy 1.25 so native numpy types
# are not printed with dtype information.
if np.__version__[0] == "2":
    np.set_printoptions(legacy="1.25", precision=3)  # type: ignore
else:
    np.set_printoptions(precision=3)

# _ = pl.Config.set_tbl_rows(20)

# =============================================================================
# Constants for unified dataset
# =============================================================================

CLASSES = ["person", "car", "boat", "plane"]
INDEX2LABEL = dict(enumerate(CLASSES))
NUM_CLASSES = len(CLASSES)

METADATA_FACTORS = {
    "time_of_day": ["dawn", "day", "dusk", "night"],
    "weather": ["clear", "cloudy", "rainy"],
    "angle": ["overhead", "eye_level", "low_angle"],
    "location": ["urban", "suburban", "rural", "maritime"],
}

# =============================================================================
# Unified image generation
# =============================================================================


def _generate_images(
    n_images: int = 50,
    shape: tuple[int, int, int] = (3, 64, 64),
    seed: int = 42,
    include_duplicates: bool = True,
    include_outliers: bool = True,
) -> list[NDArray[np.float32]]:
    """Generate images with optional duplicates and outliers for testing.

    Parameters
    ----------
    n_images : int
        Number of images to generate.
    shape : tuple
        Shape of images as (channels, height, width).
    seed : int
        Random seed for reproducibility.
    include_duplicates : bool
        If True, include duplicate images at indices 3==20 and 16==37.
    include_outliers : bool
        If True, include outlier images at indices 7, 11, 18, 25.

    Returns
    -------
    list[NDArray[np.float32]]
        List of generated images with values in [0, 1].
    """
    rng = np.random.default_rng(seed)
    images: list[NDArray[np.float32]] = []

    for _ in range(n_images):
        # Create base image with normal distribution, normalized to [0, 1]
        img = rng.normal(0.5, 0.12, shape).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)
        images.append(img)

    if include_duplicates and n_images >= 38:
        # Create exact duplicates for testing duplicate detection
        images[16] = images[37].copy()
        images[3] = images[20].copy()

    if include_outliers and n_images >= 26:
        # Create outlier images (very bright)
        for idx in [7, 11, 18, 25]:
            if idx < n_images:
                images[idx] = np.full(shape, 0.98, dtype=np.float32)

    return images


def _generate_boxes(n_images: int, seed: int = 42) -> list[list[list[int]]]:
    """Generate bounding boxes for each image.

    Returns boxes in [x1, y1, x2, y2] format.
    """
    rng = np.random.default_rng(seed)
    boxes = []

    for _ in range(n_images):
        n_boxes = rng.integers(1, 4)  # 1-3 boxes per image
        img_boxes = []
        for _ in range(n_boxes):
            x1 = rng.integers(5, 30)
            y1 = rng.integers(5, 30)
            w = rng.integers(10, 25)
            h = rng.integers(10, 25)
            img_boxes.append([x1, y1, x1 + w, y1 + h])
        boxes.append(img_boxes)

    return boxes


def _generate_labels(boxes: list[list[list[int]]], n_classes: int = NUM_CLASSES, seed: int = 42) -> list[list[int]]:
    """Generate class labels for each bounding box."""
    rng = np.random.default_rng(seed)
    labels = []
    for img_boxes in boxes:
        img_labels = list(rng.integers(0, n_classes, len(img_boxes)))
        labels.append(img_labels)
    return labels


def _generate_metadata(n_images: int, seed: int = 42) -> list[dict[str, Any]]:
    """Generate per-image metadata with CV-relevant factors."""
    rng = np.random.default_rng(seed)
    metadata = []

    for i in range(n_images):
        meta = {
            "time_of_day": rng.choice(METADATA_FACTORS["time_of_day"]),
            "weather": rng.choice(METADATA_FACTORS["weather"]),
            "angle": rng.choice(METADATA_FACTORS["angle"]),
            "location": rng.choice(METADATA_FACTORS["location"]),
            "id": i,
        }
        metadata.append(meta)

    return metadata


# =============================================================================
# Mock metadata for bias/balance metrics
# =============================================================================


@dataclass
class MockMetadata:
    """Simple Metadata implementation for doctests."""

    class_labels: NDArray[np.intp]
    factor_data: NDArray[np.int64]
    factor_names: Sequence[str]
    is_discrete: Sequence[bool]
    index2label: Mapping[int, str]


# =============================================================================
# Dataset classes
# =============================================================================


class ImageDataset:
    """Simple image classification dataset for doctests.

    Supports slicing and implements the AnnotatedDataset protocol.
    """

    def __init__(
        self,
        images: list[NDArray[Any]],
        class_labels: list[int],
        image_metadata: list[dict[str, Any]],
        n_classes: int = NUM_CLASSES,
        dataset_id: str = "image_dataset",
    ) -> None:
        self._images = images
        self._class_labels = class_labels
        self._image_metadata = image_metadata
        self._n_classes = n_classes
        self._metadata = {"id": dataset_id, "index2label": INDEX2LABEL}

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, key: int | slice) -> tuple[Any, NDArray[np.float32], dict[str, Any]] | Self:
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self._images))
            indices = range(start, stop, step)
            return ImageDataset(
                images=[self._images[i] for i in indices],
                class_labels=[self._class_labels[i] for i in indices],
                image_metadata=[self._image_metadata[i] for i in indices],
                n_classes=self._n_classes,
                dataset_id=self._metadata["id"],
            )  # type: ignore
        # Return one-hot encoded labels
        one_hot = np.zeros(self._n_classes, dtype=np.float32)
        one_hot[self._class_labels[key]] = 1.0
        return torch.from_numpy(self._images[key]), one_hot, self._image_metadata[key]


class ObjectDetectionDataset:
    """Object detection dataset for doctests.

    Returns (image, target, metadata) tuples where target has labels, boxes, scores.
    """

    def __init__(
        self,
        images: list[NDArray[Any]],
        labels: list[list[int]],
        boxes: list[list[list[int]]],
        image_metadata: list[dict[str, Any]],
        classes: list[str] = CLASSES,
        dataset_id: str = "od_dataset",
    ) -> None:
        self._images = images
        self._labels = labels
        self._boxes = boxes
        self._image_metadata = image_metadata
        self._classes = classes
        self._index2label = dict(enumerate(classes))
        self._metadata = {"id": dataset_id, "index2label": self._index2label}

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def __len__(self) -> int:
        return len(self._images)

    def _get_target(self, idx: int) -> MagicMock:
        """Create target object with labels, boxes, and scores."""
        target = MagicMock()
        target.labels = np.array(self._labels[idx], dtype=np.intp)
        target.boxes = np.array(self._boxes[idx], dtype=np.float32)

        # Create one-hot scores
        n_detections = len(self._labels[idx])
        scores = np.zeros((n_detections, len(self._classes)), dtype=np.float32)
        for i, label in enumerate(self._labels[idx]):
            scores[i, label] = 1.0
        target.scores = scores

        return target

    def __getitem__(self, key: int | slice) -> tuple[Any, MagicMock, dict[str, Any]] | Self:
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self._images))
            indices = range(start, stop, step)
            return ObjectDetectionDataset(
                images=[self._images[i] for i in indices],
                labels=[self._labels[i] for i in indices],
                boxes=[self._boxes[i] for i in indices],
                image_metadata=[self._image_metadata[i] for i in indices],
                classes=self._classes,
                dataset_id=self._metadata["id"],
            )  # type: ignore
        return self._images[key], self._get_target(key), self._image_metadata[key]

    def __str__(self) -> str:
        return f"ObjectDetectionDataset(n_images={len(self)}, classes={self._classes})"

    def __repr__(self) -> str:
        return self.__str__()


# =============================================================================
# Model factory
# =============================================================================


def _create_model() -> torch.nn.Module:
    """Create a simple embedding model for doctests."""
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((4, 4)),
        torch.nn.Flatten(),
        torch.nn.LazyLinear(64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
    )
    # Initialize lazy modules
    dummy_input = torch.randn(1, 3, 64, 64)
    model(dummy_input)
    return model


# =============================================================================
# Pytest fixtures
# =============================================================================


@pytest.fixture(autouse=True, scope="function")
def reset_config() -> Any:
    """Reset config state before each test for deterministic behavior."""
    # Save original config state
    old_device = _config.device
    old_batch_size = _config.batch_size
    old_seed = _config.seed

    # Set defaults for tests
    set_seed(0, all_generators=True)
    set_device("cpu")
    set_batch_size(32)

    yield

    # Restore original state
    _config.device = old_device
    _config.batch_size = old_batch_size
    _config.seed = old_seed


@pytest.fixture(autouse=True, scope="session")
def doctest_unified_fixtures(doctest_namespace: dict[str, Any]) -> None:
    """Create all unified fixtures for doctests."""
    # -------------------------------------------------------------------------
    # Generate base data
    # -------------------------------------------------------------------------
    n_images = 50
    images = _generate_images(n_images=n_images, shape=(3, 64, 64), seed=42)
    boxes = _generate_boxes(n_images=n_images, seed=42)
    labels = _generate_labels(boxes, seed=42)
    image_metadata = _generate_metadata(n_images=n_images, seed=42)

    # Class labels for each image (use first detection's label or random)
    rng = np.random.default_rng(42)
    class_labels = [img_labels[0] if img_labels else int(rng.integers(0, NUM_CLASSES)) for img_labels in labels]

    # -------------------------------------------------------------------------
    # Core fixtures
    # -------------------------------------------------------------------------

    # Raw images array (for Duplicates, Outliers, calculate, etc.)
    doctest_namespace["images"] = images

    # Bounding boxes
    doctest_namespace["boxes"] = boxes

    # Object detection dataset
    dataset = ObjectDetectionDataset(
        images=images,
        labels=labels,
        boxes=boxes,
        image_metadata=image_metadata,
        classes=CLASSES,
        dataset_id="doctest_od_dataset",
    )
    doctest_namespace["dataset"] = dataset

    # Also expose as od_dataset for backwards compatibility
    doctest_namespace["od_dataset"] = dataset

    # -------------------------------------------------------------------------
    # Train/test splits
    # -------------------------------------------------------------------------
    train_ds = ImageDataset(
        images=images[:40],
        class_labels=class_labels[:40],
        image_metadata=image_metadata[:40],
        n_classes=NUM_CLASSES,
        dataset_id="train_dataset",
    )
    test_ds = ImageDataset(
        images=images[40:],
        class_labels=class_labels[40:],
        image_metadata=image_metadata[40:],
        n_classes=NUM_CLASSES,
        dataset_id="test_dataset",
    )

    doctest_namespace["train_ds"] = train_ds
    doctest_namespace["test_ds"] = test_ds
    doctest_namespace["train_dataset"] = train_ds
    doctest_namespace["test_dataset"] = test_ds

    # -------------------------------------------------------------------------
    # Stats fixtures for Outliers.from_stats
    # -------------------------------------------------------------------------
    from dataeval.core import calculate
    from dataeval.flags import ImageStats

    stats1 = calculate(images[:25], stats=ImageStats.PIXEL)
    stats2 = calculate(images[25:], stats=ImageStats.PIXEL)
    doctest_namespace["stats1"] = stats1
    doctest_namespace["stats2"] = stats2

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = _create_model()
    doctest_namespace["model"] = model
    doctest_namespace["my_model"] = model

    # -------------------------------------------------------------------------
    # Prioritize fixtures
    # -------------------------------------------------------------------------
    unlabeled_data = ImageDataset(
        images=images,
        class_labels=class_labels,
        image_metadata=image_metadata,
        n_classes=NUM_CLASSES,
        dataset_id="unlabeled_dataset",
    )
    labeled_data = ImageDataset(
        images=images[:25],
        class_labels=class_labels[:25],
        image_metadata=image_metadata[:25],
        n_classes=NUM_CLASSES,
        dataset_id="labeled_dataset",
    )
    reference_data = ImageDataset(
        images=images[25:],
        class_labels=class_labels[25:],
        image_metadata=image_metadata[25:],
        n_classes=NUM_CLASSES,
        dataset_id="reference_dataset",
    )

    doctest_namespace["unlabeled_data"] = unlabeled_data
    doctest_namespace["labeled_data"] = labeled_data
    doctest_namespace["reference_data"] = reference_data
    doctest_namespace["class_labels"] = np.array(class_labels, dtype=np.intp)

    # -------------------------------------------------------------------------
    # Sufficiency fixtures (mock train/test with __len__)
    # -------------------------------------------------------------------------
    sufficiency_train = MagicMock()
    sufficiency_train.__len__.return_value = 100
    sufficiency_test = MagicMock()
    sufficiency_test.__len__.return_value = 10

    class EvaluationStrategy:
        def __init__(self, batch_size: int = 16) -> None:
            self.batch_size = batch_size

        def evaluate(self, model: Any, dataset: Any) -> dict[str, float]:
            return {"test": 1.0}

    class TrainingStrategy:
        def __init__(self, learning_rate: float = 0.5, epochs: int = 10) -> None:
            pass

        def train(self, model: torch.nn.Module, dataset: Any, indices: Sequence[int]) -> None:
            pass

    doctest_namespace["CustomTrainingStrategy"] = TrainingStrategy
    doctest_namespace["CustomEvaluationStrategy"] = EvaluationStrategy

    # -------------------------------------------------------------------------
    # Utility functions and classes
    # -------------------------------------------------------------------------
    doctest_namespace["np"] = np
