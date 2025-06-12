from dataclasses import dataclass

import numpy as np
import pytest

from dataeval.typing import Array
from dataeval.utils.data._validate import (
    ValidationMessages,
    _detect_target_type,
    _validate_dataset_metadata,
    _validate_dataset_type,
    _validate_datum_image,
    _validate_datum_metadata,
    _validate_datum_target,
    _validate_datum_target_ic,
    _validate_datum_target_od,
    _validate_datum_type,
    validate_dataset,
)


def make_dataset(data, metadata):
    class MockDataset:
        def __init__(self, data):
            self.data = data

        def __getitem__(self, key):
            return self.data[key]

        def __len__(self):
            return len(self.data)

    dataset = MockDataset(data)

    if metadata is not None:
        setattr(dataset, "metadata", metadata)

    return dataset


@dataclass
class ObjectDetectionTarget:
    labels: Array
    boxes: Array
    scores: Array


@pytest.fixture
def valid_ic_target():
    return np.array([0, 1, 0])


@pytest.fixture
def valid_od_target():
    return ObjectDetectionTarget(
        labels=np.array([1, 2]), boxes=np.array([[10, 20, 30, 40], [50, 60, 70, 80]]), scores=np.array([[0.9], [0.8]])
    )


@pytest.fixture
def valid_image():
    return np.random.rand(3, 100, 100)  # CHW format


@pytest.fixture
def valid_metadata():
    return {"id": "image_001"}


@pytest.fixture
def valid_ic_dataset(valid_image, valid_ic_target, valid_metadata):
    return make_dataset([(valid_image, valid_ic_target, valid_metadata)], valid_metadata)


@pytest.fixture
def valid_od_dataset(valid_image, valid_od_target, valid_metadata):
    return make_dataset([(valid_image, valid_od_target, valid_metadata)], valid_metadata)


class NotSized:
    def __getitem__(self, key):
        return 0


class NotIndexable:
    def __len__(self):
        return 1


def test_validate_dataset_type():
    """Tests for _validate_dataset_type function."""
    assert ValidationMessages.DATASET_SIZED in _validate_dataset_type(NotSized())
    assert ValidationMessages.DATASET_INDEXABLE in _validate_dataset_type(NotIndexable())
    assert ValidationMessages.DATASET_NONEMPTY in _validate_dataset_type([])


def test_validate_dataset_metadata():
    """Tests for _validate_dataset_type function."""
    assert ValidationMessages.DATASET_METADATA in _validate_dataset_metadata([])
    assert ValidationMessages.DATASET_METADATA_TYPE in _validate_dataset_metadata([])
    assert ValidationMessages.DATASET_METADATA_FORMAT in _validate_dataset_metadata([])


def test_validate_datum_type():
    """Tests for _validate_datum_type function."""
    assert _validate_datum_type(("img", "tgt", "meta")) == []
    assert ValidationMessages.DATUM_TYPE in _validate_datum_type(None)
    assert ValidationMessages.DATUM_TYPE in _validate_datum_type([1, 2, 3])
    assert ValidationMessages.DATUM_FORMAT in _validate_datum_type((1, 2))


def test_validate_datum_image(valid_image):
    """Tests for _validate_datum_image function."""
    assert _validate_datum_image(valid_image) == []
    assert ValidationMessages.DATUM_IMAGE_TYPE in _validate_datum_image([1, 2, 3])
    assert ValidationMessages.DATUM_IMAGE_TYPE in _validate_datum_image(np.random.rand(100, 100))
    assert ValidationMessages.DATUM_IMAGE_FORMAT in _validate_datum_image(np.random.rand(100, 100, 3))


def test_detect_target_type(valid_ic_target, valid_od_target):
    """Tests for _detect_target_type function."""
    assert _detect_target_type(valid_ic_target) == "ic"
    assert _detect_target_type(valid_od_target) == "od"
    assert _detect_target_type("some_string") == "auto"
    assert _detect_target_type(123) == "auto"


def test_validate_datum_target_ic(valid_ic_target):
    """Tests for _validate_datum_target_ic function."""
    assert _validate_datum_target_ic(valid_ic_target) == []
    assert _validate_datum_target_ic(np.array([0.5, 0.5])) == []
    assert ValidationMessages.DATUM_TARGET_IC_TYPE in _validate_datum_target_ic([0, 1, 0])
    assert ValidationMessages.DATUM_TARGET_IC_TYPE in _validate_datum_target_ic(np.array([[1]]))
    assert ValidationMessages.DATUM_TARGET_IC_FORMAT in _validate_datum_target_ic(np.array([0.8, 0.8]))


def test_validate_datum_target_od(valid_od_target):
    """Tests for _validate_datum_target_od function."""
    assert _validate_datum_target_od(valid_od_target) == []
    # Test with empty target
    empty_target = ObjectDetectionTarget(np.array([]), np.array([]), np.array([]))
    assert _validate_datum_target_od(empty_target)

    assert ValidationMessages.DATUM_TARGET_OD_TYPE in _validate_datum_target_od("not_a_target")

    bad_labels = ObjectDetectionTarget(np.array([[1]]), valid_od_target.boxes, valid_od_target.scores)
    assert ValidationMessages.DATUM_TARGET_OD_LABELS_TYPE in _validate_datum_target_od(bad_labels)

    bad_boxes_shape = ObjectDetectionTarget(valid_od_target.labels, np.array([1, 2, 3]), valid_od_target.scores)
    assert ValidationMessages.DATUM_TARGET_OD_BOXES_TYPE in _validate_datum_target_od(bad_boxes_shape)

    bad_boxes_format = ObjectDetectionTarget(valid_od_target.labels, np.array([[1, 2, 3]]), valid_od_target.scores)
    assert ValidationMessages.DATUM_TARGET_OD_BOXES_TYPE in _validate_datum_target_od(bad_boxes_format)

    bad_scores = ObjectDetectionTarget(valid_od_target.labels, valid_od_target.boxes, np.array([[[0.9]]]))
    assert ValidationMessages.DATUM_TARGET_OD_SCORES_TYPE in _validate_datum_target_od(bad_scores)


def test_validate_datum_target(valid_ic_target, valid_od_target):
    """Tests for _validate_datum_target function."""
    assert _validate_datum_target(valid_ic_target, "ic") == []
    assert _validate_datum_target(valid_od_target, "od") == []
    assert _validate_datum_target(valid_ic_target, "auto") == []
    assert _validate_datum_target(valid_od_target, "auto") == []

    assert ValidationMessages.DATUM_TARGET_TYPE in _validate_datum_target(None, "auto")
    assert ValidationMessages.DATUM_TARGET_TYPE in _validate_datum_target("invalid", "auto")
    assert ValidationMessages.DATUM_TARGET_IC_TYPE in _validate_datum_target([1], "ic")
    assert ValidationMessages.DATUM_TARGET_OD_TYPE in _validate_datum_target([1], "od")


def test_validate_datum_metadata(valid_metadata):
    """Tests for _validate_datum_metadata function."""
    assert _validate_datum_metadata(valid_metadata) == []
    assert ValidationMessages.DATUM_METADATA_TYPE in _validate_datum_metadata(None)
    assert ValidationMessages.DATUM_METADATA_TYPE in _validate_datum_metadata("not_a_dict")
    assert ValidationMessages.DATUM_METADATA_FORMAT in _validate_datum_metadata({"name": "test"})


def test_validate_dataset_success(valid_ic_dataset, valid_od_dataset):
    """Tests successful validation of full datasets."""
    # pytest.does_not_raise is implicit if no exception occurs
    validate_dataset(valid_ic_dataset, "ic")
    validate_dataset(valid_od_dataset, "od")
    validate_dataset(valid_ic_dataset, "auto")
    validate_dataset(valid_od_dataset, "auto")


def test_validate_dataset_raises_errors(valid_ic_dataset):
    """Tests that validate_dataset raises ValueError for various issues."""
    with pytest.raises(ValueError, match=ValidationMessages.DATASET_NONEMPTY):
        validate_dataset([])

    with pytest.raises(ValueError, match=ValidationMessages.DATUM_TYPE):
        validate_dataset([["img", "tgt", "meta"]])

    with pytest.raises(ValueError, match=ValidationMessages.DATUM_IMAGE_TYPE):
        bad_dataset = [(["not", "an", "image"], valid_ic_dataset[0][1], valid_ic_dataset[0][2])]
        validate_dataset(bad_dataset)

    with pytest.raises(ValueError, match=ValidationMessages.DATUM_TARGET_IC_FORMAT):
        bad_dataset = [(valid_ic_dataset[0][0], np.array([0.9, 0.9]), valid_ic_dataset[0][2])]
        validate_dataset(bad_dataset, "ic")

    with pytest.raises(ValueError, match=ValidationMessages.DATASET_METADATA):
        bad_dataset = [(valid_ic_dataset[0][0], valid_ic_dataset[0][1], {"name": "no_id"})]
        validate_dataset(bad_dataset)

    # Test that multiple errors are reported
    with pytest.raises(ValueError) as excinfo:
        bad_dataset = [(["img"], np.array([0.9, 0.9]), {"name": "no_id"})]
        validate_dataset(bad_dataset, "ic")

    assert ValidationMessages.DATASET_METADATA in str(excinfo.value)
    assert ValidationMessages.DATASET_METADATA_FORMAT in str(excinfo.value)
    assert ValidationMessages.DATUM_IMAGE_TYPE in str(excinfo.value)
    assert ValidationMessages.DATUM_IMAGE_FORMAT in str(excinfo.value)
    assert ValidationMessages.DATUM_TARGET_IC_FORMAT in str(excinfo.value)
    assert ValidationMessages.DATUM_METADATA_FORMAT in str(excinfo.value)
