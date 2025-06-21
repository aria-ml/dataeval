from __future__ import annotations

__all__ = []

from collections.abc import Sequence, Sized
from typing import Any, Literal

from dataeval.config import EPSILON
from dataeval.typing import Array, ObjectDetectionTarget
from dataeval.utils._array import as_numpy


class ValidationMessages:
    DATASET_SIZED = "Dataset must be sized."
    DATASET_INDEXABLE = "Dataset must be indexable."
    DATASET_NONEMPTY = "Dataset must be non-empty."
    DATASET_METADATA = "Dataset must have a 'metadata' attribute."
    DATASET_METADATA_TYPE = "Dataset metadata must be a dictionary."
    DATASET_METADATA_FORMAT = "Dataset metadata must contain an 'id' key."
    DATUM_TYPE = "Dataset datum must be a tuple."
    DATUM_FORMAT = "Dataset datum must contain 3 elements: image, target, metadata."
    DATUM_IMAGE_TYPE = "Images must be 3-dimensional arrays."
    DATUM_IMAGE_FORMAT = "Images must be in CHW format."
    DATUM_TARGET_IC_TYPE = "ImageClassificationDataset targets must be one-dimensional arrays."
    DATUM_TARGET_IC_FORMAT = "ImageClassificationDataset targets must be one-hot encoded or pseudo-probabilities."
    DATUM_TARGET_OD_TYPE = "ObjectDetectionDataset targets must be have 'boxes', 'labels' and 'scores'."
    DATUM_TARGET_OD_LABELS_TYPE = "ObjectDetectionTarget labels must be one-dimensional (N,) arrays."
    DATUM_TARGET_OD_BOXES_TYPE = "ObjectDetectionTarget boxes must be two-dimensional (N, 4) arrays in xxyy format."
    DATUM_TARGET_OD_SCORES_TYPE = "ObjectDetectionTarget scores must be one (N,) or two-dimensional (N, M) arrays."
    DATUM_TARGET_TYPE = "Target is not a valid ImageClassification or ObjectDetection target type."
    DATUM_METADATA_TYPE = "Datum metadata must be a dictionary."
    DATUM_METADATA_FORMAT = "Datum metadata must contain an 'id' key."


def _validate_dataset_type(dataset: Any) -> list[str]:
    issues = []
    is_sized = isinstance(dataset, Sized)
    is_indexable = hasattr(dataset, "__getitem__")
    if not is_sized:
        issues.append(ValidationMessages.DATASET_SIZED)
    if not is_indexable:
        issues.append(ValidationMessages.DATASET_INDEXABLE)
    if is_sized and len(dataset) == 0:
        issues.append(ValidationMessages.DATASET_NONEMPTY)
    return issues


def _validate_dataset_metadata(dataset: Any) -> list[str]:
    issues = []
    if not hasattr(dataset, "metadata"):
        issues.append(ValidationMessages.DATASET_METADATA)
    metadata = getattr(dataset, "metadata", None)
    if not isinstance(metadata, dict):
        issues.append(ValidationMessages.DATASET_METADATA_TYPE)
    if not isinstance(metadata, dict) or "id" not in metadata:
        issues.append(ValidationMessages.DATASET_METADATA_FORMAT)
    return issues


def _validate_datum_type(datum: Any) -> list[str]:
    issues = []
    if not isinstance(datum, tuple):
        issues.append(ValidationMessages.DATUM_TYPE)
    if datum is None or isinstance(datum, Sized) and len(datum) != 3:
        issues.append(ValidationMessages.DATUM_FORMAT)
    return issues


def _validate_datum_image(image: Any) -> list[str]:
    issues = []
    if not isinstance(image, Array) or len(image.shape) != 3:
        issues.append(ValidationMessages.DATUM_IMAGE_TYPE)
    if (
        not isinstance(image, Array)
        or len(image.shape) == 3
        and (image.shape[0] > image.shape[1] or image.shape[0] > image.shape[2])
    ):
        issues.append(ValidationMessages.DATUM_IMAGE_FORMAT)
    return issues


def _validate_datum_target_ic(target: Any) -> list[str]:
    issues = []
    if not isinstance(target, Array) or len(target.shape) != 1:
        issues.append(ValidationMessages.DATUM_TARGET_IC_TYPE)
    if target is None or sum(target) > 1 + EPSILON or sum(target) < 1 - EPSILON:
        issues.append(ValidationMessages.DATUM_TARGET_IC_FORMAT)
    return issues


def _validate_datum_target_od(target: Any) -> list[str]:
    issues = []
    if not isinstance(target, ObjectDetectionTarget):
        issues.append(ValidationMessages.DATUM_TARGET_OD_TYPE)
    od_target: ObjectDetectionTarget | None = target if isinstance(target, ObjectDetectionTarget) else None
    if od_target is None or len(as_numpy(od_target.labels).shape) != 1:
        issues.append(ValidationMessages.DATUM_TARGET_OD_LABELS_TYPE)
    if (
        od_target is None
        or len(as_numpy(od_target.boxes).shape) != 2
        or (len(as_numpy(od_target.boxes).shape) == 2 and as_numpy(od_target.boxes).shape[1] != 4)
    ):
        issues.append(ValidationMessages.DATUM_TARGET_OD_BOXES_TYPE)
    if od_target is None or len(as_numpy(od_target.scores).shape) not in (1, 2):
        issues.append(ValidationMessages.DATUM_TARGET_OD_SCORES_TYPE)
    return issues


def _detect_target_type(target: Any) -> Literal["ic", "od", "auto"]:
    if isinstance(target, Array):
        return "ic"
    if isinstance(target, ObjectDetectionTarget):
        return "od"
    return "auto"


def _validate_datum_target(target: Any, target_type: Literal["ic", "od", "auto"]) -> list[str]:
    issues = []
    target_type = _detect_target_type(target) if target_type == "auto" else target_type
    if target_type == "ic":
        issues.extend(_validate_datum_target_ic(target))
    elif target_type == "od":
        issues.extend(_validate_datum_target_od(target))
    else:
        issues.append(ValidationMessages.DATUM_TARGET_TYPE)
    return issues


def _validate_datum_metadata(metadata: Any) -> list[str]:
    issues = []
    if metadata is None or not isinstance(metadata, dict):
        issues.append(ValidationMessages.DATUM_METADATA_TYPE)
    if metadata is None or isinstance(metadata, dict) and "id" not in metadata:
        issues.append(ValidationMessages.DATUM_METADATA_FORMAT)
    return issues


def validate_dataset(dataset: Any, dataset_type: Literal["ic", "od", "auto"] = "auto") -> None:
    """
    Validate a dataset for compliance with MAITE protocol.

    Parameters
    ----------
    dataset: Any
        Dataset to validate.
    dataset_type: "ic", "od", or "auto", default "auto"
        Dataset type, if known.

    Raises
    ------
    ValueError
        Raises exception if dataset is invalid with a list of validation issues.
    """
    issues = []
    issues.extend(_validate_dataset_type(dataset))
    datum = None if issues else dataset[0]  # type: ignore
    issues.extend(_validate_dataset_metadata(dataset))
    issues.extend(_validate_datum_type(datum))

    is_seq = isinstance(datum, Sequence)
    datum_len = len(datum) if is_seq else 0
    image = datum[0] if is_seq and datum_len > 0 else None
    target = datum[1] if is_seq and datum_len > 1 else None
    metadata = datum[2] if is_seq and datum_len > 2 else None
    issues.extend(_validate_datum_image(image))
    issues.extend(_validate_datum_target(target, dataset_type))
    issues.extend(_validate_datum_metadata(metadata))

    if issues:
        raise ValueError("Dataset validation issues found:\n - " + "\n - ".join(issues))
