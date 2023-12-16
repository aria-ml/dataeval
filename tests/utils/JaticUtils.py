import maite.protocols as pr
import numpy as np


def check_jatic_classification(dataset):
    """From JATIC toolbox notebook"""
    # Data has an image
    assert pr.is_typed_dict(dataset[0], pr.HasDataImage)
    # Data has a label
    assert pr.is_typed_dict(dataset[0], pr.HasDataLabel)
    # Data supports image classification
    assert pr.is_typed_dict(dataset[0], pr.SupportsImageClassification)


def check_jatic_object_detection(dataset):
    # Data has an image
    assert pr.is_typed_dict(dataset[0], pr.HasDataImage)
    # Data has object data (label, boxes)
    assert pr.is_typed_dict(dataset[0], pr.HasDataObjects)
    # Data supports object detection
    assert pr.is_typed_dict(dataset[0], pr.SupportsObjectDetection)


classification_data: pr.SupportsImageClassification = {
    "image": np.ones((3, 28, 28)),
    "label": np.array([1]),
}

object_detection_data: pr.SupportsObjectDetection = {
    "image": np.ones((3, 28, 28)),
    "objects": {
        "labels": np.array([1]),
        "boxes": np.array([0.0, 0.0, 1.0, 1.0]),
    },
}


class MockJaticImageClassificationDataset:
    def __init__(self):
        self.data = classification_data

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> pr.SupportsImageClassification:
        return self.data


class MockJaticObjectDetectionDataset:
    def __init__(self):
        self.data = object_detection_data

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index) -> pr.SupportsObjectDetection:
        return self.data


class JaticImageClassificationDataset:
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self._images: np.ndarray = images
        self._labels: np.ndarray = labels

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> pr.SupportsImageClassification:
        output: pr.SupportsImageClassification = {
            "image": self._images[index],
            "label": self._labels[index],
        }
        return output
