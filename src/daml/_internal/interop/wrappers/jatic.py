import maite.protocols as pr
import numpy as np

from daml._internal.datasets.datasets import DamlDataset


class JaticClassificationDatasetWrapper(DamlDataset):
    "Reformats Jatic data into the DAML format"

    def __init__(self, dataset: pr.VisionDataset):
        images, labels = self._split_data(dataset)
        super().__init__(images, labels)

    def _split_data(self, dataset: pr.VisionDataset):
        images = []
        labels = []
        for idx in range(len(dataset)):
            images.append(np.array(dataset[idx]["image"]))
            labels.append(np.array(dataset[idx]["label"]))

        images = np.array(images)
        labels = np.array(labels)

        return images, labels


class JaticObjectDetectionWrapper(DamlDataset):
    "Reformats Jatic data into the DAML format"

    def __init__(self, dataset: pr.ObjectDetectionDataset):
        images, labels, boxes = self._split_data(dataset)
        super().__init__(images=images, labels=labels, boxes=boxes)

    def _split_data(self, dataset: pr.ObjectDetectionDataset):
        images = []
        labels = []
        boxes = []
        for idx in range(len(dataset)):
            objects: pr.HasDataBoxesLabels = dataset[idx]["objects"]  # type: ignore

            images.append(np.array(dataset[idx]["image"]))
            labels.append(np.array(objects["labels"]))
            boxes.append(np.array(objects["boxes"]))

        images = np.array(images)
        labels = np.array(labels)
        boxes = np.array(boxes)

        return images, labels, boxes
