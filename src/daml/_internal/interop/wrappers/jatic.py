import jatic_toolbox.protocols as pr
import numpy as np

from daml._internal.datasets.datasets import DamlDataset


class JaticClassificationDatasetWrapper(DamlDataset):
    "Reformats Jatic data into the DAML format"

    def __init__(self, dataset: pr.VisionDataset):
        self._split_data(dataset)

    def _split_data(self, dataset: pr.VisionDataset):
        X_list = []
        y_list = []
        for data in dataset:
            X_list.append(np.array(data["image"]))
            y_list.append(np.array(data["label"]))
        X = np.array(X_list)
        y = np.array(y_list)
        self._set_data(X, y)

    def __getitem__(self, index: int) -> pr.SupportsImageClassification:
        return {"image": self._images[index], "label": self._labels[index]}
