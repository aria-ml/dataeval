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
        for i in range(len(dataset)):
            X_list.append(np.array(dataset[i]["image"]))
            y_list.append(np.array(dataset[i]["label"]))
        X = np.array(X_list)
        y = np.array(y_list)
        self._set_data(X, y)

    def __getitem__(self, index: int) -> pr.SupportsImageClassification:
        return {"image": self._images[index], "label": self._labels[index]}
