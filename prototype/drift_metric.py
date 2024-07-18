import numpy as np
import torch
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

from dataeval.detectors import DriftKS


class DriftMetric(Metric):
    def __init__(self, x_ref, **kwargs):
        super().__init__(**kwargs)
        if isinstance(x_ref, torch.Tensor):
            x_ref = x_ref.detach().cpu().numpy()
        self.x_ref: np.ndarray = x_ref
        self._drift = DriftKS(x_ref)
        self.add_state("images", default=[], dist_reduce_fx="cat")

    def update(self, images: torch.Tensor):
        self.images.append(images)

    def compute(self) -> dict:
        images: np.ndarray = dim_zero_cat(self.images).detach().cpu().numpy()
        results = self._drift.predict(images)

        return results
