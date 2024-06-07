import numpy as np
import torch
from torchmetrics import Metric

from daml.metrics.drift import KSDrift


class DriftMetric(Metric):
    def __init__(self, x_ref):
        if isinstance(x_ref, torch.Tensor):
            x_ref = x_ref.detach().cpu().numpy()
        self.x_ref: np.ndarray = x_ref
        self._drift = KSDrift(x_ref)
        self.add_state("images", default=[])

    def update(self, images: torch.Tensor):
        self.images.append(images)

    def compute(self):
        images: np.ndarray = self.images.detach().cpu().numpy()
        results = self._drift.predict(images)

        return results
