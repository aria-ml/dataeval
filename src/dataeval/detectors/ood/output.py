from __future__ import annotations

__all__ = []

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from dataeval.output import Output


@dataclass(frozen=True)
class OODOutput(Output):
    """
    Output class for predictions from out-of-distribution detectors.

    Attributes
    ----------
    is_ood : NDArray
        Array of images that are detected as :term:Out-of-Distribution (OOD)`
    instance_score : NDArray
        Instance score of the evaluated dataset
    feature_score : NDArray | None
        Feature score, if available, of the evaluated dataset
    """

    is_ood: NDArray[np.bool_]
    instance_score: NDArray[np.float32]
    feature_score: NDArray[np.float32] | None


@dataclass(frozen=True)
class OODScoreOutput(Output):
    """
    Output class for instance and feature scores from out-of-distribution detectors.

    Parameters
    ----------
    instance_score : NDArray
        Instance score of the evaluated dataset.
    feature_score : NDArray | None, default None
        Feature score, if available, of the evaluated dataset.
    """

    instance_score: NDArray[np.float32]
    feature_score: NDArray[np.float32] | None = None

    def get(self, ood_type: Literal["instance", "feature"]) -> NDArray[np.float32]:
        """
        Returns either the instance or feature score

        Parameters
        ----------
        ood_type : "instance" | "feature"

        Returns
        -------
        NDArray
            Either the instance or feature score based on input selection
        """
        return self.instance_score if ood_type == "instance" or self.feature_score is None else self.feature_score
