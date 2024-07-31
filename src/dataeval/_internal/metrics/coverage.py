from typing import Literal, Tuple

import numpy as np

from dataeval._internal.functional.coverage import coverage


class Coverage:
    """
    Class for evaluating coverage and identifying images/samples that are in undercovered regions.

    This implementation is based on https://dl.acm.org/doi/abs/10.1145/3448016.3457315.

    Parameters
    ----------
    embeddings : np.ndarray
        n x p array of image embeddings from the dataset.
    radius_type : Literal["adaptive", "naive"], default "adaptive"
        The function used to determine radius.
    k: int, default 20
        Number of observations required in order to be covered.
    percent: np.float64, default np.float(0.01)
        Percent of observations to be considered uncovered. Only applies to adaptive radius.

    Note
    ----
    Embeddings should be on the unit interval.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        radius_type: Literal["adaptive", "naive"] = "adaptive",
        k: int = 20,
        percent: np.float64 = np.float64(0.01),
    ):
        self.embeddings = embeddings
        self.radius_type = radius_type
        self.k = k
        self.percent = percent

    def evaluate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identifies images which come from under sampled regions.
        Also returns the corresponding radius in embedding space within
        which we expect to see a particular number of images.

        Returns
        -------
        np.ndarray
            Array of uncovered indices
        np.ndarray
            Array of critical value radii

        Raises
        ------
        ValueError
            If length of embeddings is less than or equal to k
        ValueError
            If radius_type is unknown
        """

        if self.radius_type not in ["adaptive", "naive"]:
            raise ValueError(f"Invalid radius type {self.radius_type}")

        return coverage(self.embeddings, self.k, self.radius_type, self.percent)
