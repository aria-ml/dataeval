"""
This module contains the implementation of Dp Divergence
using the First Nearest Neighbor and Minimum Spanning Tree algorithms
"""
from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
from torch import nn

from daml._internal.metrics.aria.base import _BaseMetric
from daml._internal.metrics.outputs import DivergenceOutput

from .utils import compute_neighbors, minimum_spanning_tree


class _DpDivergence(_BaseMetric):
    """
    For more information about this divergence, its formal definition,
    and its associated estimators
    see https://arxiv.org/abs/1412.6534.
    """

    def __init__(
        self,
        images_a: np.ndarray,
        images_b: np.ndarray,
        encode: bool = False,
        model: Optional[nn.Module] = None,
        fit: Optional[bool] = None,
        epochs: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Constructor method"""
        super().__init__(
            images_a,
            encode,
            model=model,
            fit=fit,
            epochs=epochs,
            device=device,
        )
        self.images_a = images_a
        self.images_b = images_b

    @abstractmethod
    def calculate_errors(self, data: np.ndarray, labels: np.ndarray) -> int:
        """Abstract method for the implementation of divergence calculation"""

    def _encode_and_vstack(self) -> np.ndarray:
        emb_a = self._encode(self.images_a)
        emb_b = self._encode(self.images_b)
        if self.encode:
            emb_a = emb_a.flatten()
            emb_b = emb_b.flatten()
        return np.vstack((emb_a, emb_b))

    def _evaluate(self) -> DivergenceOutput:
        """
        Calculates the divergence and any errors between the datasets

        Returns
        -------
        DivergenceOutput
            Dataclass containing the dp divergence and errors during calculation
        """
        N = self.images_a.shape[0]
        M = self.images_b.shape[0]

        images = self._encode_and_vstack()
        labels = np.vstack([np.zeros([N, 1]), np.ones([M, 1])])

        errors = self.calculate_errors(images, labels)
        dp = 1 - ((M + N) / (2 * M * N)) * errors
        return DivergenceOutput(dpdivergence=dp, error=errors)


class DpDivergenceMST(_DpDivergence):
    """
    A minimum spanning tree implementation of dp divergence

    Warning
    -------
        MST is very slow in this implementation, this is unlike matlab where
        they have comparable speeds
        Overall, MST takes ~25x LONGER!!
        Source of slowdown:
        conversion to and from CSR format adds ~10% of the time diff between
        1nn and scipy mst function the remaining 90%
    """

    # TODO
    # - validate the input algorithm
    # - improve speed for MST, requires a fast mst implementation
    # mst is at least 10x slower than knn approach

    def calculate_errors(self, data: np.ndarray, labels: np.ndarray) -> int:
        """
        Returns the divergence between two datasets using a minimum spanning tree

        Parameters
        ----------
        data : np.ndarray
            Array containing images from two datasets
        labels : np.ndarray
            Array showing which dataset each image belonged to

        Returns
        -------
        int
            Number of edges connecting the two datasets

        Note
        ----
        We add a small constant to the distance matrix to ensure scipy interprets
        the input graph as fully-connected.
        """
        mst = minimum_spanning_tree(data).toarray()
        edgelist = np.transpose(np.nonzero(mst))
        errors = np.sum(labels[edgelist[:, 0]] != labels[edgelist[:, 1]])
        return errors


class DpDivergenceFNN(_DpDivergence):
    """
    A first nearest neighbor implementation of dp divergence
    """

    def calculate_errors(self, data: np.ndarray, labels: np.ndarray) -> int:
        """
        Returns the divergence between two datasets using first nearest neighbor

        Parameters
        ----------
        data : np.ndarray
            Array containing images from two datasets
        labels : np.ndarray
            Array showing which dataset each image belonged to

        Returns
        -------
        int
            Number of edges connecting the two datasets
        """
        nn_indices = compute_neighbors(data, data)
        errors = np.sum(np.abs(labels[nn_indices] - labels))
        return errors
