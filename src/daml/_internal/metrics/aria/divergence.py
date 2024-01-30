"""
This module contains the implementation of Dp Divergence
using the First Nearest Neighbor and Minimum Spanning Tree algorithms
"""
from abc import abstractmethod
from typing import Optional, Union

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from daml._internal.datasets.datasets import DamlDataset
from daml._internal.metrics.aria.base import _AriaMetric
from daml._internal.metrics.outputs import DivergenceOutput

from .utils import permute_to_numpy, permute_to_torch


class _DpDivergence(_AriaMetric):
    """
    For more information about this divergence, its formal definition,
    and its associated estimators
    see https://arxiv.org/abs/1412.6534.
    """

    def __init__(
        self, encode: bool = False, device: Union[str, torch.device] = "cpu"
    ) -> None:
        """Constructor method"""
        super().__init__(encode, device)

    @abstractmethod
    def calculate_errors(self, data: np.ndarray, labels: np.ndarray) -> int:
        """Abstract method for the implementation of divergence calculation"""

    def create_encoding(self, imgs_1: torch.Tensor, imgs_2: torch.Tensor) -> np.ndarray:
        """Reduces image dimensions using a trained autoencoder and stacks results"""
        if not self._is_trained:
            raise ValueError(
                "Tried to encode data without fitting a model.\
                    Try calling Metric.fit_dataset(dataset) first."
            )
        if self.model is None:
            raise TypeError(
                "Model is None. Try calling Metric.fit_dataset(dataset) first."
            )

        # Pass inputs through model
        tensor_a = self.model.encode(imgs_1).flatten(start_dim=1)
        tensor_b = self.model.encode(imgs_2).flatten(start_dim=1)

        # Combine data into one dataset and return
        return torch.vstack((tensor_a, tensor_b)).detach().cpu().numpy()

    def evaluate(
        self,
        dataset_a: DamlDataset,
        dataset_b: DamlDataset,
        encode: Optional[bool] = None,
    ) -> DivergenceOutput:
        """
        Calculates the divergence and any errors between two datasets

        Parameters
        ----------
        dataset_a, dataset_b : DamlDataset
            Datasets to calculate the divergence between

        Returns
        -------
        DivergenceOutput
            Dataclass containing the dp divergence and errors during calculation

        Note
        ----
        A and B must be 2 dimensions, and equivalent in size on the second dimension
        """

        imgs_a: np.ndarray = dataset_a.images
        imgs_b: np.ndarray = dataset_b.images

        do_encode = self.encode if encode is None else encode
        # Pass dataset_a and dataset_b through an autoencoder
        # before evaluating dp divergence
        if do_encode:
            i1 = (
                imgs_a if isinstance(imgs_a, torch.Tensor) else permute_to_torch(imgs_a)
            )
            i2 = (
                imgs_b if isinstance(imgs_b, torch.Tensor) else permute_to_torch(imgs_b)
            )
            data = self.create_encoding(i1, i2)
        else:
            # Input could be a torch.Tensor (future update)
            imgs_a = (
                imgs_a if isinstance(imgs_a, np.ndarray) else permute_to_numpy(imgs_a)
            )
            imgs_b = (
                imgs_b if isinstance(imgs_b, np.ndarray) else permute_to_numpy(imgs_b)
            )

            # Combine data into one dataset
            data = np.vstack((imgs_a, imgs_b))

        N = imgs_a.shape[0]
        M = imgs_b.shape[0]
        labels = np.vstack([np.zeros([N, 1]), np.ones([M, 1])])
        errors = self.calculate_errors(data, labels)
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
        dense_eudist = squareform(pdist(data)) + 1e-4
        eudist_csr = csr_matrix(dense_eudist)
        mst = minimum_spanning_tree(eudist_csr)
        mst = mst.toarray()
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
        nn_indices = self._compute_neighbors(data, data)
        errors = np.sum(np.abs(labels[nn_indices] - labels))
        return errors
