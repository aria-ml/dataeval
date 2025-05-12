from __future__ import annotations

__all__ = []

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Literal, overload

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from dataeval.config import EPSILON, DeviceLike, get_seed
from dataeval.data import Embeddings, Select
from dataeval.data._selection import Selection, SelectionStage

_logger = logging.getLogger(__name__)


class _Clusters:
    __slots__ = ["labels", "cluster_centers", "unique_labels"]

    labels: NDArray[np.intp]
    cluster_centers: NDArray[np.float64]
    unique_labels: NDArray[np.intp]

    def __init__(self, labels: NDArray[np.intp], cluster_centers: NDArray[np.float64]) -> None:
        self.labels = labels
        self.cluster_centers = cluster_centers
        self.unique_labels = np.unique(labels)

    def _dist2center(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        dist = np.zeros(self.labels.shape)
        for lab in self.unique_labels:
            dist[self.labels == lab] = np.linalg.norm(X[self.labels == lab, :] - self.cluster_centers[lab, :], axis=1)
        return dist

    def _complexity(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        num_clst_intra = int(np.maximum(np.minimum(int(self.unique_labels.shape[0] / 5), 20), 1))
        d_intra = np.zeros(self.unique_labels.shape)
        d_inter = np.zeros(self.unique_labels.shape)
        for cdx, lab in enumerate(self.unique_labels):
            d_intra[cdx] = np.mean(np.linalg.norm(X[self.labels == lab, :] - self.cluster_centers[cdx, :], axis=1))
            d_inter[cdx] = np.mean(
                np.linalg.norm(self.cluster_centers - self.cluster_centers[cdx, :], axis=1)[:num_clst_intra]
            )
        cj = d_intra * d_inter
        tau = 0.1
        exp = np.exp(cj / tau)
        prob: NDArray[np.float64] = exp / np.sum(exp)
        return prob

    def _sort_by_weights(self, X: NDArray[np.float64]) -> NDArray[np.intp]:
        pr = self._complexity(X)
        d2c = self._dist2center(X)
        inds_per_clst: list[NDArray[np.intp]] = []
        for lab in zip(self.unique_labels):
            inds = np.nonzero(self.labels == lab)[0]
            # 'hardest' first
            srt_inds = np.argsort(d2c[inds])[::-1]
            inds_per_clst.append(inds[srt_inds])
        glob_inds: list[NDArray[np.intp]] = []
        while not bool(np.all([arr.size == 0 for arr in inds_per_clst])):
            clst_ind = np.random.choice(self.unique_labels, 1, p=pr)[0]
            if inds_per_clst[clst_ind].size > 0:
                glob_inds.append(inds_per_clst[clst_ind][0])
            else:
                continue
            inds_per_clst[clst_ind] = inds_per_clst[clst_ind][1:]
        # sorted hardest first; reverse for consistency
        return np.array(glob_inds[::-1])


class _Sorter(ABC):
    @abstractmethod
    def _sort(self, embeddings: NDArray[Any], reference: NDArray[Any] | None = None) -> NDArray[np.intp]: ...


class _KNNSorter(_Sorter):
    def __init__(self, samples: int, k: int | None) -> None:
        if k is None or k <= 0:
            k = int(np.sqrt(samples))
            _logger._log(logging.INFO, f"Setting k to default value of {k}", {"k": k, "samples": samples})
        elif k >= samples:
            raise ValueError(f"k={k} should be less than dataset size ({samples})")
        elif k >= samples / 10 and k > np.sqrt(samples):
            warnings.warn(
                f"Variable k={k} is large with respect to dataset size but valid; "
                + f"a nominal recommendation is k={int(np.sqrt(samples))}"
            )
        self._k = k

    def _sort(self, embeddings: NDArray[Any], reference: NDArray[Any] | None = None) -> NDArray[np.intp]:
        if reference is None:
            dists = pairwise_distances(embeddings, embeddings)
            np.fill_diagonal(dists, np.inf)
        else:
            dists = pairwise_distances(embeddings, reference)
        return np.argsort(np.sort(dists, axis=1)[:, self._k])


class _KMeansSorter(_Sorter):
    def __init__(self, samples: int, c: int | None, n_init: int | Literal["auto", "warn"] = "auto") -> None:
        if c is None or c <= 0:
            c = int(np.sqrt(samples))
            _logger._log(logging.INFO, f"Setting the value of num_clusters to a default value of {c}", {})
        if c >= samples:
            raise ValueError(f"c={c} should be less than dataset size ({samples})")
        self._c = c
        self._n_init = n_init

    def _get_clusters(self, embeddings: NDArray[Any]) -> _Clusters:
        clst = KMeans(n_clusters=self._c, init="k-means++", n_init=self._n_init, random_state=get_seed())  # type: ignore - n_init allows int but is typed as str
        clst.fit(embeddings)
        if clst.labels_ is None or clst.cluster_centers_ is None:
            raise ValueError("Clustering failed to produce labels or cluster centers")
        return _Clusters(clst.labels_, clst.cluster_centers_)


class _KMeansDistanceSorter(_KMeansSorter):
    def _sort(self, embeddings: NDArray[Any], reference: NDArray[Any] | None = None) -> NDArray[np.intp]:
        clst = self._get_clusters(embeddings if reference is None else reference)
        return np.argsort(clst._dist2center(embeddings))


class _KMeansComplexitySorter(_KMeansSorter):
    def _sort(self, embeddings: NDArray[Any], reference: NDArray[Any] | None = None) -> NDArray[np.intp]:
        clst = self._get_clusters(embeddings if reference is None else reference)
        return clst._sort_by_weights(embeddings)


class Prioritize(Selection[Any]):
    """
    Prioritizes the dataset by sort order in the embedding space.

    Parameters
    ----------
    model : torch.nn.Module
        Model to use for encoding images
    batch_size : int
        Batch size to use when encoding images
    device : DeviceLike or None
        Device to use for encoding images
    method : Literal["knn", "kmeans_distance", "kmeans_complexity"]
        Method to use for prioritization
    k : int | None, default None
        Number of nearest neighbors to use for prioritization (knn only)
    c : int | None, default None
        Number of clusters to use for prioritization (kmeans only)
    """

    stage = SelectionStage.ORDER

    @overload
    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int,
        device: DeviceLike | None,
        method: Literal["knn"],
        *,
        k: int | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int,
        device: DeviceLike | None,
        method: Literal["kmeans_distance", "kmeans_complexity"],
        *,
        c: int | None = None,
    ) -> None: ...

    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int,
        device: DeviceLike | None,
        method: Literal["knn", "kmeans_distance", "kmeans_complexity"],
        *,
        k: int | None = None,
        c: int | None = None,
    ) -> None:
        if method not in ("knn", "kmeans_distance", "kmeans_complexity"):
            raise ValueError(f"Invalid prioritization method: {method}")
        self._model = model
        self._batch_size = batch_size
        self._device = device
        self._method = method
        self._embeddings: Embeddings | None = None
        self._reference: Embeddings | None = None
        self._k = k
        self._c = c

    @overload
    @classmethod
    def using(
        cls,
        method: Literal["knn"],
        *,
        k: int | None = None,
        embeddings: Embeddings | None = None,
        reference: Embeddings | None = None,
    ) -> Prioritize: ...

    @overload
    @classmethod
    def using(
        cls,
        method: Literal["kmeans_distance", "kmeans_complexity"],
        *,
        c: int | None = None,
        embeddings: Embeddings | None = None,
        reference: Embeddings | None = None,
    ) -> Prioritize: ...

    @classmethod
    def using(
        cls,
        method: Literal["knn", "kmeans_distance", "kmeans_complexity"],
        *,
        k: int | None = None,
        c: int | None = None,
        embeddings: Embeddings | None = None,
        reference: Embeddings | None = None,
    ) -> Prioritize:
        """
        Prioritizes the dataset by sort order in the embedding space using existing
        embeddings and/or reference dataset embeddings.

        Parameters
        ----------
        method : Literal["knn", "kmeans_distance", "kmeans_complexity"]
            Method to use for prioritization
        embeddings : Embeddings or None, default None
            Embeddings to use for prioritization
        reference : Embeddings or None, default None
            Reference embeddings to prioritize relative to
        k : int or None, default None
            Number of nearest neighbors to use for prioritization (knn only)
        c : int or None, default None
            Number of clusters to use for prioritization (kmeans, cluster only)

        Notes
        -----
        At least one of `embeddings` or `reference` must be provided.
        """
        emb_params: Embeddings | None = embeddings if embeddings is not None else reference
        if emb_params is None:
            raise ValueError("Must provide at least embeddings or reference embeddings.")
        prioritize = Prioritize(emb_params._model, emb_params.batch_size, emb_params.device, method)
        prioritize._k = k
        prioritize._c = c
        prioritize._embeddings = embeddings
        prioritize._reference = reference
        return prioritize

    def _get_sorter(self, samples: int) -> _Sorter:
        if self._method == "knn":
            return _KNNSorter(samples, self._k)
        if self._method == "kmeans_distance":
            return _KMeansDistanceSorter(samples, self._c)
        # self._method == "kmeans_complexity"
        return _KMeansComplexitySorter(samples, self._c)

    def _to_normalized_ndarray(self, embeddings: Embeddings, selection: list[int] | None = None) -> NDArray[Any]:
        emb: NDArray[Any] = embeddings.to_numpy(selection)
        emb /= max(np.max(np.linalg.norm(emb, axis=1)), EPSILON)
        return emb

    def __call__(self, dataset: Select[Any]) -> None:
        # Initialize sorter
        self._sorter = self._get_sorter(len(dataset._selection))
        # Extract and normalize embeddings
        embeddings = (
            Embeddings(dataset, batch_size=self._batch_size, model=self._model, device=self._device)
            if self._embeddings is None
            else self._embeddings
        )
        if len(dataset._selection) != len(embeddings):
            raise ValueError(
                "Size of embeddings do not match the size of the selection: "
                + f"embeddings={len(embeddings)}, selection={len(dataset._selection)}"
            )
        emb = self._to_normalized_ndarray(embeddings, dataset._selection)
        ref = None if self._reference is None else self._to_normalized_ndarray(self._reference)
        # Sort indices
        dataset._selection = self._sorter._sort(emb, ref).tolist()
