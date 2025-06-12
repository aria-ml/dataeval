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

    def _dist2center(self, X: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
        dist = np.zeros(self.labels.shape, dtype=np.float32)
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
    scores: NDArray[np.float32] | None = None

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
            dists = pairwise_distances(embeddings, embeddings).astype(np.float32)
            np.fill_diagonal(dists, np.inf)
        else:
            dists = pairwise_distances(embeddings, reference).astype(np.float32)
        self.scores = np.sort(dists, axis=1)[:, self._k]
        return np.argsort(self.scores)


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
        self.scores = clst._dist2center(embeddings)
        return np.argsort(self.scores)


class _KMeansComplexitySorter(_KMeansSorter):
    def _sort(self, embeddings: NDArray[Any], reference: NDArray[Any] | None = None) -> NDArray[np.intp]:
        clst = self._get_clusters(embeddings if reference is None else reference)
        return clst._sort_by_weights(embeddings)


class Prioritize(Selection[Any]):
    """
    Sort the dataset indices in order of highest priority data in the embedding space.

    Parameters
    ----------
    model : torch.nn.Module | None
        Model to use for encoding images
    batch_size : int
        Batch size to use when encoding images
    device : DeviceLike or None
        Device to use for encoding images
    method : Literal["knn", "kmeans_distance", "kmeans_complexity"]
        Method to use for prioritization
    k : int or None, default None
        Number of nearest neighbors to use for prioritization.
        If None, uses the square_root of the number of samples. Only used for method="knn", ignored otherwise.
    c : int or None, default None
        Number of clusters to use for prioritization. If None, uses the square_root of the number of samples.
        Only used for method="kmeans_*", ignored otherwise.

    Notes
    -----
    1. `k` is only used for method ["knn"].
    2. `c` is only used for methods ["kmeans_distance", "kmeans_complexity"].

    Raises
    ------
    ValueError
        If method not in supported methods

    """

    stage = SelectionStage.ORDER

    @overload
    def __init__(
        self,
        model: torch.nn.Module | None,
        batch_size: int,
        device: DeviceLike | None,
        method: Literal["knn"],
        policy: Literal["hard_first", "easy_first", "stratified", "class_balance"],
        *,
        k: int | None = None,
        class_label: NDArray[np.integer[Any]] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        model: torch.nn.Module | None,
        batch_size: int,
        device: DeviceLike | None,
        method: Literal["kmeans_distance", "kmeans_complexity"],
        policy: Literal["hard_first", "easy_first", "stratified", "class_balance"],
        *,
        c: int | None = None,
        class_label: NDArray[np.integer[Any]] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        model: torch.nn.Module | None,
        batch_size: int,
        device: DeviceLike | None,
        method: Literal["knn", "kmeans_distance", "kmeans_complexity"],
        policy: Literal["class_balance"],
        *,
        k: int | None = None,
        c: int | None = None,
        class_label: NDArray[np.integer[Any]] | None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        model: torch.nn.Module | None,
        batch_size: int,
        device: DeviceLike | None,
        method: Literal["knn", "kmeans_distance", "kmeans_complexity"],
        policy: Literal["hard_first", "easy_first", "stratified"],
        *,
        k: int | None = None,
        c: int | None = None,
        class_label: NDArray[np.integer[Any]] | None = None,
    ) -> None: ...

    def __init__(
        self,
        model: torch.nn.Module | None,
        batch_size: int,
        device: DeviceLike | None,
        method: Literal["knn", "kmeans_distance", "kmeans_complexity"],
        policy: Literal["hard_first", "easy_first", "stratified", "class_balance"],
        *,
        k: int | None = None,
        c: int | None = None,
        class_label: NDArray[np.integer[Any]] | None = None,
    ) -> None:
        if method not in {"knn", "kmeans_distance", "kmeans_complexity"}:
            raise ValueError(f"Invalid prioritization method: {method}")
        if policy not in ("hard_first", "easy_first", "stratified", "class_balance"):
            raise ValueError(f"Invalid selection policy: {policy}")
        self._model = model
        self._batch_size = batch_size
        self._device = device
        self._method = method
        self._policy = policy
        self._embeddings: Embeddings | None = None
        self._reference: Embeddings | None = None
        self._k = k
        self._c = c
        self.class_label = class_label

    @overload
    @classmethod
    def using(
        cls,
        method: Literal["knn"],
        policy: Literal["hard_first", "easy_first", "stratified", "class_balance"],
        *,
        k: int | None = None,
        embeddings: Embeddings | None = None,
        reference: Embeddings | None = None,
        class_label: NDArray[np.integer[Any]] | None = None,
    ) -> Prioritize: ...

    @overload
    @classmethod
    def using(
        cls,
        method: Literal["kmeans_distance", "kmeans_complexity"],
        policy: Literal["hard_first", "easy_first", "stratified", "class_balance"],
        *,
        c: int | None = None,
        embeddings: Embeddings | None = None,
        reference: Embeddings | None = None,
        class_label: NDArray[np.integer[Any]] | None = None,
    ) -> Prioritize: ...

    @classmethod
    def using(
        cls,
        method: Literal["knn", "kmeans_distance", "kmeans_complexity"],
        policy: Literal["hard_first", "easy_first", "stratified", "class_balance"],
        *,
        k: int | None = None,
        c: int | None = None,
        embeddings: Embeddings | None = None,
        reference: Embeddings | None = None,
        class_label: NDArray[np.integer[Any]] | None = None,
    ) -> Prioritize:
        """
        Use precalculated embeddings to sort the dataset indices in order of
        highest priority data in the embedding space.

        Parameters
        ----------
        method : Literal["knn", "kmeans_distance", "kmeans_complexity"]
            Method to use for sample scoring during prioritization.
        policy : Literal["hard_first","easy_first","stratified","class_balance"]
            Selection policy for prioritizing scored samples.
        embeddings : Embeddings or None, default None
            Embeddings to use during prioritization. If None, `reference` must be set.
        reference : Embeddings or None, default None
            Reference embeddings used to prioritize the calculated dataset embeddings relative to them.
            If `embeddings` is None, this will be used instead.
        k : int or None, default None
            Number of nearest neighbors to use for prioritization.
            If None, uses the square_root of the number of samples. Only used for method="knn", ignored otherwise.
        c : int or None, default None
            Number of clusters to use for prioritization. If None, uses the square_root of the number of samples.
            Only used for method="kmeans_*", ignored otherwise.

        Notes
        -----
        1. `k` is only used for method ["knn"].
        2. `c` is only used for methods ["kmeans_distance", "kmeans_complexity"].

        Raises
        ------
        ValueError
            If both `embeddings` and `reference` are None

        """
        emb_params: Embeddings | None = embeddings if embeddings is not None else reference
        if emb_params is None:
            raise ValueError("Must provide at least embeddings or reference embeddings.")
        prioritize = Prioritize(
            emb_params._model,
            emb_params.batch_size,
            emb_params.device,
            method,
            policy,
            k=k,
            c=c,
            class_label=class_label,
        )
        prioritize._embeddings = embeddings
        prioritize._reference = reference
        return prioritize

    def _get_sorter(self, samples: int) -> _Sorter:
        if self._method == "knn":
            return _KNNSorter(samples, self._k)
        if self._method == "kmeans_distance":
            return _KMeansDistanceSorter(samples, self._c)
        return _KMeansComplexitySorter(samples, self._c)

    def _compute_bin_extents(self, scores: NDArray[np.floating[Any]]) -> tuple[np.float64, np.float64]:
        """
        Compute min/max bin extents for `scores`, padding outward by epsilon

        Parameters
        ----------
        scores: NDArray[np.float64])
            Array of floats to bin

        Returns
        -------
        tuple[np.float64, np.float64]
            (min,max) scores padded outward by epsilon = 1e-6*range(scores).
        """
        # ensure binning captures all samples in range
        scores = scores.astype(np.float64)
        min_score = np.min(scores)
        max_score = np.max(scores)
        rng = max_score - min_score
        eps = rng * 1e-6
        return min_score - eps, max_score + eps

    def _select_ordered_by_label(self, labels: NDArray[np.integer[Any]]) -> NDArray[np.intp]:
        """
        Given labels (class, group, bin, etc) sorted with decreasing priority,
        rerank so that we have approximate class/group balance.  This function
        is used for both stratified and class-balance rerank methods.

        We could require and return prioritization scores and re-sorted class
        labels, but it is more compact to return indices.  This allows us to
        resort other quantities, as well, outside the function.

        Parameters
        ---------
        labels: NDArray[np.integer[Any]]
            Class label or group ID per instance in order of decreasing priority

        Returns
        -------
        NDArray[np.intp]
            Indices that sort samples according to uniform class balance or
            group membership while respecting priority of the initial ordering.
        """
        labels = np.array(labels)
        num_samp = labels.shape[0]
        selected = np.zeros(num_samp, dtype=bool)
        # preserve ordering
        _, index = np.unique(labels, return_index=True)
        u_lab = labels[np.sort(index)]
        n_cls = len(u_lab)

        resort_inds = []
        cls_idx = 0
        n = 0
        while len(resort_inds) < num_samp:
            c0 = u_lab[cls_idx % n_cls]
            samples_available = (~selected) * (labels == c0)
            if any(samples_available):
                i0 = np.argmax(samples_available)  # selects first occurrence
                resort_inds.append(i0)
                selected[i0] = True
            cls_idx += 1
            n += 1
        return np.array(resort_inds).astype(np.intp)

    def _stratified_rerank(
        self,
        scores: NDArray[np.floating[Any]],
        indices: NDArray[np.integer[Any]],
        num_bins: int = 50,
    ) -> NDArray[np.intp]:
        """
        Re-rank samples by sampling uniformly over binned scores.  This
        de-weights selection of samples with similar scores and encourages both
        prototypical and challenging samples near the decision boundary.

        Inputs
        ------
        scores: NDArray[float]
            prioritization scores sorted in order of decreasing priority
        indices: NDArray[int]
            Indices to be re-sorted according to stratified sampling of scores.
            Indices are ordered by decreasing priority.
        num_bins: int


        Returns
        -------
        NDArray[int]
            re-ranked indices

        """
        mn, mx = self._compute_bin_extents(scores)
        bin_edges = np.linspace(mn, mx, num=num_bins + 1, endpoint=True)
        bin_label = np.digitize(scores, bin_edges)
        srt_inds = self._select_ordered_by_label(bin_label)
        return indices[srt_inds].astype(np.intp)

    def _rerank(
        self,
        indices: NDArray[np.integer[Any]],
    ) -> NDArray[np.intp]:
        """
        Re-rank samples according to the re-rank policy, self._policy.  Values
        from the 'indices' and optional 'scores' and 'class_label' variables are
        assumed to correspond by index---i.e. indices[i], scores[i], and
        class_label[i] should all refer to the same instance in the dataset.

        Note: indices are assumed to be sorted with easy/prototypical samples
        first--increasing order by most prioritization scoring methods.

        Parameters
        ----------
        indices: NDArray[np.intp]
            Indices that sort samples by increasing prioritization score, where
            low scores indicate high prototypicality ('easy') and high scores
            indicate challenging samples near the decision boundary ('hard').
        """

        if self._policy == "easy_first":
            return indices.astype(np.intp)
        if self._policy == "stratified":
            if self._sorter.scores is None:
                raise (
                    ValueError(
                        "Prioritization scores are necessary in order to use "
                        "stratified re-rank.  Use 'knn' or 'kmeans_distance' "
                        "methods to populate scores."
                    )
                )
            return self._stratified_rerank(self._sorter.scores[::-1], indices[::-1])
        if self._policy == "class_balance":
            if self.class_label is None:
                raise (ValueError("Class labels are necessary in order to use class_balance re-rank"))
            indices_reversed = self._select_ordered_by_label(self.class_label[indices[::-1]]).astype(np.int32)
            n = len(indices_reversed)
            return (n - 1 - indices_reversed).astype(np.intp)
        # elif self._policy == "hard_first" (default)
        return indices[::-1].astype(np.intp)

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
        indices = self._sorter._sort(emb, ref)
        dataset._selection = indices[self._rerank(indices)].astype(int).tolist()
