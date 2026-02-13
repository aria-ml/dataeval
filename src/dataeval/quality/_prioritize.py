"""
Dataset-aware prioritization for quality evaluation.

This module provides convenient wrappers around core ranking algorithms
that handle dataset loading and embedding computation.
"""

__all__ = []

import logging
from collections.abc import Iterator
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from dataeval import Metadata as _Metadata
from dataeval.core._rank import (
    RankResult,
    rank_hdbscan_complexity,
    rank_hdbscan_distance,
    rank_kmeans_complexity,
    rank_kmeans_distance,
    rank_knn,
    rank_result_class_balanced,
    rank_result_stratified,
)
from dataeval.protocols import AnnotatedDataset, Array, FeatureExtractor
from dataeval.types import Evaluator, EvaluatorConfig, Output, set_metadata

_logger = logging.getLogger(__name__)


MethodType = Literal["knn", "kmeans_distance", "kmeans_complexity", "hdbscan_distance", "hdbscan_complexity"]
OrderType = Literal["easy_first", "hard_first"]
PolicyType = Literal["difficulty", "stratified", "class_balanced"]

# Default configuration values
DEFAULT_PRIORITIZE_METHOD: MethodType = "knn"
DEFAULT_PRIORITIZE_POLICY: PolicyType = "difficulty"
DEFAULT_PRIORITIZE_ORDER: OrderType = "easy_first"
DEFAULT_PRIORITIZE_N_INIT: int | Literal["auto"] = "auto"
DEFAULT_PRIORITIZE_NUM_BINS: int = 50


class PrioritizeOutput(Output[NDArray[np.intp]]):
    """
    Ranking result with lazy index computation based on order and policy.

    Stores the source ranking (always in easy_first order) and computes
    the final indices lazily based on the configured order and policy.
    All transformation methods return new PriorityOutput instances that
    operate on the same source data.
    """

    _fields: tuple[str, ...] = ("indices", "scores", "method", "order", "policy")

    def __init__(
        self,
        rank_result: RankResult,
        method: MethodType,
        order: OrderType = "easy_first",
        policy: PolicyType = "difficulty",
        num_bins: int = 50,
        class_labels: NDArray[np.integer[Any]] | None = None,
    ) -> None:
        """
        Initialize a PriorityOutput.

        Parameters
        ----------
        rank_result : RankResult
            Original ranking results containing indices and optional scores.
        method : {"knn", "kmeans_distance", "kmeans_complexity", "hdbscan_distance", "hdbscan_complexity"}
            The ranking method used.
        order : {"easy_first", "hard_first"}, default "easy_first"
            Sort direction for output indices.
        policy : {"difficulty", "stratified", "class_balanced"}, default "difficulty"
            Selection policy to apply.
        num_bins : int, default 50
            Number of bins for stratified policy.
        class_labels : NDArray[np.integer] | None, default None
            Class labels for class_balanced policy.
        """
        self._rank_result = rank_result
        self._method: MethodType = method
        self._order: OrderType = order
        self._policy: PolicyType = policy
        self._cached_indices: NDArray[np.intp] | None = None

        self.num_bins: int = num_bins
        self.class_labels: NDArray[np.integer[Any]] | None = class_labels

    def _compute_indices(self) -> NDArray[np.intp]:
        """Compute indices based on current order and policy."""
        if self._policy == "stratified" and self._rank_result["scores"] is None:
            raise ValueError("Cannot apply stratified policy: ranking scores are not available")

        if self._policy == "class_balanced" and self.class_labels is None:
            raise ValueError("Cannot apply class_balanced policy: class_labels not provided")

        # Create a PriorityOutput from source data (always easy_first)
        indices = (
            rank_result_stratified(self._rank_result, self.num_bins)
            if self._policy == "stratified"
            else rank_result_class_balanced(self._rank_result, cast(NDArray[np.integer[Any]], self.class_labels))
            if self._policy == "class_balanced"
            else self._rank_result["indices"].copy()
        )

        return indices[::-1] if self._order == "hard_first" else indices

    @property
    def indices(self) -> NDArray[np.intp]:
        """NDArray[np.intp] : Indices sorted according to configured order and policy (lazily computed)."""
        if self._cached_indices is None:
            self._cached_indices = self._compute_indices()
        return self._cached_indices

    @property
    def scores(self) -> NDArray[np.float32] | None:
        """NDArray[np.float32] | None : Ranking scores in configured order if available else None."""
        scores = self._rank_result["scores"]
        return scores if scores is None else scores[self.indices]

    @property
    def method(self) -> MethodType:
        """The ranking method that was used.

        {"knn", "kmeans_distance", "kmeans_complexity", "hdbscan_distance", "hdbscan_complexity"}
        """
        return self._method

    @property
    def order(self) -> OrderType:
        """{"easy_first", "hard_first"} : Sort direction."""
        return self._order

    @property
    def policy(self) -> PolicyType:
        """{"difficulty", "stratified", "class_balanced"} : Selection policy."""
        return self._policy

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __contains__(self, key: object) -> bool:
        return key in self._fields

    def keys(self) -> Iterator[str]:
        return iter(self._fields)

    def values(self) -> Iterator[Any]:
        return (getattr(self, k) for k in self._fields)

    def items(self) -> Iterator[tuple[str, Any]]:
        return ((k, getattr(self, k)) for k in self._fields)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._fields:
            return getattr(self, key)
        return default

    def __repr__(self) -> str:
        return (
            f"PriorityOutput(method={self._method!r}, order={self._order!r}, "
            f"policy={self._policy!r}, n_samples={len(self._rank_result['indices'])})"
        )

    def easy_first(self) -> Self:
        """
        Return a new PriorityOutput with easy_first sort order.

        Easy samples (prototypical, close to cluster centers) come first.
        Preserves the current policy. Idempotent if already easy_first.

        Returns
        -------
        PriorityOutput
            New result with easy_first order.

        Examples
        --------
        >>> # Get result from Prioritize
        >>> result = Prioritize.knn(extractor, k=5).evaluate(unlabeled_data)
        >>> # Transform to easy_first
        >>> easy_result = result.easy_first()
        >>> easy_result.order
        'easy_first'
        """
        if self._order == "easy_first":
            return self
        _logger.info(
            "Setting easy_first order: method=%s, policy=%s",
            self._method,
            self._policy,
        )
        return self.__class__(
            rank_result=self._rank_result,
            method=self._method,
            order="easy_first",
            policy=self._policy,
            num_bins=self.num_bins,
            class_labels=self.class_labels,
        )

    def hard_first(self) -> Self:
        """
        Return a new PriorityOutput with hard_first sort order.

        Hard samples (outliers, far from cluster centers) come first.
        Preserves the current policy. Idempotent if already hard_first.

        Returns
        -------
        PriorityOutput
            New result with hard_first order.

        Examples
        --------
        >>> # Get result from Prioritize
        >>> result = Prioritize.knn(extractor, k=5).evaluate(unlabeled_data)
        >>> # Transform to hard_first
        >>> hard_result = result.hard_first()
        >>> hard_result.order
        'hard_first'
        """
        if self._order == "hard_first":
            return self
        _logger.info(
            "Setting hard_first order: method=%s, policy=%s",
            self._method,
            self._policy,
        )
        return self.__class__(
            rank_result=self._rank_result,
            method=self._method,
            order="hard_first",
            policy=self._policy,
            num_bins=self.num_bins,
            class_labels=self.class_labels,
        )

    def stratified(self, num_bins: int = 50) -> Self:
        """
        Return a new PriorityOutput with stratified sampling policy.

        Applies stratified sampling to balance selection across score bins.
        This encourages diversity by de-weighting samples with similar scores.

        Parameters
        ----------
        num_bins : int, default 50
            Number of bins for stratification.

        Returns
        -------
        PriorityOutput
            New result with stratified policy.

        Raises
        ------
        ValueError
            If scores are not available (computed lazily when indices accessed).

        Examples
        --------
        >>> # Get result from Prioritize
        >>> result = Prioritize.knn(extractor, k=5).evaluate(unlabeled_data)
        >>> # Apply stratification to the result
        >>> strat_result = result.stratified(num_bins=10)
        >>> strat_result.policy
        'stratified'
        """
        if self._rank_result["scores"] is None:
            raise ValueError("Cannot apply stratified policy: ranking scores are not available")

        _logger.info(
            "Setting stratified policy: method=%s, num_bins=%d, order=%s",
            self._method,
            num_bins,
            self._order,
        )
        return self.__class__(
            rank_result=self._rank_result,
            method=self._method,
            order=self._order,
            policy="stratified",
            num_bins=num_bins,
            class_labels=self.class_labels,
        )

    def class_balanced(self, class_labels: NDArray[np.integer[Any]] | None = None) -> Self:
        """
        Return a new PriorityOutput with class-balanced sampling policy.

        Reorders to ensure balanced representation across classes while
        maintaining priority order within each class.

        Parameters
        ----------
        class_labels : NDArray[np.integer]
            Class label for each sample in the original dataset.

        Returns
        -------
        PriorityOutput
            New result with class_balanced policy.

        Examples
        --------
        >>> # Get result
        >>> result = Prioritize.knn(extractor, k=5).evaluate(unlabeled_data)
        >>> # Rebucket based on classes (class_labels typically from metadata)
        >>> balanced = result.class_balanced(class_labels)
        >>> balanced.policy
        'class_balanced'
        """
        class_labels = self.class_labels if class_labels is None else class_labels
        if class_labels is None:
            raise ValueError("class_labels must be provided either in method or as class attribute")

        num_classes = len(np.unique(class_labels))
        _logger.info(
            "Setting class_balanced policy: method=%s, %d classes, order=%s",
            self._method,
            num_classes,
            self._order,
        )
        return self.__class__(
            rank_result=self._rank_result,
            method=self._method,
            order=self._order,
            policy="class_balanced",
            num_bins=self.num_bins,
            class_labels=class_labels,
        )


class Prioritize(Evaluator):
    """
    Prioritize dataset samples based on their position in the embedding space.

    This class provides factory methods for common configurations and supports
    both direct instantiation and fluent policy configuration via the Output.

    Parameters
    ----------
    extractor : FeatureExtractor
        Feature extractor instance to use for extracting embeddings from data.
    method : {"knn", "kmeans_distance", "kmeans_complexity", "hdbscan_distance", \
"hdbscan_complexity"}, default "knn"
        Ranking method to use:

        - "knn": K-nearest neighbors distance ranking
        - "kmeans_distance": Distance to assigned K-means cluster center
        - "kmeans_complexity": Weighted sampling based on K-means cluster structure
        - "hdbscan_distance": Distance to assigned HDBSCAN cluster center
        - "hdbscan_complexity": Weighted sampling based on HDBSCAN cluster structure
    k : int or None, default None
        Number of nearest neighbors for "knn" method. If None, uses sqrt(n_samples).
    c : int or None, default None
        Number of clusters for clustering methods. If None, uses sqrt(n_samples).
    n_init : int or "auto", default "auto"
        Number of K-means initializations (kmeans methods only).
    max_cluster_size : int or None, default None
        Maximum cluster size for HDBSCAN methods.
    order : {"easy_first", "hard_first"}, default "easy_first"
        Sort direction for output indices:

        - "easy_first": Prototypical samples first (low distance)
        - "hard_first": Challenging samples first (high distance)
    policy : {"difficulty", "stratified", "class_balanced"}, default "difficulty"
        Selection policy:

        - "difficulty": Direct ordering by ranking results (no additional reordering)
        - "stratified": Balanced selection across difficulty bins
        - "class_balanced": Balanced selection across class labels
    num_bins : int, default 50
        Number of bins for "stratified" policy.
    reference : AnnotatedDataset or Array or None, default None
        Optional reference dataset or pre-computed embeddings. When provided,
        incoming datasets will be prioritized relative to this reference set.
        Useful for active learning (reference = labeled data) or quality
        filtering (reference = high-quality corpus).
    config : Prioritize.Config or None, default None
        Optional configuration object with default parameters. Parameters
        specified directly in __init__ will override config defaults.

    See Also
    --------
    :class:`~dataeval.quality.Outliers`
    :class:`~dataeval.selection.Indices`
    :func:`dataeval.core.rank_knn`
    :func:`dataeval.core.rank_kmeans_distance`
    :func:`dataeval.core.rank_kmeans_complexity`
    :func:`dataeval.core.rank_hdbscan_distance`
    :func:`dataeval.core.rank_hdbscan_complexity`

    Examples
    --------
    Using factory methods (recommended):

    >>> from dataeval.quality import Prioritize
    >>>
    >>> # KNN with default policy (difficulty/easy_first)
    >>> result = Prioritize.knn(extractor, k=10).evaluate(unlabeled_data)
    >>>
    >>> # Configure specific policy in factory
    >>> result = Prioritize.knn(extractor, k=10).evaluate(unlabeled_data)
    >>>
    >>> # Re-bucket results (Cheap operation)
    >>> stratified_res = result.stratified(num_bins=20)
    >>> hard_res = result.hard_first()

    Direct instantiation:

    >>> prioritizer = Prioritize(
    ...     extractor=extractor,
    ...     method="knn",
    ...     k=10,
    ...     policy="stratified",
    ...     num_bins=20,
    ... )
    >>> result = prioritizer.evaluate(unlabeled_data)

    Active learning with reference data:

    >>> # Prioritize unlabeled data based on distance to labeled data
    >>> prioritizer = Prioritize.knn(extractor, k=10, reference=labeled_data)
    >>> result = prioritizer.evaluate(unlabeled_data)
    >>> # Get the items most unlike the reference data
    >>> most_novel = result.hard_first().indices

    Using configuration:

    >>> config = Prioritize.Config(extractor=extractor, method="knn", k=10)
    >>> prioritizer = Prioritize(config=config)

    Applying class-balanced policy with class labels from metadata:

    >>> prioritizer = Prioritize.knn(extractor, k=5)
    >>> # evaluate() extracts labels from dataset metadata automatically
    >>> result = prioritizer.evaluate(unlabeled_data)
    >>> # Apply balancing
    >>> balanced = result.class_balanced()
    """

    class Config(EvaluatorConfig):
        """
        Configuration for Prioritize evaluator.

        Attributes
        ----------
        extractor : FeatureExtractor or None
            Feature extractor instance to use for extracting embeddings
            from data.
        method : {"knn", "kmeans_distance", "kmeans_complexity", "hdbscan_distance", \
"hdbscan_complexity"}, default "knn"
            Ranking method to use.
        k : int or None, default None
            Number of nearest neighbors for "knn" method.
        c : int or None, default None
            Number of clusters for clustering methods.
        n_init : int or "auto", default "auto"
            Number of K-means initializations (kmeans methods only).
        max_cluster_size : int or None, default None
            Maximum cluster size for HDBSCAN methods.
        order : {"easy_first", "hard_first"}, default "easy_first"
            Sort direction for output indices.
        policy : {"difficulty", "stratified", "class_balanced"}, default "difficulty"
            Selection policy to apply after ranking.
        num_bins : int, default 50
            Number of bins for "stratified" policy.
        """

        extractor: FeatureExtractor | None = None
        method: MethodType = DEFAULT_PRIORITIZE_METHOD
        k: int | None = None
        c: int | None = None
        n_init: int | Literal["auto"] = DEFAULT_PRIORITIZE_N_INIT
        max_cluster_size: int | None = None
        policy: PolicyType = DEFAULT_PRIORITIZE_POLICY
        order: OrderType = DEFAULT_PRIORITIZE_ORDER
        num_bins: int = DEFAULT_PRIORITIZE_NUM_BINS

    # Type declarations for attributes set by apply_config
    extractor: FeatureExtractor
    method: MethodType
    k: int | None
    c: int | None
    n_init: int | Literal["auto"]
    max_cluster_size: int | None
    order: OrderType
    policy: PolicyType
    num_bins: int
    config: Config

    def __init__(
        self,
        extractor: FeatureExtractor | None = None,
        method: MethodType | None = None,
        k: int | None = None,
        c: int | None = None,
        n_init: int | Literal["auto"] | None = None,
        max_cluster_size: int | None = None,
        order: OrderType | None = None,
        policy: PolicyType | None = None,
        num_bins: int | None = None,
        reference: AnnotatedDataset[Any] | Array | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(locals(), exclude={"reference"})
        self._reference = reference

        if self.extractor is None:
            raise ValueError("extractor must be provided either in __init__ or config")

    # ==================== Factory Class Methods ====================

    @classmethod
    def knn(
        cls,
        extractor: FeatureExtractor,
        k: int | None = None,
        reference: AnnotatedDataset[Any] | Array | None = None,
    ) -> Self:
        """
        Create a Prioritize instance using k-nearest neighbors method.

        Parameters
        ----------
        extractor : FeatureExtractor
            Feature extractor instance for embedding extraction.
        k : int or None, default None
            Number of nearest neighbors. If None, uses sqrt(n_samples).
        reference : AnnotatedDataset or Array or None, default None
            Optional reference dataset for relative prioritization.

        Returns
        -------
        Prioritize
            Configured instance ready for evaluation.

        Examples
        --------
        >>> # Default KNN
        >>> result = Prioritize.knn(extractor, k=10).evaluate(unlabeled_data)
        >>> # KNN relative to a reference (Active Learning)
        >>> result = Prioritize.knn(extractor, k=5, reference=labeled_data).evaluate(unlabeled_data)
        """
        return cls(extractor=extractor, method="knn", k=k, reference=reference)

    @classmethod
    def kmeans_distance(
        cls,
        extractor: FeatureExtractor,
        c: int | None = None,
        n_init: int | Literal["auto"] = "auto",
        reference: AnnotatedDataset[Any] | Array | None = None,
    ) -> Self:
        """
        Create a Prioritize instance using K-means distance method.

        Ranks samples by distance to their assigned cluster centers.
        Returns samples in easy-first order (low distance = prototypical).

        Parameters
        ----------
        extractor : FeatureExtractor
            Feature extractor instance for embedding extraction.
        c : int or None, default None
            Number of clusters. If None, uses sqrt(n_samples).
        n_init : int or "auto", default "auto"
            Number of K-means initializations.
        reference : AnnotatedDataset or Array or None, default None
            Optional reference dataset for relative prioritization.

        Returns
        -------
        Prioritize
            Configured instance ready for evaluation.

        Examples
        --------
        >>> result = Prioritize.kmeans_distance(extractor, c=15).evaluate(unlabeled_data)
        """
        return cls(extractor=extractor, method="kmeans_distance", c=c, n_init=n_init, reference=reference)

    @classmethod
    def kmeans_complexity(
        cls,
        extractor: FeatureExtractor,
        c: int | None = None,
        n_init: int | Literal["auto"] = "auto",
        reference: AnnotatedDataset[Any] | Array | None = None,
    ) -> Self:
        """
        Create a Prioritize instance using K-means complexity method.

        Uses weighted sampling based on intra/inter-cluster distances.
        Returns samples in easy-first order.

        Note: This method does not produce scores, so "stratified" policy
        is not available.

        Parameters
        ----------
        extractor : FeatureExtractor
            Feature extractor instance for embedding extraction.
        c : int or None, default None
            Number of clusters. If None, uses sqrt(n_samples).
        n_init : int or "auto", default "auto"
            Number of K-means initializations.
        reference : AnnotatedDataset or Array or None, default None
            Optional reference dataset for relative prioritization.

        Returns
        -------
        Prioritize
            Configured instance ready for evaluation.

        Examples
        --------
        >>> result = Prioritize.kmeans_complexity(extractor, c=10).evaluate(unlabeled_data)
        """
        return cls(extractor=extractor, method="kmeans_complexity", c=c, n_init=n_init, reference=reference)

    @classmethod
    def hdbscan_distance(
        cls,
        extractor: FeatureExtractor,
        c: int | None = None,
        max_cluster_size: int | None = None,
        reference: AnnotatedDataset[Any] | Array | None = None,
    ) -> Self:
        """
        Create a Prioritize instance using HDBSCAN distance method.

        Clusters embeddings using HDBSCAN and ranks by distance to assigned cluster
        centers. Returns samples in easy-first order (low distance = prototypical).

        Parameters
        ----------
        extractor : FeatureExtractor
            Feature extractor instance for embedding extraction.
        c : int or None, default None
            Expected number of clusters (used as hint for min_cluster_size).
            If None, uses sqrt(n_samples).
        max_cluster_size : int or None, default None
            Maximum size limit for identified clusters.
        reference : AnnotatedDataset or Array or None, default None
            Optional reference dataset for relative prioritization.

        Returns
        -------
        Prioritize
            Configured instance ready for evaluation.

        Examples
        --------
        >>> result = Prioritize.hdbscan_distance(extractor, c=15).evaluate(unlabeled_data)
        """
        return cls(
            extractor=extractor,
            method="hdbscan_distance",
            c=c,
            max_cluster_size=max_cluster_size,
            reference=reference,
        )

    @classmethod
    def hdbscan_complexity(
        cls,
        extractor: FeatureExtractor,
        c: int | None = None,
        max_cluster_size: int | None = None,
        reference: AnnotatedDataset[Any] | Array | None = None,
    ) -> Self:
        """
        Create a Prioritize instance using HDBSCAN complexity method.

        Uses weighted sampling based on intra/inter-cluster distances from
        HDBSCAN clustering. Returns samples in easy-first order.

        Note: This method does not produce scores, so "stratified" policy
        is not available.

        Parameters
        ----------
        extractor : FeatureExtractor
            Feature extractor instance for embedding extraction.
        c : int or None, default None
            Expected number of clusters (used as hint for min_cluster_size).
            If None, uses sqrt(n_samples).
        max_cluster_size : int or None, default None
            Maximum size limit for identified clusters.
        reference : AnnotatedDataset or Array or None, default None
            Optional reference dataset for relative prioritization.

        Returns
        -------
        Prioritize
            Configured instance ready for evaluation.

        Examples
        --------
        >>> result = Prioritize.hdbscan_complexity(extractor, c=10).evaluate(unlabeled_data)
        """
        return cls(
            extractor=extractor,
            method="hdbscan_complexity",
            c=c,
            max_cluster_size=max_cluster_size,
            reference=reference,
        )

    @set_metadata(state=["method", "k", "c", "n_init", "policy", "num_bins"])
    def evaluate(
        self,
        dataset: AnnotatedDataset[Any] | Array,
        class_labels: NDArray[np.integer[Any]] | None = None,
    ) -> PrioritizeOutput:
        """
        Evaluate the dataset and return prioritized indices.

        Uses the configured method and policy to rank samples.

        Parameters
        ----------
        dataset : AnnotatedDataset[Any] | Array
            The incoming dataset to prioritize. Can be either:

            - AnnotatedDataset: Will compute embeddings using the extractor
            - Array: Pre-computed embeddings (e.g. from Embeddings or numpy)

        class_labels : NDArray[np.integer] | None, default None
            Optional class labels for class_balanced policy. If not provided,
            will attempt to extract from dataset metadata.

        Returns
        -------
        PriorityOutput
            Output containing prioritized indices, scores (if available),
            and configuration information.

        Raises
        ------
        ValueError
            If class_balanced policy is used with a dataset that lacks metadata
            (e.g., raw arrays).
            If stratified policy is used with complexity methods (no scores).
        TypeError
            If dataset is neither an AnnotatedDataset nor Array.

        Examples
        --------
        Using factory methods:

        >>> result = Prioritize.knn(extractor, k=5).evaluate(unlabeled_data)

        Using direct instantiation:

        >>> prioritizer = Prioritize(extractor=extractor, method="knn", k=5, order="hard_first")
        >>> result = prioritizer.evaluate(unlabeled_data)
        """
        # Validate stratified + complexity method combinations
        if self.policy == "stratified" and self.method in ("kmeans_complexity", "hdbscan_complexity"):
            raise ValueError(
                f"stratified policy is not available with {self.method} method ({self.method} does not produce scores)",
            )

        # Check if dataset is an Array (pre-computed) or AnnotatedDataset
        if isinstance(dataset, Array):
            # Pre-computed embeddings - use directly
            embeddings_array = np.asarray(dataset)
        else:
            # Assume dataset is an AnnotatedDataset - compute embeddings
            try:
                from dataeval._embeddings import Embeddings as _Embeddings

                embeddings_array = np.asarray(_Embeddings(dataset, extractor=self.extractor))
                if class_labels is None:
                    self._metadata = _Metadata(dataset)
                    class_labels = self._metadata.class_labels
            except Exception as e:
                raise TypeError(
                    f"dataset must be either an AnnotatedDataset or Array, but got {type(dataset).__name__}",
                ) from e

        if self._reference is None:
            reference_array = None
        elif isinstance(self._reference, Array):
            reference_array = np.asarray(self._reference)
        else:
            from dataeval._embeddings import Embeddings as _Embeddings

            reference_array = np.asarray(_Embeddings(self._reference, extractor=self.extractor))

        # Check if we have labels for the requested policy
        if self.policy == "class_balanced" and class_labels is None:
            raise ValueError(
                "Policy 'class_balanced' requires an AnnotatedDataset with metadata. "
                "For raw arrays, use result.class_balanced(labels) instead.",
            )
        result = self._perform_ranking(embeddings_array, reference_array)

        return PrioritizeOutput(
            rank_result=result,
            method=self.method,
            order=self.order,
            policy=self.policy,
            num_bins=self.num_bins,
            class_labels=class_labels,
        )

    def _perform_ranking(
        self,
        embeddings_array: NDArray[np.floating[Any]],
        reference_array: NDArray[np.floating[Any]] | None,
    ) -> Any:
        """Perform initial ranking based on configured method."""
        if self.method == "knn":
            return rank_knn(embeddings_array, k=self.k, reference=reference_array)
        if self.method == "kmeans_distance":
            return rank_kmeans_distance(embeddings_array, c=self.c, n_init=self.n_init, reference=reference_array)
        if self.method == "kmeans_complexity":
            return rank_kmeans_complexity(embeddings_array, c=self.c, n_init=self.n_init, reference=reference_array)
        if self.method == "hdbscan_distance":
            return rank_hdbscan_distance(
                embeddings_array,
                c=self.c,
                max_cluster_size=self.max_cluster_size,
                reference=reference_array,
            )
        if self.method == "hdbscan_complexity":
            return rank_hdbscan_complexity(
                embeddings_array,
                c=self.c,
                max_cluster_size=self.max_cluster_size,
                reference=reference_array,
            )
        raise ValueError(f"Invalid method: {self.method}")
