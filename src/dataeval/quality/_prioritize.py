"""
Dataset-aware prioritization for quality evaluation.

This module provides convenient wrappers around core ranking algorithms
that handle dataset loading and embedding computation.
"""

__all__ = []

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from dataeval import Embeddings
from dataeval import Metadata as _Metadata
from dataeval.core._rank import rank_kmeans_complexity, rank_kmeans_distance, rank_knn
from dataeval.core._rerank import rerank_class_balance, rerank_hard_first, rerank_stratified
from dataeval.protocols import AnnotatedDataset, EmbeddingEncoder, Metadata
from dataeval.types import Evaluator, EvaluatorConfig, Output, set_metadata

# Default configuration values
DEFAULT_PRIORITIZE_METHOD: Literal["knn", "kmeans_distance", "kmeans_complexity"] = "knn"
DEFAULT_PRIORITIZE_POLICY: Literal["hard_first", "easy_first", "stratified", "class_balance"] = "hard_first"
DEFAULT_PRIORITIZE_N_INIT: int | Literal["auto"] = "auto"
DEFAULT_PRIORITIZE_NUM_BINS: int = 50


@dataclass(frozen=True)
class PrioritizeOutput(Output[NDArray[np.intp]]):
    """
    Output class for :class:`.Prioritize` quality evaluator.

    Attributes
    ----------
    indices : NDArray[np.intp]
        Indices that sort the dataset in order of priority according to the
        specified method and policy. These indices can be used with the
        :class:`~dataeval.selection.Indices` selection class.
    scores : NDArray[np.float32] | None
        Prioritization scores for each sample (only available for methods
        that compute scores: "knn" and "kmeans_distance"). Scores are ordered
        according to the original dataset order, not the prioritized order.
    method : Literal["knn", "kmeans_distance", "kmeans_complexity"]
        The prioritization method that was used.
    policy : Literal["hard_first", "easy_first", "stratified", "class_balance"]
        The selection policy that was applied.
    """

    indices: NDArray[np.intp]
    scores: NDArray[np.float32] | None
    method: Literal["knn", "kmeans_distance", "kmeans_complexity"]
    policy: Literal["hard_first", "easy_first", "stratified", "class_balance"]

    def data(self) -> NDArray[np.intp]:
        """Returns the prioritized indices."""
        return self.indices

    def __len__(self) -> int:
        """Returns the number of prioritized samples."""
        return len(self.indices)


class Prioritize(Evaluator):
    """
    Prioritize dataset samples based on their position in the embedding space.

    This class provides factory methods for common configurations and supports
    both direct instantiation and fluent policy configuration.

    Parameters
    ----------
    encoder : EmbeddingEncoder
        Encoder to use for extracting embeddings from data.
    method : {"knn", "kmeans_distance", "kmeans_complexity"}, default "knn"
        Ranking method to use:

        - "knn": K-nearest neighbors distance ranking
        - "kmeans_distance": Distance to assigned cluster center
        - "kmeans_complexity": Weighted sampling based on cluster structure
    k : int or None, default None
        Number of nearest neighbors for "knn" method. If None, uses sqrt(n_samples).
    c : int or None, default None
        Number of clusters for kmeans methods. If None, uses sqrt(n_samples).
    n_init : int or "auto", default "auto"
        Number of K-means initializations for kmeans methods.
    policy : {"hard_first", "easy_first", "stratified", "class_balance"}, default "hard_first"
        Selection policy:

        - "hard_first": Challenging samples first (high distance)
        - "easy_first": Prototypical samples first (low distance)
        - "stratified": Balanced selection across difficulty bins
        - "class_balance": Balanced selection across class labels
    num_bins : int, default 50
        Number of bins for "stratified" policy.
    class_labels : NDArray[np.integer] or None, default None
        Class labels for "class_balance" policy. If None, extracted from
        AnnotatedDataset metadata during evaluate().
    reference : AnnotatedDataset or Embeddings or None, default None
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

    Examples
    --------
    Using factory methods (recommended):

    >>> from dataeval.quality import Prioritize
    >>>
    >>> # KNN with hard samples first (default policy)
    >>> result = Prioritize.knn(encoder, k=10).evaluate(dataset)
    >>>
    >>> # KNN with easy samples first
    >>> result = Prioritize.knn(encoder, k=10).easy_first().evaluate(dataset)
    >>>
    >>> # K-means distance with stratified sampling
    >>> result = Prioritize.kmeans_distance(encoder, c=15).stratified(num_bins=20).evaluate(dataset)
    >>>
    >>> # K-means complexity with class-balanced selection
    >>> result = Prioritize.kmeans_complexity(encoder, c=10).class_balanced().evaluate(labeled_data)

    Direct instantiation:

    >>> prioritizer = Prioritize(
    ...     encoder=encoder,
    ...     method="knn",
    ...     k=10,
    ...     policy="stratified",
    ...     num_bins=20,
    ... )
    >>> result = prioritizer.evaluate(dataset)

    Active learning with reference data:

    >>> prioritizer = Prioritize.knn(encoder, k=10, reference=labeled_data)
    >>> result = prioritizer.hard_first().evaluate(unlabeled_data)

    Using configuration:

    >>> config = Prioritize.Config(encoder=encoder, method="knn", k=10)
    >>> prioritizer = Prioritize(config=config)
    """

    class Config(EvaluatorConfig):
        """
        Configuration for Prioritize evaluator.

        Attributes
        ----------
        encoder : EmbeddingEncoder or None
            Encoder to use for extracting embeddings from data.
        method : {"knn", "kmeans_distance", "kmeans_complexity"}, default "knn"
            Ranking method to use.
        k : int or None, default None
            Number of nearest neighbors for "knn" method.
        c : int or None, default None
            Number of clusters for kmeans methods.
        n_init : int or "auto", default "auto"
            Number of K-means initializations.
        policy : {"hard_first", "easy_first", "stratified", "class_balance"}, default "hard_first"
            Selection policy to apply after ranking.
        num_bins : int, default 50
            Number of bins for "stratified" policy.
        class_labels : NDArray[np.integer] or None, default None
            Class labels for "class_balance" policy.
        """

        encoder: EmbeddingEncoder | None = None
        # Method configuration
        method: Literal["knn", "kmeans_distance", "kmeans_complexity"] = DEFAULT_PRIORITIZE_METHOD
        k: int | None = None
        c: int | None = None
        n_init: int | Literal["auto"] = DEFAULT_PRIORITIZE_N_INIT
        # Policy configuration
        policy: Literal["hard_first", "easy_first", "stratified", "class_balance"] = DEFAULT_PRIORITIZE_POLICY
        num_bins: int = DEFAULT_PRIORITIZE_NUM_BINS
        class_labels: NDArray[np.integer[Any]] | None = None

    # Type declarations for attributes set by apply_config
    encoder: EmbeddingEncoder
    method: Literal["knn", "kmeans_distance", "kmeans_complexity"]
    k: int | None
    c: int | None
    n_init: int | Literal["auto"]
    policy: Literal["hard_first", "easy_first", "stratified", "class_balance"]
    num_bins: int
    class_labels: NDArray[np.integer[Any]] | None
    config: Config

    def __init__(
        self,
        encoder: EmbeddingEncoder | None = None,
        method: Literal["knn", "kmeans_distance", "kmeans_complexity"] | None = None,
        k: int | None = None,
        c: int | None = None,
        n_init: int | Literal["auto"] | None = None,
        policy: Literal["hard_first", "easy_first", "stratified", "class_balance"] | None = None,
        num_bins: int | None = None,
        class_labels: NDArray[np.integer[Any]] | None = None,
        reference: AnnotatedDataset[Any] | Embeddings | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(locals(), exclude={"reference"})
        self._reference = reference

        if self.encoder is None:
            raise ValueError("encoder must be provided either in __init__ or config")

        # Internal state populated during evaluate
        self.embeddings: Embeddings | None = None
        self.ref_embeddings: Embeddings | None = None
        self.metadata: Metadata | None = None

    # ==================== Factory Class Methods ====================

    @classmethod
    def knn(
        cls,
        encoder: EmbeddingEncoder,
        k: int | None = None,
        reference: AnnotatedDataset[Any] | Embeddings | None = None,
    ) -> Self:
        """
        Create a Prioritize instance using k-nearest neighbors method.

        Parameters
        ----------
        encoder : EmbeddingEncoder
            Encoder to use for extracting embeddings from data.
        k : int or None, default None
            Number of nearest neighbors. If None, uses sqrt(n_samples).
        reference : AnnotatedDataset or Embeddings or None, default None
            Optional reference dataset for relative prioritization.

        Returns
        -------
        Prioritize
            Configured instance ready for policy selection and evaluation.

        Examples
        --------
        >>> result = Prioritize.knn(encoder, k=10).hard_first().evaluate(dataset)
        >>> result = Prioritize.knn(encoder, k=5).easy_first().evaluate(dataset)
        """
        return cls(encoder=encoder, method="knn", k=k, reference=reference)

    @classmethod
    def kmeans_distance(
        cls,
        encoder: EmbeddingEncoder,
        c: int | None = None,
        n_init: int | Literal["auto"] = "auto",
        reference: AnnotatedDataset[Any] | Embeddings | None = None,
    ) -> Self:
        """
        Create a Prioritize instance using K-means distance method.

        Ranks samples by distance to their assigned cluster centers.

        Parameters
        ----------
        encoder : EmbeddingEncoder
            Encoder to use for extracting embeddings from data.
        c : int or None, default None
            Number of clusters. If None, uses sqrt(n_samples).
        n_init : int or "auto", default "auto"
            Number of K-means initializations.
        reference : AnnotatedDataset or Embeddings or None, default None
            Optional reference dataset for relative prioritization.

        Returns
        -------
        Prioritize
            Configured instance ready for policy selection and evaluation.

        Examples
        --------
        >>> result = Prioritize.kmeans_distance(encoder, c=15).stratified().evaluate(dataset)
        """
        return cls(encoder=encoder, method="kmeans_distance", c=c, n_init=n_init, reference=reference)

    @classmethod
    def kmeans_complexity(
        cls,
        encoder: EmbeddingEncoder,
        c: int | None = None,
        n_init: int | Literal["auto"] = "auto",
        reference: AnnotatedDataset[Any] | Embeddings | None = None,
    ) -> Self:
        """
        Create a Prioritize instance using K-means complexity method.

        Uses weighted sampling based on intra/inter-cluster distances.

        Note: This method does not produce scores, so "stratified" policy
        is not available.

        Parameters
        ----------
        encoder : EmbeddingEncoder
            Encoder to use for extracting embeddings from data.
        c : int or None, default None
            Number of clusters. If None, uses sqrt(n_samples).
        n_init : int or "auto", default "auto"
            Number of K-means initializations.
        reference : AnnotatedDataset or Embeddings or None, default None
            Optional reference dataset for relative prioritization.

        Returns
        -------
        Prioritize
            Configured instance ready for policy selection and evaluation.

        Examples
        --------
        >>> result = Prioritize.kmeans_complexity(encoder, c=10).hard_first().evaluate(dataset)
        """
        return cls(encoder=encoder, method="kmeans_complexity", c=c, n_init=n_init, reference=reference)

    # ==================== Policy Methods (return new instances) ====================

    def easy_first(self) -> Self:
        """
        Return a new instance configured with easy_first policy.

        Selects easy/prototypical samples first (ascending order of difficulty).

        Returns
        -------
        Prioritize
            New instance with policy set to "easy_first".

        Examples
        --------
        >>> result = Prioritize.knn(encoder, k=5).easy_first().evaluate(dataset)
        """
        return self.__class__(
            config=self.config.model_copy(update={"policy": "easy_first"}),
            reference=self._reference,
        )

    def hard_first(self) -> Self:
        """
        Return a new instance configured with hard_first policy.

        Selects hard/challenging samples first (descending order of difficulty).

        Returns
        -------
        Prioritize
            New instance with policy set to "hard_first".

        Examples
        --------
        >>> result = Prioritize.knn(encoder, k=5).hard_first().evaluate(dataset)
        """
        return self.__class__(
            config=self.config.model_copy(update={"policy": "hard_first"}),
            reference=self._reference,
        )

    def stratified(self, num_bins: int = DEFAULT_PRIORITIZE_NUM_BINS) -> Self:
        """
        Return a new instance configured with stratified policy.

        Balances selection across different difficulty levels by binning scores
        and sampling uniformly from bins.

        Note: Only available with methods that produce scores ("knn", "kmeans_distance").

        Parameters
        ----------
        num_bins : int, default 50
            Number of bins for stratification.

        Returns
        -------
        Prioritize
            New instance with policy set to "stratified".

        Examples
        --------
        >>> result = Prioritize.knn(encoder, k=5).stratified(num_bins=20).evaluate(dataset)
        """
        return self.__class__(
            config=self.config.model_copy(update={"policy": "stratified", "num_bins": num_bins}),
            reference=self._reference,
        )

    def class_balanced(self, class_labels: NDArray[np.integer[Any]] | None = None) -> Self:
        """
        Return a new instance configured with class_balance policy.

        Ensures balanced representation across class labels while maintaining
        priority order within each class.

        Parameters
        ----------
        class_labels : NDArray[np.integer] or None, default None
            Class labels for each sample. If None, will be extracted from
            AnnotatedDataset metadata during evaluate().

        Returns
        -------
        Prioritize
            New instance with policy set to "class_balance".

        Examples
        --------
        >>> result = Prioritize.knn(encoder, k=5).class_balanced(class_labels).evaluate(dataset)

        With AnnotatedDataset (labels extracted automatically):

        >>> result = Prioritize.knn(encoder, k=5).class_balanced().evaluate(labeled_data)
        """
        return self.__class__(
            config=self.config.model_copy(update={"policy": "class_balance", "class_labels": class_labels}),
            reference=self._reference,
        )

    @set_metadata(state=["method", "k", "c", "n_init", "policy", "num_bins"])
    def evaluate(
        self,
        dataset: AnnotatedDataset[Any] | Embeddings,
    ) -> PrioritizeOutput:
        """
        Evaluate the dataset and return prioritized indices.

        Uses the configured method and policy to rank samples.

        Parameters
        ----------
        dataset : AnnotatedDataset[Any] | Embeddings
            The incoming dataset to prioritize. Can be either:

            - AnnotatedDataset: Will compute embeddings using the encoder
            - Embeddings: Pre-computed embeddings

        Returns
        -------
        PrioritizeOutput
            Output containing prioritized indices, scores (if available),
            and configuration information.

        Raises
        ------
        ValueError
            If class_labels is None when using class_balance policy with Embeddings.
            If stratified policy is used with kmeans_complexity method.
        TypeError
            If dataset is neither an AnnotatedDataset nor Embeddings.

        Examples
        --------
        Using factory methods:

        >>> result = Prioritize.knn(encoder, k=5).hard_first().evaluate(dataset)

        Using direct instantiation:

        >>> prioritizer = Prioritize(encoder=encoder, method="knn", k=5, policy="hard_first")
        >>> result = prioritizer.evaluate(dataset)
        """
        # Validate stratified + kmeans_complexity combination
        if self.policy == "stratified" and self.method == "kmeans_complexity":
            raise ValueError(
                "stratified policy is not available with kmeans_complexity method "
                "(kmeans_complexity does not produce scores)"
            )

        # Check if dataset is Embeddings (pre-computed) or AnnotatedDataset
        if isinstance(dataset, Embeddings):
            # Pre-computed embeddings - use directly
            self.embeddings = dataset
        else:
            # Assume dataset is an AnnotatedDataset - compute embeddings
            try:
                self.embeddings = Embeddings(dataset, encoder=self.encoder)
            except Exception as e:
                raise TypeError(
                    f"dataset must be either an AnnotatedDataset or Embeddings, but got {type(dataset).__name__}"
                ) from e

        if self._reference is None:
            self.ref_embeddings = None
        elif isinstance(self._reference, Embeddings):
            # Pre-computed embeddings
            self.ref_embeddings = self._reference
        else:
            # Reference dataset - compute embeddings
            self.ref_embeddings = Embeddings(self._reference, encoder=self.encoder)

        # Extract class_labels if not provided in configuration
        class_labels = self.class_labels
        if self.policy == "class_balance" and class_labels is None:
            if isinstance(dataset, AnnotatedDataset):
                self.metadata = _Metadata(dataset)
                class_labels = self.metadata.class_labels
            else:
                raise ValueError(
                    "class_labels must be provided when using class_balance policy with Embeddings dataset"
                )

        # Perform ranking and reranking
        result = self._rank_and_rerank(class_labels)

        return PrioritizeOutput(
            indices=result["indices"],
            scores=result["scores"],
            method=result["method"],
            policy=result["policy"],
        )

    def _rank_and_rerank(self, class_labels: NDArray[np.integer[Any]] | None) -> Any:
        """Helper method to perform ranking and reranking using configured method/policy."""
        # Step 1: Perform initial ranking (always returns easy_first)
        embeddings_array = np.asarray(self.embeddings)
        reference_array = None if self.ref_embeddings is None else np.asarray(self.ref_embeddings)

        if self.method == "knn":
            result = rank_knn(embeddings_array, k=self.k, reference=reference_array)
        elif self.method == "kmeans_distance":
            result = rank_kmeans_distance(
                embeddings_array,
                c=self.c,
                n_init=self.n_init,
                reference=reference_array,
            )
        elif self.method == "kmeans_complexity":
            result = rank_kmeans_complexity(
                embeddings_array,
                c=self.c,
                n_init=self.n_init,
                reference=reference_array,
            )
        else:
            raise ValueError(f"Invalid method: {self.method}")

        # Step 2: Apply reranking policy
        if self.policy == "easy_first":
            # Already in easy_first order, no reranking needed
            pass
        elif self.policy == "hard_first":
            result = rerank_hard_first(result)
        elif self.policy == "stratified":
            result = rerank_stratified(result, num_bins=self.num_bins)
        elif self.policy == "class_balance":
            if class_labels is None:
                raise ValueError("class_labels is required for class_balance policy")
            result = rerank_class_balance(result, class_labels=class_labels)
        else:
            raise ValueError(f"Invalid policy: {self.policy}")

        return result
