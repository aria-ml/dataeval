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
from dataeval.config import DeviceLike, get_device
from dataeval.core._rank import rank_kmeans_complexity, rank_kmeans_distance, rank_knn
from dataeval.core._rerank import rerank_class_balance, rerank_hard_first, rerank_stratified
from dataeval.protocols import AnnotatedDataset, EmbeddingModel, Metadata
from dataeval.types import Output, set_metadata


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


class Prioritize:
    """
    Prioritize dataset samples based on their position in the embedding space.

    This class uses a builder pattern to configure ranking method and policy,
    then evaluates datasets to produce prioritized sample orderings.

    Parameters
    ----------
    model : EmbeddingModel
        Model to use for encoding data.
    reference : AnnotatedDataset[Any] | Embeddings | None, default None
        Optional reference dataset or pre-computed embeddings. When provided,
        incoming datasets will be prioritized relative to this reference set.
        Useful for active learning (reference = labeled data) or quality
        filtering (reference = high-quality corpus).
    batch_size : int | None, default None
        Default batch size to use when encoding data. Can be overridden in evaluate().
    device : DeviceLike | None, default None
        Default device to use for encoding data. Can be overridden in evaluate().

    See Also
    --------
    :class:`~dataeval.quality.Outliers`
    :class:`~dataeval.selection.Indices`
    :func:`dataeval.core.rank_knn`
    :func:`dataeval.core.rank_kmeans_distance`
    :func:`dataeval.core.rank_kmeans_complexity`

    Examples
    --------
    Basic prioritization using builder pattern:

    >>> from dataeval.quality import Prioritize
    >>> prioritizer = Prioritize(model)
    >>>
    >>> # Configure method and policy, then evaluate
    >>> result = prioritizer.with_knn(k=10).hard_first().evaluate(unlabeled_data)

    Different policies:

    >>> # Easy samples first
    >>> result = prioritizer.with_knn(k=5).easy_first().evaluate(unlabeled_data)
    >>>
    >>> # Stratified sampling
    >>> result = prioritizer.with_knn(k=5).stratified(num_bins=20).evaluate(unlabeled_data)
    >>>
    >>> # Class-balanced selection
    >>> result = prioritizer.with_kmeans_distance(c=10).class_balanced(class_labels).evaluate(unlabeled_data)

    Reconfigure and reuse:

    >>> # Can reconfigure the same instance
    >>> result = prioritizer.with_kmeans_complexity(c=15).easy_first().evaluate(unlabeled_data)

    Active learning with reference:

    >>> # Initialize with labeled data as reference
    >>> prioritizer = Prioritize(model, reference=labeled_data)
    >>> result = prioritizer.with_knn(k=10).hard_first().evaluate(reference_data)
    """

    def __init__(
        self,
        model: EmbeddingModel,
        reference: AnnotatedDataset[Any] | Embeddings | None = None,
        batch_size: int | None = None,
        device: DeviceLike | None = None,
    ) -> None:
        self.model = model
        self._reference = reference
        self.batch_size = batch_size
        self.device: DeviceLike = get_device(device)

        # Internal state populated during evaluate
        self.embeddings: Embeddings | None = None
        self.ref_embeddings: Embeddings | None = None
        self.metadata: Metadata | None = None

        # Configuration state
        self._method_name: Literal["knn", "kmeans_distance", "kmeans_complexity"] | None = None
        self._method_params: dict[str, Any] = {}
        self._policy_name: Literal["hard_first", "easy_first", "stratified", "class_balance"] | None = None
        self._policy_params: dict[str, Any] = {}

    # Method configuration (returns self for chaining)

    def with_knn(self, k: int | None = None) -> Self:
        """
        Configure k-nearest neighbors ranking method.

        Parameters
        ----------
        k : int | None, default None
            Number of nearest neighbors. If None, uses sqrt(n_samples).

        Returns
        -------
        Prioritize
            Self for method chaining.

        Examples
        --------
        >>> prioritizer = Prioritize(model)
        >>> result = prioritizer.with_knn(k=5).hard_first().evaluate(unlabeled_data)
        """
        self._method_name = "knn"
        self._method_params = {"k": k}
        return self

    def with_kmeans_distance(
        self,
        c: int | None = None,
        n_init: int | Literal["auto"] = "auto",
    ) -> Self:
        """
        Configure K-means distance ranking method.

        Ranks samples by distance to assigned cluster centers.

        Parameters
        ----------
        c : int | None, default None
            Number of clusters. If None, uses sqrt(n_samples).
        n_init : int | "auto", default "auto"
            Number of K-means initializations.

        Returns
        -------
        Prioritize
            Self for method chaining.

        Examples
        --------
        >>> prioritizer = Prioritize(model)
        >>> result = prioritizer.with_kmeans_distance(c=10).easy_first().evaluate(unlabeled_data)
        """
        self._method_name = "kmeans_distance"
        self._method_params = {"c": c, "n_init": n_init}
        return self

    def with_kmeans_complexity(
        self,
        c: int | None = None,
        n_init: int | Literal["auto"] = "auto",
    ) -> Self:
        """
        Configure K-means complexity ranking method.

        Uses weighted sampling based on intra/inter-cluster distances.
        Note: This method does not produce scores, so stratified() policy is not available.

        Parameters
        ----------
        c : int | None, default None
            Number of clusters. If None, uses sqrt(n_samples).
        n_init : int | "auto", default "auto"
            Number of K-means initializations.

        Returns
        -------
        Prioritize
            Self for method chaining.

        Examples
        --------
        >>> prioritizer = Prioritize(model)
        >>> result = prioritizer.with_kmeans_complexity(c=15).hard_first().evaluate(unlabeled_data)
        """
        self._method_name = "kmeans_complexity"
        self._method_params = {"c": c, "n_init": n_init}
        return self

    # Policy configuration (returns self for chaining)

    def easy_first(self) -> Self:
        """
        Configure policy to select easy/prototypical samples first.

        Returns samples in ascending order of difficulty (low distance = easy).

        Returns
        -------
        Prioritize
            Self for method chaining.

        Examples
        --------
        >>> prioritizer = Prioritize(model)
        >>> result = prioritizer.with_knn(k=5).easy_first().evaluate(unlabeled_data)
        """
        self._policy_name = "easy_first"
        self._policy_params = {}
        return self

    def hard_first(self) -> Self:
        """
        Configure policy to select hard/challenging samples first.

        Returns samples in descending order of difficulty (high distance = hard).

        Returns
        -------
        Prioritize
            Self for method chaining.

        Examples
        --------
        >>> prioritizer = Prioritize(model)
        >>> result = prioritizer.with_knn(k=5).hard_first().evaluate(unlabeled_data)
        """
        self._policy_name = "hard_first"
        self._policy_params = {}
        return self

    def stratified(self, num_bins: int = 50) -> Self:
        """
        Configure stratified sampling policy across score bins.

        Balances selection across different difficulty levels by binning scores
        and sampling uniformly from bins. Encourages diversity.

        Note: Only available with methods that produce scores (knn, kmeans_distance).

        Parameters
        ----------
        num_bins : int, default 50
            Number of bins for stratification.

        Returns
        -------
        Prioritize
            Self for method chaining.

        Examples
        --------
        >>> prioritizer = Prioritize(model)
        >>> result = prioritizer.with_knn(k=5).stratified(num_bins=20).evaluate(unlabeled_data)
        """
        self._policy_name = "stratified"
        self._policy_params = {"num_bins": num_bins}
        return self

    def class_balanced(self, class_labels: NDArray[np.integer[Any]] | None = None) -> Self:
        """
        Configure class-balanced selection policy.

        Ensures balanced representation across class labels while maintaining
        priority order within each class.

        Parameters
        ----------
        class_labels : NDArray[np.integer] | None, default None
            Class labels for each sample in the dataset. If None, will be
            extracted from AnnotatedDataset metadata during evaluate().

        Returns
        -------
        Prioritize
            Self for method chaining.

        Examples
        --------
        >>> prioritizer = Prioritize(model)
        >>> result = prioritizer.with_knn(k=5).class_balanced(class_labels).evaluate(unlabeled_data)

        With AnnotatedDataset (labels extracted automatically):

        >>> result = prioritizer.with_knn(k=5).class_balanced().evaluate(labeled_data)
        """
        self._policy_name = "class_balance"
        self._policy_params = {"class_labels": class_labels}
        return self

    @set_metadata
    def evaluate(
        self,
        dataset: AnnotatedDataset[Any] | Embeddings,
        batch_size: int | None = None,
        device: DeviceLike | None = None,
    ) -> PrioritizeOutput:
        """
        Evaluate the dataset and return prioritized indices.

        Uses the configured method and policy to rank samples. Method and policy
        must be configured using the builder methods (with_*, easy_first, etc.)
        before calling evaluate().

        Parameters
        ----------
        dataset : AnnotatedDataset[Any] | Embeddings
            The incoming dataset to prioritize. Can be either:

            - AnnotatedDataset: Will compute embeddings using the model
            - Embeddings: Pre-computed embeddings
        batch_size : int | None, default None
            Batch size for encoding the incoming dataset. If None, uses the value
            from __init__. Only used when dataset is an AnnotatedDataset.
        device : DeviceLike | None, default None
            Device for encoding the incoming dataset. If None, uses the value
            from __init__. Only used when dataset is an AnnotatedDataset.

        Returns
        -------
        PrioritizeOutput
            Output containing prioritized indices, scores (if available),
            and configuration information.

        Raises
        ------
        ValueError
            If method or policy not configured.
            If class_labels is None when using class_balanced policy with Embeddings.
            If stratified policy is used with kmeans_complexity method.
        TypeError
            If dataset is neither an AnnotatedDataset nor Embeddings.

        Examples
        --------
        Basic usage:

        >>> prioritizer = Prioritize(model)
        >>> result = prioritizer.with_knn(k=5).hard_first().evaluate(labeled_data)

        Override encoding parameters:

        >>> result = prioritizer.with_knn(k=10).easy_first().evaluate(labeled_data, batch_size=64)

        Reconfigure and evaluate different dataset:

        >>> result2 = prioritizer.with_kmeans_distance(c=15).stratified().evaluate(reference_data)
        """
        # Validate configuration
        if self._method_name is None:
            raise ValueError(
                "Method not configured. Call with_knn(), with_kmeans_distance(), or with_kmeans_complexity() first."
            )
        if self._policy_name is None:
            raise ValueError(
                "Policy not configured. Call easy_first(), hard_first(), stratified(), or class_balanced() first."
            )

        # Check if dataset is Embeddings (pre-computed) or AnnotatedDataset
        if isinstance(dataset, Embeddings):
            # Pre-computed embeddings - use directly
            self.embeddings = dataset
        else:
            # Assume dataset is an AnnotatedDataset - compute embeddings
            effective_batch_size = batch_size if batch_size is not None else self.batch_size
            effective_device = device if device is not None else self.device

            try:
                self.embeddings = Embeddings(
                    dataset,
                    batch_size=effective_batch_size,
                    model=self.model,
                    device=effective_device,
                )
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
            # Use instance defaults for reference encoding
            self.ref_embeddings = Embeddings(
                self._reference,
                batch_size=self.batch_size,
                model=self.model,
                device=self.device,
            )

        # Extract class_labels if not provided in configuration
        class_labels = self._policy_params.get("class_labels")
        if self._policy_name == "class_balance" and class_labels is None:
            if isinstance(dataset, AnnotatedDataset):
                self.metadata = _Metadata(dataset)
                class_labels = self.metadata.class_labels
            else:
                raise ValueError(
                    "class_labels must be provided when using class_balanced() policy with Embeddings dataset"
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

        if self._method_name == "knn":
            result = rank_knn(embeddings_array, k=self._method_params["k"], reference=reference_array)
        elif self._method_name == "kmeans_distance":
            result = rank_kmeans_distance(
                embeddings_array,
                c=self._method_params["c"],
                n_init=self._method_params["n_init"],
                reference=reference_array,
            )
        elif self._method_name == "kmeans_complexity":
            result = rank_kmeans_complexity(
                embeddings_array,
                c=self._method_params["c"],
                n_init=self._method_params["n_init"],
                reference=reference_array,
            )
        else:
            raise ValueError(f"Invalid method: {self._method_name}")

        # Step 2: Apply reranking policy
        if self._policy_name == "easy_first":
            # Already in easy_first order, no reranking needed
            pass
        elif self._policy_name == "hard_first":
            result = rerank_hard_first(result)
        elif self._policy_name == "stratified":
            result = rerank_stratified(result, num_bins=self._policy_params["num_bins"])
        elif self._policy_name == "class_balance":
            if class_labels is None:
                raise ValueError("class_labels is required for class_balance policy")
            result = rerank_class_balance(result, class_labels=class_labels)
        else:
            raise ValueError(f"Invalid policy: {self._policy_name}")

        return result
