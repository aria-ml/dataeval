"""
Feature extractors for drift detection.

This module provides feature extraction implementations that convert various
data types (datasets, embeddings, metadata) into arrays suitable for drift detection.
"""

from __future__ import annotations

__all__ = []


from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np
import torch
from scipy.special import softmax
from scipy.stats import entropy

from dataeval import Embeddings, Metadata
from dataeval.config import get_batch_size, get_device
from dataeval.core import calculate
from dataeval.flags import ImageStats
from dataeval.protocols import Array, DeviceLike, Transform
from dataeval.utils._array import as_numpy
from dataeval.utils._predict import predict


def _classifier_uncertainty(
    preds: Array,
    preds_type: Literal["probs", "logits"] = "probs",
) -> torch.Tensor:
    """Convert model predictions to uncertainty scores using entropy.

    Computes prediction uncertainty as the entropy of the predicted class
    probability distribution. Higher entropy indicates greater model uncertainty,
    with maximum uncertainty at uniform distributions and minimum at confident
    single-class predictions.

    Parameters
    ----------
    preds : Array
        Model predictions for a batch of instances. For "probs" type, should
        contain class probabilities that sum to 1 across the last dimension.
        For "logits" type, contains raw model outputs before softmax.
    preds_type : "probs" or "logits", default "probs"
        Type of prediction values. "probs" expects probabilities in [0,1] that
        sum to 1. "logits" expects raw outputs in [-inf,inf] and applies softmax.
        Default "probs" assumes model outputs normalized probabilities.

    Returns
    -------
    torch.Tensor
        Uncertainty scores for each instance with shape (n_samples, 1).
        Values are always >= 0, with higher values indicating greater uncertainty.

    Raises
    ------
    ValueError
        If preds_type is "probs" but probabilities don't sum to 1 within tolerance.
    NotImplementedError
        If preds_type is not "probs" or "logits".

    Notes
    -----
    Uncertainty is computed as Shannon entropy: -sum(p * log(p)) where p are
    the predicted class probabilities. This provides a principled measure of
    model confidence that is widely used in uncertainty quantification.
    """
    preds_np = as_numpy(preds)
    if preds_type == "probs":
        if np.abs(1 - np.nan_to_num(np.nansum(preds_np, axis=-1))).mean() > 1e-6:
            raise ValueError("Probabilities across labels should sum to 1")
        probs = preds_np
    elif preds_type == "logits":
        probs = softmax(preds_np, axis=-1)
    else:
        raise NotImplementedError("Only prediction types 'probs' and 'logits' supported.")

    uncertainties = cast(np.ndarray, entropy(probs, axis=-1))
    return torch.as_tensor(uncertainties[:, None])


class EmbeddingsFeatureExtractor:
    """Extract embeddings from datasets for drift detection.

    This class implements the :class:`~dataeval.protocols.FeatureExtractor` protocol
    for use with drift detectors. It converts raw datasets into embeddings using a
    neural network model, with support for reusing pre-computed embeddings to avoid
    redundant computation.

    The extractor maintains state to cache reference embeddings and avoid recomputation
    when the same dataset is passed multiple times (e.g., for reference data initialization
    vs. actual drift detection).

    Parameters
    ----------
    model : torch.nn.Module or None, default None
        Model to extract embeddings with. When None and embeddings is provided,
        uses the model from the embeddings object.
    batch_size : int or None, default None
        Batch size for processing images through the model. Uses global batch_size if
        not provided.
    transforms : Transform or Sequence[Transform] or None, default None
        Preprocessing transforms to apply before model inference.
    layer_name : str or None, default None
        Network layer from which to extract embeddings. When None, uses model output.
    use_output : bool, default True
        If True, captures output tensors from layer_name. If False, captures input tensors.
    device : DeviceLike or None, default None
        Hardware device for computation. When None, uses DataEval's configured device.
    embeddings : Embeddings or None, default None
        Pre-computed Embeddings object to reuse. When provided, avoids recomputation
        for the same dataset. This is useful when you've already computed embeddings
        and want to use them for drift detection without redundant processing.

    Attributes
    ----------
    model : torch.nn.Module or None
        The embedding model.
    batch_size : int
        Batch size for inference.
    device : torch.device
        Hardware device for computation.

    Example
    -------
    Basic usage with a dataset:

    >>> import numpy as np
    >>> import torch.nn as nn
    >>> from dataeval.shift import DriftUnivariate, EmbeddingsFeatureExtractor
    >>>
    >>> # Create dummy data
    >>> train_data = np.random.randn(100, 16).astype(np.float32)
    >>> test_data = np.random.randn(50, 16).astype(np.float32)
    >>>
    >>> # Create feature extractor
    >>> model = nn.Sequential(nn.Linear(16, 128), nn.ReLU(), nn.Linear(128, 64))
    >>> embeddings_extractor = EmbeddingsFeatureExtractor(model=model, batch_size=32)
    >>>
    >>> # Use with drift detector on raw datasets
    >>> drift_detector = DriftUnivariate(
    ...     data=train_data,
    ...     method="ks",
    ...     feature_extractor=embeddings_extractor,
    ... )
    >>> result = drift_detector.predict(test_data)
    >>> print(f"Drift detected: {result.drifted}")
    Drift detected: False

    Reusing pre-computed embeddings:

    >>> from dataeval import Embeddings
    >>>
    >>> # Use ExampleDataset for structured data (1x4x4 images for simple example)
    >>> train_ds = ExampleDataset(100, image_shape=(1, 4, 4), n_classes=10, seed=42)
    >>> model_emb = nn.Sequential(nn.Flatten(), nn.Linear(16, 128), nn.ReLU(), nn.Linear(128, 64))
    >>>
    >>> # Compute embeddings once
    >>> train_embeddings = Embeddings(train_ds, batch_size=32, model=model_emb).compute()
    >>>
    >>> # Reuse embeddings with drift detector
    >>> embeddings_extractor = EmbeddingsFeatureExtractor(embeddings=train_embeddings)
    >>> drift_detector = DriftUnivariate(
    ...     data=train_ds,
    ...     method="ks",
    ...     feature_extractor=embeddings_extractor,
    ... )

    Notes
    -----
    The extractor caches a reference to the dataset used during initialization
    to avoid redundant embedding computation when the same dataset is passed
    multiple times (common in reference data initialization).

    See Also
    --------
    Embeddings : Underlying embeddings computation class
    DriftUnivariate : Univariate drift detection with multiple statistical tests
    """

    batch_size: int
    device: DeviceLike
    transforms: Transform[torch.Tensor] | Sequence[Transform[torch.Tensor]] | None
    layer_name: str | None
    use_output: bool

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        batch_size: int | None = None,
        transforms: Transform[torch.Tensor] | Sequence[Transform[torch.Tensor]] | None = None,
        layer_name: str | None = None,
        use_output: bool = True,
        device: DeviceLike | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        # If embeddings provided, extract configuration from it
        if embeddings is not None:
            if not isinstance(embeddings, Embeddings):
                raise TypeError(f"embeddings must be an Embeddings instance, got {type(embeddings)}")

            self._reference_embeddings = embeddings
            self.model = embeddings._model if model is None else model
            self.batch_size = embeddings.batch_size if batch_size is None else batch_size
            self.transforms = embeddings._transforms if transforms is None else transforms
            self.layer_name = embeddings.layer_name if layer_name is None else layer_name
            self.use_output = embeddings.use_output if use_output else use_output
            self.device = embeddings.device if device is None else get_device(device)
            # Track the dataset used in the pre-computed embeddings
            self._reference_dataset_id = id(embeddings._dataset)
        else:
            if model is None:
                raise ValueError("Either model or embeddings must be provided")

            self._reference_embeddings = None
            self.model = model
            self.batch_size = get_batch_size(batch_size)
            self.transforms = transforms
            self.layer_name = layer_name
            self.use_output = use_output
            self.device = get_device(device)
            self._reference_dataset_id = None

        # Cache for avoiding re-extraction on same dataset
        self._dataset_cache: dict[int, np.ndarray] = {}

    def __call__(self, data: Any) -> Array:
        """Extract embeddings from dataset or return cached embeddings.

        Parameters
        ----------
        data : Dataset or Any
            Input dataset to extract embeddings from. Can be a Dataset or
            any data type that the underlying Embeddings class supports.

        Returns
        -------
        Array
            Embeddings array of shape (n_samples, embedding_dim).
        """
        dataset_id = id(data)

        # Check if we've already processed this exact dataset object
        if dataset_id in self._dataset_cache:
            return self._dataset_cache[dataset_id]

        # If this is the reference dataset and we have pre-computed embeddings, use them
        if self._reference_embeddings is not None and dataset_id == self._reference_dataset_id:
            # Force computation if not already done and cache result
            result = self._reference_embeddings.compute()[:]
            self._dataset_cache[dataset_id] = result
            return result

        # Need to compute new embeddings for this dataset
        embeddings = Embeddings(
            dataset=data,
            batch_size=self.batch_size,
            transforms=self.transforms,
            model=self.model,
            layer_name=self.layer_name,
            use_output=self.use_output,
            device=self.device,
        )

        # Compute and cache
        result = embeddings.compute()[:]
        self._dataset_cache[dataset_id] = result
        return result

    def __repr__(self) -> str:
        """Return string representation of the extractor."""
        model_name = self.model.__class__.__name__ if self.model is not None else "None"
        return f"{self.__class__.__name__}(model={model_name}, batch_size={self.batch_size}, device={self.device})"


class MetadataFeatureExtractor:
    """Extract metadata factors from datasets for drift detection.

    This class implements the :class:`~dataeval.protocols.FeatureExtractor` protocol
    for use with drift detectors. It extracts and bins metadata factors from annotated
    datasets, with support for reusing pre-computed metadata to avoid redundant processing.

    The extractor maintains state to cache reference metadata and avoid recomputation
    when the same dataset is passed multiple times.

    Parameters
    ----------
    continuous_factor_bins : Mapping[str, int | Sequence[float]] or None, default None
        Binning configuration for continuous factors. Maps factor names to either
        the number of bins or explicit bin edges.
    auto_bin_method : {"uniform_width", "uniform_count", "clusters"}, default "uniform_width"
        Automatic binning strategy for continuous factors without explicit bins.
    exclude : Sequence[str] or None, default None
        Factor names to exclude from processing.
    include : Sequence[str] or None, default None
        Factor names to include in processing.
    use_binned : bool, default True
        If True, returns binned_data (discrete integers). If False, returns factor_data
        (original continuous/categorical values).
    metadata : Metadata or None, default None
        Pre-computed Metadata object to reuse. When provided, avoids recomputation
        for the same dataset. This is useful when you've already processed metadata
        and want to use it for drift detection without redundant binning.

    Attributes
    ----------
    continuous_factor_bins : Mapping[str, int | Sequence[float]]
        Binning configuration for continuous factors.
    auto_bin_method : {"uniform_width", "uniform_count", "clusters"}
        Automatic binning strategy.
    use_binned : bool
        Whether to return binned or raw factor data.

    Example
    -------
    Basic usage with a dataset:

    >>> from dataeval.flags import ImageStats
    >>> from dataeval.shift import DriftUnivariate, MetadataFeatureExtractor
    >>>
    >>> # Use ExampleDataset from conftest
    >>> train_dataset = ExampleDataset(100, seed=42)
    >>> test_dataset = ExampleDataset(50, seed=43)
    >>>
    >>> # Create metadata extractor
    >>> metadata_extractor = MetadataFeatureExtractor(
    ...     continuous_factor_bins={"brightness": 10, "contrast": 10},
    ...     use_binned=False,
    ...     add_stats=ImageStats.VISUAL_BRIGHTNESS | ImageStats.VISUAL_CONTRAST,
    ... )
    >>>
    >>> # Use with drift detector on raw datasets
    >>> drift_detector = DriftUnivariate(
    ...     data=train_dataset,
    ...     method="ks",
    ...     feature_extractor=metadata_extractor,
    ... )
    >>> result = drift_detector.predict(test_dataset)
    >>> print(f"Drift detected: {result.drifted}")
    Drift detected: True

    Reusing pre-computed metadata:

    >>> from dataeval import Metadata
    >>> from dataeval.core import calculate
    >>> from dataeval.flags import ImageStats
    >>>
    >>> # Create dataset for metadata extraction
    >>> train_ds_meta = ExampleDataset(100, seed=42)
    >>>
    >>> # Compute metadata once with additional image statistics
    >>> stats_flags = ImageStats.VISUAL_BRIGHTNESS | ImageStats.VISUAL_CONTRAST
    >>> stats = calculate(train_ds_meta, stats=stats_flags)
    >>> train_metadata = Metadata(
    ...     train_ds_meta,
    ...     continuous_factor_bins={"brightness": 10, "contrast": 10},
    ... )
    >>> train_metadata.add_factors(stats["stats"])
    >>>
    >>> # Reuse metadata with drift detector
    >>> metadata_extractor = MetadataFeatureExtractor(metadata=train_metadata, use_binned=False, add_stats=stats_flags)
    >>> drift_detector = DriftUnivariate(
    ...     data=train_ds_meta,
    ...     method="ks",
    ...     feature_extractor=metadata_extractor,
    ... )

    Notes
    -----
    The extractor caches a reference to the dataset used during initialization
    to avoid redundant metadata processing when the same dataset is passed
    multiple times (common in reference data initialization).

    Binning configuration is preserved when reusing metadata to ensure consistent
    discretization across reference and test data.

    See Also
    --------
    Metadata : Underlying metadata processing class
    ImageStats : Supported image statistics
    DriftUnivariate : Univariate drift detection with multiple statistical tests
    """

    continuous_factor_bins: Mapping[str, int | Sequence[float]] | None
    auto_bin_method: Literal["uniform_width", "uniform_count", "clusters"]
    use_binned: bool
    add_stats: ImageStats | None
    exclude: set[str]
    include: set[str]

    def __init__(
        self,
        continuous_factor_bins: Mapping[str, int | Sequence[float]] | None = None,
        auto_bin_method: Literal["uniform_width", "uniform_count", "clusters"] | None = None,
        exclude: Sequence[str] | None = None,
        include: Sequence[str] | None = None,
        use_binned: bool = True,
        add_stats: ImageStats | None = None,
        metadata: Metadata | None = None,
    ) -> None:
        # If metadata provided, extract configuration from it
        if metadata is not None:
            if not isinstance(metadata, Metadata):
                raise TypeError(f"metadata must be a Metadata instance, got {type(metadata)}")

            self._reference_metadata = metadata
            self.continuous_factor_bins = (
                metadata.continuous_factor_bins if continuous_factor_bins is None else continuous_factor_bins
            )
            self.auto_bin_method = metadata.auto_bin_method if auto_bin_method is None else "uniform_width"
            self.exclude = metadata.exclude if exclude is None else set(exclude)
            self.include = metadata.include if include is None else set(include)
            self.use_binned = use_binned
            self.add_stats = add_stats
            # Track the dataset used in the pre-computed metadata
            self._reference_dataset_id = id(metadata._dataset)
        else:
            self._reference_metadata = None
            self.continuous_factor_bins = continuous_factor_bins
            self.auto_bin_method: Literal["uniform_width", "uniform_count", "clusters"] = (
                "uniform_width" if auto_bin_method is None else auto_bin_method
            )
            self.exclude = set(exclude or ())
            self.include = set(include or ())
            self.use_binned: bool = use_binned
            self.add_stats = add_stats
            self._reference_dataset_id = None

        # Cache for avoiding re-extraction on same dataset
        self._dataset_cache: dict[int, np.ndarray] = {}

    def __call__(self, data: Any) -> Array:
        """Extract metadata factors from dataset or return cached factors.

        Parameters
        ----------
        data : AnnotatedDataset
            Input dataset with metadata to extract factors from.

        Returns
        -------
        Array
            Metadata factors array of shape (n_samples, n_factors).
        """
        dataset_id = id(data)

        # Check if we've already processed this exact dataset object
        if dataset_id in self._dataset_cache:
            return self._dataset_cache[dataset_id]

        # If this is the reference dataset and we have pre-computed metadata, use it
        if self._reference_metadata is not None and dataset_id == self._reference_dataset_id:
            # Get the appropriate data representation
            result = self._reference_metadata.binned_data if self.use_binned else self._reference_metadata.factor_data
            self._dataset_cache[dataset_id] = result
            return result

        # Need to compute new metadata for this dataset
        metadata = Metadata(
            dataset=data,
            continuous_factor_bins=self.continuous_factor_bins,
            auto_bin_method=self.auto_bin_method,
            exclude=list(self.exclude) if self.exclude else None,
            include=list(self.include) if self.include else None,
        )

        # Add additional statistics if requested
        if self.add_stats is not None:
            stats = calculate(data, stats=self.add_stats)
            metadata.add_factors(stats["stats"])

        # Get the appropriate data representation and cache
        result = metadata.binned_data if self.use_binned else metadata.factor_data
        self._dataset_cache[dataset_id] = result
        return result

    def __repr__(self) -> str:
        """Return string representation of the extractor."""
        return f"{self.__class__.__name__}(use_binned={self.use_binned}, auto_bin_method={self.auto_bin_method!r})"


class UncertaintyFeatureExtractor:
    """Feature extractor that converts data to model uncertainty scores.

    This class implements the :class:`~dataeval.protocols.FeatureExtractor` protocol
    for use with drift detectors (e.g., :class:`~dataeval.shift.DriftUnivariate`).
    It computes prediction uncertainty (entropy) from a classification model.

    Uncertainty-based drift detection monitors changes in model confidence rather
    than raw input features. This approach is particularly effective for detecting
    drift that affects model performance even when input statistics remain similar,
    such as out-of-domain samples or adversarial examples.

    Parameters
    ----------
    model : torch.nn.Module
        Classification model to compute predictions and uncertainties.
        Should output class probabilities or logits.
    preds_type : "probs" or "logits", default "probs"
        Format of model outputs. "probs" expects normalized probabilities
        summing to 1. "logits" expects raw model outputs and applies softmax.
    batch_size : int, default 32
        Batch size for model inference. Larger batches improve GPU
        utilization but require more memory.
    transforms : Transform or Sequence[Transform] or None, default None
        Preprocessing transforms to apply before model inference. Should match
        preprocessing used during model training for consistent predictions.
    device : DeviceLike or None, default None
        Hardware device for computation. When None, uses DataEval's
        configured device or PyTorch's default.

    Attributes
    ----------
    model : torch.nn.Module
        The classification model used for predictions.
    preds_type : {"probs", "logits"}
        Format of model outputs.
    batch_size : int
        Batch size for inference.
    device : torch.device
        Hardware device for computation.

    Example
    -------
    Basic usage with DriftUnivariate

    >>> import numpy as np
    >>> import torch.nn as nn
    >>> from dataeval.shift import DriftUnivariate, UncertaintyFeatureExtractor
    >>>
    >>> # Create dummy datasets
    >>> train_dataset = np.random.randn(100, 16).astype(np.float32)
    >>> test_dataset = np.random.randn(20, 16).astype(np.float32)
    >>>
    >>> # Create a simple model
    >>> model = nn.Sequential(nn.Linear(16, 10), nn.Softmax(dim=-1))
    >>>
    >>> # Create uncertainty feature extractor
    >>> uncertainty_extractor = UncertaintyFeatureExtractor(model=model, preds_type="probs", batch_size=32)
    >>>
    >>> # Use with DriftUnivariate for uncertainty-based drift detection
    >>> drift_detector = DriftUnivariate(train_dataset, method="ks", feature_extractor=uncertainty_extractor)
    >>>
    >>> # Detect drift on new data
    >>> result = drift_detector.predict(test_dataset)
    >>> print(f"Drift detected: {result.drifted}")
    Drift detected: False

    With data preprocessing transforms

    >>> import torch
    >>>
    >>> # Create new datasets and model for this example
    >>> train_dataset = np.random.randn(100, 16).astype(np.float32)
    >>> test_dataset = np.random.randn(20, 16).astype(np.float32)
    >>> model = nn.Sequential(nn.Linear(16, 10), nn.Softmax(dim=-1))
    >>>
    >>> # Simple transform (no normalization needed for this dummy data)
    >>> transforms = lambda x: x.float() if not x.is_floating_point() else x
    >>>
    >>> uncertainty_extractor = UncertaintyFeatureExtractor(model=model, transforms=transforms, device="cpu")
    >>>
    >>> drift_detector = DriftUnivariate(train_dataset, method="ks", feature_extractor=uncertainty_extractor)

    Using different statistical methods

    >>> # Create datasets and model for this example
    >>> train_dataset = np.random.randn(100, 16).astype(np.float32)
    >>> test_dataset = np.random.randn(20, 16).astype(np.float32)
    >>> model = nn.Sequential(nn.Linear(16, 10), nn.Softmax(dim=-1))
    >>> uncertainty_extractor = UncertaintyFeatureExtractor(model=model, preds_type="probs", batch_size=32)
    >>>
    >>> # Use Cramér-von Mises test instead of Kolmogorov-Smirnov
    >>> drift_detector = DriftUnivariate(
    ...     train_dataset,
    ...     method="cvm",  # More sensitive to overall distributional changes
    ...     feature_extractor=uncertainty_extractor,
    ... )
    >>>
    >>> # Or use Mann-Whitney U test for robust median shift detection
    >>> drift_detector = DriftUnivariate(
    ...     train_dataset,
    ...     method="mwu",  # Robust to outliers
    ...     feature_extractor=uncertainty_extractor,
    ... )

    Notes
    -----
    The uncertainty extractor computes Shannon entropy: -sum(p * log(p)) where p
    are the predicted class probabilities. Higher entropy indicates greater model
    uncertainty.

    This approach works best with well-calibrated models trained on representative
    data. Poorly calibrated models may produce misleading uncertainty estimates
    that don't reliably indicate data quality issues.

    Uncertainty-based drift detection is complementary to feature-based methods
    and can detect semantic drift (changes in data meaning) that may not be
    apparent in raw feature statistics.

    See Also
    --------
    DriftUnivariate : Univariate drift detection with multiple statistical tests
    """

    def __init__(
        self,
        model: torch.nn.Module,
        preds_type: Literal["probs", "logits"] = "probs",
        batch_size: int = 32,
        transforms: Transform[torch.Tensor] | Sequence[Transform[torch.Tensor]] | None = None,
        device: DeviceLike | None = None,
    ) -> None:
        self.model = model
        self.preds_type: Literal["probs", "logits"] = preds_type
        self.batch_size: int = batch_size
        self.device: DeviceLike = get_device(device)
        self._transforms: list[Transform[torch.Tensor]] = (
            [] if transforms is None else [transforms] if isinstance(transforms, Transform) else list(transforms)
        )

    def _apply_transforms(self, x: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing transforms to input data."""
        for transform in self._transforms:
            x = transform(x)
        return x

    def __call__(self, data: Any) -> Array:
        """Extract uncertainty features from raw data.

        Parameters
        ----------
        data : Any
            Raw input data to compute uncertainties for.

        Returns
        -------
        Array
            Uncertainty scores as numpy array of shape (n_samples, 1).
        """
        preds = predict(data, self.model, self.device, self.batch_size, self._apply_transforms)
        uncertainties = _classifier_uncertainty(preds, self.preds_type)
        return uncertainties.cpu().numpy()

    def __repr__(self) -> str:
        """Return string representation of the extractor."""
        return (
            f"{self.__class__.__name__}("
            f"model={self.model.__class__.__name__}, "
            f"preds_type={self.preds_type!r}, "
            f"batch_size={self.batch_size}, "
            f"device={self.device})"
        )
