__all__ = []

import logging
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Sized
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray
from typing_extensions import Self

from dataeval.core._bin import bin_data, digitize_data, is_continuous
from dataeval.protocols import (
    AnnotatedDataset,
    Array,
    DatumMetadata,
    FeatureExtractor,
    ObjectDetectionTarget,
    ProgressCallback,
)
from dataeval.types import Array1D
from dataeval.utils.arrays import as_numpy
from dataeval.utils.data import merge_metadata

_logger = logging.getLogger(__name__)


def _binned(name: str) -> str:
    return f"{name}↕"


def _digitized(name: str) -> str:
    return f"{name}#"


@dataclass
class FactorInfo:
    factor_type: Literal["categorical", "continuous", "discrete"]
    is_binned: bool = False
    is_digitized: bool = False


def _to_col(name: str, info: FactorInfo, binned: bool = True) -> str:
    if binned and info.is_binned:
        return _binned(name)
    if info.is_digitized:
        return _digitized(name)
    return name


class Metadata(Array, FeatureExtractor):
    """Collection of binned metadata using Polars DataFrames.

    Processes dataset metadata by automatically binning continuous factors and digitizing
    categorical factors for analysis and visualization workflows.

    This class also implements the :class:`~dataeval.protocols.FeatureExtractor` protocol,
    allowing it to be used directly with drift detectors that accept feature extractors.

    Parameters
    ----------
    dataset : ImageClassificationDataset, ObjectDetectionDataset, or None, default None
        Dataset that provides original targets and metadata for processing. When None,
        creates an unbound instance that can be used as a reusable feature extractor.
        Use :meth:`bind` to attach a dataset later, or pass data directly to :meth:`__call__`.
    continuous_factor_bins : Mapping[str, int | Sequence[float]] | None, default None
        Mapping from continuous factor names to bin counts or explicit bin edges.
        When None, uses automatic discretization.
    auto_bin_method : Literal["uniform_width", "uniform_count", "clusters"], default "uniform_width"
        Binning strategy for continuous factors without explicit bins. Default "uniform_width"
        provides intuitive equal-width intervals for most distributions.
    exclude : Sequence[str] | None, default None
        Factor names to exclude from processing. Cannot be used with `include` parameter.
        When None, processes all available factors.
    include : Sequence[str] | None, default None
        Factor names to include in processing. Cannot be used with `exclude` parameter.
        When None, processes all available factors.

    Raises
    ------
    ValueError
        When both exclude and include parameters are specified simultaneously.

    Example
    -------
    Using as a feature extractor with drift detection:

    >>> from dataeval import Metadata
    >>> from dataeval.shift import DriftUnivariate
    >>>
    >>> # Create reusable extractor (no dataset bound)
    >>> extractor = Metadata(continuous_factor_bins={"brightness": 10})
    >>>
    >>> # Use with drift detector
    >>> drift = DriftUnivariate(data=train_dataset, feature_extractor=extractor)
    >>> result = drift.predict(test_dataset)

    Using with a bound dataset:

    >>> # Create with dataset bound
    >>> metadata = Metadata(train_dataset, continuous_factor_bins={"brightness": 10})
    >>> train_factors = metadata()  # Extract from bound dataset
    >>> test_factors = metadata(test_dataset)  # Extract from new dataset
    """

    def __init__(
        self,
        dataset: AnnotatedDataset[tuple[Any, Any, DatumMetadata]] | None = None,
        *,
        continuous_factor_bins: Mapping[str, int | Sequence[float]] | None = None,
        auto_bin_method: Literal["uniform_width", "uniform_count", "clusters"] = "uniform_width",
        exclude: Sequence[str] | None = None,
        include: Sequence[str] | None = None,
    ) -> None:
        self._class_labels: NDArray[np.intp]
        self._item_indices: NDArray[np.intp]
        self._factors: dict[str, FactorInfo | None]
        self._dropped_factors: dict[str, list[str]]
        self._dataframe: pl.DataFrame
        self._raw: Sequence[Mapping[str, Any]]
        self._has_targets: bool | None = None

        self._is_structured = False
        self._is_binned = False

        self._dataset = dataset
        self._count = len(dataset) if dataset is not None and isinstance(dataset, Sized) else 0
        self._continuous_factor_bins = dict(continuous_factor_bins) if continuous_factor_bins else {}
        self._auto_bin_method: Literal["uniform_width", "uniform_count", "clusters"] = auto_bin_method

        if exclude is not None and include is not None:
            raise ValueError("Filters for `exclude` and `include` are mutually exclusive.")

        self._exclude = set(exclude or ())
        self._include = set(include or ())
        self._target_factors_only = False

    @property
    def is_bound(self) -> bool:
        """Whether this instance is bound to a dataset.

        Returns
        -------
        bool
            True if a dataset is bound, False otherwise.
        """
        return self._dataset is not None

    def bind(self, dataset: AnnotatedDataset[tuple[Any, Any, DatumMetadata]]) -> Self:
        """Bind this instance to a dataset.

        Attaches a dataset to this Metadata instance for metadata extraction.
        Any previously processed metadata is cleared.

        Parameters
        ----------
        dataset : ImageClassificationDataset or ObjectDetectionDataset
            Dataset to bind for metadata extraction.

        Returns
        -------
        Self
            Returns self for method chaining.

        Example
        -------
        >>> from dataeval import Metadata
        >>>
        >>> extractor = Metadata(continuous_factor_bins={"brightness": 10})
        >>> _ = extractor.bind(train_dataset)
        """
        self._dataset = dataset
        self._count = len(dataset) if isinstance(dataset, Sized) else 0
        # Clear cached state
        self._is_structured = False
        self._is_binned = False
        self._has_targets = None
        return self

    def __array__(self) -> NDArray[np.int64]:
        """NumPy array representation of binned metadata.

        Returns
        -------
        NDArray[np.int64]
            Binned metadata as a NumPy array of shape (n_samples, n_factors).

        Notes
        -----
        This property triggers factor binning analysis on first access.
        Use this for interoperability with libraries expecting NumPy arrays.
        """
        return self.factor_data

    def __len__(self) -> int:
        """Number of items in the bound dataset.

        Returns
        -------
        int
            Number of items in the bound dataset.

        Raises
        ------
        ValueError
            If no dataset is bound.
        """
        if self._dataset is None:
            raise ValueError("No dataset bound. Call bind() first.")
        return self._count

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the binned metadata array.

        Returns
        -------
        tuple[int, ...]
            Shape of the binned metadata as (n_samples, n_factors).

        Raises
        ------
        ValueError
            If no dataset is bound.
        """
        if self._dataset is None:
            raise ValueError("No dataset bound. Call bind() first.")
        return (len(self._dataset), len(self.factor_names))

    def __iter__(self) -> Iterator[NDArray[np.int64]]:
        """Iterate over rows of the binned metadata.

        Yields
        ------
        Iterator[NDArray[np.int64]]
            Rows of the binned metadata array, one at a time.

        Raises
        ------
        ValueError
            If no dataset is bound.
        """
        if self._dataset is None:
            raise ValueError("No dataset bound. Call bind() first.")
        yield from self.factor_data

    def __getitem__(self, index: int | str | slice) -> Array:
        """Get binned metadata for specific indices or factors.

        Parameters
        ----------
        index : int, str, or slice
            Index or slice to select specific rows (by integer index)
            or columns (by factor name) from the binned metadata.

        Returns
        -------
        Array

            Binned metadata for the specified indices or factors.

        Raises
        ------
        ValueError
            If no dataset is bound.
        KeyError
            If a specified factor name is not found in the metadata.
        """
        if self._dataset is None:
            raise ValueError("No dataset bound. Call bind() first.")

        data = self.factor_data

        if isinstance(index, int):
            return data[index]
        if isinstance(index, str):
            if index not in self.factor_names:
                raise KeyError(f"Factor '{index}' not found in metadata.")
            col_index = self.factor_names.index(index)
            return data[:, col_index]
        if isinstance(index, slice):
            return data[index]
        raise TypeError("Index must be an int, str, or slice.")

    def new(self, dataset: AnnotatedDataset[tuple[Any, Any, DatumMetadata]]) -> Self:
        """Create new Metadata instance with a different dataset.

        Generate a new Metadata object using the same configuration
        but with a different dataset.

        Parameters
        ----------
        dataset : ImageClassificationDataset or ObjectDetectionDataset
            Dataset that provides metadata for the new Metadata instance.

        Returns
        -------
        Metadata
            New Metadata object configured identically to the current instance.
        """
        return self.__class__(
            dataset,
            continuous_factor_bins=self._continuous_factor_bins,
            auto_bin_method=self._auto_bin_method,
            exclude=list(self._exclude) if self._exclude else None,
            include=list(self._include) if self._include else None,
        )

    def __call__(self, data: Any | None = None) -> Array:
        """Extract metadata factors from data.

        Implements the :class:`~dataeval.protocols.FeatureExtractor` protocol,
        allowing this instance to be used directly with drift detectors.

        Parameters
        ----------
        data : Any or None, default None
            Dataset to extract metadata from. If None, uses the bound dataset.

        Returns
        -------
        Array
            Binned metadata array of shape (n_samples, n_factors).

        Raises
        ------
        ValueError
            If data is None and no dataset is bound.

        Example
        -------
        >>> from dataeval import Metadata
        >>>
        >>> metadata = Metadata(train_dataset, continuous_factor_bins={"brightness": 10})
        >>>
        >>> # Extract from bound dataset
        >>> train_factors = metadata()
        >>>
        >>> # Extract from new dataset
        >>> test_factors = metadata(test_dataset)
        """
        if data is None:
            if self._dataset is None:
                raise ValueError("No dataset bound. Provide data or call bind() first.")
            # Return factors for bound dataset
            return self.factor_data

        # Check if same as bound dataset (by identity)
        if self._dataset is not None and data is self._dataset:
            return self.factor_data

        # Compute metadata for new data using this config
        return self.new(data).factor_data

    @property
    def raw(self) -> Sequence[Mapping[str, Any]]:
        """Original metadata dictionaries extracted from the dataset.

        Access the unprocessed metadata as it was provided in the original dataset before
        any binning, filtering, or transformation operations.

        Returns
        -------
        Sequence[Mapping[str, Any]]
            List of metadata dictionaries, one per dataset item, containing the original key-value
            pairs as provided in the source data

        Notes
        -----
            This property triggers dataset structure analysis on first access.
        """
        self._structure()
        return self._raw

    @property
    def exclude(self) -> set[str]:
        """Factor names excluded from metadata processing.

        Returns
        -------
        set[str]
            Set of factor names that are filtered out during processing.
            Empty set when no exclusions are active.

        """
        return self._exclude

    @exclude.setter
    def exclude(self, value: Sequence[str]) -> None:
        """Set factor names to exclude from processing.

        Automatically clears include filter and resets binning state when exclusion list changes.

        Parameters
        ----------
        value : Sequence[str]
            Factor names to exclude from metadata analysis.
        """
        exclude = set(value)
        if self._exclude != exclude:
            self._exclude = exclude
            self._include = set()
            self._is_binned = False

    @property
    def include(self) -> set[str]:
        """Factor names included in metadata processing.

        Returns
        -------
        set[str]
            Set of factor names that are processed during analysis. Empty set when no inclusion filter is active.
        """
        return self._include

    @include.setter
    def include(self, value: Sequence[str]) -> None:
        """Set factor names to include in processing.

        Automatically clears exclude filter and resets binning state when
        inclusion list changes.

        Parameters
        ----------
        value : Sequence[str]
            Factor names to include in metadata analysis.
        """
        include = set(value)
        if self._include != include:
            self._include = include
            self._exclude = set()
            self._is_binned = False

    @property
    def continuous_factor_bins(self) -> Mapping[str, int | Sequence[float]]:
        """Binning configuration for continuous factors.

        Returns
        -------
        Mapping[str, int | Sequence[float]]
            Mapping of factor names to either the number of bins
            (int) or explicit bin edges (sequence of floats).
        """
        return self._continuous_factor_bins

    @continuous_factor_bins.setter
    def continuous_factor_bins(self, bins: Mapping[str, int | Sequence[float]]) -> None:
        """Update binning configuration for continuous factors.

        Triggers re-binning when configuration changes to ensure data
        consistency with new bin specifications.

        Parameters
        ----------
        bins : Mapping[str, int | Sequence[float]]
            Mapping of factor names to bin counts or explicit edges.
        """
        if self._continuous_factor_bins != bins:
            self._continuous_factor_bins = dict(bins)
            self._reset_bins(bins)

    @property
    def auto_bin_method(self) -> Literal["uniform_width", "uniform_count", "clusters"]:
        """Automatic binning strategy for continuous factors.

        Returns
        -------
        {"uniform_width", "uniform_count", "clusters"}
            Current method used for automatic discretization of continuous
            factors that lack explicit bin specifications.
        """
        return self._auto_bin_method

    @auto_bin_method.setter
    def auto_bin_method(self, method: Literal["uniform_width", "uniform_count", "clusters"]) -> None:
        """Set automatic binning strategy for continuous factors.

        Triggers re-binning with the new method when strategy changes to
        ensure consistent discretization across all factors.

        Parameters
        ----------
        method : {"uniform_width", "uniform_count", "clusters"}
            Binning strategy to apply for continuous factors without
            explicit bin configurations.
        """
        if self._auto_bin_method != method:
            self._auto_bin_method = method
            self._reset_bins()

    @property
    def target_factors_only(self) -> bool:
        """Whether only target-level factors are included from the factors list.

        Returns
        -------
        bool
            True if image-level factors are excluded, False if included (default).
        """
        return self._target_factors_only

    @target_factors_only.setter
    def target_factors_only(self, value: bool) -> None:
        """Set whether to only include target-level factors.

        Rebuilds the factor list when the value changes to ensure proper
        inclusion or exclusion of image-level factors.

        Parameters
        ----------
        value : bool
            True to exclude image-level factors, False to include them.
        """
        if self._target_factors_only != value:
            self._target_factors_only = value
            # Reset bins before building factors so _build_factors can properly clean up
            if self._is_binned:
                self._reset_bins()
            self._build_factors()
            self._is_binned = False

    @property
    def dataframe(self) -> pl.DataFrame:
        """Processed DataFrame containing both image-level and target-level rows.

        Access the main data structure with both image-level metadata and
        target-level information (class labels, scores, bounding boxes).
        Use `image_data` or `target_data` properties to filter to specific row types.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns for image_index, target_index, class_label, scores,
            bounding boxes (when applicable), and all processed metadata factors.
            Rows where target_index is None contain image-level data.
            Rows where target_index is an integer contain target/detection-level data.

        Notes
        -----
        This property triggers dataset structure analysis on first access.
        Factor binning occurs automatically when accessing factor-related data.

        For Object Detection datasets, the dataframe now contains:
        - Image-level rows (target_index=None): One per image with image-level factors
        - Target-level rows (target_index=0,1,2...): One per detection with detection data

        See Also
        --------
        image_data : Filter to image-level rows only
        target_data : Filter to target-level rows only
        """
        self._structure()
        return self._dataframe

    @property
    def dropped_factors(self) -> Mapping[str, Sequence[str]]:
        """Factors removed during preprocessing with removal reasons.

        Returns
        -------
        Mapping[str, Sequence[str]]
            Mapping of dropped factor names to lists of reasons
            why they were excluded from the final dataset.

        Notes
        -----
        This property triggers dataset structure analysis on first access.
        Common removal reasons include incompatible data types, excessive
        missing values, or insufficient variation.
        """
        self._structure()
        return self._dropped_factors

    @property
    def factor_data(self) -> NDArray[np.int64]:
        """Factor data with continuous values discretized into bins.

        Access fully processed factor data where both categorical and
        continuous factors are converted to integer bin indices.

        Returns
        -------
        NDArray[np.int64]
            Array with shape (n_samples, n_factors) containing binned integer
            data ready for categorical analysis algorithms. Returns empty array
            when no factors are available.
            For OD datasets, returns only target-level rows to align with class_label.

        Notes
        -----
        This property triggers factor binning analysis on first access.
        Use this for algorithms requiring purely discrete input data.

        For object detection datasets, this returns target-level data only to
        ensure alignment with class_labels (one row per detection).
        """
        if not self.factor_names:
            return np.array([], dtype=np.int64)

        self._bin()

        # For datasets with targets, use only target-level rows to align with class_label
        df = self.target_data if self.has_targets() else self.dataframe

        return df.select([_to_col(k, v, True) for k, v in self.factor_info.items()]).to_numpy().astype(np.int64)

    @property
    def factor_names(self) -> Sequence[str]:
        """Names of all processed metadata factors.

        Returns
        -------
        Sequence[str]
            List of factor names that passed filtering and preprocessing steps.
            Order matches columns in factor_data and binned_data.

        Notes
        -----
        This property triggers dataset structure analysis on first access.
        Factor names respect include/exclude filtering settings.
        """
        self._structure()
        return sorted(filter(self._filter, self._factors))

    @property
    def factor_info(self) -> Mapping[str, FactorInfo]:
        """Type information and processing status for each factor.

        Returns
        -------
        Mapping[str, FactorInfo]
            Mapping of factor names to FactorInfo objects containing
            data type classification and processing flags (binned, digitized).

        Notes
        -----
        This property triggers factor binning analysis on first access.
        Only includes factors that survived preprocessing and filtering.
        """
        self._bin()
        filtered = dict(filter(self._filter, ((k, v) for k, v in self._factors.items() if v is not None)))
        return {k: filtered[k] for k in sorted(filtered)}

    @property
    def is_discrete(self) -> Sequence[bool]:
        """Whether each factor is discrete (True) or continuous (False).

        Returns
        -------
        Sequence[bool]
            Boolean sequence with length equal to factor_names, where True
            indicates a discrete factor (categorical or discrete numeric)
            and False indicates a continuous factor.

        Notes
        -----
        This property is part of the :class:`~dataeval.protocols.Metadata`
        and aligns with scientific computing conventions where discrete factors
        are treated differently from continuous ones in statistical analyses.
        """
        return [info.factor_type != "continuous" for info in self.factor_info.values()]

    @property
    def raw_data(self) -> NDArray[Any]:
        """Raw factor values before binning or digitization.

        Access unprocessed factor data in its original numeric form before
        any categorical encoding or binning transformations are applied.

        Returns
        -------
        NDArray[Any]
            Array with shape (n_samples, n_factors) containing original factor
            values. Returns empty array when no factors are available.
            For OD datasets, returns only target-level rows to align with class_labels.

        Notes
        -----
        Use this for algorithms that can work with mixed data types or when
        you need access to original continuous values. For analysis-ready
        numeric data, use binned_data.

        For object detection datasets, this returns target-level data only to
        ensure alignment with class_labels (one row per detection).
        """
        if not self.factor_names:
            return np.array([], dtype=np.float64)

        # For datasets with targets, use only target-level rows to align with class_label
        df = self.target_data if self.has_targets() else self.dataframe

        # Extract continuous columns and convert to NumPy array
        return df.select(self.factor_names).to_numpy()

    @property
    def class_labels(self) -> NDArray[np.intp]:
        """Target class labels as integer indices.

        Returns
        -------
        NDArray[np.intp]
            Array of class indices corresponding to dataset targets. For
            object detection datasets, contains one label per detection.

        Notes
        -----
        This property triggers dataset structure analysis on first access.
        Use index2label property to get human-readable label names.
        """
        self._structure()
        return self._class_labels

    @property
    def index2label(self) -> Mapping[int, str]:
        self._structure()
        return self._index2label

    @property
    def item_indices(self) -> NDArray[np.intp]:
        """Dataset indices linking targets back to source item.

        Returns
        -------
        NDArray[np.intp]
            Array mapping each target/detection back to its source item
            index in the original dataset. Essential for object detection
            datasets where multiple detections come from a single item.

        Notes
        -----
        This property triggers dataset structure analysis on first access.
        """
        self._structure()
        return self._item_indices

    @property
    def item_count(self) -> int:
        """Total number of items in the dataset.

        Returns
        -------
        int
            Count of unique items in the source dataset, regardless of
            how many targets/detections each item contains.
        """
        if self._count == 0:
            self._structure()
        return self._count

    @property
    def image_data(self) -> pl.DataFrame:
        """Dataframe containing only image-level rows.

        Returns a view of the metadata dataframe filtered to rows where
        target_index is None, containing one row per image with image-level
        factors.

        Returns
        -------
        pl.DataFrame
            Dataframe with image-level metadata. For Object Detection datasets,
            this provides per-image analysis without target-level duplication.

        Notes
        -----
        This property triggers dataset structure analysis on first access.
        Image-level factors are stored only in these rows to avoid duplication.

        Examples
        --------
        >>> metadata = Metadata(dataset)
        >>> metadata.image_data.select("image_index", "time_of_day", "weather", "location").head(5)
        shape: (5, 4)
        ┌─────────────┬─────────────┬─────────┬──────────┐
        │ image_index ┆ time_of_day ┆ weather ┆ location │
        │ ---         ┆ ---         ┆ ---     ┆ ---      │
        │ i64         ┆ str         ┆ str     ┆ str      │
        ╞═════════════╪═════════════╪═════════╪══════════╡
        │ 0           ┆ dawn        ┆ rainy   ┆ suburban │
        │ 1           ┆ day         ┆ rainy   ┆ rural    │
        │ 2           ┆ dawn        ┆ clear   ┆ maritime │
        │ 3           ┆ dusk        ┆ rainy   ┆ maritime │
        │ 4           ┆ dusk        ┆ clear   ┆ suburban │
        └─────────────┴─────────────┴─────────┴──────────┘
        """
        self._structure()
        if self.has_targets():
            return self._dataframe.filter(pl.col("target_index").is_null())

        # Return target data as image data for classification datasets
        return self.target_data

    @property
    def target_data(self) -> pl.DataFrame:
        """Dataframe containing only target-level rows.

        Returns a view of the metadata dataframe filtered to rows where
        target_index is not None, containing target/detection-level data.

        Returns
        -------
        pl.DataFrame
            Dataframe with target-level metadata. Each row represents a
            single target or detection with its associated class, score,
            and bounding box information.

        Notes
        -----
        This property triggers dataset structure analysis on first access.
        This is similar to the legacy behavior where only target-level rows
        existed, but now image-level metadata is stored separately in image_data.

        Examples
        --------
        >>> metadata = Metadata(dataset)
        >>> metadata.target_data.select("image_index", "target_index", "class_label").head(5)
        shape: (5, 3)
        ┌─────────────┬──────────────┬─────────────┐
        │ image_index ┆ target_index ┆ class_label │
        │ ---         ┆ ---          ┆ ---         │
        │ i64         ┆ i64          ┆ i64         │
        ╞═════════════╪══════════════╪═════════════╡
        │ 0           ┆ 0            ┆ 0           │
        │ 1           ┆ 0            ┆ 3           │
        │ 1           ┆ 1            ┆ 2           │
        │ 1           ┆ 2            ┆ 1           │
        │ 2           ┆ 0            ┆ 1           │
        └─────────────┴──────────────┴─────────────┘
        """
        self._structure()
        return self._dataframe.filter(pl.col("target_index").is_not_null())

    def get_image_factors(self, image_idx: int) -> dict[str, Any]:
        """Get all factors for a specific image.

        Parameters
        ----------
        image_idx : int
            Index of the image to retrieve factors for

        Returns
        -------
        dict[str, Any]
            Dictionary mapping factor names to their values for the specified image

        Examples
        --------
        >>> metadata = Metadata(dataset)
        >>> factors = metadata.get_image_factors(0)
        >>> factors["time_of_day"]
        'dawn'
        >>> factors["weather"]
        'rainy'
        >>> factors["location"]
        'suburban'
        """
        self._structure()
        row = self.image_data.filter(pl.col("image_index") == image_idx)
        if row.height == 0:
            raise ValueError(f"No image found with index {image_idx}")
        return row.to_dicts()[0]

    def get_target_factors(self, image_idx: int, target_idx: int) -> dict[str, Any]:
        """Get all factors for a specific target within an image.

        Parameters
        ----------
        image_idx : int
            Index of the image containing the target
        target_idx : int
            Index of the target within the image (0-indexed per image)

        Returns
        -------
        dict[str, Any]
            Dictionary mapping factor names to their values for the specified target

        Examples
        --------
        >>> metadata = Metadata(dataset)
        >>> factors = metadata.get_target_factors(1, 1)
        >>> factors["image_index"]
        1
        >>> factors["target_index"]
        1
        >>> factors["class_label"]
        2
        """
        self._structure()
        row = self.target_data.filter((pl.col("image_index") == image_idx) & (pl.col("target_index") == target_idx))
        if row.height == 0:
            raise ValueError(f"No target found with image_index={image_idx}, target_index={target_idx}")
        return row.to_dicts()[0]

    def _filter(self, factor: str | tuple[str, Any]) -> bool:
        factor = factor[0] if isinstance(factor, tuple) else factor
        return factor in self.include if self.include else factor not in self.exclude

    def _reset_bins(self, cols: Iterable[str] | None = None) -> None:
        if self._is_binned:
            columns = self._dataframe.columns
            for col in cols or columns:
                # Track if we removed any processed columns for this factor
                removed = False
                # Remove binned columns
                if _binned(col) in columns:
                    self._dataframe.drop_in_place(_binned(col))
                    removed = True
                # Remove digitized columns
                if _digitized(col) in columns:
                    self._dataframe.drop_in_place(_digitized(col))
                    removed = True
                # Reset factor info only if we actually removed a processed column
                if removed and hasattr(self, "_factors") and col in self._factors:
                    self._factors[col] = None
            self._is_binned = False

    def _compute_target_indices(self, srcidx: NDArray[np.intp], datum_count: int, is_od: bool) -> NDArray[np.intp]:
        """Compute per-image target indices (0, 1, 2, ... within each image)."""
        target_idx = np.zeros_like(srcidx, dtype=np.intp)
        if is_od and len(srcidx) > 0:
            for img_idx in range(datum_count):
                mask = srcidx == img_idx
                target_idx[mask] = np.arange(mask.sum())
        return target_idx

    def _build_target_rows(
        self,
        srcidx: NDArray[np.intp],
        target_idx: NDArray[np.intp],
        labels: NDArray[np.intp],
        scores: NDArray[np.float32],
        bboxes: NDArray[np.float32] | None,
        factor_dict: dict[str, Any],
        is_od: bool,
        image_factor_names: set[str] | None = None,
        image_factor_dict: dict[str, Any] | None = None,
    ) -> dict[str, list]:
        """Build target-level rows with detection data.

        Parameters
        ----------
        factor_dict : dict[str, Any]
            For OD datasets: target-level factors (from merge with ignore_lists=False)
            For IC datasets: image-level factors (from merge with ignore_lists=False)
        image_factor_dict : dict[str, Any] | None
            For OD datasets: image-level factors (from merge with ignore_lists=True)
            Used to replicate image metadata to target rows
        """
        target_rows = {
            "image_index": srcidx.tolist() if isinstance(srcidx, np.ndarray) else list(srcidx),
            "target_index": target_idx.tolist() if isinstance(target_idx, np.ndarray) else list(target_idx),
            "class_label": labels.tolist() if isinstance(labels, np.ndarray) else list(labels),
            "score": scores.tolist() if isinstance(scores, np.ndarray) else list(scores),
            "box": (bboxes.tolist() if isinstance(bboxes, np.ndarray) else list(bboxes))
            if bboxes is not None
            else [None] * len(labels),
        }
        # Add factor values to target rows
        for factor_name, factor_values in factor_dict.items():
            target_rows[factor_name] = self._get_target_factor_values(
                factor_name, factor_values, srcidx, is_od, image_factor_names, image_factor_dict
            )
        return target_rows

    def _get_target_factor_values(
        self,
        factor_name: str,
        factor_values: Any,
        srcidx: NDArray[np.intp],
        is_od: bool,
        image_factor_names: set[str] | None,
        image_factor_dict: dict[str, Any] | None = None,
    ) -> list:
        """Get factor values for target rows, handling OD vs IC datasets."""
        if is_od and image_factor_names is not None and factor_name in image_factor_names:
            # Image-level metadata for OD: replicate to target rows using srcidx mapping
            # Use image_factor_dict if provided, otherwise fall back to factor_values
            source_values = image_factor_dict[factor_name] if image_factor_dict is not None else factor_values
            if isinstance(source_values, np.ndarray):
                return source_values[srcidx].tolist()
            if isinstance(source_values, list):
                return [source_values[i] for i in srcidx]
            return [list(source_values)[i] for i in srcidx]

        if is_od:
            # Target-level metadata for OD: use values as-is
            if isinstance(factor_values, np.ndarray):
                return factor_values.tolist()
            if isinstance(factor_values, list):
                return factor_values
            return list(factor_values)

        # For IC datasets, map image factors to target rows using srcidx
        if isinstance(factor_values, np.ndarray):
            return factor_values[srcidx].tolist()
        if isinstance(factor_values, list):
            return [factor_values[i] for i in srcidx]
        return [list(factor_values)[i] for i in srcidx]

    def _build_image_rows(self, datum_count: int, image_factor_dict: dict[str, Any]) -> dict[str, list]:
        """Build image-level rows with metadata."""
        image_rows = {
            "image_index": list(range(datum_count)),
            "target_index": [None] * datum_count,
            "class_label": [None] * datum_count,
            "score": [None] * datum_count,
            "box": [None] * datum_count,
        }
        # Add image-level factors to image rows
        for factor_name, factor_values in image_factor_dict.items():
            if isinstance(factor_values, np.ndarray):
                image_rows[factor_name] = factor_values.tolist()
            elif isinstance(factor_values, list):
                image_rows[factor_name] = factor_values
            else:
                image_rows[factor_name] = list(factor_values)
        return image_rows

    def _combine_rows(self, image_rows: dict[str, list], target_rows: dict[str, list]) -> dict[str, list]:
        """Combine image-level and target-level rows into a single dictionary."""
        combined_rows = {}
        num_image_rows = len(image_rows["image_index"])

        for key in target_rows:
            if key in image_rows:
                combined_rows[key] = image_rows[key] + target_rows[key]
            else:
                # Key exists in target_rows but not image_rows (e.g., list-type metadata)
                # Add None values for image rows
                combined_rows[key] = [None] * num_image_rows + target_rows[key]
        return combined_rows

    def _infer_factor_level(
        self, factors: Mapping[str, Array1D[Any]], num_image_rows: int, num_target_rows: int
    ) -> Literal["image", "target"]:
        """Infer factor level based on array lengths."""
        factor_lengths = {len(v) for v in factors.values()}
        if len(factor_lengths) > 1:
            raise ValueError("All factors must have the same length when using level='auto'")
        factor_len = factor_lengths.pop()

        if factor_len == num_image_rows:
            return "image"
        if factor_len == num_target_rows:
            return "target"
        raise ValueError(
            "The lists/arrays in the provided factors have a different length "
            f"than the current metadata factors. Expected {num_image_rows} (image count) "
            f"or {num_target_rows} (target count), got {factor_len}."
        )

    def _validate_factor_lengths(
        self, factors: Mapping[str, Array1D[Any]], level: str, num_image_rows: int, num_target_rows: int
    ) -> None:
        """Validate that factor lengths match the specified level."""
        if level == "image":
            expected_len = num_image_rows
            if not all(len(v) == expected_len for v in factors.values()):
                raise ValueError(f"All image-level factors must have length {expected_len} (image count)")
        elif level == "target":
            expected_len = num_target_rows
            if not all(len(v) == expected_len for v in factors.values()):
                raise ValueError(f"All target-level factors must have length {expected_len} (target count)")
        else:
            raise ValueError(f"Invalid level: {level}. Must be 'image', 'target', or 'auto'")

    def _create_factor_column(self, data_array: NDArray, level: str, num_image_rows: int) -> list:
        """Create a factor column with values at the appropriate level."""
        if level == "image":
            # Create column: image-level values in image rows, None in target rows
            full_data = [None] * len(self.dataframe)
            for idx, val in enumerate(data_array):
                full_data[idx] = val  # Image rows come first in our structure
            return full_data
        # level == "target"
        # Create column: None in image rows, target-level values in target rows
        return [None] * num_image_rows + list(data_array)

    def has_targets(self) -> bool:
        """Check if the source dataset has targets.

        Returns
        -------
        bool
            True if dataset contains targets, False for classification datasets.
        """
        if self._has_targets is None:
            self._structure()
        return bool(self._has_targets)

    def _process_targets(
        self,
        raw: list,
        labels: list,
        bboxes: list,
        scores: list,
        srcidx: list,
        datum_count: int,
        progress_callback: ProgressCallback | None,
    ) -> bool | None:
        """Process dataset targets and extract labels, bboxes, scores.

        Returns
        -------
        bool | None
            True if OD dataset, False if IC dataset, None if empty dataset
        """
        if self._dataset is None:
            return None

        is_od = None
        datum_count = len(self._dataset)
        for i in range(datum_count):
            _, target, metadata = self._dataset[i]
            raw.append(metadata)

            if is_od_target := isinstance(target, ObjectDetectionTarget):
                target_labels = as_numpy(target.labels)
                target_len = len(target_labels)
                if target_len:
                    labels.append(target_labels)
                    bboxes.append(as_numpy(target.boxes))
                    scores.append(as_numpy(target.scores))
                    srcidx.extend([i] * target_len)
            elif isinstance(target, Array):
                target_scores = as_numpy(target)
                if len(target_scores):
                    labels.append([np.argmax(target_scores)])
                    scores.append([target_scores])
                    srcidx.append(i)
            else:
                raise TypeError("Encountered unsupported target type in dataset")

            is_od = is_od_target if is_od is None else is_od
            if is_od != is_od_target:
                raise ValueError("Encountered unexpected target type in dataset")

            if progress_callback:
                progress_callback(i, total=datum_count)

        return is_od

    def _merge_od_metadata(
        self, raw: Sequence[Mapping[str, Any]], datum_count: int, srcidx: NDArray[np.intp], reserved: list[str]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[str]]]:
        """Merge OD metadata at both target and image levels.

        Returns
        -------
        tuple
            (target_factor_dict, image_factor_dict, dropped_factors)
        """
        targets_per_image = [np.sum(srcidx == i) for i in range(datum_count)]

        # Target-level merge
        merged_target_level = merge_metadata(
            raw, return_dropped=True, ignore_lists=False, targets_per_image=targets_per_image
        )
        target_factor_dict = {
            f"metadata_{k}" if k in reserved else k: v for k, v in merged_target_level[0].items() if k != "_image_index"
        }

        # Image-level merge
        merged_image_level = merge_metadata(raw, return_dropped=True, ignore_lists=True, targets_per_image=None)
        image_factor_dict = {
            f"metadata_{k}" if k in reserved else k: v for k, v in merged_image_level[0].items() if k != "_image_index"
        }

        return target_factor_dict, image_factor_dict, merged_target_level[1]

    def _build_od_rows(
        self,
        srcidx: NDArray[np.intp],
        target_idx: NDArray[np.intp],
        labels: NDArray[np.intp],
        scores: NDArray[np.float32],
        bboxes: NDArray[np.float32] | None,
        target_factor_dict: dict[str, Any],
        image_factor_dict: dict[str, Any],
        datum_count: int,
    ) -> dict[str, list]:
        """Build combined rows for OD dataset."""
        target_rows = self._build_target_rows(
            srcidx,
            target_idx,
            labels,
            scores,
            bboxes,
            target_factor_dict,
            True,
            set(image_factor_dict.keys()),
            image_factor_dict,
        )
        image_rows = self._build_image_rows(datum_count, image_factor_dict)
        return self._combine_rows(image_rows, target_rows)

    def _build_factors(self) -> None:
        """Build the _factors dict from stored factor names."""
        if not self._is_structured:
            self._factors = {}
            return

        factors = (
            self._target_factors - self._image_factors
            if self._has_targets and self._target_factors_only
            else self._target_factors.union(self._image_factors)
        )

        usable_factors = {
            k for k in factors if not isinstance(self._dataframe.schema.get(k), pl.List | pl.Struct | pl.Array)
        }

        self._factors = dict.fromkeys(usable_factors, None)

    def _structure(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        if self._is_structured:
            return

        if self._dataset is None:
            raise ValueError("No dataset bound. Call bind() first.")

        raw: list[Mapping[str, Any]] = []
        labels = []
        bboxes = []
        scores = []
        srcidx = []
        datum_count = len(self._dataset)

        self._has_targets = self._process_targets(raw, labels, bboxes, scores, srcidx, datum_count, progress_callback)

        np_asarray: Callable[..., np.ndarray] = np.concatenate if srcidx else np.asarray
        labels = np_asarray(labels, dtype=np.intp)
        scores = np_asarray(scores, dtype=np.float32)
        bboxes = np_asarray(bboxes, dtype=np.float32) if self._has_targets else None
        srcidx = np.asarray(srcidx, dtype=np.intp)

        index2label = self._dataset.metadata.get("index2label", {i: str(i) for i in np.unique(labels)})
        target_idx = self._compute_target_indices(srcidx, datum_count, bool(self._has_targets))
        reserved = ["image_index", "target_index", "class_label", "score", "box"]
        target_factor_dict = {}

        # Build target-level and image-level rows
        if self._has_targets:
            target_factor_dict, image_factor_dict, dropped_factors = self._merge_od_metadata(
                raw, datum_count, srcidx, reserved
            )
            combined_rows = self._build_od_rows(
                srcidx, target_idx, labels, scores, bboxes, target_factor_dict, image_factor_dict, datum_count
            )
            self._dropped_factors = dropped_factors
            self._target_factors = set(target_factor_dict)
            self._image_factors = set(image_factor_dict)
        else:
            # For IC datasets, only need target-level rows (which are same as image-level)
            merged_image_level = merge_metadata(raw, return_dropped=True, ignore_lists=False, targets_per_image=None)
            image_factor_dict = {
                f"metadata_{k}" if k in reserved else k: v
                for k, v in merged_image_level[0].items()
                if k != "_image_index"
            }
            target_rows = self._build_target_rows(
                srcidx, target_idx, labels, scores, bboxes, image_factor_dict, bool(self._has_targets)
            )
            combined_rows = target_rows
            self._dropped_factors = merged_image_level[1]
            self._image_factors = set(image_factor_dict)
            self._target_factors = set()

        self._raw = raw
        self._index2label = index2label
        self._class_labels = labels
        self._item_indices = srcidx

        self._dataframe = pl.DataFrame(combined_rows)
        self._is_structured = True

        # Build _factors dict from stored factor dictionaries
        self._build_factors()

    def _add_column_with_padding(self, df: pl.DataFrame, col_name: str, values: NDArray, is_od: bool) -> pl.DataFrame:
        """Add a column to dataframe with None padding for OD image rows."""
        if is_od:
            num_image_rows = len(self.image_data)
            full_values = [None] * num_image_rows + values.tolist()
            return df.with_columns(pl.Series(name=col_name, values=full_values))
        return df.with_columns(pl.Series(name=col_name, values=values))

    def _process_binned_factor(
        self, df: pl.DataFrame, col: str, data: NDArray, bins: int | Sequence[float], is_od: bool
    ) -> tuple[pl.DataFrame, FactorInfo]:
        """Process a factor with user-provided bins."""
        col_bn = _binned(col)
        binned_values = digitize_data(data, bins).astype(np.int64)
        df = self._add_column_with_padding(df, col_bn, binned_values, is_od)
        return df, FactorInfo("continuous", is_binned=True)

    def _process_categorical_factor(
        self, df: pl.DataFrame, col: str, ordinal: NDArray, is_od: bool
    ) -> tuple[pl.DataFrame, FactorInfo]:
        """Process a non-numeric categorical factor."""
        col_dg = _digitized(col)
        df = self._add_column_with_padding(df, col_dg, ordinal.astype(np.int64), is_od)
        return df, FactorInfo("categorical", is_digitized=True)

    def _process_continuous_factor(
        self, df: pl.DataFrame, col: str, data: NDArray, is_od: bool
    ) -> tuple[pl.DataFrame, FactorInfo]:
        """Process a continuous numeric factor with automatic binning."""
        _logger.warning(
            f"A user defined binning was not provided for {col}. "
            f"Using the {self.auto_bin_method} method to discretize the data. "
            "It is recommended that the user rerun and supply the desired "
            "bins using the continuous_factor_bins parameter."
        )
        binned_data = bin_data(data, self.auto_bin_method)
        col_bn = _binned(col)
        df = self._add_column_with_padding(df, col_bn, binned_data.astype(np.int64), is_od)
        return df, FactorInfo("continuous", is_binned=True)

    def _process_factor(
        self, df: pl.DataFrame, col: str, data: NDArray, factor_bins: Mapping[str, int | Sequence[float]], is_od: bool
    ) -> tuple[pl.DataFrame, FactorInfo]:
        """Process a single factor based on its type."""
        if col in factor_bins:
            return self._process_binned_factor(df, col, data, factor_bins[col], is_od)

        _, ordinal = np.unique(data, return_inverse=True)
        if not np.issubdtype(data.dtype, np.number):
            return self._process_categorical_factor(df, col, ordinal, is_od)
        if is_continuous(data, self.item_indices):
            return self._process_continuous_factor(df, col, data, is_od)
        return df, FactorInfo("discrete")

    def _bin(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Populate factor info and bin non-categorical factors."""
        if self._is_binned:
            return

        factor_info: dict[str, FactorInfo] = {}
        df = self.dataframe.clone()
        factor_bins = self.continuous_factor_bins

        # For OD datasets, use target_data to extract data for analysis (avoids None values)
        is_od = self.has_targets()
        data_df = self.target_data if is_od else df

        # Check for invalid keys
        invalid_keys = set(factor_bins.keys()) - set(df.columns)
        if invalid_keys:
            _logger.warning(
                f"The keys - {invalid_keys} - are present in the `continuous_factor_bins` dictionary "
                "but are not columns in the metadata DataFrame. Unknown keys will be ignored."
            )

        column_set = set(df.columns)
        factors_to_process = [col for col in self.factor_names if not {_binned(col), _digitized(col)} & column_set]
        total_factors = len(factors_to_process)

        for i, col in enumerate(factors_to_process):
            data = data_df[col].to_numpy()
            df, info = self._process_factor(df, col, data, factor_bins, is_od)
            factor_info[col] = info

            if progress_callback:
                progress_callback(i + 1, total=total_factors)

        # Store the results
        self._dataframe = df
        self._factors.update(factor_info)
        self._is_binned = True

    def add_factors(
        self, factors: Mapping[str, Array1D[Any]], level: Literal["image", "target", "auto"] = "auto"
    ) -> None:
        """Add additional factors to metadata collection.

        Extend the current metadata with new factors at either image or target level.
        For image-level factors, values are stored only in image-level rows.
        For target-level factors, values are stored only in target-level rows.

        Parameters
        ----------
        factors : Mapping[str, _1DArray[Any]]
            Mapping of factor names to their values. Factor length must match
            the specified level (image count or target count).
        level : {"image", "target", "auto"}, default="auto"
            Level at which to store the factors:
            - "image": Array length must match image count, stored in image-level rows only
            - "target": Array length must match target count, stored in target-level rows only
            - "auto": Automatically infers level based on array length

        Raises
        ------
        ValueError
            When factor lengths do not match the specified level's dimensions.

        Examples
        --------
        >>> metadata = Metadata(od_dataset)
        >>> # Add image-level factors (e.g., from imagestats)
        >>> image_factors = {
        ...     "brightness": np.random.rand(50),  # One per image
        ...     "contrast": np.random.rand(50),  # One per image
        ... }
        >>> metadata.add_factors(image_factors, level="image")
        >>>
        >>> # Add target-level factors (e.g., detection confidence scores)
        >>> target_factors = {
        ...     "iou": np.random.rand(93),  # One per target/detection
        ... }
        >>> metadata.add_factors(target_factors, level="target")
        """
        self._structure()

        # Early return for empty factors
        if not factors:
            return

        num_image_rows = len(self.image_data)
        num_target_rows = len(self.target_data)

        # Determine the level
        if level == "auto":
            level = self._infer_factor_level(factors, num_image_rows, num_target_rows)

        # Validate factor lengths match the specified level
        self._validate_factor_lengths(factors, level, num_image_rows, num_target_rows)

        # Add factors to the appropriate rows
        new_columns = []
        for k, v in factors.items():
            data_array = as_numpy(v)
            full_data = self._create_factor_column(data_array, level, num_image_rows)
            new_columns.append(pl.Series(name=k, values=full_data))
            self._factors[k] = None

        if new_columns:
            self._dataframe = self.dataframe.with_columns(new_columns)
            self._is_binned = False
            if level == "image":
                self._image_factors.update(factors)
            elif level == "target":
                self._target_factors.update(factors)
            self._build_factors()

    def filter_by_factor(self, condition: Callable[[str, FactorInfo], bool]) -> NDArray[np.float64]:
        """Filters metadata factors by factor name or FactorInfo.

        Parameters
        ----------
        condition : Callable[[str, FactorInfo], bool]
            A condition to include the factor in the output.

        Returns
        -------
        NDArray[np.float64]
            Array with shape (n_samples, n_factors) where the factors
            are filtered by the user provided condition.
        """
        if not self.factor_names:
            return np.array([], dtype=np.float64)

        self._bin()
        filtered = [name for name, info in self.factor_info.items() if condition(name, info)]
        return self.dataframe[filtered].to_numpy().astype(np.float64)

    def filter_by_factor_type(
        self, factor_type: Literal["categorical", "discrete", "continuous"]
    ) -> NDArray[np.float64]:
        """Filters metadata factors by factor type.

        Parameters
        ----------
        factor_type : "categorical", "discrete" or "continuous"
            The factor type to include in the output.

        Returns
        -------
        NDArray[np.float64]
            Array with shape (n_samples, n_factors) where the factors
            are filtered by the user provided factor type.
        """
        return self.filter_by_factor(lambda _, fi: fi.factor_type == factor_type)
