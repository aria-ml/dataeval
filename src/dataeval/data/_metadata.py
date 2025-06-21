from __future__ import annotations

__all__ = []

import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray
from tqdm.auto import tqdm

from dataeval.typing import (
    AnnotatedDataset,
    Array,
    ObjectDetectionTarget,
)
from dataeval.utils._array import as_numpy
from dataeval.utils._bin import bin_data, digitize_data, is_continuous
from dataeval.utils.data._merge import merge


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


class Metadata:
    """Collection of binned metadata using Polars DataFrames.

    Processes dataset metadata by automatically binning continuous factors and digitizing
    categorical factors for analysis and visualization workflows.

    Parameters
    ----------
    dataset : ImageClassificationDataset or ObjectDetectionDataset
        Dataset that provides original targets and metadata for processing.
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
    """

    def __init__(
        self,
        dataset: AnnotatedDataset[tuple[Any, Any, Mapping[str, Any]]],
        *,
        continuous_factor_bins: Mapping[str, int | Sequence[float]] | None = None,
        auto_bin_method: Literal["uniform_width", "uniform_count", "clusters"] = "uniform_width",
        exclude: Sequence[str] | None = None,
        include: Sequence[str] | None = None,
    ) -> None:
        self._class_labels: NDArray[np.intp]
        self._class_names: list[str]
        self._image_indices: NDArray[np.intp]
        self._factors: dict[str, FactorInfo | None]
        self._dropped_factors: dict[str, list[str]]
        self._dataframe: pl.DataFrame
        self._raw: Sequence[Mapping[str, Any]]

        self._is_structured = False
        self._is_binned = False

        self._dataset = dataset
        self._count = len(dataset) if isinstance(dataset, Sized) else 0
        self._continuous_factor_bins = dict(continuous_factor_bins) if continuous_factor_bins else {}
        self._auto_bin_method: Literal["uniform_width", "uniform_count", "clusters"] = auto_bin_method

        if exclude is not None and include is not None:
            raise ValueError("Filters for `exclude` and `include` are mutually exclusive.")

        self._exclude = set(exclude or ())
        self._include = set(include or ())

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
            Dictionary mapping factor names to either the number of bins
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
            Dictionary mapping factor names to bin counts or explicit edges.
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
    def dataframe(self) -> pl.DataFrame:
        """Processed DataFrame containing targets and metadata factors.

        Access the main data structure with target information (class labels,
        scores, bounding boxes) and processed metadata factors ready for analysis.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns for image indices, class labels, scores,
            bounding boxes (when applicable), and all processed metadata factors.

        Notes
        -----
        This property triggers dataset structure analysis on first access.
        Factor binning occurs automatically when accessing factor-related data.
        """
        self._structure()
        return self._dataframe

    @property
    def dropped_factors(self) -> Mapping[str, Sequence[str]]:
        """Factors removed during preprocessing with removal reasons.

        Returns
        -------
        Mapping[str, Sequence[str]]
            Dictionary mapping dropped factor names to lists of reasons
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
    def digitized_data(self) -> NDArray[np.int64]:
        """Factor data with categorical values converted to integer codes.

        Access processed factor data where categorical factors are digitized
        to integer codes but continuous factors remain in their original form.

        Returns
        -------
        NDArray[np.int64]
            Array with shape (n_samples, n_factors) containing integer-coded
            categorical data. Returns empty array when no factors are available.

        Notes
        -----
        This property triggers factor binning analysis on first access.
        Use this for algorithms that can handle mixed categorical and
        continuous data types.
        """
        if not self.factor_names:
            return np.array([], dtype=np.int64)

        self._bin()
        return (
            self.dataframe.select([_to_col(k, v, False) for k, v in self.factor_info.items()])
            .to_numpy()
            .astype(np.int64)
        )

    @property
    def binned_data(self) -> NDArray[np.int64]:
        """Factor data with continuous values discretized into bins.

        Access fully processed factor data where both categorical and
        continuous factors are converted to integer bin indices.

        Returns
        -------
        NDArray[np.int64]
            Array with shape (n_samples, n_factors) containing binned integer
            data ready for categorical analysis algorithms. Returns empty array
            when no factors are available.

        Notes
        -----
        This property triggers factor binning analysis on first access.
        Use this for algorithms requiring purely discrete input data.
        """
        if not self.factor_names:
            return np.array([], dtype=np.int64)

        self._bin()
        return (
            self.dataframe.select([_to_col(k, v, True) for k, v in self.factor_info.items()])
            .to_numpy()
            .astype(np.int64)
        )

    @property
    def factor_names(self) -> Sequence[str]:
        """Names of all processed metadata factors.

        Returns
        -------
        Sequence[str]
            List of factor names that passed filtering and preprocessing steps.
            Order matches columns in factor_data, digitized_data, and binned_data.

        Notes
        -----
        This property triggers dataset structure analysis on first access.
        Factor names respect include/exclude filtering settings.
        """
        self._structure()
        return list(filter(self._filter, self._factors))

    @property
    def factor_info(self) -> Mapping[str, FactorInfo]:
        """Type information and processing status for each factor.

        Returns
        -------
        Mapping[str, FactorInfo]
            Dictionary mapping factor names to FactorInfo objects containing
            data type classification and processing flags (binned, digitized).

        Notes
        -----
        This property triggers factor binning analysis on first access.
        Only includes factors that survived preprocessing and filtering.
        """
        self._bin()
        return dict(filter(self._filter, ((k, v) for k, v in self._factors.items() if v is not None)))

    @property
    def factor_data(self) -> NDArray[Any]:
        """Raw factor values before binning or digitization.

        Access unprocessed factor data in its original numeric form before
        any categorical encoding or binning transformations are applied.

        Returns
        -------
        NDArray[Any]
            Array with shape (n_samples, n_factors) containing original factor
            values. Returns empty array when no factors are available.

        Notes
        -----
        Use this for algorithms that can work with mixed data types or when
        you need access to original continuous values. For analysis-ready
        integer data, use binned_data or digitized_data instead.
        """
        if not self.factor_names:
            return np.array([], dtype=np.float64)

        # Extract continuous columns and convert to NumPy array
        return self.dataframe.select(self.factor_names).to_numpy()

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
        Use class_names property to get human-readable label names.
        """
        self._structure()
        return self._class_labels

    @property
    def class_names(self) -> Sequence[str]:
        """Human-readable names corresponding to class labels.

        Returns
        -------
        Sequence[str]
            List of class names where index corresponds to class label value.
            Derived from dataset metadata or auto-generated from label indices.

        Notes
        -----
        This property triggers dataset structure analysis on first access.
        """
        self._structure()
        return self._class_names

    @property
    def image_indices(self) -> NDArray[np.intp]:
        """Dataset indices linking targets back to source images.

        Returns
        -------
        NDArray[np.intp]
            Array mapping each target/detection back to its source image
            index in the original dataset. Essential for object detection
            datasets where multiple detections come from single images.

        Notes
        -----
        This property triggers dataset structure analysis on first access.
        """
        self._structure()
        return self._image_indices

    @property
    def image_count(self) -> int:
        """Total number of images in the dataset.

        Returns
        -------
        int
            Count of unique images in the source dataset, regardless of
            how many targets/detections each image contains.
        """
        if self._count == 0:
            self._structure()
        return self._count

    def _filter(self, factor: str | tuple[str, Any]) -> bool:
        factor = factor[0] if isinstance(factor, tuple) else factor
        return factor in self.include if self.include else factor not in self.exclude

    def _reset_bins(self, cols: Iterable[str] | None = None) -> None:
        if self._is_binned:
            columns = self._dataframe.columns
            for col in (col for col in cols or columns if _binned(col) in columns):
                self._dataframe.drop_in_place(_binned(col))
                self._factors[col] = None
            self._is_binned = False

    def _structure(self) -> None:
        if self._is_structured:
            return

        raw: Sequence[Mapping[str, Any]] = []

        labels = []
        bboxes = []
        scores = []
        srcidx = []
        is_od = None
        for i in tqdm(range(len(self._dataset))):
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

        np_asarray: Callable[..., np.ndarray] = np.concatenate if srcidx else np.asarray
        labels = np_asarray(labels, dtype=np.intp)
        scores = np_asarray(scores, dtype=np.float32)
        bboxes = np_asarray(bboxes, dtype=np.float32) if is_od else None
        srcidx = np.asarray(srcidx, dtype=np.intp)

        index2label = self._dataset.metadata.get("index2label", {i: str(i) for i in np.unique(labels)})

        targets_per_image = np.bincount(srcidx, minlength=len(self._dataset)).tolist() if is_od else None
        merged = merge(raw, return_dropped=True, ignore_lists=False, targets_per_image=targets_per_image)

        reserved = ["image_index", "class_label", "score", "box"]
        factor_dict = {f"metadata_{k}" if k in reserved else k: v for k, v in merged[0].items() if k != "_image_index"}

        target_dict = {
            "image_index": srcidx,
            "class_label": labels,
            "score": scores,
            "box": bboxes if bboxes is not None else [None] * len(labels),
        }

        self._raw = raw
        self._index2label = index2label
        self._class_labels = labels
        self._class_names = list(index2label.values())
        self._image_indices = target_dict["image_index"]
        self._factors = dict.fromkeys(factor_dict, None)
        self._dataframe = pl.DataFrame({**target_dict, **factor_dict})
        self._dropped_factors = merged[1]
        self._is_structured = True

    def _bin(self) -> None:
        """Populate factor info and bin non-categorical factors."""
        if self._is_binned:
            return

        # Start with an empty set of factor info
        factor_info: dict[str, FactorInfo] = {}

        # Create a mutable DataFrame for updates
        df = self.dataframe.clone()
        factor_bins = self.continuous_factor_bins

        # Check for invalid keys
        invalid_keys = set(factor_bins.keys()) - set(df.columns)
        if invalid_keys:
            warnings.warn(
                f"The keys - {invalid_keys} - are present in the `continuous_factor_bins` dictionary "
                "but are not columns in the metadata DataFrame. Unknown keys will be ignored."
            )

        column_set = set(df.columns)
        for col in (col for col in self.factor_names if not {_binned(col), _digitized(col)} & column_set):
            # Get data as numpy array for processing
            data = df[col].to_numpy()
            if col in factor_bins:
                # User provided binning
                bins = factor_bins[col]
                col_bn = _binned(col)
                df = df.with_columns(pl.Series(name=col_bn, values=digitize_data(data, bins).astype(np.int64)))
                factor_info[col] = FactorInfo("continuous", is_binned=True)
            else:
                # Check if data is numeric
                _, ordinal = np.unique(data, return_inverse=True)
                if not np.issubdtype(data.dtype, np.number):
                    # Non-numeric data - convert to categorical
                    col_dg = _digitized(col)
                    df = df.with_columns(pl.Series(name=col_dg, values=ordinal.astype(np.int64)))
                    factor_info[col] = FactorInfo("categorical", is_digitized=True)
                elif is_continuous(data, self.image_indices):
                    # Continuous values - discretize by binning
                    warnings.warn(
                        f"A user defined binning was not provided for {col}. "
                        f"Using the {self.auto_bin_method} method to discretize the data. "
                        "It is recommended that the user rerun and supply the desired "
                        "bins using the continuous_factor_bins parameter.",
                        UserWarning,
                    )
                    # Create binned version
                    binned_data = bin_data(data, self.auto_bin_method)
                    col_bn = _binned(col)
                    df = df.with_columns(pl.Series(name=col_bn, values=binned_data.astype(np.int64)))
                    factor_info[col] = FactorInfo("continuous", is_binned=True)
                else:
                    # Non-continuous values - treat as discrete
                    factor_info[col] = FactorInfo("discrete")

        # Store the results
        self._dataframe = df
        self._factors.update(factor_info)
        self._is_binned = True

    def add_factors(self, factors: Mapping[str, Array | Sequence[Any]]) -> None:
        """Add additional factors to metadata collection.

        Extend the current metadata with new factors, automatically handling
        length validation and integration with existing data structures.

        Parameters
        ----------
        factors : Mapping[str, Array | Sequence[Any]]
            Dictionary mapping factor names to their values. Factor length must
            match either the number of images or number of detections in the dataset.

        Raises
        ------
        ValueError
            When factor lengths do not match dataset dimensions.

        Examples
        --------
        >>> metadata = Metadata(dataset)
        >>> new_factors = {
        ...     "brightness": [0.2, 0.8, 0.5, 0.3, 0.4, 0.1, 0.3, 0.2],
        ...     "contrast": [1.1, 0.9, 1.0, 0.8, 1.2, 1.0, 0.7, 1.3],
        ... }
        >>> metadata.add_factors(new_factors)
        """
        self._structure()

        targets = len(self.dataframe)
        images = self.image_count
        targets_match = all(len(v) == targets for v in factors.values())
        images_match = targets_match if images == targets else all(len(v) == images for v in factors.values())
        if not targets_match and not images_match:
            raise ValueError(
                "The lists/arrays in the provided factors have a different length than the current metadata factors."
            )

        new_columns = []
        for k, v in factors.items():
            data = as_numpy(v)[self.image_indices]
            new_columns.append(pl.Series(name=k, values=data))
            self._factors[k] = None

        if new_columns:
            self._dataframe = self.dataframe.with_columns(new_columns)
            self._is_binned = False
