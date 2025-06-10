from __future__ import annotations

__all__ = []

import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence, Sized

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval.typing import (
    AnnotatedDataset,
    Array,
    ObjectDetectionTarget,
)
from dataeval.utils._array import as_numpy
from dataeval.utils._bin import bin_data, digitize_data, is_continuous
from dataeval.utils.data.metadata import merge


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
    """
    Class containing binned metadata using Polars DataFrames.

    Parameters
    ----------
    dataset : ImageClassificationDataset or ObjectDetectionDataset
        Dataset to access original targets and metadata from.
    continuous_factor_bins : Mapping[str, int | Sequence[float]] | None, default None
        Mapping from continuous factor name to the number of bins or bin edges
    auto_bin_method : Literal["uniform_width", "uniform_count", "clusters"], default "uniform_width"
        Method for automatically determining the number of bins for continuous factors
    exclude : Sequence[str] | None, default None
        Filter metadata factors to exclude the specified factors, cannot be set with `include`
    include : Sequence[str] | None, default None
        Filter metadata factors to include the specified factors, cannot be set with `exclude`
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
        """The raw list of metadata dictionaries for the dataset."""
        self._structure()
        return self._raw

    @property
    def exclude(self) -> set[str]:
        """Factors to exclude from the metadata."""
        return self._exclude

    @exclude.setter
    def exclude(self, value: Sequence[str]) -> None:
        exclude = set(value)
        if self._exclude != exclude:
            self._exclude = exclude
            self._include = set()
            self._is_binned = False

    @property
    def include(self) -> set[str]:
        """Factors to include from the metadata."""
        return self._include

    @include.setter
    def include(self, value: Sequence[str]) -> None:
        include = set(value)
        if self._include != include:
            self._include = include
            self._exclude = set()
            self._is_binned = False

    @property
    def continuous_factor_bins(self) -> Mapping[str, int | Sequence[float]]:
        """Map of factor names to bin counts or bin edges."""
        return self._continuous_factor_bins

    @continuous_factor_bins.setter
    def continuous_factor_bins(self, bins: Mapping[str, int | Sequence[float]]) -> None:
        if self._continuous_factor_bins != bins:
            self._continuous_factor_bins = dict(bins)
            self._reset_bins(bins)

    @property
    def auto_bin_method(self) -> Literal["uniform_width", "uniform_count", "clusters"]:
        """Binning method to use when continuous_factor_bins is not defined."""
        return self._auto_bin_method

    @auto_bin_method.setter
    def auto_bin_method(self, method: Literal["uniform_width", "uniform_count", "clusters"]) -> None:
        if self._auto_bin_method != method:
            self._auto_bin_method = method
            self._reset_bins()

    @property
    def dataframe(self) -> pl.DataFrame:
        """Dataframe containing target information and metadata factors."""
        self._structure()
        return self._dataframe

    @property
    def dropped_factors(self) -> Mapping[str, Sequence[str]]:
        """Factors that were dropped during preprocessing and the reasons why they were dropped."""
        self._structure()
        return self._dropped_factors

    @property
    def digitized_data(self) -> NDArray[np.int64]:
        """Factor data with digitized categorical data."""
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
        """Factor data with binned continuous data."""
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
        """Factor names of the metadata."""
        self._structure()
        return list(filter(self._filter, self._factors))

    @property
    def factor_info(self) -> Mapping[str, FactorInfo]:
        """Factor types of the metadata."""
        self._bin()
        return dict(filter(self._filter, ((k, v) for k, v in self._factors.items() if v is not None)))

    @property
    def factor_data(self) -> NDArray[Any]:
        """Factor data as a NumPy array."""
        if not self.factor_names:
            return np.array([], dtype=np.float64)

        # Extract continuous columns and convert to NumPy array
        return self.dataframe.select(self.factor_names).to_numpy()

    @property
    def class_labels(self) -> NDArray[np.intp]:
        """Class labels as a NumPy array."""
        self._structure()
        return self._class_labels

    @property
    def class_names(self) -> Sequence[str]:
        """Class names as a list of strings."""
        self._structure()
        return self._class_names

    @property
    def image_indices(self) -> NDArray[np.intp]:
        """Indices of images as a NumPy array."""
        self._structure()
        return self._image_indices

    @property
    def image_count(self) -> int:
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
        for i in range(len(self._dataset)):
            _, target, metadata = self._dataset[i]

            raw.append(metadata)

            if is_od_target := isinstance(target, ObjectDetectionTarget):
                target_labels = as_numpy(target.labels)
                target_len = len(target_labels)
                if target_len:
                    labels.extend(target_labels.tolist())
                    bboxes.extend(as_numpy(target.boxes).tolist())
                    scores.extend(as_numpy(target.scores).tolist())
                    srcidx.extend([i] * target_len)
            elif isinstance(target, Array):
                if len(target):
                    target_len = 1
                    labels.append(int(np.argmax(as_numpy(target))))
                    scores.append(target)
                    srcidx.append(i)
            else:
                raise TypeError("Encountered unsupported target type in dataset")

            is_od = is_od_target if is_od is None else is_od
            if is_od != is_od_target:
                raise ValueError("Encountered unexpected target type in dataset")

        labels = as_numpy(labels).astype(np.intp)
        scores = as_numpy(scores).astype(np.float32)
        bboxes = as_numpy(bboxes).astype(np.float32) if is_od else None
        srcidx = as_numpy(srcidx).astype(np.intp)

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
        """
        Add additional factors to the metadata.

        The number of measures per factor must match the number of images
        in the dataset or the number of detections in the dataset.

        Parameters
        ----------
        factors : Mapping[str, Array | Sequence[Any]]
            Dictionary of factors to add to the metadata.
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
