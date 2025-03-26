from __future__ import annotations

__all__ = []

import warnings
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from dataeval.typing import (
    AnnotatedDataset,
    Array,
    ArrayLike,
    ObjectDetectionTarget,
)
from dataeval.utils._array import as_numpy, to_numpy
from dataeval.utils._bin import bin_data, digitize_data, is_continuous
from dataeval.utils.metadata import merge

if TYPE_CHECKING:
    from dataeval.utils.data import Targets
else:
    from dataeval.utils.data._targets import Targets


class Metadata:
    """
    Class containing binned metadata.

    Attributes
    ----------
    discrete_factor_names : list[str]
        List containing factor names for the original data that was discrete and
        the binned continuous data
    discrete_data : NDArray[np.int64]
        Array containing values for the original data that was discrete and the
        binned continuous data
    continuous_factor_names : list[str]
        List containing factor names for the original continuous data
    continuous_data : NDArray[np.float64] | None
        Array containing values for the original continuous data or None if there
        was no continuous data
    class_labels : NDArray[np.int]
        Numerical class labels for the images/objects
    class_names : list[str]
        List of unique class names
    total_num_factors : int
        Sum of discrete_factor_names and continuous_factor_names plus 1 for class
    image_indices : NDArray[np.intp]
        Array of the image index that is mapped by the index of the factor

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
        dataset: AnnotatedDataset[tuple[Any, Any, dict[str, Any]]],
        *,
        continuous_factor_bins: Mapping[str, int | Sequence[float]] | None = None,
        auto_bin_method: Literal["uniform_width", "uniform_count", "clusters"] = "uniform_width",
        exclude: Sequence[str] | None = None,
        include: Sequence[str] | None = None,
    ) -> None:
        self._collated = False
        self._merged = None
        self._processed = False

        self._dataset = dataset
        self._continuous_factor_bins = dict(continuous_factor_bins) if continuous_factor_bins else {}
        self._auto_bin_method = auto_bin_method

        if exclude is not None and include is not None:
            raise ValueError("Filters for `exclude` and `include` are mutually exclusive.")

        self._exclude = set(exclude or ())
        self._include = set(include or ())

    @property
    def targets(self) -> Targets:
        self._collate()
        return self._targets

    @property
    def raw(self) -> list[dict[str, Any]]:
        self._collate()
        return self._raw

    @property
    def exclude(self) -> set[str]:
        return self._exclude

    @exclude.setter
    def exclude(self, value: Sequence[str]) -> None:
        exclude = set(value)
        if self._exclude != exclude:
            self._exclude = exclude
            self._include = set()
            self._processed = False

    @property
    def include(self) -> set[str]:
        return self._include

    @include.setter
    def include(self, value: Sequence[str]) -> None:
        include = set(value)
        if self._include != include:
            self._include = include
            self._exclude = set()
            self._processed = False

    @property
    def continuous_factor_bins(self) -> Mapping[str, int | Sequence[float]]:
        return self._continuous_factor_bins

    @continuous_factor_bins.setter
    def continuous_factor_bins(self, bins: Mapping[str, int | Sequence[float]]) -> None:
        if self._continuous_factor_bins != bins:
            self._continuous_factor_bins = dict(bins)
            self._processed = False

    @property
    def auto_bin_method(self) -> str:
        return self._auto_bin_method

    @auto_bin_method.setter
    def auto_bin_method(self, method: Literal["uniform_width", "uniform_count", "clusters"]) -> None:
        if self._auto_bin_method != method:
            self._auto_bin_method = method
            self._processed = False

    @property
    def merged(self) -> dict[str, Any]:
        self._merge()
        return {} if self._merged is None else self._merged[0]

    @property
    def dropped_factors(self) -> dict[str, list[str]]:
        self._merge()
        return {} if self._merged is None else self._merged[1]

    @property
    def discrete_factor_names(self) -> list[str]:
        self._process()
        return self._discrete_factor_names

    @property
    def discrete_data(self) -> NDArray[np.int64]:
        self._process()
        return self._discrete_data

    @property
    def continuous_factor_names(self) -> list[str]:
        self._process()
        return self._continuous_factor_names

    @property
    def continuous_data(self) -> NDArray[np.float64]:
        self._process()
        return self._continuous_data

    @property
    def class_labels(self) -> NDArray[np.intp]:
        self._collate()
        return self._class_labels

    @property
    def class_names(self) -> list[str]:
        self._collate()
        return self._class_names

    @property
    def total_num_factors(self) -> int:
        self._process()
        return self._total_num_factors

    @property
    def image_indices(self) -> NDArray[np.intp]:
        self._process()
        return self._image_indices

    def _collate(self, force: bool = False):
        if self._collated and not force:
            return

        raw: list[dict[str, Any]] = []

        labels = []
        bboxes = []
        scores = []
        srcidx = []
        is_od = None
        for i in range(len(self._dataset)):
            _, target, metadata = self._dataset[i]

            raw.append(metadata)

            if is_od_target := isinstance(target, ObjectDetectionTarget):
                target_len = len(target.labels)
                labels.extend(as_numpy(target.labels).tolist())
                bboxes.extend(as_numpy(target.boxes).tolist())
                scores.extend(as_numpy(target.scores).tolist())
                srcidx.extend([i] * target_len)
            elif isinstance(target, Array):
                target_len = 1
                labels.append(int(np.argmax(as_numpy(target))))
                scores.append(target)
            else:
                raise TypeError("Encountered unsupported target type in dataset")

            is_od = is_od_target if is_od is None else is_od
            if is_od != is_od_target:
                raise ValueError("Encountered unexpected target type in dataset")

        labels = as_numpy(labels).astype(np.intp)
        scores = as_numpy(scores).astype(np.float32)
        bboxes = as_numpy(bboxes).astype(np.float32) if is_od else None
        srcidx = as_numpy(srcidx).astype(np.intp) if is_od else None

        self._targets = Targets(labels, scores, bboxes, srcidx)
        self._raw = raw

        index2label = self._dataset.metadata.get("index2label", {})
        self._class_labels = self._targets.labels
        self._class_names = [index2label.get(i, str(i)) for i in np.unique(self._class_labels)]
        self._collated = True

    def _merge(self, force: bool = False):
        if self._merged is not None and not force:
            return

        targets_per_image = (
            None if self.targets.source is None else np.unique(self.targets.source, return_counts=True)[1].tolist()
        )
        self._merged = merge(self.raw, return_dropped=True, ignore_lists=False, targets_per_image=targets_per_image)

    def _validate(self) -> None:
        # Check that metadata is a single, flattened dictionary with uniform array lengths
        check_length = None
        if self._targets.labels.ndim > 1:
            raise ValueError(
                f"Got class labels with {self._targets.labels.ndim}-dimensional "
                f"shape {self._targets.labels.shape}, but expected a 1-dimensional array."
            )
        for v in self.merged.values():
            if not isinstance(v, (list, tuple, np.ndarray)):
                raise TypeError(
                    "Metadata dictionary needs to be a single dictionary whose values "
                    "are arraylike containing the metadata on a per image or per object basis."
                )
            else:
                check_length = len(v) if check_length is None else check_length
                if check_length != len(v):
                    raise ValueError(
                        "The lists/arrays in the metadata dict have varying lengths. "
                        "Metadata requires them to be uniform in length."
                    )
        if len(self._class_labels) != check_length:
            raise ValueError(
                f"The length of the label array {len(self._class_labels)} is not the same as "
                f"the length of the metadata arrays {check_length}."
            )

    def _process(self, force: bool = False) -> None:
        if self._processed and not force:
            return

        # Create image indices from targets
        self._image_indices = np.arange(len(self.raw)) if self.targets.source is None else self.targets.source

        # Validate the metadata dimensions
        self._validate()

        # Include specified metadata keys
        if self.include:
            metadata = {i: self.merged[i] for i in self.include if i in self.merged}
            continuous_factor_bins = (
                {i: self.continuous_factor_bins[i] for i in self.include if i in self.continuous_factor_bins}
                if self.continuous_factor_bins
                else {}
            )
        else:
            metadata = self.merged
            continuous_factor_bins = dict(self.continuous_factor_bins) if self.continuous_factor_bins else {}
            for k in self.exclude:
                metadata.pop(k, None)
                continuous_factor_bins.pop(k, None)

        # Remove generated "_image_index" if present
        if "_image_index" in metadata:
            metadata.pop("_image_index", None)

        # Bin according to user supplied bins
        continuous_metadata = {}
        discrete_metadata = {}
        if continuous_factor_bins:
            invalid_keys = set(continuous_factor_bins.keys()) - set(metadata.keys())
            if invalid_keys:
                raise KeyError(
                    f"The keys - {invalid_keys} - are present in the `continuous_factor_bins` dictionary "
                    "but are not keys in the `metadata` dictionary. Delete these keys from `continuous_factor_bins` "
                    "or add corresponding entries to the `metadata` dictionary."
                )
            for factor, bins in continuous_factor_bins.items():
                discrete_metadata[factor] = digitize_data(metadata[factor], bins)
                continuous_metadata[factor] = metadata[factor]

        # Determine category of the rest of the keys
        remaining_keys = set(metadata.keys()) - set(continuous_metadata.keys())
        for key in remaining_keys:
            data = to_numpy(metadata[key])
            if np.issubdtype(data.dtype, np.number):
                result = is_continuous(data, self._image_indices)
                if result:
                    continuous_metadata[key] = data
                unique_samples, ordinal_data = np.unique(data, return_inverse=True)
                if unique_samples.size <= np.max([20, data.size * 0.01]):
                    discrete_metadata[key] = ordinal_data
                else:
                    warnings.warn(
                        f"A user defined binning was not provided for {key}. "
                        f"Using the {self.auto_bin_method} method to discretize the data. "
                        "It is recommended that the user rerun and supply the desired "
                        "bins using the continuous_factor_bins parameter.",
                        UserWarning,
                    )
                    discrete_metadata[key] = bin_data(data, self.auto_bin_method)
            else:
                _, discrete_metadata[key] = np.unique(data, return_inverse=True)

        # Split out the dictionaries into the keys and values
        self._discrete_factor_names = list(discrete_metadata.keys())
        self._discrete_data = (
            np.stack(list(discrete_metadata.values()), axis=-1, dtype=np.int64)
            if discrete_metadata
            else np.array([], dtype=np.int64)
        )
        self._continuous_factor_names = list(continuous_metadata.keys())
        self._continuous_data = (
            np.stack(list(continuous_metadata.values()), axis=-1, dtype=np.float64)
            if continuous_metadata
            else np.array([], dtype=np.float64)
        )
        self._total_num_factors = len(self._discrete_factor_names + self._continuous_factor_names) + 1
        self._processed = True

    def add_factors(self, factors: Mapping[str, ArrayLike]) -> None:
        self._merge()
        self._processed = False
        target_len = len(self.targets.source) if self.targets.source is not None else len(self.targets)
        if any(len(v) != target_len for v in factors.values()):
            raise ValueError(
                "The lists/arrays in the provided factors have a different length than the current metadata factors."
            )
        merged = cast(tuple[dict[str, ArrayLike], dict[str, list[str]]], self._merged)[0]
        for k, v in factors.items():
            merged[k] = v
